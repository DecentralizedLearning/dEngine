from abc import abstractmethod

from typing import Optional, Literal, Dict
from warnings import deprecated
import copy
import json
import logging

from tqdm import tqdm
import numpy as np
import torch
from dataclasses import dataclass
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, CosineAnnealingLR
from torch.utils.data import DataLoader, Subset

from dengine.dataset import SupervisedDataset
from dengine.dataset.utils import get_targets
from dengine.interfaces import ClientCallbackInterface, LocalTrainingEngineInterface, ModuleBase
from dengine.callbacks import DummyCallback
from dengine.utils.utils import softmax_with_argmax, get_output_in_production, assert_no_nans
from dengine.interfaces import ScenarioEngineInterface, ClientInterface

from .decorators import register_local_training


@dataclass
class training_step_output:
    output: Tensor
    reduced_loss: float


class EarlyStoppingBase(LocalTrainingEngineInterface):
    def __init__(
        self,
        training_data: Subset,
        validation_data: Subset,
        scenario: ScenarioEngineInterface,
        client: ClientInterface,
        # Additional args
        training_batch_size: int,
        validation_batch_size: int,
        epochs: int,
        callback: Optional[ClientCallbackInterface] = None,
        # Additional kwargs
        verbose: bool = True,
        patience: int = 0,
    ):
        self._scenario = scenario
        self._client = client
        self._callback = callback or DummyCallback()

        self._verbose = verbose
        self._patience = patience
        self._training_batch_size = training_batch_size
        self._validation_batch_size = validation_batch_size

        self._loss_func = torch.nn.CrossEntropyLoss(reduction='none')
        self._val_err_func = self._loss_func

        self._local_epochs = epochs

        if len(training_data) == 0:
            self._ldr_train = None
        else:
            self._ldr_train = DataLoader(
                training_data,
                self._training_batch_size,
                shuffle=True
            )
        self._training_data = training_data
        self._validation_data = validation_data

    @property
    def callback(self) -> ClientCallbackInterface:
        return self._callback

    @callback.setter
    def callback(self, value: ClientCallbackInterface):
        self._callback = value

    def _load_optimizer(self, net: Module) -> Optimizer:
        raise NotImplementedError()

    def _load_scheduler(self, optimizer: Optimizer) -> Optional[LRScheduler]:
        raise NotImplementedError()

    @property
    def _logging(self):
        def mock_call(s: str):
            pass

        if not self._verbose:
            return mock_call
        return logging.info

    @property
    def _logging_warning(self):
        def mock_call(s: str):
            pass

        if not self._verbose:
            return mock_call
        return logging.warning

    # ...................................... #
    # TRAINING LOOP
    # ...................................... #
    @abstractmethod
    def training_step(
        self,
        net: Module,
        optimizer: Optimizer,
        *args
    ) -> training_step_output:
        raise NotImplementedError()

    def train(self, model: ModuleBase, current_time: float) -> ModuleBase:
        if self._ldr_train is None:
            self._logging_warning("Empty dataset, training is ignored!")
            return model

        self._callback.on_local_training_start(current_time)

        best_performing_net = copy.deepcopy(model)
        best_Y_hat_train = np.nan
        best_Y_hat_valid = np.nan
        best_training_loss = np.nan
        best_valid_loss = np.nan

        current_valid_loss = np.inf
        # self._valid_loss(
        #     get_output_in_production(
        #         net=model,
        #         dataset=self._validation_data,
        #         idxs=None,
        #         batch_size=self._validation_batch_size
        #     )
        # )

        patience = 0
        optimizer = self._load_optimizer(model)
        scheduler = self._load_scheduler(optimizer)

        epoch_pbar = tqdm(
            range(self._local_epochs),
            ascii=True,
            leave=False,
            total=self._local_epochs
        )
        for current_epoch in epoch_pbar:
            Y_hat_train = []
            epoch_pbar.set_description(f'Epoch {current_epoch}/{self._local_epochs} (patience: {patience}/{self._patience}): ')
            self.callback.on_training_epoch_start(current_epoch)

            # 2. Looping over batches
            total_samples = len(self._training_data)
            batch_iter = tqdm(
                enumerate(self._ldr_train),
                ascii=True,
                leave=False,
                total=(total_samples // self._training_batch_size)
            )
            cumulated_loss = 0
            for batch_idx, batch_data in batch_iter:
                self.callback.on_training_batch_start(batch_idx, *batch_data)
                step_output = self.training_step(model, optimizer, *batch_data)
                cumulated_loss += step_output.reduced_loss
                Y_hat_train.extend(step_output.output)
                self.callback.on_training_batch_end(batch_idx, *batch_data, cumulated_loss=cumulated_loss)
            epoch_loss = cumulated_loss / (batch_idx + 1)

            if scheduler:
                scheduler.step()

            # 3. Computing validation loss
            Y_hat_validation = get_output_in_production(
                net=model,
                dataset=self._validation_data,
                idxs=None,
                batch_size=self._validation_batch_size
            )
            new_valid_loss = self._valid_loss(Y_hat_validation)
            new_acc = self._valid_accuracy(Y_hat_validation)

            epoch_pbar_postfix_dict = {
                'loss': float(round(epoch_loss, 5)),
                'vl_loss': round(float(new_valid_loss), 5),
                'vl_acc': round(float(new_acc), 5)
            }
            epoch_pbar.set_postfix(epoch_pbar_postfix_dict)

            # 4. Early stopping
            if new_valid_loss >= current_valid_loss:
                patience += 1
                if patience >= self._patience:
                    new_valid_loss_str = str(round(float(new_valid_loss), 5))
                    current_valid_loss_str = str(round(float(current_valid_loss), 5))
                    self._logging(
                        f'Triggered early stop at local epoch {current_epoch} '
                        f'for increasing valid loss [new:{new_valid_loss_str} > prev:{current_valid_loss_str}] '
                        f'(patience: {patience}/{self._patience})'
                    )
                    break
            else:
                patience = 0
                best_performing_net = copy.deepcopy(model)
                current_valid_loss = float(new_valid_loss)

                best_Y_hat_train = torch.stack(Y_hat_train)
                best_Y_hat_valid = Y_hat_validation
                best_training_loss = epoch_loss
                best_valid_loss = current_valid_loss

            if current_valid_loss == 0:
                break

            self.callback.on_training_epoch_end(
                current_epoch,
                training_data=self._training_data,
                validation_data=self._validation_data,
                Y_hat_train=best_Y_hat_train,
                Y_hat_valid=best_Y_hat_valid,
                training_loss=best_training_loss,
                validation_loss=best_valid_loss,
            )

        self._logging(
            f'Metrics on the last epoch: '
            f'{json.dumps(epoch_pbar_postfix_dict)}'
        )
        self.callback.on_local_training_end(
            current_time,
            training_data=self._training_data,
            validation_data=self._validation_data,
            Y_hat_train=best_Y_hat_train,
            Y_hat_valid=best_Y_hat_valid,
            training_loss=best_training_loss,
            validation_loss=best_valid_loss,
        )
        return best_performing_net

    # ...................................... #
    # VALIDATION AND METRICS
    # ...................................... #
    @torch.no_grad()
    def compute_loss_in_production(
        self,
        model: Module,
        data: SupervisedDataset
    ) -> Dict[Literal["output", "loss"], Tensor]:
        Y_log_probs = get_output_in_production(
            model,
            data,
            None,
            self._validation_batch_size
        )
        ground_truth = data.targets.to(Y_log_probs.device).type(data.targets.dtype)
        loss = self._loss_func(Y_log_probs, ground_truth)
        return {"output": Y_log_probs, "loss": loss}

    @torch.no_grad()
    def _valid_loss(self, log_probs: Tensor) -> Tensor:
        ground_truth = get_targets(self._validation_data)
        ground_truth = ground_truth.to(log_probs.device)
        return self._val_err_func(log_probs, ground_truth).mean()

    @torch.no_grad()
    def _valid_accuracy(self, log_probs: Tensor) -> Tensor:
        ground_truth = get_targets(self._validation_data)
        ground_truth = ground_truth.to(log_probs.device)
        y_pred = softmax_with_argmax(log_probs)
        accuracy = (ground_truth == y_pred).type(torch.float)
        return accuracy.mean()


@register_local_training()
class TrainingEngine(EarlyStoppingBase):
    def __init__(
        self,
        *args,
        optimizer: Literal["SGD", "Adam"],
        lr: float,
        sgd_momentum: float = 0,
        adam_weight_decay: float = 0,
        scheduler: Literal["cosine"] | None = None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self._lr = lr
        self._sgd_momentum = sgd_momentum
        self._adam_weigth_decay = adam_weight_decay

        self._selected_clients = []
        self._optimizer_type = optimizer
        self._scheduler_type = scheduler

    def _load_optimizer(self, net) -> Optimizer:
        if self._optimizer_type == "SGD":
            return torch.optim.SGD(
                net.parameters(),
                lr=self._lr,
                momentum=self._sgd_momentum
            )
        elif self._optimizer_type == "Adam":
            return torch.optim.Adam(
                net.parameters(),
                lr=self._lr,
                weight_decay=self._adam_weigth_decay
            )
        raise ValueError(f"Optimizer {self._optimizer_type} is not supported")

    def _load_scheduler(self, optimizer: Optimizer) -> Optional[LRScheduler]:
        if not self._scheduler_type:
            return
        if self._scheduler_type == "cosine":
            return CosineAnnealingLR(
                optimizer,
                T_max=self._local_epochs
            )
        raise ValueError(f"Scheduler {self._scheduler_type} is not supported")

    def training_step(
        self,
        net: Module,
        optimizer: Optimizer,
        images: Tensor,
        labels: Tensor,
    ) -> training_step_output:
        net.train()
        device = next(net.parameters()).device
        images = images.to(device)
        labels = labels.to(device)

        output = net(images)
        assert_no_nans(output)

        erm = self._loss_func(output, labels)
        loss = torch.nanmean(erm, dim=0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return training_step_output(output=output, reduced_loss=loss.item())


@register_local_training()
class EngineWithoutEarlyStopping(TrainingEngine):
    def train(self, model: ModuleBase, current_time: float) -> ModuleBase:
        if self._ldr_train is None:
            self._logging_warning("Empty dataset, training is ignored!")
            return model

        optimizer = self._load_optimizer(model)
        scheduler = self._load_scheduler(optimizer)

        epoch_pbar = tqdm(
            range(self._local_epochs),
            ascii=True,
            leave=False,
            total=self._local_epochs
        )

        for current_epoch in epoch_pbar:
            epoch_pbar.set_description(f'Epoch {current_epoch}/{self._local_epochs}')
            self.callback.on_training_epoch_start(current_epoch)

            batch_iter = enumerate(self._ldr_train)
            total_samples = len(self._training_data)
            batch_iter = tqdm(
                batch_iter,
                ascii=True,
                leave=False,
                total=(total_samples // self._training_batch_size)
            )

            cumulated_loss = 0
            for batch_idx, batch_data in batch_iter:
                self.callback.on_training_batch_start(batch_idx, *batch_data)
                cumulated_loss += self.training_step(model, optimizer, *batch_data).reduced_loss
                self.callback.on_training_batch_end(batch_idx, cumulated_loss=cumulated_loss)

            if scheduler:
                scheduler.step()

            # 3. Computing validation loss
            epoch_loss = cumulated_loss / (batch_idx + 1)
            epoch_pbar_postfix_dict = {
                'loss': f"{epoch_loss:012.6f}",
            }
            epoch_pbar.set_postfix(epoch_pbar_postfix_dict)

            # 5. Finalization
            self.callback.on_training_epoch_end(current_epoch, epoch_loss=epoch_loss)

        self._logging(
            f'Metrics on the last epoch: '
            f'{json.dumps(epoch_pbar_postfix_dict)}'
        )
        self.callback.on_local_training_end(
            current_time,
            training_data=self._training_data,
            validation_data=self._validation_data,
            training_loss=epoch_loss,
        )
        return model


@deprecated("This class has been refactored to TrainingEngine, please consider refactoring.")
@register_local_training()
class StandardLocalUpdateV1(TrainingEngine):
    def __init__(
        self,
        kd_alpha: float,
        vteacher_generator: float,
        skd_beta: float,
        momentum: float,
        local_ep: int,
        weight_decay: float,
        *args,
        **kwargs,
    ):
        super().__init__(
            *args,
            adam_weight_decay=weight_decay,
            epochs=local_ep,
            sgd_momentum=momentum,
            **kwargs
        )
