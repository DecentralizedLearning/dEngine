import logging
from typing import Dict, Literal, Optional

import torch
from torch.utils.data import Subset
import numpy as np
from torch import Tensor
from npy_append_array import NpyAppendArray

from dengine.config.constants import CHECKPOINT_DIR_NAME, METRICS_DIR_NAME
from dengine.utils.utils import get_output_in_production
from dengine.utils.confusion_matrix import confusion_matrix_from_model_output

from .callback import PeriodicCallback
from .decorators import register_callback


@register_callback()
class ModelDumpOnEpochEndCallback(PeriodicCallback):
    LAST_CHECKPOINT_NAME = 'last.ckpt'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        output_path_root = self.experiment_configuration.output_directory
        self._outfolder = output_path_root / f'{CHECKPOINT_DIR_NAME}/{self.client.UUID}'
        self._outfolder.mkdir(exist_ok=True, parents=True)
        self._fpath = self._outfolder / self.LAST_CHECKPOINT_NAME

    def training_epoch_end(self, epoch: int, **kwargs):
        logging.info(f"Saved model checkpoint at epoch: {self._fpath}")
        torch.save(self.client.model.state_dict(), self._fpath)


@register_callback()
class ModelDumpOnLocalTrainEndCallback(PeriodicCallback):
    LAST_CHECKPOINT_NAME = 'last.ckpt'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        output_path_root = self.experiment_configuration.output_directory
        self._outfolder = output_path_root / f'{CHECKPOINT_DIR_NAME}/{self.client.UUID}'
        self._outfolder.mkdir(exist_ok=True, parents=True)
        self._fpath = self._outfolder / self.LAST_CHECKPOINT_NAME

    def local_training_end(self, *args, **kwargs):
        logging.info(f"Saved model checkpoint at: {self._fpath}")
        torch.save(self.client.model.state_dict(), self._fpath)


@register_callback()
class LossDumpCallback(PeriodicCallback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        output_path_root = self.experiment_configuration.output_directory
        self._outfolder = output_path_root / f'{METRICS_DIR_NAME}/{self.client.UUID}'

        self._tr_outfile = self._outfolder / 'round_loss_tr.npy'
        self._tr_outfile.parent.mkdir(parents=True, exist_ok=True)

        self._valid_outfile = self._outfolder / 'round_loss_valid.npy'
        self._valid_outfile.parent.mkdir(parents=True, exist_ok=True)

        self._test_outfile = self._outfolder / 'round_loss_test.npy'
        self._test_outfile.parent.mkdir(parents=True, exist_ok=True)

        self._training_output = []

    def test_inference_end(
        self,
        current_time: float,
        output: Dict[Literal["output", "loss"], Tensor],
    ):
        test_loss = float(output['loss'].mean())
        with NpyAppendArray(self._test_outfile) as f:
            f.append(np.array([test_loss]))
        logging.info(f'Appended test loss {round(test_loss, 5)} to: {self._test_outfile}')

    def local_training_end(
        self,
        current_time: float,
        training_loss: float,
        validation_loss: Optional[float] = None,
        **kwargs
    ):
        with NpyAppendArray(self._tr_outfile) as tr_out_file_array:
            tr_out_file_array.append(np.array([training_loss]))
        logging.info(f'Appended trainining loss {round(training_loss, 5)} at {current_time} to: {self._tr_outfile}')

        if not validation_loss:
            return

        with NpyAppendArray(self._valid_outfile) as valid_out_file_array:
            valid_out_file_array.append(np.array([validation_loss]))
        logging.info(f'Appended validation loss {round(validation_loss, 5)} at {current_time} to: {self._valid_outfile}')


class ConfusionMatrixDumpBase(PeriodicCallback):
    def __init__(self, batch_size: Optional[int] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        training_engine_args = self.experiment_configuration.client.training_engine.arguments
        if training_engine_args and (bs := training_engine_args.get("validation_batch_size")):
            batch_size = int(bs)
        if batch_size is None:
            raise ValueError("Neither batch_size or `client > training_engine > arguments` was found.")
        self.batch_size = batch_size

        self.num_classes = len(self.test_data.unique_targets)


@register_callback()
class ConfusionMatrixDumpOnCommRoundEndCallback(ConfusionMatrixDumpBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        output_path_root = self.experiment_configuration.output_directory
        self._outfolder = output_path_root / f'{METRICS_DIR_NAME}/{self.client.UUID}'
        self._num_classes = len(self.test_data.unique_targets)

        self._tr_outfile = self._outfolder / 'confusion_matrix_tr.npy'
        self._tr_outfile.parent.mkdir(parents=True, exist_ok=True)

        self._valid_outfile = self._outfolder / 'confusion_matrix_valid.npy'
        self._valid_outfile.parent.mkdir(parents=True, exist_ok=True)

        self._test_outfile = self._outfolder / 'confusion_matrix_test.npy'
        self._test_outfile.parent.mkdir(parents=True, exist_ok=True)

        self._training_output = []

        self._last_Y_hat_train = None
        self._last_Y_hat_valid = None

    def test_inference_end(
        self,
        current_time: float,
        output: Dict[Literal["output", "loss"], Tensor],
    ):
        test_conf_matrix = confusion_matrix_from_model_output(output['output'], self.test_data, None, self._num_classes)
        with NpyAppendArray(self._test_outfile) as f:
            f.append(test_conf_matrix)
        logging.info(f'Appended test confusion matrix to: {self._test_outfile}')

    def local_training_end(
        self,
        *args,
        Y_hat_train: Tensor,
        training_data: Subset,
        Y_hat_valid: Tensor,
        validation_data: Subset,
        **kwargs
    ):
        # No improvment since last epoch, copy-pasting last known value
        if not torch.is_tensor(Y_hat_train) and np.isnan(Y_hat_train):
            if self._last_Y_hat_train is None:
                self._last_Y_hat_train = get_output_in_production(self.client.model, training_data, None, self.batch_size)
            Y_hat_train = self._last_Y_hat_train
        else:
            self._last_Y_hat_train = Y_hat_train

        if not torch.is_tensor(Y_hat_valid) and np.isnan(Y_hat_valid):
            if self._last_Y_hat_valid is None:
                self._last_Y_hat_valid = get_output_in_production(self.client.model, validation_data, None, self.batch_size)
            Y_hat_valid = self._last_Y_hat_valid
        else:
            self._last_Y_hat_valid = Y_hat_valid

        tr_conf_matrix = confusion_matrix_from_model_output(Y_hat_train, training_data.dataset, training_data.indices, self._num_classes)  # type: ignore
        with NpyAppendArray(self._tr_outfile) as tr_out_file_array:
            tr_out_file_array.append(tr_conf_matrix)
        logging.info(f'Appended train confusion matrix to: {self._tr_outfile}')

        vl_conf_matrix = confusion_matrix_from_model_output(Y_hat_valid, validation_data.dataset, validation_data.indices, self._num_classes)  # type: ignore
        with NpyAppendArray(self._valid_outfile) as valid_out_file_array:
            valid_out_file_array.append(vl_conf_matrix)
        logging.info(f'Appended validation confusion matrix to: {self._valid_outfile}')


@register_callback()
class ConfusionMatrixDumpOnAggregationEnd(ConfusionMatrixDumpBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        output_path_root = self.experiment_configuration.output_directory
        outfolder = output_path_root / f'{METRICS_DIR_NAME}/{self.client.UUID}'
        self._test_outfile = outfolder / 'confusion_matrix_test_after_agg_before_tr.npy'
        self._test_outfile.parent.mkdir(parents=True, exist_ok=True)

    def aggregation_end(self, current_time: float):
        Y_hat = get_output_in_production(self.client.model, self.test_data, None, self.batch_size)
        test_conf_matrix = confusion_matrix_from_model_output(Y_hat, self.test_data, None, self.num_classes)
        with NpyAppendArray(self._test_outfile) as f:
            f.append(test_conf_matrix)
        logging.info(f'Appended test confusion matrix after agg and before local train to: {self._test_outfile}')


@register_callback()
class ConfusionMatrixDumpOnEpochEndCallback(ConfusionMatrixDumpBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        output_path_root = self.experiment_configuration.output_directory
        outfolder = output_path_root / f'{METRICS_DIR_NAME}/{self.client.UUID}'

        self._tr_outfile = outfolder / 'confusion_matrix_tr.npy'
        self._tr_outfile.parent.mkdir(parents=True, exist_ok=True)

        self._test_outfile = outfolder / 'confusion_matrix_test.npy'
        self._test_outfile.parent.mkdir(parents=True, exist_ok=True)

        self._last_Y_hat_train = None

    def training_epoch_end(
        self,
        *args,
        Y_hat_train: Tensor,
        training_data: Subset,
        **kwargs
    ):
        # No improvment since last epoch, copy-pasting last known value
        if not torch.is_tensor(Y_hat_train) and np.isnan(Y_hat_train):
            if self._last_Y_hat_train is None:
                self._last_Y_hat_valid = get_output_in_production(self.client.model, training_data, None, self.batch_size)
            Y_hat_train = self._last_Y_hat_train
        else:
            self._last_Y_hat_train = Y_hat_train

        tr_conf_matrix = confusion_matrix_from_model_output(Y_hat_train, training_data.dataset, training_data.indices, self.num_classes)  # type: ignore
        with NpyAppendArray(self._tr_outfile) as tr_out_file_array:
            tr_out_file_array.append(tr_conf_matrix)
        logging.info(f'Appended train confusion matrix to: {self._tr_outfile}')

        Y_hat_test = get_output_in_production(self.client.model, self.test_data, None, self.batch_size)
        test_conf_matrix = confusion_matrix_from_model_output(Y_hat_test, self.test_data, None, self.num_classes)
        with NpyAppendArray(self._test_outfile) as f:
            f.append(test_conf_matrix)
        logging.info(f'Appended test confusion matrix to: {self._test_outfile}')


@register_callback()
class ConfusionMatrixDumpOnTestInferenceEnd(PeriodicCallback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        output_path_root = self.experiment_configuration.output_directory
        self._outfolder = output_path_root / f'{METRICS_DIR_NAME}/{self.client.UUID}'
        self._num_classes = len(self.test_data.unique_targets)

        self._test_outfile = self._outfolder / 'confusion_matrix_test.npy'
        self._time_outfile = self._outfolder / '.confusion_matrix_test.time.npy'
        self._test_outfile.parent.mkdir(parents=True, exist_ok=True)

    def test_inference_end(
        self,
        current_time: float,
        output: Dict[Literal["output", "loss"], Tensor],
    ):
        test_conf_matrix = confusion_matrix_from_model_output(output['output'], self.test_data, None, self._num_classes)
        with NpyAppendArray(self._test_outfile) as f:
            f.append(test_conf_matrix)
        with NpyAppendArray(self._time_outfile) as f:
            f.append(np.array([current_time]))
        logging.info(f'Appended test confusion matrix to: {self._test_outfile}')
