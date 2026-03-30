from __future__ import annotations

from typing import List, Tuple
import logging
from dataclasses import dataclass

from torch import nn, Tensor
from tqdm import tqdm
import numpy as np
from torch.utils.data import Subset
import torch
from torch.optim import Optimizer

from dengine import SupervisedDataset, DecAvgClient, VanillaDecentralizedMessage, TrainingEngine
from dengine.interfaces import ModuleBase
from dengine.scenarios.decentralized import model_wise_weighted_average
from dengine.models.memnet import MemNet
from dengine.training_strategies.decorators import register_local_training
from dengine.scenarios.decorators import register_client


class WithIndexDataset(Subset):
    def __getitem__(self, index):
        return self.indices[index], super().__getitem__(index)

    def __getitems__(self, indices: list):
        return [(index, self.dataset[index]) for index in indices]


@register_local_training()
class GANLocalUpdate(TrainingEngine):
    _training_data: WithIndexDataset

    def __init__(self, training_data: Subset, *args, **kwargs):
        dataset = WithIndexDataset(training_data, training_data.indices)
        super().__init__(*args, training_data=dataset, **kwargs)

    def _load_optimizer(self, net: MemNet) -> Tuple[Optimizer, Optimizer]:
        return (net.optimizer_G, net.optimizer_D)

    def train(self, model: MemNet, current_time: float) -> ModuleBase:
        model.train(True)
        epoch_pbar = tqdm(
            range(self._local_epochs),
            ascii=True,
            leave=False,
            total=self._local_epochs
        )

        for iter in epoch_pbar:
            self.callback.on_training_epoch_start(epoch=iter)
            epoch_pbar.set_description(f'Epoch {iter}/{self._local_epochs}')
            g_losses = []
            d_losses = []
            total_samples = len(self._training_data)
            batch_iter = tqdm(
                self._ldr_train,
                ascii=True,
                leave=False,
                total=(total_samples // self._training_batch_size)
            )

            for batch_idx, (images, labels) in batch_iter:
                g_loss, d_loss = self.train_step(model, images, batch_idx)
                g_losses.append(g_loss)
                d_losses.append(d_loss)
                epoch_pbar.set_postfix(g_loss=g_loss, d_loss=d_loss)
            self.callback.on_training_epoch_end(
                epoch=iter,
                g_loss=float(np.array(g_loss).mean()),
                d_loss=float(np.array(d_loss).mean()),
            )

        model.train(False)
        self.callback.on_local_training_end(
            current_time=0,
            g_loss=float(np.array(g_loss).mean()),
            d_loss=float(np.array(d_loss).mean()),
        )
        return model

    def train_step(self, net: MemNet, images: Tensor, batch_idxs: Tensor):
        device = next(net.parameters()).device
        images = images.to(device)
        batch_idxs = batch_idxs.to(device)
        batch_size = batch_idxs.size(0)

        # Generator input from embedding
        z = net.embedding_layer(batch_idxs)
        gen_imgs = net.generator(z)

        # Train Discriminator
        net.optimizer_D.zero_grad()

        real_images_discr_output = net.discriminator(images)
        fake_images_discr_output = net.discriminator(gen_imgs.detach())

        discriminator_output_size = real_images_discr_output.size(-1)
        valid = torch.ones((batch_size, 1, discriminator_output_size, discriminator_output_size), device=device)
        fake = torch.zeros((batch_size, 1, discriminator_output_size, discriminator_output_size), device=device)

        real_loss = net.adversarial_loss(real_images_discr_output, valid)
        fake_loss = net.adversarial_loss(fake_images_discr_output, fake)

        d_loss = 0.5 * (real_loss + fake_loss)
        d_loss.backward()
        net.optimizer_D.step()

        # Train Generator
        net.optimizer_G.zero_grad()
        net.optimizer_E.zero_grad()
        discriminator_output = net.discriminator(gen_imgs)
        g_loss_adv = net.adversarial_loss(discriminator_output, valid)
        g_loss_pixel = net.pixelwise_loss(gen_imgs, images)
        g_loss = g_loss_adv + 100 * g_loss_pixel
        g_loss.backward()
        net.optimizer_E.step()
        net.optimizer_G.step()

        return g_loss.item(), d_loss.item()


@dataclass
class GANPaivSyncMessage(VanillaDecentralizedMessage):
    source_paiv: GANPaiv


@register_client()
class GANPaiv(DecAvgClient):
    def __init__(
        self,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        assert isinstance(self.model, MemNet)
        self.model: MemNet

    def aggregation(self, messages: List[GANPaivSyncMessage]):
        messages = self.filter_message_from_neighborhood(messages)
        if len(messages) == 0:
            return
        message_origins = [_p.source_paiv.UUID for _p in messages]
        logging.info(f"PAIV-{self.UUID}, aggregating models sent by: {message_origins}")

        embeddings = [m.source_paiv.model.embedding_layer.state_dict() for m in messages]
        generators = [m.source_paiv.model.generator.state_dict() for m in messages]
        discriminators = [m.source_paiv.model.discriminator.state_dict() for m in messages]
        embedding_avg = model_wise_weighted_average(
            embeddings,
            self._get_alphas(messages)
        )
        generator_avg = model_wise_weighted_average(
            generators,
            self._get_alphas(messages)
        )
        discrimnator_avg = model_wise_weighted_average(
            discriminators,
            self._get_alphas(messages)
        )
        self.model.embedding_layer.load_state_dict(embedding_avg)
        self.model.generator.load_state_dict(generator_avg)
        self.model.discriminator.load_state_dict(discrimnator_avg)

    def test(self, dataset: SupervisedDataset, model: nn.Module | None = None):
        pass
