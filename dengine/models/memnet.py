from __future__ import annotations

from typing import List
from dataclasses import dataclass
from math import log2, ceil

from PIL.Image import Image
from torch import nn, Tensor
from torch.utils.data import Subset
import numpy as np
import torch
import torchvision.transforms.functional as TF

from dengine.dataset.utils import get_targets

from .utils import get_image_channels, get_image_size
from .decorators import ModuleBase, register_model
from .modules import UpScalingBlock, PatchDiscriminator


@dataclass
class InterpolationResult:
    source_label_idx: int
    target_label_idx: int
    frames: List[Image]


class Decoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_size: int,
        output_channels: int,
        decoder_spreadout_channels: int,
        decoder_output_channels: int,
        decoder_latent_dim: int,
    ):
        super().__init__()
        self._decoder_input_channels = decoder_spreadout_channels
        self._decoder_input_size = decoder_latent_dim

        num_upscaling_blocks = log2(output_size / decoder_latent_dim)
        num_upscaling_blocks = ceil(num_upscaling_blocks)

        max_power = int(log2(decoder_spreadout_channels))
        min_power = int(log2(decoder_output_channels))
        layers_output_channels = []
        for i in range(num_upscaling_blocks + 1):
            layers_output_channels.append(
                max_power - i % (max_power - min_power + 1)
            )
        layers_output_channels = [2 ** x for x in sorted(layers_output_channels, reverse=True)]

        decoder_blocks = []
        for block_id in range(len(layers_output_channels) - 1):
            ch_prev = layers_output_channels[block_id]
            ch_next = layers_output_channels[block_id + 1]
            decoder_blocks.append(UpScalingBlock(ch_prev, ch_next))
        decoder_blocks.extend([
            nn.Conv2d(ch_next, output_channels, 3, padding=1),
            nn.Tanh()
        ])

        self.latent_to_decoder_input_mapping = nn.Sequential(
            nn.Linear(input_dim, decoder_spreadout_channels * decoder_latent_dim ** 2)
        )
        self.decoder = nn.Sequential(*decoder_blocks)

    def forward(self, z: Tensor):
        out = self.latent_to_decoder_input_mapping(z)
        out = out.view(
            out.shape[0],
            self._decoder_input_channels,
            self._decoder_input_size,
            self._decoder_input_size
        )
        img = self.decoder(out)
        return img

    @torch.no_grad()
    def decode_to_pil(self, z: Tensor) -> List[Image]:
        out = self(z)
        out = (out + 1) / 2
        return [TF.to_pil_image(x.cpu().clamp(0, 1)) for x in out]


@register_model()
class MemNet(ModuleBase):
    def __init__(
        self,
        dataset: Subset,
        # Config params
        embedding_layer_size: int,
        discriminator_encoder_spreadout_channels: int,
        discriminator_encoder_output_channels: int,
        discriminator_output_size: int,
        generator_decoder_spreadout_channels: int,
        generator_decoder_output_channels: int,
        generator_input_size: int,
        label_flip_on_strength_limit: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__(dataset)

        self._allow_label_flip = label_flip_on_strength_limit
        self.corrupted_indexes = []
        self._train_idx = dataset.indices
        self._memsize = len(self._train_idx)
        self._labels = get_targets(dataset)
        self._dataset = dataset.dataset if self._allow_label_flip else None
        self._unique_labels = self._labels.unique().tolist()

        self.embedding_layer = nn.Embedding(self._memsize, embedding_layer_size)
        nn.init.uniform_(self.embedding_layer.weight, -0.05, 0.05)

        dataset_channels = get_image_channels(dataset)
        dataset_size = get_image_size(dataset)
        self.generator = Decoder(
            input_dim=embedding_layer_size,
            output_size=dataset_size,
            output_channels=dataset_channels,
            decoder_spreadout_channels=generator_decoder_spreadout_channels,
            decoder_output_channels=generator_decoder_output_channels,
            decoder_latent_dim=generator_input_size
        )
        self.discriminator = PatchDiscriminator(
            input_size=dataset_size,
            input_channels=dataset_channels,
            output_size=discriminator_output_size,
            encoder_spreadout_channels=discriminator_encoder_spreadout_channels,
            encoder_output_channels=discriminator_encoder_output_channels,
        )

        self.adversarial_loss = nn.MSELoss()
        self.pixelwise_loss = nn.L1Loss()

        self.optimizer_E = torch.optim.Adam(
            self.embedding_layer.parameters(),
            lr=0.001, betas=(0.5, 0.999)
        )
        self.optimizer_G = torch.optim.Adam(
            self.generator.parameters(),
            lr=0.0002, betas=(0.5, 0.999)
        )
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))

    def get_random_label_idxs(self, label: int, n: int) -> Tensor:
        device = next(self.embedding_layer.parameters()).device
        dataset_indexes = np.where(self._labels == label)[0]
        embeddings_idxs = np.intersect1d(dataset_indexes, self._train_idx)
        embeddings_idxs = np.setdiff1d(embeddings_idxs, self.corrupted_indexes)
        random_indexes = np.random.choice(
            embeddings_idxs,
            size=n,
            replace=False
        )
        return torch.from_numpy(random_indexes).to(device)

    def get_random_label_embeddings(self, label: int, n: int) -> Tensor:
        random_indexes = self.get_random_label_idxs(label, n)
        self.corrupted_indexes.extend(map(int, random_indexes))
        return self.embedding_layer(random_indexes)

    def get_random_label_data(self, label: int, n: int) -> Tensor:
        if not self._dataset:
            raise ValueError('Dataset not tracked. Is label_flip_on_strength_limit active?')
        random_indexes = self.get_random_label_idxs(label, n)
        self.corrupted_indexes.extend(map(int, random_indexes))
        return self._dataset.data[random_indexes]

    @torch.no_grad()
    def sample_validation_batch(self, frames: int = 100) -> List[InterpolationResult]:
        res = []
        for source_label in range(len(self._unique_labels)):
            destination_label = (source_label + 1) % len(self._unique_labels)
            source_embeddings = self.get_random_label_embeddings(source_label, 1)
            target_embeddings = self.get_random_label_embeddings(destination_label, 1)
            alphas_values = np.linspace(0, 1, frames)

            interpolation = InterpolationResult(source_label_idx=source_label, target_label_idx=destination_label, frames=[])
            for alpha in reversed(alphas_values):
                interpolated_embeddings = alpha * source_embeddings + (1 - alpha) * target_embeddings
                output_frame = self.generator.decode_to_pil(interpolated_embeddings)
                for source_label, f in enumerate(output_frame):
                    interpolation.frames.append(f)
            res.append(interpolation)
        return res

    @torch.no_grad()
    def sample_interpolations(
        self,
        source_label: int,
        target_label: int,
        frames: int = 100,
        n: int = 1,
    ) -> InterpolationResult:
        source_embeddings = self.get_random_label_embeddings(source_label, n)
        target_embeddings = self.get_random_label_embeddings(target_label, n)
        alphas_values = np.linspace(0, 1, frames)
        interpolation = InterpolationResult(source_label_idx=source_label, target_label_idx=target_label, frames=[])
        for alpha in reversed(alphas_values):
            interpolated_embeddings = alpha * source_embeddings + (1 - alpha) * target_embeddings
            output_frame = self.generator.decode_to_pil(interpolated_embeddings)
            for source_label, f in enumerate(output_frame):
                interpolation.frames.append(f)
        return interpolation

    @torch.no_grad()
    def random_interpolation(
        self,
        source_label: int,
        destination_label: int,
        strength: float,
        batch_size: int,
    ) -> np.ndarray:
        if (not self._allow_label_flip) or (0 < strength < 1):
            source_embeddings = self.get_random_label_embeddings(source_label, batch_size)
            target_embeddings = self.get_random_label_embeddings(destination_label, batch_size)
            interpolated_embeddings = strength * target_embeddings + (1 - strength) * source_embeddings
            imgs = self.generator.decode_to_pil(interpolated_embeddings)
            return np.stack([np.array(x) for x in imgs])

        if strength == 0:
            return self.get_random_label_data(source_label, batch_size).numpy()
        elif strength == 1:
            return self.get_random_label_data(destination_label, batch_size).numpy()

        raise ValueError('You should not end up here')

    @torch.no_grad()
    def interpolate_with_random_dst(
        self,
        source: Tensor,
        destination_label: int,
        strength: float,
        batch_size: int,
    ) -> np.ndarray:
        if strength == 1 and self._allow_label_flip:
            return self.get_random_label_data(destination_label, batch_size).numpy()

        src_embeddings_idxs = np.setdiff1d(source, self.corrupted_indexes)
        self.corrupted_indexes.extend(map(int, src_embeddings_idxs))
        source_embeddings = self.embedding_layer(
            torch.from_numpy(src_embeddings_idxs)
        )

        if strength == 0:
            if self._allow_label_flip:
                if not self._dataset:
                    raise ValueError('Dataset not tracked. Is label_flip_on_strength_limit active?')
                return self._dataset.data[src_embeddings_idxs].numpy()
            interpolated_embeddings = source_embeddings
        else:
            target_embeddings = self.get_random_label_embeddings(destination_label, batch_size)
            interpolated_embeddings = strength * target_embeddings + (1 - strength) * source_embeddings
        imgs = self.generator.decode_to_pil(interpolated_embeddings)
        return np.stack([np.array(x) for x in imgs])
