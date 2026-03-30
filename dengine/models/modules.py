from __future__ import annotations

from math import log2, ceil

import torch
from torch import nn, Tensor


class NormGeluDownScalingBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self._downscaling_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(num_groups=1, num_channels=out_channels, eps=1e-6, affine=True)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self._downscaling_block(x)


class DownScalingBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, batch_normalization: bool = True):
        super().__init__()
        upscaling_block = [
            nn.Conv2d(input_channels, output_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        if batch_normalization:
            upscaling_block.append(nn.BatchNorm2d(output_channels, momentum=0.8))
        self._upscaling_block = nn.Sequential(*upscaling_block)

    def forward(self, x: Tensor) -> Tensor:
        return self._upscaling_block(x)


class PatchDiscriminator(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        input_channels: int,
        encoder_spreadout_channels,
        encoder_output_channels,
    ):
        super().__init__()

        num_downscaling_blocks = log2(input_size / output_size)
        assert num_downscaling_blocks.is_integer()
        num_downscaling_blocks = ceil(num_downscaling_blocks)

        layers_channels = []
        max_power = int(log2(encoder_output_channels))
        min_power = int(log2(encoder_spreadout_channels))
        for i in range(num_downscaling_blocks):
            layers_channels.append(
                max_power - i % (max_power - min_power + 1)
            )
        layers_channels = [2 ** x for x in sorted(layers_channels)]
        layers_channels.append(2 ** max_power)

        downscaling_blocks = []
        for block_id in range(len(layers_channels) - 1):
            ch_prev = layers_channels[block_id]
            ch_next = layers_channels[block_id + 1]
            downscaling_blocks.append(DownScalingBlock(ch_prev, ch_next, batch_normalization=(block_id != 0)))

        self.input_to_encoder_input_mapping = nn.Conv2d(
            input_channels,
            layers_channels[0],
            kernel_size=3,
            padding=1
        )
        self.encoder = nn.Sequential(*downscaling_blocks)
        self.output_features = nn.Sequential(
            nn.Conv2d(in_channels=layers_channels[-1], out_channels=1, kernel_size=3, padding=1),
        )

    def forward(self, img: Tensor):
        x = self.input_to_encoder_input_mapping(img)
        out = self.encoder(x)
        validity = self.output_features(out)
        return validity


class UpScalingBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int):
        super().__init__()
        self._upscaling_block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_channels, momentum=0.8),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self._upscaling_block(x)


class Nonlinear2DClassificationHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_dimension: int
    ) -> None:
        super().__init__()
        self._head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.GroupNorm(num_groups=1, num_channels=in_channels, eps=1e-3, affine=True),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_channels, out_dimension),
        )

    def forward(self, x: Tensor):
        return self._head(x)


class LayerScale(nn.Module):
    def __init__(self, dim, init_value=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(
            init_value * torch.ones((dim, 1, 1))
        )

    def forward(self, x):
        return x * self.gamma


class DepthWiseSkipConvolution(nn.Module):
    def __init__(
        self,
        in_channels: int,
        projection_factor: int,
    ):
        super().__init__()
        hidden_size = in_channels * projection_factor
        self._layers = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                groups=in_channels,
                out_channels=hidden_size,
                kernel_size=3,
                padding='same'
            ),
            nn.GroupNorm(num_groups=1, num_channels=hidden_size, eps=1e-3, affine=True),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Conv2d(hidden_size, in_channels, kernel_size=1, bias=True),
            LayerScale(in_channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self._layers(x) + x
