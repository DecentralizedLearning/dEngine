from typing import List

import PIL.Image as pil
from PIL.Image import Image
import numpy as np
import torch
from torch import Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import Compose
from torchvision.transforms import v2

from .utils import filter_integer_targets, balanance_class_samples, _unnormalize_tensor
from .dataset import SupervisedDataset
from .decorators import register_dataset


"""
MNIST normalization constants (mean, std), in float32.

Computed 2026-11-19 with:
    python bin/torchvision_dataset_normalization.py \
        --dataset MNIST \
        --dataset_output_path datasets/ \
        --precision float32

Environment:
    Python 3.13.11
    torch==2.15.0+cpu
    torchvision==0.29.2+cpu
    dengine-vit-pytorch @ git+https://github.com/DecentralizedLearning/dengine-vit-pytorch.git@f09a28ae2c014275168895d0f4364bdc74d5f8c6
"""
_MNIST_TRANSFORM = Compose([
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize((0.1309,), (0.3084,)),
])


def training_tensor_to_image(X: Tensor) -> List[Image]:
    X_inverse = (
        _unnormalize_tensor(X, _MNIST_TRANSFORM)
        .squeeze()
        .cpu()
        .numpy()
        .astype(np.uint8)
    )
    return [pil.fromarray(xi) for xi in X_inverse]


@register_dataset('mnist')
def load_mnist(
    train: bool,
    output_path: str,
    target_labels: List[int] = [],
    class_balance: bool = True,
    subset_fraction: float = 1,
    use_cache: bool = False,
    *args, **kwargs
) -> SupervisedDataset:
    mnist = MNIST(output_path, train=train, download=True, transform=_MNIST_TRANSFORM)
    assert 0 < subset_fraction <= 1
    end = int(len(mnist.data) * subset_fraction)
    dataset = SupervisedDataset(
        data=mnist.data[:end].unsqueeze(1),
        targets=mnist.targets[:end],
        transform=mnist.transform,
        use_cache=use_cache
    )

    if class_balance:
        dataset = balanance_class_samples(dataset)

    if len(target_labels) == 0:
        return dataset
    dataset = filter_integer_targets(dataset, target_labels)
    return dataset
