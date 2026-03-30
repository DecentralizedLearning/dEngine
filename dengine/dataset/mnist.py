from typing import List

import PIL.Image as pil
from PIL.Image import Image
import numpy as np
from torch import Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize

from .utils import filter_integer_targets, balanance_class_samples, _unnormalize_tensor
from .dataset import SupervisedDataset
from .decorators import register_dataset


_MNIST_TRANSFORM = Compose([
    ToTensor(),
    Normalize((0.5,), (0.5,))
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
    *args, **kwargs
) -> SupervisedDataset:
    mnist = MNIST(output_path, train=train, download=True, transform=_MNIST_TRANSFORM)
    assert 0 < subset_fraction <= 1
    end = int(len(mnist.data) * subset_fraction)
    dataset = SupervisedDataset(
        data=mnist.data[:end],
        targets=mnist.targets[:end],
        transform=mnist.transform
    )

    if class_balance:
        dataset = balanance_class_samples(dataset)

    if len(target_labels) == 0:
        return dataset
    dataset = filter_integer_targets(dataset, target_labels)
    return dataset
