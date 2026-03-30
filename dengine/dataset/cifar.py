from typing import List, Tuple

import torch
import numpy as np
from torch import Tensor
import PIL.Image as pil
from PIL.Image import Image
from torchvision.datasets import CIFAR100, CIFAR10
from torchvision.transforms import (
    Compose,
    ToTensor,
    Normalize,
    RandomCrop,
    RandomHorizontalFlip,
    ColorJitter,
    RandomRotation,
    RandomErasing,
)

from .dataset import SupervisedDataset
from .utils import filter_integer_targets, balanance_class_samples, _unnormalize_tensor
from .decorators import register_dataset, register_postprocessing


CIFAR_TRANSFORM = Compose([
    ToTensor(),
    Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

CIFAR_TRAINING_TRANSFORM = Compose([
    RandomCrop(32, padding=4),
    RandomHorizontalFlip(),
    ColorJitter(0.2, 0.2, 0.2, 0.1),
    RandomRotation(30),
    ToTensor(),
    RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
    Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


def training_tensor_to_image(X: Tensor) -> List[Image]:
    X_inverse = (
        _unnormalize_tensor(X, CIFAR_TRANSFORM)
        .squeeze()
        .cpu()
        .numpy()
        .astype(np.uint8)
    )
    return [pil.fromarray(xi) for xi in X_inverse]


@register_dataset('cifar100')
def load_cifar100(
    train: bool,
    output_path: str,
    target_labels: List[int] = [],
    class_balance: bool = True,
    augmentations: bool = True,
    *args, **kwargs
) -> SupervisedDataset:
    if train and augmentations:
        transforms = CIFAR_TRAINING_TRANSFORM
    else:
        transforms = CIFAR_TRANSFORM

    cifar = CIFAR100(
        output_path,
        train=train,
        download=True,
        transform=transforms
    )
    dataset = SupervisedDataset(
        data=torch.from_numpy(cifar.data),
        targets=Tensor(cifar.targets).type(torch.int64),
        transform=cifar.transform
    )

    if class_balance:
        dataset = balanance_class_samples(dataset)

    if len(target_labels) == 0:
        return dataset
    dataset = filter_integer_targets(dataset, target_labels)
    return dataset


@register_dataset('cifar10')
def load_cifar10(
    train: bool,
    output_path: str,
    target_labels: List[int] = [],
    class_balance: bool = True,
    augmentations: bool = True,
    *args, **kwargs
) -> SupervisedDataset:
    if train and augmentations:
        transforms = CIFAR_TRAINING_TRANSFORM
    else:
        transforms = CIFAR_TRANSFORM

    cifar = CIFAR10(
        output_path,
        train=train,
        download=True,
        transform=transforms
    )
    dataset = SupervisedDataset(
        data=torch.from_numpy(cifar.data),
        targets=Tensor(cifar.targets).type(torch.int64),
        transform=cifar.transform
    )

    if class_balance:
        dataset = balanance_class_samples(dataset)

    if len(target_labels) == 0:
        return dataset
    dataset = filter_integer_targets(dataset, target_labels)
    return dataset


_toronto_edu_coarse_mapping = {
    "aquatic mammals": ["beaver", "dolphin", "otter", "seal", "whale"],
    "fish": ["aquarium_fish", "flatfish", "ray", "shark", "trout"],
    "flowers": ["orchid", "poppy", "rose", "sunflower", "tulip"],
    "food containers": ["bottle", "bowl", "can", "cup", "plate"],
    "fruit and vegetables": ["apple", "mushroom", "orange", "pear", "sweet_pepper"],
    "household electrical devices": ["clock", "keyboard", "lamp", "telephone", "television"],
    "household furniture": ["bed", "chair", "couch", "table", "wardrobe"],
    "insects": ["bee", "beetle", "butterfly", "caterpillar", "cockroach"],
    "large carnivores": ["bear", "leopard", "lion", "tiger", "wolf"],
    "large man-made outdoor things": ["bridge", "castle", "house", "road", "skyscraper"],
    "large natural outdoor scenes": ["cloud", "forest", "mountain", "plain", "sea"],
    "large omnivores and herbivores": ["camel", "cattle", "chimpanzee", "elephant", "kangaroo"],
    "medium-sized mammals": ["fox", "porcupine", "possum", "raccoon", "skunk"],
    "non-insect invertebrates": ["crab", "lobster", "snail", "spider", "worm"],
    "people": ["baby", "boy", "girl", "man", "woman"],
    "reptiles": ["crocodile", "dinosaur", "lizard", "snake", "turtle"],
    "small mammals": ["hamster", "mouse", "rabbit", "shrew", "squirrel"],
    "trees": ["maple_tree", "oak_tree", "palm_tree", "pine_tree", "willow_tree"],
    "vehicles 1": ["bicycle", "bus", "motorcycle", "pickup_truck", "train"],
    "vehicles 2": ["lawn_mower", "rocket", "streetcar", "tank", "tractor"]
}
_cifar100_classses = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver',
    'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl',
    'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can',
    'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee',
    'clock', 'cloud', 'cockroach', 'couch', 'crab',
    'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant',
    'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house',
    'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard',
    'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle',
    'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange',
    'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck',
    'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum',
    'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea',
    'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
    'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper',
    'table', 'tank', 'telephone', 'television', 'tiger', 'tractor',
    'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale',
    'willow_tree', 'wolf', 'woman', 'worm'
]
_CIFAR100_FINE_TO_COARSE_MAPPING = torch.zeros(100)
for _coarse_idx, (_, _fine_labels) in enumerate(_toronto_edu_coarse_mapping.items()):
    for _fine_label in _fine_labels:
        _fine_idx = _cifar100_classses.index(_fine_label)
        _CIFAR100_FINE_TO_COARSE_MAPPING[_fine_idx] = _coarse_idx


@register_postprocessing('cifar100_fine2coarse')
def cifar100_fine2coarse_postprocessing(
    dataset: SupervisedDataset,
    test: SupervisedDataset,
    *args, **kwargs
) -> Tuple[SupervisedDataset, SupervisedDataset]:
    dataset.targets = _CIFAR100_FINE_TO_COARSE_MAPPING[dataset.targets].type_as(dataset.targets)
    test.targets = _CIFAR100_FINE_TO_COARSE_MAPPING[test.targets].type_as(test.targets)
    return dataset, test
