import logging
from typing import List

import torch
from torchvision.datasets import EMNIST
from torchvision.transforms import Compose
from torchvision.transforms import v2

from .utils import filter_integer_targets, balanance_class_samples
from .dataset import SupervisedDataset
from .decorators import register_dataset


def _emnist_trasnform_by_split(split: str):
    """
    Per-split EMNIST normalization constants (mean, std), in float32.

    Computed with:
        dEngine/bin/torchvision_dataset_normalization.py \
            --dataset EMNIST \
            --dataset_output_path datasets/ \
            --precision float32 \
            split=<split>

    Environment:
        Python 3.13.11
        torch==2.15.0+cpu
        torchvision==0.29.2+cpu
        dengine-vit-pytorch @ git+https://github.com/DecentralizedLearning/dengine-vit-pytorch.git@f09a28ae2c014275168895d0f4364bdc74d5f8c6
    """
    return Compose([
        v2.ToDtype(torch.float32, scale=True),
        {
            "byclass": v2.Normalize((0.1307,), (0.3081,)),
            "bymerge": v2.Normalize((0.1736,), (0.3317,)),
            "balanced": v2.Normalize((0.1752,), (0.3333,)),
            "letters": v2.Normalize((0.1722,), (0.3309,)),
            "digits": v2.Normalize((0.1733,), (0.3317,)),
            "mnist": v2.Normalize((0.1733,), (0.3317,))
        }[split]
    ])


def _patch_torchvision_emnist_urls():
    """https://github.com/pytorch/vision/blob/5181a854d8b127cf465cd22a67c1b5aaf6ccae05/torchvision/datasets/mnist.py#L279
    """
    updated_url = 'https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip'
    if EMNIST.url != updated_url:
        EMNIST.url = updated_url
        logging.warning(
            'Found unexpected url for emnist dataset, this is due to torchvision <= v0.17.2. '
            'Patching the url with "https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip"'
        )


_patch_torchvision_emnist_urls()


@register_dataset('emnist')
def load_emnist(
    train: bool,
    output_path: str,
    target_labels: List[int] = [],
    class_balance: bool = True,
    split: str = 'letters',
    use_cache: bool = False,
    *args, **kwargs
) -> SupervisedDataset:
    emnist = EMNIST(output_path, split=split, train=train, download=True, transform=_emnist_trasnform_by_split(split))
    dataset = SupervisedDataset(
        data=emnist.data.unsqueeze(1),
        targets=emnist.targets,
        transform=emnist.transform,
        use_cache=use_cache
    )

    if class_balance:
        dataset = balanance_class_samples(dataset)

    if len(target_labels) == 0:
        return dataset
    dataset = filter_integer_targets(dataset, target_labels)
    return dataset
