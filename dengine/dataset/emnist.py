import logging
from typing import List

from torchvision.datasets import EMNIST

from .mnist import _MNIST_TRANSFORM
from .utils import filter_integer_targets, balanance_class_samples
from .dataset import SupervisedDataset
from .decorators import register_dataset


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
    emnist = EMNIST(output_path, split=split, train=train, download=True, transform=_MNIST_TRANSFORM)
    dataset = SupervisedDataset(
        data=emnist.data,
        targets=emnist.targets,
        trasnform=emnist.transform,
        use_cache=use_cache
    )

    if class_balance:
        dataset = balanance_class_samples(dataset)

    if len(target_labels) == 0:
        return dataset
    dataset = filter_integer_targets(dataset, target_labels)
    return dataset
