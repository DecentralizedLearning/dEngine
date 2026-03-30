from typing import Optional, Sequence, TypeVar, List
from copy import deepcopy
from collections import defaultdict
import logging

import torch
from torch.utils.data import Dataset, Subset
from torchvision.transforms import Compose, Normalize
from torch import Tensor

from .dataset import SupervisedDataset


def get_idxs_per_class(dataset: SupervisedDataset):
    class_idxs = {}
    unique_values = torch.unique(dataset.targets)
    for label in unique_values:
        class_idxs[label.item()] = (dataset.targets == label).nonzero().squeeze().tolist()
    return class_idxs


T = TypeVar('T', bound=SupervisedDataset)


def filter_integer_targets(
    dataset: T,
    allowed_targets: Optional[Sequence[int]] = None,
    exclude_targets: Optional[Sequence[int]] = None
) -> T:
    if len(dataset) == 0:
        logging.warning('Empty dataset')
        return dataset

    if (allowed_targets is None) and (exclude_targets is None):
        logging.warning(
            'No filters specified, returning raw dataset'
        )
        return dataset

    if (allowed_targets is not None) and (exclude_targets is not None):
        raise ValueError('Arguments "exclude" and "include" are mutually exclusive')

    mask = torch.ones(len(dataset)).type(torch.bool)
    if allowed_targets:
        final_classes = [dataset.unique_targets[c] for c in allowed_targets]
        t_allowed_classes = torch.Tensor(allowed_targets)
        mask &= torch.isin(dataset.targets, t_allowed_classes)

    if exclude_targets:
        final_classes: List[str] = []
        for class_idx, class_label in enumerate(dataset.unique_targets):
            if class_idx in exclude_targets:
                continue
            final_classes.append(class_label)
        t_exclude_classes = torch.Tensor(exclude_targets)
        mask &= ~torch.isin(dataset.targets, t_exclude_classes)

    allowed_classes_idxs = torch.where(mask)
    dataset.data = dataset.data[allowed_classes_idxs]
    dataset.targets = dataset.targets[allowed_classes_idxs]
    dataset.unique_targets = final_classes
    return dataset


TT = TypeVar('TT')


def merge_into_empty_dataset(
    datasets: List[Dataset],
    empty_dataset: TT,
    fill_missing_keys: bool = True
) -> TT:
    dataset_tensors_shapes = {}
    for data in datasets:
        for key, item in vars(data).items():
            if not isinstance(item, Tensor):
                continue

            # Note: we exclude dim zero since we are going to concat
            item_shape = item.shape[1:]
            if key in dataset_tensors_shapes:
                assert not (
                    dataset_tensors_shapes[key] != item_shape and
                    len(item) != 0 and
                    len(dataset_tensors_shapes[key]) != 0
                )
            dataset_tensors_shapes[key] = item_shape

    merged_dataset = defaultdict(list)
    for key, shape in dataset_tensors_shapes.items():
        for i, data in enumerate(datasets):
            item = getattr(data, key, None)
            if item is not None:
                merged_dataset[key].append(item)
                continue
            if not fill_missing_keys:
                raise ValueError(
                    f"Datasets have different keys: {key} "
                    f"is missing in dataset at position {i}"
                )
            final_shape = (len(data), *shape)
            item = torch.zeros(final_shape) + torch.nan
            merged_dataset[key].append(item)

    dataset_tensors_dict = {key: torch.concat(X) for key, X in merged_dataset.items()}
    for key, value in dataset_tensors_dict.items():
        setattr(empty_dataset, key, value)
    return empty_dataset


def balanance_class_samples(dataset: SupervisedDataset):
    dataset = deepcopy(dataset)
    unique_targets, counts = dataset.targets.unique(return_counts=True)
    min_count = min(counts)
    indxs = []
    for y in unique_targets:
        y_idxs = torch.where(dataset.targets == y)[0][:min_count]
        indxs.append(y_idxs)
    indxs = torch.concat(indxs)
    dataset.data = dataset.data[indxs]
    dataset.targets = dataset.targets[indxs]
    return dataset


def _unnormalize_tensor(X: Tensor, transform: Compose) -> Tensor:
    normalization_layers = [t for t in transform.transforms if isinstance(t, Normalize)]
    if len(normalization_layers) == 0:
        return X
    elif len(normalization_layers) > 1:
        raise ValueError('Multiple normalizations layers are not supported')

    normalization = normalization_layers[0]
    mean = normalization.mean[0]
    std = normalization.std[0]

    inverse = Normalize((-mean / std,), (1.0 / std,))
    X_im = inverse(X) * 255
    return X_im.type(torch.uint8)


def get_targets(dataset: Subset) -> Tensor:
    return dataset.dataset.targets[dataset.indices]  # type: ignore


def get_data(dataset: Subset) -> Tensor:
    return dataset.dataset.data[dataset.indices]  # type: ignore
