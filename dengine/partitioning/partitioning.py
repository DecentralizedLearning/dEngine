from typing import Dict, Any, Union, List, Optional, Literal
from collections import defaultdict

import torch
from torch import Tensor
from copy import deepcopy
from torch import Generator

from dengine.dataset import SupervisedDataset


TYPE_PARTITION = Dict[Literal["train", "validation"], Tensor]
TYPE_DATASET_PARTITIONING = Dict[str, TYPE_PARTITION]


def dataset_partitions_with_rolling_probability(
    dataset: SupervisedDataset,
    probabilities: Tensor,
    minimum_class_samples: int = 0,
) -> Dict[str, Tensor]:
    partitions = defaultdict(list)
    unique_targets = torch.unique(dataset.targets).tolist()
    for i, class_id in enumerate(unique_targets):
        j_class_indexes = torch.where(dataset.targets == class_id)[0]
        user_probs = torch.roll(probabilities, i)

        offset = 0
        for current_user_index in range(len(user_probs)):
            partitions[current_user_index].append(
                j_class_indexes[offset:offset + minimum_class_samples]
            )
            offset = offset + minimum_class_samples
        j_class_cardinality_after_min_samples = j_class_indexes.shape[0] - offset

        for current_user_index, current_user_probability in enumerate(user_probs):
            user_sample_num = (
                (j_class_cardinality_after_min_samples * current_user_probability)
                .round().int()
            )
            partitions[current_user_index].append(
                j_class_indexes[offset:offset + user_sample_num]
            )
            offset = offset + user_sample_num
        partitions[current_user_index].append(
            j_class_indexes[offset:]
        )
    return {str(_k): torch.hstack(_x) for _k, _x in partitions.items()}


def split_partitions_into_train_and_test(
    val_split: float,
    partitions: Dict[Any, Tensor],
) -> TYPE_DATASET_PARTITIONING:
    new_data_partitions = {}
    train_split_percentage = 1 - val_split
    for user_id, user_partition_indexes in partitions.items():
        partition_size = len(user_partition_indexes)
        train_size = int(partition_size * train_split_percentage)
        randidx = torch.randperm(partition_size)

        train_idx = user_partition_indexes[randidx[:train_size]]
        val_idx = user_partition_indexes[randidx[train_size:]]

        new_data_partitions[user_id] = {
            'train': train_idx,
            'validation': val_idx
        }
    return new_data_partitions


def split_partitions_targets_into_train_and_test(
    dataset: SupervisedDataset,
    val_split: float,
    partitions: Dict[Any, Tensor],
) -> TYPE_DATASET_PARTITIONING:
    new_data_partitions = {}
    train_split_percentage = 1 - val_split
    unique_targets = torch.unique(dataset.targets).tolist()
    for user_id, user_partition_indexes in partitions.items():
        train_set_idxs = []
        test_set_idxs = []
        user_dataset_targets_view = dataset.targets[user_partition_indexes]
        for jth_class_label in unique_targets:
            jth_target_view_idxs = torch.where(user_dataset_targets_view == jth_class_label)[0]
            jth_samples_num = len(jth_target_view_idxs)
            train_size = int(jth_samples_num * train_split_percentage)

            randidx = torch.randperm(jth_samples_num)
            target_view_train_idx = jth_target_view_idxs[randidx[:train_size]]
            target_view_test_idx = jth_target_view_idxs[randidx[train_size:]]

            train_set_idxs.extend(
                user_partition_indexes[target_view_train_idx]
            )
            test_set_idxs.extend(
                user_partition_indexes[target_view_test_idx]
            )
        new_data_partitions[user_id] = {
            'train': torch.stack(train_set_idxs),
            'validation': torch.stack(test_set_idxs) if len(test_set_idxs) > 0 else torch.Tensor([])
        }
    return new_data_partitions


def _filter_partition(
    partitions: TYPE_DATASET_PARTITIONING,
    allowed_keys: Union[str, List[str]]
) -> TYPE_DATASET_PARTITIONING:
    if isinstance(allowed_keys, str):
        allowed_keys = [allowed_keys]

    filtered_partitions = {}
    for user_id, p in partitions.items():
        filtered_p = {}
        for key, tensor_indexes in p.items():
            if key not in allowed_keys:
                continue
            filtered_p[key] = tensor_indexes
        filtered_partitions[user_id] = filtered_p
    return filtered_partitions


def permute_paiv_training_partition(
    partitions: TYPE_DATASET_PARTITIONING,
    generator: Optional[Generator] = None,
) -> TYPE_DATASET_PARTITIONING:
    partitions = deepcopy(partitions)
    for p in partitions.values():
        train = p['train']
        index_permutation = torch.randperm(
            train.size()[0],
            generator=generator
        )
        p['train'] = train[index_permutation]
    return partitions


def partitions_length(
    partitions: TYPE_DATASET_PARTITIONING,
    keys: Optional[Union[str, List[str]]] = None
) -> int:
    if keys is not None:
        partitions = _filter_partition(partitions, keys)
    return len(merge_partitions_into_tensor(partitions))


def merge_partitions_into_tensor(partitions: TYPE_DATASET_PARTITIONING) -> Tensor:
    cumulative_partition = torch.concat([
        torch.concat(
            list(X.values())
        ) for X in partitions.values()
    ])
    return cumulative_partition


def clip_iid_partitions_samples(
    dataset: SupervisedDataset,
    partitioning: TYPE_DATASET_PARTITIONING,
    mapping: Dict[int, float] = {},
) -> TYPE_DATASET_PARTITIONING:
    for key in partitioning:  # train / validation
        idxs = partitioning[0][key]
        max_class_available_samples = torch.sum(
            dataset.targets[idxs] == dataset.unique_targets[0]
        )
        for device_id, device_partition in partitioning.items():
            counters = [0] * len(dataset.unique_targets)
            new_ith_partition = []

            for ith_idx in device_partition[key]:
                ith_target = dataset.targets[ith_idx]
                mapping_target = str(int(ith_target))
                if mapping_target not in mapping.keys():
                    new_ith_partition.append(ith_idx)
                    continue

                allowed_samples = int(mapping[mapping_target] * max_class_available_samples)
                if counters[ith_target] <= allowed_samples:
                    new_ith_partition.append(ith_idx)
                    counters[ith_target] += 1
                    continue
            partitioning[device_id][key] = torch.stack(new_ith_partition)
    return partitioning
