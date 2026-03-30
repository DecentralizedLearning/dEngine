from typing import List, TypeVar, Tuple, Any
from pathlib import Path
import csv
from collections import defaultdict

import torch
from torch.utils.data import Dataset

from dengine.dataset import SupervisedDataset
from dengine.dataset.utils import merge_into_empty_dataset
from dengine.utils.utils import tensor_report

from .partitioning import TYPE_DATASET_PARTITIONING


def partitions_report(
    dataset: SupervisedDataset,
    partitions: TYPE_DATASET_PARTITIONING
) -> str:
    lines = []
    lines.append(tensor_report(dataset.targets))

    for id, data in partitions.items():
        for key, idxs in data.items():
            if len(idxs) == 0:
                lines.append(f'Empty set: {key}')
                continue
            key_report = tensor_report(dataset.targets[idxs], f'### PAIV-{id} {key}')
            lines.append(key_report)
        lines.append('---')
    return '\n'.join(lines)


T = TypeVar('T')


def merge_dataset_and_partitions(
    datasets: List[Dataset],
    partitions: List[TYPE_DATASET_PARTITIONING],
    dataset: T,
) -> Tuple[T, TYPE_DATASET_PARTITIONING]:
    merged_dataset = merge_into_empty_dataset(datasets, dataset)

    merged_partition: Any = defaultdict(
        lambda: defaultdict(list)
    )
    offset = 0
    for ith_dataset, ith_partition in zip(datasets, partitions):
        for user, dataset_split in ith_partition.items():
            for key, items in dataset_split.items():
                merged_partition[user][key].append(items + offset)
        offset += len(ith_dataset)

    for user, dataset_split in merged_partition.items():
        merged_partition[user] = {k: torch.concat(v) for k, v in dataset_split.items()}

    return merged_dataset, merged_partition


def dump_partition(
    fpath: Path,
    dataset_train: SupervisedDataset,
    data_partitions: TYPE_DATASET_PARTITIONING,
):
    fpath.parent.mkdir(parents=True, exist_ok=True)

    with open(fpath, 'w') as f:
        wr = csv.writer(f)
        wr.writerow(["nodeid"] + [str(i) for i in dataset_train.unique_targets.tolist()])

        for key, idxs in data_partitions.items():
            labels, counters = torch.unique(
                dataset_train.targets[idxs['train']],
                return_counts=True
            )
            all_labels_at_zero = dict(zip(labels.tolist(), counters.tolist()))

            # Needed because not all labels are assigned (e.g., EMNIST 'N/A' not in train)
            res = dict.fromkeys(dataset_train.unique_targets.tolist(), 0) | all_labels_at_zero

            wr.writerow([key] + list(res.values()))


def nodes_sorted_by_class_size(
    dataset: SupervisedDataset,
    train_test_partitions: TYPE_DATASET_PARTITIONING,
    target_label: int,
    desc: bool,
) -> List[str]:
    majority_class_cardinality = []
    for client_id, dataset_idxs in train_test_partitions.items():
        Y = dataset.targets[dataset_idxs["train"]]
        counts = len(Y[Y == target_label])
        majority_class_cardinality.append((client_id, counts))
    sorted_indexed_list = sorted(majority_class_cardinality, key=lambda x: x[1], reverse=desc)
    return [nodeid for nodeid, count in sorted_indexed_list]
