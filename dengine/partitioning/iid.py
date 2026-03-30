from sys import maxsize
from collections import defaultdict
import random

import torch
from torch import Tensor

from dengine.dataset import SupervisedDataset
from dengine.dataset.utils import get_idxs_per_class
from dengine.graph import Graph

from .partitioning import TYPE_DATASET_PARTITIONING, split_partitions_into_train_and_test
from .decorators import register_partitioning


@register_partitioning()
def random_iid(
    dataset: SupervisedDataset,
    *args,
    graph: Graph,
    validation_percentage: float,
    **kwargs
) -> TYPE_DATASET_PARTITIONING:
    num_users = len(graph.nodes)
    num_items = len(dataset) // num_users
    partitions = {}
    all_idxs = set(range(len(dataset)))

    for i in range(num_users):
        ith_user_idxs = random.sample(list(all_idxs), num_items)
        all_idxs -= set(ith_user_idxs)
        partitions[str(i)] = Tensor(ith_user_idxs).type(torch.int)

    train_test_partitions = split_partitions_into_train_and_test(validation_percentage, partitions)

    return train_test_partitions


@register_partitioning()
def random_iid_balanced(
    dataset: SupervisedDataset,
    *args,
    graph: Graph,
    validation_percentage: float,
    max_per_label_samples: int = maxsize,
    **kwargs
) -> TYPE_DATASET_PARTITIONING:
    num_users = len(graph.nodes)
    labels_with_idx = get_idxs_per_class(dataset)
    min_label_cardinality = min([len(samples) for samples in labels_with_idx.values()])
    samples_to_draw_per_label = min(min_label_cardinality // num_users, max_per_label_samples)

    labels_partitions = defaultdict(lambda: defaultdict(list))
    for idxs in labels_with_idx.values():
        ith_label_partitions = {}
        available_idxs = set(idxs)

        for user in map(str, range(num_users)):
            ith_user_idxs = random.sample(list(available_idxs), samples_to_draw_per_label)
            available_idxs -= set(ith_user_idxs)
            ith_label_partitions[user] = Tensor(ith_user_idxs).type(torch.int)

        ith_label_train_test_partitions = split_partitions_into_train_and_test(validation_percentage, ith_label_partitions)

        for user, train_test_idxs in ith_label_train_test_partitions.items():
            labels_partitions[user]["train"].append(train_test_idxs["train"])
            labels_partitions[user]["validation"].append(train_test_idxs["validation"])

    train_test_partitions = {}
    for user, train_test_idxs in labels_partitions.items():
        train_test_partitions[user] = {
            "train": torch.concatenate(train_test_idxs["train"]),
            "validation": torch.concatenate(train_test_idxs["validation"])
        }

    return train_test_partitions
