from sys import maxsize
from collections import defaultdict
from typing import Dict
import random

import torch
from torch import Tensor

from dengine.dataset import SupervisedDataset
from dengine.dataset.utils import get_idxs_per_class
from dengine.graph import Graph
from dengine.graph.nx_graph import NXGraph

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

        for user in graph.nodes:
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


@register_partitioning()
def random_iid_balanced_exclude_hub(
    dataset: SupervisedDataset,
    *args,
    graph: NXGraph,
    **kwargs
) -> TYPE_DATASET_PARTITIONING:
    if not isinstance(graph, NXGraph):
        raise ValueError(f"Unsupported class {graph.__class__}, only graph.nx_graph.NXGraph is supported.")

    g = graph.nx_graph.copy()

    nodes_and_degrees = dict(g.degree())
    max_node = max(nodes_and_degrees, key=nodes_and_degrees.get)  # type: ignore
    g.remove_node(max_node)

    nx_g = NXGraph(experiment_cfg=graph._experiment_cfg, graph=g)
    partitioning = random_iid_balanced(dataset, *args, graph=nx_g, **kwargs)
    partitioning[f"{max_node}"] = {
        "train": Tensor([]),
        "validation": Tensor([])
    }
    return partitioning


@register_partitioning()
def subsample_classes_in_partitions(
    dataset: SupervisedDataset,
    test: SupervisedDataset,
    partitions: TYPE_DATASET_PARTITIONING,
    graph: Graph,
    class_sample_limits: Dict[str, int],
    *args,
    **kwargs
) -> TYPE_DATASET_PARTITIONING:
    target_classes_tensor = Tensor([int(x) for x in class_sample_limits.keys()])
    targets_to_keep_int = {int(k): v for k, v in class_sample_limits.items()}

    new_partition = defaultdict(dict)
    for client_key, data_distribution in partitions.items():
        for partition_key, partition_idx in data_distribution.items():
            if len(partition_idx) == 0:
                new_partition[client_key][partition_key] = partition_idx
                continue

            targets = dataset.targets[partition_idx]

            # Base mask: keep everything that is NOT a filtered class
            keep_mask = ~torch.isin(targets, target_classes_tensor)

            # Re-admit up to N samples for each filtered class
            for cls, num_to_keep in targets_to_keep_int.items():
                cls_positions = (targets == cls).nonzero(as_tuple=True)[0]
                selected = cls_positions[:num_to_keep]  # truncates if over budget
                keep_mask[selected] = True

            new_partition[client_key][partition_key] = partition_idx[keep_mask]

    return new_partition
