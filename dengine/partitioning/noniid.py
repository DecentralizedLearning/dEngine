import math
from typing import Optional, Iterable, Union, Dict
import random
from sys import maxsize
from collections import defaultdict

import numpy as np
import torch
from torch import Tensor

from dengine.dataset import SupervisedDataset
from dengine.graph import Graph
from dengine.dataset.utils import get_idxs_per_class
from dengine.graph.utils import nodes_sorted_by_neighbor_count

from .partitioning import (
    TYPE_DATASET_PARTITIONING,
    dataset_partitions_with_rolling_probability,
    split_partitions_targets_into_train_and_test,
    split_partitions_into_train_and_test,
    clip_iid_partitions_samples
)
from .iid import random_iid_balanced
from .decorators import register_partitioning


def _get_zipf_probabilities(
    dataset: SupervisedDataset,
    zipf_alpha: float,
    graph: Graph,
    *args,
    **kwargs
) -> Tensor:
    """We generate a zipfs sample but we keep it only if its sum is <= than the number of images in the least numerous class
    (otherwise we cannot guarantee that each node has got >=1 image for each class)
    TO DO: check what is the minimum allowed
    """
    num_users = len(graph.nodes)
    _, items_per_class = torch.unique(dataset.targets, return_counts=True)
    items_per_class = items_per_class.tolist()

    count_discarded = 0
    while True:
        zipfs_samples = np.random.zipf(zipf_alpha, num_users)
        if np.sum(zipfs_samples) <= min(items_per_class):
            break
        else:
            count_discarded = count_discarded + 1
    return torch.tensor(zipfs_samples)


@register_partitioning()
def zipf_noniid_truncated(
    dataset: SupervisedDataset,
    *args,
    graph: Graph,
    validation_percentage: float,
    zipf_alpha: Optional[float] = None,
    prob_vec: Optional[Iterable[float]] = None,
    minimum_training_samples_per_class: int = 0,
    seed: Optional[int] = None,
    **kwargs
) -> TYPE_DATASET_PARTITIONING:
    """
    Sample non-I.I.D client data from {args.dataset} dataset on a subset of classes
    """
    # Compute zipf probability vector
    if seed:
        np.random.seed(seed)

    if zipf_alpha is not None:
        user_probabilities = _get_zipf_probabilities(dataset=dataset, graph=graph, zipf_alpha=zipf_alpha)
    elif prob_vec is not None:
        user_probabilities = torch.tensor(prob_vec)
    else:
        raise ValueError('Unable to compute probabilities, specify either zipf_alpha or prob_vector')
    rescaled_probs = user_probabilities / torch.sum(user_probabilities, dim=0)

    # Partition the dataset between users
    partitions = dataset_partitions_with_rolling_probability(
        dataset,
        rescaled_probs,
        minimum_class_samples=math.ceil(
            minimum_training_samples_per_class / (1 - validation_percentage)
        )
    )

    # Train + Test split
    train_test_partitions = split_partitions_targets_into_train_and_test(
        dataset,
        validation_percentage,
        partitions
    )

    return train_test_partitions


@register_partitioning()
def random_iid_balanced_with_class_majority_hub(
    dataset: SupervisedDataset,
    *args,
    graph: Graph,
    validation_percentage: float,
    hub_training_samples: Union[int, float],
    classes: Union[Iterable[int], int],
    max_per_label_samples: int = maxsize,
    **kwargs
) -> TYPE_DATASET_PARTITIONING:
    if isinstance(classes, int):
        classes = [classes]
    num_users = len(graph.nodes)
    hub_id = nodes_sorted_by_neighbor_count(graph)[0]
    labels_with_idx = get_idxs_per_class(dataset)
    min_label_cardinality = min([len(samples) for samples in labels_with_idx.values()])

    # 1. Samples to draw for uniform distribution
    samples_to_draw_per_label = min(min_label_cardinality // num_users, max_per_label_samples)

    # 2. Iterate over key and related indexes to populate labels_partitions
    labels_partitions = defaultdict(lambda: defaultdict(list))
    for key, idxs in labels_with_idx.items():
        ith_label_partitions = {}
        available_idxs = set(idxs)

        users = set(map(str, range(num_users)))
        samples_to_draw_per_user = samples_to_draw_per_label

        # 2.1 Non-iid class, we need to use hub_training_samples
        if key in classes:
            if isinstance(hub_training_samples, int):
                hub_training_samples_to_draw = hub_training_samples
            else:
                hub_training_samples_to_draw = int(len(available_idxs) * hub_training_samples)
            hub_idxs = random.sample(list(available_idxs), hub_training_samples_to_draw)
            available_idxs -= set(hub_idxs)
            ith_label_partitions[hub_id] = Tensor(hub_idxs).type(torch.int)
            users.remove(hub_id)
            samples_to_draw_per_user = len(available_idxs) // (num_users - 1)

        # 2.2 Distribute samples uniformaly accross the nodes
        for user in users:
            ith_user_idxs = random.sample(list(available_idxs), samples_to_draw_per_user)
            available_idxs -= set(ith_user_idxs)
            ith_label_partitions[user] = Tensor(ith_user_idxs).type(torch.int)

        # 2.3 Final train/validation spli
        ith_label_train_test_partitions = split_partitions_into_train_and_test(validation_percentage, ith_label_partitions)
        for user, train_test_idxs in ith_label_train_test_partitions.items():
            labels_partitions[user]["train"].append(train_test_idxs["train"])
            labels_partitions[user]["validation"].append(train_test_idxs["validation"])

    # 3. Final partition concatenation
    train_test_partitions = {}
    for user, train_test_idxs in labels_partitions.items():
        train_test_partitions[user] = {
            "train": torch.concatenate(train_test_idxs["train"]),
            "validation": torch.concatenate(train_test_idxs["validation"])
        }
    return train_test_partitions


@register_partitioning()
def default_random_iid_balanced(
    dataset: SupervisedDataset,
    *args,
    graph: Graph,
    validation_percentage: float,
    mapping: Dict[int, float] = {},
    **kwargs
) -> TYPE_DATASET_PARTITIONING:
    partitioning = random_iid_balanced(
        dataset=dataset,
        validation_percentage=validation_percentage,
        graph=graph,
        *args,
        **kwargs
    )
    return clip_iid_partitions_samples(dataset, partitioning, mapping)


@register_partitioning()
def default_random_iid_balanced_with_class_majority_hub(
    dataset: SupervisedDataset,
    *args,
    graph: Graph,
    validation_percentage: float,
    hub_training_samples: Union[int, float],
    classes: Union[Iterable[int], int],
    max_per_label_samples: int = maxsize,
    mapping: Dict[int, float] = {},
    **kwargs
) -> TYPE_DATASET_PARTITIONING:
    partitioning = random_iid_balanced_with_class_majority_hub(
        dataset=dataset,
        validation_percentage=validation_percentage,
        graph=graph,
        classes=classes,
        hub_training_samples=hub_training_samples,
        max_per_label_samples=max_per_label_samples,
        *args,
        **kwargs
    )
    return clip_iid_partitions_samples(dataset=dataset, partitioning=partitioning, mapping=mapping)
