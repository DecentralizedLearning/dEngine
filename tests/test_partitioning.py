"""Tests for ``dengine.partitioning``: how a dataset is split among the nodes of a graph."""
from pathlib import Path
from typing import Any

import networkx as nx
import pytest
import torch
from torch import Tensor

from dengine.dataset import SupervisedDataset
from dengine.graph.nx_graph import NXGraph
from dengine.partitioning import TYPE_DATASET_PARTITIONING, dump_partition
from dengine.partitioning.decorators import BUILTIN_PARTITIONING, TPartitioning
from dengine.partitioning.iid import (
    random_iid,
    random_iid_balanced,
    random_iid_balanced_exclude_hub,
    subsample_classes_in_partitions,
)
from dengine.partitioning.noniid import (
    default_random_iid_balanced,
    random_iid_balanced_with_class_majority_hub,
    zipf_noniid_truncated,
)
from dengine.partitioning.partitioning import (
    dataset_partitions_with_rolling_probability,
    merge_partitions_into_tensor,
    partitions_length,
    permute_paiv_training_partition,
    split_partitions_into_train_and_test,
    split_partitions_targets_into_train_and_test,
)
from dengine.partitioning.utils import merge_dataset_and_partitions, nodes_sorted_by_class_size

from tests.helpers import all_indices, make_dataset, make_graph


def run_partitioner(fn: TPartitioning, dataset: SupervisedDataset, graph: NXGraph, **kwargs: Any) -> TYPE_DATASET_PARTITIONING:
    """Call a partitioner the way ``bin/simulation.py`` does."""
    return fn(dataset, test=dataset, partitions={}, graph=graph, **kwargs)


def labels(dataset: SupervisedDataset, idxs: Tensor) -> list[int]:
    """The labels of the given samples."""
    return dataset.targets[idxs.long()].tolist()


def two_node_partitioning() -> TYPE_DATASET_PARTITIONING:
    """A hand-written partitioning: 10 train and 3 validation indices in total."""
    return {
        "0": {"train": torch.arange(0, 6), "validation": torch.arange(6, 8)},
        "1": {"train": torch.arange(8, 12), "validation": torch.arange(12, 13)},
    }


# --- Train / validation splits ------------------------------------------------

class TestSplitIntoTrainAndValidation:
    """``split_partitions_into_train_and_test``: random split of each node's samples."""

    def test_respects_validation_percentage(self):
        split = split_partitions_into_train_and_test(0.2, {"0": torch.arange(10)})

        assert len(split["0"]["train"]) == 8
        assert len(split["0"]["validation"]) == 2

    def test_keeps_every_index_exactly_once(self):
        split = split_partitions_into_train_and_test(0.3, {"0": torch.arange(10)})

        assert sorted(all_indices(split)) == list(range(10))


class TestStratifiedSplit:
    """``split_partitions_targets_into_train_and_test``: same split, done class by class."""

    def test_every_class_appears_in_both_splits(self, dataset: SupervisedDataset):
        split = split_partitions_targets_into_train_and_test(dataset, 0.2, {"0": torch.arange(40)})

        assert labels(dataset, split["0"]["train"]).count(0) == 8
        assert labels(dataset, split["0"]["validation"]).count(0) == 2
        assert set(labels(dataset, split["0"]["validation"])) == {0, 1, 2, 3}


# --- Partitioning helpers -----------------------------------------------------

class TestPartitionHelpers:
    """Small utilities that count, merge or shuffle partitions."""

    def test_length_can_be_filtered_by_split(self):
        partitioning = two_node_partitioning()

        assert partitions_length(partitioning) == 13
        assert partitions_length(partitioning, "train") == 10
        assert partitions_length(partitioning, "validation") == 3

    def test_merge_concatenates_all_indices(self):
        merged = merge_partitions_into_tensor(two_node_partitioning())

        assert sorted(merged.tolist()) == list(range(13))

    def test_permute_shuffles_train_only_and_returns_a_copy(self):
        original = two_node_partitioning()

        permuted = permute_paiv_training_partition(original, generator=torch.Generator().manual_seed(0))

        assert torch.equal(original["0"]["train"], torch.arange(0, 6))
        assert sorted(permuted["0"]["train"].tolist()) == list(range(6))
        assert torch.equal(permuted["0"]["validation"], original["0"]["validation"])


class TestRollingProbability:
    """``dataset_partitions_with_rolling_probability``: class i favours node i."""

    def test_one_hot_probabilities_give_one_class_per_node(self, dataset: SupervisedDataset):
        partitions = dataset_partitions_with_rolling_probability(dataset, torch.tensor([1.0, 0.0, 0.0, 0.0]))

        for node in range(4):
            assert set(labels(dataset, partitions[str(node)])) == {node}

    def test_minimum_class_samples_go_to_every_node(self, dataset: SupervisedDataset):
        partitions = dataset_partitions_with_rolling_probability(
            dataset, torch.tensor([1.0, 0.0, 0.0, 0.0]), minimum_class_samples=2
        )

        for idxs in partitions.values():
            node_labels = labels(dataset, idxs)
            assert all(node_labels.count(c) >= 2 for c in range(4))


# --- IID partitioners ---------------------------------------------------------

class TestRandomIID:
    """``random_iid``: same number of random samples for every node."""

    def test_one_partition_per_node(self, dataset: SupervisedDataset, ring_graph: NXGraph):
        partitioning = run_partitioner(random_iid, dataset, ring_graph, validation_percentage=0.2)

        assert set(partitioning) == set(ring_graph.nodes)
        for split in partitioning.values():
            assert len(split["train"]) == 8      # 40 samples / 4 nodes = 10, 80/20 split
            assert len(split["validation"]) == 2

    def test_partitions_are_disjoint(self, dataset: SupervisedDataset, ring_graph: NXGraph):
        partitioning = run_partitioner(random_iid, dataset, ring_graph, validation_percentage=0.2)

        indices = all_indices(partitioning)
        assert len(indices) == len(set(indices)) == 40


class TestRandomIIDBalanced:
    """``random_iid_balanced``: every node gets the same amount of every class."""

    def test_same_class_counts_on_every_node(self, dataset: SupervisedDataset, ring_graph: NXGraph):
        partitioning = run_partitioner(random_iid_balanced, dataset, ring_graph, validation_percentage=0.0)

        for split in partitioning.values():
            node_labels = labels(dataset, split["train"])
            assert [node_labels.count(c) for c in range(4)] == [2, 2, 2, 2]  # 10 // 4 nodes

    def test_max_per_label_samples_caps_each_class(self, dataset: SupervisedDataset, ring_graph: NXGraph):
        partitioning = run_partitioner(
            random_iid_balanced, dataset, ring_graph, validation_percentage=0.0, max_per_label_samples=1
        )

        assert all(len(split["train"]) == 4 for split in partitioning.values())

    def test_exclude_hub_leaves_the_hub_empty(self, dataset: SupervisedDataset):
        star = make_graph(graph=nx.star_graph(3))  # node 0 is the hub

        partitioning = run_partitioner(random_iid_balanced_exclude_hub, dataset, star, validation_percentage=0.0)

        assert len(partitioning["0"]["train"]) == 0
        assert all(len(partitioning[n]["train"]) > 0 for n in ("1", "2", "3"))


# --- Non-IID partitioners -----------------------------------------------------

class TestZipfNonIID:
    """``zipf_noniid_truncated``: class shares follow a probability vector."""

    def test_prob_vec_assigns_every_sample_once(self, dataset: SupervisedDataset, ring_graph: NXGraph):
        partitioning = run_partitioner(
            zipf_noniid_truncated, dataset, ring_graph, validation_percentage=0.2, prob_vec=[0.4, 0.3, 0.2, 0.1]
        )

        assert sorted(all_indices(partitioning)) == list(range(40))

    def test_requires_zipf_alpha_or_prob_vec(self, dataset: SupervisedDataset, ring_graph: NXGraph):
        with pytest.raises(ValueError):
            run_partitioner(zipf_noniid_truncated, dataset, ring_graph, validation_percentage=0.2)


class TestClassMajorityHub:
    """``random_iid_balanced_with_class_majority_hub``: the hub gets most of some classes."""

    def test_hub_receives_requested_samples_of_the_class(self, dataset: SupervisedDataset):
        star = make_graph(graph=nx.star_graph(3))

        partitioning = run_partitioner(
            random_iid_balanced_with_class_majority_hub, dataset, star,
            validation_percentage=0.0, hub_training_samples=7, classes=0,
        )

        assert labels(dataset, partitioning["0"]["train"]).count(0) == 7
        assert all(labels(dataset, partitioning[n]["train"]).count(0) == 1 for n in ("1", "2", "3"))


class TestSubsampleClasses:
    """``subsample_classes_in_partitions``: caps the samples of selected classes."""

    def test_limits_only_the_selected_class(self, dataset: SupervisedDataset, ring_graph: NXGraph):
        balanced = run_partitioner(random_iid_balanced, dataset, ring_graph, validation_percentage=0.0)

        limited = subsample_classes_in_partitions(
            dataset, test=dataset, partitions=balanced, graph=ring_graph, class_sample_limits={"0": 1}
        )

        for split in limited.values():
            node_labels = labels(dataset, split["train"])
            assert node_labels.count(0) == 1
            assert node_labels.count(1) == 2


@pytest.mark.xfail(
    strict=True, raises=KeyError,
    reason="Bug: clip_iid_partitions_samples indexes partitioning[0] with an int and loops over nodes as split keys",
)
def test_default_random_iid_balanced_runs(dataset: SupervisedDataset, ring_graph: NXGraph):
    """``default_random_iid_balanced`` should behave like ``random_iid_balanced`` with an empty mapping."""
    partitioning = run_partitioner(default_random_iid_balanced, dataset, ring_graph, validation_percentage=0.0)

    assert set(partitioning) == set(ring_graph.nodes)


# --- Utilities -----------------------------------------------------------------

class TestPartitioningUtils:
    """Helpers in ``dengine.partitioning.utils``."""

    def test_merge_dataset_offsets_second_partitioning(self):
        first, second = make_dataset(samples_per_class=2), make_dataset(samples_per_class=1)
        empty = SupervisedDataset(torch.empty(0, 28, 28), torch.empty(0))
        no_validation = torch.tensor([], dtype=torch.long)
        part_a: TYPE_DATASET_PARTITIONING = {"0": {"train": torch.tensor([0, 1]), "validation": no_validation}}
        part_b: TYPE_DATASET_PARTITIONING = {"0": {"train": torch.tensor([0]), "validation": no_validation}}

        merged, partitioning = merge_dataset_and_partitions([first, second], [part_a, part_b], empty)

        assert len(merged) == len(first) + len(second)
        assert partitioning["0"]["train"].tolist() == [0, 1, len(first)]  # second dataset is shifted

    def test_nodes_sorted_by_class_size(self, dataset: SupervisedDataset):
        partitioning: TYPE_DATASET_PARTITIONING = {
            "a": {"train": torch.tensor([0, 1, 10]), "validation": torch.tensor([])},  # two samples of class 0
            "b": {"train": torch.tensor([2]), "validation": torch.tensor([])},         # one
            "c": {"train": torch.tensor([10, 11]), "validation": torch.tensor([])},    # none
        }

        assert nodes_sorted_by_class_size(dataset, partitioning, target_label=0, desc=True) == ["a", "b", "c"]

    def test_dump_partition_writes_class_counts(self, dataset: SupervisedDataset, tmp_path: Path):
        partitioning: TYPE_DATASET_PARTITIONING = {
            "0": {"train": torch.tensor([0, 1, 10]), "validation": torch.tensor([])},
        }

        dump_partition(tmp_path / "partition.csv", dataset, partitioning)

        rows = (tmp_path / "partition.csv").read_text().splitlines()
        assert rows == ["nodeid,0,1,2,3", "0,2,1,0,0"]


def test_builtin_partitioners_are_registered():
    """Partitioners are found by name when configs refer to them."""
    for name in ("random_iid", "random_iid_balanced", "zipf_noniid_truncated"):
        assert name in BUILTIN_PARTITIONING
