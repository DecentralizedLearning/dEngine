"""CPU-only regression tests for built-in dataset allocation."""
import random
import unittest

import numpy as np
import torch

from dengine.graph import NXGraph
from dengine.partitioning import (
    random_iid, random_iid_balanced,
    random_iid_balanced_with_class_majority_hub,
    subsample_classes_in_partitions, zipf_noniid_truncated,
)
from dengine.tests.mock import MockSupervisedDataset


class BuiltinPartitioningTests(unittest.TestCase):
    def setUp(self):
        self.python_state = random.getstate()
        self.numpy_state = np.random.get_state()
        self.torch_state = torch.get_rng_state()
        self.addCleanup(random.setstate, self.python_state)
        self.addCleanup(np.random.set_state, self.numpy_state)
        self.addCleanup(torch.set_rng_state, self.torch_state)
        self.seed()
        self.dataset = MockSupervisedDataset(samples=120, data_size=(1,), unique_targets=3)
        # Explicit labels eliminate random class imbalance in the mock.
        self.dataset.targets[:] = torch.arange(120) % 3
        self.graph = NXGraph(experiment_cfg=None, nx_class="star_graph", n=3)

    @staticmethod
    def seed():
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)

    def allocate(self, algorithm, **kwargs):
        return algorithm(dataset=self.dataset, test=self.dataset, partitions={},
                         graph=self.graph, validation_percentage=.2, **kwargs)

    def assert_allocation(self, result, count):
        self.assertEqual(set(result), set(self.graph.nodes))
        seen = set()
        for splits in result.values():
            self.assertEqual(set(splits), {"train", "validation"})
            for indices in splits.values():
                self.assertEqual(indices.device.type, "cpu")
                values = indices.tolist()
                self.assertEqual(len(values), len(set(values)))
                self.assertTrue(all(isinstance(i, int) and 0 <= i < 120 for i in values))
                self.assertTrue(seen.isdisjoint(values))
                seen.update(values)
        self.assertEqual(len(seen), count)
        if count == 120:
            self.assertEqual(seen, set(range(120)))

    def test_random_iid_disjoint_complete_and_repeatable(self):
        self.seed()
        first = self.allocate(random_iid)
        self.assert_allocation(first, 120)
        for splits in first.values():
            self.assertEqual(len(splits["train"]), 24)
            self.assertEqual(len(splits["validation"]), 6)
        self.seed()
        second = self.allocate(random_iid)
        for node in first:
            for split in first[node]:
                self.assertTrue(torch.equal(first[node][split], second[node][split]))

    def test_balanced_counts_and_sample_cap(self):
        for cap, count in ((10, 120), (5, 60)):
            with self.subTest(cap=cap):
                result = self.allocate(random_iid_balanced, max_per_label_samples=cap)
                self.assert_allocation(result, count)
                for splits in result.values():
                    for split, expected in (("train", int(cap * .8)), ("validation", cap - int(cap * .8))):
                        labels = self.dataset.targets[splits[split]]
                        self.assertEqual(torch.bincount(labels, minlength=3).tolist(), [expected] * 3)

    def test_hub_majority_preserves_coverage(self):
        result = self.allocate(random_iid_balanced_with_class_majority_hub,
                               classes=[0], hub_training_samples=25)
        self.assert_allocation(result, 120)
        # Pre-split class-0 counts: hub gets 25; (40 - 25) / 3 = 5 per leaf.
        for node, splits in result.items():
            indices = torch.cat(list(splits.values()))
            self.assertEqual(int((self.dataset.targets[indices] == 0).sum()), 25 if node == "0" else 5)

    def test_probability_vector_preserves_coverage(self):
        result = self.allocate(zipf_noniid_truncated, prob_vec=[1, 2, 3, 4])
        self.assert_allocation(result, 120)
        # Each class has 40 examples: normalized weights give 4, 8, 12, 16.
        # Rolling one position per class changes which node receives each count.
        expected = {
            "0": [4, 16, 12], "1": [8, 4, 16],
            "2": [12, 8, 4], "3": [16, 12, 8],
        }
        for node, splits in result.items():
            indices = torch.cat(list(splits.values()))
            counts = torch.bincount(self.dataset.targets[indices], minlength=3)
            self.assertEqual(counts.tolist(), expected[node])

    def test_subsample_keeps_unfiltered_classes_and_split_boundaries(self):
        original = self.allocate(random_iid_balanced)
        snapshot = {node: {key: value.clone() for key, value in splits.items()}
                    for node, splits in original.items()}
        result = subsample_classes_in_partitions(
            dataset=self.dataset, test=self.dataset, partitions=original,
            graph=self.graph, class_sample_limits={"0": 1})
        for node, splits in result.items():
            for key, indices in splits.items():
                source = snapshot[node][key]
                labels = self.dataset.targets[source]
                # Both original splits exceed the limit (8 train, 2 validation).
                self.assertGreater(int((labels == 0).sum()), 1)
                result_labels = self.dataset.targets[indices]
                self.assertEqual(int((result_labels == 0).sum()), 1)
                self.assertEqual(len(indices), len(source) - int((labels == 0).sum()) + 1)
                self.assertTrue(torch.equal(indices[result_labels != 0], source[labels != 0]))
                positions = {value: position for position, value in enumerate(source.tolist())}
                self.assertTrue(all(value in positions for value in indices.tolist()))
                retained_positions = [positions[value] for value in indices.tolist()]
                self.assertEqual(retained_positions, sorted(set(retained_positions)))
                self.assertTrue(torch.equal(original[node][key], source))


if __name__ == "__main__":
    unittest.main()
