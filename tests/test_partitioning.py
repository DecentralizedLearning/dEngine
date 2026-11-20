import random
import unittest

import torch
import yaml

from dengine import BUILTINS
from dengine.config import instantiate_configuration_module
from dengine.config.configuration import DynamicModuleConfigBase
from dengine.dataset import SupervisedDataset
from dengine.graph.nx_graph import NXGraph
from dengine.partitioning import TYPE_DATASET_PARTITIONING
from dengine.partitioning.decorators import BUILTIN_PARTITIONING


class TestBuiltinPartitioning(unittest.TestCase):
    def setUp(self):
        self.addCleanup(random.setstate, random.getstate())
        self.enterContext(torch.random.fork_rng(devices=[]))
        random.seed(42)
        torch.random.default_generator.manual_seed(42)

        self.dataset = SupervisedDataset(
            torch.zeros((24, 1)),
            torch.arange(2).repeat_interleave(12),
            root=None,
        )
        self.graph = NXGraph(experiment_cfg=None, nx_class="star_graph", n=2)

    def run_preset(self, preset, **overrides):
        raw = yaml.safe_load(preset.read_text(encoding="utf-8"))["partitioning"]
        config = DynamicModuleConfigBase(
            target=raw["target"],
            arguments={**raw.get("arguments", {}), **overrides},
        )
        partitions = instantiate_configuration_module(
            config,
            superclass=TYPE_DATASET_PARTITIONING,
            allowed_cls=BUILTIN_PARTITIONING,
            dataset=self.dataset,
            test=self.dataset,
            partitions={},
            graph=self.graph,
        )

        self.assertEqual(set(partitions), set(self.graph.nodes))
        indices = []
        for splits in partitions.values():
            self.assertEqual(set(splits), {"train", "validation"})
            for split in splits.values():
                self.assertIn(split.dtype, (torch.int32, torch.int64))
                indices.extend(split.tolist())
        self.assertEqual(sorted(indices), list(range(len(self.dataset))))
        return partitions

    def class_counts(self, indices):
        return self.dataset.targets[indices].bincount(minlength=2).tolist()

    def test_iid_balances_each_split_across_clients(self):
        partitions = self.run_preset(BUILTINS.CORE.PARTITIONING.IID)

        for node, splits in partitions.items():
            with self.subTest(node=node):
                self.assertEqual(self.class_counts(splits["train"]), [3, 3])
                self.assertEqual(self.class_counts(splits["validation"]), [1, 1])

    def test_iid_leaves_the_hub_empty(self):
        partitions = self.run_preset(BUILTINS.CORE.PARTITIONING.IID_BALANCED_EXCLUDE_HUB)

        self.assertEqual(self.graph.nx_graph.degree[0], 2)
        self.assertEqual(self.class_counts(partitions["0"]["train"]), [0, 0])
        self.assertEqual(self.class_counts(partitions["0"]["validation"]), [0, 0])
        for node in ("1", "2"):
            with self.subTest(node=node):
                self.assertEqual(self.class_counts(partitions[node]["train"]), [4, 4])
                self.assertEqual(self.class_counts(partitions[node]["validation"]), [2, 2])

    def test_hub_gets_the_requested_class_majority(self):
        partitions = self.run_preset(
            BUILTINS.CORE.PARTITIONING.IID_NONIID_HUB,
            hub_training_samples=8,
            classes=[0],
        )

        expected_train = {"0": [6, 3], "1": [1, 3], "2": [1, 3]}
        expected_validation = {"0": [2, 1], "1": [1, 1], "2": [1, 1]}
        for node, splits in partitions.items():
            with self.subTest(node=node):
                self.assertEqual(self.class_counts(splits["train"]), expected_train[node])
                self.assertEqual(self.class_counts(splits["validation"]), expected_validation[node])

    def test_noniid_rotates_class_preferences_across_clients(self):
        partitions = self.run_preset(
            BUILTINS.CORE.PARTITIONING.NONIID,
            zipf_alpha=None,
            prob_vec=[1, 2, 3],
            minimum_training_samples_per_class=0,
        )

        expected_train = {"0": [1, 4], "1": [3, 1], "2": [4, 3]}
        expected_validation = {"0": [1, 2], "1": [1, 1], "2": [2, 1]}
        for node, splits in partitions.items():
            with self.subTest(node=node):
                self.assertEqual(self.class_counts(splits["train"]), expected_train[node])
                self.assertEqual(self.class_counts(splits["validation"]), expected_validation[node])
