"""Exercise the built-in dataset loaders with synthetic TorchVision data."""

import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from dengine.dataset import cifar, emnist, mnist
from dengine.dataset.decorators import BUILTIN_DATASETS
from dengine.dataset.utils import filter_integer_targets


def source_dataset(data, targets):
    def build(_root, *, transform, **_options):
        return SimpleNamespace(data=data, targets=targets, transform=transform)

    return build


class DatasetLoaderTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = directory.name

    def test_mnist_selects_the_requested_prefix_and_normalizes_grayscale(self):
        raw = torch.zeros((4, 28, 28), dtype=torch.uint8)
        raw[1].fill_(255)
        source = source_dataset(raw, torch.tensor([0, 1, 2, 3]))
        with patch.object(mnist, "MNIST", side_effect=source) as dataset_type:
            dataset = BUILTIN_DATASETS["mnist"](
                train=False, output_path=self.path,
                class_balance=False, subset_fraction=0.5,
            )

        self.assertEqual(dataset_type.call_args.args, (self.path,))
        self.assertFalse(dataset_type.call_args.kwargs["train"])
        self.assertTrue(dataset_type.call_args.kwargs["download"])
        self.assertEqual(len(dataset), 2)
        image, label = dataset[1]
        self.assertEqual(label, 1)
        self.assertEqual(image.dtype, torch.float32)
        torch.testing.assert_close(image, torch.full((1, 28, 28), (1 - 0.1309) / 0.3084))

    def test_emnist_uses_split_and_filters_by_actual_label(self):
        raw = torch.zeros((4, 28, 28), dtype=torch.uint8)
        source = source_dataset(raw, torch.tensor([1, 3, 3, 5]))
        with patch.object(emnist, "EMNIST", side_effect=source) as dataset_type:
            dataset = BUILTIN_DATASETS["emnist"](
                train=True, output_path=self.path,
                split="letters", target_labels=[3], class_balance=False,
            )

        self.assertEqual(dataset_type.call_args.kwargs["split"], "letters")
        self.assertTrue(dataset_type.call_args.kwargs["train"])
        self.assertEqual(dataset.targets.tolist(), [3, 3])
        self.assertEqual(dataset.unique_targets.tolist(), [3])
        image, label = dataset[0]
        self.assertEqual(label, 3)
        torch.testing.assert_close(image, torch.full((1, 28, 28), -0.1722 / 0.3309))

    def test_cifar_loaders_return_balanced_channel_first_images(self):
        raw = np.empty((5, 32, 32, 3), dtype=np.uint8)
        raw[:] = [255, 128, 0]
        targets = [0, 1, 0, 1, 0]
        expected = [
            (1 - 0.4914) / 0.2470,
            (128 / 255 - 0.4822) / 0.2435,
            -0.4465 / 0.2616,
        ]
        for name, class_name in (("cifar10", "CIFAR10"), ("cifar100", "CIFAR100")):
            with self.subTest(dataset=name):
                source = source_dataset(raw, targets)
                with patch.object(cifar, class_name, side_effect=source) as dataset_type:
                    dataset = BUILTIN_DATASETS[name](
                        train=False, output_path=self.path, augmentations=True, use_cache=True,
                    )

                self.assertIs(dataset_type.call_args.kwargs["transform"], cifar.CIFAR_TRANSFORM)
                self.assertEqual(torch.bincount(dataset.targets).tolist(), [2, 2])
                self.assertEqual(dataset.targets.dtype, torch.int64)
                image, label = dataset[0]
                self.assertEqual(label, int(dataset.targets[0]))
                self.assertEqual(image.shape, (3, 32, 32))
                torch.testing.assert_close(image[:, 0, 0], torch.tensor(expected))

    def test_cifar_training_augmentation_is_opt_in(self):
        source = source_dataset(np.zeros((1, 32, 32, 3), dtype=np.uint8), [0])
        for augment, expected in ((True, cifar.CIFAR_TRAINING_TRANSFORM), (False, cifar.CIFAR_TRANSFORM)):
            with self.subTest(augment=augment):
                with patch.object(cifar, "CIFAR10", side_effect=source) as dataset_type:
                    BUILTIN_DATASETS["cifar10"](
                        train=True, output_path=self.path, augmentations=augment,
                    )
                self.assertIs(dataset_type.call_args.kwargs["transform"], expected)

    def test_subsampling_can_return_an_empty_balanced_dataset(self):
        source = source_dataset(torch.zeros((1, 28, 28), dtype=torch.uint8), torch.tensor([4]))
        with patch.object(mnist, "MNIST", side_effect=source):
            dataset = BUILTIN_DATASETS["mnist"](
                train=True, output_path=self.path, subset_fraction=0.5,
            )
        self.assertEqual(len(dataset), 0)
        self.assertEqual(dataset.unique_targets.tolist(), [])

    def test_excluding_a_label_keeps_samples_and_cached_labels_aligned(self):
        raw = torch.zeros((3, 28, 28), dtype=torch.uint8)
        raw[1].fill_(255)
        source = source_dataset(raw, torch.tensor([1, 3, 5]))
        with patch.object(mnist, "MNIST", side_effect=source):
            dataset = BUILTIN_DATASETS["mnist"](
                train=True, output_path=self.path, class_balance=False, use_cache=True,
            )

        before, label = dataset[1]
        self.assertEqual(label, 3)
        filter_integer_targets(dataset, exclude_targets=[1])
        after, label = dataset[0]
        self.assertEqual(label, 3)
        self.assertEqual(dataset.unique_targets.tolist(), [3, 5])
        torch.testing.assert_close(after, before)
