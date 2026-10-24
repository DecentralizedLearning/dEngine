"""Shared fixtures. Everything is tiny, CPU-only and needs no downloads."""
import random

import pytest
import torch

from dengine.dataset import SupervisedDataset
from dengine.graph.nx_graph import NXGraph

from tests.helpers import make_dataset, ring


@pytest.fixture(autouse=True)
def _seed() -> None:
    """Make every test deterministic."""
    random.seed(0)
    torch.manual_seed(0)


@pytest.fixture
def dataset() -> SupervisedDataset:
    """40 samples: 10 for each of the labels 0, 1, 2 and 3."""
    return make_dataset(num_classes=4, samples_per_class=10)


@pytest.fixture
def ring_graph() -> NXGraph:
    """A 4-node ring: 0-1-2-3-0."""
    return ring(4)
