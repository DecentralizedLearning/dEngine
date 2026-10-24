"""Small factories shared by the tests."""
from typing import Any, Dict, Optional, cast

import networkx as nx
import torch

from dengine.config import ClientModuleConfig, DynamicModuleConfigBase, ExperimentConfiguration
from dengine.dataset import SupervisedDataset
from dengine.graph.nx_graph import NXGraph
from dengine.interfaces import ClientInterface
from dengine.partitioning import TYPE_DATASET_PARTITIONING

from tests.fakes import RecordingTrainingEngine, ScalarModel


# --- Data and graphs -----------------------------------------------------------

def make_dataset(num_classes: int = 4, samples_per_class: int = 10) -> SupervisedDataset:
    """Blank 28x28 images with sorted labels: [0]*n + [1]*n + ..."""
    targets = torch.arange(num_classes).repeat_interleave(samples_per_class)
    return SupervisedDataset(torch.zeros(len(targets), 28, 28), targets)


def make_graph(**kwargs: Any) -> NXGraph:
    """Build an NXGraph without an experiment config (only ``dump()`` needs one)."""
    return NXGraph(experiment_cfg=cast(ExperimentConfiguration, None), **kwargs)


def ring(num_nodes: int = 4) -> NXGraph:
    """Ring graph: node i is connected to i-1 and i+1."""
    return make_graph(graph=nx.cycle_graph(num_nodes))


def all_indices(partitioning: TYPE_DATASET_PARTITIONING) -> list[int]:
    """Every dataset index used by a partitioning, train and validation together."""
    return [i for split in partitioning.values() for idxs in split.values() for i in idxs.tolist()]


# --- Scenarios -----------------------------------------------------------------

def round_robin_partitions(dataset: SupervisedDataset, graph: NXGraph) -> TYPE_DATASET_PARTITIONING:
    """Deal samples to nodes like cards; each node keeps its first sample for validation."""
    nodes = graph.nodes
    partitions: TYPE_DATASET_PARTITIONING = {}
    for i, node in enumerate(nodes):
        idxs = torch.arange(i, len(dataset), len(nodes))
        partitions[node] = {"train": idxs[1:], "validation": idxs[:1]}
    return partitions


def scenario_inputs(
    graph: NXGraph,
    client: str = "DecAvgClient",
    client_arguments: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Arguments shared by every scenario engine, using the fake model and engine."""
    dataset = make_dataset()
    return {
        "graph": graph,
        "training_data": dataset,
        "test_data": dataset,
        "data_partitions": round_robin_partitions(dataset, graph),
        "client_configuration": ClientModuleConfig(
            target=client,
            arguments=client_arguments,
            training_engine=DynamicModuleConfigBase(target=RecordingTrainingEngine.__name__),
            local_model=DynamicModuleConfigBase(target=ScalarModel.__name__),
        ),
    }


def weight(client: ClientInterface) -> float:
    """The scalar weight of a client's ``ScalarModel``."""
    return float(cast(ScalarModel, client.model).weight.item())


def set_weight(client: ClientInterface, value: float) -> None:
    """Overwrite the scalar weight of a client's ``ScalarModel``."""
    with torch.no_grad():
        cast(ScalarModel, client.model).weight.fill_(value)


def engine_of(client: ClientInterface) -> RecordingTrainingEngine:
    """The fake training engine of a client, to inspect what was called."""
    return cast(RecordingTrainingEngine, client.local_training_engine)
