from typing import Optional

import networkx as nx
import torch
from torch.nn import init
from torch.utils.data import Subset

from dengine.models.classifier import TinyCNN
from dengine.dataset.mnist import load_mnist
from dengine.graph import Graph
from dengine.interfaces import ScenarioEngineInterface, ModuleBase
from dengine.scenarios.decentralized import DecAvgClient


MOCK_VALIDATION_SUBSET = Subset(load_mnist(True, "datasets"), [100, 101, 102, 103, 104])


def constant_model(c: int) -> ModuleBase:
    model = TinyCNN(MOCK_VALIDATION_SUBSET)

    for p in model.parameters():
        init.constant_(p, c)
    return model


@torch.no_grad()
def constant_model_value(model: ModuleBase):
    return float(next(model.parameters()).unique())


class MockScenario(ScenarioEngineInterface):
    def __init__(self, graph: Graph):
        self._graph = graph

    @property
    def clients(self):
        return {}

    @property
    def graph(self) -> Graph:
        return self._graph

    def run(self):
        pass


class MockScenarioEmptyGraph(MockScenario):
    def __init__(self):
        super().__init__(nx.empty_graph(0))


class MockClient(DecAvgClient):
    def __init__(
        self,
        uuid: int,
        include_myself: bool = False,
        use_weighted_avg: bool = False,
        scenario: Optional[MockScenario] = None,
    ):
        super().__init__(
            uuid=uuid,
            include_myself=include_myself,
            scenario=scenario or MockScenarioEmptyGraph(),
            use_weighted_avg=use_weighted_avg,
            training_engine=None,
            local_model=constant_model(uuid),
            training_data=Subset(load_mnist(True, "datasets"), range(0, uuid)),
            validation_data=MOCK_VALIDATION_SUBSET,
            force_time_synchronization=True,
            verbose=True,
        )

    def execute_local_train_strategy(self, current_time: float):
        return constant_model(int(self.UUID) + int((current_time + 1) * 10))
