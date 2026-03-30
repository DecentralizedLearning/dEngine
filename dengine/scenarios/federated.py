from warnings import deprecated
from typing import List, TypeVar
from dataclasses import dataclass
from copy import deepcopy
from queue import PriorityQueue
import logging

import torch
from torch import nn
from torch.utils.data import Subset

from dengine.models.utils import model_wise_weighted_average
from dengine.interfaces import MessageBase, Graph
from dengine.graph import NXGraph
from dengine.scenarios.scenario import AbstractClient

from .decentralized import DecentralizedScenarioEngineBase
from .decorators import register_scenario, register_client


def _fed_avg(w):
    w_avg = deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg


@dataclass
class VanillaFederatedMessage(MessageBase):
    global_model: nn.Module


TGenericVanillaFederatedMessage = TypeVar("TGenericVanillaFederatedMessage", bound=VanillaFederatedMessage)


class ServerMockClient(AbstractClient[TGenericVanillaFederatedMessage]):
    def __init__(self, *args, **kwargs):
        self._uuid = -1
        self._msg_buffer: PriorityQueue[TGenericVanillaFederatedMessage] = PriorityQueue()

    @property
    def message_buffer(self) -> PriorityQueue[TGenericVanillaFederatedMessage]:
        return self._msg_buffer

    def synchronization(self, *args, **kwargs):
        raise NotImplementedError()

    def aggregation(self, *args, **kwargs):
        raise NotImplementedError()

    def update(self, *args, **kwargs):
        raise NotImplementedError()

    def validate(self, *args, **kwargs):
        raise NotImplementedError()

    def test(self, *args, **kwargs):
        raise NotImplementedError()


class FederatedClient(AbstractClient[TGenericVanillaFederatedMessage]):
    def __init__(self, training_data: Subset, *args, **kwargs):
        super().__init__(*args, training_data=training_data, **kwargs)
        self.train_size = len(training_data)

    @property
    def message_buffer(self) -> PriorityQueue[TGenericVanillaFederatedMessage]:
        return self._msg_buffer

    def aggregation(self, messages: List[TGenericVanillaFederatedMessage]):
        if len(messages) == 0:
            return
        assert len(messages) == 1
        self.model.load_state_dict(
            messages[0].global_model.state_dict()
        )


@deprecated("This class has been refactored to FederatedClient, please consider refactoring.")
@register_client()
class FedClientPaiv(FederatedClient):
    def __init__(
        self,
        *args,
        test_batch_size,
        **kwargs
    ):
        super().__init__(*args, **kwargs)


def _convert_dengine_graph_to_fed_topology(graph: Graph) -> Graph:
    if not isinstance(graph, NXGraph):
        raise NotImplementedError("Currently only NXGraph is supported")

    G = graph.nx_graph
    server = "C"
    G.clear_edges()
    G.add_node(server)
    for node in list(G.nodes):
        if node != server:
            G.add_edge(server, node)
    return graph


@register_scenario()
class VanillaFederatedScenario(DecentralizedScenarioEngineBase):
    def __init__(
        self,
        use_weighted_avg: bool,
        graph: Graph,
        *args,
        **kwargs
    ):
        super().__init__(*args, graph=graph, **kwargs)

        # This is a total useless operation since the graph is not used at all in this scenario
        # it has been introduced only for visualization purposes and to calm down the user
        _convert_dengine_graph_to_fed_topology(graph)

        self._server_mock_client = ServerMockClient()
        self._weight_avg = use_weighted_avg

        # Will be overwritten at the first agg.
        self._global_model = deepcopy(
            list(self.clients.values())[0].model
        )

    def get_client_by_id(self, id) -> FederatedClient:
        return super().get_client_by_id(id)  # type: ignore

    def get_active_clients(self) -> List[FederatedClient]:
        return super().get_active_clients()  # type: ignore

    def on_communication_round_end(self, round_idx: int):
        super().on_communication_round_end(round_idx)

        models = []
        alphas = []
        for p in self.get_active_clients():
            models.append(p.model.state_dict())
            alphas.append(p.train_size)

        logging.info(
            "``` \n"
            f"Merging {len(models)} models coming from: {[p.UUID for p in self.get_active_clients()]} \n"
            "```"
        )
        if self._weight_avg:
            tot_alpha = sum(alphas)
            alphas = [i / tot_alpha for i in alphas]
            avg_model = model_wise_weighted_average(models, alphas)
        else:
            avg_model = _fed_avg(models)

        self._global_model.load_state_dict(avg_model)
        msg = VanillaFederatedMessage(
            time=round_idx,
            global_model=self._global_model,
            source_client=self._server_mock_client
        )
        for p in self.get_active_clients():
            p.message_buffer.put(msg)

    def synchronize_client(self, communication_round: int, client):
        pass
