from typing import Dict, Optional, List, Sequence
from datetime import datetime
import sys
from queue import PriorityQueue

import torch

from dengine.scenarios.utils import client_on_device_context
from dengine.graph import Graph, DistributedGraph
from dengine.scenarios.decorators import register_scenario
from dengine.config import ClientModuleConfig
from dengine.dataset import SupervisedDataset
from dengine.interfaces import (
    GenericClient,
    TYPE_CLIENT_CALLBACK_FACTORY,
)
from dengine.partitioning import TYPE_DATASET_PARTITIONING
from dengine.scenarios.event_api.events import Event

from .remote_engine import RemoteEngine, RemoteClient
from .sync_engine import SyncEngine


@register_scenario()
class DistributedEngine(SyncEngine[GenericClient]):
    def __init__(
        self,
        *args,
        peers_api_base_url: List[str] = [],
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self._graph = None
        self._event_queue: PriorityQueue[Event] = PriorityQueue()
        self._clients: Dict[str, GenericClient] = {}
        self._max_communication_rounds = sys.maxsize

        self._test_data_mapping: Dict[str, SupervisedDataset] = {}
        self._train_data_mapping: Dict[str, SupervisedDataset] = {}
        self._partitioning_mapping: Dict[str, TYPE_DATASET_PARTITIONING] = {}

        self._peers: List[RemoteEngine] = [RemoteEngine(url) for url in peers_api_base_url]

    def load(
        self,
        graph: Graph,
        training_data: SupervisedDataset,
        data_partitions: TYPE_DATASET_PARTITIONING,
        test_data: SupervisedDataset,
        client_configuration: ClientModuleConfig,
        callback_factory: Optional[TYPE_CLIENT_CALLBACK_FACTORY] = None,
        # Additional args
        max_communication_rounds: int = sys.maxsize,
        common_init: bool = False,
    ):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = torch.device(device)

        if self.graph is None:
            updated_graph = graph
        else:
            assert isinstance(graph, DistributedGraph)
            assert isinstance(self.graph, DistributedGraph)
            updated_graph = self.graph.merge(graph)

        self._graph = graph
        new_clients = self.init_clients(
            client_configuration,
            training_data,
            data_partitions,
            common_init,
            callback_factory
        )
        self._graph = updated_graph

        self.clients.update(new_clients)

        self._test_data_mapping.update({
            k: test_data for k in new_clients.keys()
        })
        self._train_data_mapping.update({
            k: training_data for k in new_clients.keys()
        })
        self._partitioning_mapping.update({
            k: data_partitions for k in new_clients.keys()
        })

    def _client_step(self, timestamp: datetime, client: GenericClient):
        with client_on_device_context(client, self._device):
            client.update(
                current_time=timestamp.timestamp(),
            )
            if not self._disable_testing:
                client.test(timestamp.timestamp(), self._test_data_mapping[client.UUID])

    def get_all_clients(self) -> Sequence[GenericClient]:
        local_clients = list(self.clients.values())
        remote_clients = []
        for r_engine in self._peers:
            remote_clients.extend(
                r_engine.get_all_clients()
            )
        return [*local_clients, *remote_clients]


def get_local_clients(engine: SyncEngine):
    return [c for c in engine.get_all_clients() if not isinstance(c, RemoteClient)]


def get_remote_clients(engine: SyncEngine):
    return [c for c in engine.get_all_clients() if isinstance(c, RemoteClient)]
