import sys
from copy import deepcopy
from datetime import datetime
from typing import Sequence, Dict, Literal, Optional
from dataclasses import dataclass
from queue import PriorityQueue

import torch

from dengine.config import ClientModuleConfig
from dengine.utils.utils import model_on_device_context
from dengine.dataset import SupervisedDataset
from dengine.graph import Graph, DynamicGraph
from dengine.interfaces import MessageBase, TYPE_CLIENT_CALLBACK_FACTORY
from dengine.partitioning import TYPE_DATASET_PARTITIONING
from dengine.scenarios.decorators import register_scenario
from dengine.scenarios.event_api.events import Event, StartEvent, Synchronization, LocalTraining
from dengine.scenarios.scenario import GenericClient

from .scenario import ScenarioEventEngine, Depends


@dataclass
class DynamicUpdateMessage(MessageBase):
    contact_time: float
    time: float


@register_scenario()
class SyncEngine(ScenarioEventEngine[GenericClient]):
    def __init__(
        self,
        synchronization_mode: Literal['auto', 'manual'] = 'auto',
        testing_mode: Literal['on_local_training_end', 'manual'] = 'on_local_training_end',
        raise_for_unknown_event: bool = True,
    ):
        super().__init__()
        self._disable_synchronization = (synchronization_mode == 'manual')
        self._disable_testing = (testing_mode == 'manual')
        self._raise_for_unknown_event = raise_for_unknown_event

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
        self._graph = graph
        self.training_data = training_data
        self.test_data = test_data
        self.partitioning = data_partitions

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = torch.device(device)

        self._event_queue: PriorityQueue[Event] = PriorityQueue()
        self._max_communication_rounds = max_communication_rounds

        self._clients = self.init_clients(
            client_configuration,
            training_data,
            data_partitions,
            common_init,
            callback_factory
        )

    def pop_next_event(self) -> Event:
        return self._event_queue.get()

    def add_events(self, events: Event | Sequence[Event]):
        if not isinstance(events, Sequence):
            events = [events]
        for ith_event in events:
            self._event_queue.put(ith_event)

    def consume_event(self, event: Event) -> Sequence[Event] | None:
        if isinstance(event, StartEvent):
            return
        elif isinstance(event, Synchronization):
            self._synchronize_client(
                event.timestamp,
                self.get_client(event.src),
                self.get_many_clients(event.destinations)
            )
            return
        elif isinstance(event, LocalTraining):
            self._client_step(event.timestamp, self.get_client(event.client))
            return

        if self._raise_for_unknown_event:
            raise ValueError(f"Unknown event received: \n{event.model_dump()}")

    def get_client(
        self,
        uuids: str
    ) -> GenericClient:
        return self.clients[uuids]

    def get_all_clients(self):
        return list(self.clients.values())

    def get_many_clients(
        self,
        uuids: str | Sequence[str] | None = None
    ) -> Sequence[GenericClient]:
        if not uuids:
            return []
        return [self.clients[ith_uuid] for ith_uuid in uuids]

    def _client_step(self, timestamp: datetime, client: GenericClient):
        with model_on_device_context(client.model, self._device):
            client.update(
                current_time=timestamp.timestamp(),
            )
            if not self._disable_testing:
                client.test(timestamp.timestamp(), self.test_data)

    def _synchronize_client(
        self,
        timestamp: datetime,
        src: GenericClient,
        destinations: Sequence[GenericClient] | None
    ):
        if self._disable_synchronization:
            return
        if isinstance(self.graph, DynamicGraph):
            self._synchronize_client_dynamic_graph(timestamp, src, destinations)
        else:
            self._synchronize_client_static_graph(timestamp, src, destinations)

    def _synchronize_client_static_graph(
        self,
        timestamp: datetime,
        client: GenericClient,
        destinations: Sequence[GenericClient] | None
    ):
        assert isinstance(self.graph, Graph)
        if destinations is None:
            destinations_idxs = self.graph.neighbors(client.UUID)
        else:
            destinations_idxs = [p.UUID for p in destinations]
        self._logging(f"Sending the local model to: {destinations_idxs}")

        client_checkpoint = deepcopy(client)
        for dst_id in destinations_idxs:
            dst = self.get_client(dst_id)
            dst.message_buffer.put(
                DynamicUpdateMessage(
                    time=timestamp.timestamp(),
                    source_client=client_checkpoint,
                    contact_time=1
                )
            )

    def _synchronize_client_dynamic_graph(
        self,
        timestamp: datetime,
        client: GenericClient,
        destinations: Sequence[GenericClient] | None
    ):
        assert isinstance(self.graph, DynamicGraph)
        if destinations is None:
            destinations_idxs = self.graph.neighbors(
                client.UUID,
                timestamp
            )
        else:
            destinations_idxs = [p.UUID for p in destinations]
        self._logging(f"Sending the local model to: {destinations_idxs}")

        client_checkpoint = deepcopy(client)
        for dst_id in destinations_idxs:
            dst = self.get_client(dst_id)
            contact_time = self.graph.contact_time(client.UUID, dst.UUID, timestamp)
            dst.message_buffer.put(
                DynamicUpdateMessage(
                    time=timestamp.timestamp(),
                    source_client=client_checkpoint,
                    contact_time=contact_time,
                )
            )


# Dependencies ..... ..... #
# ........................ #
def get_graph(engine: SyncEngine):
    return engine.graph


def get_all_clients(engine: SyncEngine):
    return engine.get_all_clients()


def get_training_client(engine: SyncEngine, event: LocalTraining):
    assert isinstance(event, LocalTraining)
    return engine.get_client(event.client)


def get_destinations(
    engine: SyncEngine,
    event: Synchronization,
    graph: Graph | DynamicGraph = Depends(get_graph),
):
    assert isinstance(event, Synchronization)
    if event.destinations:
        destinations = event.destinations
    elif isinstance(graph, DynamicGraph):
        destinations = graph.neighbors(event.src, event.timestamp)
    else:
        destinations = graph.neighbors(event.src)
    destinations = [str(id) for id in destinations]
    return engine.get_many_clients(destinations)


def get_contact_time(
    event: Synchronization,
    graph: Graph | DynamicGraph = Depends(get_graph),
    destinations: Sequence[GenericClient] = Depends(get_destinations),
) -> Dict[str, float | None]:
    assert isinstance(event, Synchronization)
    destinations_uuids = [dst.UUID for dst in destinations]
    if isinstance(graph, DynamicGraph):
        return {
            dst: graph.contact_time(
                source=event.src,
                destination=dst,
                time=event.timestamp
            )
            for dst in destinations_uuids
        }
    return {dst: None for dst in destinations_uuids}


def get_src(engine: SyncEngine, event: Synchronization):
    assert isinstance(event, Synchronization)
    return engine.get_client(event.src)
