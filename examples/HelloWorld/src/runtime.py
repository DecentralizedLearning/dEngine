from typing import Sequence, Dict
from copy import deepcopy

from datetime import timedelta

from dengine import (
    LocalTraining,
    Synchronization,
    Depends,
)
from dengine.scenarios.event_api.sync_engine import (
    SyncEngine,
    get_all_clients,
    get_training_client,
    get_src,
    get_destinations,
    get_contact_time
)

from .client import CustomClient, CustomMessage


engine = SyncEngine[CustomClient](synchronization_mode='manual')


@engine.start()
def entrypoint(
    clients: Sequence[CustomClient] = Depends(get_all_clients)
):
    for ith_client in clients:
        e = LocalTraining(
            timestamp=(engine.timestamp + timedelta(seconds=10)),
            client=ith_client.UUID
        )
        engine.add_events(e)


@engine.local_training("end")
def schedule_synch_and_training(
    event: LocalTraining,
    clients: CustomClient = Depends(get_training_client)
):
    engine.add_events([
        Synchronization(timestamp=event.timestamp, src=clients.UUID, destinations=None),
        LocalTraining(timestamp=event.timestamp + timedelta(seconds=1), client=clients.UUID)
    ])


@engine.synchronization("start")
def update_buffers(
    event: Synchronization,
    trained_client: CustomClient = Depends(get_src),
    destinations: Sequence[CustomClient] = Depends(get_destinations),
    contact_times: Dict[str, float] = Depends(get_contact_time)
):
    client_checkpoint = deepcopy(trained_client)
    for dst in destinations:
        # overall_time = contact_times[dst.UUID]
        dst.message_buffer.put(
            CustomMessage(
                time=event.timestamp.timestamp(),
                source_client=client_checkpoint,
                normalized_contact_time=2.12,
            )
        )
