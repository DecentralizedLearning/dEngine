import threading
import time
import os
from typing import Sequence, List

from dengine import Depends, Event
from dengine.scenarios.decentralized import DecAvgClient
from dengine.scenarios.event_api.distributed_engine import DistributedEngine
from dengine.scenarios.event_api.remote_engine import RemoteClient
from dengine.interfaces import ClientInterface
from dengine.scenarios.event_api.sync_engine import get_all_clients
from dengine.scenarios.event_api.distributed_engine import get_remote_clients, get_local_clients


def _get_peers_from_env() -> List[str]:
    raw_val = os.getenv("PEERS_API_BASE_URL", "").strip()
    if not raw_val:
        return []
    return [url.strip() for url in raw_val.split(",") if url.strip()]


def format_params(state_dict) -> str:
    total = sum(t.numel() for t in state_dict.values())

    if total >= 1e9:
        readable = f"{total / 1e9:.2f}B"
    elif total >= 1e6:
        readable = f"{total / 1e6:.2f}M"
    elif total >= 1e3:
        readable = f"{total / 1e3:.2f}K"
    else:
        readable = str(total)

    return f"{readable} ({total:,})"


def format_client(client: ClientInterface) -> str:
    is_remote = isinstance(client, RemoteClient)
    client_kind = "remote" if is_remote else "local"

    params_str = format_params(client.model.state_dict())
    return f"{client.UUID} ({client_kind}) | {params_str} params"


def create_engine() -> DistributedEngine[DecAvgClient]:
    """Configures and initializes an asynchronous, discrete-event simulation engine.

    Timing Characteristics:
        - Internal Simulation Clock: Advances non-linearly, jumping directly
          to scheduled event timestamps as quickly as compute allows.
        - Wall-Clock Synchronization: The engine loop does not inherently align
          with physical real time. Background threads are required when pacing
          events against real-world wall-clock delays.
    """
    engine = DistributedEngine[DecAvgClient](
        peers_api_base_url=_get_peers_from_env(),
        synchronization_mode="manual",
        testing_mode="manual",
        raise_for_unknown_event=False,
    )

    @engine.start()
    def handle_engine_start(
        event: Event,
        clients: Sequence[ClientInterface] = Depends(get_all_clients),
    ):
        """Initializes dependencies and starts the simulation run."""
        print(f"[{engine.timestamp}] Initialized dEngine with {len(clients)} client(s).")
        details = ", \n".join(format_client(c) for c in clients)
        print(details)

    @engine.user_event(tag="hello")
    def handle_hello(
        event: Event,
        clients: Sequence[ClientInterface] = Depends(get_all_clients),
        remote_clients: Sequence[RemoteClient] = Depends(get_remote_clients),
        local_clients: Sequence[DecAvgClient] = Depends(get_local_clients)
    ):
        print(f"[{engine.timestamp}] Initialized dEngine with {len(clients)} client(s).")
        details = ", \n".join(format_client(c) for c in clients)
        print(details)

    def _schedule_wall_clock_tick(delay_seconds: float = 5.0):
        time.sleep(delay_seconds)
        engine.add_events(Event(timestamp=engine.timestamp, tag="slow loop"))

    @engine.user_event(tag="loop")
    def handle_slow_loop(event: Event):
        print(f"[{engine.timestamp}] Received wall-clock event; arming background delay...")
        threading.Thread(
            target=_schedule_wall_clock_tick,
            args=(5.0,),
            daemon=True,
        ).start()

    return engine
