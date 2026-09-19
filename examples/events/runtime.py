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


def _get_peers_from_env() -> List[str]:
    raw_val = os.getenv("PEERS_API_BASE_URL", "").strip()
    if not raw_val:
        return []
    return [url.strip() for url in raw_val.split(",") if url.strip()]


def format_client(client: ClientInterface) -> str:
    is_remote = isinstance(client, RemoteClient)
    client_kind = "remote" if is_remote else "local"
    return f"{client.UUID} ({client_kind})"


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

    # -------------------------------------------------------------------------
    # Lifecycle & Basic Events
    # -------------------------------------------------------------------------

    @engine.start()
    def handle_engine_start(
        event: Event,
        clients: Sequence[DecAvgClient] = Depends(get_all_clients),
    ):
        """Initializes dependencies and starts the simulation run."""
        print(f"[{engine.timestamp}] Initialized dEngine with {len(clients)} client(s).")
        details = ", ".join(format_client(c) for c in clients)
        print(details)

    @engine.user_event(tag="say_hello")
    def handle_hello(
        event: Event,
        clients: Sequence[DecAvgClient] = Depends(get_all_clients),
    ):
        """Processes one-off diagnostic or handshake events."""
        print(f"[{engine.timestamp}] Initialized dEngine with {len(clients)} client(s).")
        details = ", ".join(format_client(c) for c in clients)
        print(details)

    def _schedule_wall_clock_tick(delay_seconds: float = 5.0):
        """Paces event injection against physical wall-clock time."""
        time.sleep(delay_seconds)
        # Injects the event using the engine's current simulation timestamp at wakeup
        engine.add_events(Event(timestamp=engine.timestamp, tag="slow loop"))

    @engine.user_event(tag="slow loop")
    def handle_slow_loop(event: Event):
        """Demonstrates physical pacing by offloading delays to a background daemon thread.

        Because the engine's internal clock does not track real-world seconds,
        a daemon worker thread sleeps in real time before enqueuing the next event.
        """
        print(f"[{engine.timestamp}] Received wall-clock event; arming background delay...")
        threading.Thread(
            target=_schedule_wall_clock_tick,
            args=(5.0,),
            daemon=True,
            name="WallClockPacerThread",
        ).start()

    return engine
