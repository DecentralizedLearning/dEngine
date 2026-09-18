from threading import Thread
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request

from dengine.scenarios.event_api.distributed_engine import DistributedEngine
from dengine.scenarios.event_api.events import EndEvent


class CurrentEngineDependency:
    def __init__(self):
        self._engine = DistributedEngine()
        self._thread: Thread | None = None

    @property
    def engine(self) -> DistributedEngine:
        return self._engine

    def run_on_daemon_thread(self) -> None:
        """Starts the engine's blocking run loop in a background daemon thread."""
        self._thread = Thread(
            target=self._engine.run,
            name="DistributedEngineWorker",
            daemon=True,
        )
        self._thread.start()

    def shutdown(self) -> None:
        """Signals shutdown to the engine if it supports graceful stopping."""
        self._engine.add_events(EndEvent())
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=10.0)


@asynccontextmanager
async def lifespan(app: FastAPI):
    daemon_engine = CurrentEngineDependency()
    daemon_engine.run_on_daemon_thread()
    try:
        yield {"current_active_engine": daemon_engine}
    finally:
        daemon_engine.shutdown()


def get_engine(request: Request) -> DistributedEngine:
    return request.state.current_active_engine.engine
