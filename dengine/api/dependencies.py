import importlib
import os
from contextlib import asynccontextmanager
import sys
from threading import Thread
from typing import Callable

from fastapi import FastAPI, Request

from dengine.scenarios.event_api.distributed_engine import DistributedEngine
from dengine.scenarios.event_api.events import EndEvent


def _resolve_engine_factory() -> Callable[[], DistributedEngine]:
    factory_path = os.getenv("DENGINE_ENGINE_FACTORY") or os.getenv("ENGINE_FACTORY")
    if not factory_path:
        return DistributedEngine

    if ":" not in factory_path:
        raise ValueError(
            f"Invalid factory format: '{factory_path}'. Expected format: 'module.submodule:callable'"
        )

    module_name, attr_name = factory_path.split(":", 1)

    # Ensure the current working directory is searchable
    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.insert(0, cwd)

    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"Could not import module '{module_name}' from factory string '{factory_path}'. "
            f"Searched in sys.path (including current directory: '{cwd}')."
        ) from exc

    try:
        factory = getattr(module, attr_name)
    except AttributeError as exc:
        raise AttributeError(
            f"Module '{module_name}' has no attribute '{attr_name}'."
        ) from exc

    if not callable(factory):
        raise TypeError(f"Resolved factory '{factory_path}' is not callable.")

    return factory


class CurrentEngineDependency:
    def __init__(self):
        engine_factory = _resolve_engine_factory()
        self._engine: DistributedEngine = engine_factory()
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
