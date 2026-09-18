from contextlib import asynccontextmanager
from fastapi import FastAPI

from dengine.scenarios.event_api.distributed_engine import DistributedEngine


class CurrentEngineDependency:
    def __init__(self):
        self._engine = DistributedEngine()

    @property
    def engine(self):
        return self._engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.current_active_engine = CurrentEngineDependency()
    yield
    del app.state.current_active_engine
