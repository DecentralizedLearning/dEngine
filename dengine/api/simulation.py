from pathlib import Path
from typing import List, Union

from fastapi import APIRouter, Response, status, Depends

from dengine.config.configuration import ExperimentConfiguration
from dengine.scenarios.event_api.events import Event
from dengine.scenarios.event_api.distributed_engine import DistributedEngine
from dengine.scenarios.scenario import GenericClient

from dengine.bin.simulation import VerbosityLevel
from dengine.bin.simulation import load_engine, SimulationArguments
from .dependencies import get_engine
from .schemas import ClientListSchema


class SimulationAPI:
    def __init__(
        self,
        output_directory: Path,
        datasets_directory: Path,
    ):
        self._output_directory = output_directory
        self._datasets_directory = datasets_directory

        self._router = APIRouter(
            tags=["simulation"],
            prefix="/simulation"
        )
        self._router.add_api_route("/sanity_check", self.sanity_check, methods=["POST"])
        self._router.add_api_route("/event/new", self.add_events, methods=["POST"])
        self._router.add_api_route("/event/all", self.list_all_events, methods=["GET"])
        self._router.add_api_route("/load", self.load, methods=["POST"])
        self._router.add_api_route("/client/all", self.list_all_clients, methods=["GET"])

    async def sanity_check(self, config: ExperimentConfiguration):
        return Response(status_code=status.HTTP_200_OK)

    async def add_events(
        self,
        payload: Union[List[Event], Event],
        engine: DistributedEngine = Depends(get_engine)
    ):
        events = payload if isinstance(payload, list) else [payload]
        engine.add_events(events)
        return {"status": "success", "count": len(events)}

    async def list_all_events(
        self,
        engine: DistributedEngine = Depends(get_engine)
    ) -> List[Event]:
        return engine._event_queue.queue

    async def load(
        self,
        config: ExperimentConfiguration,
        engine: DistributedEngine = Depends(get_engine)
    ):
        args = SimulationArguments(
            gpus=[-1], torch_num_threads=1, verbosity=VerbosityLevel.debug,
            dataset_directory=self._datasets_directory,
            dump_stdout=False,
            output_directory=self._output_directory,
            resume_checkpoints=True,
            sanity_check=True,
            seed=123
        )
        load_engine(args, config, engine)
        return {"status": "success"}

    async def list_all_clients(
        self,
        engine: DistributedEngine[GenericClient] = Depends(get_engine)
    ) -> List[ClientListSchema]:
        clients = engine.get_all_clients()
        return [ClientListSchema.model_validate(c.UUID) for c in clients]
