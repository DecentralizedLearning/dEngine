from pathlib import Path
from typing import List, Union
from urllib.parse import urljoin
from datetime import datetime, timezone
import jwt
from jwt import DecodeError
import io

import torch
from fastapi.responses import StreamingResponse
from fastapi import APIRouter, Response, status, Depends, HTTPException

from dengine.config.configuration import ExperimentConfiguration
from dengine.scenarios.event_api.events import Event
from dengine.scenarios.event_api.distributed_engine import DistributedEngine
from dengine.scenarios.scenario import GenericClient

from dengine.bin.simulation import VerbosityLevel
from dengine.bin.simulation import load_engine, SimulationArguments
from .dependencies import get_engine
from .schemas import ClientListSchema, PresignedModelDownloadURL, JWTTokenPresignedModelDownloadURL


def decode_token_or_raise_400(token: str, secret: str):
    try:
        return jwt.decode(token, secret, algorithms="HS256")  # type: ignore
    except DecodeError:
        raise HTTPException(status_code=401, detail="JWT Error")


class SimulationAPI:
    def __init__(
        self,
        output_directory: Path,
        datasets_directory: Path,
        jwt_secret: str,
        api_base_url: str,
    ):
        self._output_directory = output_directory
        self._datasets_directory = datasets_directory

        self._jwt_secret = jwt_secret
        self._presigned_url_get_endpoint = "client/model/download"
        self._api_base_url = api_base_url

        self._router = APIRouter(
            tags=["simulation"],
            prefix="/simulation"
        )
        self._router.add_api_route("/sanity_check", self.sanity_check, methods=["POST"])
        self._router.add_api_route("/event/new", self.add_events, methods=["POST"])
        self._router.add_api_route("/event/all", self.list_all_events, methods=["GET"])
        self._router.add_api_route("/load", self.load, methods=["POST"])
        self._router.add_api_route("/client/all", self.list_all_clients, methods=["GET"])
        self._router.add_api_route(f"/{self._presigned_url_get_endpoint}", self.client_model_download, methods=["GET"])
        self._router.add_api_route("/client/{UUID}/model/download", self.client_model_presigned_url, methods=["GET"])

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
        config.experiments_directory_root = str(
            (self._output_directory / config.experiments_directory_root).absolute()
        )
        load_engine(args, config, engine)
        return {"status": "success"}

    async def list_all_clients(
        self,
        engine: DistributedEngine[GenericClient] = Depends(get_engine)
    ) -> List[ClientListSchema]:
        clients = engine.get_all_clients()
        return [
            ClientListSchema.model_validate({
                "UUID": c.UUID
            })
            for c in clients
        ]

    async def client_model_presigned_url(
        self,
        UUID: str,
        engine: DistributedEngine[GenericClient] = Depends(get_engine)
    ) -> PresignedModelDownloadURL:
        try:
            engine.get_client(UUID)
        except KeyError:
            raise HTTPException(404, "Unknown client")

        payload = JWTTokenPresignedModelDownloadURL(
            UUID=UUID,
            token_created_at=datetime.now(timezone.utc).isoformat(),
        )
        token = jwt.encode(payload.model_dump(), self._jwt_secret)
        return PresignedModelDownloadURL(
            url=urljoin(self._api_base_url, f"simulation/client/model/download?token={token}")
        )

    async def client_model_download(
        self,
        token: str,
        engine: DistributedEngine[GenericClient] = Depends(get_engine)
    ):
        payload = decode_token_or_raise_400(token, self._jwt_secret)
        presigned_url_data = JWTTokenPresignedModelDownloadURL.model_validate(payload)
        try:
            client = engine.get_client(presigned_url_data.UUID)
        except KeyError:
            raise HTTPException(404, "Unknown client")

        buffer = io.BytesIO()
        torch.save(client.model.state_dict(), buffer)
        buffer.seek(0)

        return StreamingResponse(
            content=buffer,
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": f'attachment; filename="{presigned_url_data.UUID}.pt"'
            }
        )
