from __future__ import annotations

from io import BytesIO
from typing import List, Union, Dict, Optional, Any

import httpx
import torch
from torch import Tensor

from dengine.interfaces import ClientInterface
from dengine.scenarios.event_api.events import Event

from .sync_engine import SyncEngine


class _ModuleMock:
    def __init__(self, state_dict):
        self._state_dict = state_dict

    def state_dict(self):
        return self._state_dict


class RemoteClient(ClientInterface):
    def __init__(
        self,
        scenario: RemoteEngine,
        uuid: str,
    ):
        self._uuid = uuid
        self._scenario = scenario

    @property
    def UUID(self) -> str:
        return self._uuid

    @property
    def model(self):
        return _ModuleMock(
            self._scenario.download_model(self.UUID)
        )

    def synchronization(self, current_time: int):
        raise NotImplementedError()

    def aggregation(self, messages):
        raise NotImplementedError()

    def update(self, current_time: float):
        raise NotImplementedError()

    def test(self, current_time: float, dataset):
        raise NotImplementedError()


class RemoteEngine(SyncEngine[RemoteClient]):
    API_PATH_LIST_ALL_CLIENTS = "/simulation/client/all"
    API_PATH_ADD_EVENTS = "/simulation/event/new"
    API_PATH_CLIENT_MODEL_DOWNLOAD = "/simulation/client/{UUID}/model/download"

    def __init__(self, api_base_url: str):
        self._base_url = api_base_url.rstrip("/") + "/"
        self._httpx_client = httpx.Client(base_url=self._base_url)

    def list_all_clients(self) -> List[RemoteClient]:
        """Fetch all client UUIDs from the API and instantiate RemoteClient wrappers."""
        url = self.API_PATH_LIST_ALL_CLIENTS.lstrip("/")
        response = self._httpx_client.get(url)
        response.raise_for_status()

        raw_clients = response.json()
        return [RemoteClient(scenario=self, uuid=c["UUID"]) for c in raw_clients]

    def add_events(self, payload: Union[List[Event], Event]) -> dict:
        """Serialize one or more events using Pydantic and post to the API."""
        events = payload if isinstance(payload, list) else [payload]
        json_data = [e.model_dump(mode="json") for e in events]

        url = self.API_PATH_ADD_EVENTS.lstrip("/")
        response = self._httpx_client.post(url, json=json_data)
        response.raise_for_status()

        return response.json()

    async def close(self):
        self._httpx_client.close()

    @property
    def clients(self) -> Dict[str, RemoteClient]:
        return {client.UUID: client for client in self.list_all_clients()}

    def download_model(
        self,
        uuid: str,
        map_location: Optional[Any] = "cpu",
        weights_only: bool = True,
    ) -> Dict[str, Tensor]:
        # 1. Fetch presigned URL payload
        presigned_url_endpoint = self.API_PATH_CLIENT_MODEL_DOWNLOAD.format(UUID=uuid).lstrip("/")
        url_resp = self._httpx_client.get(presigned_url_endpoint)
        url_resp.raise_for_status()
        download_url = url_resp.json()["url"]

        # 2. Download the serialized model bytes
        download_resp = self._httpx_client.get(download_url)
        download_resp.raise_for_status()

        # 3. Read into BytesIO buffer and load the state dict onto the specified device
        buffer = BytesIO(download_resp.content)
        state_dict: Dict[str, Tensor] = torch.load(
            buffer,
            map_location=map_location,
            weights_only=weights_only,
        )

        return state_dict
