from pathlib import Path

from fastapi import APIRouter, Response, status

from dengine.config.utils import ExperimentConfiguration


class SimulationAPI:
    def __init__(self, output_directory: Path):
        self._output_directory = output_directory

        self._router = APIRouter(
            tags=["simulation"],
            prefix="/simulation"
        )
        self._router.add_api_route("/sanity_check", self.sanity_check, methods=["POST"])

    async def sanity_check(self, config: ExperimentConfiguration):
        return Response(status_code=status.HTTP_200_OK)
