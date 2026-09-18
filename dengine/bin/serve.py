from pathlib import Path

import uvicorn
from fastapi import FastAPI
from pydantic_settings import BaseSettings, SettingsConfigDict

from dengine.api import simulation, docs


# ╔══════════════════════════════════════════════════════════╗
# ║  Settings                                                ║
# ╚══════════════════════════════════════════════════════════╝
class ServerSettings(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"
    dengine_reload_on_change: bool = False

    output_directory: str = "logs/"

    model_config = SettingsConfigDict(
        env_file=".env.local",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = ServerSettings()


# ╔══════════════════════════════════════════════════════════╗
# ║  Main App                                                ║
# ╚══════════════════════════════════════════════════════════╝
def inizialize_fastapi_app():
    app = FastAPI()
    app.include_router(docs.router)

    output_directory = Path(settings.output_directory)
    assert output_directory.exists()
    app.include_router(
        simulation.SimulationAPI(output_directory)._router
    )
    return app


# ╔══════════════════════════════════════════════════════════╗
# ║  Uvicorn Main                                            ║
# ╚══════════════════════════════════════════════════════════╝
fastapi_app = inizialize_fastapi_app()


def _main():
    if settings.dengine_reload_on_change:
        uvicorn.run(
            "dengine.bin.serve:app",
            host=settings.host,
            port=settings.port,
            log_level=settings.log_level,
            reload=True,
        )
    else:
        uvicorn.run(
            fastapi_app,
            host=settings.host,
            port=settings.port,
            log_level=settings.log_level,
        )
