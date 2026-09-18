from pathlib import Path

import uvicorn
from fastapi import FastAPI
from pydantic_settings import BaseSettings, SettingsConfigDict

from dengine.api import simulation, docs, dependencies


# ╔══════════════════════════════════════════════════════════╗
# ║  Settings                                                ║
# ╚══════════════════════════════════════════════════════════╝
class ServerSettings(BaseSettings):
    dengine_host: str = "0.0.0.0"
    dengine_port: int = 8000
    log_level: str = "info"
    dengine_reload_on_change: bool = False

    dengine_output_directory: str = "logs/"
    dengine_datasets_directory: str = "datasets/"

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
    app = FastAPI(lifespan=dependencies.lifespan)
    app.include_router(docs.router)

    output_directory = Path(settings.dengine_output_directory)
    assert output_directory.exists()
    datasets_directory = Path(settings.dengine_datasets_directory)
    assert datasets_directory.exists()

    app.include_router(
        simulation.SimulationAPI(output_directory, datasets_directory)._router
    )
    return app


# ╔══════════════════════════════════════════════════════════╗
# ║  Uvicorn Main                                            ║
# ╚══════════════════════════════════════════════════════════╝
fastapi_app = inizialize_fastapi_app()


def _main():
    if settings.dengine_reload_on_change:
        uvicorn.run(
            "dengine.bin.serve:fastapi_app",
            host=settings.dengine_host,
            port=settings.dengine_port,
            log_level=settings.log_level,
            reload=True,
        )
    else:
        uvicorn.run(
            fastapi_app,
            host=settings.dengine_host,
            port=settings.dengine_port,
            log_level=settings.log_level,
        )
