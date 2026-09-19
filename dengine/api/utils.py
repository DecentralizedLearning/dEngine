from typing import Callable
from fastapi import FastAPI


def url_path_for(fastapi_app: FastAPI, f: Callable, **kwargs) -> str:
    return fastapi_app.router.url_path_for(f.__name__, **kwargs)
