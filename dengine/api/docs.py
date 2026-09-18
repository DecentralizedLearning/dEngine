from fastapi import APIRouter
from fastapi.responses import RedirectResponse


router = APIRouter(
    tags=["entrypoint"],
)


@router.get("/")
async def greetings() -> RedirectResponse:
    return RedirectResponse("/docs")


@router.get("/health")
async def health():
    return {
        "status": "ok",
        "version": "2.0/ananas-pizza"
    }
