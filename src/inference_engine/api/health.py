from fastapi import APIRouter

from .. import __version__
from .state import app_state

router = APIRouter()


@router.get("/v1/health")
async def health() -> dict:
    return {
        "status": "ok",
        "version": __version__,
        "backend": app_state.backend_name,
        "loaded_models": app_state.manager.loaded_summary(),
        "loaded_bytes": app_state.manager.loaded_bytes,
        "memory_budget_bytes": app_state.manager.memory_budget_bytes,
    }
