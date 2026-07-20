from fastapi import APIRouter, Response, status

from .. import __version__
from ..config import settings
from .state import app_state

router = APIRouter()


@router.get("/v1/health")
async def health() -> dict:
    readiness = app_state.readiness()
    return {
        "status": "ok" if readiness["ready"] else readiness["status"],
        "ready": readiness["ready"],
        "readiness": readiness,
        "version": __version__,
        "workload_surface": settings.model_plane_workload_surface,
        "backend": app_state.backend_name,
        "loaded_models": app_state.manager.loaded_summary(),
        "loaded_bytes": app_state.manager.loaded_bytes,
        "memory_budget_bytes": app_state.manager.memory_budget_bytes,
    }


@router.get("/v1/ready")
async def ready(response: Response) -> dict:
    readiness = app_state.readiness()
    if not readiness["ready"]:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        response.headers["Retry-After"] = "5"
    return {
        "status": readiness["status"],
        "ready": readiness["ready"],
        "readiness": readiness,
        "version": __version__,
        "workload_surface": settings.model_plane_workload_surface,
        "backend": app_state.backend_name,
    }
