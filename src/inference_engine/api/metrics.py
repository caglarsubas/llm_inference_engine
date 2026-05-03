"""Lightweight Prometheus-compatible /v1/metrics endpoint.

Hand-rolled rather than pulling in `prometheus-client` so the dependency
surface stays small. Outputs the OpenMetrics text format Prometheus scrapes.
"""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import PlainTextResponse

from .. import __version__
from .state import app_state

router = APIRouter()


@router.get("/v1/metrics", response_class=PlainTextResponse)
async def metrics() -> str:
    lines: list[str] = []

    lines.append("# HELP inference_engine_info Engine build info.")
    lines.append("# TYPE inference_engine_info gauge")
    lines.append(
        f'inference_engine_info{{version="{__version__}",backend="{app_state.backend_name}"}} 1'
    )

    lines.append("# HELP inference_engine_models_loaded Number of models currently loaded.")
    lines.append("# TYPE inference_engine_models_loaded gauge")
    lines.append(f"inference_engine_models_loaded {len(app_state.manager.loaded_models())}")

    lines.append("# HELP inference_engine_loaded_bytes Total bytes of loaded model weights on disk.")
    lines.append("# TYPE inference_engine_loaded_bytes gauge")
    lines.append(f"inference_engine_loaded_bytes {app_state.manager.loaded_bytes}")

    lines.append("# HELP inference_engine_memory_budget_bytes Configured hot-keep memory budget.")
    lines.append("# TYPE inference_engine_memory_budget_bytes gauge")
    lines.append(f"inference_engine_memory_budget_bytes {app_state.manager.memory_budget_bytes}")

    lines.append("# HELP inference_engine_models_available Total models discoverable in the registry.")
    lines.append("# TYPE inference_engine_models_available gauge")
    lines.append(f"inference_engine_models_available {len(app_state.registry.list_models())}")

    # Per-loaded-adapter prefix cache state (one series per model).
    lines.append("# HELP inference_engine_prefix_cache_capacity_bytes Configured prefix-cache capacity (per loaded model).")
    lines.append("# TYPE inference_engine_prefix_cache_capacity_bytes gauge")
    lines.append("# HELP inference_engine_prefix_cache_size_bytes Bytes of cached prefix state currently held.")
    lines.append("# TYPE inference_engine_prefix_cache_size_bytes gauge")
    for name, adapter, _desc in app_state.manager.iter_loaded():
        cap = getattr(adapter, "prefix_cache_capacity_bytes", 0)
        size = getattr(adapter, "prefix_cache_size_bytes", 0)
        labels = f'{{model="{name}",backend="{adapter.backend_name}"}}'
        lines.append(f"inference_engine_prefix_cache_capacity_bytes{labels} {cap}")
        lines.append(f"inference_engine_prefix_cache_size_bytes{labels} {size}")

    return "\n".join(lines) + "\n"
