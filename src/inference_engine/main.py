import asyncio
import time
from contextlib import asynccontextmanager, suppress

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from . import __version__
from .api import admin, chat, completions, embeddings, evals, health, metrics, models, rerank
from .api.state import app_state
from .auth import load_keys
from .config import settings
from .evals import load_policy
from .model_routing import activate_model_routing_policy_from_settings
from .model_routing_runtime import (
    build_model_routing_runtime_state,
    load_model_routing_pricing_catalog,
)
from .observability import configure_logging, get_logger
from .otel import configure_tracing, instrument_fastapi, is_enabled, shutdown_tracing
from .registry import get_openrouter_probe, get_probe, get_vllm_probe

# Configure tracing at import time so the global TracerProvider is set before
# any span is created or any FastAPI middleware is built. configure_tracing()
# is idempotent and a no-op when OTEL_ENABLED=false.
configure_tracing()

_READINESS_EXEMPT_PATHS = {"/v1/health", "/v1/ready", "/v1/metrics"}
_STARTING_RETRY_AFTER_SECONDS = 5


def _collect_startup_model_summary(n_keys: int) -> dict:
    t0 = time.perf_counter()

    # Walk the composite registry once with probe-aware partitioning so the
    # startup log honestly reflects the reachable surface — not just what's
    # on disk. GGUFs that llama-cpp-python can't load fall through to the
    # ollama_http source automatically and land in ``available``. Anything
    # every source rejects (or that registry parsing skipped) lands in
    # ``unavailable`` / ``skipped`` with structured reasons.
    probe = get_probe()

    def _accept(desc):
        if desc.format == "gguf":
            return probe.probe(desc).loadable
        if desc.format == "vllm":
            return get_vllm_probe().probe(desc).loadable
        if desc.format == "openrouter":
            return get_openrouter_probe().probe(desc).loadable
        return True

    loadable, rejected = app_state.registry.list_loadable(_accept)

    available_summary = [{"model": d.qualified_name, "format": d.format} for d in loadable]
    unavailable = []
    for desc in rejected:
        if desc.format == "gguf":
            result = probe.probe(desc)
            unavailable.append(
                {
                    "model": desc.qualified_name,
                    "reason": result.reason or "load_failed",
                    "detail": result.detail,
                }
            )
        elif desc.format == "vllm":
            result = get_vllm_probe().probe(desc)
            unavailable.append(
                {
                    "model": desc.qualified_name,
                    "reason": result.reason or "vllm_unavailable",
                    "detail": result.detail,
                }
            )
        elif desc.format == "openrouter":
            result = get_openrouter_probe().probe(desc)
            unavailable.append(
                {
                    "model": desc.qualified_name,
                    "reason": result.reason or "openrouter_unavailable",
                    "detail": result.detail,
                }
            )
        else:
            unavailable.append({"model": desc.qualified_name, "reason": "rejected_by_accept"})
    skipped = [
        {"model": s.qualified_name, "reason": s.reason}
        for source in getattr(app_state.registry, "_sources", ())
        for s in (getattr(source, "list_skipped", lambda: [])() or [])
    ]

    routing_policy = app_state.model_routing_policy
    routing_pricing = app_state.model_routing_pricing
    return {
        "version": __version__,
        "backend": app_state.backend_name,
        "ollama_models_dir": str(settings.ollama_models_dir),
        "mlx_models_dir": str(settings.mlx_models_dir),
        "ollama_http_endpoint": settings.ollama_http_endpoint or "<disabled>",
        "openrouter_models_file": str(settings.openrouter_models_file),
        "n_available": len(loadable),
        "n_unavailable": len(unavailable),
        "n_skipped": len(skipped),
        "available": available_summary,
        "unavailable": unavailable,
        "skipped": skipped,
        "memory_budget_gb": settings.memory_budget_gb,
        "otel_enabled": is_enabled(),
        "auth_enabled": settings.auth_enabled,
        "n_keys": n_keys,
        "n_policies": len(app_state.policy_registry),
        "model_routing_policy_required": settings.model_routing_policy_required,
        "model_routing_policy_active": routing_policy is not None,
        "model_routing_policy_id": (
            routing_policy.policy_id if routing_policy is not None else None
        ),
        "model_routing_policy_revision": (
            routing_policy.revision if routing_policy is not None else None
        ),
        "model_routing_policy_digest": (
            routing_policy.digest if routing_policy is not None else None
        ),
        "model_routing_policy_source": (
            routing_policy.source if routing_policy is not None else None
        ),
        "model_routing_request_enforcement": routing_policy is not None,
        "model_routing_pricing_digest": (
            routing_pricing.digest if routing_pricing is not None else None
        ),
        "startup_probe_duration_ms": round((time.perf_counter() - t0) * 1000, 2),
    }


async def _finish_startup(log, n_keys: int) -> None:
    try:
        summary = await asyncio.to_thread(_collect_startup_model_summary, n_keys)
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 - startup should fail typed
        app_state.mark_startup_failed(exc)
        log.error(
            "engine_startup_failed",
            error=str(exc),
            error_type=type(exc).__name__,
        )
        return

    app_state.mark_ready()
    log.info("engine_ready", **summary)


def _readiness_error_response() -> JSONResponse:
    readiness = app_state.readiness()
    error_type = "engine_starting"
    message = "Inference engine is starting; startup model probes are still running."
    if readiness["status"] == "error":
        error_type = "engine_startup_failed"
        message = "Inference engine startup failed; check engine logs."

    detail = {
        "message": message,
        "type": error_type,
        "code": error_type,
        "param": None,
        "status": readiness["status"],
        "ready": False,
        "retry_after_seconds": _STARTING_RETRY_AFTER_SECONDS,
    }
    if readiness.get("error"):
        detail["startup_error"] = readiness["error"]

    return JSONResponse(
        status_code=503,
        content={"detail": detail},
        headers={"Retry-After": str(_STARTING_RETRY_AFTER_SECONDS)},
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging(settings.log_level)
    n_keys = load_keys()
    app_state.policy_registry = load_policy(settings.auto_eval_policies_file)
    routing_policy = activate_model_routing_policy_from_settings()
    routing_pricing = load_model_routing_pricing_catalog(
        settings.model_routing_pricing_file,
        max_bytes=settings.model_routing_max_file_bytes,
    )
    app_state.model_routing_runtime = build_model_routing_runtime_state(
        routing_policy,
        routing_pricing,
        auth_enabled=settings.auth_enabled,
        expected_org_id=settings.model_routing_expected_org_id,
    )
    log = get_logger("startup")
    app_state.mark_starting()
    startup_task = asyncio.create_task(
        _finish_startup(log, n_keys),
        name="inference-engine-startup-probes",
    )
    try:
        yield
    finally:
        if not startup_task.done():
            startup_task.cancel()
            with suppress(asyncio.CancelledError):
                await startup_task
        await app_state.manager.shutdown()
        shutdown_tracing()


app = FastAPI(
    title="Local LLM Inference Engine",
    version=__version__,
    description="Backend-agnostic, OpenAI-compatible inference service.",
    lifespan=lifespan,
)

# Instrument right after construction so the ASGI middleware wraps every route
# below. Must run before the first request — module-level call is fine because
# Uvicorn imports this module before binding the socket.
instrument_fastapi(app)


@app.middleware("http")
async def startup_readiness_gate(request: Request, call_next):
    if (
        request.url.path.startswith("/v1/")
        and request.url.path not in _READINESS_EXEMPT_PATHS
        and not app_state.is_ready
    ):
        return _readiness_error_response()
    return await call_next(request)


app.include_router(health.router, tags=["health"])
app.include_router(metrics.router, tags=["metrics"])
app.include_router(models.router, tags=["models"])
app.include_router(chat.router, tags=["chat"])
app.include_router(completions.router, tags=["completions"])
app.include_router(embeddings.router, tags=["embeddings"])
app.include_router(rerank.router, tags=["rerank"])
app.include_router(evals.router, tags=["evals"])
app.include_router(admin.router, tags=["admin"])
