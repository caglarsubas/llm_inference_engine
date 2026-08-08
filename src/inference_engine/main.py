import asyncio
import time
import uuid
from contextlib import asynccontextmanager, suppress

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from . import __version__, usage_ledger
from .api import (
    admin,
    chat,
    completions,
    embeddings,
    evals,
    health,
    metrics,
    models,
    rerank,
    tokenize,
)
from .api._models_snapshot import run_models_snapshot_refresher
from .api.errors import error_response, install_error_handlers
from .api.state import app_state
from .auth import load_keys
from .config import settings
from .evals import load_policy
from .model_plane_observer import (
    ModelPlaneObservationReporter,
    load_model_plane_observation_config,
)
from .model_routing import activate_model_routing_policy_from_settings
from .model_routing_runtime import (
    build_model_routing_rate_limiter,
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

# Priced inference surfaces, and the only paths that produce a billing record.
# /v1/rerank is excluded on purpose: it has no routing or pricing integration
# and already sits outside CERTIFIED_MODEL_WORKLOAD_SURFACE.
_USAGE_LEDGER_PATHS = {"/v1/chat/completions", "/v1/completions", "/v1/embeddings"}

_READINESS_EXEMPT_PATHS = {"/v1/health", "/v1/ready", "/v1/metrics"}
# Inference paths that don't live under /v1/. ``/tokenize`` and ``/detokenize``
# follow the vLLM/TGI convention of sitting at the root, but they still touch
# the model manager, so the startup gate has to cover them explicitly.
_READINESS_GUARDED_PATHS = {"/tokenize", "/detokenize"}
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
        "model_routing_rate_limit_scope": app_state.model_routing_rate_limiter.scope,
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


async def _run_observer_after_startup(
    startup_task: asyncio.Task,
    observer: ModelPlaneObservationReporter,
) -> None:
    try:
        await asyncio.shield(startup_task)
        await observer.run()
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 - observer cannot take inference down
        get_logger("model_plane.observer").error(
            "model_plane_observer_stopped",
            error_type=type(exc).__name__,
        )


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

    # Built directly rather than raised, so it bypasses the exception handlers;
    # go through error_response() so a 503 during startup carries the same
    # ``error`` envelope as every other failure.
    return error_response(
        detail,
        503,
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
    observer_config = load_model_plane_observation_config(settings)
    observer = (
        ModelPlaneObservationReporter(
            observer_config,
            app_state,
            models.collect_model_list,
            models.is_model_available,
        )
        if observer_config is not None
        else None
    )
    previous_rate_limiter = app_state.model_routing_rate_limiter
    rate_limiter = build_model_routing_rate_limiter(settings)
    try:
        await asyncio.to_thread(rate_limiter.ping)
    except Exception:
        await asyncio.to_thread(rate_limiter.close)
        raise
    app_state.model_routing_rate_limiter = rate_limiter
    app_state.model_plane_observer = observer
    log = get_logger("startup")
    app_state.mark_starting()
    startup_task = asyncio.create_task(
        _finish_startup(log, n_keys),
        name="inference-engine-startup-probes",
    )
    observer_task = (
        asyncio.create_task(
            _run_observer_after_startup(startup_task, observer),
            name="inference-engine-model-plane-observer",
        )
        if observer is not None
        else None
    )
    # Background-refresh the /v1/models snapshot so metadata discovery never
    # probe-loads on the request path (issue #69). Waits for the startup probe
    # pass so its first build reuses the warm probe cache. 0 disables it and
    # routes fall back to an off-thread fresh compute per request.
    snapshot_task = (
        asyncio.create_task(
            run_models_snapshot_refresher(
                models.snapshot_cache,
                models.build_model_views,
                interval_seconds=settings.models_snapshot_refresh_seconds,
                wait_for=startup_task,
            ),
            name="inference-engine-models-snapshot-refresher",
        )
        if settings.models_snapshot_refresh_seconds > 0
        else None
    )
    ledger_task = (
        asyncio.create_task(
            usage_ledger.run_usage_ledger_drain(
                usage_ledger.usage_ledger,
                interval_seconds=settings.usage_ledger_drain_interval_seconds,
            ),
            name="inference-engine-usage-ledger-drain",
        )
        if settings.usage_ledger_enabled
        else None
    )
    try:
        yield
    finally:
        if ledger_task is not None:
            if not ledger_task.done():
                ledger_task.cancel()
            with suppress(asyncio.CancelledError):
                await ledger_task
            # One last pass so the records buffered between the final drain
            # and shutdown are still billed.
            await usage_ledger.drain_once(usage_ledger.usage_ledger)
        if snapshot_task is not None:
            if not snapshot_task.done():
                snapshot_task.cancel()
            with suppress(asyncio.CancelledError):
                await snapshot_task
        models.snapshot_cache.clear()
        if observer_task is not None:
            if not observer_task.done():
                observer_task.cancel()
            with suppress(asyncio.CancelledError):
                await observer_task
        if not startup_task.done():
            startup_task.cancel()
            with suppress(asyncio.CancelledError):
                await startup_task
        try:
            await app_state.manager.shutdown()
        finally:
            try:
                await asyncio.to_thread(rate_limiter.close)
            finally:
                app_state.model_routing_rate_limiter = previous_rate_limiter
                app_state.model_plane_observer = None
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

# OpenAI-shaped {"error": {...}} alongside FastAPI's {"detail": ...}.
install_error_handlers(app)


class _LedgerFlushOnSend:
    """ASGI wrapper that flushes a streamed record once its response is done.

    The SSE generator's own teardown owns the flush so the record carries real
    token counts, but that teardown runs only if something iterates the response
    body. A response whose iterator never starts — the client vanished, the
    first ``send`` failed — would emit no record at all, and a request that was
    served and never billed is the one failure this channel exists to prevent.
    Wrapping the send is what makes the emission unconditional: it runs whether
    the body completed, raised, or never began.

    ``flush`` is idempotent, so on the normal path the generator's earlier and
    richer flush is the one that counts and this is a no-op.
    """

    def __init__(self, response, record) -> None:
        self._response = response
        self._record = record

    def __getattr__(self, name):
        return getattr(self._response, name)

    async def __call__(self, scope, receive, send) -> None:
        try:
            await self._response(scope, receive, send)
        finally:
            usage_ledger.flush(self._record)


@app.middleware("http")
async def model_usage_ledger(request: Request, call_next):
    """Open and close the per-request ``prometa.model-usage.v1`` record.

    Defined first so it ends up *innermost*: ``add_middleware`` inserts at
    position 0, which makes the last-defined middleware outermost. Innermost
    means ``request.state.request_id`` is already stamped when this runs and
    the status observed here is the route's, not the readiness gate's.
    """
    if request.url.path not in _USAGE_LEDGER_PATHS:
        return await call_next(request)

    record = usage_ledger.begin(
        request_id=getattr(request.state, "request_id", ""),
        route=request.url.path,
    )
    try:
        response = await call_next(request)
    except Exception:
        if record is not None:
            # Every handler that answers a status of its own sits *below* this
            # middleware, so an exception arriving here has none left to run:
            # Starlette's server-error boundary above us answers 500, and that
            # is the status the caller is billed against.
            record.http_status = 500
        usage_ledger.flush(record)
        raise
    if record is None:
        return response
    record.http_status = response.status_code
    # A streaming response has not produced a token yet at this point, so the
    # SSE generator's own teardown owns the flush, with _LedgerFlushOnSend as
    # the backstop for the body that never runs. Everything rejected before the
    # stream opened answers non-2xx and is flushed here, because that generator
    # never runs at all.
    if record.stream and 200 <= response.status_code < 300:
        return _LedgerFlushOnSend(response, record)
    usage_ledger.flush(record)
    return response


@app.middleware("http")
async def request_id_header(request: Request, call_next):
    """Stamp ``x-request-id`` on every response.

    An inbound id wins so a caller's correlation survives the hop — the
    orchestra-python-sdk model gateway sends its own under
    ``x-orchestra-runtime-request-id``, and generic clients use
    ``x-request-id``. Otherwise we mint one, which is what makes a support
    conversation about a single bad completion tractable at all.
    """
    incoming = (
        request.headers.get("x-request-id")
        or request.headers.get("x-orchestra-runtime-request-id")
        or ""
    ).strip()
    # Bound it: this value is echoed back and lands in logs, so an unbounded
    # caller-controlled string is not something we want to propagate.
    request_id = incoming[:200] if incoming else f"req_{uuid.uuid4().hex}"
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["x-request-id"] = request_id
    return response


@app.middleware("http")
async def startup_readiness_gate(request: Request, call_next):
    path = request.url.path
    guarded = (
        path.startswith("/v1/") and path not in _READINESS_EXEMPT_PATHS
    ) or path in _READINESS_GUARDED_PATHS
    if guarded and not app_state.is_ready:
        return _readiness_error_response()
    return await call_next(request)


app.include_router(health.router, tags=["health"])
app.include_router(metrics.router, tags=["metrics"])
app.include_router(models.router, tags=["models"])
app.include_router(chat.router, tags=["chat"])
app.include_router(completions.router, tags=["completions"])
app.include_router(embeddings.router, tags=["embeddings"])
app.include_router(rerank.router, tags=["rerank"])
app.include_router(tokenize.router, tags=["tokenize"])
app.include_router(evals.router, tags=["evals"])
app.include_router(admin.router, tags=["admin"])
