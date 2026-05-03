from contextlib import asynccontextmanager

from fastapi import FastAPI

from . import __version__
from .api import admin, chat, completions, embeddings, evals, health, metrics, models, rerank
from .api.state import app_state
from .auth import load_keys
from .config import settings
from .evals import load_policy
from .observability import configure_logging, get_logger
from .otel import configure_tracing, instrument_fastapi, is_enabled, shutdown_tracing

# Configure tracing at import time so the global TracerProvider is set before
# any span is created or any FastAPI middleware is built. configure_tracing()
# is idempotent and a no-op when OTEL_ENABLED=false.
configure_tracing()


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging(settings.log_level)
    n_keys = load_keys()
    app_state.policy_registry = load_policy(settings.auto_eval_policies_file)
    log = get_logger("startup")
    log.info(
        "engine_ready",
        version=__version__,
        backend=app_state.backend_name,
        ollama_models_dir=str(settings.ollama_models_dir),
        mlx_models_dir=str(settings.mlx_models_dir),
        n_models=len(app_state.registry.list_models()),
        memory_budget_gb=settings.memory_budget_gb,
        otel_enabled=is_enabled(),
        auth_enabled=settings.auth_enabled,
        n_keys=n_keys,
        n_policies=len(app_state.policy_registry),
    )
    yield
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

app.include_router(health.router, tags=["health"])
app.include_router(metrics.router, tags=["metrics"])
app.include_router(models.router, tags=["models"])
app.include_router(chat.router, tags=["chat"])
app.include_router(completions.router, tags=["completions"])
app.include_router(embeddings.router, tags=["embeddings"])
app.include_router(rerank.router, tags=["rerank"])
app.include_router(evals.router, tags=["evals"])
app.include_router(admin.router, tags=["admin"])
