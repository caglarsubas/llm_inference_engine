"""Process-wide singletons: registry + ModelManager.

Two registries (Ollama GGUF + MLX) are composed behind a single
``CompositeRegistry``. The adapter factory dispatches per-descriptor so each
loaded model uses the runtime that matches its on-disk format.
"""

from __future__ import annotations

import time
from threading import RLock

from ..adapters import InferenceAdapter
from ..adapters.llama_cpp import LlamaCppAdapter
from ..config import settings
from ..evals import EvalRunner, PolicyRegistry, RubricRegistry
from ..manager import ModelManager
from ..model_routing import ActivatedModelRoutingPolicy
from ..observability import get_logger
from ..registry import (
    CompositeRegistry,
    MLXRegistry,
    ModelDescriptor,
    OllamaHttpRegistry,
    OllamaRegistry,
    OpenRouterRegistry,
    VLLMRegistry,
    get_openrouter_probe,
    get_probe,
    get_vllm_probe,
)


def _build_adapter_for(descriptor: ModelDescriptor) -> InferenceAdapter:
    if descriptor.format == "gguf":
        return LlamaCppAdapter()
    if descriptor.format == "mlx":
        # Lazy import — avoids requiring mlx-lm to be installed when the
        # store has no MLX models. The user installs it via `make install-mlx`.
        from ..adapters.mlx_lm import MLXAdapter  # noqa: PLC0415

        return MLXAdapter()
    if descriptor.format == "vllm":
        # Lazy import for symmetry — keeps the adapter cold-import path quick
        # in deployments that have no vLLM-served models configured.
        from ..adapters.vllm_adapter import VLLMAdapter  # noqa: PLC0415

        return VLLMAdapter()
    if descriptor.format == "openrouter":
        from ..adapters.openrouter_adapter import OpenRouterAdapter  # noqa: PLC0415

        return OpenRouterAdapter()
    if descriptor.format == "ollama_http":
        # Lazy import — keeps the cold-import path quick when the operator
        # hasn't configured an Ollama fallback (OLLAMA_HTTP_ENDPOINT="").
        from ..adapters.ollama_http import OllamaHttpAdapter  # noqa: PLC0415

        return OllamaHttpAdapter()
    raise ValueError(f"unsupported model format: {descriptor.format!r}")


class AppState:
    def __init__(self) -> None:
        log = get_logger("startup.appstate")
        ollama = OllamaRegistry(settings.ollama_models_dir)
        mlx = MLXRegistry(settings.mlx_models_dir)
        vllm = VLLMRegistry(
            settings.vllm_models_file,
            demanded_config_path=settings.vllm_demanded_models_file,
            local_snapshot_root=settings.hf_vlm_models_dir,
        )
        openrouter = OpenRouterRegistry(
            settings.openrouter_models_file,
            default_endpoint=settings.openrouter_endpoint,
            min_parameter_count_b=settings.openrouter_min_parameter_count_b,
        )
        # vLLM listed first so an explicitly-configured continuous-batching
        # path always wins over a local GGUF/MLX of the same qualified_name.
        # Local-format ordering (mlx-vs-gguf) follows the existing toggle.
        local_sources = (mlx, ollama) if settings.prefer_mlx_over_gguf else (ollama, mlx)

        # Ollama-HTTP fallback comes *after* the local sources: anything
        # llama.cpp / MLX can serve in-process wins on latency, only
        # llama.cpp-rejected GGUFs fall through to the HTTP path.  Empty
        # endpoint → registry stays inert; nothing to wire up.
        sources: tuple = (openrouter, vllm, *local_sources)
        if settings.ollama_http_endpoint:
            ollama_http = OllamaHttpRegistry(settings.ollama_http_endpoint)
            sources = (*sources, ollama_http)
            log.info(
                "ollama_http.fallback_enabled",
                endpoint=settings.ollama_http_endpoint,
            )

        self.registry = CompositeRegistry(sources)

        # Probe-aware resolver: for each model id, walk sources in order and
        # pick the first descriptor whose adapter would actually load.  GGUFs
        # are checked via the load probe (gemma4 etc. fail here and fall
        # through to ollama_http); vLLM descriptors must also be present on
        # the upstream's own /v1/models surface. This keeps ``manager.get(id)``
        # and ``/v1/models`` agreeing on what's reachable.
        def _accept(desc: ModelDescriptor) -> bool:
            if desc.format == "gguf":
                return get_probe().probe(desc).loadable
            if desc.format == "vllm":
                return get_vllm_probe().probe(desc).loadable
            if desc.format == "openrouter":
                return get_openrouter_probe().probe(desc).loadable
            return True

        self._accept = _accept

        def _resolve(model_id: str) -> ModelDescriptor | None:
            return self.registry.resolve(model_id, _accept)

        self.manager = ModelManager(
            registry=self.registry,
            adapter_factory=_build_adapter_for,
            memory_budget_bytes=settings.memory_budget_bytes,
            resolver=_resolve,
        )
        # Heterogeneous backends — report "auto" rather than a single backend.
        self.backend_name = "auto"

        # LLM-as-a-Judge plumbing.
        self.rubric_registry = RubricRegistry.with_builtins()
        self.eval_runner = EvalRunner(self.manager)
        # Policy is reloaded from disk in main.py's lifespan so a startup
        # failure surfaces with a clear log line rather than at first request.
        self.policy_registry: PolicyRegistry = PolicyRegistry([])
        self.model_routing_policy: ActivatedModelRoutingPolicy | None = None

        # Dynamic-batching coalescer for /v1/embeddings. Lazy: queues are
        # created per-adapter on first submit, automatically replaced when
        # ModelManager reloads an adapter (we key by id(adapter)).
        from ._batcher import EmbedCoalescer  # noqa: PLC0415 — avoid early import cycle

        self.embed_coalescer = EmbedCoalescer()

        from ..scheduler import TenantScheduler  # noqa: PLC0415 — avoid eager settings cycles

        self.scheduler = TenantScheduler()

        # TestClient route tests often bypass ASGI lifespan, so the default
        # state is ready. The real lifespan flips this to "starting" before
        # yielding and then marks it ready after the background probe pass.
        self._readiness_lock = RLock()
        self._readiness: dict = {
            "ready": True,
            "status": "ready",
            "message": "engine is ready",
            "started_at": None,
            "ready_at": time.time(),
            "error": None,
        }

    def mark_starting(self, message: str = "startup model probes are running") -> None:
        with self._readiness_lock:
            self._readiness = {
                "ready": False,
                "status": "starting",
                "message": message,
                "started_at": time.time(),
                "ready_at": None,
                "error": None,
            }

    def mark_ready(self, message: str = "engine is ready") -> None:
        with self._readiness_lock:
            started_at = self._readiness.get("started_at")
            self._readiness = {
                "ready": True,
                "status": "ready",
                "message": message,
                "started_at": started_at,
                "ready_at": time.time(),
                "error": None,
            }

    def mark_startup_failed(self, error: BaseException) -> None:
        with self._readiness_lock:
            started_at = self._readiness.get("started_at")
            self._readiness = {
                "ready": False,
                "status": "error",
                "message": "engine startup failed",
                "started_at": started_at,
                "ready_at": None,
                "error": f"{type(error).__name__}: {error}",
            }

    def readiness(self) -> dict:
        with self._readiness_lock:
            return dict(self._readiness)

    @property
    def is_ready(self) -> bool:
        return bool(self.readiness()["ready"])


app_state = AppState()
