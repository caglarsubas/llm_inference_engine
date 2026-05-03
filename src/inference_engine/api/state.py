"""Process-wide singletons: registry + ModelManager.

Two registries (Ollama GGUF + MLX) are composed behind a single
``CompositeRegistry``. The adapter factory dispatches per-descriptor so each
loaded model uses the runtime that matches its on-disk format.
"""

from __future__ import annotations

from ..adapters import InferenceAdapter
from ..adapters.llama_cpp import LlamaCppAdapter
from ..config import settings
from ..evals import EvalRunner, PolicyRegistry, RubricRegistry
from ..manager import ModelManager
from ..registry import (
    CompositeRegistry,
    MLXRegistry,
    ModelDescriptor,
    OllamaRegistry,
    VLLMRegistry,
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
    raise ValueError(f"unsupported model format: {descriptor.format!r}")


class AppState:
    def __init__(self) -> None:
        ollama = OllamaRegistry(settings.ollama_models_dir)
        mlx = MLXRegistry(settings.mlx_models_dir)
        vllm = VLLMRegistry(settings.vllm_models_file)
        # vLLM listed first so an explicitly-configured continuous-batching
        # path always wins over a local GGUF/MLX of the same qualified_name.
        # Local-format ordering (mlx-vs-gguf) follows the existing toggle.
        local_sources = (mlx, ollama) if settings.prefer_mlx_over_gguf else (ollama, mlx)
        self.registry = CompositeRegistry((vllm, *local_sources))
        self.manager = ModelManager(
            registry=self.registry,
            adapter_factory=_build_adapter_for,
            memory_budget_bytes=settings.memory_budget_bytes,
        )
        # Heterogeneous backends — report "auto" rather than a single backend.
        self.backend_name = "auto"

        # LLM-as-a-Judge plumbing.
        self.rubric_registry = RubricRegistry.with_builtins()
        self.eval_runner = EvalRunner(self.manager)
        # Policy is reloaded from disk in main.py's lifespan so a startup
        # failure surfaces with a clear log line rather than at first request.
        self.policy_registry: PolicyRegistry = PolicyRegistry([])

        # Dynamic-batching coalescer for /v1/embeddings. Lazy: queues are
        # created per-adapter on first submit, automatically replaced when
        # ModelManager reloads an adapter (we key by id(adapter)).
        from ._batcher import EmbedCoalescer  # noqa: PLC0415 — avoid early import cycle
        self.embed_coalescer = EmbedCoalescer()


app_state = AppState()
