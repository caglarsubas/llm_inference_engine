from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    ollama_models_dir: Path = Field(
        default=Path("/Users/caglarsubasi/Desktop/prometa/pocs/auto-ml/ollama-models/models"),
        description="Root of the Ollama-format model store (contains 'manifests' and 'blobs').",
    )
    mlx_models_dir: Path = Field(
        default=Path.home() / ".cache" / "inference_engine" / "mlx",
        description="Directory holding MLX-format model directories (HF-style safetensors).",
    )
    vllm_models_file: Path = Field(
        default=Path(".vllm_models.json"),
        description=(
            "JSON config mapping engine model ids to vLLM endpoints. Empty / "
            "missing file means no vLLM-served models. Format: see "
            ".vllm_models.example.json."
        ),
    )
    prefer_mlx_over_gguf: bool = Field(
        default=True,
        description="When the same qualified name is in both registries, prefer MLX.",
    )
    default_model: str = Field(default="llama3.2:3b")

    host: str = Field(default="127.0.0.1")
    port: int = Field(default=8080)
    log_level: str = Field(default="INFO")

    # llama.cpp runtime
    n_gpu_layers: int = Field(default=-1, description="-1 = offload all layers to GPU (Metal).")
    n_ctx: int = Field(default=8192)
    n_threads: int = Field(default=0, description="0 = auto.")
    n_batch: int = Field(default=512)

    # llama.cpp prompt-prefix cache (LlamaRAMCache). 0 disables. Sized in bytes —
    # 2 GiB default leaves headroom on a 128 GB unified-memory system.
    prefix_cache_bytes: int = Field(default=2 * 1024**3)

    # Allocate the llama.cpp embedding pooling layer at load time so the same
    # adapter can serve /v1/embeddings alongside chat. Small memory cost on
    # decoder-only architectures; unsafe to disable if you want /v1/embeddings.
    llama_cpp_embedding_enabled: bool = Field(default=True)

    # Dynamic batching for /v1/embeddings. Concurrent requests for the same
    # adapter merge into one underlying batched call within a small wait
    # window. Real GPU-batch win on embedding-native models (bge/nomic/e5);
    # falls through cleanly when only a single request is in flight.
    batch_enabled: bool = Field(default=True)
    batch_max_wait_ms: float = Field(default=10.0, ge=0.0)
    batch_max_size: int = Field(default=32, ge=1)

    # MLX prompt cache. Multi-slot LRU keyed by token-prefix overlap. Each slot
    # holds an independent KV state for one prefix; lookup picks the slot with
    # the longest matching prefix. ``MLX_PREFIX_CACHE_MAX_SLOTS=1`` reproduces
    # the previous single-slot behaviour. Disable entirely with
    # ``MLX_PREFIX_CACHE_ENABLED=false``.
    mlx_prefix_cache_enabled: bool = Field(default=True)
    mlx_prefix_cache_max_slots: int = Field(default=4, ge=1)

    # adapter selection — kept for back-compat with the single-adapter mode.
    # In multi-format setups the ModelManager dispatches per-descriptor, so this
    # is effectively unused unless an external integration reads it.
    adapter: str = Field(default="llama_cpp", description="llama_cpp | mlx")

    # multi-model hot-keep budget. Default ~60 GB leaves plenty of headroom on a
    # 128 GB unified-memory M5 Max. Override via MEMORY_BUDGET_GB in .env.
    memory_budget_gb: float = Field(default=60.0)

    # OpenTelemetry. Disabled by default — flip to true with OTEL_ENABLED=true and
    # `make otel-up` (Jaeger), or point at any OTLP/gRPC collector.
    otel_enabled: bool = Field(default=False)
    otel_exporter_otlp_endpoint: str = Field(default="http://localhost:4317")
    otel_service_name: str = Field(default="inference-engine")

    # Per-key bearer auth. Off by default; flip on for any environment where
    # multiple agents/tenants share the engine.
    auth_enabled: bool = Field(default=False)
    auth_keys_file: Path = Field(default=Path(".auth_keys.json"))

    # LLM-as-a-Judge default. Override per-request via EvalRequest.judge_model.
    default_judge_model: str = Field(default="llama3.2:3b")

    # Server-side auto-eval policy. JSON file mapping (tenant, model) → rubrics.
    # Missing file is silently treated as "no policy" (per-request auto_eval
    # still works); malformed file fails startup loudly.
    auto_eval_policies_file: Path = Field(default=Path(".auto_eval_policies.json"))

    # Tool-call audit. When enabled, every chat completion emits span events
    # for inbound tool-result messages and outbound tool_calls. Argument and
    # result payloads are truncated to ``tool_audit_max_payload_chars`` to
    # keep span sizes bounded — the truncation flag is on the event so
    # downstream knows when full content is suppressed.
    tool_audit_enabled: bool = Field(default=True)
    tool_audit_max_payload_chars: int = Field(default=1024, ge=0)

    # Tool execution timing correlation. The engine records the timestamp of
    # every emitted gen_ai.tool_call event and matches it against the next
    # gen_ai.tool_result event with the same call_id, surfacing the wall-clock
    # gap as ``tool.execution_ms``. The store is bounded so a runaway agent
    # that opens calls but never closes them can't leak memory.
    tool_timing_ttl_seconds: float = Field(default=300.0, ge=0.0)
    tool_timing_max_entries: int = Field(default=10_000, ge=1)

    @property
    def memory_budget_bytes(self) -> int:
        return int(self.memory_budget_gb * 1024**3)


settings = Settings()
