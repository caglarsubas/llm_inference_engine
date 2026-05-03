# Local LLM Inference Engine (v1)

A laptop-class inference service that wraps your existing **Ollama-format GGUF model store** and exposes an **OpenAI-compatible HTTP API** — without requiring the Ollama daemon.

The service is **backend-agnostic**: a thin adapter interface (`InferenceAdapter`) sits between the API layer and the actual model runtime, so we can swap llama.cpp for MLX-LM, vLLM, or SGLang without touching the routes.

## Why this stack

Synthesised from the multi-LLM guide (GPT, Claude, Gemini, Grok all converge on the same shape):

- **Engines — `llama.cpp` (GGUF) + `mlx-lm` (Apple Silicon native).** Both implement the same `InferenceAdapter` ABC. The `ModelManager` dispatches per-descriptor so each loaded model uses the runtime that matches its on-disk format. llama.cpp gives universal hardware reach + access to the existing 135 GB Ollama GGUF store; MLX is native to the M5 Max's unified Metal stack and tracks newer architectures faster than mainline llama.cpp.
- **Service layer — FastAPI.** OpenAI-compatible `/v1/chat/completions`, streaming via SSE, structured JSON mode, model registry, health checks.
- **Registry — composite, direct manifest reader.** `OllamaRegistry` parses the Docker-distribution layout (`manifests/<registry>/<ns>/<model>/<tag>` + `blobs/sha256-*`) so we use the existing store with zero copying or daemon overhead. `MLXRegistry` scans for HF-style safetensors directories. `CompositeRegistry` merges them, with `prefer_mlx_over_gguf` controlling collision resolution.
- **Roadmap — backend-agnostic from day one.** Same routes serve both backends today; `vLLM` / `SGLang` slot in as additional adapters whenever GPU-server workloads matter.

## Containerized deployment (horizontally scalable)

The repo ships a `Dockerfile` + `docker-compose.yml` that bring up N engine replicas behind an nginx load balancer. **Verified working**: 6/6 round-robin distribution at `--scale=2`, live scale-up to 4 replicas with all healthy and serving traffic.

```bash
make compose-build                       # ~3-5 min: llama-cpp-python compiles from source
make compose-up                          # 2 replicas (default)
make compose-up-scale REPLICAS=4         # bring up 4 replicas
make compose-logs                        # tail engine + nginx
make compose-down                        # tear down

# Smoke test
curl http://127.0.0.1:8080/v1/health
curl http://127.0.0.1:8080/v1/models | jq '.data[0]'
```

### Topology

```
client ──► nginx :8080 ──┬─► engine.1 :8080 (internal)
                          ├─► engine.2 :8080
                          └─► engine.N :8080
                              │
                              └─► /models (read-only mount, shared)
                                  /config (read-only mount, shared)
```

Engine replicas share a read-only mount of the Ollama model store and a read-only config dir (auth keys + auto-eval policy). Override host paths via `.env`:

```bash
OLLAMA_MODELS_HOST_DIR=/path/to/auto-ml/ollama-models/models
CONFIG_HOST_DIR=./docker/config
LB_PORT=8080
REPLICAS=2
MEMORY_BUDGET_GB=12.0
```

### Constraints — what's honest about this stack

**Out-of-the-box LB is round-robin, not header-stickiness.** nginx OSS doesn't support header-based hash routing across Docker-discovered replicas — its `hash` directive operates on the explicit list of `server` entries, and `--scale` doesn't expose per-replica hostnames. The included config uses dynamic DNS resolution (`resolver 127.0.0.11` + `set $upstream_engine` + `proxy_pass http://$upstream_engine`) so each request really does hit a different replica via Compose's embedded DNS round-robin. **Verified**: 12 requests across 2 replicas → 6/6 split.

**Trade-off**: round-robin breaks tenant cache locality across requests. Specifically:
- Prefix cache (rounds 8/12) — each replica warms independently
- Tool-execution timing correlation (round 21) — turn N hits replica A, turn N+1 hits replica B → no `tool.execution_ms`
- Embed coalescer queue (round 16) — only coalesces within a single replica

For multi-turn agent workloads where this matters, use the **HAProxy overlay** (round 26) instead — see the next section.

### Tenant-sticky routing via HAProxy overlay

`docker-compose.haproxy.yml` swaps nginx for HAProxy with `balance hdr(Authorization)`. Same Authorization Bearer token always lands on the same replica, restoring per-tenant cache locality across requests.

```bash
make compose-up-sticky                   # 2 replicas, default
make compose-up-sticky REPLICAS=4
make compose-down-sticky

# HAProxy stats UI for ops
open http://127.0.0.1:8404/stats
```

#### How it works

| capability | nginx (default) | HAProxy (overlay) |
|---|---|---|
| Service discovery | DNS at request-time via `set $upstream` + `proxy_pass` | `server-template engine- 10 engine:8080 resolvers docker init-addr none` (auto-fills slots from DNS) |
| Load balancing | round-robin across DNS round-robin | **`balance hdr(Authorization)` consistent-hash** |
| Same Authorization → same replica | ❌ | ✅ |
| Auto-discovers `--scale` changes | ✅ | ✅ |
| Active healthcheck | passive (`max_fails`) | active (`option httpchk GET /v1/health`, `inter 10s rise 2 fall 3`) |
| Stats UI | none | `:8404/stats` |
| SSE-friendly | `proxy_buffering off` | `option http-server-close`, `http-reuse never`, `timeout tunnel 1h` |
| HTTP version forwarded | 1.1 | 1.1 |

#### Verified end-to-end (round 26)

Same Authorization × 10 requests at scale=2:

```
  same-token request distribution: engine-1=10, engine-2=0
  ✓ STICKINESS CONFIRMED: all 10 same-token requests landed on one replica
```

8 distinct tokens at scale=2:

```
  8 distinct tokens distributed: engine-1=1, engine-2=7
  ✓ DISTRIBUTION CONFIRMED: different tokens hit different replicas
```

Live scale-up `--scale=2` → `--scale=4` while traffic flows:

```
  engine-1: status=UP
  engine-2: status=UP
  engine-3: status=UP                    ← came online from DNS re-resolution
  engine-4: status=UP 1/3                ← finishing healthcheck cycle
  engine-5: status=MAINT (resolution)    ← over-provisioned slot, no replica yet
  ...
```

Server slots are over-provisioned (10 by default) so scale-out up to 10 replicas requires no HAProxy restart — slots fill via active DNS resolution within ~5 seconds. Active healthchecks gate which slots actually receive traffic, so a crashed replica stops getting hit within `inter * fall = 30s`.

#### When to choose which

| workload | LB choice |
|---|---|
| Stateless, single-shot completions (eval batches, doc summarization, classification) | **nginx** — simple round-robin maximises utilisation evenly |
| Multi-turn agent traffic, conversation continuity, prefix-cache reliance | **HAProxy** — stickiness preserves cache locality + tool-timing correlation |
| Anonymous traffic only (`AUTH_ENABLED=false`) | either works — both collapse anonymous traffic into one bucket either way |
| Need stats / dashboard | **HAProxy** — built-in stats page; nginx OSS has none |

#### Honest trade-off

Anonymous traffic (no Authorization header) hashes to a single bucket and effectively all anonymous requests land on one replica. That's a documented design choice — for multi-tenant production deployments you'd run `AUTH_ENABLED=true` anyway, which gives every tenant a distinct hash bucket.

**MLX adapter doesn't run in containers.** Apple Silicon Docker Desktop runs a Linux VM with no Metal passthrough; mlx-lm needs Metal. The composite registry handles this cleanly — the MLX directory mount is empty inside the container, the registry returns zero MLX models, and llama.cpp serves everything. The container's llama.cpp build is CPU-only by default; switch to CUDA on a GPU host with `--build-arg CMAKE_ARGS="-DGGML_CUDA=on"`.

**No metric-driven auto-scaling on plain Compose.** `docker compose up --scale` is manual. The `deploy.replicas` block in the compose file is read by **Docker Swarm** (`docker stack deploy`) for declarative scaling; for true HPA-style autoscaling, deploy on Kubernetes. Both paths work without code changes — the engine itself is stateless modulo the instance-local caches called out above.

**Cross-instance state is not shared.** Process-global stores (tool-timing, embed coalescer, prefix cache) are per-replica. That's fine for cache-warming workloads (each replica warms separately) but it means signals like `tool.execution_ms` only fire when both turns of a tool exchange land on the same replica. With sticky sessions via HAProxy/Traefik, this works; with plain round-robin, it's best-effort. For real distributed state, plug Redis behind the audit module — out of scope here.

### What's installed in the container

- llama-cpp-python compiled from source (CPU-only by default, CUDA build with `CMAKE_ARGS="-DGGML_CUDA=on"`)
- The OTel extra (`uv sync --extra otel`) so `OTEL_ENABLED=true` works against an external collector
- Non-root runtime user (UID 10001) for blast-radius reduction
- Container `HEALTHCHECK` hitting `/v1/health` every 15s

What's deliberately NOT installed:
- `mlx-lm` (Apple-Silicon-only)
- `[dev]` extras
- The full model store (mounted read-only from host instead)

## Quick start

```bash
# from the project root
cd /Users/caglarsubasi/Desktop/prometa/pocs/llm_inference_engine_v1

cp .env.example .env

# install both backends (Metal-accelerated llama.cpp + MLX-LM)
make install-metal     # llama-cpp-python with -DGGML_METAL=on
make install-mlx       # mlx-lm + mlx-metal

# (optional) grab a small MLX model for the demo
make download-mlx-model    # default: mlx-community/Llama-3.2-1B-Instruct-4bit (~700 MB)

# enumerate the unified model registry (Ollama GGUF + MLX, no server needed)
make list-models

# run the API
make run            # http://127.0.0.1:8080

# in another shell — exercise the API end-to-end
make smoke
```

The Metal build links `llama-cpp-python` against your M5 Max GPU so the entire GGUF model is offloaded automatically (`N_GPU_LAYERS=-1`). MLX models always run on the unified-memory Metal stack natively.

## API surface

OpenAI-compatible — drop into any client that already speaks the OpenAI schema (Python SDK, LangChain, `curl`, etc.). Override `base_url` to `http://127.0.0.1:8080/v1`.

| Method | Path                          | Notes                                                                      |
|--------|-------------------------------|----------------------------------------------------------------------------|
| GET    | `/v1/health`                  | Liveness + every currently-loaded model + budget usage                     |
| GET    | `/v1/metrics`                 | Prometheus-format scrape: loaded count, loaded bytes, budget, total models |
| GET    | `/v1/models`                  | All models discoverable in the unified registry                            |
| GET    | `/v1/models/{model:tag}`      | Single model details (size, blob path, backend)                            |
| POST   | `/v1/chat/completions`        | Blocking + SSE streaming (`stream: true`)                                  |
| POST   | `/v1/completions`             | Legacy raw-prompt completions — bypasses chat templating                   |
| POST   | `/v1/embeddings`              | OpenAI-compatible embeddings (llama.cpp); MLX returns 501                  |
| POST   | `/v1/rerank`                  | Cohere/Jina-shaped relevance ranking via embedding cosine similarity        |
| GET    | `/v1/evals/rubrics`           | List built-in + registered rubrics                                         |
| GET    | `/v1/evals/policy`            | Active server-side auto-eval policy entries (Prometa-driven)               |
| POST   | `/v1/admin/policies:reload`   | Hot-reload `AUTO_EVAL_POLICIES_FILE`; atomic swap on success, rejects malformed |
| POST   | `/v1/evals/run`               | LLM-as-a-Judge: candidate + rubric → structured verdict                    |
| POST   | `/v1/chat/completions`        | (extension) `auto_eval: {rubrics, mode}` runs evals inline or in background |

### Multi-model, multi-backend hot-keep

The engine keeps multiple models warm in memory simultaneously — across **different backends** — and routes each request to the matching adapter. Demonstrated end-to-end with MLX and llama.cpp side-by-side:

```
cold-MLX   Llama-3.2-1B-Instruct-4bit:mlx   4247 ms   ← MLXAdapter loads safetensors
cold-GGUF  llama3.2:1b                       278 ms   ← LlamaCppAdapter loads alongside
warm-MLX   Llama-3.2-1B-Instruct-4bit:mlx    176 ms   ← already resident, no reload
warm-GGUF  llama3.2:1b                        31 ms   ← already resident, no reload

loaded_models: [
  {"model": "Llama-3.2-1B-Instruct-4bit:mlx", "backend": "mlx",        "size_bytes": 712578487},
  {"model": "llama3.2:1b",                    "backend": "llama_cpp",  "size_bytes": 1321082688}
]
loaded_bytes / budget: 2.03 GB / 60.00 GB
```

When the next load would exceed `MEMORY_BUDGET_GB` (default 60 GB), the **least-recently-used** model is unloaded first. Touch order is updated on every `get()`, so a model that's actively being hit is never the eviction victim. The same eviction policy works across formats — an MLX model can evict a GGUF model and vice versa.

### curl example

```bash
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "llama3.2:1b",
    "messages": [{"role": "user", "content": "Hello in one word."}],
    "max_tokens": 8
  }' | jq
```

## Layout

```
src/inference_engine/
├── main.py              # FastAPI app + lifespan + module-level OTel wiring + load_keys()
├── config.py            # pydantic-settings (.env-driven)
├── manager.py           # ModelManager — LRU multi-model hot-keep, per-format dispatch
├── observability.py     # span() bridges structlog + OTel; Span.bind() mutates both
├── otel.py              # OTel SDK setup, NoOp span shim, FastAPI auto-instrumentation
├── auth.py              # bearer-token auth, Identity, key index, FastAPI dependency
├── cancellation.py      # Cancellation flag + watch_disconnect() watchdog
├── schemas.py           # OpenAI-compatible request/response models
├── evals/
│   ├── rubrics.py       # RubricSpec, built-in helpfulness/correctness/safety, RubricRegistry
│   ├── runner.py        # EvalRunner: candidate + rubric → judge → Verdict (clean/repaired/failed)
│   ├── policy.py        # PolicyMatch / PolicyEntry / PolicyRegistry — server-side auto-eval rules
│   └── schemas.py       # EvalRequest, EvalResponse, Verdict, PolicyList
├── api/
│   ├── state.py         # composite registry + ModelManager + adapter dispatch + EvalRunner
│   ├── health.py
│   ├── metrics.py       # /v1/metrics (Prometheus format)
│   ├── models.py        # gated by require_identity when auth on
│   ├── embeddings.py    # /v1/embeddings (OpenAI-compatible; llama.cpp only, MLX 501)
│   ├── rerank.py        # /v1/rerank — Cohere/Jina-shaped relevance via embedding cosine
│   ├── evals.py         # /v1/evals/rubrics + /v1/evals/run
│   ├── admin.py         # /v1/admin/policies:reload (hot-reload of auto-eval policy)
│   ├── _auto_eval.py    # blocking + background batch helpers for chat-attached eval
│   ├── _batcher.py      # EmbedCoalescer — dynamic batching for /v1/embeddings
│   ├── _tool_audit.py   # gen_ai.tool_call / gen_ai.tool_result event emission with truncation
│   └── chat.py          # /v1/chat/completions (+ SSE, gen_ai.* spans, watchdog, tenant, auto_eval, tool audit)
├── adapters/
│   ├── base.py          # InferenceAdapter ABC (stream/generate accept cancel=)
│   ├── llama_cpp.py     # llama-cpp-python implementation (GGUF) — streaming cancel
│   ├── mlx_lm.py        # mlx-lm implementation (Apple Silicon native) — streaming cancel
│   └── vllm_adapter.py  # vLLM HTTP client (continuous batching on a CUDA upstream)
└── registry/
    ├── ollama.py        # parses Ollama manifests → ModelDescriptor
    ├── mlx.py           # scans MLX model directories
    ├── vllm.py          # parses .vllm_models.json (HTTP endpoints, no local files)
    └── composite.py     # merges multiple registry sources
Dockerfile                      # multi-stage build, llama-cpp-python from source, non-root runtime
docker-compose.yml              # N engine replicas + nginx LB + healthchecks + volume mounts
docker-compose.haproxy.yml      # overlay: HAProxy LB with header-based tenant stickiness
docker-compose.vllm.yml         # overlay: single-GPU vLLM sidecar (count: 1)
docker-compose.vllm-multigpu.yml# overlay: multi-GPU vLLM (two services pinned via device_ids)
docker-compose.otel.yml         # overlay: Jaeger sidecar for OTel trace UI
docker/nginx.conf               # dynamic-resolution upstream + SSE-friendly buffering
docker/haproxy.cfg              # balance hdr(Authorization) + dynamic DNS + active healthcheck
docker/config/                  # mount target for auth_keys.json + auto_eval_policies.json
scripts/
├── list_models.py            # CLI to enumerate the unified registry
├── download_mlx_model.py     # snapshot_download from mlx-community/*
├── smoke_test.py             # blocking + streaming end-to-end check
└── stress_test.py            # concurrent-traffic harness with p50/p95/p99 + throughput
tests/
├── test_registry.py            # Ollama manifest parser
├── test_mlx_registry.py        # MLX directory scanner
├── test_composite_registry.py  # merge / collision / fallthrough
├── test_manager.py             # ModelManager LRU + budget enforcement
├── test_state_dispatch.py      # adapter factory picks the right backend
├── test_observability.py       # span() bridges structlog + OTel via in-memory exporter
├── test_auth.py                # bearer-token resolution, anonymous fallback, file loading
├── test_cancellation.py        # Cancellation flag + watch_disconnect watchdog
├── test_chat_streaming.py      # chat.py disconnect → cancel → adapter break wire
├── test_concurrency.py         # load dedup, parallel cold loads, cache-hit-during-load, watchdog cleanup
├── test_evals.py               # rubric registry, runner with fakes, JSON repair, schema mismatches
├── test_prefix_cache.py        # LlamaRAMCache install/teardown gating + introspection
├── test_auto_eval.py           # AutoEvalSpec validation + blocking/background batch + per-rubric isolation
├── test_auto_eval_policy.py    # Policy file loading, wildcard match, first-match-wins, resolver vs request
├── test_mlx_prefix_cache.py    # MLX cache state machine: miss / full / trimmed / disabled / unload
├── test_tool_audit.py          # tool_call / tool_result span events, payload truncation, disable gate
├── test_embeddings.py          # /v1/embeddings schema + 200 / 400 / 404 / 501 paths + span attrs
├── test_dynamic_batching.py    # adapter capability fallback + coalescer flush semantics + slicing
├── test_completions.py         # /v1/completions raw-prompt path + multi-prompt + spans
├── test_pairwise.py            # pairwise rubric: score mapping, runner enforcement, route + spans
├── test_admin_policies.py      # POST /v1/admin/policies:reload — atomic swap + auth gate + bad-file rejection
├── test_streaming_tool_audit.py # ToolCallReassembler unit tests + end-of-stream gen_ai.tool_call events
├── test_tool_timing.py          # ToolCallTimingStore (TTL/LRU) + tool.execution_ms cross-event correlation
├── test_per_rubric_judges.py    # AutoEvalSpec.judge_models override + resolver precedence + dispatch
├── test_rerank.py               # /v1/rerank schema + cosine ranking + top_n + 404/422/501 + spans
└── test_vllm_adapter.py         # VLLMRegistry parsing + VLLMAdapter HTTP behaviour via httpx.MockTransport
```

## Configuration

All knobs live in `.env` (see `.env.example`):

| var                      | default                                                                                  | meaning                                                  |
|--------------------------|------------------------------------------------------------------------------------------|----------------------------------------------------------|
| `OLLAMA_MODELS_DIR`      | `/Users/caglarsubasi/Desktop/prometa/pocs/auto-ml/ollama-models/models`                  | Root with `manifests/` and `blobs/`                      |
| `MLX_MODELS_DIR`         | `~/.cache/inference_engine/mlx`                                                          | Where `download_mlx_model.py` snapshots HF repos         |
| `PREFER_MLX_OVER_GGUF`   | `true`                                                                                   | On a name collision, MLX wins (faster on Apple Silicon)  |
| `DEFAULT_MODEL`          | `llama3.2:3b`                                                                            | Used by smoke test by default                            |
| `N_GPU_LAYERS`           | `-1`                                                                                     | llama.cpp: `-1` = offload all layers to Metal            |
| `N_CTX`                  | `8192`                                                                                   | Context window                                           |
| `N_THREADS`              | `0`                                                                                      | `0` = auto                                               |
| `ADAPTER`                | `llama_cpp`                                                                              | Legacy single-adapter mode (manager dispatch ignores it) |
| `MEMORY_BUDGET_GB`       | `60.0`                                                                                   | LRU evicts past this                                     |
| `PREFIX_CACHE_BYTES`     | `2147483648` (2 GiB)                                                                     | llama.cpp `LlamaRAMCache` capacity; `0` disables         |
| `MLX_PREFIX_CACHE_ENABLED` | `true`                                                                                 | MLX prompt cache master switch                            |
| `MLX_PREFIX_CACHE_MAX_SLOTS` | `4`                                                                                  | Number of independent prefix slots per loaded MLX model; `1` = single-slot legacy behaviour |
| `LLAMA_CPP_EMBEDDING_ENABLED` | `true`                                                                              | Allocate llama.cpp embedding pooling layer at load; needed for `/v1/embeddings`             |
| `BATCH_ENABLED`          | `true`                                                                                   | Coalesce concurrent `/v1/embeddings` requests; `false` = pass-through                      |
| `BATCH_MAX_WAIT_MS`      | `10`                                                                                     | Wait window before flushing a partial batch                                                |
| `BATCH_MAX_SIZE`         | `32`                                                                                     | Force flush when queued inputs hit this count                                              |
| `OTEL_ENABLED`           | `false`                                                                                  | Master switch — when true, sets up OTLP/gRPC exporter    |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | `http://localhost:4317`                                                              | Any OTLP/gRPC collector (Jaeger, otel-collector, …)      |
| `OTEL_SERVICE_NAME`      | `inference-engine`                                                                       | `service.name` resource attribute                        |
| `AUTH_ENABLED`           | `false`                                                                                  | Bearer-token gate on `/v1/models` and `/v1/chat/completions` |
| `AUTH_KEYS_FILE`         | `.auth_keys.json`                                                                        | JSON array of `{"key": "...", "tenant": "..."}` records  |
| `DEFAULT_JUDGE_MODEL`    | `llama3.2:3b`                                                                            | Used by `/v1/evals/run` when `judge_model` is not set    |
| `AUTO_EVAL_POLICIES_FILE`| `.auto_eval_policies.json`                                                               | JSON array of `{name, match, auto_eval}` rules; missing = no policy |
| `TOOL_AUDIT_ENABLED`     | `true`                                                                                   | Emit `gen_ai.tool_*` span events on every chat completion           |
| `TOOL_AUDIT_MAX_PAYLOAD_CHARS` | `1024`                                                                             | Per-event truncation cap for arguments / result content             |
| `TOOL_TIMING_TTL_SECONDS` | `300`                                                                                  | TTL for the call_id → emit-timestamp store; older entries swept on insert |
| `TOOL_TIMING_MAX_ENTRIES` | `10000`                                                                                | Hard cap on the timing store; oldest entries LRU-evicted past this        |

## Observability — real OpenTelemetry, dual-emission

`observability.span()` now emits to **both** sinks at once:

* **structlog** — human-readable `span.start` / `span.end` records with all attributes (always on).
* **OpenTelemetry** — real OTLP/gRPC spans with the [Generative AI semantic-convention](https://opentelemetry.io/docs/specs/semconv/gen-ai/) attributes (when `OTEL_ENABLED=true`).

`Span.bind(...)` mutates both sinks in place, so attributes added mid-flight (output token counts, finish reason) reach the final `span.end` record and the exported OTel span — the previous version silently discarded the inner `bind()` and lost those attrs.

### Run with traces

```bash
make install-otel        # one-time: pulls opentelemetry-* into the venv
make otel-up             # docker compose: starts Jaeger on :16686 (UI) + :4317 (OTLP gRPC)
make run-otel            # starts the engine with OTEL_ENABLED=true
# … hit a few endpoints …
open http://127.0.0.1:16686    # Jaeger UI; pick service "inference-engine"
make otel-down           # tear down
```

### What a single `POST /v1/chat/completions` looks like in Jaeger

```
POST /v1/chat/completions                      ← HTTP server span (FastAPIInstrumentor)
├── POST /v1/chat/completions http receive
├── model.acquire                              ← ModelManager.get(): cache hit / cold load
├── chat.generate                              ← inference, carrying gen_ai.* attrs
│     gen_ai.system               = llama_cpp
│     gen_ai.request.model        = llama3.2:1b
│     gen_ai.request.max_tokens   = 8
│     gen_ai.request.temperature  = 0.2
│     gen_ai.usage.input_tokens   = 41         ← post-bind: filled after generate() returns
│     gen_ai.usage.output_tokens  = 5          ← post-bind
│     gen_ai.response.finish_reason = stop     ← post-bind
│     duration_ms                 = 24.38
└── POST /v1/chat/completions http send
```

Cold model load shows up as a long `model.acquire` (e.g. 263 ms) above an unchanged `chat.generate`; warm hits drop `model.acquire` to <1 ms. That's exactly the kind of evidence Prometa's signal layer needs to attribute latency to load vs. compute.

### Plugging into Prometa

Point any OTLP/gRPC collector at `OTEL_EXPORTER_OTLP_ENDPOINT` instead of Jaeger. Standard OTel env vars (`OTEL_RESOURCE_ATTRIBUTES`, `OTEL_TRACES_SAMPLER`, …) are honored by the SDK directly. Service identity is pre-set to `service.name=inference-engine`, `service.version=<package version>`.

### Grafana dashboards — full observability stack

`make obs-up` brings up the engine plus a complete Grafana / Prometheus / Jaeger / OTel-Collector stack. All metrics on the dashboard are derived from the same OTLP traces the engine already emits — no extra instrumentation. The pipeline is:

```
engine ── OTLP/gRPC ──▶ otel-collector ──┬──▶ jaeger        (trace search UI)
                              │          │
                              │          ├──▶ spanmetrics ──▶ prometheus ──▶ grafana
                              │          │       calls_total, duration_ms histogram
                              │          │
                              │          └──▶ sumconnector ──▶ prometheus ──▶ grafana
                              │                  inference_tokens_input/output_total
                              │
                              └─ engine /v1/metrics scraped directly by prometheus
                                   inference_engine_models_loaded, prefix_cache_size_bytes, …
```

`spanmetrics` derives request rate + latency histograms from spans, preserving `prometa.tenant`, `gen_ai.request.model`, and `gen_ai.system` as labels. `sumconnector` reads `gen_ai.usage.input_tokens` / `output_tokens` off each span and sums into counters — that's how the tokens/sec panels work without the engine emitting any direct metrics.

```bash
make obs-up         # engine + grafana + prometheus + jaeger + otel-collector
make obs-load       # drives ~40 requests across 2 tenants × 2 models so panels populate
make obs-down
```

URLs (host-side):

| Service     | URL                                | Notes                                  |
| ----------- | ---------------------------------- | -------------------------------------- |
| Grafana     | `http://127.0.0.1:3000`            | admin / admin; dashboard is provisioned |
| Prometheus  | `http://127.0.0.1:9090`            | for ad-hoc PromQL                      |
| Jaeger      | `http://127.0.0.1:16686`           | trace search; click any series → trace |
| Engine LB   | `http://127.0.0.1:8090`            | unchanged (nginx fronts engine ×N)     |

Open Grafana, then **Dashboards → Inference Engine → Inference Engine — Overview**. The dashboard is split into seven rows:

* **Traffic overview** — req/sec, active tenants, active models, error rate (single-stat tiles).
* **Request rate breakdowns** — req/sec sliced by route, tenant, model, backend.
* **Latency** — `p50 / p95 / p99` per route + `p95` per model, computed from the spanmetrics histogram.
* **Token throughput** — input / output tokens/sec by tenant, plus a combined-by-model panel.
* **Tenant × model traffic matrix** — last-5m request count per `(tenant, model)` pair as a heatmap, the panel Prometa uses to spot tenant-specific routing skew.
* **Eval signals (LLM-as-a-Judge)** — `eval.run` rate by rubric + p95 latency by rubric, populated by the auto-eval policy or explicit `/v1/evals/*` calls.
* **Operational health** — loaded models per replica, loaded model bytes, prefix-cache size by model. These are scraped from the engine's own `/v1/metrics` (Prometheus exposition), not from OTel — they're cheap counters/gauges the engine maintains directly.

Trace correlation is wired both ways: any series in a metric panel can be clicked to land on the matching trace in Jaeger (via `exemplarTraceIdDestinations` on the Prometheus datasource), and Jaeger spans link back to the metric series via `tracesToMetrics`.

#### Caveats and notes

* **Tenant labels show `anonymous` until `AUTH_ENABLED=true`.** The default obs stack has auth off, so all bearer tokens hit the same anonymous tenant — the dashboard surfaces this in the "Active tenants" tile (will read 1, not 2). Turn auth on (`AUTH_ENABLED=true` + a configured `auth_token_map`) and the tenant breakdowns light up.
* **First minute is empty.** spanmetrics flushes every 15 s and Prometheus scrapes every 15 s, so panels need ~30 s of traffic before they read non-zero. `make obs-load` is the friction-free way to seed.
* **Cumulative temporality.** spanmetrics is configured with `AGGREGATION_TEMPORALITY_CUMULATIVE`, so all PromQL panels use `rate(...)` over a 5-minute window. Don't switch to `irate` — the cumulative reset behavior at 5-minute reset windows produces spikes that aren't real traffic.
* **Engine-native metrics aren't OTLP.** `inference_engine_*` gauges (loaded models, prefix-cache size) are scraped straight off `engine:8080/v1/metrics` — they don't need the OTel collector at all. The collector pipeline only carries the trace-derived metrics.

Need a different overlay? `obs-up` composes on top of `docker-compose.yml`, so you can add `-f docker-compose.haproxy.yml` for sticky LB or `-f docker-compose.vllm.yml` for a vLLM upstream and the dashboard keeps working — `gen_ai.system` will surface `vllm` alongside `llama_cpp` and the panels split by backend.

## LLM-as-a-Judge — `/v1/evals/*`

The most Prometa-aligned slice of the engine: a candidate response goes in, a structured verdict comes out, and the verdict is stamped onto an OTel span carrying provenance back to the original completion. That's the substrate for continuous evaluation, regression detection, and self-healing loops.

### Built-in rubrics

| name              | shape    | requires           | output keys                   | score extraction              |
|-------------------|----------|--------------------|-------------------------------|-------------------------------|
| `helpfulness`     | scalar   | —                  | `score` (1-5), `justification` | `score` as float              |
| `correctness`     | scalar   | `expected`         | `correct` (bool), `reason`     | `1.0` if correct else `0.0`   |
| `safety`          | scalar   | —                  | `safe` (bool), `concerns` (list) | `1.0` if safe else `0.0`      |
| `pairwise_quality`| pairwise | `response_b`       | `winner` (`A`/`B`/`tie`), `reason` | `1.0`/`0.0`/`0.5` (unknown→`0.0`)|

All four converge on a single `[0, 1]` (or 1–5) numeric `score` so downstream aggregation treats them the same way. Custom rubrics drop in via `RubricRegistry.register(...)` — that's the seam where Prometa's control plane can ship org-specific judges.

#### Per-rubric judge model overrides

A single `auto_eval` spec can route different rubrics to different judge models — fast cheap judge for high-volume rubrics, stronger one for accuracy-sensitive ones. Per-rubric resolution precedence: `judge_models[rubric]` > `judge_model` > `DEFAULT_JUDGE_MODEL`.

```json
{
  "rubrics": ["safety", "correctness"],
  "judge_model": "llama3.2:1b",
  "judge_models": {"correctness": "llama3.2:3b"},
  "mode": "blocking",
  "expected": "..."
}
```

Each `eval.run` span carries its own `eval.judge.model` attribute, so Prometa sees per-rubric judge usage and cost split out of the box. This works in policy entries too — set `judge_models` on the policy's `auto_eval` and the engine routes per-rubric for every covered chat without client coordination.

#### Pairwise comparison

`pairwise_quality` evaluates **two** candidate responses to the same prompt and picks the better one. Foundation for preference data (DPO-style training sets), A/B model evaluation, and tournament-style model selection. The request shape adds one field:

```bash
curl -X POST .../v1/evals/run -d '{
  "rubric": "pairwise_quality",
  "prompt": "What is the capital of France?",
  "response": "Paris.",
  "response_b": "The capital of France is the city known as Paris ...",
  "judge_model": "llama3.2:3b",
  "candidate_completion_id": "chatcmpl-A",
  "candidate_b_completion_id": "chatcmpl-B"
}'
# → {"verdict": {"parsed": {"winner": "A", "reason": "..."}, "score": 1.0}, ...}
```

Both candidate completion ids stamp onto the `eval.run` span (`eval.candidate.completion_id` and `eval.candidate_b.completion_id`), so Prometa can join the pairwise verdict back to **both** original chat completions automatically.

### Endpoints

```bash
# list rubrics
curl http://127.0.0.1:8080/v1/evals/rubrics

# run a single eval
curl -s -X POST http://127.0.0.1:8080/v1/evals/run \
  -H 'content-type: application/json' \
  -d '{
    "rubric": "correctness",
    "prompt": "What is 7 * 8?",
    "response": "The answer is 54.",
    "expected": "56",
    "judge_model": "llama3.2:3b",
    "candidate_model": "llama3.2:1b",
    "candidate_completion_id": "chatcmpl-abc123"
  }'
```

Sample response:

```json
{
  "id": "eval-e570f56b8cee466b8c7762c0bc7c6bb0",
  "object": "eval",
  "rubric": "correctness",
  "judge_model": "llama3.2:3b",
  "candidate_model": "llama3.2:1b",
  "candidate_completion_id": "chatcmpl-abc123",
  "verdict": {
    "score": 0.0,
    "parsed": {"correct": false, "reason": "Incorrect calculation, 7 * 8 = 56"},
    "raw": "{\"correct\": false, \"reason\": \"...\"}",
    "parse_status": "clean"
  },
  "duration_ms": 699.04
}
```

### Provenance — every eval emits a self-contained signal

The runner builds an OTel span (`eval.run`) carrying:

```
eval.rubric.name              = correctness
eval.judge.model              = llama3.2:3b
eval.candidate.model          = llama3.2:1b           ← what produced the response
eval.candidate.completion_id  = chatcmpl-abc123       ← correlation back to the chat span
eval.score                    = 0.0
eval.parse_status             = clean | repaired | failed
gen_ai.usage.input_tokens     = 126
gen_ai.usage.output_tokens    = 25
gen_ai.system                 = llama_cpp
prometa.tenant                = <caller>
```

Send these spans to Prometa's OTLP endpoint (`OTEL_EXPORTER_OTLP_ENDPOINT`) and every eval becomes a first-class signal — joinable to the candidate completion via `eval.candidate.completion_id`, sliceable by tenant, rubric, judge model.

### `/v1/rerank` (Cohere/Jina-shaped)

Relevance ranking on top of the existing embedding pathway. Same model registry, same auth, same observability surface, same dynamic-batching coalescer (round 16) — query + documents go through one batched embed call, then we cosine-similarity rank.

```bash
curl -X POST .../v1/rerank -d '{
  "model": "bge-small-en-v1.5:gguf",
  "query": "What is Python?",
  "documents": [
    "Python is a popular programming language used for AI.",
    "Rust is a systems programming language.",
    "The cookie recipe calls for butter and sugar."
  ],
  "top_n": 2,
  "return_documents": true
}'

# →
# {
#   "results": [
#     {"index": 0, "relevance_score": 0.91, "document": "Python is a popular..."},
#     {"index": 1, "relevance_score": 0.62, "document": "Rust is a systems..."}
#   ],
#   ...
# }
```

#### Span surface

```
rerank.run
  gen_ai.system               = llama_cpp
  gen_ai.request.model        = bge-small-en-v1.5:gguf
  rerank.documents_count      = 5
  rerank.top_n                = 3
  rerank.results_returned     = 3
  embedding.dimensions        = 384
  gen_ai.usage.input_tokens   = 96
  batch.adapter_action        = batch | serial | fallback
```

#### Quality caveat (same as `/v1/embeddings`)

The cosine-similarity approach is **only as discriminative as the loaded embedding model**. With `llama3.2:1b` (a chat model), the E2E returns scores of essentially `1.000` across all docs because chat models produce near-identical vectors for any input. For production RAG, drop a purpose-built embedding GGUF (`bge-small-en-v1.5`, `nomic-embed-text-v1.5`, `e5-small-v2`) into the model store and the ranking becomes meaningful.

#### Out of scope: dedicated cross-encoder rerankers

Real cross-encoder rerankers (`bge-reranker`, `jina-reranker`) take `(query, doc)` pairs through a classification head and produce a single relevance scalar per pair — substantially higher quality than embedding-cosine. They need a new adapter capability (`rerank_pair(query, doc) -> float`) that's distinct enough to warrant its own round; the current implementation gets you the API shape, the observability, and a working signal on real embedding models without that complexity.

### Continuous chat batching via vLLM-as-subprocess

`/v1/chat/completions` continuous-batching for autoregressive decode is the largest engineering line on the roadmap and the one we deliberately stayed out of for rounds 16+ — implementing it well means owning a paged-attention scheduler. Rather than reimplementing vLLM, the engine plugs vLLM in as an upstream sidecar via a third adapter slot.

```
                    ┌──────────────────────────────────────────────┐
                    │              inference engine                │
                    │  /v1/chat/completions (auth, eval, audit)    │
                    │                  │                           │
                    │      ┌───────────┼───────────┐               │
                    │      ▼           ▼           ▼               │
                    │  llama.cpp     MLX        VLLMAdapter        │
                    │  (local)      (local)    (HTTP client)       │
                    └─────────────────────────────────┬────────────┘
                                                      │
                          ┌───────────────────────────┘
                          ▼
                  ┌────────────────────┐
                  │   vLLM sidecar     │  GPU host (CUDA)
                  │ continuous-batching│  /v1/chat/completions
                  │   PagedAttention   │
                  └────────────────────┘
```

What the engine adds for vLLM-served traffic that vLLM alone doesn't:

- **Same auth + tenant attribution** (round 5)
- **Same auto-eval policy enforcement** including pairwise + per-rubric judges (rounds 13/22)
- **Same tool-call audit + execution timing correlation** (rounds 14/20/21)
- **Same OTel `gen_ai.*` spans** with everything joinable in Prometa
- **Same OpenAI-compat surface** so clients don't change endpoints when traffic moves between local llama.cpp / MLX and remote vLLM

#### Configuring vLLM-served models

Drop a `.vllm_models.json` (config path overridable via `VLLM_MODELS_FILE`) listing each vLLM upstream:

```json
[
  {
    "name": "llama-3.2-1b-instruct",
    "tag": "vllm",
    "endpoint": "http://vllm:8000",
    "model_id": "meta-llama/Llama-3.2-1B-Instruct",
    "size_bytes": 2400000000
  }
]
```

Clients then send `model: "llama-3.2-1b-instruct:vllm"` and the engine routes through `VLLMAdapter` to the upstream. Mixing local and remote: a single engine instance can serve `llama3.2:1b` (Ollama GGUF, llama.cpp), `Llama-3.2-1B-Instruct-4bit:mlx` (MLX), and `llama-3.2-1b-instruct:vllm` (vLLM remote) at the same time — different `model` ids in the same registry, different adapters under the hood, identical observability surface.

#### Deployment — `make compose-vllm-up`

`docker-compose.vllm.yml` overlays a vLLM service onto the engine compose stack:

```bash
# On a CUDA host with nvidia-container-toolkit installed:
make compose-vllm-up                    # engine x N + vllm + nginx
make compose-vllm-down

# Configurable via .env:
VLLM_MODEL=meta-llama/Llama-3.2-1B-Instruct
VLLM_GPU_COUNT=1                        # 1 (default) or "all"; for pinning use multi-GPU overlay
VLLM_GPU_UTIL=0.85
VLLM_MAX_MODEL_LEN=4096
HF_TOKEN=<your hf token>                # for gated models
HF_CACHE_HOST_DIR=./.cache/huggingface  # persists weights across restarts
```

The vLLM service has a 180-second healthcheck `start_period` because cold-start (model download + CUDA-graph compilation) can run several minutes on first boot.

#### GPU configuration — prerequisites and pinning

GPU access flows through Compose's standard `deploy.resources.reservations.devices` block (the same spec [Docker documents officially](https://docs.docker.com/compose/how-tos/gpu-support/)). Prerequisites on the host:

* NVIDIA GPU driver installed
* [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) configured for the Docker daemon
* `nvidia-smi` works (use it to discover GPU IDs for pinning)

The single-GPU overlay (`docker-compose.vllm.yml`) uses `count: ${VLLM_GPU_COUNT:-1}` — it grabs the first available GPU by default; set `VLLM_GPU_COUNT=all` to give one vLLM container every GPU on the host (rarely what you want in multi-tenant; pin instead).

For **multiple vLLM services on a multi-GPU host**, use `docker-compose.vllm-multigpu.yml`. It runs two vLLM containers pinned via `device_ids: ['0']` and `['1']`:

```bash
nvidia-smi                              # discover GPU IDs (0, 1, ...)

# Default: vllm on GPU 0 (1B), vllm-secondary on GPU 1 (3B). Override via .env:
#   VLLM_GPU_ID_PRIMARY=0
#   VLLM_GPU_ID_SECONDARY=1
#   VLLM_MODEL_PRIMARY=meta-llama/Llama-3.2-1B-Instruct
#   VLLM_MODEL_SECONDARY=meta-llama/Llama-3.2-3B-Instruct
make compose-vllm-multigpu-up
```

The matching `.vllm_models.json` lists both upstreams (already in `.vllm_models.example.json` — copy it across):

```json
[
  {"name": "llama-3.2-1b-instruct", "tag": "vllm",
   "endpoint": "http://vllm:8000",
   "model_id": "meta-llama/Llama-3.2-1B-Instruct"},
  {"name": "llama-3.2-3b-instruct", "tag": "vllm",
   "endpoint": "http://vllm-secondary:8000",
   "model_id": "meta-llama/Llama-3.2-3B-Instruct"}
]
```

Clients then send `model: "llama-3.2-1b-instruct:vllm"` or `"llama-3.2-3b-instruct:vllm"` and the engine routes each through its own GPU-pinned vLLM upstream. Adding a third model is two more entries: another vLLM service block in compose with `device_ids: ['2']`, and another `.vllm_models.json` entry pointing at it.

`count` and `device_ids` are mutually exclusive in the Compose GPU spec, which is why the single-GPU and multi-GPU paths live in separate overlay files rather than as one parameterised service.

#### Honest constraints

- **vLLM doesn't run on Apple Silicon, macOS Docker Desktop, or pure-CPU containers.** It needs CUDA (or supported AMD/Intel GPUs). On the M5 Max laptop the user's working from, `make compose-vllm-up` will start the vLLM container which will then crash-loop trying to find a GPU. The engine itself stays up and serves local llama.cpp / MLX models normally; vLLM-routed model ids return upstream-503 until vLLM is reachable.
- **One model per vLLM process.** Multi-model = multiple vLLM containers on different ports + multiple `.vllm_models.json` entries. The engine is the multiplexer.
- **No prefix-cache introspection on vLLM.** vLLM's PagedAttention is excellent but its OpenAI-compatible HTTP API doesn't expose per-call hit counts the way our local adapters do. `prefix_cache_*` properties on `VLLMAdapter` report `disabled` so the chat span attrs stay uniform across backends.
- **Embeddings unsupported.** `VLLMAdapter.embed()` raises `EmbeddingsNotSupportedError`; `/v1/embeddings` against a vLLM model returns 501. Continue to use llama.cpp for embeddings (round 15) or wire a separate vLLM container with an embedding model + a custom adapter override.

### Dynamic batching (embeddings)

Concurrent `/v1/embeddings` requests for the same loaded adapter merge into a single underlying `adapter.embed()` call within a small wait window. **Per-adapter capability detection** picks the right inner path automatically:

* **Encoder embedding GGUFs** (bge, nomic, e5, …) → `adapter_action="batch"`. One forward pass on the concatenated inputs; real GPU batching.
* **Chat-model GGUFs misused for embedding** → first batch attempt hits `llama_decode returned -1`, the adapter caches `supports_batched_embed=False`, and every subsequent call goes straight to `adapter_action="serial"` (or `"fallback"` on the very first call where the probe runs and falls through). HTTP-level coalescing still happens; the inner GPU work just serializes.

Three knobs:

| env var               | default | what it does                                                                 |
|-----------------------|---------|------------------------------------------------------------------------------|
| `BATCH_ENABLED`       | `true`  | Master switch. `false` is a clean pass-through with single-request semantics. |
| `BATCH_MAX_WAIT_MS`   | `10`    | How long a request waits for siblings to coalesce. Solo traffic pays this latency once. |
| `BATCH_MAX_SIZE`      | `32`    | Force a flush when total queued inputs hit this, ignoring the wait window.   |

Every embedding span carries the full coalescing story:

```
embeddings.run
  batch.id              = 1            ← join key — same id on every coalesced span
  batch.coalesced_with  = 10           ← total requests merged into this batch
  batch.total_inputs    = 10           ← inputs across all merged requests
  batch.wait_ms         = 12.05        ← this caller's wait time
  batch.adapter_action  = batch | serial | fallback
```

#### Verified end-to-end

```
fire 10 concurrent /v1/embeddings on llama3.2:1b
total wall: 125 ms                             ← all 10 in flight at once
batch.flushed coalesced=10 total_inputs=10     ← one adapter.embed call, not 10
batch.adapter_action=fallback                  ← chat model, GPU batch failed once,
                                                 cached + went serial after
```

Same workload on an encoder embedding GGUF reports `adapter_action="batch"` instead, with the inner forward pass running as a true GPU batch.

#### Out of scope: dynamic batching for chat completions

`/v1/chat/completions` is **not** dynamically batched. Continuous batching for autoregressive decode (vLLM-style) requires reimplementing the inference loop on top of `llama_cpp` ctypes / `mlx.core` — multi-round project, not a one-shot. Today, concurrent chat requests for the same adapter serialize through the per-adapter lock (round 6's concurrency design). Documented here so the trade-off is visible rather than implied: for high-QPS chat workloads, plug a dedicated continuous-batching backend (vLLM, SGLang) behind the existing `InferenceAdapter` ABC.

### `/v1/embeddings`

OpenAI-compatible embeddings endpoint for RAG retrievers — same model registry, same auth, same observability surface as chat completions. Closes the last big OpenAI-compat gap.

```bash
curl -s -X POST http://127.0.0.1:8080/v1/embeddings \
  -H 'content-type: application/json' \
  -d '{"model": "llama3.2:1b", "input": "Vector this please"}'

# →
# {
#   "object": "list",
#   "data": [{"object": "embedding", "index": 0, "embedding": [0.943, 0.200, 0.632, ...]}],
#   "model": "llama3.2:1b",
#   "usage": {"prompt_tokens": 4, "completion_tokens": 0, "total_tokens": 4}
# }
```

Batch inputs work too: `"input": ["alpha", "beta", "gamma"]` returns one vector per string in request order.

#### Architecture

* **llama.cpp** — adapters load with `embedding=True` so the same loaded model serves both `/v1/chat/completions` and `/v1/embeddings`. Inputs are processed **serially** inside `adapter.embed()` rather than batched, because batched embedding decode can fail with `llama_decode returned -1` on decoder-only chat models. Throughput cost is negligible for RAG-sized batches; reliability across architectures is the win.
* **MLX** — no first-class embeddings API, so the adapter raises `EmbeddingsNotSupportedError` and the route returns **HTTP 501** with the backend name in the body. That's the signal a deployment needs to load a llama.cpp embedding model alongside its MLX chat model.

```bash
curl -s -X POST .../v1/embeddings -d '{"model": "Llama-3.2-1B-Instruct-4bit:mlx", "input": "x"}'
# → HTTP 501  {"detail": "embeddings not supported by mlx backend"}
```

#### Span surface

Same `gen_ai.*` semconv shape as chat plus embedding-specific attrs:

```
span = embeddings.run
  gen_ai.system             = llama_cpp
  gen_ai.request.model      = bge-small-en-v1.5:gguf
  gen_ai.usage.input_tokens = 14
  embedding.batch_size      = 3
  embedding.dimensions      = 384
  prometa.tenant            = ...
```

#### Embedding quality caveat

The `embedding=True` flag lets any GGUF emit a vector — but **chat models produce low-quality embeddings for retrieval**. For production RAG drop a purpose-built embedding GGUF (`bge-small-en-v1.5`, `nomic-embed-text-v1.5`, `e5-small-v2`, etc.) into the Ollama model store; the registry will pick it up automatically. The endpoint shape and observability are identical regardless.

### Tool-call audit logs

Tool-using agents are dark matter without observability into what tools they actually invoked. Every `/v1/chat/completions` now emits **OpenTelemetry span events** for both halves of the tool-calling lifecycle:

| event                  | when                                           | key attributes                                                  |
|------------------------|------------------------------------------------|------------------------------------------------------------------|
| `gen_ai.tool_result`   | inbound — agent passed in a `role="tool"` message  | `gen_ai.tool.call.id`, `gen_ai.tool.name`, `gen_ai.tool.result.content` |
| `gen_ai.tool_call`     | outbound — model emitted `tool_calls` in its reply | `gen_ai.tool.call.id`, `gen_ai.tool.name`, `gen_ai.tool.call.arguments` |

Both event shapes truncate the variable-length payload (`content` / `arguments`) to `TOOL_AUDIT_MAX_PAYLOAD_CHARS` (default 1024) and surface a `*_truncated=true` flag so downstream readers don't mistake a clipped string for the original.

The chat span itself binds aggregate counts (`tool_audit.tool_results_in`, `tool_audit.tool_calls_out`) so dashboards can plot "tool intensity per turn" without scanning every event.

#### Verified end-to-end

```bash
# Synthetic agent conversation: assistant called get_weather earlier; tool
# result is being fed back to the model.
curl -X POST .../v1/chat/completions -d '{
  "model": "llama3.2:1b",
  "messages": [
    {"role": "user", "content": "What is the weather in SF?"},
    {"role": "assistant", "content": null, "tool_calls": [
      {"id": "call_abc123", "type": "function",
       "function": {"name": "get_weather", "arguments": "{\"city\":\"San Francisco\"}"}}
    ]},
    {"role": "tool", "tool_call_id": "call_abc123", "name": "get_weather", "content": "Cloudy, 12C"}
  ]
}'

# → chat.generate span carries:
#     event=gen_ai.tool_result
#       gen_ai.tool.call.id              = call_abc123
#       gen_ai.tool.name                 = get_weather
#       gen_ai.tool.result.content       = "Cloudy, 12C"
#       gen_ai.tool.result.content_truncated = false
#     bound:
#       tool_audit.tool_results_in       = 1
#       tool_audit.tool_calls_out        = 0
```

#### Prometa correlation

The platform's agent-graph view joins on `gen_ai.tool.call.id`. A single tool call generates a `gen_ai.tool_call` event in the chat span where the model invoked it, and a `gen_ai.tool_result` event in the next chat span where the agent fed the result back — same id, two events, two spans. Prometa stitches them into one tool-execution record.

#### Execution timing correlation (round 21)

Every `gen_ai.tool_call` event records its emit timestamp in a process-global, TTL+LRU bounded store keyed by `gen_ai.tool.call.id`. When a `gen_ai.tool_result` event arrives later for the same id, the elapsed wall-clock is stamped as **`tool.execution_ms`** on the result event — the agent-side latency between "model decided to call this tool" and "agent fed back the result".

```
turn N                              span event
  emit_tool_calls                   gen_ai.tool_call    call.id=call_xyz  (timestamp recorded)
       ↓ ~1700 ms wall clock        (agent runs the tool externally)
turn N+1
  emit_tool_results                 gen_ai.tool_result  call.id=call_xyz
                                                        tool.execution_ms = 1701.71
```

Bounds:

* **TTL** — `TOOL_TIMING_TTL_SECONDS` (default 300s). Entries older than this are swept on every record() so a tool that's never resolved doesn't pin memory.
* **LRU max_entries** — `TOOL_TIMING_MAX_ENTRIES` (default 10000). Hard cap against runaway agents that open calls but never close them.

Edge cases handled (and unit-tested):

* Tool result with no matching prior call → the event still fires, just without `tool.execution_ms`. We never fabricate a value.
* Same call_id resolved twice → only the first result event gets timing; the second is unannotated.
* Empty / blank call_id → silently skipped (defensive against malformed events).

**Single-process caveat**: the store is in-memory per uvicorn worker. With multiple workers, a tool result that arrives at a different worker than the one that emitted the call sees no timing. Single-worker is the default deployment shape; documented for completeness.

#### Streaming coverage (round 20)

Streaming-mode tool calls are now audited too. OpenAI streams tool calls as a sequence of per-`index` fragments — `id` and `function.name` typically arrive in the first chunk, `function.arguments` arrives as a string concatenated across chunks. The chat-stream path runs a `ToolCallReassembler` over the deltas, passes the raw deltas through to the SSE client unchanged (so OpenAI clients reassemble themselves on the wire), and emits **one** `gen_ai.tool_call` span event per call at end-of-stream with the fully reassembled `arguments`. `tool_audit.tool_calls_out` on the chat span gives the per-stream count.

When a streaming request is cancelled mid-flight (round 5), the partially-assembled tool call is **not** emitted as an event — half-formed arguments would be misleading.

### Server-side auto-eval policy (Prometa-authoritative)

Per-request `auto_eval` lets clients attach rubrics to a single chat completion. That works for ad-hoc debugging — but it's a **coordination point**: every agent has to know which rubrics to send. For platform compliance (safety must run, always) the source of truth needs to be the engine, not the client.

The auto-eval policy file (`AUTO_EVAL_POLICIES_FILE`, default `.auto_eval_policies.json`) is a JSON array of `(match → auto_eval)` rules. Prometa's control plane writes it; the engine reads it at startup. **When a policy entry matches a request, the request's own `auto_eval` field is ignored** — the policy plane is authoritative over which rubrics run. Clients that want fully request-driven evals are simply not covered by a policy.

```json
[
  {
    "name": "agent-runtime-quality",
    "match": {"tenant": "agent-runtime", "model": "llama3.2:1b"},
    "auto_eval": {
      "rubrics": ["safety", "helpfulness"],
      "mode": "background",
      "judge_model": "llama3.2:3b"
    }
  },
  {
    "name": "compliance-baseline",
    "match": {"tenant": "*", "model": "*"},
    "auto_eval": {
      "rubrics": ["safety"],
      "mode": "background",
      "judge_model": "llama3.2:3b"
    }
  }
]
```

`"*"` matches anything; resolution is **first-match-wins**, so list specific entries before the wildcard fallback.

`GET /v1/evals/policy` returns the active policy entries (in priority order) so the control plane can verify what's installed.

#### Hot-reload — rotate rubric coverage without restarting

`POST /v1/admin/policies:reload` re-reads `AUTO_EVAL_POLICIES_FILE` and atomically replaces the in-memory registry. The reload is **strictly validated** by the same code path startup uses: a malformed file returns HTTP 400 and the existing registry is preserved untouched. In-flight requests that already resolved through the previous registry continue to use it (the resolver returns by value, not by reference), so there's no torn-state failure mode.

```bash
# Prometa rotates compliance rubrics
$ cat > .auto_eval_policies.json <<EOF
[{"name": "compliance", "match": {"tenant": "*"}, "auto_eval": {"rubrics": ["safety"]}}]
EOF
$ curl -X POST .../v1/admin/policies:reload
{"object":"policy.reload","reloaded_at":1777759282,"policies_loaded":1,"source":".auto_eval_policies.json"}
```

Span: `admin.policies.reload` carries `policy.previous_count`, `policy.loaded_count`, and the file path so audit logs show exactly which reload bumped coverage from N to M policies. Same auth gate as every other tenant-scoped endpoint.

#### Provenance — every chat span shows whether eval was policy-driven

```
span=chat.generate
  auto_eval.from_policy            = true
  auto_eval.policy.name            = compliance-baseline
  auto_eval.policy.match_tenant    = *
  auto_eval.policy.match_model     = *
  auto_eval.rubrics                = [safety]
  auto_eval.mode                   = background
  auto_eval.judge_model            = llama3.2:3b
```

Combined with the existing `eval.candidate.completion_id` linkage on every `eval.run` span, Prometa can reconstruct the full chain *which policy → which chat → which eval verdict* purely from joined OTel spans. No coordination protocol with clients required.

#### Verified end-to-end

```bash
# install a policy file
cat > .auto_eval_policies.json <<'EOF'
[{"name": "compliance-baseline", "match": {"tenant": "*", "model": "*"},
  "auto_eval": {"rubrics": ["safety"], "mode": "background", "judge_model": "llama3.2:3b"}}]
EOF

# send a chat with NO auto_eval field
curl -X POST .../v1/chat/completions -d '{
  "model": "llama3.2:1b",
  "messages": [{"role":"user","content":"What is 2+2?"}],
  "max_tokens": 8
}'

# → response.evals is null (background mode)
# → spans: chat.generate auto_eval.from_policy=true,
#          eval.run    eval.score=1.0 eval.candidate.completion_id=chatcmpl-...
```

### Auto-judge attached to `/v1/chat/completions`

Closes the continuous-evaluation loop end-to-end: every chat completion can carry a per-request `auto_eval` directive that fires rubrics against the assistant's response. Two modes:

| mode | latency | verdicts visible via |
|---|---|---|
| `blocking` | chat latency + slowest rubric (rubrics run concurrently) | inline `evals: [...]` on the response **and** `eval.run` spans |
| `background` | unchanged from a vanilla chat completion | `eval.run` spans only, joined back via `eval.candidate.completion_id` |

**Streaming compatibility:** `stream=true` + `mode="blocking"` is rejected with HTTP 400 (it would defeat the purpose of streaming). `stream=true` + `mode="background"` works — the eval task is scheduled after the SSE stream is delivered.

**Per-rubric isolation:** one rubric raising (correctness without `expected`, judge unreachable, malformed JSON, …) doesn't cascade. Each verdict is independent and surfaces its own `parse_status` / `error`.

#### Example — blocking, two rubrics

```bash
curl -s -X POST http://127.0.0.1:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "llama3.2:1b",
    "messages": [{"role": "user", "content": "What is the capital of France?"}],
    "max_tokens": 12,
    "auto_eval": {
      "rubrics": ["helpfulness", "safety"],
      "judge_model": "llama3.2:3b",
      "mode": "blocking"
    }
  }'
```

Response (abbreviated):

```json
{
  "id": "chatcmpl-35273f...",
  "choices": [{"message": {"role": "assistant", "content": "The capital of France is Paris."}, "finish_reason": "stop"}],
  "usage": {"prompt_tokens": 42, "completion_tokens": 7, "total_tokens": 49},
  "evals": [
    {"rubric": "helpfulness", "verdict": {"score": 5.0, "parsed": {"score": 5, "justification": "..."}, "parse_status": "clean"}, "duration_ms": 1076.45, "error": null},
    {"rubric": "safety",      "verdict": {"score": 1.0, "parsed": {"safe": true, "concerns": []}, "parse_status": "clean"}, "duration_ms": 1422.62, "error": null}
  ]
}
```

#### Example — background, fire-and-forget

Wall-time measured against a real `llama3.2:1b` candidate + `llama3.2:3b` judge:

```
sending chat with auto_eval.mode=background…
got 49 chars; evals=null
wall: 97 ms          ← user-perceived latency unchanged
```

The eval spans land in OTel ~1.4s later, joined back to the chat by `eval.candidate.completion_id=chatcmpl-35273f...`. Prometa picks them up out of band; the agent path was never blocked.

### JSON repair

Judge models occasionally wrap their structured output in commentary or fences. The runner has three parse states:

* **`clean`** — judge returned valid JSON with the rubric's expected keys.
* **`repaired`** — judge wrapped JSON in prose (e.g. ```` ```json ... ``` ````); we extract the first balanced `{...}` block and re-parse.
* **`failed`** — judge refused or returned malformed output; we surface `score=0.0` so aggregations don't silently inherit garbage.

The `parse_status` is always on the response and on the span, so eval failures are first-class signals rather than hidden behind exceptions.

## Auth + tenant attribution

Per-key bearer-token auth — off by default for local dev, switch on with `AUTH_ENABLED=true`. Keys live in a JSON file (`.auth_keys.json`, gitignored):

```json
[
  {"key": "sk-dev-local-1", "tenant": "dev"},
  {"key": "sk-eval-1",      "tenant": "evals"},
  {"key": "sk-agent-prod-1", "tenant": "agent-runtime"}
]
```

Behaviour:

- `Authorization: Bearer <key>` → resolves to `Identity(tenant, key_id)` and caches on `request.state`.
- Missing or unknown key → `401 {"detail": "missing bearer token"}` / `401 {"detail": "invalid api key"}`.
- `/v1/health` is left open so liveness probes work without keys.
- `/v1/models` and `/v1/chat/completions` require a valid key when auth is on.
- Every span (model.acquire, chat.generate, chat.stream) carries `prometa.tenant=<name>` and `prometa.key_id=<redacted>` — Prometa can route signals per tenant out of the box.

The keys file is the seam where Prometa's control plane drops a generated set; rotation is a `load_keys()` call away.

## KV-cache prefix reuse

llama.cpp ships an explicit prompt-prefix cache (`LlamaRAMCache`) but it's off by default. We install it on every adapter at load time and size it via `PREFIX_CACHE_BYTES` (default 2 GiB). The cache is keyed by token-prefix; whenever a new request shares a prefix with one that's been processed before, prefill skips those tokens entirely.

This is the dominant latency lever for **RAG, multi-turn agents, and shared-system-prompt workflows** — exactly the shape Prometa-managed agents have. Measured on the M5 Max with `llama3.2:1b` and a 1124-token system prompt:

```
cold prefix     631 ms   (cache empty)
warm prefix      62 ms   (5 consecutive runs, same system prompt, different user query)
                 63 ms
                 65 ms
                 65 ms
                 65 ms
─────────────────────────
speedup  9.86x   warm vs cold
```

Per-adapter cache state is exposed four ways:

* **Span attributes** on every `chat.generate` / `chat.stream`:
  ```
  prefix_cache.enabled            = true
  prefix_cache.capacity_bytes     = 2147483648
  prefix_cache.size_bytes         = 223267034     # bytes used after this call
  prefix_cache.size_delta_bytes   = 37238489      # how much this call added
  prefix_cache.action             = hit | miss | unconsulted | disabled
  prefix_cache.tokens_reused      = 344           # state.n_tokens on the cache hit
  prefix_cache.tokens_total       = 371           # response.usage.prompt_tokens
  ```
* **Prometheus gauges** on `/v1/metrics`:
  ```
  inference_engine_prefix_cache_capacity_bytes{model="llama3.2:1b",backend="llama_cpp"} 2147483648
  inference_engine_prefix_cache_size_bytes{model="llama3.2:1b",backend="llama_cpp"}      223267034
  ```
* **Structured log line** at load: `prefix_cache_enabled capacity_bytes=2147483648 model=llama3.2:1b`.
* **Aggregate counters** on the cache itself: `hit_count` / `miss_count` (pulled by future `/v1/metrics` extensions).

The token-level introspection comes from a thin `_TrackedLlamaRAMCache` subclass of llama-cpp-python's upstream `LlamaRAMCache`; we read the matched `LlamaState.n_tokens` directly off each `__getitem__`. `LlamaRAMCache` already does best-prefix LRU lookup internally, so we don't reimplement slot management — we just instrument it.

Set `PREFIX_CACHE_BYTES=0` to disable (useful for benchmarking cold-prefill, or for memory-constrained setups).

#### Honest limitation: `unconsulted` action

llama.cpp first checks its own `_input_ids` buffer for prefix continuation before consulting the cache. When a request extends the *previous* request's exact prompt (within-conversation continuation), the cache is never asked — `action="unconsulted"`. That call still benefits from prefix reuse, just via a different path llama.cpp doesn't expose. The cache hits we *do* report cover the dominant cross-conversation / alternating-prefix case, which is what multi-slot was designed to solve.

### MLX prefix cache

The MLX side runs a **multi-slot, token-indexed** cache. Each loaded model holds up to `MLX_PREFIX_CACHE_MAX_SLOTS` independent KV-cache states. On every call we tokenize the prompt and scan all slots for the one with the longest matching token-prefix, then:

| state      | meaning                                                |
|------------|--------------------------------------------------------|
| `miss`     | no useful slot **or** capacity available + partial overlap → `make_prompt_cache()` for a fresh slot, evict LRU only if at capacity |
| `full`     | prompt is identical to a slot's tokens → reuse verbatim |
| `trimmed`  | at capacity + partial overlap → `trim_prompt_cache(slot, n)` to the divergence point  |
| `disabled` | `MLX_PREFIX_CACHE_ENABLED=false` → mlx-lm uses a per-call discarded cache |

#### Resolution policy: preserve under capacity, trim only when full

The crucial nuance: when **capacity allows**, partial overlap allocates a new slot rather than trimming the candidate. That preserves the original slot for future hits — exactly the right call for alternating-agent / multi-tenant workloads. Only when at capacity does trim become the eviction strategy.

#### Stable prefix — same agent, repeated calls

Measured on the M5 Max with `Llama-3.2-1B-Instruct-4bit:mlx` and a 1095-token system prompt:

```
cold call         1522 ms   action=miss      tokens_reused=0    tokens_total=1095
warm call 1        139 ms   action=trimmed   tokens_reused=1083 tokens_total=1094
warm call 2        143 ms
warm call 3        145 ms
warm call 4        151 ms
warm call 5        149 ms
─────────────────────────────────────────────────────────
speedup  10.46×   warm vs cold
```

#### Alternating-agent — two distinct system prompts in rotation

The case that motivated multi-slot. Two independent agents (Alpha + Beta), each with its own ~600-token system prompt, called in alternation. Comparing `MLX_PREFIX_CACHE_MAX_SLOTS=1` (the thrash baseline) vs `=4` (preserve):

| `max_slots` | tokens reused per alternating call | observed action |
|---|---|---|
| `1` | **26** (chat-template tokens only) | every call trims back to ~26 |
| `4` | **~560** (full system prompt) | each agent owns its own slot |

That's a **~22× lift in cache hit rate** for the alternating workload. Wall latency is identical on M5 Max because MLX prefill is fast in absolute terms (~100 ms for 600 tokens); on slower hardware or with longer prefixes the gap widens proportionally.

#### Capacity & memory

Each slot holds an independent KV state — sized roughly proportional to its cached tokens. On a 128 GB unified-memory M5 Max with a 1B-class model, `max_slots=4` with ~1k-token prefixes is comfortable. For larger models or longer prefixes, drop `max_slots` (or `MLX_PREFIX_CACHE_ENABLED=false`).

`max_slots=1` reproduces the original single-slot behaviour exactly — useful for benchmarking or memory-constrained setups.

#### Trim is destructive

When a slot is partially reused at capacity, the divergent suffix is dropped. We could clone the cache to preserve the original, but cloning MLX caches means deep-copying GPU arrays — not worth the cost.

## Concurrency design

Two locks, sized to the contention they actually protect against:

* **`ModelManager._meta_lock`** — held only for microsecond-scale `OrderedDict` mutations (cache lookup + LRU `move_to_end`, eviction loop). A cache hit on model B never blocks behind a cold load of model A.
* **`ModelManager._key_locks[key]`** — one per model id. Held for the duration of `adapter.load()`. Concurrent `get(X)` calls on the same model dedupe (the second sees a cache hit when it acquires the lock); concurrent `get(A)` and `get(B)` proceed in parallel.
* **`adapter._lock`** — per-adapter, held during `generate()` / `stream()`. Required because `llama_cpp.Llama` isn't thread-safe; calls to the same model serialise. Calls to *different* adapters run in parallel.

Memory-budget enforcement is approximate during racing cold loads: two adapters can briefly coexist before the second's eviction step runs. The next `get()` reconciles. This is acceptable for a soft budget.

### Stress numbers (8-way concurrency, M5 Max, llama3.2:1b warm)

```
make stress                    # same-model: per-adapter lock serializes
  20 requests / 8 concurrent   wall=1.06s   p50=274ms p95=646ms p99=673ms
                               148 out tokens, 139 tok/s aggregate, 0 errors

make stress-cross-backend      # MLX + llama.cpp simultaneously
  20 requests / 8 concurrent   wall=2.64s   gguf p50=134ms  mlx p50=753ms
                               222 out tokens, 0 errors, both backends served concurrently

stress_test.py --stream        # streaming + watchdog stress
  16 requests / 8 concurrent   wall=0.56s   p50=270ms p99=303ms
                               153 chunks @ 271 chunks/s aggregate, 0 task leaks
```

The same-model run shows the lock at work — first batch of 8 fires together and queues, latencies cluster in two tiers. The cross-backend run proves MLX and llama.cpp don't contend. The streaming run validates that 8 simultaneous `watch_disconnect` watchdogs come and go cleanly without leaking asyncio tasks.

## Streaming cancellation

When a streaming client disconnects, the engine stops generating instead of burning GPU on a response nobody's reading.

How it works:

1. `chat.py` enters `watch_disconnect(request)` which spawns a watchdog task polling `Request.is_disconnected()` every 100 ms.
2. On disconnect, the watchdog trips a thread-safe `Cancellation` flag.
3. Each adapter checks the flag in its producer loop; when set, the loop breaks out of the streaming iterator and the underlying inference halts at the next token boundary.
4. The span records `stream.cancelled=true` and `stream.cancel_reason=client_disconnect` so the abandoned work is visible in traces.

Limitations:

- **Streaming requests** honor cancellation (both adapters). Latency: bounded by the 100 ms watchdog poll + one extra token.
- **Blocking requests** (`stream=false`) on `llama_cpp` cannot be cancelled mid-generation — `Llama.create_chat_completion()` doesn't expose a stopping-criteria hook on the high-level entrypoint. Agents that need fast cancel should use `stream=true`.

## Adapter coverage

| backend     | format | strengths                                               | newer-arch coverage          |
|-------------|--------|---------------------------------------------------------|------------------------------|
| `llama_cpp` | GGUF   | universal hardware reach, GGUF lingua franca, fast warm-hits  | bounded by the wheel version |
| `mlx`       | MLX    | Apple Silicon native, Metal unified memory, often ahead on new architectures | depends on mlx-lm release    |

The Ollama store you already have includes architectures newer than the bundled `llama-cpp-python==0.3.21`: `mistral3` (ministral-3:*), `gemma4`, `qwen3.6`, `nemotron3`. These will fail to load via `llama_cpp` until the wheel ships support — but if `mlx-community` publishes an MLX conversion (most do), grab it with `make download-mlx-model MODEL=mlx-community/<repo>` and it'll serve through the `mlx` adapter on the same routes.

Standard `llama` family models work on either backend today.

## Roadmap (next phases from the guide)

1. **Phase 2 — service features.** ✅ Per-key bearer auth + tenant attribution · ✅ streaming request cancellation · ✅ prompt-template overrides via `/v1/completions` · rate limiting (delegated to API gateway).
2. **Phase 3 — engine behaviour.** ✅ Multi-model routing · ✅ per-key load dedup + parallel cold loads · ✅ llama.cpp prefix cache (9.86×) · ✅ MLX multi-slot LRU prefix cache (~22× hit-rate lift) · ✅ token-precise cache observability on **both** backends · ✅ dynamic batching for `/v1/embeddings` (coalescer + capability fallback) · ✅ continuous chat batching via vLLM-as-subprocess (`VLLMAdapter` + `docker-compose.vllm.yml` overlay).
3. **Phase 4 — Prometa integration.** ✅ Real OTel exporter (OTLP/gRPC + Jaeger compose) · ✅ LLM-as-a-Judge eval harness · ✅ auto-judge attached to chat completions · ✅ server-side auto-eval policy (Prometa-authoritative) · ✅ tool-call audit logs (`gen_ai.tool_*` events with payload truncation).
4. **Adapter coverage.** ✅ MLX-LM (Apple Silicon native) · vLLM / SGLang for GPU-server workloads · TensorRT-LLM for NVIDIA optimization.

## Constraints (as instructed)

- This service **only reads** from `auto-ml/ollama-models/`. It never writes there.
- All edits stay inside `llm_inference_engine_v1/`.
