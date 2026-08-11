# Local LLM Inference Engine (v1)

An **OpenAI-compatible model plane**. It began as a laptop-class service wrapping an existing **Ollama-format GGUF model store** without needing the Ollama daemon, and it still runs that way — but the same routes now also front vLLM, OpenRouter, and Ollama-HTTP upstreams, and the whole thing ships as a signed, fail-closed tenant deployment on OpenShift.

The service is **backend-agnostic**: a thin adapter interface (`InferenceAdapter`) sits between the API layer and the actual model runtime, so llama.cpp, MLX-LM, vLLM, OpenRouter, and a remote Ollama server all serve the same routes — and a new runtime slots in without touching them.

Scope note: this is the **model plane**, not a full API gateway. It governs and serves models — signed routing, tenant admission, evals, and OTel evidence — while ingress termination, cross-tenant quota, and the control plane itself stay outside it. Read the gaps below against that boundary.

## Why this stack

Synthesised from the multi-LLM guide (GPT, Claude, Gemini, Grok all converge on the same shape):

- **Engines — `llama.cpp` (GGUF) + `mlx-lm` (Apple Silicon native).** Both implement the same `InferenceAdapter` ABC. The `ModelManager` dispatches per-descriptor so each loaded model uses the runtime that matches its on-disk format. llama.cpp gives universal hardware reach + access to the existing 135 GB Ollama GGUF store; MLX is native to the M5 Max's unified Metal stack and tracks newer architectures faster than mainline llama.cpp.
- **Service layer — FastAPI.** OpenAI-compatible `/v1/chat/completions`, streaming via SSE, structured JSON mode, model registry, health checks.
- **Registry — composite, direct manifest reader.** `OllamaRegistry` parses the Docker-distribution layout (`manifests/<registry>/<ns>/<model>/<tag>` + `blobs/sha256-*`) so we use the existing store with zero copying or daemon overhead. `MLXRegistry` scans for HF-style safetensors directories. `CompositeRegistry` merges them, with `prefer_mlx_over_gguf` controlling collision resolution.
- **Remote runtimes — same routes, no client change.** `vLLM` (continuous batching on a CUDA upstream), `ollama_http` (architectures the bundled llama.cpp wheel can't load yet), and `openrouter` (large open-weight models beyond local memory) are all live adapters behind the identical API. `SGLang` / TensorRT-LLM would slot in the same way.
- **Governance — signed desired state, verified locally.** An Ed25519 routing-policy envelope from the control plane pins routes, token/cost ceilings, and RPM per tenant. Verification is entirely local: the engine never calls the control plane on the request path.

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

### Canonical release artifacts

Release tags and explicit `publish-release-artifacts` workflow dispatches
produce two digest-addressable images:

| Image | Deployment profile |
|---|---|
| `ghcr.io/caglarsubas/llm_inference_engine/inference-engine` | Minimal Debian runtime |
| `ghcr.io/caglarsubas/llm_inference_engine/inference-engine-ubi` | Red Hat UBI9 runtime for OpenShift estates |

Both published variants include the `otel` extra, run non-root, and are tested
under an arbitrary high UID in GID 0 with a read-only root filesystem. The
release workflow emits OCI provenance, SPDX and CycloneDX SBOMs, a keyless
cosign signature, and a CycloneDX attestation for the exact image digest.
Production deployments should use the digest printed in the workflow summary,
not a mutable tag:

```bash
docker pull \
  ghcr.io/caglarsubas/llm_inference_engine/inference-engine@sha256:<digest>
```

Version tags also publish the independently versioned standalone Helm chart at
`oci://ghcr.io/caglarsubas/llm_inference_engine/charts/orchestra-inference-engine`.
The chart digest is keylessly signed and carries a CycloneDX attestation; its
packaged tarball, SPDX/CycloneDX SBOMs, and GitHub build-provenance attestation
are retained by the same workflow. Chart version and engine application version
are intentionally separate and are both printed in the release summary.

Image verification, registry relocation, air-gap transfer, supported
architecture, and the precise UBI/FIPS boundary are documented in
[`docs/CONTAINER_IMAGES.md`](docs/CONTAINER_IMAGES.md).

### Standalone tenant model plane

[`deploy/helm/inference-engine`](deploy/helm/inference-engine/README.md) deploys
the engine as its own tenant-plane StatefulSet release. Its fail-closed
OpenShift overlay requires the pinned `orchestra-ocp-4.20-amd64-v1` profile,
immutable UBI digest, mounted signed artifacts and purpose-scoped credentials,
external Sentinel state, persistent per-replica LKG storage, explicit network
peers, and asynchronous evidence export. It creates no credentials, public
Route, runtime host, or Orchestra control-plane workload.

This makes the deployment topology honest: the tenant runtime calls the model
plane directly while Orchestra receives observations out of band. A successful
chart render is contract evidence, not OpenShift production certification.

The chart carries two mutually incompatible profiles:

| profile | values file | what it is |
|---|---|---|
| Production | `values.openshift-production.yaml` | Fail-closed pinned `orchestra-ocp-4.20-amd64-v1` contract — multi-replica, PDB, topology spread, ServiceMonitor, external Sentinel state |
| SNO engineering trial | `values.openshift-sno-trial.yaml` | Source-only single-node `orchestra-ocp-sno-trial-amd64-v1` contract — one replica, deliberately non-HA, PDB / topology spread / backup claims / ServiceMonitor all disabled |

The SNO profile names only a public image repository and the objects an
operator would have to create; it carries no credentials or model artifacts,
and its image digest is deliberately empty until a release containing both this
profile and the secure OTLP/HTTP transport is published. Rendering it neither
installs anything nor supports a production or certification claim. Both
profiles have an adversarial render proof under `ci/`:

```bash
./deploy/helm/inference-engine/ci/render-production-profile.sh /tmp/production.yaml
./deploy/helm/inference-engine/ci/render-sno-trial-profile.sh  /tmp/sno-trial.yaml
```

Each script first proves the values file fails unchanged, then injects a
synthetic digest to render. Chart version and engine `appVersion` are
independent by design; `scripts/verify_release_contract.sh` enforces that
`appVersion` matches the Python runtime version and the release tag.

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

**MLX adapter doesn't run in containers.** Apple Silicon Docker Desktop runs a Linux VM with no Metal passthrough; mlx-lm needs Metal. The composite registry handles this cleanly — the MLX directory mount is empty inside the container, the registry returns zero MLX models, and llama.cpp serves everything. The container's llama.cpp build is CPU-only and disables host-native instruction tuning by default so published images remain portable; switch to CUDA on a GPU host with `--build-arg CMAKE_ARGS="-DGGML_CUDA=on"`.

**No metric-driven auto-scaling on plain Compose.** `docker compose up --scale` is manual. The `deploy.replicas` block in the compose file is read by **Docker Swarm** (`docker stack deploy`) for declarative scaling; for true HPA-style autoscaling, deploy on Kubernetes. Both paths work without code changes — the engine itself is stateless modulo the instance-local caches called out above.

**Cross-instance state is not shared.** Process-global stores (tool-timing, embed coalescer, prefix cache) are per-replica. That's fine for cache-warming workloads (each replica warms separately) but it means signals like `tool.execution_ms` only fire when both turns of a tool exchange land on the same replica. With sticky sessions via HAProxy/Traefik, this works; with plain round-robin, it's best-effort. For real distributed state, plug Redis behind the audit module — out of scope here.

### Tenant-aware scheduling inside each replica

Horizontal replicas add capacity; they do not, by themselves, protect one tenant from another tenant's burst once requests land on the same replica/model. The engine adds a process-local tenant scheduler in front of backend calls for `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, and `/v1/rerank`.

Dispatch policy:

- Requests enter per-tenant queues, not one process-wide FIFO.
- Local serialized backends (`llama_cpp`, `mlx`) default to one active slot per model resource, so requests do not pile up inside `adapter._lock` ahead of the scheduler.
- vLLM-backed resources default to eight active slots so the upstream continuous batcher can still work.
- Each tenant gets a soft active-slot reservation (`SCHEDULER_TENANT_RESERVED_IN_FLIGHT`). A tenant can borrow extra idle capacity while nobody else waits; when another tenant queues, new dispatches favor tenants below their reservation.
- Workload lanes are prioritized as: streaming chat, blocking chat, single completions, bulk completions/rerank, embeddings. Wait-time aging and tenant-fairness boost keep lower-priority work from starving.
- A full tenant queue returns `429 tenant_queue_full` with `Retry-After`; a queue wait timeout returns `503 tenant_queue_timeout`.

Every generation/embed/rerank span carries `scheduler.*` attributes: enabled flag, resource, workload, priority, estimated tokens, queue depth at submit, tenant queue depth at submit, and wait time.

### Horizontal scaling policy

For local Compose, scale manually:

```bash
make compose-up-scale REPLICAS=4
make compose-up-sticky REPLICAS=4
```

For production, keep the same engine contract but run under Swarm/Kubernetes and autoscale on saturation signals rather than CPU alone:

- Scale out when `inference_engine_scheduler_queued > 0` for a sustained window, when `inference_engine_scheduler_max_queue_wait_seconds` crosses the tenant SLO, or when the rate of `inference_engine_scheduler_rejected_total` / `inference_engine_scheduler_timed_out_total` is non-zero.
- Scale out when p95/p99 route latency rises while `inference_engine_scheduler_in_flight_by_resource` is pinned at the configured cap.
- Scale out vLLM lanes on token throughput plus queue wait; scale local GGUF/MLX lanes on queue wait plus active resource saturation.
- Scale in only when queues are empty, scheduler in-flight is low, and loaded model churn is stable for several windows.
- Use nginx round-robin for stateless/batch traffic; use HAProxy sticky routing for multi-turn agent traffic where prefix cache and tool timing matter.

### What's installed in the container

- llama-cpp-python compiled from source (portable CPU baseline by default, CUDA build with `CMAKE_ARGS="-DGGML_CUDA=on"`)
- The OTel extra in Compose and canonical release builds, so `OTEL_ENABLED=true` works against an external collector
- A non-root default user plus GID-0 permissions for OpenShift-assigned arbitrary UIDs
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

# (optional) materialize local FraudGuard VLM snapshots before serving/probing
make download-vlm-models CORE_ONLY=1   # writes to ~/.cache/inference_engine/hf-vlm

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
| GET    | `/v1/ready`                   | Readiness probe; returns 503 + `Retry-After` while startup probes run      |
| GET    | `/v1/metrics`, `/metrics`     | Prometheus scrape — engine internals plus OTel GenAI semconv instruments   |
| GET    | `/v1/models`                  | All models discoverable in the unified registry                            |
| GET    | `/v1/models.data`             | Machine-readable catalog — provider, modality, image / strict-JSON capability, plus a typed `unavailable[]` with the reason each id is not servable |
| GET    | `/v1/models/{model:tag}`      | Single model details (size, blob path, backend, `context_length`)          |
| POST   | `/v1/chat/completions`        | Blocking + SSE streaming (`stream: true`)                                  |
| POST   | `/v1/completions`             | Legacy raw-prompt completions — bypasses chat templating                   |
| POST   | `/v1/embeddings`              | OpenAI-compatible embeddings (llama.cpp); MLX returns 501                  |
| POST   | `/v1/rerank`                  | Cohere/Jina-shaped relevance ranking via embedding cosine similarity        |
| POST   | `/tokenize`                   | vLLM/TGI-shaped token count for a prompt or templated messages; 501 on HTTP-proxy backends |
| POST   | `/detokenize`                 | Token ids → text, same backend support as `/tokenize`                      |
| GET    | `/v1/evals/rubrics`           | List built-in + registered rubrics                                         |
| GET    | `/v1/evals/policy`            | Active server-side auto-eval policy entries (Prometa-driven)               |
| POST   | `/v1/admin/policies:reload`   | Hot-reload `AUTO_EVAL_POLICIES_FILE`; atomic swap on success, rejects malformed |
| GET    | `/v1/admin/auth-keys`         | Secret-free loaded key IDs, validity windows, digest, and active count     |
| POST   | `/v1/admin/auth-keys:reload`  | Atomically activate a mounted key set while retaining the calling key      |
| GET    | `/v1/admin/model-routing-policy` | Payload-free activated signed-policy identity and LKG status             |
| POST   | `/v1/admin/model-routing-policy:reload` | Verify candidate/LKG and atomically activate on success             |
| POST   | `/v1/admin/model-routing-pricing:reload` | Validate mounted pricing against the active policy and atomically replace pricing |
| GET    | `/v1/admin/model-plane-observer` | Payload-free asynchronous reporter delivery status                    |
| POST   | `/v1/evals/run`               | LLM-as-a-Judge: candidate + rubric → structured verdict                    |
| POST   | `/v1/chat/completions`        | (extension) `auto_eval: {rubrics, mode}` runs evals inline or in background |

During native startup, the HTTP listener binds before heavyweight model probes
finish. `/v1/health` stays open and reports `status: "starting"`;
`/v1/ready` returns HTTP 503 with `Retry-After`; normal `/v1/*` model,
inference, eval, and admin routes return a typed 503 detail with
`type: "engine_starting"` until the probe pass completes.

### OpenAI request/response compatibility

Request fields honoured on `/v1/chat/completions` beyond the basics:

| Field | Behaviour |
|-------|-----------|
| `max_completion_tokens` | OpenAI's replacement for `max_tokens`. Both accepted; the newer name wins when both are sent. |
| `stream_options: {include_usage: true}` | Emits a final SSE frame with empty `choices` and a populated `usage`, immediately before `[DONE]`. **Without this a streaming caller gets no token counts at all.** |
| `response_format: {type: "json_schema", …}` | Structured Outputs — see below. |
| `logprobs` / `top_logprobs` | Returned as `choices[].logprobs.content` on llama.cpp and vLLM. |
| `frequency_penalty`, `presence_penalty`, `repetition_penalty` | Forwarded only when set, so each backend keeps its own default otherwise. `repetition_penalty` maps to llama.cpp's `repeat_penalty`. |
| `logit_bias` | OpenAI keys by string token id; converted to llama.cpp's int keys. |
| `parallel_tool_calls`, `user`, `n` | Accepted; `n` is restricted to `1` (only one choice is served). |

Responses carry `usage.prompt_tokens_details.cached_tokens` when the request
hit a prefix cache — the engine already measured this token-precisely on
llama.cpp and MLX, and now reports it in the standard field so ordinary cost
dashboards see the saving. vLLM's value is passed through from its own
accounting. The count is clamped to `prompt_tokens`: llama.cpp keys cache
entries on the full context that produced them (prompt *plus* generated
tokens), so the raw figure can legitimately exceed the current prompt.

#### Error envelope

Failures return **both** shapes:

```json
{
  "detail": {"message": "…", "type": "context_length_exceeded", "code": "…", "param": "messages", "context_window": 8192},
  "error":  {"message": "…", "type": "context_length_exceeded", "code": "…", "param": "messages", "context_window": 8192}
}
```

`error` is what every OpenAI SDK reads (`APIStatusError.body`, `BadRequestError.code`,
`RateLimitError`); `detail` is retained unchanged for existing Prometa consumers
and the admin tooling. Typed extras like `context_window` and
`retry_after_seconds` survive in both. 429s additionally carry
`x-ratelimit-limit-requests`, `x-ratelimit-remaining-requests`, and
`x-ratelimit-reset-requests` alongside `Retry-After`.

Every response the application can observe carries a server-owned
`x-request-id` (`engine_request_id`). Inbound `x-request-id` is accepted for
generic-client compatibility but never echoed or used as a server identity.
The orchestra-python-sdk instead sends three independent, optional correlation
headers: `x-orchestra-runtime-request-id`,
`x-orchestra-model-invocation-id`, and `x-orchestra-model-attempt-id`. Each must
be 1-256 visible-ASCII characters, excluding the case-insensitive flattened-null
sentinels `null`, `none`, `nil`, and `undefined`, and is rejected, never
truncated or normalized, when invalid. Those spellings are reserved because an
OTLP attribute carrier cannot distinguish them from a JSON null after collector
flattening. The headers form an ordered prefix: runtime may appear alone;
invocation requires runtime; attempt requires both runtime and invocation.

These caller-owned values are copied verbatim into billing stdout, so producers
must use opaque, non-PII, non-secret identifiers: never embed an email address,
user name, credential, token, prompt fragment, or business payload. For this
identity contract the only tracing propagation metadata in scope is W3C
`traceparent` / `tracestate`; do not put correlation data in `baggage`, and
`baggage` is never copied into the usage ledger.

With `USAGE_LEDGER_ENABLED=true`, priced routes return
`x-orchestra-usage-record-id` only when that request has an actual billable
ledger record after crossing the bind seam. Ledger-disabled responses and
pre-bind 401, 422, early 400, or unhandled failures do not advertise a record
that will never be emitted. The exact v2 field and header contract is
`contracts/prometa-model-usage-v2.schema.json`.

After the platform dual-reader is deployed, roll out the engine before the SDK
**only after an identity preflight passes**. Inventory every configured or
caller-supplied legacy runtime-request-id source and verify representative live
requests use 1-256 visible-ASCII characters, do not use the reserved sentinel
spellings above, and do not depend on trimming or Unicode normalization. Migrate
any nonconforming producer before engine activation; do not silently rewrite an
identity, because that can create a correlation collision. A contract-conformant
old SDK + new engine is compatible because its runtime-only header is a valid
prefix. This statement does not cover older/custom kernels that admitted spaces,
Unicode, or the reserved sentinel spellings: the new engine deliberately rejects
those values with 400 before priced execution.

New SDK + old engine is a safe partial-rollout or rollback state: the old engine
echoes the runtime id as `x-request-id`, but the new SDK accepts an engine id only
when it matches the canonical `req_` plus 32-lowercase-hex pattern, so that
noncanonical echo degrades to an absent `engine_request_id` instead of becoming
a false identity. The old engine has no usage-record response header, which
likewise remains absent.

#### Structured Outputs

`response_format: {"type": "json_schema", "json_schema": {"name": …, "strict": true, "schema": {…}}}`
is **grammar-enforced** wherever the backend can do it, not merely requested in
the prompt:

| Backend | Enforcement |
|---------|-------------|
| `llama_cpp` | Schema compiled to GBNF; the sampler cannot emit a non-conforming token |
| `vllm` | Forwarded natively to guided decoding (xgrammar) |
| `ollama_http`, `openrouter` | Forwarded; the upstream enforces |
| `mlx` | No constrained-decoding hook — the route validates the document and retries once with the validation error fed back, then returns a typed 502 `structured_output_invalid` |

The MLX fallback validator implements the subset of JSON Schema that OpenAI's
strict mode itself permits (`type`, `properties`, `required`,
`additionalProperties`, `items`, `enum`, `anyOf`, `$ref` into `$defs`).
Assertion keywords like `minimum` and `pattern` are **not** checked and pass
silently — it never rejects a valid document, which is the property that
matters for not breaking working requests.

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

## Sharing a public endpoint (ngrok / Cloudflare Tunnel)

The engine binds to loopback by default. To hand a consumer — or the Prometa
control plane — a public HTTPS "Engine URL" without a public IP or inbound
firewall rule, front it with a tunnel:

```bash
make share              # ngrok against $PORT from .env  (default 8080)
make share-cf           # Cloudflare Tunnel (no account needed)
make share PORT=8090    # tunnel a different port, e.g. the docker LB
```

`make share` wraps [`scripts/share_endpoint.sh`](scripts/share_endpoint.sh),
which health-checks the engine, warns loudly if `AUTH_ENABLED` isn't on (a
public URL with auth off is an open inference engine), prints the assigned URL,
and emits copy-paste `curl` + Prometa wiring for it. Direct use:

```bash
scripts/share_endpoint.sh --provider ngrok --domain my-name.ngrok-free.dev   # stable URL
scripts/share_endpoint.sh --provider cloudflared --port 8090
```

For a stable URL, claim the free static domain on the ngrok dashboard and pass
`make share NGROK_DOMAIN=my-name.ngrok-free.dev`. For a **truly always-on**
endpoint that survives crashes and reboots, install the tunnel as a `launchd`
agent (same pattern as `make native-install`):

```bash
make share-install NGROK_DOMAIN=my-name.ngrok-free.dev   # auto-start + respawn
make share-status        # state, PID, live URL
make share-uninstall     # remove the agent
```

Paste the printed URL into **Prometa → Settings → Self-hosted
(llm_inference_engine) → ENGINE URL**, and a bearer key from `.auth_keys.json`
into ENGINE TOKEN.

**The full, shareable usage guide for the URL** — base URL, auth, listing
models, and `POST` examples for every route across all backends/models, plus
SDK snippets and troubleshooting — lives in
[`docs/PUBLIC_ENDPOINT.md`](docs/PUBLIC_ENDPOINT.md). Send that to whoever
consumes the endpoint.

## Layout

```
src/inference_engine/
├── main.py              # FastAPI app + lifespan + module-level OTel wiring + load_keys()
├── server.py            # fail-closed HTTP/TLS launcher — strict env validation, mTLS, uvicorn config
├── config.py            # pydantic-settings (.env-driven)
├── manager.py           # ModelManager — LRU multi-model hot-keep, per-format dispatch
├── scheduler.py         # TenantScheduler — per-tenant queues, fair dispatch, admission limits
├── observability.py     # span() bridges structlog + OTel; Span.bind() mutates both
├── otel.py              # OTel SDK setup, NoOp span shim, FastAPI auto-instrumentation
├── genai_metrics.py     # GenAI semconv instruments — TTFT, TPOT, operation duration, token usage
├── auth.py              # bearer-token auth, Identity, key index, FastAPI dependency
├── model_routing.py     # signed-policy trust, verification, atomic LKG activation
├── model_routing_runtime.py # local route/limit/pricing/RPM enforcement
├── model_routing_status.py  # payload-free status view of the active signed policy
├── model_plane_observer.py  # asynchronous payload-free observed-state reporter
├── cancellation.py      # Cancellation flag + watch_disconnect() watchdog
├── schemas.py           # OpenAI-compatible request/response models
├── structured_outputs.py # JSON-Schema validator for backends with no grammar hook (MLX)
├── response_normalize.py # repairs raw model text into tool_calls / reasoning_content
├── evals/
│   ├── rubrics.py       # RubricSpec, built-in helpfulness/correctness/safety, RubricRegistry
│   ├── runner.py        # EvalRunner: candidate + rubric → judge → Verdict (clean/repaired/failed)
│   ├── policy.py        # PolicyMatch / PolicyEntry / PolicyRegistry — server-side auto-eval rules
│   └── schemas.py       # EvalRequest, EvalResponse, Verdict, PolicyList
├── api/
│   ├── state.py         # composite registry + ModelManager + adapter dispatch + EvalRunner
│   ├── health.py        # /v1/health + /v1/ready — readiness, version, workload surface
│   ├── metrics.py       # /v1/metrics + /metrics (Prometheus format)
│   ├── models.py        # /v1/models, /v1/models.data, /v1/models/{id} — gated by require_identity
│   ├── _models_snapshot.py # background-refreshed catalog so discovery never blocks on probes
│   ├── embeddings.py    # /v1/embeddings (OpenAI-compatible; llama.cpp only, MLX 501)
│   ├── rerank.py        # /v1/rerank — Cohere/Jina-shaped relevance via embedding cosine
│   ├── tokenize.py      # /tokenize + /detokenize (vLLM/TGI-shaped; 501 on HTTP-proxy backends)
│   ├── completions.py   # /v1/completions — legacy raw-prompt path, bypasses chat templating
│   ├── evals.py         # /v1/evals/rubrics + /v1/evals/policy + /v1/evals/run
│   ├── admin.py         # auth-key, auto-eval, and signed model-routing reload/status endpoints
│   ├── errors.py        # dual `error` + `detail` envelope, x-request-id, x-ratelimit-* headers
│   ├── _scheduling.py   # shared API helpers for scheduler admission/span attrs
│   ├── _auto_eval.py    # blocking + background batch helpers for chat-attached eval
│   ├── _batcher.py      # EmbedCoalescer — dynamic batching for /v1/embeddings
│   ├── _tool_audit.py   # gen_ai.tool_call / gen_ai.tool_result event emission with truncation
│   ├── _model_routing.py # per-request signed-policy enforcement wired into generation routes
│   ├── _fallback.py     # shared OpenRouter fallback helpers for generation endpoints
│   └── chat.py          # /v1/chat/completions (+ SSE, gen_ai.* spans, watchdog, tenant, auto_eval, tool audit)
├── adapters/
│   ├── base.py          # InferenceAdapter ABC (stream/generate accept cancel=)
│   ├── llama_cpp.py     # llama-cpp-python implementation (GGUF) — streaming cancel
│   ├── mlx_lm.py        # mlx-lm implementation (Apple Silicon native) — streaming cancel
│   ├── ollama_http.py   # Ollama server client for GGUF architectures the wheel can't load
│   ├── openrouter_adapter.py # OpenRouter OpenAI-compatible client
│   └── vllm_adapter.py  # vLLM HTTP client (continuous batching on a CUDA upstream)
└── registry/
    ├── ollama.py        # parses Ollama manifests → ModelDescriptor
    ├── ollama_http.py   # discovers models served by an Ollama HTTP server
    ├── mlx.py           # scans MLX model directories
    ├── openrouter.py    # parses .openrouter_models.json + policy gate
    ├── openrouter_probe.py # one shared catalog fetch + TTL + last-known-good window
    ├── vllm.py          # parses .vllm_models.json (HTTP endpoints, no local files)
    ├── vllm_probe.py    # upstream /v1/models reachability probe with TTL cache
    ├── probe.py         # GGUF load-probe — can llama.cpp actually open this manifest?
    └── composite.py     # merges multiple registry sources
Dockerfile                      # multi-stage build, llama-cpp-python from source, non-root runtime
Dockerfile.ubi                  # Red Hat UBI9 runtime variant for OpenShift estates
docker-compose.yml              # N engine replicas + nginx LB + healthchecks + volume mounts
docker-compose.haproxy.yml      # overlay: HAProxy LB with header-based tenant stickiness
docker-compose.vllm.yml         # overlay: single-GPU vLLM sidecar (count: 1)
docker-compose.vllm-multigpu.yml# overlay: multi-GPU vLLM (two services pinned via device_ids)
docker-compose.otel.yml         # overlay: Jaeger sidecar for OTel trace UI
docker-compose.observability.yml# overlay: Grafana + Prometheus + Jaeger + OTel Collector
docker-compose.native.yml       # native split: engine + ollama under launchd, obs stack in containers
docker/nginx.conf               # dynamic-resolution upstream + SSE-friendly buffering
docker/haproxy.cfg              # balance hdr(Authorization) + dynamic DNS + active healthcheck
docker/observability/           # collector + Prometheus config, provisioned Grafana dashboard
docker/config/                  # mount target for auth_keys.json + auto_eval_policies.json
deploy/helm/inference-engine/   # standalone tenant model-plane chart
├── templates/                      # StatefulSet, Service, NetworkPolicy, PDB, ServiceMonitor, RBAC
├── values.yaml                     # permissive defaults
├── values.openshift-production.yaml  # fail-closed pinned production profile
├── values.openshift-sno-trial.yaml   # source-only single-node engineering-trial profile
└── ci/                             # render checks for both profiles
scripts/
├── list_models.py            # CLI to enumerate the unified registry
├── download_mlx_model.py     # snapshot_download from mlx-community/*
├── download_vlm_models.py    # materialize local Hugging Face VLM snapshots
├── promote_vllm_model.py     # gate a demanded vLLM model into the live manifest
├── serve_sida_openai.py      # OpenAI-compatible worker for the SIDA reference implementation
├── serve_molmo_mlx_openai.py # OpenAI-compatible MLX worker for Molmo
├── serve_mlx_vlm_openai.py   # OpenAI-compatible MLX-VLM worker (InternVL, Ovis, …)
├── share_endpoint.sh         # expose the engine on a public HTTPS URL (ngrok/cloudflared)
├── share-service.sh          # run the ngrok tunnel as an always-on launchd agent
├── native-service.sh         # run the engine itself as an always-on launchd agent
├── relocate_images.sh        # mirror canonical images into a private / air-gapped registry
├── verify_release_contract.sh# check a release digest against the published contract
├── container_smoke.sh        # container-level smoke against a built image
├── launchd/                  # launchd plist templates (engine, ollama, ngrok-tunnel)
├── smoke_test.py             # blocking + streaming end-to-end check
├── vlm_request_matrix.py     # image-model request matrix across configured backends
├── vlm_strict_json_smoke.py  # strict image+JSON exposure gate
└── stress_test.py            # concurrent-traffic harness; p50/p95/p99 + throughput, and
                              #   --baseline-url/--baseline-model for measured proxy overhead
docs/
├── PUBLIC_ENDPOINT.md        # shareable usage guide for the public Engine URL
├── CONTAINER_IMAGES.md       # image verification, relocation, air-gap, UBI/FIPS boundary
└── MODEL_DEMAND_SHORTLIST.md # demanded-model backlog behind /v1/models.data unavailable[]
tests/                          # 67 modules, all run by `make test`
├── [registry + catalog]        # test_registry, test_mlx_registry, test_composite_registry,
│                             #   test_openrouter, test_vllm_probe, test_models_snapshot,
│                             #   test_manager, test_state_dispatch, test_promote_vllm_model
├── [adapters + backends]       # test_vllm_adapter, test_ollama_http_adapter, test_adapter_usage_paths,
│                             #   test_llama_cpp_context_window, test_llama_cpp_tool_arguments,
│                             #   test_sida_openai_worker, test_molmo_mlx_openai_worker,
│                             #   test_mlx_vlm_openai_worker, test_vlm_request_matrix, test_vlm_smoke_script
├── [OpenAI wire contract]      # test_openai_compat, test_http_contract, test_response_normalize,
│                             #   test_structured_outputs, test_structured_output_enforcement,
│                             #   test_completions, test_chat_multimodal, test_embeddings, test_rerank
├── [streaming + concurrency]   # test_cancellation, test_chat_streaming, test_generation_timeout,
│                             #   test_concurrency, test_scheduler, test_dynamic_batching
├── [upstream resilience]       # test_upstream_resilience (retry classification, backoff, breaker),
│                             #   test_upstream_retry_routing (retry-before-fallback, reservation, ledger)
├── [prefix caches]             # test_prefix_cache, test_mlx_prefix_cache
├── [evals]                     # test_evals, test_auto_eval, test_auto_eval_policy, test_pairwise,
│                             #   test_per_rubric_judges, test_admin_policies
├── [observability]             # test_observability, test_otel_exporter, test_genai_metrics, test_metrics,
│                             #   test_intent_tracing, test_tool_audit, test_streaming_tool_audit, test_tool_timing
├── [auth + governance]         # test_auth, test_admin_auth_keys, test_model_routing_policy,
│                             #   test_admin_model_routing_policy, test_model_routing_runtime,
│                             #   test_model_routing_api, test_model_routing_rate_limit, test_model_plane_observer
└── [server lifecycle]          # test_server (TLS/mTLS launcher), test_startup_readiness
```

`make test` (or `uv run pytest`) runs from a fresh clone with no environment
set up: `tests/conftest.py` creates an empty scratch model store before the
first `inference_engine` import and points `OLLAMA_MODELS_DIR` and
`MLX_MODELS_DIR` at it, so the suite never reads a real model store. Export
either variable yourself to run the suite against a store of your own; an
explicit environment variable wins over the scratch store, a value in `.env`
does not.

## Configuration

Every knob is an environment variable read through `.env`. The table below is
the full surface; `.env.example` is a commented starting point that covers the
common ones, so a var missing from that file is still settable:

| var                      | default                                                                                  | meaning                                                  |
|--------------------------|------------------------------------------------------------------------------------------|----------------------------------------------------------|
| `HOST`                   | `127.0.0.1`                                                                              | Listener bind address                                    |
| `PORT`                   | `8080`                                                                                   | Listener port                                            |
| `LOG_LEVEL`              | `INFO`                                                                                   | Uvicorn + structlog level                                |
| `OLLAMA_MODELS_DIR`      | `~/.cache/inference_engine/ollama`                                                       | Root with `manifests/` and `blobs/`; startup fails loudly if it is missing |
| `MLX_MODELS_DIR`         | `~/.cache/inference_engine/mlx`                                                          | Where `download_mlx_model.py` snapshots HF repos         |
| `MLX_MODELS_HOST_DIR`    | `~/.cache/inference_engine/mlx`                                                          | Host MLX cache mounted into Docker compose               |
| `VLLM_MODELS_FILE`       | `.vllm_models.json`                                                                      | Config file for OpenAI-compatible vLLM/DMR upstreams     |
| `VLLM_DEMANDED_MODELS_FILE` | `.vllm_models.demanded.example.json`                                                  | Catalog-only vLLM demand manifest reported as `unavailable` until configured |
| `HF_VLM_MODELS_DIR`      | `~/.cache/inference_engine/hf-vlm`                                                       | Local Hugging Face VLM snapshot cache used to report `downloaded_but_not_served` |
| `HF_VLM_MODELS_HOST_DIR` | `~/.cache/inference_engine/hf-vlm`                                                       | Host VLM snapshot cache mounted into Docker compose      |
| `VLLM_EXTRA_ARGS`        | `""`                                                                                     | Extra args for the vLLM sidecar, such as `--served-model-name` or `--trust-remote-code` |
| `OPENROUTER_MODELS_FILE` | `.openrouter_models.json`                                                               | Config file for large open-weight OpenRouter models      |
| `OPENROUTER_API_KEY`     | `""`                                                                                    | OpenRouter bearer token; keep only in ignored runtime env |
| `OPENROUTER_ENDPOINT`    | `https://openrouter.ai/api`                                                             | OpenRouter OpenAI-compatible base before `/v1`           |
| `OPENROUTER_MIN_PARAMETER_COUNT_B` | `50`                                                                           | OpenRouter gate: configured model must be larger unless it is an image-capable `benchmark_only` candidate |
| `OPENROUTER_FALLBACK_ENABLED` | `true`                                                                             | Retry eligible local generation failures on OpenRouter   |
| `OPENROUTER_FALLBACK_MODEL` | `""`                                                                                  | Explicit fallback model; empty derives `<name>:openrouter` |
| `OPENROUTER_FALLBACK_BACKENDS` | `llama_cpp,ollama_http,vllm`                                                     | Local backend names eligible for OpenRouter fallback     |
| `OPENROUTER_HTTP_REFERER` | `""`                                                                                    | Optional OpenRouter `HTTP-Referer` attribution header    |
| `OPENROUTER_APP_TITLE`   | `llm-inference-engine`                                                                   | Optional OpenRouter `X-Title` attribution header         |
| `OLLAMA_HTTP_ENDPOINT`   | `""`                                                                                     | Ollama server used as a fallback runtime for GGUF architectures the bundled llama.cpp wheel can't load; empty disables it and unsupported models are marked `unavailable` |
| `VLLM_UPSTREAM_PROBE_TIMEOUT_SECONDS` | `1.0`                                                                       | Reachability-probe timeout for vLLM / OpenAI-compatible upstreams |
| `VLLM_UPSTREAM_PROBE_TTL_SECONDS` | `30`                                                                            | How long a vLLM reachability result is cached; `0` probes on every call |
| `OPENROUTER_UPSTREAM_PROBE_TIMEOUT_SECONDS` | `8.0`                                                                 | Catalog-probe timeout, kept separate because OpenRouter is a public network hop with second-scale latency |
| `OPENROUTER_UPSTREAM_PROBE_TTL_SECONDS` | `60`                                                                      | Reuse window for one fetched OpenRouter catalog across every configured descriptor |
| `OPENROUTER_LAST_KNOWN_GOOD_SECONDS` | `900`                                                                        | Grace window that keeps serving the last good OpenRouter catalog through a transient failure; `0` prunes on the first failure |
| `PREFER_MLX_OVER_GGUF`   | `true`                                                                                   | On a name collision, MLX wins (faster on Apple Silicon)  |
| `DEFAULT_MODEL`          | `llama3.2:3b`                                                                            | Used by smoke test by default                            |
| `MODELS_SNAPSHOT_REFRESH_SECONDS` | `15`                                                                            | Rebuild cadence for the background `/v1/models` snapshot, so discovery never blocks on probe-loads; `0` disables the background refresh |
| `CHAT_COMPLETION_TIMEOUT_SECONDS` | `120`                                                                            | HTTP-backed chat deadline; `0` disables. Keep below public proxy caps. |
| `UPSTREAM_RETRY_ENABLED` | `true`                                                                                   | Retry an idempotent HTTP upstream call on the same deployment before falling back to a different model; `false` restores fallback-only resilience |
| `UPSTREAM_RETRY_MAX_ATTEMPTS` | `2`                                                                                 | Total attempts per upstream call, retries included; `1` disables retries. Attempts never outlive `CHAT_COMPLETION_TIMEOUT_SECONDS` |
| `UPSTREAM_RETRY_BASE_DELAY_SECONDS` | `0.1`                                                                         | First backoff delay before jitter; doubles per attempt   |
| `UPSTREAM_RETRY_MAX_DELAY_SECONDS` | `2.0`                                                                          | Ceiling for the computed backoff; an upstream `Retry-After` is honoured above it, or refused when it does not fit the remaining deadline |
| `UPSTREAM_BREAKER_ENABLED` | `true`                                                                                 | Open a deployment after consecutive transient failures so candidate selection skips it until a half-open probe succeeds |
| `UPSTREAM_BREAKER_FAILURE_THRESHOLD` | `3`                                                                          | Consecutive failed *requests* on one `(backend, endpoint, model_id)` before it opens. Counts logical calls, not upstream attempts — six failed round trips at the default attempt cap |
| `UPSTREAM_BREAKER_COOLDOWN_SECONDS` | `15`                                                                          | Cooldown applied the first time a deployment opens       |
| `UPSTREAM_BREAKER_MAX_COOLDOWN_SECONDS` | `120`                                                                     | Ceiling for the cooldown, which doubles each time the half-open probe fails again |
| `N_GPU_LAYERS`           | `-1`                                                                                     | llama.cpp: `-1` = offload all layers to Metal            |
| `N_CTX`                  | `32768`                                                                                  | Context-window **ceiling**, not a fixed size — each GGUF loads at `min(N_CTX, n_ctx_train)` |
| `N_THREADS`              | `0`                                                                                      | `0` = auto                                               |
| `N_BATCH`                | `512`                                                                                    | llama.cpp prompt batch size                              |
| `ADAPTER`                | `llama_cpp`                                                                              | Legacy single-adapter mode (manager dispatch ignores it) |
| `MEMORY_BUDGET_GB`       | `60.0`                                                                                   | LRU evicts past this                                     |
| `PREFIX_CACHE_BYTES`     | `2147483648` (2 GiB)                                                                     | llama.cpp `LlamaRAMCache` capacity; `0` disables         |
| `MLX_PREFIX_CACHE_ENABLED` | `true`                                                                                 | MLX prompt cache master switch                            |
| `MLX_PREFIX_CACHE_MAX_SLOTS` | `4`                                                                                  | Number of independent prefix slots per loaded MLX model; `1` = single-slot legacy behaviour |
| `LLAMA_CPP_EMBEDDING_ENABLED` | `true`                                                                              | Allocate llama.cpp embedding pooling layer at load; needed for `/v1/embeddings`             |
| `BATCH_ENABLED`          | `true`                                                                                   | Coalesce concurrent `/v1/embeddings` requests; `false` = pass-through                      |
| `BATCH_MAX_WAIT_MS`      | `10`                                                                                     | Wait window before flushing a partial batch                                                |
| `BATCH_MAX_SIZE`         | `32`                                                                                     | Force flush when queued inputs hit this count                                              |
| `SCHEDULER_ENABLED`      | `true`                                                                                   | Tenant-aware admission and fair dispatch before backend locks                              |
| `SCHEDULER_GLOBAL_MAX_IN_FLIGHT` | `32`                                                                            | Max scheduler slots active in one engine replica                                           |
| `SCHEDULER_TENANT_RESERVED_IN_FLIGHT` | `2`                                                                        | Soft per-tenant active slot reservation; tenants can borrow idle capacity                  |
| `SCHEDULER_RESOURCE_MAX_IN_FLIGHT` | `1`                                                                              | Default per-model/backend dispatch cap; keeps local serialized adapters fair               |
| `SCHEDULER_VLLM_RESOURCE_MAX_IN_FLIGHT` | `8`                                                                         | Per-model dispatch cap for vLLM-backed models                                              |
| `SCHEDULER_MAX_QUEUE_PER_TENANT` | `64`                                                                             | Per-tenant queue depth before `429 tenant_queue_full`                                      |
| `SCHEDULER_QUEUE_TIMEOUT_SECONDS` | `30`                                                                           | Max wait for scheduler capacity before `503 tenant_queue_timeout`; `0` disables timeout    |
| `SCHEDULER_RETRY_AFTER_SECONDS` | `2`                                                                              | `Retry-After` header on scheduler admission failures                                       |
| `SCHEDULER_WAIT_AGING_PRIORITY_PER_SECOND` | `0.5`                                                                    | Wait-time aging added to dispatch priority                                                 |
| `SCHEDULER_TENANT_FAIRNESS_WEIGHT` | `2.0`                                                                         | Boost for tenants that have not recently received a dispatch                               |
| `OTEL_ENABLED`           | `false`                                                                                  | Master switch for OTLP export |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | `http://localhost:4317`                                                              | OTLP/gRPC collector or OTLP/HTTP traces endpoint |
| `OTEL_EXPORTER_OTLP_PROTOCOL` | `grpc`                                                                                | `grpc` or `http/protobuf` |
| `OTEL_EXPORTER_OTLP_HEADERS` | empty                                                                                  | URL-encoded comma-separated headers; keep API keys in a Secret |
| `OTEL_SERVICE_NAME`      | `inference-engine`                                                                       | `service.name` resource attribute                        |
| `AUTH_ENABLED`           | `false`                                                                                  | Bearer-token gate on `/v1/models` and `/v1/chat/completions` |
| `AUTH_KEYS_FILE`         | `.auth_keys.json`                                                                        | JSON key records; optional `key_id`, `not_before`, and `expires_at` enable managed rotation |
| `INFERENCE_ENGINE_SERVER_TLS_CERT_FILE` | empty                                                                    | Server certificate PEM; must be paired with the private key |
| `INFERENCE_ENGINE_SERVER_TLS_KEY_FILE` | empty                                                                     | Server private-key PEM; never place the value in environment variables |
| `INFERENCE_ENGINE_SERVER_TLS_CLIENT_CA_FILE` | empty                                                               | Optional client CA PEM used when mTLS is required |
| `INFERENCE_ENGINE_SERVER_TLS_REQUIRE_CLIENT_CERTIFICATE` | `false`                                                   | Require a verified client certificate; incomplete settings fail startup |
| `MODEL_ROUTING_POLICY_REQUIRED` | `false`                                                                         | Fail startup when no signed candidate or valid LKG can activate            |
| `MODEL_ROUTING_POLICY_FILE` | `.model_routing_policy.json`                                                         | Operator-mounted signed desired-state candidate                            |
| `MODEL_ROUTING_LAST_KNOWN_GOOD_FILE` | `.model_routing_policy.lkg.json`                                           | Atomic exact-envelope LKG persisted after successful verification          |
| `MODEL_ROUTING_TRUST_STORE_FILE` | `.model_routing_trust.json`                                                   | Purpose-specific issuers/keys, org/env constraints, and revocations        |
| `MODEL_ROUTING_PRICING_FILE` | `.model_routing_pricing.json`                                                       | Explicit per-model input/output rates used for fail-closed cost ceilings    |
| `MODEL_ROUTING_EXPECTED_AUDIENCE` | `orchestra-model-plane`                                                    | Required signed audience                                                    |
| `MODEL_ROUTING_EXPECTED_ENVIRONMENT` | empty                                                                        | Optional local environment binding                                          |
| `MODEL_ROUTING_EXPECTED_ORG_ID` | empty                                                                             | Optional single-org deployment binding                                      |
| `MODEL_ROUTING_CLOCK_SKEW_SECONDS` | `30`                                                                           | Explicit verification skew, bounded to 300 seconds                          |
| `MODEL_ROUTING_MAX_FILE_BYTES` | `1048576` (1 MiB)                                                                  | Size bound applied to signed artifacts before parsing                       |
| `MODEL_ROUTING_INPUT_TOKEN_RESERVE` | `1024`                                                                        | Conservative reserve for model-side chat templates during pre-dispatch bounds |
| `MODEL_ROUTING_RATE_LIMIT_MAX_BUCKETS` | `10000`                                                                     | Fail-closed cap on process-local policy windows; one route holds one window per enforced dimension, so a route bounded by requests, tokens, and spend occupies three |
| `MODEL_ROUTING_RATE_LIMIT_MAX_WINDOW_ENTRIES` | `100000`                                                             | Fail-closed cap on reservations one token or spend window retains, local and shared alike; size it above the requests one route serves in one `budgetWindowSeconds` |
| `MODEL_ROUTING_RATE_LIMIT_SCOPE` | `process-replica`                                                                  | `process-replica` or exact `deployment-shared` policy rate/budget enforcement |
| `MODEL_ROUTING_RATE_LIMIT_REDIS_URL` | empty                                                                         | Direct Redis-compatible URL for shared scope; prefer the mounted file source |
| `MODEL_ROUTING_RATE_LIMIT_REDIS_URL_FILE` | empty                                                                    | Mounted Redis-compatible URL used by the shared limiter                     |
| `MODEL_ROUTING_RATE_LIMIT_SENTINEL_CONFIG_FILE` | empty                                                              | Mounted strict JSON config for TLS/auth Sentinel discovery and replica acknowledgement |
| `MODEL_ROUTING_RATE_LIMIT_ALLOW_INSECURE_REDIS` | `false`                                                            | Permit non-TLS remote Redis only for explicitly isolated test environments  |
| `MODEL_ROUTING_RATE_LIMIT_KEY_PREFIX` | `orchestra:model-routing`                                                   | Bounded, non-secret namespace for hashed shared RPM keys                    |
| `MODEL_ROUTING_RATE_LIMIT_CONNECT_TIMEOUT_SECONDS` | `1`                                                               | Shared-store connection timeout; an unavailable store fails startup         |
| `MODEL_ROUTING_RATE_LIMIT_OPERATION_TIMEOUT_SECONDS` | `1`                                                             | Per-request shared-store timeout; failures deny before model acquisition    |
| `MODEL_PLANE_WORKLOAD_SURFACE` | `unrestricted`                                                                     | `unrestricted`, or `orchestra-model-plane-workload-v1` to fail closed on uncertified workloads; echoed by `/v1/health` and `/v1/ready` |
| `MODEL_PLANE_OBSERVATION_ENABLED` | `false`                                                                         | Enable asynchronous payload-free observed-state reporting                   |
| `MODEL_PLANE_OBSERVATION_ENDPOINT` | empty                                                                           | Exact HTTPS Orchestra `/api/model-routing-observations` URL                  |
| `MODEL_PLANE_OBSERVATION_API_KEY` | empty                                                                            | Direct `model-plane:observe` key; prefer the rotatable file source           |
| `MODEL_PLANE_OBSERVATION_API_KEY_FILE` | empty                                                                       | Mounted key file re-read before every dispatch                              |
| `MODEL_PLANE_OBSERVATION_DEPLOYMENT_ID` | empty                                                                      | Deployment identity; must match an active signed routing policy              |
| `MODEL_PLANE_OBSERVATION_TARGET_ENVIRONMENT` | empty                                                                 | One of `dev`, `test`, `staging`, or `prod`; must match active policy         |
| `MODEL_PLANE_OBSERVATION_ENGINE_INSTANCE_ID` | empty                                                                    | Stable instance identity, normally the Kubernetes pod name                  |
| `MODEL_PLANE_OBSERVATION_VERSION` | `1`                                                                            | `1` for compatibility; `2` adds payload-free signed-route registry coverage |
| `MODEL_PLANE_OBSERVATION_INTERVAL_SECONDS` | `60`                                                                        | Report cadence, bounded to 10 seconds through 24 hours                       |
| `MODEL_PLANE_OBSERVATION_TIMEOUT_SECONDS` | `5`                                                                          | Per-dispatch timeout, bounded to 100 ms through 30 seconds                   |
| `MODEL_PLANE_OBSERVATION_JITTER_RATIO` | `0.1`                                                                          | Symmetric cadence jitter, from 0 through 0.5                                 |
| `MODEL_PLANE_RUNTIME_CONTROL_ENABLED` | `false`                                                                     | Read signed quarantine leases off the observation reply; off leaves the reply body unread. Requires the observation reporter and `MODEL_ROUTING_EXPECTED_ORG_ID` |
| `MODEL_PLANE_RUNTIME_CONTROL_TRUST_STORE_FILE` | `.model_routing_trust.json`                                        | Ed25519 trust store for leases; defaults to the routing trust store. The signing entry must name `orchestra.runtime-control-lease` in `allowedArtifactTypes` |
| `MODEL_PLANE_RUNTIME_CONTROL_MAX_LEASE_SECONDS` | `900`                                                            | Ceiling on a lease's claimed window; a longer one is refused, not shortened |
| `MODEL_PLANE_RUNTIME_CONTROL_STALE_ACTION` | `lease`                                                               | What a *matched* quarantine does after its lease expires: `lease` follows the signed `staleAction`, `continue` and `stop` override it |
| `MODEL_PLANE_RUNTIME_CONTROL_MAX_RESPONSE_BYTES` | `65536`                                                         | Ceiling on the observation reply the lease is read from                     |
| `GUARDRAIL_ENABLED`      | `false`                                                                                  | Master switch; off means no client is built and no route changes behaviour |
| `GUARDRAIL_ENDPOINT`     | empty                                                                                    | Bare origin of an `orchestra-guardrail-evaluate-v1` service; required when enabled, and a startup error when set while disabled |
| `GUARDRAIL_PROFILE`      | `default`                                                                                | Profile id the service resolves thresholds and detector parameters from |
| `GUARDRAIL_NAMES`        | empty                                                                                    | Comma-separated guardrail names sent on every evaluation; required when enabled, since an empty list asks the service to check nothing |
| `GUARDRAIL_API_KEY`      | empty                                                                                    | Direct bearer token; prefer the rotatable file source. One key source is required when enabled |
| `GUARDRAIL_API_KEY_FILE` | empty                                                                                    | Mounted bearer-token file, re-read before every evaluation           |
| `GUARDRAIL_TIMEOUT_SECONDS` | `0.05`                                                                                | Client deadline per evaluation; the wire `budgetMs` is derived back from it |
| `GUARDRAIL_FAIL_MODE`    | `closed`                                                                                 | `closed` or `open` when no usable verdict is produced; `open` is a startup error under the certified workload surface |
| `GUARDRAIL_FAIL_OPEN_MAX_CONSECUTIVE` | `20`                                                                        | Consecutive fail-opens before the client trips to fail-closed        |
| `GUARDRAIL_FAIL_OPEN_WINDOW_SECONDS` | `60`                                                                         | Longest unbroken fail-open run before the client trips; recovery needs a successful evaluation |
| `DEFAULT_JUDGE_MODEL`    | `llama3.2:3b`                                                                            | Used by `/v1/evals/run` when `judge_model` is not set    |
| `AUTO_EVAL_POLICIES_FILE`| `.auto_eval_policies.json`                                                               | JSON array of `{name, match, auto_eval}` rules; missing = no policy |
| `TOOL_AUDIT_ENABLED`     | `true`                                                                                   | Emit `gen_ai.tool_*` span events on every chat completion           |
| `TOOL_AUDIT_MAX_PAYLOAD_CHARS` | `1024`                                                                             | Per-event truncation cap for arguments / result content             |
| `TOOL_TIMING_TTL_SECONDS` | `300`                                                                                  | TTL for the call_id → emit-timestamp store; older entries swept on insert |
| `TOOL_TIMING_MAX_ENTRIES` | `10000`                                                                                | Hard cap on the timing store; oldest entries LRU-evicted past this        |
| `USAGE_LEDGER_ENABLED`   | `false`                                                                                  | Emit one `prometa.model-usage.v2` billing record per priced request    |
| `USAGE_LEDGER_MAX_BUFFER` | `10000`                                                                                 | Bounded hand-off buffer; arriving records refused (counted + logged) past this |
| `USAGE_LEDGER_DRAIN_INTERVAL_SECONDS` | `1.0`                                                                       | How often the background task ships buffered records, 0 < n <= 60      |

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
│     gen_ai.system               = llama_cpp  ← kept for existing dashboards
│     gen_ai.provider.name        = llama_cpp  ← current semconv name for the same thing
│     gen_ai.operation.name       = chat
│     gen_ai.request.model        = llama3.2:1b
│     gen_ai.request.max_tokens   = 8
│     gen_ai.request.temperature  = 0.2
│     gen_ai.request.top_p        = 0.95
│     gen_ai.request.stream       = false
│     gen_ai.usage.input_tokens   = 41         ← post-bind: filled after generate() returns
│     gen_ai.usage.output_tokens  = 5          ← post-bind
│     gen_ai.usage.cached_input_tokens = 32    ← post-bind: prefix-cache reuse
│     gen_ai.response.finish_reason = stop     ← post-bind
│     duration_ms                 = 24.38
└── POST /v1/chat/completions http send
```

Recent semconv versions renamed `gen_ai.system` to `gen_ai.provider.name`. Both
are emitted: the orchestra-python-sdk's own instrumentation still writes
`gen_ai.system`, and splitting existing dashboards across two attribute names
mid-flight would cost more than the duplicate attribute does.

Cold model load shows up as a long `model.acquire` (e.g. 263 ms) above an unchanged `chat.generate`; warm hits drop `model.acquire` to <1 ms. That's exactly the kind of evidence Prometa's signal layer needs to attribute latency to load vs. compute.

Streaming spans additionally carry `gen_ai.server.time_to_first_token`.

### GenAI metrics — TTFT, TPOT, token usage, operation duration

`/v1/metrics` (and its conventional alias `/metrics`) publishes the four
standard [GenAI semantic-convention](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
instruments alongside the existing `inference_engine_*` series:

| Metric | What it answers |
|--------|-----------------|
| `gen_ai_server_time_to_first_token_seconds` | How long until the user sees anything — the primary streaming SLI |
| `gen_ai_server_time_per_output_token_seconds` | Decode rate after prefill (inter-token latency) |
| `gen_ai_client_operation_duration_seconds` | End-to-end per operation |
| `gen_ai_client_token_usage` | Tokens in/out, split by `gen_ai_token_type` |

Labels are the semconv attribute names (`gen_ai_operation_name`,
`gen_ai_provider_name`, `gen_ai_request_model`), i.e. exactly what an OTel
Collector would produce scraping a vLLM deployment — so a stock GenAI dashboard
works against this endpoint without remapping every series.

TTFT and TPOT are recorded only for streams that finished cleanly; a cancelled
or errored stream would otherwise skew the histogram toward whatever it managed
before failing. TPOT excludes the first token by definition. Metric export is
**scrape-only** — `otel.py` installs a TracerProvider, not a MeterProvider, so
these do not go out over OTLP.

### Per-request usage ledger — `prometa.model-usage.v2`

Spans answer "what did this request do"; an invoice needs "what is this request
owed". Those are different artefacts, because the records billing cares about
most are the ones tracing is worst at keeping: a policy denial never opens a
`chat.generate` span at all, and sampling drops whichever spans it likes. So
`USAGE_LEDGER_ENABLED=true` turns on a dedicated channel that constructs
**exactly one record per priced request**, whatever the outcome. Delivery is a
best-effort bounded buffer, described below; it is not a durable outbox.

"Priced request" means one that reached the `_usage.bind_request` seam in a
handler for `/v1/chat/completions`, `/v1/completions`, or `/v1/embeddings` —
the point that has a resolved caller identity, a body the schema accepted, and
a token budget to enforce against, which is what makes a record an invoice
line. Everything short of that seam produces no record at all and cannot
consume a slot in the bounded buffer: a call with no credentials or a malformed
body is attributable to nobody, any other route is unpriced (`/v1/rerank` and
`/tokenize` included), and the two shape guards that answer 400 before the seam
— a `prompt` or an `input` that is a list containing no strings — reject a
well-formed body that still names no work to price. None of those reach a
model. A *denial* is on the other side of the line: it was answered against a
real identity, so it is recorded with `outcome: "denied"`.

Past the seam, record construction and flush eligibility are guaranteed. The
binders that enrich it are individually failure-isolated, so a bug in any one
of them costs that binder's fields and leaves the rest of the line intact.
Acceptance into the bounded buffer and delivery to stdout remain best-effort:
overflow refuses the arriving line, and a sink failure retires the failed
batch after counting it.

Accepted records are JSON lines on stdout, written by a background drain task
and handed to whatever log pipeline the deployment already runs for durability
and retention (the collector in `docker-compose.observability.yml`, or your
cluster's log shipper). There is no file sink on purpose: a billing artefact
written to a long-lived handle and rotated with `copytruncate` loses records
silently, and log shipping is the path every supported deployment shape
already has.

Not an OTLP log record, deliberately. Every `opentelemetry-*` package lives
behind the optional `otel` extra — `make install` is a plain `uv sync` — so an
OTel-only ledger would simply not exist in a default deployment, which is not a
property a chargeback signal may have. `structlog` is a hard dependency, so the
ledger works in every deployment shape. Correlation back to traces still happens:
each record carries the `trace_id` / `span_id` of the serving span when tracing
is on, via the stable `opentelemetry.trace` API. `usage_ledger._sink` is the one
seam to add an OTLP-logs sink later without touching a single call site.

| Group | Fields |
|-------|--------|
| Identity | `usage_record_id`, `engine_request_id`, `runtime_request_id`, `model_invocation_id`, `model_attempt_id`, `route`, `operation`, `stream`, `tenant`, `org_id`, `key_id` |
| Outcome | `outcome` (`ok` / `error` / `timeout` / `denied`), `finish_reason`, `http_status`, `error_type`, `denial_code` |
| Model | `requested_model`, `resolved_model`, `backend`, `request_key_source` |
| Policy | `policy_id`, `policy_digest`, `route_id`, `pricing_digest` |
| Tokens | `input_token_upper_bound`, `output_token_budget`, `input_tokens`, `output_tokens`, `cached_tokens`, `cost_micros` |
| Fallback | `fallback`, `fallback_from_model`, `fallback_from_backend`, `fallback_reason` |
| Timing | `duration_ms`, `ttft_ms` |
| Trace | `trace_id`, `span_id` |

**Metadata only.** `usage_ledger.SCHEMA_FIELDS` is the reviewed key set and the
payload is built by walking that explicit list, never from `dataclasses.asdict`,
so a field added to the internal accumulator cannot reach an emitted record
without a deliberate schema change. The two are not two lists that agree by
convention: `_check_schema_coverage` runs at import and refuses to load the
module if any accumulator attribute is neither an emitted field nor declared
control state, or if any schema key has nothing behind it. Adding a field to one
and forgetting the other fails the process at startup, not silently on an
invoice. No prompt, completion, tool argument, or embedding input is reachable
from any field above, and a test asserts that directly. This structural
guarantee does not inspect the semantic content of caller-owned correlation IDs:
their metadata-only status depends on producers following the opaque, non-PII,
non-secret requirement above.

Things worth knowing before you reconcile against it:

* **Estimated and actual are named differently on purpose.**
  `input_token_upper_bound` is a conservative enforcement bound (a canonical-JSON
  byte count plus a reserve), not a token estimate; `input_tokens` /
  `output_tokens` are what the serving path actually reported. Calling the former
  `estimated_input_tokens` would mislead anyone reconciling the two.
* **`cost_micros` is `null`, never `0`, for an unpriced model.** Absent from the
  pricing catalog means "unknown", and billing it as free under-charges silently.
  Priced values reuse the pre-flight cost ceiling's rounding, so the ledger and
  the `max_cost_micros_per_request` limit never disagree about one call.
* **A billable record can be unattributed.** `tenant`, `org_id`, and `key_id`
  are bound *after* the request is declared billable, so a fault in the
  attribution mapping leaves them `null` rather than deleting the invoice line —
  degrade, never drop. Reconciliation must treat `tenant: null` as
  **unattributed**, not as a tenant literally named `null` and not as absent
  traffic: the call was served and consumed capacity. Route those lines to an
  exceptions queue keyed by `usage_record_id`; `engine_request_id` joins back
  to the request logs that do carry the caller. `resolved_model`, the token
  counts, and `cost_micros` are still intact on the record, so the cost is
  recoverable once the identity is. This should be rare enough to alarm on —
  every occurrence also logs
  `usage_ledger.bind_failed` with `binder=_bind_attribution`, and a non-zero rate
  is an engine bug, not a billing edge case.
* **Buffered, best-effort, and never blocking.** The request path does a dict
  build and a `deque.append`; the write happens on the drain task. Under overflow
  it is the *arriving* record that is refused, never a buffered one — a record
  already accepted is an invoice line the drain task still owes. Every refusal
  increments a counter and each overflow episode logs `usage_ledger.buffer_full`
  once, so the loss is never silent: alarm on
  `inference_engine_usage_ledger_dropped_total` and
  `inference_engine_usage_ledger_sink_failures_total` on `/v1/metrics`.
* **Best-effort buffered, with one narrow redelivery window.** This is not a
  durable outbox and does not promise at-least-once delivery: overflow and sink
  failure can lose records and are surfaced by the counters above. A batch
  stays buffered until the sink accepts it, so cancellation while sink
  acceptance is uncertain may result in the final drain shipping it again.
  The server mints `usage_record_id` once when the record opens and retains it
  across that redelivery window. Deduplicate on that field, never on
  `engine_request_id` or an upstream runtime/invocation/attempt id.
* **Streaming records are flushed by the SSE generator**, not the middleware, so
  they carry real token counts and TTFT. A streamed response whose body never
  runs at all — the client vanished before the first byte — is flushed by the
  middleware's send wrapper instead, so it is still emitted; that record has
  `http_status`, attribution, and policy fields but no token counts.
* **SSE always answers HTTP 200**, so a stream that ends in an error frame is
  recorded as `outcome="error"` with `http_status=200`. A cancelled stream bills
  as `ok` with `finish_reason="cancelled"` — it consumed tokens.
* **One buffer and one drain task per worker.** Records are per-request and
  independently valid, so multi-worker deployments need no join.
* **Scope.** `/v1/chat/completions`, `/v1/completions`, and `/v1/embeddings` only.
  `/v1/rerank` is excluded because it has no routing or pricing integration and
  already sits outside `CERTIFIED_MODEL_WORKLOAD_SURFACE`; `EvalRunner`-driven
  judge inference is excluded because it does not pass through these routes.

### Caller intent labels on chat spans

`/v1/chat/completions` accepts optional generic intent metadata in the request body. When present, the engine stamps it onto `model.acquire` before cold-load or cache lookup work, then onto the `chat.generate` or `chat.stream` span before model generation begins:

```json
{
  "model": "gemma4:26b",
  "messages": [{"role": "user", "content": "Update the config and run the pipeline"}],
  "metadata": {
    "intent": {
      "labels": ["configuration_edit", "flow_execution"],
      "label_names": ["configuration_editing_execution", "flow_process_execution"],
      "source": "client_classifier",
      "preclassified": true,
      "classifier_version": "intent-router-v1"
    }
  }
}
```

The emitted span attributes are:

- `intent.labels`
- `intent.label_names`
- `intent.count`
- `intent.source`
- `intent.preclassified`
- `intent.classifier_version` when supplied

For preclassified frontend prompts, set `metadata.intent.preclassified=true`; the engine only propagates those labels and does not run another classifier. Top-level dotted `intent.*` request keys are still accepted for clients that cannot send nested metadata, but `metadata.intent` is the preferred API shape. If a downstream platform needs a vendor- or platform-specific namespace, map the emitted generic `intent.*` span attributes in the collector, SDK, or ingestion layer rather than hard-coding that namespace into the inference service.

### Plugging into Prometa

Set `OTEL_EXPORTER_OTLP_PROTOCOL=grpc` for an OTLP/gRPC collector or
`http/protobuf` for a direct HTTP traces endpoint. The exporter derives
transport security from the endpoint scheme, rejects credentials in the URL,
and parses purpose-scoped headers without logging their values. Standard OTel
resource and sampler environment variables are honored by the SDK. Service
identity is pre-set to `service.name=inference-engine`,
`service.version=<package version>`.

To wire the engine into the [Prometa platform](https://github.com/caglarsubas/agent-hook-v2) for cross-service tracing, set the resource attributes so the platform's correlation-id resolver finds the engine's role in the canonical chain. Two patterns:

**Pattern A — engine as a standalone agent.** The engine appears in the Prometa registry as its own agent (`inference-engine`). Use this when the engine isn't called from a Prometa-instrumented agent (e.g. direct HTTP from a frontend / CLI):

```bash
OTEL_ENABLED=true \
OTEL_EXPORTER_OTLP_ENDPOINT=https://prometa.example.com/api/v2/otlp/v1/traces \
OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf \
OTEL_EXPORTER_OTLP_HEADERS="x-api-key=prm_live_..." \
OTEL_RESOURCE_ATTRIBUTES="prometa.solution_id=sol_inference,prometa.stage=production"
```

**Pattern B — engine called by a Prometa-SDK-instrumented agent.** The engine's `chat.generate` / `tool.invoke` spans nest under the calling agent's span via standard OTel context propagation (W3C `traceparent` header on the inbound request). Agents using the [Python](https://github.com/prometa-ai/orchestra-python-sdk), [Node](https://github.com/prometa-ai/orchestra-node-sdk), or [Java](https://github.com/prometa-ai/orchestra-java-sdk) SDKs already propagate this header out of the box — no engine-side change needed beyond pointing at the same OTLP endpoint. The platform's resolver then attributes the engine span's identity horizontals (`agent_id`, `solution_id`) by inheriting from the parent agent's resolved chain.

Engine-side resource attributes already wired into the OTLP stream:
- `service.name=inference-engine` — primary identity
- `service.version=<package version>` — auto-discovered
- `prometa.tenant` (per-span, set from the inbound `x-prometa-tenant` header) — drives multi-tenant routing in Grafana panels
- `gen_ai.system`, `gen_ai.request.model`, `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens` — OTel GenAI semconv keys; the platform's cost rollup keys on these.

See the platform-side [`correlation-id-design.md`](https://github.com/caglarsubas/agent-hook-v2/blob/main/resources/correlation/correlation-id-design.md) for the full chain grammar and the SDK READMEs for the agent-side helpers (`set_customer_id`, `set_user_id`, `set_request_model`, `set_tool_name`, etc.) that populate the optional identity-horizontal segments.

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
* **Operational health** — loaded models per replica, loaded model bytes, prefix-cache size by model, and tenant scheduler pressure. These are scraped from the engine's own `/v1/metrics` (Prometheus exposition), not from OTel — they're cheap counters/gauges the engine maintains directly.

Trace correlation is wired both ways: any series in a metric panel can be clicked to land on the matching trace in Jaeger (via `exemplarTraceIdDestinations` on the Prometheus datasource), and Jaeger spans link back to the metric series via `tracesToMetrics`.

#### Caveats and notes

* **Tenant labels show `anonymous` until `AUTH_ENABLED=true`.** The default obs stack has auth off, so all bearer tokens hit the same anonymous tenant — the dashboard surfaces this in the "Active tenants" tile (will read 1, not 2). Turn auth on (`AUTH_ENABLED=true` + a configured `auth_token_map`) and the tenant breakdowns light up.
* **First minute is empty.** spanmetrics flushes every 15 s and Prometheus scrapes every 15 s, so panels need ~30 s of traffic before they read non-zero. `make obs-load` is the friction-free way to seed.
* **Cumulative temporality.** spanmetrics is configured with `AGGREGATION_TEMPORALITY_CUMULATIVE`, so all PromQL panels use `rate(...)` over a 5-minute window. Don't switch to `irate` — the cumulative reset behavior at 5-minute reset windows produces spikes that aren't real traffic.
* **Engine-native metrics aren't OTLP.** `inference_engine_*` gauges (loaded models, prefix-cache size, scheduler queue/in-flight pressure) are scraped straight off `engine:8080/v1/metrics` — they don't need the OTel collector at all. The collector pipeline only carries the trace-derived metrics.

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

The engine only lists a configured vLLM entry in `/v1/models.data` `data[]`
after the upstream's own `/v1/models` response contains the exact `model_id`.
If the endpoint is down or serving a different model, the id is reported under
`/v1/models.data` `unavailable[]` with `available=false`, an
`upstream_reachable` state, the upstream reason, and the same benchmark metadata
until the next probe succeeds.

On macOS with Docker Model Runner, use the host-side OpenAI base before `/v1`
as the endpoint and the exact Docker-advertised id as `model_id`:

```json
[
  {
    "name": "qwen3-vl-8b-instruct",
    "tag": "vllm",
    "endpoint": "http://127.0.0.1:12434/engines",
    "model_id": "huggingface.co/qwen/qwen3-vl-8b-instruct:latest",
    "size_bytes": 17530000000
  }
]
```

Validate the upstream directly first, then restart the native engine and run
the image smoke before handing the id to downstream benchmarks:

```bash
docker model pull hf.co/Qwen/Qwen3-VL-8B-Instruct
curl http://127.0.0.1:12434/engines/v1/models
./scripts/native-service.sh restart engine
make vlm-smoke MODEL=qwen3-vl-8b-instruct:vllm IMAGE=/path/to/vehicle.jpg
```

To promote a single demanded VLM into the ignored live manifest without copying
JSON by hand, use the promotion helper. For the FakeShield-22B FraudGuard pilot:

```bash
make vllm-fakeshield-init FAKESHIELD_ENDPOINT=http://vllm-fakeshield-22b:8000

# Stronger gate: only write the live manifest if the upstream already advertises
# zhipeixu/fakeshield-v1-22b from GET /v1/models.
make vllm-fakeshield-init FAKESHIELD_ENDPOINT=http://vllm-fakeshield-22b:8000 \
  VLLM_REQUIRE_UPSTREAM=1
```

For the SIDA-13B FraudGuard follow-up:

```bash
make download-vlm-models VLM_MODEL=saberzl/SIDA-13B VLM_MAX_WORKERS=1

# CUDA host example: run the SIDA reference implementation behind the
# OpenAI-compatible worker. SIDA_SRC_DIR must point to a checkout of
# https://github.com/hzlsaber/SIDA in a Python/CUDA environment with the SIDA
# requirements plus fastapi and uvicorn installed.
SIDA_SRC_DIR=/path/to/SIDA \
SIDA_PYTHON=/path/to/sida-env/bin/python \
make sida13b-openai-upstream

# The native engine default expects the worker at http://127.0.0.1:8000.
make vllm-sida13b-init SIDA13B_ENDPOINT=http://127.0.0.1:8000 \
  VLLM_REQUIRE_UPSTREAM=1
```

For Molmo-7B-D on Apple Silicon, prefer the MLX-converted 4-bit worker rather
than the 29.9 GiB upstream safetensors or a CUDA vLLM sidecar. The worker
serves an OpenAI-compatible surface and advertises the original upstream id, so
the engine can probe and route it through the existing vLLM/OpenAI-compatible
adapter:

```bash
make install-mlx
make molmo7b-mlx-download
make molmo7b-openai-upstream

# In another shell, once GET /v1/models lists allenai/Molmo-7B-D-0924:
make vllm-molmo7b-init MOLMO7B_ENDPOINT=http://127.0.0.1:8001 \
  VLLM_REQUIRE_UPSTREAM=1
./scripts/native-service.sh restart
make vlm-smoke MODEL=molmo-7b-d:vllm IMAGE=/path/to/vehicle.jpg
```

These targets write `fakeshield-22b:vllm`, `sida-13b:vllm`, or
`molmo-7b-d:vllm` to `.vllm_models.json` with
`supports_strict_image_json=false` and `strict_image_json_status=pending_smoke`.
After the upstream is running, restart the engine, verify the id has
`available=true` under `/v1/models.data` `data[]`, then run repeated vehicle-image
JSON smoke before promoting the descriptor to benchmark-safe. If the upstream is
offline, the id stays visible under `/v1/models.data` `unavailable[]` with typed
reachability fields instead of disappearing from the catalog. If a Hugging Face
snapshot has been downloaded under `HF_VLM_MODELS_DIR` (default
`~/.cache/inference_engine/hf-vlm`) but no live upstream is configured yet,
`/v1/models.data` reports `availability_status="downloaded_but_not_served"` so
benchmark clients can distinguish acquisition progress from serving readiness.
SIDA-13B is not expected to load in the generic vLLM compose sidecar; keep it in
`unavailable[]` until the dedicated worker lists `saberzl/SIDA-13B` from
`GET /v1/models`.
Molmo-7B-D is similarly not a generic `mlx-lm` registry entry in this engine:
use the dedicated `mlx-vlm` worker above so image content parts reach the model.

For InternVL3.5-8B on Apple Silicon, use the generic `mlx-vlm`
OpenAI-compatible worker against the local Hugging Face snapshot. The worker
advertises the original upstream id, while clients call the engine id
`internvl3.5-8b:vllm` after promotion:

```bash
make internvl35-8b-download VLM_MAX_WORKERS=4
make internvl35-8b-openai-upstream

# In another shell, once GET /v1/models lists OpenGVLab/InternVL3_5-8B:
make vllm-internvl35-8b-init VLLM_REQUIRE_UPSTREAM=1
./scripts/native-service.sh restart
make vlm-smoke MODEL=internvl3.5-8b:vllm IMAGE=/path/to/vehicle.jpg
```

Keep `supports_strict_image_json=false` until repeated vehicle-image JSON smoke
passes. The model id requested in issue text is sometimes written with a dot
(`OpenGVLab/InternVL3.5-8B`), but the public Hugging Face repo and engine
descriptor use `OpenGVLab/InternVL3_5-8B`.

For Ovis2.5-9B on Apple Silicon, use Docker Model Runner's host-side
OpenAI-compatible API. The demanded descriptor remains `ovis2.5-9b:vllm`, while
the upstream model id should match the exact id returned by
`GET http://127.0.0.1:12434/engines/v1/models`:

```bash
make ovis25-9b-dmr-pull
curl http://127.0.0.1:12434/engines/v1/models
make vllm-ovis25-9b-init VLLM_REQUIRE_UPSTREAM=1
./scripts/native-service.sh restart
make vlm-smoke MODEL=ovis2.5-9b:vllm IMAGE=/path/to/vehicle.jpg
```

The default promotion target assumes Docker Model Runner advertises
`huggingface.co/aidc-ai/ovis2.5-9b:latest`. Override
`OVIS25_9B_UPSTREAM_MODEL_ID` if the local runner reports a different id.
Keep `supports_strict_image_json=false` until repeated vehicle-image JSON smoke
passes.

For the current FraudGuard vehicle-photo model demand shortlist, including
local bakeoff candidates and VLM serving/evaluation requirements, see
[`docs/MODEL_DEMAND_SHORTLIST.md`](docs/MODEL_DEMAND_SHORTLIST.md).

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

Clients then send `model: "llama-3.2-1b-instruct:vllm"` or `"llama-3.2-3b-instruct:vllm"` after those ids appear in `/v1/models.data`, and the engine routes each through its own GPU-pinned vLLM upstream. Adding a third model is two more entries: another vLLM service block in compose with `device_ids: ['2']`, and another `.vllm_models.json` entry pointing at it.

`count` and `device_ids` are mutually exclusive in the Compose GPU spec, which is why the single-GPU and multi-GPU paths live in separate overlay files rather than as one parameterised service.

#### Honest constraints

- **The Compose vLLM overlay is CUDA-oriented.** `make compose-vllm-up` expects a Linux/WSL CUDA host with the NVIDIA Container Toolkit; it is still the production-style path for GPU-pinned vLLM services. On Apple Silicon/macOS Docker Desktop, use Docker Model Runner's host-side API when vLLM-Metal is available, and point `.vllm_models.json` at `http://127.0.0.1:12434/engines`. Model Runner may list models that still fail at load time, so keep the strict image JSON smoke as the final exposure gate.
- **One model per vLLM process.** Multi-model = multiple vLLM containers on different ports + multiple `.vllm_models.json` entries. The engine is the multiplexer.
- **No prefix-cache introspection on vLLM.** vLLM's PagedAttention is excellent but its OpenAI-compatible HTTP API doesn't expose per-call hit counts the way our local adapters do. `prefix_cache_*` properties on `VLLMAdapter` report `disabled` so the chat span attrs stay uniform across backends.
- **Embeddings unsupported.** `VLLMAdapter.embed()` raises `EmbeddingsNotSupportedError`; `/v1/embeddings` against a vLLM model returns 501. Continue to use llama.cpp for embeddings (round 15) or wire a separate vLLM container with an embedding model + a custom adapter override.

### OpenRouter gate for large open-weight models

OpenRouter is a second OpenAI-compatible HTTP lane for cases where the demanded
model is larger than the local lane should carry, or where a smaller
image-capable model is intentionally exposed as a benchmark-only candidate. It
is deliberately config-driven and policy-gated: a `.openrouter_models.json`
entry is accepted only when `parameter_count_b` is strictly greater than
`OPENROUTER_MIN_PARAMETER_COUNT_B` (default `50`) unless the entry sets
`benchmark_only=true` and `modality` includes `image`. All OpenRouter entries
must still set `open_weight=true` and `proprietary=false`; if `open_source` is
present it cannot be `false`.

The committed `.openrouter_models.example.json` is a curated operator catalog
of current OpenRouter ids verified against `https://openrouter.ai/api/v1/models`.
It includes large Llama, Qwen, Nemotron, Nous/Hermes, Mixtral, selected
Llama-finetune lanes, and smaller FraudGuard VLM candidates that remain
`benchmark_only` until a repeated strict image+JSON smoke passes. Copy only the
entries you want to expose into the ignored runtime `.openrouter_models.json`.

Bootstrap the ignored live manifest from the curated catalog:

```bash
make openrouter-models-init
```

Then delete any entries that should not be exposed in this deployment. The live
path is controlled by `OPENROUTER_MODELS_FILE` and defaults to
`.openrouter_models.json`; the example catalog is never loaded automatically.

Single-entry shape:

```json
[
  {
    "name": "llama-3.1-70b-instruct",
    "tag": "openrouter",
    "model_id": "meta-llama/llama-3.1-70b-instruct",
    "parameter_count_b": 70,
    "open_weight": true,
    "open_source": true,
    "proprietary": false
  }
]
```

Set the bearer only in ignored runtime env:

```bash
OPENROUTER_API_KEY=sk-or-v1-...
```

Clients then send `model: "llama-3.1-70b-instruct:openrouter"`. The engine
adds `Authorization: Bearer $OPENROUTER_API_KEY` upstream, probes
OpenRouter's `/v1/models` before listing the model under `/v1/models` or
`/v1/models.data`,
and stamps every inference span/response with:

| value | meaning |
|-------|---------|
| `local-inference` | local llama.cpp, MLX, vLLM, Ollama-HTTP, or other no-provider-key paths |
| `openrouter-api-key` | OpenRouter-backed request using `OPENROUTER_API_KEY` |
| `openai-api-key` | reserved flag for a future direct OpenAI provider lane |

The flag is returned as `request_key_source` on completion, embedding, rerank,
and streaming chunk payloads, and as `llm.request.key_source` on model request
spans.

When `OPENROUTER_FALLBACK_ENABLED=true`, generation errors from the configured
local backend names in `OPENROUTER_FALLBACK_BACKENDS` retry once on OpenRouter.
By default, the retry target is the same exposed model name with the
`openrouter` tag, e.g. `gemma4:26b` falls back to `gemma4:openrouter`; set
`OPENROUTER_FALLBACK_MODEL` to force a single shared fallback. Context-window
errors stay client-visible 400s and do not fallback. Streaming requests only
fallback before the first assistant chunk is emitted, so clients never receive
a response stitched across providers. For HTTP-backed upstreams this is the
*second* line of defence: bounded retries on the same deployment run first, so
a transient upstream failure no longer changes which model answers. See
[Upstream resilience](#upstream-resilience--retry-the-same-model-then-cool-the-deployment-down).

`GET /v1/models.data` is the machine-readable model catalog for benchmark
harnesses and internal UIs. It uses the same probe-aware live surface as
`/v1/models`, then adds provider details that clients need before sending image
or strict JSON workloads:

```json
{
  "object": "model_catalog",
  "data": [
    {
      "id": "qwen3-vl-235b-a22b-instruct:openrouter",
      "available": true,
      "upstream_reachable": true,
      "availability_status": "available",
      "provider": "openrouter",
      "backend": "openrouter",
      "upstream_model_id": "qwen/qwen3-vl-235b-a22b-instruct",
      "modality": "text+image->text",
      "supports_images": true,
      "context_length": 262144,
      "max_image_side_px": null,
      "supports_json_mode": true,
      "supports_strict_image_json": false,
      "strict_image_json_status": "unstable",
      "strict_image_json_checked_at": "2026-06-19",
      "strict_image_json_detail": "Issue #38 12-row FraudGuard smoke parsed 2/12; keep out until repeated full-shape smoke passes.",
      "request_key_source": "openrouter-api-key"
    }
  ],
  "unavailable": [
    {
      "id": "fakeshield-22b:vllm",
      "available": false,
      "upstream_reachable": false,
      "availability_status": "upstream_unreachable",
      "reason": "upstream_unreachable",
      "backend": "vllm",
      "endpoint": "http://vllm-fakeshield-22b:8000",
      "upstream_model_id": "zhipeixu/fakeshield-v1-22b",
      "modality": "text+image->text",
      "supports_strict_image_json": false,
      "strict_image_json_status": "pending_smoke"
    },
    {
      "id": "sida-13b:vllm",
      "available": false,
      "upstream_reachable": false,
      "availability_status": "downloaded_but_not_served",
      "reason": "downloaded_but_not_served",
      "provider": "sida",
      "backend": "vllm",
      "endpoint": "http://127.0.0.1:8000",
      "upstream_model_id": "saberzl/SIDA-13B",
      "download_status": "downloaded",
      "local_snapshot_path": "/Users/example/.cache/inference_engine/hf-vlm/saberzl--SIDA-13B",
      "supports_strict_image_json": false,
      "strict_image_json_status": "pending_smoke"
    }
  ]
}
```

`max_image_size`, `max_image_side_px`, and `max_image_pixels` are `null` when
the manifest or provider does not publish a stable limit; clients should keep
their own preprocessing cap, such as the FraudGuard 512px smoke setting, in
that case. OpenRouter routes are provider-backed external inference. Treat
them as benchmark-only until the deployment owner has approved the selected
model/provider for production and commercial use.

`supports_images` and `supports_json_mode` are independent capability hints.
Benchmark harnesses that require image input and parseable JSON in one call
should require `supports_strict_image_json=true`. If the field is `false`, the
entry stays visible for operators but should be skipped before expensive
pilots; `strict_image_json_status`, `strict_image_json_checked_at`, and
`strict_image_json_detail` explain the last strict smoke result. Upstream
OpenRouter 4xx/5xx failures surface through typed engine 502 payloads with
`detail.type="upstream_http_error"`, `detail.upstream_status_code`, and the
bounded upstream error body under `detail.detail`.

### Dynamic batching (embeddings)

Concurrent `/v1/embeddings` requests first pass through tenant-aware admission, then requests for the same loaded adapter merge into a single underlying `adapter.embed()` call within a small wait window. **Per-adapter capability detection** picks the right inner path automatically:

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

`/v1/chat/completions` is **not** dynamically batched. Continuous batching for autoregressive decode (vLLM-style) requires reimplementing the inference loop on top of `llama_cpp` ctypes / `mlx.core` — multi-round project, not a one-shot. Today, concurrent local chat requests for the same adapter are admitted by the tenant scheduler, then serialize through the per-adapter lock (round 6's concurrency design). Documented here so the trade-off is visible rather than implied: for high-QPS chat workloads, plug a dedicated continuous-batching backend (vLLM, SGLang) behind the existing `InferenceAdapter` ABC.

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

## Signed model-routing desired state

The first Orchestra model-plane increment accepts a purpose-separated Ed25519
routing-policy envelope produced by the Prometa control plane. Verification is
entirely local. The engine does not call Prometa during startup, reload, or an
inference request.

The verifier requires exact canonical payload bytes, strict schema/version,
issuer and key trust, organization/environment constraints, validity and
offline-lease windows, and non-revoked key and policy JTIs. Files are bounded
before parsing. A verified candidate is persisted with mode `0600` through a
same-directory atomic replace and becomes last known good.

Startup and reload follow this order:

1. Verify the mounted candidate and activate it when valid.
2. If the candidate is absent or invalid, verify last known good against the
   current trust store and current time.
3. Reject last known good after revocation, trust removal, binding mismatch,
   expiry, or offline-lease expiry.
4. Fail closed when `MODEL_ROUTING_POLICY_REQUIRED=true` and neither source is
   valid. A present but invalid candidate never silently disables governance,
   even when required mode is false.

Use `.model_routing_trust.example.json` as the trust-store shape. Never reuse
bundle or promotion keys for model routing. The payload-free operator endpoints
show active policy identity and reload atomically:

```bash
curl -H "Authorization: Bearer $ENGINE_ADMIN_KEY" \
  http://127.0.0.1:8080/v1/admin/model-routing-policy

curl -X POST -H "Authorization: Bearer $ENGINE_ADMIN_KEY" \
  http://127.0.0.1:8080/v1/admin/model-routing-policy:reload

curl -X POST -H "Authorization: Bearer $ENGINE_ADMIN_KEY" \
  http://127.0.0.1:8080/v1/admin/model-routing-pricing:reload
```

The pricing-only reload keeps the active signed policy unchanged. It requires
both an active policy and a nonempty mounted pricing catalog, validates every
cost-bounded route against the candidate catalog, and swaps the combined
runtime state only on success. Missing, malformed, or incomplete pricing leaves
the previous in-memory policy and pricing state untouched. Artifact mounting
and reload invocation remain tenant CI/CD responsibilities.

Chat, completion, and embedding endpoints now enforce an active policy before
model registry lookup:

1. Re-check policy validity, expiry, and offline lease at request time.
2. Require the caller's `org_id` to match the signed policy organization.
3. Resolve an exact requested-model alias or the final wildcard route.
4. Reject caller input/output bounds, RPM, TPM, worst-case fallback cost, or an
   exhausted window spend budget before loading a model or contacting an
   upstream.
5. Try only the signed primary and ordered fallback set for timeout, backend,
   or embedding-capability failures. Global OpenRouter fallback is not
   consulted in governed mode.
6. Stamp policy ID, revision, digest, release, deployment, organization,
   environment, route, selected candidate, limits, and pricing digest on route
   and workload spans. Canonical `prometa.*` aliases identify the signed
   artifact as `model-routing-policy`; existing `model_routing.*` attributes
   remain the compatibility contract.

`maxInputTokens` uses the UTF-8 byte count of model-bound request fields plus
`MODEL_ROUTING_INPUT_TOKEN_RESERVE` as a conservative tokenizer-independent
upper bound. Remote image URLs have no locally knowable image-token cost and
fail closed when the selected route bounds input or cost; inline data URLs are
counted. Embedding requests use the canonical input array and an output-token
budget of zero, so input bounds, pricing, RPM, organization binding, and signed
fallback order apply without treating vector dimensions as generated tokens.
`maxCostMicrosPerRequest` reserves the worst case across the primary
and every possible fallback using `.model_routing_pricing.json`. A costed route
cannot activate unless every signed candidate has pricing. The catalog is
deployment metadata, not hidden policy: its digest is exposed in status and on
every governed decision span.

`maxRequestsPerMinute`, `maxTokensPerMinute`, and the
`maxCostMicrosPerWindow`/`budgetWindowSeconds` pair are sliding windows keyed by
policy, route, organization, and tenant. All three are evaluated in a single
atomic step, before model acquisition, and admit together or not at all: a
request denied on its token window leaves no charge on its request window. When
more than one of them would deny, the reported code is fixed: windows are
evaluated in the order requests, tokens, spend, the entry ceiling below is
checked ahead of the limit within each, and the first denial is the answer. Both
scopes apply that order, so a signed policy denies with the same code and the
same status wherever it runs.
Each has its own stable denial code and `Retry-After` — `rate_limit_exceeded`,
`token_rate_limit_exceeded`, and `budget_exceeded`, all answered 429. A token
denial additionally carries `x-ratelimit-limit-tokens`,
`x-ratelimit-remaining-tokens`, and `x-ratelimit-reset-tokens`; a request denial
carries the `-requests` trio it always did. A spend denial carries neither:
there is no OpenAI header dimension for a budget window, and reporting the
request window as exhausted when it is not would send an SDK into an hour-long
backoff it does not owe.

`maxTokensPerMinute` and `maxCostMicrosPerWindow` are **pre-consumed, then
settled**. Admission reserves the conservative upper bound — the input estimate
plus the request's own output-token budget (its `max_tokens`, itself already
bounded by the route's `maxOutputTokens`) for the token window, and the same
worst-case-across-every-fallback figure `maxCostMicrosPerRequest` uses for the
spend window — so a request is never admitted against a budget concurrent
requests have already committed. When the call finishes, the reservation is
revised down (or up) to the tokens actually served and their catalog price.
Settlement runs on every exit: success, upstream error, structured-output
rejection, and client disconnect, with a streamed request settling when its
body ends rather than when its route returns. A request that reported no usage
at all releases its whole reservation. Two cases keep their hold instead,
because what they consumed is real and unknown rather than absent: a stream
abandoned mid-generation, whose token counts ride on a terminal frame the
client is no longer there to receive, keeps its whole reserve, so disconnecting
early is not a way around either window; and a request served by a model
missing from the pricing catalog keeps its spend hold. In flight, a window can
therefore be occupied by up to the sum of the outstanding reservations, and
never by more.

Because settlement revises entries downward rather than removing them, a token
or spend window is not bounded by its own limit the way a request window is: a
request window holds one entry per admitted request and so can never exceed
`maxRequestsPerMinute`, while a settling window holds one entry per served
request until that entry expires or settles to exactly zero.
`MODEL_ROUTING_RATE_LIMIT_MAX_WINDOW_ENTRIES` caps how many reservations one
*settling* window retains and is fail-closed: an admission that would exceed it
is denied `rate_limit_state_capacity` (503). The cap applies to the token and
spend windows in both scopes, and to request windows in neither — a signed
`maxRequestsPerMinute` above the cap is still enforced at the number that was
signed, in both scopes. A reservation settled to zero is removed outright
rather than left dating the window.

**Size the cap against the widest budget window on the deployment, not against
the default.** The reachable load is one retained entry per served request per
(policy, route, organization, tenant) for the length of that window, so a route
carrying `budgetWindowSeconds: 86400` at the default 100 000 entries fails
closed at roughly 1.2 requests per second sustained, however far under budget
the actual spend is. Set the cap to at least the requests you expect one such
route to serve in one window, and budget memory accordingly: an entry is a
sorted-set member plus a hash field per window, order of a hundred bytes.
Raising the cap costs memory, not latency: the shared script reclaims expired
entries in bounded batches spread over successive admissions, so no single
request pays for clearing a full window and no tenant waits behind one.
`/metrics` carries `inference_engine_model_routing_rate_limit_window_entries_peak`,
the most entries one window has held since the process started, alongside
`..._max_window_entries` and a `..._state_capacity_denials_total` counter — the
peak against the cap is the headroom signal, and the counter is the cliff.

The dependency-free default remains `process-replica`. **`process-replica` is a
per-replica budget, not a fleet budget**: every replica keeps its own windows,
so a deployment of N replicas will admit up to N times each signed ceiling.
Setting `MODEL_ROUTING_RATE_LIMIT_SCOPE=deployment-shared` uses a tenant-owned
Redis-compatible service and one atomic server-time script for exact aggregate
enforcement across replicas — one round trip to admit all three dimensions, one
to settle the two that settle. Exactly one direct URL, URL file, or Sentinel
config file must be selected. Store keys contain only a SHA-256 identity digest;
request, route, tenant, and organization values are not stored in clear text.
Remote connections require TLS unless the operator explicitly enables insecure
transport for an isolated test profile. Startup and request-time store failures
are fail-closed for every dimension alike, with no downgrade to the local
limiter and no call to the Orchestra control plane. A failure while *settling*
is the one exception: the request has already been served, so the failure is
logged and the reservation stands at its admission value until its window
slides past it — restrictive, never permissive. The admin reload swaps policy
and pricing as one immutable runtime snapshot, so requests cannot observe a
mixed revision.

A route that leaves the v2 limit fields null, and every v1 policy, reserves
nothing and touches no additional store keys. A route bounded by
`maxCostMicrosPerWindow` cannot activate unless every signed candidate has
pricing, exactly as `maxCostMicrosPerRequest` already required.

The verifier accepts `policyVersion` 1 and 2. Version 2 adds
`maxTokensPerMinute`, the `maxCostMicrosPerWindow` and `budgetWindowSeconds`
pair, `candidateWeights`, and `shadowModel`. The key set is exact per version:
a v1 policy carrying any v2 key, or a v2 policy omitting one, is rejected as
`malformed_claims`, and nonsensical values or combinations are rejected as
`invalid_routes` — including a `shadowModel` naming any live candidate of its
own route, that is its `primaryModel` or any of its `fallbackModels`. The v1
shape check runs before every claim-level check, so a v1 policy carrying a v2
key reports `malformed_claims` whatever else is wrong with it.
`maxTokensPerMinute` and the `maxCostMicrosPerWindow`/`budgetWindowSeconds`
pair are enforced at request time as described above; `candidateWeights` and
`shadowModel` are parsed, bounded, and reported but not yet acted on.
`/v1/admin/model-routing-policy` and the model-plane
observation report `accepted_policy_versions` so the control plane can confirm
fleet readiness before it starts signing v2.

### Emitter contract for the signer

The five v2 keys are nullable, not optional. The verifier compares the set of
v2 keys *present* on each route against the set its `policyVersion` mandates,
so signers must emit by key presence, not by value:

- **A v2 policy must serialise all five v2 keys on every route** —
  `candidateWeights` and `shadowModel` on the route, `maxTokensPerMinute`,
  `maxCostMicrosPerWindow`, and `budgetWindowSeconds` on its `limits` — using
  an explicit `null` wherever the route does not use one. Omitting a key on
  even one route rejects the whole policy as `malformed_claims`.
- **A v1 policy must emit none of the five, not even as `null`.** A `null`
  still counts as present, so a v1 route carrying `"shadowModel": null` is
  likewise `malformed_claims`.

Both directions reject the whole artifact, not the offending route: the engine
falls back to its last-known-good policy rather than partially applying the
rejected one. A route that uses no v2 feature is therefore still five keys
longer under v2:

```json
{
  "routeId": "default",
  "requestedModel": "*",
  "primaryModel": "llama3.2:3b",
  "fallbackModels": [],
  "candidateWeights": null,
  "shadowModel": null,
  "limits": {
    "maxInputTokens": null,
    "maxOutputTokens": 1024,
    "maxRequestsPerMinute": null,
    "maxCostMicrosPerRequest": null,
    "maxTokensPerMinute": null,
    "maxCostMicrosPerWindow": null,
    "budgetWindowSeconds": null
  }
}
```

`tests/fixtures/model-routing-policy-v2.json` is the reference artifact; its
first route shows the same shape with the v2 features actually populated.

For a replicated deployment, point
`MODEL_ROUTING_RATE_LIMIT_SENTINEL_CONFIG_FILE` at a Secret-mounted document:

```json
{
  "configVersion": 1,
  "serviceName": "orchestra-model-routing",
  "sentinels": [
    {"host": "sentinel-0.tenant.svc.cluster.local", "port": 26379},
    {"host": "sentinel-1.tenant.svc.cluster.local", "port": 26379},
    {"host": "sentinel-2.tenant.svc.cluster.local", "port": 26379}
  ],
  "minOtherSentinels": 1,
  "database": 0,
  "password": "replace-with-mounted-data-password",
  "sentinelPassword": "replace-with-mounted-sentinel-password",
  "tls": true,
  "caFile": "/etc/orchestra/ca/ca-bundle.crt",
  "requiredReplicaAcks": 1,
  "replicaAckTimeoutMilliseconds": 500
}
```

The schema rejects unknown fields, duplicate discovery endpoints, fewer than
three Sentinels, and a peer threshold that cannot be satisfied. Optional
`username` and `sentinelUsername` fields support ACL deployments. TLS verifies
both discovery and data-node hostnames; `caFile` may be omitted to use the
container trust store. Sentinel credentials and data credentials may differ.

An accepted bucket mutation must receive the configured number of Redis
`WAIT` acknowledgements before inference can proceed. Missing acknowledgements
deny the request as `rate_limit_backend_unavailable`; because the primary may
already contain the mutation, that failure can conservatively consume budget.
The acknowledgement timeout must be strictly lower than
`MODEL_ROUTING_RATE_LIMIT_OPERATION_TIMEOUT_SECONDS` so the socket remains
available for the bounded `WAIT` response.
Sentinel rediscovery reconnects later requests to the promoted primary. The
engine does not retry an indeterminate mutation, choose a replica to promote,
provision quorum, or rotate this Secret in place. Those remain tenant topology
and rollout responsibilities.

Enforcement currently covers `/v1/chat/completions`, `/v1/completions`, and
`/v1/embeddings`. While a policy is active, chat auto-eval plus `/v1/rerank`
and `/v1/evals/run` return a payload-free
`model_routing_workload_not_integrated` denial instead of reaching
`ModelManager` outside governance. Standalone tenant chart wiring now ships;
those remaining workload integrations plus OpenShift lifecycle,
backup/recovery, multi-replica load, and SLO certification remain open.

The pinned OpenShift profile additionally sets
`MODEL_PLANE_WORKLOAD_SURFACE=orchestra-model-plane-workload-v1`. Under that
named contract, the same excluded workloads fail closed with
`model_plane_workload_not_certified` even before a policy can be activated.
`/v1/health` and `/v1/ready` expose the configured surface identity so a
deployment observer can bind the running process to the release contract.

### Asynchronous model-plane observations

The engine can continuously report observed state to Orchestra without making
the control plane a request-time dependency. The reporter is disabled by
default. When enabled, startup validates the exact HTTPS endpoint, deployment
and environment scope, instance identity, and one API-key source. Plain HTTP is
accepted only for loopback integration tests.

Each cycle computes the sorted, deduplicated `/v1/models` ID digest locally and
sends only that digest, available/unavailable counts, readiness, engine
version, and the exact payload-free routing-policy status. Observation v1 is
the compatibility default. Opt-in v2 also resolves each unique candidate from
the active signed policy against the same probe-aware local registry used by
requests. It reports only aggregate candidate and ready/unavailable route
counts plus `ready`, `degraded`, or `unavailable`; it does not report candidate
or route names. Model names, routes, prompts, responses, credentials, and
inference payloads never leave through this path. An active policy must match
the configured deployment and environment or the cycle fails closed locally.
Observation-report spans carry the same typed artifact, policy, release,
deployment, and environment aliases without changing either wire version.

Use a platform API key scoped to `model-plane:observe`. A mounted key file is
re-read before every request, which permits rotation without engine restart.
Transient transport, authentication, throttling, and server failures retain
the same observation ID for idempotent retry. Permanent contract rejection
drops that payload so the next cycle can report corrected state. Redirects are
never followed.

Reporting begins after startup model probes and runs on a separate background
task. A platform outage cannot change engine readiness, routing, or inference;
the last-known-good policy continues to govern locally. Inspect delivery state
at `/v1/admin/model-plane-observer` and through the
`inference_engine_model_plane_observer_*` metrics on `/v1/metrics`.

### Runtime control (quarantine) — a signed lease on the reply

The engine has no inbound control channel and does not grow one for this. The
observation POST above already leaves the engine on a timer; the control plane
answers that POST with a `runtimeControls` delivery block, and the engine reads
it off the reply. No new listener, no new connection, no polling, and nothing
on the inference request path calls out.

The state travels as an **Ed25519-signed, short-TTL lease**
(`orchestra.runtime-control-lease`) verified locally against the same trust
store the routing policy uses, through the same trust-resolution and Ed25519
code path and the same `signed-payload-json-v1` canonicalization. The engine
also checks the **signing key's purpose**: the trust entry must name
`orchestra.runtime-control-lease` in its `allowedArtifactTypes`, so a routing,
bundle or promotion key is refused here even though its signature verifies. The
issuer is expected to refuse those keys as well, but a check performed only at
issue is not a check: a misconfigured or compromised issuer is the case it
exists for, which is why the verifier repeats it.

`tests/fixtures/lease-vectors/` mirrors the cross-implementation vector set,
run in full by `tests/test_model_plane_runtime_control.py`: nine lease-direction
vectors and ten acknowledgement-direction cases, the same bytes the Python SDK
host and the control plane run. They pin the envelope, the claim names, the
canonicalization, the trust-entry purpose field, the expected digests and the
acknowledgement's wire shape, so a divergence between this runtime and any
other consumer of the same fixtures fails a test rather than a deployment.

**Scope is typed, and this runtime says what it can enforce.** A control names
its subject at one of five scopes. The engine holds three identities — the org
and deployment it is configured as, and the tenant a request authenticated as —
so it enforces `org`, `tenant` and `deployment`, and **ignores `solution` and
`agent`**, which it has no way to resolve. Every acknowledgement declares that
set in `enforceable_scopes` and counts ignored controls in
`ignored_control_count`, so an agent-scoped quarantine issued into a fleet of
engines renders as unenforced with a reason rather than as a green tick.
Agent- and solution-granular refusal belongs to the SDK host, which knows which
agent is running.

`enforcedControlCount` is **not** the number of controls the lease names. It
counts only controls that are at an enforceable scope, matched to a subject
this replica holds, *and* in state `quarantined`. Its own org id and deployment
id are matched the moment a lease is applied; a tenant is only matched once a
request from that tenant arrives, so a quarantine naming a tenant this replica
never serves counts for nothing here. A matched control in state `serving` is
governed, not enforced, and counts for nothing either. A lease this replica
enforces nothing under reports `enforcement: "advisory"` and
`enforcedControlCount: 0`, whatever the lease asked for, because reporting
otherwise would let the control plane count this replica towards a quarantine
it does not apply.

**Precedence is fixed, and `mode` is decided first.**

```
1. mode == "advisory"                          -> never refuse. Report only.
2. no control matched this request             -> serve.
3. matched quarantine, lease live              -> refuse.
4. matched quarantine, lease expired           -> the stale action decides how
                                                  much is refused, not whether.
5. matched control in state "serving"          -> serve while the lease is live.
```

An advisory lease can therefore never refuse a request under any staleness
condition, and a replica holding no lease at all matches nothing and so stops
nothing. An enforced quarantine stops `/v1/chat/completions`, `/v1/completions`,
`/v1/embeddings` and `/v1/rerank` with a typed, payload-free `503`:

```json
{"detail": {"message": "runtime control has quarantined this subject",
            "type": "model_plane_quarantined",
            "code": "model_plane_quarantined", "scope": "tenant"},
 "error": {"message": "runtime control has quarantined this subject",
           "type": "model_plane_quarantined",
           "code": "model_plane_quarantined", "scope": "tenant", "param": null}}
```

`scope` names which scope matched, never which subject. The refusal happens
before admission accounting, so a quarantined request takes no rate-limit
reservation. The check itself reads a cached projection of the lease:
signatures are verified when a lease arrives on the observer's cadence, never
per request.

**The trade, stated plainly.** `MODEL_PLANE_RUNTIME_CONTROL_MAX_LEASE_SECONDS`
(default 900s) is a ceiling on how long any one lease may claim to be valid: a
lease whose `expiresAt` is further than that from its `notBefore` is refused
rather than silently shortened. When controls actually go stale is set by the
lease's own window, which this ceiling only bounds.
What a *matched* quarantine does once its lease expires with no refresh is a
deployment choice. All three settings decide what happens to **enforcement**,
never to traffic directly, and none of them ever resumes serving a quarantined
subject: a quarantine that lapsed on expiry could be lifted by cutting this
replica off from the issuer, which would make a network partition a kill-switch
override.

| `MODEL_PLANE_RUNTIME_CONTROL_STALE_ACTION` | After a matched quarantine's lease expires |
|---|---|
| `lease` (default) | Whichever the lease's own signed `staleAction` asks for, `continue` or `stop`. The claim is inside the signature precisely because it decides behaviour when the issuer is unreachable. |
| `continue` | Keep *enforcing* the last verified lease: the quarantine that was in force stays in force, refusing with `503 model_plane_quarantined`, and the subjects that lease named `serving` keep being served. |
| `stop` | The wider opt-in: every subject that lease governed here stops with `503 model_plane_control_lease_stale`, the ones it named `serving` included. The cost is that traffic which was never quarantined also fails during an outage. |

None of the three touches a request no control matched, so no setting here can
turn a control-plane outage into a fleet-wide inference outage.

**Desired is not enforced.** Every surface reports both, so neither the operator
nor the control plane has to guess which one it is looking at:

- The observation payload carries a `runtimeControlAck` object. It is an
  **additive field on the existing observation version** — no version bump, so
  a control plane that predates this contract ignores the key instead of
  rejecting the whole payload. Its shape is pinned to nine camelCase members:
  `leaseId`, `revision`, `enforcement`, `enforceableScopes`,
  `enforcedControlCount`, `ignoredControlCount`, `stale`, `leaseExpiresAt` and
  `leaseParseFailed`. There is deliberately no `mode` — that is the issuer's
  instruction and already travels in the lease, while this object reports what
  the replica did. It is emitted on every observation, not only on a state
  change, and while a lease is stale it still reports what is still being
  enforced rather than zero.
- `GET /v1/admin/model-plane-runtime-control` returns that acknowledgement plus
  the local policy: `enforceable_scopes` and `ignored_scopes`, `matched_state`,
  the control counts, `stale_action_policy` and the `effective_stale_action` it
  resolves to, the lease's own `lease_stale_action` and `lease_mode`, the
  delivery status of the last reply, and the last refresh error.
- `/v1/metrics` — `inference_engine_model_plane_runtime_control_lease_held`,
  `…_enforcing`, `…_stale`, `…_stopping_while_stale`, `…_controls`,
  `…_matched_controls`, `…_ignored_controls`, `…_enforced_controls`,
  `…_revision`, `…_lease_seconds_remaining`, `…_leases_accepted_total`,
  `…_leases_rejected_total`, `…_refusals_total`.

Counts and flags are the whole of it. Neither `/v1/metrics` nor `/v1/ready`
renders a subject id, a reason code, a lease id or any operator free text, and
neither does the `503` a refused caller receives — `/v1/ready` says nothing
about runtime control at all. The lease id and its revision appear only on the
admin status and in this replica's own log.

A refresh that fails to verify — wrong signature, a key not provisioned for
this artifact type, wrong org or environment, a revision below the highest this
replica has accepted, or a replay of an older instance of the revision it holds
— is counted and logged, and the lease already held keeps governing until its
own clock runs out. Revisions are ordered globally per `(orgId,
targetEnvironment)` rather than per `leaseId`, so an authentic older `serving`
lease cannot lift a live quarantine by arriving under a different `leaseId`.
An identical redelivery of the lease already held is not a replay: it is the
ordinary case on every observation tick, and it is accepted rather than
counted as a rejection. A lease that cannot be parsed is recorded as `lease_parse_failed` and is
never read as "no quarantine". A reply whose delivery block reports `disabled`
or `unavailable` is the control plane declining to deliver, not a release:
nothing is lifted, the status is recorded for the operator, and the held lease
still expires on schedule.

Off is the default and it is a real off: `MODEL_PLANE_RUNTIME_CONTROL_ENABLED`
false means the reply body is never read and the request path takes no branch.
Enabling it requires the observation reporter — that reply is the only channel
it uses — plus `MODEL_ROUTING_EXPECTED_ORG_ID`, without which an org-scoped
control would have no subject to match, and a readable trust store at startup.
Anything missing is a startup failure rather than a deployment that looks
controlled and is not.

## Auth + tenant attribution

Per-key bearer-token auth — off by default for local dev, switch on with `AUTH_ENABLED=true`. Keys live in a JSON file (`.auth_keys.json`, gitignored):

```json
[
  {
    "key": "mpk-old-secret",
    "key_id": "mpk_orders_r3",
    "tenant": "orders-runtime",
    "org_id": "org-acme",
    "not_before": "2026-07-14T00:00:00Z",
    "expires_at": "2026-07-15T00:00:00Z"
  },
  {
    "key": "mpk-new-secret",
    "key_id": "mpk_orders_r4",
    "tenant": "orders-runtime",
    "org_id": "org-acme",
    "not_before": "2026-07-14T00:00:00Z",
    "expires_at": "2026-10-12T00:00:00Z"
  }
]
```

Behaviour:

- `Authorization: Bearer <key>` → resolves to `Identity(tenant, key_id, org_id)` and caches on `request.state`.
- Missing or unknown key → `401 {"detail": "missing bearer token"}` / `401 {"detail": "invalid api key"}`.
- A key before `not_before` or at/after `expires_at` is rejected as an invalid key. Both fields are optional for legacy files but timezone-aware when present.
- `/v1/health` and `/v1/ready` are left open so liveness/readiness probes work without keys.
- `/v1/models` and `/v1/chat/completions` require a valid key when auth is on.
- Every span (model.acquire, chat.generate, chat.stream) carries `prometa.tenant=<name>` and `prometa.key_id=<redacted>` — Prometa can route signals per tenant out of the box.
- Governed routing requires `org_id` on every auth-key record. With auth off,
  local development must set `MODEL_ROUTING_EXPECTED_ORG_ID` to the signed
  policy organization.

The keys file is the seam where Prometa's control plane hands a generated set
to tenant automation. Rotation remains tenant-controlled:

1. Mount a candidate containing the current key plus the new key. Give the old
   key a short `expires_at` overlap and the new key its full validity window.
2. Call `POST /v1/admin/auth-keys:reload` with the current key. The engine
   parses the complete file before one atomic swap and refuses a candidate
   that does not retain the credential invoking reload.
3. Distribute the new key to callers. The old key stops authenticating at its
   declared expiry without another engine restart.

`GET /v1/admin/auth-keys` and the reload response expose only key IDs,
bindings, validity metadata, counts, and the file digest. Secret values are
never returned or logged; legacy records without `key_id` use an irreversible
SHA-256 fingerprint in status output. A malformed, oversized, missing, or
lockout-causing candidate preserves the previous in-memory key set.

## In-process TLS and mTLS

`python -m inference_engine.server` is the container entrypoint and a
fail-closed launcher around uvicorn. It validates listener configuration
strictly **before** binding, so a half-configured deployment never comes up
quietly serving plaintext. It reads the four `INFERENCE_ENGINE_SERVER_TLS_*`
vars from the configuration table above (abbreviated to `…` below) plus `HOST`,
`PORT`, and `LOG_LEVEL`:

| condition | result |
|---|---|
| certificate set without key, or key without certificate | exit 2, `server_tls_configuration_invalid` |
| client CA or `…REQUIRE_CLIENT_CERTIFICATE=true` with no server certificate | exit 2, `server_tls_configuration_invalid` |
| `…REQUIRE_CLIENT_CERTIFICATE=true` with no client CA | exit 2, `server_tls_client_ca_required` |
| unreadable or malformed PEM material | exit 2, `server_tls_material_invalid` |
| `HOST` / `PORT` outside their allowed ranges | exit 2, `server_listen_address_invalid` |
| `LOG_LEVEL` not a known level | exit 2, `server_log_level_invalid` |
| a boolean env var that is not exactly `true` / `false` | exit 2, `server_tls_configuration_invalid` |

Failures print one line of JSON to stderr —
`{"type":"inference-engine.server","status":"failed","code":"…"}` — and
deliberately omit certificate paths, so the code is greppable without leaking
mount layout into logs.

With a certificate configured the context is TLS 1.2 minimum with compression
disabled; `…REQUIRE_CLIENT_CERTIFICATE=true` additionally sets `CERT_REQUIRED`
against the mounted client CA, which is the mTLS mode the OpenShift profiles
run in. With no certificate configured the launcher serves plain HTTP — the
default for local development.

`make run` bypasses this launcher and starts uvicorn directly on loopback. It
is a dev convenience, not the deployment path.

The container `HEALTHCHECK` follows the same switch: it probes `https://` when
`INFERENCE_ENGINE_SERVER_TLS_CERT_FILE` is set, `http://` otherwise. Under mTLS
the probe needs its own client credential, or it cannot authenticate to the
listener it is checking:

| var | meaning |
|---|---|
| `INFERENCE_ENGINE_PROBE_TLS_CERT_FILE` | Client certificate the healthcheck presents |
| `INFERENCE_ENGINE_PROBE_TLS_KEY_FILE`  | Matching private key |
| `INFERENCE_ENGINE_PROBE_TLS_CA_FILE`   | CA used to verify the listener; falls back to the server certificate |

The Helm chart wires all of this from `serverTls.*` — it names the container
port `https`, sets `scheme: HTTPS` on the liveness, readiness, and startup
probes, and mounts the probe client credential when `requireClientCertificate`
is on.

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
* **`TenantScheduler`** — one per engine replica. Holds per-tenant queues and dispatches into backend resources with soft per-tenant reservations, idle borrowing, wait-time aging, and per-resource caps.
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

### Proxy overhead — measured, including where it misses

The blueprint budgets what this engine may **add** to a call: p50 < 5 ms and
p99 < 20 ms of added latency on the cache-miss path, and < 10 ms of added
time-to-first-token when streaming. Nothing here had ever been measured against
those numbers, so `scripts/stress_test.py` now measures it and these are the
real results.

The harness fires the identical workload twice — once through the engine, once
straight at the OpenAI-compatible upstream the engine proxies to — interleaved
request by request so upstream drift lands on both series. Added latency is the
difference between the two series **at each percentile**; it is a
distribution-level statement, not a per-request delta, and it includes the
client's own loopback hop to the engine (which a caller pays too).

```bash
# blocking, cache-miss path
uv run python scripts/stress_test.py --base-url http://127.0.0.1:8080 \
  --requests 200 --concurrency 1 --max-tokens 1 --models llama3.2:1b \
  --baseline-url http://127.0.0.1:11434 --baseline-model llama3.2:1b --quiet
```

Measured on an M-series laptop, engine → `ollama_http` → local Ollama 0.22.1
serving `llama3.2:1b` warm, 200 request pairs at concurrency 1, distinct prompt
per request (so neither side is served from a prefix cache the other warmed):

```
blocking, max_tokens=1        upstream direct   p50= 57.55  p95= 67.99  p99= 73.11 ms
                              through engine    p50= 61.79  p95= 73.91  p99= 87.96 ms
                              ADDED             p50=  4.24  p95=  5.93  p99= 14.85 ms
                              target p50 < 5 ms  → OK
                              target p99 < 20 ms → OK

streaming, max_tokens=16      upstream direct   p50=106.77  p95=117.19  p99=126.89 ms
 (total latency)              through engine    p50=113.07  p95=125.60  p99=147.80 ms
                              ADDED             p50=  6.30  p95=  8.41  p99= 20.91 ms
                              target p50 < 5 ms  → MISS
                              target p99 < 20 ms → MISS

streaming (TTFT)              upstream direct   p50= 52.93  p95= 63.83  p99= 71.73 ms
                              through engine    p50= 80.22  p95= 95.25  p99=109.45 ms
                              ADDED             p50= 27.29  p95= 31.42  p99= 37.72 ms
                              target p50 < 10 ms → MISS
```

Read those as one sample, not a specification. At n=200 on a laptop the p50
rows move by 1–2 ms between runs and the p99 rows by considerably more (a
re-run of the blocking case measured p50 6.13 ms and p99 15.68 ms against the
4.24 / 14.85 above), which is enough to flip a row that sits near its target.
Re-measure on the hardware you care about before treating any single row as a
pass or a fail; what is stable across runs is the shape — blocking cheap,
streaming TTFT expensive by tens of milliseconds.

**The blocking path meets both targets on this sample. Streaming misses all
three, and the TTFT miss is the large one.** It is not proxy plumbing: `StreamNormalizer`
holds back the trailing 24 characters of the text stream so a vendor tag split
across two chunks (`<think>`, `<tool_call>`) is never emitted as content it
would have to retract. The first content chunk therefore cannot leave the
engine until the upstream has produced 25 characters.

That decomposition is measured, not assumed. Against the same upstream, the
point at which 24 characters have arrived is p50 **78.40 ms**, versus **55.02
ms** for the first content chunk — so of the 27.29 ms of added TTFT, about
23 ms is the holdback and roughly 2 ms is everything else the request path
does. The streaming total-latency miss is the same delay carried to the end of
the stream.

Buying the TTFT budget back means making the holdback conditional on a model
that can actually emit those tags, not shortening it globally — the guarantee
it buys (no retracted content) is worth more than 20 ms to a caller rendering
tokens. That work is not in this round; the number above is what the engine
does today.

### Judge latency and large-model guidance

LLM-as-a-judge traffic has a much tighter SLO than general chat: target p95
latency is **1-10 seconds** for judge-sized prompts with a short output cap
(typically 32-128 tokens, `temperature=0`). A multi-minute completion should be
treated as "not a viable judge model on this hardware", not as normal queue
behavior.

For `gemma4:26b`, use this checklist before routing production judge traffic:

* Prefer a quantized model that fits entirely on the accelerator. A 26B Q4
  model plus KV cache usually needs a large-memory GPU or high-end unified
  memory system; CPU-only or partial-offload runs can land in the multi-minute
  range.
* Confirm GPU/Metal/CUDA offload in the upstream runtime. If the engine is
  reaching `gemma4:26b` through `OLLAMA_HTTP_ENDPOINT`, verify the Ollama server
  is actually using the accelerator and not falling back to CPU.
* Keep judge generations small: `max_tokens=64` is a better default for rubric
  scoring than the chat default of 512.
* Keep same-model concurrency at 1 unless the backend has a real decode
  scheduler. Local `llama_cpp`/MLX adapters serialize per adapter; use vLLM or
  another continuous-batching upstream for high-QPS judge workloads.
* Measure before enabling the pilot:

```bash
uv run python scripts/stress_test.py \
  --base-url http://127.0.0.1:8080 \
  --models gemma4:26b \
  --requests 5 \
  --concurrency 1 \
  --max-tokens 64 \
  --prompt "Grade this answer with one JSON object containing score and reason."
```

The report prints p50/p95/p99 latency and output tokens/sec. If p95 is above
the judge SLO, route the judge role to a smaller/quantized model or a dedicated
GPU-backed batching runtime instead of exposing the slow model through ngrok.

## Server-side generation timeout

`CHAT_COMPLETION_TIMEOUT_SECONDS` bounds HTTP-backed chat calls (`ollama_http`
and `vllm`). The default is 120 seconds, deliberately below common public proxy
limits such as ngrok's 5-minute free-tier upstream cap. Set it lower for judge
traffic, for example `CHAT_COMPLETION_TIMEOUT_SECONDS=30`, so a slow candidate
fails clearly instead of stalling the caller's scoring run.

Timeout behavior:

* Blocking `/v1/chat/completions` returns HTTP `504` with
  `detail.type="generation_timeout"`, plus the backend, model, and configured
  timeout.
* Streaming `/v1/chat/completions` emits a terminal SSE `error` event with the
  same typed payload, then closes without a `[DONE]` trailer.
* Local blocking native calls (`llama_cpp`, MLX) are not forcibly interrupted
  mid-generation because the underlying high-level APIs do not expose a safe
  cancellation hook. Use `stream=true` when clients need fast disconnect
  cancellation on local backends.

It is a **total-elapsed** budget on both paths, which is not what
`httpx.Timeout` gives you. httpx's read timeout is per read operation, so a
response that keeps arriving — a slow model, or a stream dripping a token a
second — resets it forever and never trips it. The blocking path is capped by
the route (`asyncio.wait_for`); the streaming path is capped inside the HTTP
adapters, which bound every await on the upstream by what is *left* of the
budget and start no further read once it is gone.

That matters beyond latency, because the scheduler lease is held for the whole
call. `ollama_http` takes the default dispatch cap of
`SCHEDULER_RESOURCE_MAX_IN_FLIGHT=1`, so while one call holds the slot no other
request for that model dispatches: each waits in the tenant queue and is
refused with a 503 `tenant_queue_timeout` once it has waited
`SCHEDULER_QUEUE_TIMEOUT_SECONDS` (30 s by default). Bounding the stream bounds
how long that costs — it does not change how the queue behaves while it lasts,
and it does not address the separate leak in **Known issue: a leaked
`ollama_http` dispatch slot** below.

A streaming call that reaches the budget therefore ends with the same terminal
SSE `error` event as any other timeout, mid-stream, instead of running on. If
your workload legitimately streams for longer than 120 s, raise
`CHAT_COMPLETION_TIMEOUT_SECONDS` — it now means what it says.

### Known issue: a leaked `ollama_http` dispatch slot

**Not fixed, and not caused by the resilience work above** — it reproduces on a
pristine tree with none of that code present, which two of us confirmed
separately before deferring it. Recorded here so the next person does not
rediscover it from scratch.

Observed signature: a single **streaming** request ends up holding an
`ollama_http` dispatch slot indefinitely. With that backend's resource cap of 1
(above), every subsequent request for the same model queues behind it and is
then refused with `tenant_queue_timeout`, on an otherwise idle process — the
model looks wedged while nothing is generating. Look for it in
`inference_engine_scheduler_in_flight_by_resource{resource="ollama_http:<model>"}`
staying at 1 with no generation running; only a restart clears it.

What is *not* yet established: the trigger, and whether the lease is leaked
before the response body starts (the route acquires the slot in
`chat_completions()` before returning the `EventSourceResponse`, while the
release lives in the streaming generator's `finally`, so a stream whose body is
never iterated has no path to the release) or somewhere later. Confirm the
trigger before writing a fix — that reading is a hypothesis, not a diagnosis.

## Upstream resilience — retry the same model, then cool the deployment down

Resilience used to be fallback-only: any exception out of an adapter sent the
request to the *next model in the signed route*. A single transient 503 from a
vLLM upstream therefore changed which model answered the caller — a silent
model substitution caused by a dropped packet. Two mechanisms sit in front of
fallback now, both scoped to HTTP-backed upstreams (`vllm`, `ollama_http`,
`openrouter`); in-process backends have no transport to retry.

### Bounded retries, below the fallback loop

An idempotent upstream call is reissued on the **same deployment** before
candidate selection is allowed to consider a different model. Two attempts by
default, exponential backoff with equal jitter, `Retry-After` honoured when the
upstream sends one.

Two rules are load-bearing:

* **Never retry after the first token has been streamed.** A partial stream
  cannot be resumed upstream, and rerunning the prompt would splice two
  different completions into one answer. The adapter tracks whether a chunk
  reached the consumer and refuses the retry from that moment on — which is the
  same cut-off the streaming fallback path already used.
* **Classify before spending the deadline.** 408/425/429/500/502/503/504/529
  and transport failures that produced no response are retried. A 400, 404 or
  422 is the request being wrong; resending it buys the same rejection at the
  cost of the caller's remaining time. A *read* timeout is not retried either —
  the upstream had the request when the clock ran out.

Every attempt and every backoff sleep is drawn from one
`CHAT_COMPLETION_TIMEOUT_SECONDS` budget — on the streaming path as well as the
blocking one. Each await on the upstream is bounded by what is *left* of the
budget, and no attempt, sleep or upstream read is *started* once the budget is
spent. The request still ends a hair past it, because unwinding the timeout and
mapping the error take real time: against a mock upstream that would have
streamed for 10 s, a 300 ms budget ended the stream at 301–303 ms over five
runs. Retries cannot carry a request materially past the operator's timeout;
they do not promise ending on the microsecond. When the remaining budget cannot
fit the backoff plus a usable attempt, the retry is refused and the real
upstream error surfaces instead of a timeout produced by our own optimism.

Retries do not disturb the two things that now ride on the same request: policy
enforcement (and therefore the budget reservation) runs once per request above
the adapter, and the usage ledger is a per-request record — so a retried
request still places one reservation and emits one invoice line.

### Per-deployment cooldown and breaker

Consecutive transient failures on one deployment open it for a cooldown, after
which exactly one half-open probe is admitted: success closes it, failure
re-opens it with a doubled cooldown up to the configured ceiling. A
*deployment* is `(backend, endpoint, model_id)` — one Ollama host serving ten
models does not lose all ten because one of them keeps OOMing.

This is wired into the registry probe layer rather than standing beside it. The
probes already answer "can this upstream serve right now?", cache it with a
TTL, and feed the probe-aware resolver that picks a request's candidate, so:

* an open deployment reports `upstream_cooldown` from the probe and drops out
  of candidate selection without an HTTP round trip;
* the vLLM probe's own `/v1/models` check **is** the half-open probe — a cheap
  GET rather than a full generation;
* only failures the retry classifier calls transient count — a 400, 404 or 422
  leaves the deployment's history untouched, so an upstream *rejecting* a
  request can never cool it down for other tenants. (A request that makes the
  upstream itself fail with a 500 does count; from here that is
  indistinguishable from any other 500.)

**The threshold counts logical calls, not upstream attempts.** Read
`UPSTREAM_BREAKER_FAILURE_THRESHOLD=3` as "three consecutive *requests* that
failed after exhausting their own retries", not "three failed round trips": the
adapter retries first and reports one verdict per request. With the shipped
defaults a deployment therefore absorbs up to **six** failed upstream requests,
across three callers, before it opens. That is deliberate — counting attempts
would open the breaker twice as fast for a blip the retry already absorbed —
but it is not what the number looks like, so size it accordingly.

**A generation timeout is not a health signal, and that is deliberate.**
`CHAT_COMPLETION_TIMEOUT_SECONDS` bounds elapsed time, and elapsed time cannot
tell a wedged deployment apart from a healthy one producing a long answer — a
4k-token completion and a hung socket both end the same way. An earlier cut of
this classifier counted every timeout; once the streaming path grew a deadline
of its own, three slow-but-successful streams were enough to open the breaker
and withdraw a working model from every tenant. So a deadline expiry and
httpx's read/write/pool timeouts are all discounted. A **connect** timeout
still counts: it proves the upstream never accepted the call.

The limit that leaves, stated because it is the price of the rule: **a
deployment that accepts connections and then goes silent is never opened from
the hot path.** For vLLM something else covers it — `VLLMUpstreamProbe` GETs
`/v1/models` under `VLLM_UPSTREAM_PROBE_TIMEOUT_SECONDS` (1 s), where an
unanswered call really is evidence about the server, and its timeouts do feed
the breaker. `ollama_http` has no such probe, so a silently wedged Ollama
deployment stays in candidate selection and every request to it spends its own
full budget before failing. Fixing that needs a bounded liveness probe for the
format — not counting generation timeouts again.

State is exported both ways an operator looks for it: `chat.generate` and
`chat.stream` spans carry `llm.upstream.deployment` and
`llm.upstream.breaker.state`, and `/v1/metrics` exports
`inference_engine_upstream_breaker_state{backend,endpoint,model}` (0=closed,
1=half_open, 2=open) alongside `..._consecutive_failures`,
`..._cooldown_seconds`, `..._opened_total`, `..._half_open_probes_total`,
`..._skipped_total`, and `inference_engine_upstream_retries_total{backend}`.

### Settings

| Variable | Default | Meaning |
| --- | --- | --- |
| `UPSTREAM_RETRY_ENABLED` | `true` | Off restores fallback-only resilience. |
| `UPSTREAM_RETRY_MAX_ATTEMPTS` | `2` | Total attempts per upstream call, retries included. |
| `UPSTREAM_RETRY_BASE_DELAY_SECONDS` | `0.1` | First backoff before jitter; doubles per attempt. |
| `UPSTREAM_RETRY_MAX_DELAY_SECONDS` | `2.0` | Ceiling for the computed backoff. A `Retry-After` above it is still honoured — or refused if it does not fit the deadline. |
| `UPSTREAM_BREAKER_ENABLED` | `true` | Off leaves the deployment always eligible. |
| `UPSTREAM_BREAKER_FAILURE_THRESHOLD` | `3` | Consecutive failed *requests* (each having already exhausted its own retries) before a deployment opens — six failed upstream round trips at the default attempt cap. |
| `UPSTREAM_BREAKER_COOLDOWN_SECONDS` | `15` | Cooldown on the first open. |
| `UPSTREAM_BREAKER_MAX_COOLDOWN_SECONDS` | `120` | Ceiling as the cooldown doubles per failed half-open probe. |

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

Five adapters implement the same `InferenceAdapter` ABC; `ModelManager` picks
one per descriptor, so a single request set can span all of them concurrently.

| backend       | runs where | format / transport | strengths | newer-arch coverage |
|---------------|-----------|--------------------|-----------|---------------------|
| `llama_cpp`   | in-process | GGUF               | universal hardware reach, GGUF lingua franca, fast warm-hits, GBNF-enforced structured outputs, prefix cache | bounded by the wheel version |
| `mlx`         | in-process | MLX safetensors    | Apple Silicon native, Metal unified memory, multi-slot prefix cache, often ahead on new architectures | depends on mlx-lm release |
| `ollama_http` | HTTP       | Ollama server      | escape hatch for GGUF architectures the bundled wheel can't load; consulted only after local llama.cpp | tracks Ollama's ggml fork |
| `vllm`        | HTTP       | OpenAI-compatible  | continuous batching, guided decoding (xgrammar), GPU-pinned upstreams, logprobs | tracks the vLLM deployment |
| `openrouter`  | HTTP       | OpenAI-compatible  | large open-weight models beyond local memory; gated to open-weight, non-proprietary, `> OPENROUTER_MIN_PARAMETER_COUNT_B` | tracks the OpenRouter catalog |

Only `llama_cpp` and `mlx` load weights into this process; the other three are
HTTP clients to a runtime you operate (or, for OpenRouter, a provider you pay).
That is why `/tokenize` and `/detokenize` return `501` on the HTTP-proxy
backends — there is no local tokenizer to call. `/v1/embeddings` is narrower
still: `llama_cpp` is the only backend that serves it, and every other one —
including in-process MLX — returns `501 embeddings not supported by <backend>`.

The Ollama store you already have includes architectures newer than the bundled
`llama-cpp-python>=0.3.22`: `mistral3` (ministral-3:*), `gemma4`, `qwen3.6`,
`nemotron3`. These fail to load via `llama_cpp` until the wheel ships support.
Two ways around it: point `OLLAMA_HTTP_ENDPOINT` at an Ollama server, which
serves them over the same routes via the `ollama_http` adapter; or, if
`mlx-community` publishes an MLX conversion (most do), grab it with
`make download-mlx-model MODEL=mlx-community/<repo>` and it serves through the
`mlx` adapter.

Standard `llama` family models work on either local backend today.

## Roadmap (next phases from the guide)

1. **Phase 2 — service features.** ✅ Per-key bearer auth + tenant attribution · ✅ managed key rotation via `/v1/admin/auth-keys:reload` · ✅ streaming request cancellation · ✅ prompt-template overrides via `/v1/completions` · ✅ rate limiting (per-tenant scheduler admission + signed-policy RPM, `process-replica` or exact `deployment-shared` via Redis/Sentinel).
2. **Phase 3 — engine behaviour.** ✅ Multi-model routing · ✅ per-key load dedup + parallel cold loads · ✅ llama.cpp prefix cache (9.86×) · ✅ MLX multi-slot LRU prefix cache (~22× hit-rate lift) · ✅ token-precise cache observability on **both** backends · ✅ dynamic batching for `/v1/embeddings` (coalescer + capability fallback) · ✅ continuous chat batching via vLLM-as-subprocess (`VLLMAdapter` + `docker-compose.vllm.yml` overlay).
3. **Phase 4 — Prometa integration.** ✅ Real OTel exporter (OTLP/gRPC + Jaeger compose) · ✅ LLM-as-a-Judge eval harness · ✅ auto-judge attached to chat completions · ✅ server-side auto-eval policy (Prometa-authoritative) · ✅ tool-call audit logs (`gen_ai.tool_*` events with payload truncation).
4. **Adapter coverage.** ✅ MLX-LM (Apple Silicon native) · ✅ vLLM for GPU-server workloads · ✅ Ollama-HTTP for architectures ahead of the llama.cpp wheel · ✅ OpenRouter for large open-weight models · SGLang · TensorRT-LLM for NVIDIA optimization.
5. **Phase 5 — wire standardization.** ✅ `stream_options.include_usage` streaming usage trailer · ✅ OpenAI `error` envelope + `x-request-id` + `x-ratelimit-*` · ✅ `max_completion_tokens` · ✅ grammar-enforced Structured Outputs (`json_schema`) · ✅ logprobs, penalties, `logit_bias` · ✅ `usage.prompt_tokens_details.cached_tokens` · ✅ `/tokenize` + `/detokenize` · ✅ OTel GenAI semconv attrs + TTFT/TPOT metrics · `/v1/responses` (stateful; deferred) · batch/files/audio/moderations (deferred until a consumer needs them).
6. **Phase 6 — model plane.** ✅ Signed Ed25519 routing policy, verified locally, with atomic LKG activation and offline lease · ✅ per-route input/output token, cost, RPM, TPM, and window spend ceilings with fail-closed pricing and reserve-then-settle budgeting · ✅ exact `deployment-shared` rate and budget windows via Redis/Sentinel · ✅ asynchronous payload-free observation reporting (v1 + v2) · ✅ certified workload surface (`orchestra-model-plane-workload-v1`) · ✅ fail-closed TLS/mTLS listener · ✅ signed UBI images, SBOMs, and a standalone Helm chart with production + SNO-trial profiles · rerank, standalone eval, and chat-attached auto-eval under signed routing (today they fail closed in governed mode) · OpenShift lifecycle, backup/recovery, multi-replica load, and SLO certification.
7. **Upstream resilience.** ✅ Bounded retries on the same deployment ahead of model fallback (classified, jittered, `Retry-After`-aware, drawn from the request deadline, never after a stream's first token) · ✅ per-deployment cooldown + half-open breaker wired into the registry probe layer, exported as a Prometheus gauge and a span attribute · ✅ measured proxy overhead published above, including the streaming TTFT miss and its cause · a bounded liveness probe for `ollama_http`, without which a silently wedged Ollama deployment is never withdrawn · the leaked `ollama_http` dispatch slot recorded as a known issue above.

## Constraints (as instructed)

- This service **only reads** from `auto-ml/ollama-models/`. It never writes there.
- All edits stay inside `llm_inference_engine_v1/`.
