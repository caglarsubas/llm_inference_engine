# Public Endpoint — Usage Guide

This engine speaks the **OpenAI HTTP API**. Once it's exposed on a public URL
(via ngrok or Cloudflare Tunnel), any OpenAI-compatible client — the OpenAI
SDKs, LangChain, `curl`, or the **Prometa platform** — can talk to it by
pointing `base_url` at that URL.

This document is meant to be handed to a consumer of the endpoint. It covers
the base URL, authentication, how to list models, and how to `POST` to every
route for any model, plus the configuration knobs that change behaviour.

- [1. Anatomy of the endpoint](#1-anatomy-of-the-endpoint)
- [2. Get a public URL (operator)](#2-get-a-public-url-operator)
- [3. Authentication](#3-authentication)
- [4. Discover models](#4-discover-models)
- [5. Chat completions](#5-chat-completions)
- [6. Legacy completions](#6-legacy-completions)
- [7. Embeddings](#7-embeddings)
- [8. Rerank](#8-rerank)
- [9. Health & metrics](#9-health--metrics)
- [10. Use it from an SDK](#10-use-it-from-an-sdk)
- [11. Wire it into Prometa](#11-wire-it-into-prometa)
- [12. Errors & troubleshooting](#12-errors--troubleshooting)
- [13. Security checklist](#13-security-checklist)

---

## 1. Anatomy of the endpoint

```
https://my-name.ngrok-free.dev        ← your public host (reserved domain = stable; ephemeral = per-session)
                       /v1            ← API version prefix (always present)
                          /chat/completions   ← the route you POST to
```

So the **base URL** you give clients is the host **plus `/v1`**:

```
https://abc123.ngrok.app/v1
```

Everything below uses `$BASE` for that value:

```bash
export BASE="https://abc123.ngrok.app/v1"
export KEY="sk-..."   # only if auth is enabled — see §3
```

| Method | Path                        | Purpose                                              |
|--------|-----------------------------|------------------------------------------------------|
| GET    | `/v1/health`                | Liveness, loaded models, memory budget (no auth)     |
| GET    | `/v1/models`                | List every servable model + why others are unavailable |
| GET    | `/v1/models/{id}`           | Details for one model id                             |
| POST   | `/v1/chat/completions`      | Chat (blocking or SSE streaming), tools, JSON mode   |
| POST   | `/v1/completions`           | Raw-prompt completion (bypasses chat templating)     |
| POST   | `/v1/embeddings`            | Vector embeddings                                    |
| POST   | `/v1/rerank`                | Relevance ranking (query + documents)               |
| GET    | `/v1/metrics`               | Prometheus-format scrape                             |
| GET    | `/v1/evals/rubrics`         | List LLM-as-judge rubrics                            |
| POST   | `/v1/evals/run`             | Score a candidate answer with a rubric              |

---

## 2. Get a public URL (operator)

> Skip this section if someone already handed you a URL.

The engine binds to loopback (`127.0.0.1:8080`) by default — not reachable from
the internet. A tunnel terminates TLS in the cloud and forwards to that
loopback port, so you don't need a public IP or an inbound firewall rule.

**One command** (reads `HOST`/`PORT` from `.env`, prints the URL + copy-paste
examples):

```bash
make share                                       # ngrok  (default)
make share-cf                                    # Cloudflare Tunnel (no account needed)
make share PORT=8090                             # tunnel a different port, e.g. the docker LB
make share NGROK_DOMAIN=my-name.ngrok-free.dev   # stable, reserved ngrok domain
```

Under the hood that runs `scripts/share_endpoint.sh`. Direct use:

```bash
scripts/share_endpoint.sh --provider ngrok
scripts/share_endpoint.sh --provider cloudflared --port 8090
scripts/share_endpoint.sh --domain my-name.ngrok-free.dev   # stable URL
```

**Prerequisites**

- ngrok: `brew install ngrok`, then once: `ngrok config add-authtoken <token>`
  (free token from [dashboard.ngrok.com](https://dashboard.ngrok.com)).
- cloudflared: `brew install cloudflared` (quick `*.trycloudflare.com` tunnels
  need no login).

**Stable vs. ephemeral URL**

- An **ephemeral** ngrok run (no `--domain`) gives a URL that can change between
  sessions — fine for a quick test, annoying once it's pasted into Prometa.
- The **ngrok free plan includes one reserved static domain** (shown under
  *Domains* in the dashboard, current format `my-name.ngrok-free.dev`). Pass it
  with `make share NGROK_DOMAIN=my-name.ngrok-free.dev` (or `--domain`) for a
  URL that survives tunnel restarts.
- Use the domain **exactly** as the dashboard shows it. Passing any other
  subdomain on the free plan fails with `ERR_NGROK_313` ("only paid plans may
  create custom subdomains").

**Keeping it up (always-on)**

`make share` runs in the foreground and the tunnel stops when you do (or on
logout / reboot). For a **truly always-on** endpoint, install the tunnel as a
`launchd` user agent — it auto-starts on login and respawns on crash/reboot,
exactly like the engine service:

```bash
make share-install NGROK_DOMAIN=my-name.ngrok-free.dev   # install + start
make share-status                                        # state, PID, live URL
make share-logs                                          # tail tunnel logs
make share-restart                                       # re-read config / restart
make share-uninstall                                     # remove the agent
```

A reserved domain is **required** here (an ephemeral URL would change on every
respawn). Port defaults to `PORT` from `.env`/`8080`; override with
`PORT=8090`. Pair this with the engine's own `make native-install` agent and
both the engine and its public URL come back together after a reboot.

---

## 3. Authentication

Auth is **optional** and controlled server-side by `AUTH_ENABLED`.

- **Auth off** (default for local dev): send no credentials. Every request is
  attributed to the `anonymous` tenant.
- **Auth on**: send a bearer token on every request.

```bash
curl -s "$BASE/models" -H "Authorization: Bearer $KEY"
```

Keys live in `.auth_keys.json` on the server, mapping each key to a tenant:

```json
[
  {"key": "sk-prometa-prod", "tenant": "prometa"},
  {"key": "sk-dev-1",        "tenant": "dev"}
]
```

Behaviour with auth on:

| Situation                       | Response                                        |
|---------------------------------|-------------------------------------------------|
| Valid `Authorization: Bearer …` | request proceeds, tagged with the key's tenant  |
| Missing header                  | `401 {"detail": "missing bearer token"}`        |
| Unknown key                     | `401 {"detail": "invalid api key"}`             |
| `/v1/health`                    | always open (so liveness probes work)           |

> **Anyone with a public URL and auth OFF can run your models for free.** Turn
> auth on before sharing the URL beyond your own machine. See §13.

---

## 4. Discover models

You don't guess model names — you list them. The `model` field in every
request below is an **id from this list**.

```bash
curl -s "$BASE/models" | jq '.data[].id'
```

```json
{
  "object": "list",
  "data": [
    {
      "id": "llama3.2:3b",
      "backend": "llama_cpp",
      "format": "gguf",
      "size_bytes": 2019377440,
      "reasoning": false,
      "tool_calling_mode": "native"
    },
    {
      "id": "Llama-3.2-1B-Instruct-4bit:mlx",
      "backend": "mlx",
      "format": "mlx",
      "tool_calling_mode": "native"
    }
  ],
  "unavailable": [
    {
      "id": "fakeshield-22b:vllm",
      "available": false,
      "upstream_reachable": false,
      "availability_status": "upstream_unreachable",
      "reason": "upstream_unreachable",
      "detail": "connection refused",
      "backend": "vllm",
      "format": "vllm",
      "endpoint": "http://vllm-fakeshield-22b:8000",
      "upstream_model_id": "zhipeixu/fakeshield-v1-22b",
      "supports_strict_image_json": false,
      "strict_image_json_status": "pending_smoke"
    },
    {
      "id": "sida-13b:vllm",
      "available": false,
      "upstream_reachable": false,
      "availability_status": "downloaded_but_not_served",
      "reason": "downloaded_but_not_served",
      "backend": "vllm",
      "format": "vllm",
      "endpoint": "http://vllm-sida-13b:8000",
      "upstream_model_id": "saberzl/SIDA-13B",
      "download_status": "downloaded",
      "local_snapshot_path": "/Users/example/.cache/inference_engine/hf-vlm/saberzl--SIDA-13B",
      "supports_strict_image_json": false,
      "strict_image_json_status": "pending_smoke"
    }
  ]
}
```

What the fields tell you:

- **`id`** — pass this verbatim as `"model"`. Ids often carry a `:tag`
  (`llama3.2:3b`) or a `:backend` suffix (`...:mlx`). URL-encode the `/` or `:`
  only on the `GET /v1/models/{id}` path, not in JSON bodies.
- **`backend` / `format`** — which runtime serves it (`llama_cpp` GGUF, `mlx`
  Apple-Silicon, `vllm` GPU, `ollama_http` fallback). You call all of them the
  same way; only latency/throughput differ.
- **`reasoning` / `thinking`** — the model emits private chain-of-thought; the
  engine strips it into a separate `reasoning_content` channel so it never
  leaks into `content`.
- **`tool_calling_mode`** — `native` means you can pass `tools`; `unsupported`
  means the model has no tool plumbing.
- **`unavailable[]`** — models the engine knows about but can't serve, with the
  reason and availability state. vLLM/OpenRouter entries can also include the
  same benchmark metadata as `data[]`, so clients can distinguish a declared
  but offline endpoint (`available=false`, `upstream_reachable=false`) from a
  reachable endpoint that still needs strict JSON validation. A vLLM candidate
  with `availability_status="downloaded_but_not_served"` has local checkpoint
  files under `local_snapshot_path`, but no configured upstream has proved it
  is callable yet. If a model you expect is here, it won't work yet — pick one
  from `data`.

The set of models is whatever the operator has installed (Ollama GGUFs, MLX
dirs, vLLM endpoints). To add models, the operator drops them in the model
store; nothing changes for you except a new id appearing in this list.

---

## 5. Chat completions

`POST /v1/chat/completions` — the main route.

### Minimal request

```bash
curl -s "$BASE/chat/completions" \
  -H 'content-type: application/json' \
  -d '{
    "model": "llama3.2:3b",
    "messages": [{"role": "user", "content": "Say hi in one word."}],
    "max_tokens": 16
  }'
```

```json
{
  "id": "chatcmpl-…",
  "object": "chat.completion",
  "model": "llama3.2:3b",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "Hi"},
    "finish_reason": "stop"
  }],
  "usage": {"prompt_tokens": 13, "completion_tokens": 2, "total_tokens": 15}
}
```

**Switching models is just changing `"model"`** to any id from `/v1/models` —
the request shape is identical across GGUF, MLX, vLLM, and Ollama-fallback
backends.

### Request parameters

| Field             | Type                       | Default | Notes                                                            |
|-------------------|----------------------------|---------|------------------------------------------------------------------|
| `model`           | string (**required**)      | —       | An id from `/v1/models`.                                         |
| `messages`        | array (**required**)       | —       | `{role, content}`; roles: `system`, `user`, `assistant`, `tool`. `content` can be a string or OpenAI-style `text`/`image_url` parts for vision backends. |
| `max_tokens`      | int                        | `512`   | Upper bound on generated tokens.                                 |
| `temperature`     | float `0.0–2.0`            | `0.7`   | Higher = more random. Use `0` for deterministic-ish output.      |
| `top_p`           | float `0.0–1.0`            | `0.95`  | Nucleus sampling.                                                |
| `top_k`           | int `≥0`                   | `40`    | Top-k sampling.                                                  |
| `stop`            | string or array of strings | `null`  | Stop sequences.                                                  |
| `seed`            | int                        | `null`  | Reproducible sampling where the backend supports it.            |
| `stream`          | bool                       | `false` | `true` → Server-Sent Events (see below).                        |
| `response_format` | object                     | `null`  | `{"type": "json_object"}` to force JSON output.                 |
| `tools`           | array                      | `null`  | OpenAI tool definitions; only for `tool_calling_mode: native`.   |
| `tool_choice`     | string or object           | `null`  | `"auto"`, `"none"`, or a specific function.                    |
| `auto_eval`       | object                     | `null`  | Inline LLM-as-judge scoring (see below).                        |

### Vision / multimodal content

For VLM backends, send OpenAI-style content parts. The engine accepts the
request shape and forwards it to HTTP backends such as vLLM/Ollama fallback;
actual image understanding depends on the selected model and runtime.

```bash
curl -s "$BASE/chat/completions" \
  -H 'content-type: application/json' \
  -d '{
    "model": "qwen3-vl-8b:vllm",
    "temperature": 0,
    "response_format": {"type": "json_object"},
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "Assess this vehicle photo. Return JSON."},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,<base64>", "detail": "low"}}
      ]
    }]
  }'
```

### Streaming (SSE)

Set `"stream": true`. The response is `text/event-stream`: a sequence of
`data: {chunk}` lines ending with `data: [DONE]`.

```bash
curl -N "$BASE/chat/completions" \
  -H 'content-type: application/json' \
  -d '{
    "model": "llama3.2:3b",
    "messages": [{"role":"user","content":"Count to 5."}],
    "stream": true,
    "max_tokens": 64
  }'
```

Each chunk: `{"choices":[{"delta":{"content":"..."}}]}`. Reasoning models also
emit `delta.reasoning_content` (render it separately or drop it). Use `-N` with
curl to disable buffering.

### JSON mode

```bash
curl -s "$BASE/chat/completions" \
  -H 'content-type: application/json' \
  -d '{
    "model": "llama3.2:3b",
    "response_format": {"type": "json_object"},
    "messages": [
      {"role":"system","content":"Reply with JSON only."},
      {"role":"user","content":"name and age of a fictional person"}
    ]
  }'
```

### Tool calling

Only for models with `tool_calling_mode: "native"`.

```bash
curl -s "$BASE/chat/completions" \
  -H 'content-type: application/json' \
  -d '{
    "model": "llama3.2:3b",
    "messages": [{"role":"user","content":"What is the weather in Paris?"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get current weather for a city",
        "parameters": {
          "type": "object",
          "properties": {"city": {"type": "string"}},
          "required": ["city"]
        }
      }
    }],
    "tool_choice": "auto"
  }'
```

When the model decides to call a tool, `finish_reason` is `tool_calls` and
`message.tool_calls[]` carries `{id, function: {name, arguments}}` where
`arguments` is a JSON-encoded **string**. Execute the tool yourself, then send
the result back as a `{"role":"tool","tool_call_id":"…","content":"…"}` message
in the next request.

### Reasoning models

For models flagged `reasoning: true`, the private chain-of-thought is returned
in `message.reasoning_content` (non-stream) or `delta.reasoning_content`
(stream), kept out of `content`. Treat `content` as the user-facing answer.

### Inline evaluation (`auto_eval`)

Attach an LLM-as-judge pass to a completion. `mode: "blocking"` (non-stream
only) returns verdicts inline on `response.evals`; `mode: "background"` returns
immediately and scores asynchronously.

```bash
curl -s "$BASE/chat/completions" \
  -H 'content-type: application/json' \
  -d '{
    "model": "llama3.2:3b",
    "messages": [{"role":"user","content":"Is 17 prime?"}],
    "auto_eval": {"rubrics": ["correctness"], "expected": "Yes, 17 is prime.", "mode": "blocking"}
  }'
```

> Note: the server can enforce its own auto-eval policy per tenant/model, which
> overrides the request's `auto_eval`. List rubrics at `/v1/evals/rubrics`.

---

## 6. Legacy completions

`POST /v1/completions` — sends the prompt **verbatim**, no chat template. Use
for base models, custom prompt formats, or raw text-in/text-out. Streaming is
not implemented here (use chat streaming).

```bash
curl -s "$BASE/completions" \
  -H 'content-type: application/json' \
  -d '{
    "model": "llama3.2:3b",
    "prompt": "The capital of France is",
    "max_tokens": 8
  }'
```

`prompt` may be an array of strings; choices come back ordered by index.
Default `max_tokens` here is `128`.

---

## 7. Embeddings

`POST /v1/embeddings` — OpenAI-compatible vectors for RAG/retrieval. Served by
`llama_cpp` backends; MLX models return `501`. For quality retrieval the
operator should load a real embedding model (`bge`, `nomic`, `e5`).

```bash
curl -s "$BASE/embeddings" \
  -H 'content-type: application/json' \
  -d '{
    "model": "llama3.2:1b",
    "input": ["first text", "second text"]
  }'
```

`input` accepts a single string or an array. Response: `data[].embedding`
float vectors in input order.

---

## 8. Rerank

`POST /v1/rerank` — Cohere/Jina-shaped relevance ranking via embedding cosine
similarity.

```bash
curl -s "$BASE/rerank" \
  -H 'content-type: application/json' \
  -d '{
    "model": "llama3.2:1b",
    "query": "programming languages",
    "documents": ["Python is popular", "The sky is blue", "Rust is fast"],
    "top_n": 2,
    "return_documents": true
  }'
```

```json
{
  "results": [
    {"index": 0, "relevance_score": 0.91, "document": "Python is popular"},
    {"index": 2, "relevance_score": 0.62, "document": "Rust is fast"}
  ]
}
```

---

## 9. Health & metrics

```bash
curl -s "$BASE/health" | jq          # no auth; loaded models + budget
curl -s "$BASE/metrics"              # Prometheus exposition format
```

Use `/v1/health` as a cheap readiness check before sending traffic — it returns
which models are currently warm in memory.

---

## 10. Use it from an SDK

Because it's the OpenAI API, point any OpenAI client at `$BASE`.

**Python (`openai`)**

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://abc123.ngrok.app/v1",
    api_key="sk-...",  # any non-empty string when auth is off
)

resp = client.chat.completions.create(
    model="llama3.2:3b",
    messages=[{"role": "user", "content": "Hello"}],
    max_tokens=32,
)
print(resp.choices[0].message.content)
```

**Node (`openai`)**

```js
import OpenAI from "openai";
const client = new OpenAI({
  baseURL: "https://abc123.ngrok.app/v1",
  apiKey: "sk-...",
});
const r = await client.chat.completions.create({
  model: "llama3.2:3b",
  messages: [{ role: "user", content: "Hello" }],
});
```

**LangChain**

```python
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(base_url="https://abc123.ngrok.app/v1", api_key="sk-...",
                 model="llama3.2:3b")
```

---

## 11. Wire it into Prometa

In Prometa: **Settings → Primary LLM Provider → Self-hosted
(llm_inference_engine)**.

| Field          | Value                                                           |
|----------------|-----------------------------------------------------------------|
| **ENGINE URL** | the public host **without** `/v1` — e.g. `https://abc123.ngrok.app` (Prometa appends the OpenAI path itself) |
| **ENGINE TOKEN** | a bearer key from `.auth_keys.json` when auth is on; leave blank when off |
| **CANDIDATE MODEL** | an id from `/v1/models`, e.g. `gemma4:26b`, `llama3.2:3b` |

Prometa then routes its judge / assist / synthetic-data calls to your engine
on that URL. Verify the wiring from Prometa's side, or directly:

```bash
curl -s https://abc123.ngrok.app/v1/models | jq '.data[].id'
```

If you used an **ephemeral ngrok URL**, it can change between sessions — update
the ENGINE URL after each restart, or use your reserved domain
(`make share NGROK_DOMAIN=…`) for a stable value you set once.

---

## 12. Errors & troubleshooting

| Symptom                                            | Cause / fix                                                                 |
|----------------------------------------------------|----------------------------------------------------------------------------|
| `404 {"detail": "model not found: '…'"}`           | The `model` id isn't servable. Run `GET /v1/models`; pick an id from `data`. |
| `401 missing bearer token` / `invalid api key`     | Auth is on. Send `Authorization: Bearer <key>` with a valid key.           |
| `400` with `context_length_exceeded`               | Prompt + `max_tokens` overran the model window. Shorten input or lower `max_tokens`. |
| `504` with `generation_timeout`                    | The HTTP-backed model exceeded `CHAT_COMPLETION_TIMEOUT_SECONDS`. Lower `max_tokens`, use a faster/quantized judge model, or raise the timeout if this is expected long-form chat. |
| `501 embeddings not supported by mlx backend`      | That model can't embed. Use a `llama_cpp`/embedding model for `/v1/embeddings` & `/v1/rerank`. |
| `400` "blocking is incompatible with stream=true"  | `auto_eval.mode:"blocking"` can't stream. Use `mode:"background"` or `stream:false`. |
| Browser shows an **ngrok warning page** for GETs   | Free ngrok interstitial. Add header `ngrok-skip-browser-warning: true`, or just use POST/API clients (unaffected). |
| `ERR_NGROK_313` / "only paid plans may create custom subdomains" | The `--domain` / `NGROK_DOMAIN` you passed isn't your account's reserved domain (often a typo). Copy it verbatim from the ngrok dashboard *Domains* page. |
| Connection refused / 502 from the tunnel           | The engine isn't running or you tunneled the wrong port. `curl 127.0.0.1:8080/v1/health` locally; tunnel the LB port with `PORT=` if using docker. |
| Tunnel URL works then dies                          | The `make share` process stopped (foreground job, logout, or reboot). Restart it; for persistence run the tunnel under `launchd`. A reserved domain keeps the URL identical across restarts. |

### Public judge endpoints

Do not expose a multi-minute judge model through a free ngrok tunnel and expect
Prometa scoring runs to complete. For judge candidates such as `gemma4:26b`,
measure single-call latency locally first:

```bash
uv run python scripts/stress_test.py \
  --models gemma4:26b \
  --requests 5 \
  --concurrency 1 \
  --max-tokens 64 \
  --prompt "Grade this answer with one JSON object containing score and reason."
```

Use the model as a judge only when p95 is in the 1-10 second range on the
actual serving hardware. If it is slower, choose a smaller/quantized judge
model, confirm GPU/Metal/CUDA offload in the upstream runtime, or serve the
model through a batching backend such as vLLM on suitable GPU hardware.

Set `CHAT_COMPLETION_TIMEOUT_SECONDS` below the tunnel/proxy cap. For pilots,
`30` seconds is a useful starting point because it records a typed timeout
before the caller's job budget or ngrok's 5-minute limit becomes the failure
mode.

Add `ngrok-skip-browser-warning` example:

```bash
curl -s "$BASE/models" -H 'ngrok-skip-browser-warning: true'
```

---

## 13. Security checklist

A public URL means the internet can reach your engine. Before sharing:

- [ ] **Turn auth on.** Set `AUTH_ENABLED=true`, create `.auth_keys.json`,
      restart. Without it, anyone with the URL runs your models for free.
- [ ] **Use a unique key per consumer** (one per tenant) so you can revoke one
      without breaking the rest. Rotation = edit the file + reload.
- [ ] **Prefer a reserved domain** over rotating free URLs so you're not
      tempted to disable auth for convenience.
- [ ] **Don't commit** `.auth_keys.json` or `.env` (already gitignored).
- [ ] **Stop the tunnel** (`Ctrl-C`) when you're done — the URL is revoked on
      exit.
- [ ] Treat the endpoint as compute you pay for: monitor `/v1/metrics` and the
      per-tenant spans for unexpected load.
