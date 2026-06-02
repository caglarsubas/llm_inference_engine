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
https://abc123.ngrok.app              ← your public host (changes per session unless reserved)
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
make share            # ngrok  (default)
make share-cf         # Cloudflare Tunnel (no account needed)
make share PORT=8090  # tunnel a different port, e.g. the docker LB
```

Under the hood that runs `scripts/share_endpoint.sh`. Direct use:

```bash
scripts/share_endpoint.sh --provider ngrok
scripts/share_endpoint.sh --provider cloudflared --port 8090
scripts/share_endpoint.sh --domain my-reserved.ngrok.app   # stable URL
```

**Prerequisites**

- ngrok: `brew install ngrok`, then once: `ngrok config add-authtoken <token>`
  (free token from [dashboard.ngrok.com](https://dashboard.ngrok.com)).
- cloudflared: `brew install cloudflared` (quick `*.trycloudflare.com` tunnels
  need no login).

The free ngrok URL changes every restart. Reserve a domain on the ngrok
dashboard and pass `--domain` for a URL that survives restarts — recommended
when you've pasted it into Prometa.

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
    {"id": "gemma4:27b", "reason": "unsupported_arch",
     "detail": "llama.cpp can't open this GGUF", "backend": "llama_cpp"}
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
  reason. If a model you expect is here, it won't work — pick one from `data`.

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
| `messages`        | array (**required**)       | —       | `{role, content}`; roles: `system`, `user`, `assistant`, `tool`. |
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

If you used a **free ngrok URL**, it changes on every restart — update the
ENGINE URL field after each restart, or reserve a domain (`--domain`) for a
stable value.

---

## 12. Errors & troubleshooting

| Symptom                                            | Cause / fix                                                                 |
|----------------------------------------------------|----------------------------------------------------------------------------|
| `404 {"detail": "model not found: '…'"}`           | The `model` id isn't servable. Run `GET /v1/models`; pick an id from `data`. |
| `401 missing bearer token` / `invalid api key`     | Auth is on. Send `Authorization: Bearer <key>` with a valid key.           |
| `400` with `context_length_exceeded`               | Prompt + `max_tokens` overran the model window. Shorten input or lower `max_tokens`. |
| `501 embeddings not supported by mlx backend`      | That model can't embed. Use a `llama_cpp`/embedding model for `/v1/embeddings` & `/v1/rerank`. |
| `400` "blocking is incompatible with stream=true"  | `auto_eval.mode:"blocking"` can't stream. Use `mode:"background"` or `stream:false`. |
| Browser shows an **ngrok warning page** for GETs   | Free ngrok interstitial. Add header `ngrok-skip-browser-warning: true`, or just use POST/API clients (unaffected). |
| Connection refused / 502 from the tunnel           | The engine isn't running or you tunneled the wrong port. `curl 127.0.0.1:8080/v1/health` locally; tunnel the LB port with `PORT=` if using docker. |
| Tunnel URL works then dies                          | Free tunnels are session-scoped. Keep the `make share` process running; reserve a domain for persistence. |

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
