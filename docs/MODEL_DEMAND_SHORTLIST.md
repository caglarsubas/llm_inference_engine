# Model Demand Shortlist - Vehicle-Photo Reasoning

Captured from the application request on 2026-06-15 for the FraudGuard-style
vehicle-photo reasoning workflow.

This is an operator handoff, not a declaration that every candidate is already
served by this engine. Keep three states separate:

- **installed**: appears from `uv run python scripts/list_models.py` or
  `GET /v1/models`.
- **configured**: has a runtime endpoint in `.vllm_models.json`, and that
  upstream's own `/v1/models` response lists the exact `model_id`.
- **validated**: passes the same image-input and strict-JSON bakeoff harness
  as the production application.

## Immediate Local Bakeoff

The application message called out these locally available candidates. They
were confirmed on this checkout with `uv run python scripts/list_models.py`.

| Model id | Format | Size | First action |
|---|---:|---:|---|
| `nemotron3:33b` | GGUF | 25.7 GB | Run first as the strongest currently installed baseline. |
| `qwen3.6:27b` | GGUF | 16.2 GB | Run first; compare reasoning quality and JSON parse rate. |
| `gemma4:31b` | GGUF | 18.5 GB | Run first; compare latency/cost against `qwen3.6:27b`. |
| `ministral-3:14b` | GGUF | 8.5 GB | Run as medium-size latency baseline. |
| `ministral-3:8b` | GGUF | 5.6 GB | Run as workstation/edge latency baseline. |
| `ministral-3:3b` | GGUF | 2.8 GB | Run as smallest local cost floor. |

Nearby installed models that can be included if the harness has room:
`nemotron-3-nano:30b`, `gemma4:26b`, `gemma4:e4b`, `gemma4:e2b`,
`llama3.2:3b`, `llama3.2:1b`, and `Llama-3.2-1B-Instruct-4bit:mlx`.

Before treating any installed GGUF as a real VLM candidate, send one smoke
request with an image content part. Some installed tags can be useful
text/reasoning baselines while still not being the final image-native model.

## Acquisition Queue

| Priority | Model family | Specific models to inventory | Why the application wants it |
|---|---|---|---|
| P0 | Qwen3-VL | `Qwen3-VL-8B`, `32B`, `30B-A3B`, `235B-A22B` Instruct/Thinking | First candidate for visual reasoning bakeoff; likely strong vLLM/SGLang path. |
| P0 | GLM-V | `GLM-4.6V-Flash`, `GLM-4.6V`, `GLM-4.5V`, `GLM-4.1V-9B-Thinking` | Strong multimodal reasoning, grounding, document, GUI, and image reasoning; Flash is attractive for latency. |
| P0 | MiniCPM-V | `MiniCPM-V 4.5`, `MiniCPM-V 4.6` | Strong small-model option for local or edge latency testing; Ollama/llama.cpp/vLLM routes may exist. |
| P0 | InternVL | `InternVL3.5-8B`, `20B-A4B`, `241B-A28B` | High-potential open multimodal family for image reasoning and larger GPU-cluster tests. |
| P1 | Mistral 3 / Ministral 3 | `Ministral-3-8B`, `Ministral-3-14B`, `Mistral Large 3` | Open-weight multimodal line with a deployment story; local `ministral-3` variants are already present. |
| P1 | Llama 4 | `Llama 4 Scout`, `Llama 4 Maverick` | Native multimodal open-weight MoE baseline, subject to license approval. |
| P1 | Gemma 3 / Gemma vision line | `Gemma-3-12B-it`, `Gemma-3-27B-it` | Efficient single-GPU/workstation baseline for latency and cost comparison. |
| P1 | Kimi multimodal | `Kimi-VL-A3B`, `Kimi-K2.5` | Efficient MoE/native multimodal agentic reasoning; useful for long-context visual reasoning. |
| P2 | DeepSeek-VL2 | `DeepSeek-VL2-Tiny`, `DeepSeek-VL2-Small`, full `DeepSeek-VL2` | Smaller-footprint VLM benchmark baseline. |
| P2 | Molmo 2 | `Molmo 2 4B`, `8B`, `Molmo 2-0 7B` | Grounding, pointing, and video lineage; useful for localization-style explanations. |
| P2 | NVIDIA Nemotron VL | `Llama Nemotron Nano VL`, `Nemotron Nano V2 VL`, current `nemotron3:33b` | Enterprise/document-heavy VLM line; useful baseline if NVIDIA NIM is already in the stack. |
| P2 | Aya Vision | `Aya Vision 8B`, `Aya Vision 32B` | Multilingual vision-language baseline; license must be reviewed before production use. |

## Exact Provisioning IDs

Copy entries from `.vllm_models.demanded.example.json` into the deployment's
real `.vllm_models.json` only after each upstream server is running. The engine
model id is what clients pass to this service; the upstream model id is what the
vLLM/SGLang/NIM-compatible server must advertise from its own `/v1/models`.

| Priority | Engine model id | Upstream model id |
|---|---|---|
| P0 | `qwen3-vl-8b-instruct:vllm` | `Qwen/Qwen3-VL-8B-Instruct` |
| P0 | `qwen3-vl-32b-instruct:vllm` | `Qwen/Qwen3-VL-32B-Instruct` |
| P0 | `qwen3-vl-30b-a3b-instruct:vllm` | `Qwen/Qwen3-VL-30B-A3B-Instruct` |
| P0 | `qwen3-vl-235b-a22b-instruct:vllm` | `Qwen/Qwen3-VL-235B-A22B-Instruct` |
| P0 | `glm-4.6v-flash:vllm` | `zai-org/GLM-4.6V-Flash` |
| P0 | `glm-4.6v:vllm` | `zai-org/GLM-4.6V` |
| P0 | `glm-4.5v:vllm` | `zai-org/GLM-4.5V` |
| P0 | `glm-4.1v-9b-thinking:vllm` | `zai-org/GLM-4.1V-9B-Thinking` |
| P0 | `minicpm-v-4.5:vllm` | `openbmb/MiniCPM-V-4_5` |
| P0 | `minicpm-v-4.6:vllm` | `openbmb/MiniCPM-V-4.6` |
| P0 | `internvl3.5-8b:vllm` | `OpenGVLab/InternVL3_5-8B` |
| P0 | `internvl3.5-20b-a4b-preview:vllm` | `OpenGVLab/InternVL3_5-20B-A4B-Preview` |
| P0 | `internvl3.5-241b-a28b:vllm` | `OpenGVLab/InternVL3_5-241B-A28B` |
| P1 | `llama-4-scout-17b-16e-instruct:vllm` | `meta-llama/Llama-4-Scout-17B-16E-Instruct` |
| P1 | `llama-4-maverick-17b-128e-instruct:vllm` | `meta-llama/Llama-4-Maverick-17B-128E-Instruct` |
| P1 | `kimi-vl-a3b-instruct:vllm` | `moonshotai/Kimi-VL-A3B-Instruct` |
| P1 | `kimi-k2.5:vllm` | `moonshotai/Kimi-K2.5` |
| P2 | `deepseek-vl2-tiny:vllm` | `deepseek-ai/deepseek-vl2-tiny` |
| P2 | `deepseek-vl2-small:vllm` | `deepseek-ai/deepseek-vl2-small` |
| P2 | `deepseek-vl2:vllm` | `deepseek-ai/deepseek-vl2` |
| P2 | `molmo2-4b:vllm` | `allenai/Molmo2-4B` |
| P2 | `molmo2-8b:vllm` | `allenai/Molmo2-8B` |
| P2 | `molmo2-o-7b:vllm` | `allenai/Molmo2-O-7B` |
| P2 | `aya-vision-8b:vllm` | `CohereLabs/aya-vision-8b` |
| P2 | `aya-vision-32b:vllm` | `CohereLabs/aya-vision-32b` |

## Honest Rollout Gates

1. Start one upstream model server on suitable hardware.
2. Confirm that upstream returns the exact `model_id` from `GET /v1/models`.
3. Copy only that entry into the live `.vllm_models.json`.
4. Restart or reload this engine.
5. Confirm this engine lists the model under `GET /v1/models.data`, not
   `unavailable`.
6. Run the strict vehicle-image smoke:

```bash
make vlm-smoke MODEL=qwen3-vl-8b-instruct:vllm IMAGE=/path/to/vehicle.jpg
```

The smoke defaults to `max_tokens=768`. Override with `MAX_TOKENS=...` only
when the benchmark intentionally needs a different budget; 256-token local
smokes can truncate otherwise valid JSON for `gemma4:31b` and `qwen3.6:27b`.

Only models that pass that smoke should be handed to the FraudGuard benchmark
as exposed candidates.

For Docker Model Runner upstreams, use the Docker-advertised model id in
`.vllm_models.json`, for example
`huggingface.co/qwen/qwen3-vl-8b-instruct:latest` for
`qwen3-vl-8b-instruct:vllm`. The engine's `name`/`tag` stay stable for
clients, while `model_id` must match the upstream exactly.

## Benchmark Exposure Status

The current local endpoint exposes only installed local GGUF/MLX/Ollama
fallback models plus any operator-configured vLLM upstreams. The following
priority families are intentionally **not** exposed in `/v1/models.data` until
an operator installs weights, starts a serving backend, and the upstream probe
passes:

| Family | Current status | Reason |
|---|---|---|
| Qwen3-VL exact family | Partially exposed | `qwen3-vl-8b-instruct:vllm` is live through Docker Model Runner vLLM-Metal and passed the image + strict JSON smoke. Larger Qwen3-VL IDs still need upstream servers and smoke validation. Nearby `qwen3.6:27b` remains a local Ollama fallback candidate, not an exact Qwen3-VL id. |
| GLM-V | Not exposed | No GLM-V upstream is configured and probe-ready. |
| MiniCPM-V | Partially exposed | `minicpm-v-4.5-gguf-q4-k-m:dmr` is live through a locally packaged Docker Model Runner llama.cpp GGUF plus multimodal projector and passed the vehicle-image strict JSON smoke. Full upstream `openbmb/MiniCPM-V-4_5` / `MiniCPM-V-4.6` vLLM-style IDs still need dedicated upstream servers and smoke validation. |
| InternVL | Not exposed | No InternVL upstream is configured and probe-ready. |
| Llama 4 VLMs | Not exposed | No Llama 4 Scout/Maverick upstream is configured and probe-ready; license and hardware fit still need review. |
| Kimi multimodal | Not exposed | No Kimi-VL/Kimi-K2.5 upstream is configured and probe-ready. |
| DeepSeek-VL2 | Not exposed | No DeepSeek-VL2 upstream is configured and probe-ready. |
| Molmo 2 | Not exposed | `Molmo2-4B` can be pulled and listed by Docker Model Runner, but local vLLM-Metal load failed because the runner would need `trust_remote_code=True`; do not expose until actual inference smoke passes. |
| Aya Vision | Not exposed | No Aya Vision upstream is configured and probe-ready; license requires production review. |

Exposed local benchmark candidates:

| Model id | Benchmark status |
|---|---|
| `qwen3-vl-8b-instruct:vllm` | Exact demanded Qwen3-VL candidate; Docker Model Runner vLLM-Metal upstream passed engine text JSON smoke and vehicle-image strict JSON smoke with `max_tokens=768`. |
| `minicpm-v-4.5-gguf-q4-k-m:dmr` | Priority-list MiniCPM-V candidate; locally packaged `openbmb/MiniCPM-V-4_5-gguf` Q4_K_M with `mmproj-model-f16.gguf`, served by Docker Model Runner llama.cpp with `chat_template_kwargs.enable_thinking=false`, and passed vehicle-image strict JSON smoke with `max_tokens=768`. |
| `ministral-3:3b` | Image + strict JSON smoke validated in the FraudGuard endpoint report. |
| `ministral-3:8b` | Image + strict JSON smoke validated in the FraudGuard endpoint report. |
| `ministral-3:14b` | Image + strict JSON smoke validated in the FraudGuard endpoint report. |
| `gemma4:31b` | Image + strict JSON smoke validated through the engine after the Ollama empty-response retry. |
| `qwen3.6:27b` | Image + strict JSON smoke validated through the engine after empty-response retry and JSON fence cleanup; nearby local candidate, not an exact Qwen3-VL id. |
| `nemotron3:33b` | Unsuitable for this image-scoring benchmark shape today: it remains empty after the Ollama empty-response retry. Keep it out of the strict JSON benchmark until its image prompt path is fixed upstream or replaced with a dedicated Nemotron VL id. |

## Serving Contract

Expose each validated candidate through the engine's OpenAI-compatible
`POST /v1/chat/completions` route.

Required request behavior:

- Accept OpenAI multimodal `messages[].content` arrays with `text` and
  `image_url` parts.
- Support `response_format: {"type": "json_object"}` for strict parseability.
- Run the benchmark at fixed `temperature: 0`, fixed prompt, and fixed image
  preprocessing.
- Return a stable model id from `GET /v1/models`; clients must not guess names.

Example vision request:

```json
{
  "model": "qwen3-vl-8b:vllm",
  "temperature": 0,
  "response_format": {"type": "json_object"},
  "messages": [
    {
      "role": "system",
      "content": "Return JSON only."
    },
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "Assess this vehicle photo for fraud-relevant anomalies. Return anomaly_score, confidence, and reasons."
        },
        {
          "type": "image_url",
          "image_url": {
            "url": "data:image/jpeg;base64,<base64-image>",
            "detail": "low"
          }
        }
      ]
    }
  ]
}
```

Required benchmark log fields:

| Field | Purpose |
|---|---|
| `model_id` | Engine model id used in the request. |
| `model_version` | Upstream model repo/revision or artifact digest. |
| `quantization` | GGUF/MLX/vLLM quantization, or `none` for full precision. |
| `serving_stack` | `llama_cpp`, `mlx`, `vllm`, `sglang`, `ollama_http`, or external NIM. |
| `gpu_type` | Hardware used for the request. |
| `latency_ms` | End-to-end request latency. |
| `tokens_in` | Prompt/input token count when available. |
| `tokens_out` | Completion token count when available. |
| `parse_success` | Whether strict JSON parsing succeeded. |
| `anomaly_score` | Numeric score emitted by the model. |
| `confidence` | Numeric/self-reported confidence emitted by the model. |

## Runtime Notes

- vLLM and SGLang candidates need CUDA-class server hardware; this repo's
  local Mac path is still useful for GGUF/MLX baselines.
- Add `.vllm_models.json` entries only after the upstream model server exists.
  If a manifest gets ahead of runtime by accident, the engine keeps that id in
  `unavailable` until the upstream `/v1/models` probe succeeds. The engine is
  the multiplexer; one vLLM process usually serves one model.
- For OpenAI-compatible upstreams that need model-specific template switches,
  set `chat_template_kwargs` on that model's `.vllm_models.json` entry so
  clients do not need to remember per-model runtime knobs.
- The candidate list came from the application demand. Before production,
  verify each model id, license, model-card safety limits, context length,
  quantization availability, and image-input support against the upstream
  source of record.
