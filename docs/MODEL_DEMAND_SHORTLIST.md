# Model Demand Shortlist - Vehicle-Photo Reasoning

Captured from the application request on 2026-06-15 for the FraudGuard-style
vehicle-photo reasoning workflow.

This is an operator handoff, not a declaration that every candidate is already
served by this engine. Keep three states separate:

- **installed**: appears from `uv run python scripts/list_models.py` or
  `GET /v1/models`.
- **configured**: has a runtime endpoint, for example an entry in
  `.vllm_models.json`.
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
  The engine is the multiplexer; one vLLM process usually serves one model.
- The candidate list came from the application demand. Before production,
  verify each model id, license, model-card safety limits, context length,
  quantization availability, and image-input support against the upstream
  source of record.
