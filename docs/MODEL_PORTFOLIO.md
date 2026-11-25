# Model portfolio

The [portfolio manifest](../src/inference_engine/model_portfolio.json) records exact publisher IDs and Hugging Face revisions. The November 25, 2026 selection includes recent releases and earlier checkpoints that provide smaller deployment options or related versions. The license field describes the published artifact. Ai2's Olmo entries also link to its public training code and data.

The [selection guide](MODEL_PORTFOLIO_RESEARCH.md) compares candidates for specific workloads and reports local results for tabular classification and passage ranking.

The [family coverage audit](MODEL_FAMILY_COVERAGE.md) lists coverage by repository, the next candidates to test and the command to refresh the audit.

| Workload | Examples in the portfolio | Input → output |
|---|---|---|
| Decisions | Mitra 2 (75.7M), MiniCPM-OCR-Jev (2.21B), Decision 3.0 Flash (8.39B) | Labeled table + new rows → class probabilities, document crop + transcription → correctness score, or request + candidate labels → routing scores |
| Retrieval | Qwen3 Embedding (0.6B and 8B), BGE-M3, EmbeddingGemma 2, MiniLM and Qwen rerankers | Text or supported media → vector, or query + document → relevance score |
| Tokenization | OpenEuroLLM v2 (131,072 vocabulary entries), Qwen3-TTS Tokenizer (171M parameters) | Text ↔ token IDs, or audio ↔ neural codec tokens |
| Direct chat and code | Granite 4.1 3B, Qwen3 4B Instruct 2507, Olmo 3 Instruct 7B, Qwen3-Coder-Next | Text or code → answer or proposed edit |
| Dense reasoning | Qwen3.5 0.8B, Granite 4.2 8B, Qwen3.8 27B, Olmo 3.1 Think 32B | Prompt → text, with thinking modes depending on the checkpoint |
| MoE reasoning | gpt-oss-20b, Gemma 4 26B-A4B, Mistral Small 4, Ling 3.2 air, DeepSeek V4.1 Flash-1017 | Supported text or multimodal prompt → text. Size details distinguish total and active parameters |
| Image and video understanding | Florence-2 base, MiniCPM-V 4.7 (1B and 35B-A3B classes), WeMM-Video 2B | Images or video → captions, text or region coordinates. WeMM can also use the soundtrack |
| Speech | Whisper large-v3-turbo, VibeVoice streaming ASR, Kokoro and Audio8 TTS | Speech → transcript, or text → speech |
| Image generation and editing | Z-Image-Turbo, MOSS-Image, FLUX.2 klein 4B | Text → image, or references + instruction → edited image |
| Forecasting | Toto 2.0 4M, Chronos-2 120M | Time series → forecast quantiles, optionally conditioned on covariates with Chronos-2 |

## Browse and acquire

```bash
uv run python scripts/model_portfolio.py
uv run python scripts/model_portfolio.py --task decision --json
uv run python scripts/model_portfolio.py --task reranking --json | jq '.entries[] | {id, downloads_30d, quality_evidence}'
uv run python scripts/model_portfolio.py download openeurollm/tokenizer-128k-v2
```

Listing works offline. The download command accepts one exact publisher ID and fetches its pinned snapshot into the Hugging Face cache. Use `--cache-dir PATH` to choose a destination. The text tokenizer entry selects its vocabulary and configuration files. Check the publisher's file sizes before acquiring large snapshots.

The authenticated service exposes the same catalog at `GET /v1/model-portfolio`. Append `?task=reasoning`, `?task=non-reasoning`, or any exact task from the manifest to filter it. Each response entry links to its pinned model card. Inputs and outputs describe the checkpoint's upstream capabilities. `downloads_30d` is the repository's rolling Hugging Face count at `popularity_observed_on`. `quality_evidence` gives the use case, result, limitation, source and whether the evaluation came from the publisher or an independent group.

## Optimized builds

The November 25 catalog contains **175 model and tokenizer entries** and **230 selected builds from 195 distribution repositories**. Unsloth supplies 81 builds across 47 repositories. Other sources include NVIDIA, LM Studio, ggml-org, ISTA-DASLab, Handy and Bartowski. Each selection records the upstream and distribution revisions, available license information, file sizes and SHA-256 digests. A build's `downloads_30d` counts downloads for its distribution repository. Several quantizations published together share the same count. The license and evidence fields describe the specific edition. An empty variants list means we have not selected an additional build for that entry. [Unsloth model catalog](https://unsloth.ai/docs/get-started/unsloth-model-catalog)

| Format | Selected builds | Examples and requirements |
|---|---:|---|
| GGUF | 168 | Language models include compact distillations and split MoE builds. Load the specified projector for supported media input. Handy ASR builds require transcribe.cpp. Image denoisers need their pipeline companions. |
| MLX | 21 | Apple Silicon builds include Gemma 4 12B IT and Qwen3.8 27B in 4, 5, 6 and 8 bits. Use an MLX runtime that supports the checkpoint and its processor. |
| Bitsandbytes | 7 | Use compatible Transformers or Diffusers accelerators. The Z-Image bundle includes the image transformer, text encoder, VAE, scheduler and tokenizer. |
| Other safetensors | 30 | Includes NVFP4, FP8 and INT4 conversions from NVIDIA, Unsloth and Red Hat. Match the quantization recipe and architecture to the accelerator and serving engine. |
| ONNX | 1 | Load the selected ONNX graphs and external tensor data with an ONNX Runtime compatible with the model. |
| Strata | 1 | Packed Strata weights, lookup tables and tokenizer assets require the Strata runtime. |
| Edge0 | 2 | Edge0 packs include native expert and dense weights. Load them with the Edge0 runtime. |

List variants offline, then pass the distribution ID and its exact quantization to fetch one build:

```bash
uv run python scripts/model_portfolio.py variants
uv run python scripts/model_portfolio.py variants google/gemma-3-4b-it
uv run python scripts/model_portfolio.py variants --task embedding --json
uv run python scripts/model_portfolio.py download unsloth/gemma-3-4b-it-GGUF --variant UD-Q4_K_XL
uv run python scripts/model_portfolio.py download unsloth/gemma-3-12b-it-GGUF --variant UD-Q4_K_XL
```

The two Gemma 3 downloads are **3.40 GB and 8.29 GB**, including each BF16 projector. For comparison, even the smallest listed DeepSeek V4.1 Flash-1017 build is about 234 GB. These are decimal on-disk sizes. Runtime memory also depends on context length and backend. The downloader checks each selected file's size and SHA-256 and prints its absolute cache path. Use `--cache-dir PATH` to choose the cache volume. A second download can reuse files already verified by their digest. When the publisher hosts native and GGUF formats in one repository, omit `--variant` for the selected native files or provide a quantization for one GGUF build.

Set `HF_HUB_OFFLINE=1` before running the command to verify a cached build offline. If the Hub's Xet client reports a transport error, retry with `HF_HUB_DISABLE_XET=1` to use HTTP with the same revision and integrity checks.

For image inference with a Gemma GGUF, pass the returned model path as `-m` and the matching projector path as `--mmproj` to a compatible `llama-server`. To route that server through the engine, configure it as an OpenAI-compatible upstream with its advertised model ID and vision capability. Files in the Hugging Face cache become available to clients after a runtime serves them. The existing Ollama IDs for the original Gemma models are `gemma3:4b` and `gemma3:12b`. [Service configuration](../.vllm_models.example.json) · [Unsloth GGUF documentation](https://unsloth.ai/docs/basics/inference-and-deployment/saving-to-gguf)

### Local acquisition and inference checks

On November 25, 2026, six `UD-Q4_K_XL` builds were fetched into the local Hugging Face cache. All ten files matched the pinned digests and totaled 18,301,021,408 bytes. These checks used temporary local servers. They establish that the listed inputs completed successfully on this host.

| Unsloth GGUF | Observed result |
|---|---|
| Gemma 3 4B IT and 12B IT | Each identified red and blue images with the same prompt, both directly in llama-server and through the inference engine's `/v1/chat/completions` route. Each model used its own BF16 projector. |
| Qwen3.5 0.8B | Identified red and blue images through llama-server with its F16 projector. |
| EmbeddingGemma 2 | Returned finite, unit-length 768-dimensional embeddings for two texts and two images. |
| Granite 4.1 3B and Qwen3 4B Instruct 2507 | Each returned `12` when asked to add seven and five. |

Three more builds were fetched and verified on the same host. Together the nine cached selections total **21,226,338,464 bytes**. The following checks used an Apple M5 Max and temporary runtimes.

| Additional build | Inference result |
|---|---|
| Unsloth DeepSeek R1 Distill Qwen 1.5B Q4_K_M, 1,117,321,312 bytes | llama.cpp build 12721 returned 15 for 8 + 7 and selected pear as the fruit. Both answers included extra prose. Content accuracy was 2/2 and exact-format adherence was 0/2. |
| Handy Nemotron 3.5 ASR 0.6B Q4_K_M, 495,831,520 bytes | transcribe.cpp v0.4.2 Metal transcribed “The blue train arrives at the station on Friday morning.” exactly in both offline and 200 ms streaming-chunk tests. This was one 3.34-second synthetic English recording. |
| Empero Qwen3.8-2B-Distill Q4_K_M, 1,312,164,224 bytes | llama.cpp build 12721 on Metal returned exactly `15` for 8 + 7 and `pear` for the fruit choice. Both requests used temperature 0 and seed 17. This checks text inference and format adherence on two prompts. |

Reproduce these verified acquisitions offline with `HF_HUB_OFFLINE=1`, or omit it to acquire them from the Hub.

```bash
HF_HUB_OFFLINE=1 uv run python scripts/model_portfolio.py download empero-ai/Qwen3.8-2B-Distill-GGUF --variant Q4_K_M
HF_HUB_OFFLINE=1 uv run python scripts/model_portfolio.py download unsloth/DeepSeek-R1-Distill-Qwen-1.5B-GGUF --variant Q4_K_M
HF_HUB_OFFLINE=1 uv run python scripts/model_portfolio.py download handy-computer/nemotron-3.5-asr-streaming-0.6b-gguf --variant Q4_K_M
```

## Use a model in the service

| Integration | Next step |
|---|---|
| `ollama` | Pull the entry's `ollama_tag` into the server at the engine's `OLLAMA_HTTP_ENDPOINT`. Once the tag appears in the `data` array of `GET /v1/models`, use that tag as the API model ID. Ollama distributes its own build of the checkpoint. |
| `chat-upstream` | Start a compatible inference server, verify its advertised ID, and register the endpoint and ID in the deployment's [vLLM manifest](../.vllm_models.example.json). Validate the input formats your application needs. |
| `worker-required` | Provision the workload-specific runtime described in the model card. Speech recognition, speech synthesis, image generation, tabular decisions and forecasting need their own request and output formats. |
| `standalone-asset` | Load the pinned tokenizer or codec in its native library. The engine's `/tokenize` and `/detokenize` endpoints use the tokenizer bound to a loaded model. |

For a small local deployment, pull `granite4.1:3b` for `POST /v1/chat/completions` and `qwen3-embedding:0.6b` for `POST /v1/embeddings`. Text vectors can also feed the engine's cosine-similarity `/v1/rerank` route. The Qwen3 reranker uses a separate scoring model and needs a worker to execute it.

To update a release, verify its ID, license, capabilities, size and public release month against the publisher. Pin its revision, check the evidence against that variant and update the review date. When refreshing popularity, read all repository counts in the same observation and update the snapshot date. Runtime discovery and signed deployment routing determine which requests the engine accepts.
