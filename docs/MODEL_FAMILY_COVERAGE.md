# Hugging Face model family coverage

**November 25, 2026.** The portfolio covers **38 of 38 publisher repositories** and **140 of 140 additional direct quantization repositories** in the nine-family audit. Qwen accounts for 89 of the 140 quantizations. MiniMax and IBM Granite add 9 publisher and 17 quantization matches. All **9 of 9** recent high-download repositories in the separate distillation name search now have a selected GGUF. Coverage is measured per repository at the thresholds below.

## How coverage was measured

We paginated the public Hugging Face model API in descending order of trailing 30-day downloads through the 100,000-download cutoff. There were 2,386 distinct repositories across all ages and 610 with a UTC creation date from May 25 through November 25, 2026, inclusive. Downloads reflect repository activity and depend on the Hub’s file-counting rules. They do not measure unique users or model quality. [Hub API query](https://huggingface.co/api/models?sort=downloads&direction=-1&limit=500&expand=downloads&expand=createdAt&expand=tags&expand=pipeline_tag) · [Download methodology](https://huggingface.co/docs/hub/models-download-stats)

First-party scope covers family-named repositories in Qwen, deepseek-ai, moonshotai, zai-org/THUDM, meta-llama, meta-models, nvidia and google, plus the mistralai publisher, including its specialists. The quantization column counts additional repositories whose Hub metadata explicitly identifies a direct quantized parent in one of these publisher families, regardless of the parent’s age. Each repository gets one count. The exact repository ID must appear as a parent or selected build in the portfolio to earn coverage. A matching multi-quant repository earns one match even when only some of its files are selected.

Other optimizations can use several intermediate parents or omit lineage tags. Distillation often appears as fine-tuning and can involve a teacher from another family. The direct-quantization count therefore understates the wider ecosystem. See the separate distillation examples below.

## Coverage by family

A repository counts once here even when the portfolio offers several quantizations. Llama has no publisher or direct quantization repository above both thresholds in this window. The catalog includes older Llama releases separately.

| Family | Publisher repositories covered / found | Additional direct quantization repositories covered / found |
|---|---:|---:|
| Qwen | 7 / 7 | 89 / 89 |
| DeepSeek | 4 / 4 | 12 / 12 |
| Kimi | 2 / 2 | 1 / 1 |
| GLM | 3 / 3 | 6 / 6 |
| Llama | 0 / 0 | 0 / 0 |
| Muse | 3 / 3 | 1 / 1 |
| Nemotron | 11 / 11 | 7 / 7 |
| Gemma | 7 / 7 | 22 / 22 |
| Mistral | 1 / 1 | 2 / 2 |
| MiniMax | 4 / 4 | 14 / 14 |
| Granite | 5 / 5 | 3 / 3 |

## Which models to evaluate next

1. **Local vision.** Compare Gemma 4 12B IT (Unsloth UD-Q4_K_XL, 7.54 GB with projector) and Muse Glimmer 30B (19.73 GB with projector) against the already tested Gemma 3 models. Use the same labeled images and scanned pages. Record answer accuracy, first-token latency and peak memory at the service context length. Promote the best fit for each workload through the gateway. [Gemma 4 build](https://huggingface.co/unsloth/gemma-4-12b-it-GGUF) · [Muse build](https://huggingface.co/unsloth/Muse-Glimmer-30B-GGUF)
2. **Speech and decisions.** Expose the locally tested 0.6B Nemotron ASR through a speech worker and compare its transcripts with Whisper on real recordings, including background noise and required languages. For decisions, compare OTel 2.0's Q4_K_M and Q8_0 against Decision 3.0 Flash on the same labeled routing requests. Measure wrong routes, label coverage and latency. The OTel publisher observed about a 2.6-point JevBench loss with Q4_K_M. [ASR build](https://huggingface.co/handy-computer/nemotron-3.5-asr-streaming-0.6b-gguf) · [OTel decision model](https://huggingface.co/farbodtavakkoli/OTel-2.0-Decision-26B-A4B-4bit)
3. **Large reasoning models.** Use a separately provisioned inference endpoint for Kimi K3 (711.97 GB for the selected Unsloth build) and the selected DeepSeek V4.1 Flash-1017 GSQ build (249.81 GB). The local machine has 128 GiB of unified memory. Size the endpoint for weights, KV cache and concurrency, then test the actual upstream ID through the service. [Kimi build](https://huggingface.co/unsloth/Kimi-K3-GGUF) · [DeepSeek build](https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF)
4. **Community Qwen editions.** Compare a direct Qwen3.8 quantization with its full-precision source on the same workload. Use separate comparisons for QUASAR's trained NVFP4 student, the REAP 288-expert model and the ternary Bonsai students. Model edits, expert pruning and learned compression can change behavior even when the Hub tags say “quantized.” For Holo 3.1, the family metadata includes Qwen 0.8B, while the model card and files identify the 35B-A3B Holo student as the build's parent. [QUASAR](https://huggingface.co/QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4) · [REAP](https://huggingface.co/sh0wie/Qwen3.8-Flash-Next-REAP-288-MLX-4bit) · [Holo GGUF](https://huggingface.co/Hcompany/Holo-3.1-35B-A3B-GGUF)

To refresh the audit, set the UTC creation window and the trailing download threshold. The first command prints family counts and the highest-download gaps. The second returns every matching repository, its declared upstream and catalog status. The JSON also contains a separate distillation candidate list deduplicated by repository ID. Later runs use the Hub's current download counts, so a fixed creation window can yield different results.

```bash
uv run python scripts/audit_model_coverage.py --since 2026-05-25 --until 2026-11-25
uv run python scripts/audit_model_coverage.py --since 2026-05-25 --min-downloads 100000 --json
```

## Releases outside the numerical cutoff

The catalog also contains Gemma 4 12B IT, announced June 3 after its repository was created May 23, and Kimi K3.1 Pro and Mistral Large 4, whose publisher counts were below the download floor. Older Llama 3.1 8B, Llama 3.2 1B and 3B, Llama 4 Scout, and Bartowski's Llama 3.1 8B GGUF cover established Llama deployments. [Google release](https://blog.google/innovation-and-ai/technology/developers-tools/introducing-gemma-4-12B/) · [Kimi K3.1 Pro](https://huggingface.co/moonshotai/Kimi-K3.1-Pro) · [Mistral Large 4](https://huggingface.co/mistralai/Mistral-Large-4-1T-A52B) · [Llama 3.1 GGUF](https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF)

Distilled models have distinct student entries, including DeepSeek R1 Distill Qwen 1.5B, 7B and 32B, Meta Muse Glimmer 30B, and the community Qwen3.5 9B student trained on Kimi K3 traces. The selected GGUF for that last student has a different parent from Unsloth's direct Kimi K3 quantization. [DeepSeek student](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B) · [Muse Glimmer](https://huggingface.co/meta-models/Muse-Glimmer-30B) · [Community Kimi student](https://huggingface.co/khazarai/Qwen3.5-9B-Kimi-k3-Distilled)

## What is on this host

The authenticated portfolio endpoint returned 175 catalog entries and 230 selected builds from 195 distribution repositories. The live serving registry advertised 22 model IDs. Two existing routes, `llama3.2:1b` and `ministral-3:3b`, each returned exactly `42` for 17 + 25 through the gateway. The registry also exposes Nemotron Nano 30B, additional Ministral and Gemma 4 sizes and Qwen models. Ollama distributes its own weights. A local Gemma 4 size tag alone did not establish an exact publisher instruction-checkpoint match.

Nine selected builds passed byte-size and SHA-256 checks in the default Hub cache. Six previous local inference checks included Gemma 3 vision and EmbeddingGemma image and text embeddings. Three additional builds ran on temporary local runtimes. Empero’s Qwen 2B student returned both requested answers in the exact format. DeepSeek Distill 1.5B returned correct answers to both prompts with extra prose. Nemotron 0.6B transcribed one synthetic English sentence exactly in offline and streaming modes. These local checks cover the stated inputs. [Detailed checks](MODEL_PORTFOLIO.md#local-acquisition-and-inference-checks)

## Repository-level evidence

Each row links to the Hugging Face repository and records its creation date, trailing downloads at observation and exact catalog match. Quantization rows also give the direct parent declared to the Hub. The last two publisher groups are additional checks outside the nine-family total.

<details><summary>Qwen — 7 first-party, 89 additional quantization repositories</summary>

**First-party**

| Repository | Created (UTC) | Downloads (30d) | Catalog |
|---|---|---:|---|
| [Qwen/Qwen3.8-27B-FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8) | 2026-08-13 | 7,028,433 | Present |
| [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) | 2026-08-05 | 6,047,011 | Present |
| [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) | 2026-08-24 | 1,658,768 | Present |
| [Qwen/Qwen3-ASR-0.6B-hf](https://huggingface.co/Qwen/Qwen3-ASR-0.6B-hf) | 2026-06-26 | 504,923 | Present |
| [Qwen/Qwen3.8-Flash-Next-FP8](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8) | 2026-08-24 | 378,050 | Present |
| [Qwen/Qwen3-ASR-1.7B-hf](https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf) | 2026-06-26 | 351,117 | Present |
| [Qwen/Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1) | 2026-09-14 | 283,561 | Present |

**Other direct quantizations**

| Repository | Created (UTC) | Downloads (30d) | Catalog | Declared parent |
|---|---|---:|---|---|
| [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) | 2026-08-13 | 8,181,570 | Present | Qwen/Qwen3.8-27B |
| [ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF) | 2026-09-07 | 7,683,242 | Present | Qwen/Qwen3.8-Flash-Next |
| [huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF](https://huggingface.co/huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF) | 2026-08-16 | 4,628,141 | Present | Qwen/Qwen3.8-27B |
| [unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) | 2026-08-26 | 4,051,762 | Present | Qwen/Qwen3.8-Flash-Next |
| [nvidia/Qwen3.6-35B-A3B-NVFP4](https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4) | 2026-05-27 | 3,593,391 | Present | Qwen/Qwen3.6-35B-A3B |
| [cdiamond/Qwen3.8-27B-iMatrix-NVFP4-MTP-GGUF](https://huggingface.co/cdiamond/Qwen3.8-27B-iMatrix-NVFP4-MTP-GGUF) | 2026-08-17 | 3,335,303 | Present | Qwen/Qwen3.8-27B |
| [lmstudio-community/Qwen3.8-27B-MLX-4bit](https://huggingface.co/lmstudio-community/Qwen3.8-27B-MLX-4bit) | 2026-08-14 | 2,554,128 | Present | Qwen/Qwen3.8-27B |
| [agentionai/Qwen3.8-Flash-Next-Gyro-GGUF](https://huggingface.co/agentionai/Qwen3.8-Flash-Next-Gyro-GGUF) | 2026-09-30 | 2,518,137 | Present | Qwen/Qwen3.8-Flash-Next |
| [lmstudio-community/Qwen3.8-27B-MLX-8bit](https://huggingface.co/lmstudio-community/Qwen3.8-27B-MLX-8bit) | 2026-08-14 | 2,401,990 | Present | Qwen/Qwen3.8-27B |
| [lmstudio-community/Qwen3.8-27B-MLX-6bit](https://huggingface.co/lmstudio-community/Qwen3.8-27B-MLX-6bit) | 2026-08-14 | 2,399,515 | Present | Qwen/Qwen3.8-27B |
| [lmstudio-community/Qwen3.8-27B-MLX-5bit](https://huggingface.co/lmstudio-community/Qwen3.8-27B-MLX-5bit) | 2026-08-14 | 2,370,733 | Present | Qwen/Qwen3.8-27B |
| [byteshape/Qwen3.8-27B-GGUF](https://huggingface.co/byteshape/Qwen3.8-27B-GGUF) | 2026-08-18 | 2,171,142 | Present | Qwen/Qwen3.8-27B |
| [JonathanColetti/Qwen3.8-27B-Uncensored-GGUF](https://huggingface.co/JonathanColetti/Qwen3.8-27B-Uncensored-GGUF) | 2026-08-14 | 2,135,073 | Present | Qwen/Qwen3.8-27B |
| [unsloth/Qwen-Image-2.1-Turbo-GGUF](https://huggingface.co/unsloth/Qwen-Image-2.1-Turbo-GGUF) | 2026-10-10 | 2,121,983 | Present | Qwen/Qwen-Image-2.1-Turbo |
| [prism-ml/Ternary-Bonsai-2-9B-gguf](https://huggingface.co/prism-ml/Ternary-Bonsai-2-9B-gguf) | 2026-09-23 | 2,024,327 | Present | Qwen/Qwen3.5-9B |
| [prism-ml/Ternary-Bonsai-2-2B-gguf](https://huggingface.co/prism-ml/Ternary-Bonsai-2-2B-gguf) | 2026-09-23 | 1,898,793 | Present | Qwen/Qwen3.5-2B |
| [unsloth/Qwen3.8-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4) | 2026-08-13 | 1,878,785 | Present | Qwen/Qwen3.8-27B |
| [HauhauCS/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF](https://huggingface.co/HauhauCS/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF) | 2026-08-17 | 1,480,657 | Present | Qwen/Qwen3.8-27B |
| [unsloth/Qwen-Image-2.1-GGUF](https://huggingface.co/unsloth/Qwen-Image-2.1-GGUF) | 2026-09-21 | 1,458,783 | Present | Qwen/Qwen-Image-2.1 |
| [peculiar-ragdoll/Dirk-Qwen3.8-27B-GGUF](https://huggingface.co/peculiar-ragdoll/Dirk-Qwen3.8-27B-GGUF) | 2026-08-14 | 1,419,205 | Present | Qwen/Qwen3.8-27B |
| [ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF) | 2026-08-28 | 1,310,656 | Present | Qwen/Qwen3.8-27B |
| [prism-ml/Ternary-Bonsai-2-27B-gguf](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf) | 2026-09-16 | 1,283,338 | Present | Qwen/Qwen3.8-27B |
| [0bserverx/Qwen3.8-27B-Heretic-Abliterated-Uncensored-GGUF](https://huggingface.co/0bserverx/Qwen3.8-27B-Heretic-Abliterated-Uncensored-GGUF) | 2026-08-14 | 1,242,305 | Present | Qwen/Qwen3.8-27B |
| [huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated-v2-GGUF](https://huggingface.co/huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated-v2-GGUF) | 2026-10-11 | 1,202,673 | Present | Qwen/Qwen3.8-Flash-Next |
| [ggml-org/Qwen3.8-27B-GGUF](https://huggingface.co/ggml-org/Qwen3.8-27B-GGUF) | 2026-08-14 | 1,050,205 | Present | Qwen/Qwen3.8-27B |
| [lmstudio-community/Qwen3.8-27B-GGUF](https://huggingface.co/lmstudio-community/Qwen3.8-27B-GGUF) | 2026-08-14 | 1,031,653 | Present | Qwen/Qwen3.8-27B |
| [turintech/Qwen3.5-4B-Artemis-GGUF](https://huggingface.co/turintech/Qwen3.5-4B-Artemis-GGUF) | 2026-10-14 | 995,512 | Present | Qwen/Qwen3.5-4B |
| [turintech/Qwen3.5-9B-Artemis-GGUF](https://huggingface.co/turintech/Qwen3.5-9B-Artemis-GGUF) | 2026-10-15 | 991,098 | Present | Qwen/Qwen3.5-9B |
| [DavidAU/Qwen3.8-27B-Turbo-Brilliance-50X-Reasoning-Instruct-NEO-MAX-MTP-GGUF](https://huggingface.co/DavidAU/Qwen3.8-27B-Turbo-Brilliance-50X-Reasoning-Instruct-NEO-MAX-MTP-GGUF) | 2026-09-29 | 980,184 | Present | Qwen/Qwen3.8-27B |
| [ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF) | 2026-09-26 | 970,097 | Present | Qwen/Qwen3.8-Flash-Next |
| [cyankiwi/Qwen3.8-27B-AWQ-INT4](https://huggingface.co/cyankiwi/Qwen3.8-27B-AWQ-INT4) | 2026-08-15 | 897,610 | Present | Qwen/Qwen3.8-27B |
| [turintech/Qwen3.5-0.8B-Artemis-GGUF](https://huggingface.co/turintech/Qwen3.5-0.8B-Artemis-GGUF) | 2026-10-15 | 872,277 | Present | Qwen/Qwen3.5-0.8B |
| [agentionai/Qwen3.8-27B-AP-GGUF](https://huggingface.co/agentionai/Qwen3.8-27B-AP-GGUF) | 2026-09-22 | 853,220 | Present | Qwen/Qwen3.8-27B |
| [Maximilian228/Qwen3.8-Flash-Next-NVFP4-GPTQ-Strata](https://huggingface.co/Maximilian228/Qwen3.8-Flash-Next-NVFP4-GPTQ-Strata) | 2026-10-09 | 841,208 | Present | Qwen/Qwen3.8-Flash-Next |
| [lmstudio-community/Qwen3.5-9B-MLX-4bit](https://huggingface.co/lmstudio-community/Qwen3.5-9B-MLX-4bit) | 2026-06-02 | 792,668 | Present | Qwen/Qwen3.5-9B |
| [lmstudio-community/Qwen3.5-9B-MLX-8bit](https://huggingface.co/lmstudio-community/Qwen3.5-9B-MLX-8bit) | 2026-06-02 | 739,868 | Present | Qwen/Qwen3.5-9B |
| [unsloth/Qwen3.6-35B-A3B-NVFP4-Fast](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-NVFP4-Fast) | 2026-07-10 | 688,778 | Present | Qwen/Qwen3.6-35B-A3B |
| [turintech/Qwen3.5-2B-Artemis-GGUF](https://huggingface.co/turintech/Qwen3.5-2B-Artemis-GGUF) | 2026-10-15 | 680,270 | Present | Qwen/Qwen3.5-2B |
| [mlx-community/Qwen3.8-27B-4bit](https://huggingface.co/mlx-community/Qwen3.8-27B-4bit) | 2026-08-14 | 660,816 | Present | Qwen/Qwen3.8-27B |
| [OBLITERATUS/Qwen3.8-27B-OBLITERATED](https://huggingface.co/OBLITERATUS/Qwen3.8-27B-OBLITERATED) | 2026-08-19 | 632,256 | Present | Qwen/Qwen3.8-27B |
| [RadixArk/Qwen3.8-27B-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-27B-NVFP4) | 2026-08-14 | 596,574 | Present | Qwen/Qwen3.8-27B |
| [orcarouter/Qwen3.8-Flash-Next-Uncensored-GGUF](https://huggingface.co/orcarouter/Qwen3.8-Flash-Next-Uncensored-GGUF) | 2026-08-26 | 549,816 | Present | Qwen/Qwen3.8-Flash-Next |
| [AtomicChat/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/AtomicChat/Qwen3.8-Flash-Next-GGUF) | 2026-08-26 | 527,835 | Present | Qwen/Qwen3.8-Flash-Next |
| [chatrlabs/Qwen-Image-2.1-Uncensored-Omni-Quantized](https://huggingface.co/chatrlabs/Qwen-Image-2.1-Uncensored-Omni-Quantized) | 2026-11-03 | 515,591 | Present | Qwen/Qwen-Image-2.1 |
| [z-lab/Qwen3.8-27B-DFlash2-GGUF](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2-GGUF) | 2026-08-18 | 468,881 | Present | Qwen/Qwen3.8-27B |
| [Inferact/Qwen3.8-27B-NVFP4](https://huggingface.co/Inferact/Qwen3.8-27B-NVFP4) | 2026-08-13 | 449,955 | Present | Qwen/Qwen3.8-27B |
| [nicosuter/Qwen3.8-27B-AWQ](https://huggingface.co/nicosuter/Qwen3.8-27B-AWQ) | 2026-08-14 | 435,818 | Present | Qwen/Qwen3.8-27B |
| [PassingByPixels/Qwen3.8-27B-NVFP4](https://huggingface.co/PassingByPixels/Qwen3.8-27B-NVFP4) | 2026-08-14 | 412,921 | Present | Qwen/Qwen3.8-27B |
| [runmyai/Qwen3.8-Flash-Next-TokenWing](https://huggingface.co/runmyai/Qwen3.8-Flash-Next-TokenWing) | 2026-11-17 | 361,470 | Present | Qwen/Qwen3.8-Flash-Next |
| [nvidia/Qwen3.8-27B-NVFP4](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4) | 2026-09-04 | 356,807 | Present | Qwen/Qwen3.8-27B |
| [antirez/qwen3.8-flash-next-gguf](https://huggingface.co/antirez/qwen3.8-flash-next-gguf) | 2026-09-14 | 343,754 | Present | Qwen/Qwen3.8-Flash-Next |
| [lmstudio-community/Qwen3.6-27B-MLX-8bit](https://huggingface.co/lmstudio-community/Qwen3.6-27B-MLX-8bit) | 2026-06-02 | 341,491 | Present | Qwen/Qwen3.6-27B |
| [lmstudio-community/Qwen3.6-27B-MLX-5bit](https://huggingface.co/lmstudio-community/Qwen3.6-27B-MLX-5bit) | 2026-06-02 | 340,802 | Present | Qwen/Qwen3.6-27B |
| [lmstudio-community/Qwen3.6-27B-MLX-6bit](https://huggingface.co/lmstudio-community/Qwen3.6-27B-MLX-6bit) | 2026-06-02 | 336,290 | Present | Qwen/Qwen3.6-27B |
| [bartowski/Qwen3.8-27B-GGUF](https://huggingface.co/bartowski/Qwen3.8-27B-GGUF) | 2026-08-14 | 329,179 | Present | Qwen/Qwen3.8-27B |
| [gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090](https://huggingface.co/gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090) | 2026-08-14 | 318,285 | Present | Qwen/Qwen3.8-27B |
| [empero-ai/Qwen3.8-27B-Ridge-GGUF](https://huggingface.co/empero-ai/Qwen3.8-27B-Ridge-GGUF) | 2026-08-15 | 281,151 | Present | Qwen/Qwen3.8-27B |
| [abenzerps/Qwen-Image-2.1-Turbo-Quantized](https://huggingface.co/abenzerps/Qwen-Image-2.1-Turbo-Quantized) | 2026-09-20 | 280,378 | Present | Qwen/Qwen-Image-2.1-Turbo |
| [philbert440/Qwen3.8-27B-W4A16-AWQ](https://huggingface.co/philbert440/Qwen3.8-27B-W4A16-AWQ) | 2026-08-14 | 219,114 | Present | Qwen/Qwen3.8-27B |
| [QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4](https://huggingface.co/QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4) | 2026-08-23 | 218,626 | Present | Qwen/Qwen3.8-27B |
| [RedHatAI/Qwen3.8-27B-INT4](https://huggingface.co/RedHatAI/Qwen3.8-27B-INT4) | 2026-08-17 | 213,545 | Present | Qwen/Qwen3.8-27B |
| [huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated-GGUF](https://huggingface.co/huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated-GGUF) | 2026-09-06 | 208,487 | Present | Qwen/Qwen3.8-Flash-Next |
| [RadixArk/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4) | 2026-08-25 | 204,205 | Present | Qwen/Qwen3.8-Flash-Next |
| [prism-ml/Ternary-Bonsai-2-4B-gguf](https://huggingface.co/prism-ml/Ternary-Bonsai-2-4B-gguf) | 2026-09-23 | 199,748 | Present | Qwen/Qwen3.5-4B |
| [Edge0/Edge0-145B-q3k](https://huggingface.co/Edge0/Edge0-145B-q3k) | 2026-11-11 | 197,160 | Present | Qwen/Qwen3.8-Flash-Next |
| [AtomicChat/Qwen-Image-2.1-Turbo-GGUF](https://huggingface.co/AtomicChat/Qwen-Image-2.1-Turbo-GGUF) | 2026-10-09 | 194,751 | Present | Qwen/Qwen-Image-2.1-Turbo |
| [leejet/Qwen-Image-2.1-GGUF](https://huggingface.co/leejet/Qwen-Image-2.1-GGUF) | 2026-09-20 | 176,618 | Present | Qwen/Qwen-Image-2.1 |
| [dark-side-of-the-code/Qwen3-Coder-30B-A3B-Instruct-AWQ](https://huggingface.co/dark-side-of-the-code/Qwen3-Coder-30B-A3B-Instruct-AWQ) | 2026-07-13 | 169,031 | Present | Qwen/Qwen3-Coder-30B-A3B-Instruct |
| [SergiioB/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16](https://huggingface.co/SergiioB/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16) | 2026-08-15 | 142,293 | Present | Qwen/Qwen3.8-27B |
| [Alennnndy/Qwen3.6-27B-AWQ-4bit](https://huggingface.co/Alennnndy/Qwen3.6-27B-AWQ-4bit) | 2026-07-03 | 140,365 | Present | Qwen/Qwen3.6-27B |
| [sh0wie/Qwen3.8-Flash-Next-REAP-288-MLX-4bit](https://huggingface.co/sh0wie/Qwen3.8-Flash-Next-REAP-288-MLX-4bit) | 2026-08-27 | 134,236 | Present | Qwen/Qwen3.8-Flash-Next |
| [Inferact/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/Inferact/Qwen3.8-Flash-Next-NVFP4) | 2026-08-26 | 132,598 | Present | Qwen/Qwen3.8-Flash-Next |
| [ggml-org/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/ggml-org/Qwen3.8-Flash-Next-GGUF) | 2026-08-27 | 132,076 | Present | Qwen/Qwen3.8-Flash-Next |
| [Youssofal/Qwen3.8-Flash-Next-MTPLX-Optimized-Speed](https://huggingface.co/Youssofal/Qwen3.8-Flash-Next-MTPLX-Optimized-Speed) | 2026-08-28 | 131,588 | Present | Qwen/Qwen3.8-Flash-Next |
| [primitive-ai/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/primitive-ai/Qwen3.8-Flash-Next-NVFP4) | 2026-08-26 | 130,799 | Present | Qwen/Qwen3.8-Flash-Next |
| [ggml-org/Qwen3.5-4B-GGUF](https://huggingface.co/ggml-org/Qwen3.5-4B-GGUF) | 2026-11-12 | 129,208 | Present | Qwen/Qwen3.5-4B |
| [TensorFold/Qwen3.8-Flash-Next-MLX-4bit](https://huggingface.co/TensorFold/Qwen3.8-Flash-Next-MLX-4bit) | 2026-08-26 | 127,894 | Present | Qwen/Qwen3.8-Flash-Next |
| [Intel/Qwen3.8-Flash-Next-W4A16-AutoRound](https://huggingface.co/Intel/Qwen3.8-Flash-Next-W4A16-AutoRound) | 2026-08-28 | 126,989 | Present | Qwen/Qwen3.8-Flash-Next |
| [Sawfwair/Qwen3.8-Flash-Next-MLX-Mixed-2bit](https://huggingface.co/Sawfwair/Qwen3.8-Flash-Next-MLX-Mixed-2bit) | 2026-08-26 | 126,492 | Present | Qwen/Qwen3.8-Flash-Next |
| [azukivc/Qwen3.8-27B-Uncensored-GGUF](https://huggingface.co/azukivc/Qwen3.8-27B-Uncensored-GGUF) | 2026-08-22 | 123,744 | Present | Qwen/Qwen3.8-27B |
| [KasugaiSakura/Qwen-Image-2.1-Uncensored-Abenzerps-GGUF](https://huggingface.co/KasugaiSakura/Qwen-Image-2.1-Uncensored-Abenzerps-GGUF) | 2026-09-20 | 123,061 | Present | Qwen/Qwen-Image-2.1 |
| [orcarouter/Qwen3.8-27B-Uncensored-GGUF](https://huggingface.co/orcarouter/Qwen3.8-27B-Uncensored-GGUF) | 2026-08-16 | 120,587 | Present | Qwen/Qwen3.8-27B |
| [ConceptCountry/qwen-concept-27b](https://huggingface.co/ConceptCountry/qwen-concept-27b) | 2026-11-08 | 120,418 | Present | Qwen/Qwen3.8-27B |
| [ggml-org/Qwen3-Coder-30B-A3B-Instruct-GGUF](https://huggingface.co/ggml-org/Qwen3-Coder-30B-A3B-Instruct-GGUF) | 2026-10-22 | 118,627 | Present | Qwen/Qwen3-Coder-30B-A3B-Instruct |
| [Hcompany/Holo-3.1-35B-A3B-GGUF](https://huggingface.co/Hcompany/Holo-3.1-35B-A3B-GGUF) | 2026-06-02 | 117,576 | Present | Qwen/Qwen3.5-0.8B |
| [prithivMLmods/Qwen-Image-2.1-PE-I2I-GGUF](https://huggingface.co/prithivMLmods/Qwen-Image-2.1-PE-I2I-GGUF) | 2026-09-20 | 110,258 | Present | Qwen/Qwen-Image-2.1-PE-I2I |
| [taronaeo/Qwen3.6-35B-A3B-BE-GGUF](https://huggingface.co/taronaeo/Qwen3.6-35B-A3B-BE-GGUF) | 2026-11-01 | 107,683 | Present | Qwen/Qwen3.6-35B-A3B |
| [Edge0/Edge0-145B-NVFP4](https://huggingface.co/Edge0/Edge0-145B-NVFP4) | 2026-11-11 | 104,914 | Present | Qwen/Qwen3.8-Flash-Next |
| [nvidia/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4) | 2026-09-02 | 100,681 | Present | Qwen/Qwen3.8-Flash-Next |

</details>

<details><summary>DeepSeek — 4 first-party, 12 additional quantization repositories</summary>

**First-party**

| Repository | Created (UTC) | Downloads (30d) | Catalog |
|---|---|---:|---|
| [deepseek-ai/DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) | 2026-07-31 | 1,134,785 | Present |
| [deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) | 2026-09-10 | 1,096,257 | Present |
| [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) | 2026-10-17 | 550,396 | Present |
| [deepseek-ai/DeepSeek-V4-Flash-DSpark](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-DSpark) | 2026-06-27 | 162,925 | Present |

**Other direct quantizations**

| Repository | Created (UTC) | Downloads (30d) | Catalog | Declared parent |
|---|---|---:|---|---|
| [ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF) | 2026-10-23 | 3,682,279 | Present | deepseek-ai/DeepSeek-V4.1-Flash-1017 |
| [unsloth/DeepSeek-V4.1-Flash-1017-GGUF](https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF) | 2026-10-17 | 1,595,364 | Present | deepseek-ai/DeepSeek-V4.1-Flash-1017 |
| [antirez/deepseek-v4.1-flash-gguf](https://huggingface.co/antirez/deepseek-v4.1-flash-gguf) | 2026-09-11 | 680,279 | Present | deepseek-ai/DeepSeek-V4.1-Flash |
| [huihui-ai/Huihui-DeepSeek-V4-Flash-abliterated-ds4-GGUF](https://huggingface.co/huihui-ai/Huihui-DeepSeek-V4-Flash-abliterated-ds4-GGUF) | 2026-05-26 | 620,304 | Present | deepseek-ai/DeepSeek-V4-Flash |
| [ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-Coder-GGUF](https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-Coder-GGUF) | 2026-11-02 | 592,600 | Present | deepseek-ai/DeepSeek-V4.1-Flash-1017 |
| [prem-research/DeepSeek-V4-Flash-0731-abliterated](https://huggingface.co/prem-research/DeepSeek-V4-Flash-0731-abliterated) | 2026-08-12 | 323,659 | Present | deepseek-ai/DeepSeek-V4-Flash-0731 |
| [huihui-ai/Huihui-DeepSeek-V4-Flash-0731-abliterated-GGUF](https://huggingface.co/huihui-ai/Huihui-DeepSeek-V4-Flash-0731-abliterated-GGUF) | 2026-08-02 | 250,483 | Present | deepseek-ai/DeepSeek-V4-Flash-0731 |
| [peasantsmith/DeepSeek-V4.1-Flash-1017-Maya-GGUF](https://huggingface.co/peasantsmith/DeepSeek-V4.1-Flash-1017-Maya-GGUF) | 2026-10-19 | 177,723 | Present | deepseek-ai/DeepSeek-V4.1-Flash-1017 |
| [huihui-ai/Huihui-DeepSeek-V4.1-Flash-1017-abliterated-GGUF](https://huggingface.co/huihui-ai/Huihui-DeepSeek-V4.1-Flash-1017-abliterated-GGUF) | 2026-10-25 | 153,301 | Present | deepseek-ai/DeepSeek-V4.1-Flash-1017 |
| [orcarouter/DeepSeek-V4.1-Flash-1017-abliterated-GGUF](https://huggingface.co/orcarouter/DeepSeek-V4.1-Flash-1017-abliterated-GGUF) | 2026-10-22 | 145,360 | Present | deepseek-ai/DeepSeek-V4.1-Flash-1017 |
| [lxe/DeepSeek-V4.1-Flash-1017-dsx-t16](https://huggingface.co/lxe/DeepSeek-V4.1-Flash-1017-dsx-t16) | 2026-10-22 | 128,822 | Present | deepseek-ai/DeepSeek-V4.1-Flash-1017 |
| [unsloth/DeepSeek-V4-Flash-0731-GGUF](https://huggingface.co/unsloth/DeepSeek-V4-Flash-0731-GGUF) | 2026-07-31 | 119,088 | Present | deepseek-ai/DeepSeek-V4-Flash-0731 |

</details>

<details><summary>Kimi — 2 first-party, 1 additional quantization repositories</summary>

**First-party**

| Repository | Created (UTC) | Downloads (30d) | Catalog |
|---|---|---:|---|
| [moonshotai/Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3) | 2026-06-13 | 674,720 | Present |
| [moonshotai/Kimi-K2.7-Code](https://huggingface.co/moonshotai/Kimi-K2.7-Code) | 2026-06-11 | 327,197 | Present |

**Other direct quantizations**

| Repository | Created (UTC) | Downloads (30d) | Catalog | Declared parent |
|---|---|---:|---|---|
| [unsloth/Kimi-K3-GGUF](https://huggingface.co/unsloth/Kimi-K3-GGUF) | 2026-07-27 | 181,290 | Present | moonshotai/Kimi-K3 |

</details>

<details><summary>GLM — 3 first-party, 6 additional quantization repositories</summary>

**First-party**

| Repository | Created (UTC) | Downloads (30d) | Catalog |
|---|---|---:|---|
| [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) | 2026-08-25 | 6,226,276 | Present |
| [zai-org/GLM-5.3](https://huggingface.co/zai-org/GLM-5.3) | 2026-08-25 | 1,081,011 | Present |
| [zai-org/GLM-5.2](https://huggingface.co/zai-org/GLM-5.2) | 2026-06-16 | 1,008,689 | Present |

**Other direct quantizations**

| Repository | Created (UTC) | Downloads (30d) | Catalog | Declared parent |
|---|---|---:|---|---|
| [peasantsmith/GLM-5.3-Flash-Maya-GGUF](https://huggingface.co/peasantsmith/GLM-5.3-Flash-Maya-GGUF) | 2026-10-07 | 1,979,344 | Present | zai-org/GLM-5.3-Flash |
| [unsloth/GLM-5.3-Flash-GGUF](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF) | 2026-08-26 | 738,989 | Present | zai-org/GLM-5.3-Flash |
| [unsloth/GLM-5.3-GGUF](https://huggingface.co/unsloth/GLM-5.3-GGUF) | 2026-08-28 | 337,109 | Present | zai-org/GLM-5.3 |
| [nvidia/GLM-5.3-NVFP4](https://huggingface.co/nvidia/GLM-5.3-NVFP4) | 2026-09-14 | 229,578 | Present | zai-org/GLM-5.3 |
| [BoldingBuilds/GLM-5.3-Flash-Uncensored-GGUF](https://huggingface.co/BoldingBuilds/GLM-5.3-Flash-Uncensored-GGUF) | 2026-09-05 | 194,604 | Present | zai-org/GLM-5.3-Flash |
| [tattrongvu/GLM-5.3-Flash-NVFP4](https://huggingface.co/tattrongvu/GLM-5.3-Flash-NVFP4) | 2026-08-31 | 110,749 | Present | zai-org/GLM-5.3-Flash |

</details>

<details><summary>Muse — 3 first-party, 1 additional quantization repositories</summary>

**First-party**

| Repository | Created (UTC) | Downloads (30d) | Catalog |
|---|---|---:|---|
| [meta-models/Muse-Glimmer-30B-GGUF](https://huggingface.co/meta-models/Muse-Glimmer-30B-GGUF) | 2026-08-09 | 161,635 | Present |
| [meta-models/Muse-Glimmer-30B](https://huggingface.co/meta-models/Muse-Glimmer-30B) | 2026-08-09 | 148,511 | Present |
| [meta-models/Muse-Spark-Open-816B-A42B-NVFP4](https://huggingface.co/meta-models/Muse-Spark-Open-816B-A42B-NVFP4) | 2026-09-15 | 123,538 | Present |

**Other direct quantizations**

| Repository | Created (UTC) | Downloads (30d) | Catalog | Declared parent |
|---|---|---:|---|---|
| [unsloth/Muse-Glimmer-30B-GGUF](https://huggingface.co/unsloth/Muse-Glimmer-30B-GGUF) | 2026-08-10 | 380,304 | Present | meta-models/Muse-Glimmer-30B |

</details>

<details><summary>Nemotron — 11 first-party, 7 additional quantization repositories</summary>

**First-party**

| Repository | Created (UTC) | Downloads (30d) | Catalog |
|---|---|---:|---|
| [nvidia/NVIDIA-Nemotron-Parse-2.0](https://huggingface.co/nvidia/NVIDIA-Nemotron-Parse-2.0) | 2026-06-30 | 859,053 | Present |
| [nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4) | 2026-08-04 | 781,354 | Present |
| [nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16) | 2026-06-03 | 748,562 | Present |
| [nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16) | 2026-08-01 | 541,056 | Present |
| [nvidia/Nemotron-3-Embed-1B-BF16](https://huggingface.co/nvidia/Nemotron-3-Embed-1B-BF16) | 2026-07-14 | 242,847 | Present |
| [nvidia/Nemotron-3-Diarization](https://huggingface.co/nvidia/Nemotron-3-Diarization) | 2026-09-01 | 241,983 | Present |
| [nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-4GB](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-4GB) | 2026-10-05 | 196,058 | Present |
| [nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF) | 2026-10-07 | 188,328 | Present |
| [nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4) | 2026-06-03 | 142,279 | Present |
| [nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4) | 2026-10-12 | 132,519 | Present |
| [nvidia/NVIDIA-Nemotron-Labs-3-Puzzle-75B-A9B-NVFP4](https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-3-Puzzle-75B-A9B-NVFP4) | 2026-06-24 | 126,507 | Present |

**Other direct quantizations**

| Repository | Created (UTC) | Downloads (30d) | Catalog | Declared parent |
|---|---|---:|---|---|
| [handy-computer/nemotron-3.5-asr-streaming-0.6b-gguf](https://huggingface.co/handy-computer/nemotron-3.5-asr-streaming-0.6b-gguf) | 2026-06-07 | 1,139,680 | Present | nvidia/nemotron-3.5-asr-streaming-0.6b |
| [unsloth/NVIDIA-Nemotron-3.5-Bolt-E4B-GGUF](https://huggingface.co/unsloth/NVIDIA-Nemotron-3.5-Bolt-E4B-GGUF) | 2026-10-23 | 818,871 | Present | nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-BF16 |
| [RedHatAI/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-FP8](https://huggingface.co/RedHatAI/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-FP8) | 2026-08-13 | 286,234 | Present | nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 |
| [ggml-org/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF](https://huggingface.co/ggml-org/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF) | 2026-08-11 | 212,490 | Present | nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16 |
| [Jmica/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-GGUF](https://huggingface.co/Jmica/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-GGUF) | 2026-10-29 | 178,622 | Present | nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B |
| [AtomicChat/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-GGUF](https://huggingface.co/AtomicChat/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-GGUF) | 2026-10-28 | 146,186 | Present | nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B |
| [OpenYourMind/OpenYourMind-NVIDIA-Nemotron-3-Ultra-550B-A55B-abliterated-uncensored-NVFP4](https://huggingface.co/OpenYourMind/OpenYourMind-NVIDIA-Nemotron-3-Ultra-550B-A55B-abliterated-uncensored-NVFP4) | 2026-06-05 | 124,829 | Present | nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16 |

</details>

<details><summary>Gemma — 7 first-party, 22 additional quantization repositories</summary>

**First-party**

| Repository | Created (UTC) | Downloads (30d) | Catalog |
|---|---|---:|---|
| [google/gemma-4-12B-it-qat-q4_0-gguf](https://huggingface.co/google/gemma-4-12B-it-qat-q4_0-gguf) | 2026-06-05 | 1,187,972 | Present |
| [google/embeddinggemma-2](https://huggingface.co/google/embeddinggemma-2) | 2026-09-14 | 586,665 | Present |
| [google/gemma-4-12B-it-qat-w4a16-ct](https://huggingface.co/google/gemma-4-12B-it-qat-w4a16-ct) | 2026-06-05 | 573,173 | Present |
| [google/diffusiongemma-26B-A4B-it](https://huggingface.co/google/diffusiongemma-26B-A4B-it) | 2026-06-09 | 485,861 | Present |
| [google/gemma-4-E2B-it-qat-w4a16-ct](https://huggingface.co/google/gemma-4-E2B-it-qat-w4a16-ct) | 2026-06-04 | 340,536 | Present |
| [google/gemma-4-31B-it-qat-w4a16-ct](https://huggingface.co/google/gemma-4-31B-it-qat-w4a16-ct) | 2026-06-04 | 295,115 | Present |
| [google/gemma-4-E4B-it-qat-w4a16-ct](https://huggingface.co/google/gemma-4-E4B-it-qat-w4a16-ct) | 2026-06-04 | 237,945 | Present |

**Other direct quantizations**

| Repository | Created (UTC) | Downloads (30d) | Catalog | Declared parent |
|---|---|---:|---|---|
| [farbodtavakkoli/OTel-2.0-Decision-26B-A4B-4bit](https://huggingface.co/farbodtavakkoli/OTel-2.0-Decision-26B-A4B-4bit) | 2026-10-06 | 3,193,842 | Present | google/gemma-4-26B-A4B-it |
| [unsloth/gemma-4-12b-it-GGUF](https://huggingface.co/unsloth/gemma-4-12b-it-GGUF) | 2026-05-29 | 2,510,138 | Present | google/gemma-4-12B-it |
| [turintech/gemma-4-E2B-it-Artemis-GGUF](https://huggingface.co/turintech/gemma-4-E2B-it-Artemis-GGUF) | 2026-10-12 | 1,492,314 | Present | google/gemma-4-E2B-it-qat-q4_0-unquantized |
| [turintech/gemma-4-E4B-it-Artemis-GGUF](https://huggingface.co/turintech/gemma-4-E4B-it-Artemis-GGUF) | 2026-10-12 | 1,158,229 | Present | google/gemma-4-E4B-it-qat-q4_0-unquantized |
| [HauhauCS/Gemma4-12B-QAT-Uncensored-HauhauCS-Balanced](https://huggingface.co/HauhauCS/Gemma4-12B-QAT-Uncensored-HauhauCS-Balanced) | 2026-06-22 | 797,323 | Present | google/gemma-4-12B-it |
| [HauhauCS/Gemma4-26B-A4B-QAT-Uncensored-HauhauCS-Balanced-MTP](https://huggingface.co/HauhauCS/Gemma4-26B-A4B-QAT-Uncensored-HauhauCS-Balanced-MTP) | 2026-06-24 | 544,928 | Present | google/gemma-4-26B-A4B-it |
| [onnx-community/embeddinggemma-2-ONNX](https://huggingface.co/onnx-community/embeddinggemma-2-ONNX) | 2026-10-04 | 539,909 | Present | google/embeddinggemma-2 |
| [lmstudio-community/gemma-4-12B-it-MLX-4bit](https://huggingface.co/lmstudio-community/gemma-4-12B-it-MLX-4bit) | 2026-06-11 | 338,769 | Present | google/gemma-4-12B-it |
| [lmstudio-community/gemma-4-12B-it-MLX-8bit](https://huggingface.co/lmstudio-community/gemma-4-12B-it-MLX-8bit) | 2026-06-11 | 322,600 | Present | google/gemma-4-12B-it |
| [lmstudio-community/gemma-4-12B-it-QAT-GGUF](https://huggingface.co/lmstudio-community/gemma-4-12B-it-QAT-GGUF) | 2026-06-04 | 321,429 | Present | google/gemma-4-12B-it-qat-q4_0-unquantized |
| [lmstudio-community/gemma-4-12B-it-MLX-6bit](https://huggingface.co/lmstudio-community/gemma-4-12B-it-MLX-6bit) | 2026-06-11 | 320,820 | Present | google/gemma-4-12B-it |
| [lmstudio-community/gemma-4-12B-it-MLX-5bit](https://huggingface.co/lmstudio-community/gemma-4-12B-it-MLX-5bit) | 2026-06-11 | 319,988 | Present | google/gemma-4-12B-it |
| [melcheikh/gemma-4-31B-it-qat-NVFP4-Blackwell](https://huggingface.co/melcheikh/gemma-4-31B-it-qat-NVFP4-Blackwell) | 2026-06-05 | 259,859 | Present | google/gemma-4-31B-it-qat-q4_0-unquantized |
| [unsloth/gemma-4-26B-A4B-it-qat-GGUF](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-qat-GGUF) | 2026-06-05 | 246,963 | Present | google/gemma-4-26B-A4B-it-qat-q4_0-unquantized |
| [unsloth/gemma-4-E2B-it-qat-mobile-GGUF](https://huggingface.co/unsloth/gemma-4-E2B-it-qat-mobile-GGUF) | 2026-06-05 | 225,641 | Present | google/gemma-4-E2B-it-qat-mobile-transformers |
| [unsloth/gemma-4-12B-it-qat-GGUF](https://huggingface.co/unsloth/gemma-4-12B-it-qat-GGUF) | 2026-06-05 | 167,002 | Present | google/gemma-4-12B-it-qat-q4_0-unquantized |
| [lmstudio-community/gemma-4-12B-it-GGUF](https://huggingface.co/lmstudio-community/gemma-4-12B-it-GGUF) | 2026-06-01 | 153,665 | Present | google/gemma-4-12B-it |
| [lmstudio-community/gemma-4-26B-A4B-it-QAT-GGUF](https://huggingface.co/lmstudio-community/gemma-4-26B-A4B-it-QAT-GGUF) | 2026-06-03 | 143,456 | Present | google/gemma-4-26B-A4B-it-qat-q4_0-unquantized |
| [ggml-org/gemma-4-12B-it-GGUF](https://huggingface.co/ggml-org/gemma-4-12B-it-GGUF) | 2026-06-03 | 136,272 | Present | google/gemma-4-12B-it |
| [ggml-org/embeddinggemma-2-GGUF](https://huggingface.co/ggml-org/embeddinggemma-2-GGUF) | 2026-10-06 | 110,732 | Present | google/embeddinggemma-2 |
| [yuxinlu1/gemma-4-12B-coder-fable5-composer2.5-v1-GGUF](https://huggingface.co/yuxinlu1/gemma-4-12B-coder-fable5-composer2.5-v1-GGUF) | 2026-06-10 | 106,667 | Present | google/gemma-4-12B-it |
| [FINAL-Bench/POCKET-26B-GGUF](https://huggingface.co/FINAL-Bench/POCKET-26B-GGUF) | 2026-07-24 | 100,629 | Present | google/gemma-4-26B-A4B-it |

</details>

<details><summary>Mistral — 1 first-party, 2 additional quantization repositories</summary>

**First-party**

| Repository | Created (UTC) | Downloads (30d) | Catalog |
|---|---|---:|---|
| [mistralai/Shieldstral-1.0-3B](https://huggingface.co/mistralai/Shieldstral-1.0-3B) | 2026-07-16 | 103,042 | Present |

**Other direct quantizations**

| Repository | Created (UTC) | Downloads (30d) | Catalog | Declared parent |
|---|---|---:|---|---|
| [handy-computer/Voxtral-Mini-4B-Realtime-2602-gguf](https://huggingface.co/handy-computer/Voxtral-Mini-4B-Realtime-2602-gguf) | 2026-06-05 | 465,114 | Present | mistralai/Voxtral-Mini-4B-Realtime-2602 |
| [handy-computer/Voxtral-Mini-3B-2507-gguf](https://huggingface.co/handy-computer/Voxtral-Mini-3B-2507-gguf) | 2026-06-04 | 222,527 | Present | mistralai/Voxtral-Mini-3B-2507 |

</details>

<details><summary>MiniMax — 4 first-party, 14 additional quantization repositories</summary>

**First-party**

| Repository | Created (UTC) | Downloads (30d) | Catalog |
|---|---|---:|---|
| [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) | 2026-07-28 | 6,012,871 | Present |
| [MiniMaxAI/MiniMax-H3-Image](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image) | 2026-11-03 | 1,036,133 | Present |
| [MiniMaxAI/MiniMax-M3](https://huggingface.co/MiniMaxAI/MiniMax-M3) | 2026-06-02 | 334,245 | Present |
| [MiniMaxAI/MiniMax-M3.1-Flash](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash) | 2026-10-17 | 149,243 | Present |

**Other direct quantizations**

| Repository | Created (UTC) | Downloads (30d) | Catalog | Declared parent |
|---|---|---:|---|---|
| [unsloth/MiniMax-H3-GGUF](https://huggingface.co/unsloth/MiniMax-H3-GGUF) | 2026-08-07 | 3,674,003 | Present | MiniMaxAI/MiniMax-H3 |
| [VideoRebirthAI/hyperflow-h3-flash](https://huggingface.co/VideoRebirthAI/hyperflow-h3-flash) | 2026-11-04 | 952,617 | Present | MiniMaxAI/MiniMax-H3 |
| [Abiray/MiniMax-H3-Pruned-GGUF](https://huggingface.co/Abiray/MiniMax-H3-Pruned-GGUF) | 2026-08-07 | 944,647 | Present | MiniMaxAI/MiniMax-H3 |
| [Serveurperso/MiniMax-Music3-GGUF](https://huggingface.co/Serveurperso/MiniMax-Music3-GGUF) | 2026-08-14 | 917,557 | Present | MiniMaxAI/MiniMax-Music3 |
| [Hearthscale/MiniMax-H3-Image-GGUF](https://huggingface.co/Hearthscale/MiniMax-H3-Image-GGUF) | 2026-11-10 | 747,780 | Present | MiniMaxAI/MiniMax-H3-Image |
| [audio-cpp/MiniMax-Music3-GGUF](https://huggingface.co/audio-cpp/MiniMax-Music3-GGUF) | 2026-08-14 | 647,738 | Present | MiniMaxAI/MiniMax-Music3 |
| [Abiray/MiniMax-H3-Image-GGUF](https://huggingface.co/Abiray/MiniMax-H3-Image-GGUF) | 2026-11-10 | 317,457 | Present | MiniMaxAI/MiniMax-H3-Image |
| [leejet/MiniMax-H3-GGUF](https://huggingface.co/leejet/MiniMax-H3-GGUF) | 2026-08-03 | 312,974 | Present | MiniMaxAI/MiniMax-H3 |
| [scragnog/MiniMax-Music3-GGUF](https://huggingface.co/scragnog/MiniMax-Music3-GGUF) | 2026-08-13 | 294,129 | Present | MiniMaxAI/MiniMax-Music3 |
| [unsloth/MiniMax-H3-Image-GGUF](https://huggingface.co/unsloth/MiniMax-H3-Image-GGUF) | 2026-11-12 | 288,328 | Present | MiniMaxAI/MiniMax-H3-Image |
| [Abiray/MiniMax-H3-GGUF](https://huggingface.co/Abiray/MiniMax-H3-GGUF) | 2026-08-03 | 189,879 | Present | MiniMaxAI/MiniMax-H3 |
| [Abiray/Minimax-H3-nvfp4-INT4-INT8-Convrot](https://huggingface.co/Abiray/Minimax-H3-nvfp4-INT4-INT8-Convrot) | 2026-08-03 | 130,436 | Present | MiniMaxAI/MiniMax-H3 |
| [molbal/MiniMax-H3-GGUF](https://huggingface.co/molbal/MiniMax-H3-GGUF) | 2026-08-03 | 109,776 | Present | MiniMaxAI/MiniMax-H3 |
| [webmp3/Sakura-MiniMax-H3-Uncensored-GGUF](https://huggingface.co/webmp3/Sakura-MiniMax-H3-Uncensored-GGUF) | 2026-10-13 | 108,923 | Present | MiniMaxAI/MiniMax-H3 |

</details>

<details><summary>Granite — 5 first-party, 3 additional quantization repositories</summary>

**First-party**

| Repository | Created (UTC) | Downloads (30d) | Catalog |
|---|---|---:|---|
| [ibm-granite/granite-timeseries-patchtst-fm-r2](https://huggingface.co/ibm-granite/granite-timeseries-patchtst-fm-r2) | 2026-08-07 | 1,024,428 | Present |
| [ibm-granite/granite-4.2-8b-GGUF](https://huggingface.co/ibm-granite/granite-4.2-8b-GGUF) | 2026-08-12 | 283,904 | Present |
| [ibm-granite/granite-4.2-3b-GGUF](https://huggingface.co/ibm-granite/granite-4.2-3b-GGUF) | 2026-08-12 | 270,139 | Present |
| [ibm-granite/granite-4.2-30b-GGUF](https://huggingface.co/ibm-granite/granite-4.2-30b-GGUF) | 2026-08-12 | 245,156 | Present |
| [ibm-granite/granite-speech-5.0-470m-turboctc-nc](https://huggingface.co/ibm-granite/granite-speech-5.0-470m-turboctc-nc) | 2026-08-17 | 104,423 | Present |

**Other direct quantizations**

| Repository | Created (UTC) | Downloads (30d) | Catalog | Declared parent |
|---|---|---:|---|---|
| [taronaeo/Granite-4.2-3B-BE-GGUF](https://huggingface.co/taronaeo/Granite-4.2-3B-BE-GGUF) | 2026-10-18 | 359,038 | Present | ibm-granite/granite-4.2-3b |
| [taronaeo/Granite-4.0-3B-Vision-BE-GGUF](https://huggingface.co/taronaeo/Granite-4.0-3B-Vision-BE-GGUF) | 2026-10-18 | 266,015 | Present | ibm-granite/granite-4.0-3b-vision |
| [taronaeo/Granite-4.0-H-1B-BE-GGUF](https://huggingface.co/taronaeo/Granite-4.0-H-1B-BE-GGUF) | 2026-10-18 | 160,293 | Present | ibm-granite/granite-4.0-h-1b |

</details>

**GGUFs of distilled checkpoints.** A supplementary scan for `distill` in recent, high-download repository names returned the following nine. All nine are covered. The Empero 2B, 4B and 9B distribution cards use shorter upstream aliases that the Hub resolves to their respective Distill repositories. Ma7ee7’s 4B checkpoint uses a different Qwen3-4B-Thinking-2507 student. For KhazarAI’s Fable-named checkpoint the pinned card describes coding traces without identifying the teacher. IsValorum also changed model behavior after Empero’s 35B distillation. Repository names identify candidates here. The model cards provide the publisher’s training claims. Students with other naming conventions, including Muse Glimmer, require separate identification.

| Repository | Created (UTC) | Downloads (30d) | Reported direct base |
|---|---|---:|---|
| [empero-ai/Qwen3.8-35B-A3B-Distill-GGUF](https://huggingface.co/empero-ai/Qwen3.8-35B-A3B-Distill-GGUF) | 2026-09-16 | 1,285,670 | empero-ai/Qwen3.8-35B-A3B-Distill |
| [bartowski/MiMo-V2.6-Distill-Qwen-9B-GGUF](https://huggingface.co/bartowski/MiMo-V2.6-Distill-Qwen-9B-GGUF) | 2026-09-21 | 991,748 | XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B |
| [mradermacher/Qwen3.8-27B-Fable-5-Coding-Distilled-GGUF](https://huggingface.co/mradermacher/Qwen3.8-27B-Fable-5-Coding-Distilled-GGUF) | 2026-09-16 | 915,329 | khazarai/Qwen3.8-27B-Fable-5-Coding-Distilled |
| [mradermacher/Qwen3.5-9B-Kimi-k3-Distilled-GGUF](https://huggingface.co/mradermacher/Qwen3.5-9B-Kimi-k3-Distilled-GGUF) | 2026-09-16 | 879,346 | khazarai/Qwen3.5-9B-Kimi-k3-Distilled |
| [empero-ai/Qwen3.8-4B-Distill-GGUF](https://huggingface.co/empero-ai/Qwen3.8-4B-Distill-GGUF) | 2026-08-15 | 564,504 | empero-ai/Qwen3.8-4B-Distill |
| [empero-ai/Qwen3.8-9B-Distill-GGUF](https://huggingface.co/empero-ai/Qwen3.8-9B-Distill-GGUF) | 2026-08-15 | 524,464 | empero-ai/Qwen3.8-9B-Distill |
| [empero-ai/Qwen3.8-2B-Distill-GGUF](https://huggingface.co/empero-ai/Qwen3.8-2B-Distill-GGUF) | 2026-08-15 | 517,117 | empero-ai/Qwen3.8-2B-Distill |
| [Ma7ee7/Qwen3.8_4B_Distilled_GGUF](https://huggingface.co/Ma7ee7/Qwen3.8_4B_Distilled_GGUF) | 2026-07-30 | 129,168 | Ma7ee7/Qwen3.8_4B_Distilled |
| [IsValorum/Qwen3.8-35B-A3B-Distill-MTP-APEX-I-MiniPlus-V2.1-Abliterated-GGUF](https://huggingface.co/IsValorum/Qwen3.8-35B-A3B-Distill-MTP-APEX-I-MiniPlus-V2.1-Abliterated-GGUF) | 2026-09-20 | 122,476 | empero-ai/Qwen3.8-35B-A3B-Distill |
