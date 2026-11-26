# Local route qualification

On 26 November 2026, the gateway ran three labeled checks per route against
17 installed builds. Twelve passed the stated workload and five failed. All
51 labeled cases completed. The results include prompts, expected answers,
visible model outputs, finish reasons, scores, timestamps and build digests.
[Per-route results](results.json)

## Workloads

- **Short text extraction.** Each chat model received three independent notes.
  The requested values were the current loading bay, the rejected lot ID and
  the final carton count. Each visible answer had to equal the labeled string
  after trimming surrounding whitespace. Temperature was zero, top-p was 1,
  seed was 17 and the output limit was 128 tokens. Model-native reasoning
  behavior was retained.
- **Short text retrieval.** Each embedding route received three passages and
  three paraphrased queries in one batch through `POST /v1/embeddings`. Every
  vector had to be finite, nonzero and of the dimension declared by the local
  model. Every query had to assign its labeled passage a strictly higher
  cosine similarity than both alternatives. The test used the raw strings in
  the results file.

Of 15 chat routes, 10 passed all three extraction cases. Ministral 3B wrapped
one correct value in Markdown, so it passed two exact matches. Gemma 4 E2B
reached the token limit with empty visible content on two cases. Qwen3.6 27B
did so on all three. Both Nemotron routes returned empty visible content
with a stop finish reason on all three cases. The scores use visible content
alone. Different budgets and prompts can produce different results.

Both embedding routes ranked all three labeled passages first. EmbeddingGemma
produced six valid 768-dimensional vectors and Qwen3 Embedding produced six
valid 1,024-dimensional vectors. These three short English passages evaluate
basic retrieval on this fixture. Production accuracy, multilingual quality,
tool use, long contexts and performance under concurrent load require their
own evaluation.

## Runtime and binding

A dedicated loopback inference gateway used an isolated Ollama 0.32.14 daemon
with cloud access disabled, a 4,096-token context, one parallel slot and a
one-model residency limit. Copied Ollama manifests pointed at the existing
local blobs. The gateway made sequential, non-streaming requests with model
substitution disabled. Sixteen routes were tested this way. The remaining
MiniCPM-V route used the same gateway and the local Docker Model Runner.
The command exercised text extraction on MiniCPM-V. A separate benchmark
exercised Gemma 3 4B and 12B on images.
[Gemma document extraction](../gemma-document-extraction/)

For Ollama, the gateway-reported manifest digest was compared with the
daemon's installed tag before requests and the resident model after requests.
For Docker Model Runner, the gateway's reported model digest was corroborated
against the matching model in its native `/api/tags` inventory before and
after the three calls. All 17 digests matched. The timing fields include model
loading when it occurred and should be interpreted per request.

## Reproduce

Start a dedicated loopback Ollama daemon with the selected local manifests,
`OLLAMA_NO_CLOUD=1`, `OLLAMA_MAX_LOADED_MODELS=1` and
`OLLAMA_NUM_PARALLEL=1`. Point a local inference gateway at it using
`OLLAMA_HTTP_ENDPOINT` and `N_CTX=4096`. Disable auth, model substitution
and external fallbacks on that isolated gateway. To include a Docker Model
Runner route, register its local OpenAI-compatible endpoint in the gateway's
vLLM manifest. Supply exact advertised route IDs, the local URLs and a new
output path to the driver.

```sh
python scripts/qualify_local_model_routes.py \
  --gateway-url "$GATEWAY_URL" --ollama-url "$OLLAMA_URL" \
  --model ministral-3:8b --model qwen3-embedding:0.6b \
  --output /tmp/local-route-checks.json
```

The `requested_routes` array in the original results gives the full roster
and order. A route with a request failure or an unconfirmed artifact is
recorded as `incomplete`. The driver proceeds to the remaining requests and
checkpoints after each route. [Qualification driver](../../../scripts/qualify_local_model_routes.py)
