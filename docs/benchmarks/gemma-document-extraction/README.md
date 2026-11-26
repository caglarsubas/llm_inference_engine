# Gemma 3 invoice extraction

**Gemma 3 12B passed the invoice fixture with 39 of 40 fields correct.**
It is the recommended first choice for supervised extraction on pages like these.
Gemma 3 4B got 37 fields correct and missed the predeclared pass threshold. For
this workload the fallback is review or abstention. Both models filled in USD
when the invoice omitted its currency.

| November 26, 2026 · Apple M5 Max, 128 GiB | Gemma 3 4B | Gemma 3 12B |
|---|---:|---:|
| Completed requests / valid JSON / schema-valid responses | 8 / 8 / 8 | 8 / 8 / 8 |
| Exact fields | 37/40 (92.5%) | 39/40 (97.5%) |
| Pages with all fields correct | 5/8 | 7/8 |
| First-token time, median / p95 | 0.85 / 2.46 s | 1.46 / 4.04 s |
| Complete gateway response, median / p95 | 1.26 / 2.89 s | 2.27 / 4.81 s |
| Peak macOS daemon-tree physical footprint | 3.99 GiB | 8.82 GiB |
| Peak allocation reported by Ollama | 3.58 GiB | 8.44 GiB |
| Workload result | Failed | Passed |

On the paid invoice (page 05), 4B returned the gross total of 384.00 even though
the remaining balance was 0.00. On page 06 it copied the payment deadline into
the absent issue date. On page 07 both models supplied a currency for which
the document gave no evidence. The raw predictions and per-page timing are
in the [measured results](results.json).

## Method

We defined the eight English synthetic invoice images, five exact labels per
image and pass thresholds before sending any requests. The pages include
distractor totals, invoice and purchase-order IDs, a buyer and seller, zero
balance, missing fields, rotated text and a blurred copy. Only the image,
common extraction instructions and JSON schema were sent to the model.
A pass required eight completed schema-valid JSON responses, seven fully correct
pages and at least 38 correct fields. A failed or incomplete stream earns zero
field and page credit. Dates use ISO format, amounts use two decimal places,
names preserve case, and missing values use JSON null.
[Scored images and ground truth](cases.json).

Both models ran serially via the same loopback gateway at revision
`f3383c9f40708561881b17e2b905d3047d61c969` and a dedicated Ollama 0.32.14
daemon. The daemon used a separate model directory with the deployed manifests
and shared local blobs. The gateway's streaming chat path forwarded each image
with a strict JSON schema, temperature 0, top-p 1, seed 21, a 192-token
output limit and model substitution disabled. Ollama reported a 32,768-token
context and Q4_K_M for both manifests. The driver verified the running model's
manifest digest against the installed tag after each completed response.

The first request for each model started cold. The seven subsequent requests
started warm, with median complete-response times of 1.26 s for 4B and 2.23 s
for 12B. The cold page took 2.89 s and 4.81 s respectively. Nearest-rank p95
is the maximum of eight observations here. The timing includes the gateway,
loopback transport, image processing and generation. Physical footprint sums
the daemon and its runner processes. Ollama's process list provides a separate
allocation measurement. Sampling targeted 250 ms, with all 47 and 83 samples
returning footprint measurements. Shorter peaks may be missed.

This comparison covers eight machine-rendered documents and one request per
model per page. Camera captures, handwriting, multilingual documents and
production traffic need separate evaluation. The result supports a
workload-specific selection for the two measured manifest digests.

## Reproduce

Use a dedicated local Ollama daemon containing the two Gemma 3 tags. Start a
gateway pointing to that daemon with a 32,768-token context and no model
substitution. Leave the daemon empty at startup. The driver releases each
model through Ollama after its eight requests. For an authenticated gateway,
supply its token in `INFERENCE_ENGINE_API_KEY`.

```sh
python scripts/benchmark_gemma_document_extraction.py \
  --gateway-url "$GATEWAY_URL" --dedicated-ollama-url "$OLLAMA_URL" \
  --ollama-pid "$OLLAMA_PID" --gateway-revision "$GATEWAY_COMMIT" \
  --output /tmp/gemma-invoice-results.json
```

The driver verifies each PNG checksum against the [fixture](cases.json) and
writes a new result file containing the fixture checksum, exact build identities,
timestamps, prompt, schema, unprocessed answer text, scores and sampled peaks.
