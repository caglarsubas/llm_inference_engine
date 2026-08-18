# Onboarding `qwen3.8:27b`

Captured 2026-08-18. This is an operator handoff for taking `qwen3.8:27b` from
"pulled" to "reachable by every tenant" on a deployment that enforces a signed
model-routing policy.

Keep three states separate, because a model can sit in the first two and still
refuse every tenant request:

- **served**: appears in `GET /v1/models.data`, and a chat call returns content.
- **routed**: the active signed policy carries a route whose `primaryModel` is
  this exact model id.
- **reachable**: the calling key carries an `org_id` matching the policy's
  `orgId`, and the route's limits admit the request.

## Engine side — done on the native host

No code in this repo declares a servable model list; the surface is discovered.
Making this model serve was operational, not a source change.

| Step | Detail |
|---|---|
| Runtime floor | The registry config declares `"requires": "0.32.12"`. Homebrew's ollama was 0.32.8 — below the floor. Upstream v0.32.14 is installed at `~/.local/ollama-0.32.14/`. |
| Managed daemon | `com.prometa.ollama` launchd agent now runs that binary. Port `11434` and store `~/.cache/inference_engine/ollama` are unchanged, so engine config stayed valid. |
| Weights | `ollama pull qwen3.8:27b` — 17.74 GB (16.81 GB model + 0.93 GB projector). |
| Fallback wiring | `OLLAMA_HTTP_ENDPOINT` was absent from `.env`. It is now set. Without it neither `qwen3.8:27b` nor the pre-existing `qwen3.6:27b` can serve at all. |

Both models fail the in-process GGUF load probe — `llama-cpp-python` 0.3.22
cannot open the architecture — so they resolve to the `ollama_http` backend.
That is the designed path, not a workaround.

Verify:

```bash
curl -s 127.0.0.1:11434/api/version
```

## Control plane — required, cannot be done from this repo

`model_routing.py` verifies signed policy bytes and never calls the control
plane, so the route below has to be issued and signed upstream.

Request a route with **this exact shape** in the policy's `routes` array:

```json
{
  "routeId": "qwen38-27b",
  "requestedModel": "qwen3.8:27b",
  "primaryModel": "qwen3.8:27b",
  "fallbackModels": [],
  "limits": {
    "maxInputTokens": null,
    "maxOutputTokens": null,
    "maxRequestsPerMinute": null,
    "maxCostMicrosPerRequest": null,
    "maxTokensPerMinute": null,
    "maxCostMicrosPerWindow": null,
    "budgetWindowSeconds": null
  },
  "candidateWeights": null,
  "shadowModel": null
}
```

Three traps, each of which fails differently:

1. **A wildcard route is not enough.** `_select_route` falls back to any route
   with `requestedModel: "*"`, and the served candidates then come from *that*
   route's `primaryModel` — not from what the caller asked for. A tenant
   requesting `qwen3.8:27b` under a wildcard-only policy is served the
   wildcard's model instead, with no error and a `200` response. The route must
   name the model.

2. **Pricing must land before the route, or the replica will not boot.** If the
   route sets `maxCostMicrosPerRequest` or `maxCostMicrosPerWindow` and the
   model is absent from `.model_routing_pricing.json`, startup raises
   `pricing_model_missing` — a fatal config error, not a per-request failure.
   The pricing file is operator-side and gitignored; the route is control-plane
   side. Ship pricing first. The entry now present locally:

   ```json
   { "model": "qwen3.8:27b",
     "inputCostMicrosPerMillionTokens": 0,
     "outputCostMicrosPerMillionTokens": 0 }
   ```

   Zero because the model is self-hosted, matching how `qwen3:32b` is priced in
   `.model_routing_pricing.example.json`. Revisit if the deployment bills
   internally for GPU time.

3. **Keys need an org identity.** `enforce_model_routing_request` raises
   `org_identity_missing` before it looks at any route when `identity.org_id`
   is `None`, and `org_identity_mismatch` when it differs from the policy's
   `orgId`. Seven of the eight keys in `.auth_keys.json` carried no `org_id`;
   all eight now carry `org-prometa`. Confirm that matches the `orgId` the
   control plane signs, and `MODEL_ROUTING_EXPECTED_ORG_ID` if it is set.

## Known limitation — vision survives only via `ollama_http`

The weights ship a projector layer. This engine has no projector support: the
`llama_cpp` adapter cannot consume one, and the descriptor built from the
on-disk manifest carries `params["projector_path"]` purely as a signal.

Vision works today only because the in-process path *cannot load the
architecture at all*, so `CompositeRegistry.resolve` falls through to
`ollama_http`, where ollama owns the projector. Verified end-to-end: an image
request through `/v1/chat/completions` returns a correct answer.

If a future `llama-cpp-python` gains this architecture, the probe starts
succeeding, resolution flips to the earlier in-process source, and **vision
degrades to text-only with no error**. `llama-cpp-python>=0.3.22` has no upper
bound, so a routine `uv sync` is enough to trigger it.

Deciding what to do about that is a routing-semantics call and is deliberately
left open. The options, in the order they narrow behaviour:

- have the composite prefer `ollama_http` for descriptors carrying
  `projector_path`, so multimodal models never silently lose half their input;
- teach the GGUF probe to reject projector-bearing manifests for the
  in-process path;
- pin an upper bound on `llama-cpp-python`, which only defers the question.

`test_projector_layer_is_surfaced_not_dropped` in `tests/test_registry.py`
locks in that the projector is at least visible.
