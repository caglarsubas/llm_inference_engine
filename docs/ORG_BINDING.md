# Org binding — one engine, one org

Captured 2026-08-24, from a question that had no answer written down: this
deployment carries auth keys for two customer orgs (`org-planeon` and `org-2`),
and someone was about to activate a signed model-routing policy on it.

The short version: **a policy-enforcing engine process serves exactly one org.**
Activating a policy on a deployment whose key file spans two orgs locks out
every tenant of the other one, on every request, immediately.

## The constraint

`ModelRoutingRuntimeState` holds one policy, not a collection keyed by org
([`model_routing_runtime.py:349`](../src/inference_engine/model_routing_runtime.py#L349)):

```python
class ModelRoutingRuntimeState:
    policy: ActivatedModelRoutingPolicy | None = None
```

and that policy names a single org — `org_id: StrictStr` on
`ModelRoutingPolicyClaims`
([`model_routing.py:174`](../src/inference_engine/model_routing.py#L174)),
not a list.

Request-time enforcement then compares the **calling key's** org against the
**active policy's** org, and refuses on any difference
([`model_routing_runtime.py:1452`](../src/inference_engine/model_routing_runtime.py#L1452)):

```python
if identity.org_id is None:
    raise ModelRoutingEnforcementError("org_identity_missing", ...)
if identity.org_id != claims.org_id:
    raise ModelRoutingEnforcementError("org_identity_mismatch", ...)
```

This is a hard refusal, not a degraded route or a fallback. So with a key file
spanning two orgs there are only three reachable states:

| Active policy | Result |
|---|---|
| bound to `org-planeon` | every `org-2` tenant refused — `org_identity_mismatch` |
| bound to `org-2` | every `org-planeon` tenant refused — same error |
| none | no governed routing for anyone; all traffic passes |

The third row is where this deployment sits today: `GET
/v1/admin/model-routing-policy` reports `active: false`,
`request_time_enforcement: false`, `route_count: 0`, and no policy file exists
on disk. That is why a two-org key file has been harmless so far.

## Two things that look like escape hatches and are not

**`org_binding_mode`** in the policy status is a label, not a switch. It reports
`"auth-key-org"` when auth is on and `"deployment-org"` when it is off, and
nothing reads it back
([`model_routing_status.py:74`](../src/inference_engine/model_routing_status.py#L74)).

**`allowed_org_ids`** on `ModelRoutingTrustEntry`
([`model_routing.py:209`](../src/inference_engine/model_routing.py#L209)) *is* a
list, which makes it read like multi-org support. It is not: it scopes which
orgs a **signing key** may issue policies for. One signer can serve both orgs
happily. One running engine still cannot.

Keep these three org comparisons distinct — they fail in different places:

| Comparison | Where | Meaning |
|---|---|---|
| `claims.org_id` ∈ `entry.allowed_org_ids` | [`model_routing.py:594`](../src/inference_engine/model_routing.py#L594) | may this signer issue for this org |
| `claims.org_id` == `MODEL_ROUTING_EXPECTED_ORG_ID` | [`model_routing.py:598`](../src/inference_engine/model_routing.py#L598) | is this the policy this deployment expects |
| `identity.org_id` == `claims.org_id` | [`model_routing_runtime.py:1457`](../src/inference_engine/model_routing_runtime.py#L1457) | may this **caller** use this policy |

Only the third involves the auth key file, and it is the one that produces a
per-tenant outage.

## What `org_id` does while no policy is active

Not nothing — which is why the field still deserves to be correct on every key
even on an unenforced deployment. It flows into:

- the **usage ledger** ([`api/_usage.py:101`](../src/inference_engine/api/_usage.py#L101)),
  so per-org token accounting and any rollup built on it are already split;
- **span attributes** as `planeon.org_id` on chat, completions, and embeddings;
- the **guardrail subject** payload, as `{"tenant": ..., "orgId": ...}`
  ([`guardrail.py:457`](../src/inference_engine/guardrail.py#L457)).

An `org_id` that is wrong today is invisible until it is suddenly an outage.

## Current deployment

| Org | Tenants |
|---|---|
| `org-planeon` | `dev`, `evals`, `planeon`, `declarai-auto-ml`, `deepfaked-claims-detection-v1`, `aishraq-platform-assistant`, `merge-forensics`, `gcp-audit`, `manufacturing-quality-management` |
| `org-2` | `football-transfer-copilot` |

`org-2` is a real second customer org, not a placeholder. Do not "fix" it to
`org-planeon` — that would file a customer's usage under the wrong org and give
them access under a policy written for someone else.

## When governed routing goes live

Split the deployment. One engine process per org, each with its own key file
and its own policy — which is the shape the status payload already assumes,
carrying `deployment_id` alongside `policy_id` and `org_id`. Running both orgs
under one policy is not a configuration that exists.

Until then, activating any policy on this box is a breaking change for one
customer or nine internal tenants. There is no ordering of the rollout that
avoids it while the key file spans two orgs.

## Shared compute, separately

Worth stating because it is adjacent and easy to conflate: org binding is an
authorization boundary, not an isolation one. Both orgs share one node, one GPU,
one `MEMORY_BUDGET_GB`, and one scheduler.
`SCHEDULER_MAX_QUEUE_PER_TENANT` bounds queue depth per tenant; it does not
reserve compute share. A long local generation for an internal eval delays a
customer request up to `SCHEDULER_QUEUE_TIMEOUT_SECONDS`, after which it is shed
with `429` and a `Retry-After`.

That is an appropriate arrangement for a POC a customer knows is a POC, and an
inappropriate one to leave undiscussed if they believe otherwise.

## See also

- [`QWEN38_ONBOARDING.md`](QWEN38_ONBOARDING.md) — the served / routed /
  **reachable** distinction, where reachability is exactly this org match
- [`PUBLIC_ENDPOINT.md`](PUBLIC_ENDPOINT.md) — the consumer-facing endpoint guide
- `README.md` § signed model-routing policy — envelope format and rotation
