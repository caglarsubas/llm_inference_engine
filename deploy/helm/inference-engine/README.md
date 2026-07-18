# Standalone Orchestra model-plane chart

This chart deploys `llm_inference_engine` as a tenant-owned model-plane service.
It is deliberately separate from the Orchestra control-plane release and the
tenant runtime-host release.

The resulting request path is:

```text
tenant runtime -> internal model-plane Service -> approved model backends
                         |
                         +-> asynchronous evidence to Orchestra
```

Orchestra remains outside synchronous production inference. Tenant CI/CD or
GitOps owns installation, policy activation, Secret rotation, upgrade, and
rollback.

## Resource boundary

The chart creates only:

- a ServiceAccount with token automount disabled;
- an internal ClusterIP Service plus a headless Service;
- a StatefulSet with one retained last-known-good policy PVC per replica;
- a PodDisruptionBudget;
- a default-deny workload NetworkPolicy with explicit DNS, caller, Sentinel,
  model/backend, observation, and OTLP destinations; and
- an optional ServiceMonitor.

It creates no Secret, external datastore, public Route/Ingress, tenant runtime,
or Orchestra control-plane workload. The customer pre-creates every referenced
Secret, ConfigMap, PVC, StorageClass, namespace-wide default deny, and external
dependency.

## OpenShift production contract

[`values.openshift-production.yaml`](values.openshift-production.yaml) is the
model-plane overlay for `orchestra-ocp-4.20-amd64-v1`. It is intentionally not
renderable unchanged. A customer overlay must provide:

- a mirrored UBI9 image by exact digest and its pull Secret;
- release, deployment, organization, and environment bindings;
- separate auth, signed-routing, observation, and Sentinel Secret references;
- an existing trusted-CA ConfigMap;
- a StorageClass plus explicit acknowledgement that the external storage
  operator owns backup and restore;
- separate runtime and monitoring ingress plus
  Sentinel/model/observation/OTLP egress rules; and
- acknowledgement that namespace-wide default deny exists before Helm runs.

The profile also requires two or more replicas, `restricted-v2` identity
delegation, a read-only root filesystem, dropped capabilities,
`RuntimeDefault` seccomp, persistent LKG state, signed policy enforcement,
deployment-shared Sentinel limits, HTTPS observation/OTLP, a PDB, topology
spread, resource bounds, and a ServiceMonitor.

A successful render means those declared inputs are complete. It does not prove
OpenShift CNI/CSI behavior, Secret-operator convergence, pod/node loss,
Sentinel failover, backup/restore, load, RPO/RTO, or soak. Until those drills run
on the pinned cluster profile, the status remains `declared-not-certified`.

## Required objects

The chart reads values from existing objects and never accepts their contents
in `values.yaml`.

| Reference | Required keys | Purpose |
|---|---|---|
| `auth.existingSecretName` | `auth_keys.json` | Engine bearer/admin keys and rotation metadata |
| `routing.artifactsSecretName` | `model_routing_policy.json`, `model_routing_trust.json`, `model_routing_pricing.json` | Signed desired state, purpose-specific trust, and cost catalog |
| `observation.apiKeySecretName` | `api-key` | Deployment-bound `model-plane:observe` credential |
| `routing.sharedRateLimit.existingSecretName` | `sentinel-config.json` | Strict Sentinel discovery, TLS, credentials, and replica-ack contract |
| `trustedCA.configMapName` | `ca-bundle.crt` | Customer and internal trust roots |

Backend credentials belong in `extraEnvFrom.secretRef` or a customer-defined
Secret volume. `extraEnv` is for non-secret configuration and cannot override
chart-managed auth, routing, observation, OTLP, model-store, or trust variables.

Set `modelBackends.mode` to `remote`, `mounted`, or `hybrid`. Remote and hybrid
profiles require a dedicated model-backend egress lane. Mounted and hybrid
profiles require at least one tenant-owned read-only model PVC. This prevents a
render from claiming a model plane without declaring how inference is reached.

The Sentinel document must select TLS, at least three discovery endpoints, and
a satisfiable peer threshold. Its `caFile` should resolve to
`/etc/orchestra/ca/ca-bundle.crt`. The engine validates that document and fails
closed; the chart does not provision or promote Sentinel/Valkey nodes.

## Render and install

First create a customer overlay containing references and NetworkPolicy peers,
not credential values. Then render against the APIs installed in the target
cluster:

```bash
helm lint ./deploy/helm/inference-engine \
  -f ./deploy/helm/inference-engine/values.openshift-production.yaml \
  -f ./customer-model-plane.yaml

helm template orchestra-model-plane ./deploy/helm/inference-engine \
  --namespace orchestra-model-plane \
  --api-versions monitoring.coreos.com/v1 \
  -f ./deploy/helm/inference-engine/values.openshift-production.yaml \
  -f ./customer-model-plane.yaml > rendered-model-plane.yaml
```

Inspect the render and apply customer admission/signature policy before install:

```bash
helm upgrade --install orchestra-model-plane ./deploy/helm/inference-engine \
  --namespace orchestra-model-plane \
  --atomic --wait --timeout 20m \
  -f ./deploy/helm/inference-engine/values.openshift-production.yaml \
  -f ./customer-model-plane.yaml
```

The repository CI contract can be repeated locally without credentials:

```bash
./deploy/helm/inference-engine/ci/render-production-profile.sh \
  /tmp/orchestra-model-plane-profile.yaml
```

That script uses synthetic references, proves the strict render, and verifies
that weakening core controls fails. It is a chart test, not a cluster test.

## Rotation and rollback

Prefer immutable, revisioned Secret names. When the customer Secret operator
updates an existing object, change `rolloutId` so every pod receives a new pod
template and the StatefulSet drains predecessors. Keep old/new identities and
trust in overlap until all replicas use the replacement, then retire the old
material and run negative probes.

The observation key file is reread for each dispatch. Auth keys and routing
artifacts use authenticated atomic reload endpoints after projected files
converge; initial/recovery activation and Sentinel configuration changes still
require a rollout. Never assume Secret projection alone changes in-memory
state.

For image rollback, perform a new Helm upgrade with the previous image digest,
the current Secrets/trust, and a fresh `rolloutId`. A raw Helm rollback can
restore stale object references. For routing rollback, issue a new higher signed
policy revision containing the previous route intent; never replay an older
revision over a newer per-replica LKG.

Image publication, SBOM, provenance, signature, relocation, and the UBI/FIPS
boundary are documented in
[`docs/CONTAINER_IMAGES.md`](../../../docs/CONTAINER_IMAGES.md).
