# Container images and supply-chain contract

The inference engine is a tenant-deployed model-plane component. Publishing an
image does not place the Orchestra control plane in the synchronous inference
path, and the platform does not gain rollout authority.

## Published variants

| Variant | Canonical repository | Dockerfile | Intended deployment |
|---|---|---|---|
| Debian | `ghcr.io/caglarsubas/llm_inference_engine/inference-engine` | `Dockerfile` | General Kubernetes, VMs, and Compose |
| UBI9 | `ghcr.io/caglarsubas/llm_inference_engine/inference-engine-ubi` | `Dockerfile.ubi` | Red Hat UBI/OpenShift estates |

The gated `.github/workflows/publish-images.yml` workflow publishes only on a
`v*` tag or a manual dispatch. A merge by itself does not publish an image.
Each run creates the requested tag and `sha-<12-character-commit>` tag, but the
workflow summary's `repository@sha256:<digest>` reference is the production
deployment identity.

Canonical workflow artifacts are currently `linux/amd64`. Both pinned base
image indexes also contain `linux/arm64`; operators may build a native ARM
image from the same Dockerfile, but ARM is not a published/certified release
profile until it receives equivalent CI and load evidence.

## Runtime hardening

Both final images:

- contain no compiler toolchain from the llama-cpp-python build stage;
- run as a numeric non-root user by default;
- set `HOME` and `TMPDIR` to `/tmp`;
- keep code/dependencies group-readable and executable by GID 0 while making
  only `/state` group-writable;
- keep last-known-good routing state under the dedicated `/state` path; and
- expose a shallow `/v1/health` container health check.

CI overrides the image user with UID `1001230000`, GID `0`, makes the root
filesystem read-only, mounts only `/tmp` and `/state` as writable tmpfs paths,
then proves process identity, state persistence, root immutability, and health.
This matches the identity shape imposed by OpenShift `restricted-v2`. A real
cluster must still provide writable storage and the security context expected
by the deployment chart.

The UBI image is a supported UBI-based build. It is not, by that fact alone, a
Red Hat-certified image, and it does not assert that every Python dependency or
model backend is FIPS validated. FIPS claims require host crypto policy,
dependency validation, and deployment-specific certification.

## Build and smoke locally

Canonical builds include OpenTelemetry:

```bash
docker build --build-arg EXTRAS=otel -t inference-engine:debian .
docker build -f Dockerfile.ubi --build-arg EXTRAS=otel \
  -t inference-engine:ubi .

./scripts/container_smoke.sh inference-engine:debian
./scripts/container_smoke.sh inference-engine:ubi
```

Canonical CPU builds disable llama.cpp host-native instruction tuning. This
keeps an image built on a newer x86 runner portable to older tenant nodes and
avoids coupling the UBI compiler to assembler features detected on the build
host. Additional `CMAKE_ARGS`, including CUDA, are appended to that baseline.

Use a CUDA-capable builder and runtime host to create an accelerated Debian
variant:

```bash
docker build \
  --build-arg EXTRAS=otel \
  --build-arg CMAKE_ARGS=-DGGML_CUDA=on \
  -t inference-engine:cuda .
```

The canonical CPU image does not claim GPU runtime support. A CUDA release
profile needs its own base, driver compatibility matrix, and smoke/load proof.

## Verify before deployment

Resolve and pin the digest first:

```bash
IMAGE=ghcr.io/caglarsubas/llm_inference_engine/inference-engine
DIGEST=sha256:<digest-from-publish-workflow>
REF="${IMAGE}@${DIGEST}"
```

Verify the keyless signature and CycloneDX attestation against the publishing
workflow identity:

```bash
IDENTITY='^https://github\.com/caglarsubas/llm_inference_engine/\.github/workflows/publish-images\.yml@refs/(heads/main|tags/v.*)$'

cosign verify "${REF}" \
  --certificate-identity-regexp "${IDENTITY}" \
  --certificate-oidc-issuer https://token.actions.githubusercontent.com

cosign verify-attestation "${REF}" \
  --type cyclonedx \
  --certificate-identity-regexp "${IDENTITY}" \
  --certificate-oidc-issuer https://token.actions.githubusercontent.com
```

Admission policy should verify the same identity and require digest-pinned pod
references. Registry authentication and runtime credentials are separate:
image pull credentials never belong in the image, and engine/platform API keys
remain mounted runtime Secrets.

## Mirror or move into an air gap

The manual publish workflow accepts a registry namespace and uses
`REGISTRY_USER` / `REGISTRY_TOKEN` for a non-GHCR destination. For an existing
signed release, `scripts/relocate_images.sh` copies both variants and preserves
signatures and attestations:

```bash
cosign login ghcr.io -u <source-user> -p <source-token>
cosign login registry.customer.example -u <destination-user> -p <destination-token>

ORCHESTRA_ENGINE_TAG=v0.1.4 \
  ./scripts/relocate_images.sh copy registry.customer.example/orchestra
```

For disconnected transfer:

```bash
ORCHESTRA_ENGINE_TAG=v0.1.4 ./scripts/relocate_images.sh save ./engine-images
# Move ./engine-images across the boundary.
ORCHESTRA_ENGINE_TAG=v0.1.4 \
  ./scripts/relocate_images.sh load registry.airgap.example/orchestra ./engine-images
```

Verify the destination digest and signature after relocation. If the air-gap
policy cannot use Fulcio/Rekor material, re-sign inside the boundary with the
tenant's offline key and enforce that key at admission.

## Promotion responsibility

The engine image, signed routing policy, trust store, pricing catalog, and
runtime credentials are separate release inputs. Tenant CI/CD or GitOps pins
and deploys them, calls the engine reload endpoint when appropriate, and owns
rollback. Orchestra may issue policy and evaluate observed state; it remains
outside the request-time model path.
