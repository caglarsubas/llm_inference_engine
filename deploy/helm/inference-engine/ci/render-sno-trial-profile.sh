#!/usr/bin/env bash
set -euo pipefail

script_dir=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)
chart=$(CDPATH='' cd -- "$script_dir/.." && pwd)
helm_bin=${HELM_BIN:-helm}
output=${1:-}
workdir=$(mktemp -d "${TMPDIR:-/tmp}/orchestra-model-plane-sno.XXXXXX")
trap 'rm -rf "$workdir"' EXIT HUP INT TERM

manifest=${output:-"$workdir/openshift-sno-model-plane.yaml"}
if [ -n "$output" ]; then
  mkdir -p "$(dirname -- "$output")"
fi

if "$helm_bin" template orchestra-model-plane "$chart" \
  --namespace orchestra-model-plane \
  -f "$chart/values.openshift-sno-trial.yaml" >/dev/null 2>&1; then
  echo "OpenShift SNO trial values rendered without a released image digest" >&2
  exit 1
fi

required=(
  --set image.digest=sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
)

render_profile() {
  "$helm_bin" template orchestra-model-plane "$chart" \
    --namespace orchestra-model-plane \
    -f "$chart/values.openshift-sno-trial.yaml" "${required[@]}" "$@"
}

expect_profile_failure() {
  local description=$1
  shift
  if render_profile "$@" >/dev/null 2>&1; then
    echo "OpenShift SNO trial profile accepted ${description}" >&2
    exit 1
  fi
}

"$helm_bin" lint "$chart" -f "$chart/values.openshift-sno-trial.yaml" \
  "${required[@]}"
render_profile >"$manifest"

grep -qF 'kind: StatefulSet' "$manifest"
grep -qF 'replicas: 1' "$manifest"
grep -qF \
  'prometa.io/engineering-trial-profile-id: "orchestra-ocp-sno-trial-amd64-v1"' \
  "$manifest"
grep -qF \
  'image: ghcr.io/caglarsubas/llm_inference_engine/inference-engine-ubi@sha256:aaaaaaaa' \
  "$manifest"
grep -qF 'name: OTEL_EXPORTER_OTLP_PROTOCOL' "$manifest"
grep -qF 'value: "http/protobuf"' "$manifest"
grep -qF 'name: OTEL_EXPORTER_OTLP_HEADERS' "$manifest"
grep -qF 'name: "orchestra-model-plane-trial-otlp"' "$manifest"
grep -qF 'cpu: 500m' "$manifest"
grep -qF 'memory: 4096Mi' "$manifest"
grep -qF 'cpu: 1500m' "$manifest"
grep -qF 'memory: 8192Mi' "$manifest"
grep -qF 'automountServiceAccountToken: false' "$manifest"
grep -qF 'readOnlyRootFilesystem: true' "$manifest"
grep -qF 'allowPrivilegeEscalation: false' "$manifest"
grep -qF 'type: RuntimeDefault' "$manifest"
grep -qF 'dns.operator.openshift.io/daemonset-dns: default' "$manifest"
grep -qF 'port: 6379' "$manifest"
grep -qF 'port: 8080' "$manifest"
grep -qF 'port: 3443' "$manifest"

for kind in ServiceAccount Service StatefulSet NetworkPolicy; do
  grep -qF "kind: $kind" "$manifest"
done
if grep -Eq '^kind: (Secret|Route|Ingress|Deployment|PodDisruptionBudget|ServiceMonitor)$' \
  "$manifest"; then
  echo "SNO trial render emitted a forbidden object" >&2
  exit 1
fi
if grep -qF 'prometa.io/production-profile-id:' "$manifest"; then
  echo "SNO trial render claimed the production profile" >&2
  exit 1
fi
if grep -Eq '(^|[[:space:]])(runAsUser|runAsGroup|fsGroup):' "$manifest"; then
  echo "SNO trial render pinned an identity managed by restricted-v2" >&2
  exit 1
fi

expect_profile_failure "the production profile" \
  --set productionProfile.enabled=true
expect_profile_failure "an unknown profile ID" \
  --set engineeringTrialProfile.profileId=another-profile
expect_profile_failure "missing source-only acknowledgement" \
  --set engineeringTrialProfile.sourceOnlyAcknowledged=false
expect_profile_failure "two replicas" --set replicaCount=2
expect_profile_failure "a production environment" --set targetEnvironment=prod
expect_profile_failure "a mutable image" --set-string image.digest=
expect_profile_failure "an unrestricted workload surface" \
  --set workloadSurface.profileId=unrestricted
expect_profile_failure "replica-local rate limits" \
  --set routing.rateLimitScope=process-replica \
  --set-string routing.sharedRateLimit.existingSecretName=
expect_profile_failure "insecure shared state" \
  --set routing.sharedRateLimit.allowInsecureRedis=true
expect_profile_failure "disabled observation" --set observation.enabled=false
expect_profile_failure "gRPC export to the platform HTTP route" \
  --set otel.protocol=grpc
expect_profile_failure "unauthenticated OTLP export" \
  --set-string otel.headersSecretName=
expect_profile_failure "an incomplete OTLP Secret reference" \
  --set-string otel.headersSecretKey=
expect_profile_failure "plaintext serving" --set serverTls.enabled=false \
  --set-string serverTls.existingSecret= \
  --set-string serverTls.rolloutId=
expect_profile_failure "fixed UID mode" --set securityContextMode=fixed
expect_profile_failure "service-account token automount" \
  --set serviceAccount.automountServiceAccountToken=true
expect_profile_failure "disabled NetworkPolicy" --set networkPolicy.enabled=false
expect_profile_failure "a mounted model backend" --set modelBackends.mode=mounted
expect_profile_failure "a production backup claim" \
  --set persistence.externalBackupAcknowledged=true
expect_profile_failure "a PodDisruptionBudget on one node" \
  --set podDisruptionBudget.enabled=true
expect_profile_failure "topology spread on one node" \
  --set topologySpread.enabled=true
expect_profile_failure "resource request drift" \
  --set resources.requests.cpu=750m

echo "Source-only OpenShift SNO model-plane render passed: $manifest"
