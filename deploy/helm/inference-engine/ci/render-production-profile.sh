#!/usr/bin/env bash
set -euo pipefail

script_dir=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)
chart=$(CDPATH='' cd -- "$script_dir/.." && pwd)
helm_bin=${HELM_BIN:-helm}
output=${1:-}
workdir=$(mktemp -d "${TMPDIR:-/tmp}/orchestra-model-plane-chart.XXXXXX")
trap 'rm -rf "$workdir"' EXIT HUP INT TERM

manifest=${output:-"$workdir/openshift-model-plane.yaml"}
if [ -n "$output" ]; then
  mkdir -p "$(dirname -- "$output")"
fi

if "$helm_bin" template orchestra-model-plane "$chart" \
  -f "$chart/values.openshift-production.yaml" >/dev/null 2>&1; then
  echo "OpenShift production values rendered without customer inputs" >&2
  exit 1
fi

base=(
  --set deploymentId=tenant-model-plane-staging
  --set targetEnvironment=staging
  --set rolloutId=staging-v1
  --set auth.existingSecretName=engine-auth
  --set routing.artifactsSecretName=engine-routing
  --set observation.endpoint=https://orchestra.example.test/api/model-routing-observations
  --set observation.apiKeySecretName=engine-observer
)
"$helm_bin" lint "$chart" "${base[@]}"
"$helm_bin" template model-plane-staging "$chart" "${base[@]}" \
  >"$workdir/staging.yaml"

required=(
  --set productionProfile.namespaceDefaultDenyAcknowledged=true
  --set image.repository=registry.example.test/orchestra/inference-engine-ubi
  --set image.digest=sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
  --set deploymentId=tenant-model-plane
  --set rolloutId=release-2026-07
  --set auth.existingSecretName=engine-auth
  --set routing.artifactsSecretName=engine-routing
  --set routing.expectedOrgId=org-tenant
  --set routing.sharedRateLimit.existingSecretName=engine-sentinel
  --set 'routing.sharedRateLimit.networkPolicyEgress[0].to[0].ipBlock.cidr=10.40.0.0/16'
  --set 'routing.sharedRateLimit.networkPolicyEgress[0].ports[0].protocol=TCP'
  --set 'routing.sharedRateLimit.networkPolicyEgress[0].ports[0].port=26379'
  --set 'routing.sharedRateLimit.networkPolicyEgress[1].to[0].ipBlock.cidr=10.40.0.0/16'
  --set 'routing.sharedRateLimit.networkPolicyEgress[1].ports[0].protocol=TCP'
  --set 'routing.sharedRateLimit.networkPolicyEgress[1].ports[0].port=6379'
  --set observation.endpoint=https://orchestra.platform.svc/api/model-routing-observations
  --set observation.apiKeySecretName=engine-observer
  --set otel.endpoint=https://otel-collector.observability.svc:4317
  --set modelBackends.mode=remote
  --set 'modelBackends.networkPolicyEgress[0].to[0].ipBlock.cidr=10.60.0.0/16'
  --set 'modelBackends.networkPolicyEgress[0].ports[0].protocol=TCP'
  --set 'modelBackends.networkPolicyEgress[0].ports[0].port=8443'
  --set persistence.storageClassName=ocs-storagecluster-ceph-rbd
  --set persistence.externalBackupAcknowledged=true
  --set trustedCA.configMapName=orchestra-model-plane-ca
  --set 'networkPolicy.runtimeIngressFrom[0].namespaceSelector.matchLabels.kubernetes\.io/metadata\.name=tenant-runtime'
  --set 'networkPolicy.runtimeIngressFrom[0].podSelector.matchLabels.app\.kubernetes\.io/name=orchestra-runtime'
  --set 'networkPolicy.monitoringIngressFrom[0].namespaceSelector.matchLabels.kubernetes\.io/metadata\.name=openshift-user-workload-monitoring'
  --set 'networkPolicy.observationEgress[0].to[0].ipBlock.cidr=10.50.0.0/16'
  --set 'networkPolicy.observationEgress[0].ports[0].protocol=TCP'
  --set 'networkPolicy.observationEgress[0].ports[0].port=443'
  --set 'networkPolicy.otelEgress[0].to[0].ipBlock.cidr=10.70.0.0/16'
  --set 'networkPolicy.otelEgress[0].ports[0].protocol=TCP'
  --set 'networkPolicy.otelEgress[0].ports[0].port=4317'
)

render_profile() {
  "$helm_bin" template orchestra-model-plane "$chart" \
    --namespace orchestra-model-plane \
    --api-versions monitoring.coreos.com/v1 \
    -f "$chart/values.openshift-production.yaml" \
    "${required[@]}" "$@"
}

expect_profile_failure() {
  local description=$1
  shift
  if render_profile "$@" >/dev/null 2>&1; then
    echo "OpenShift production profile accepted ${description}" >&2
    exit 1
  fi
}

"$helm_bin" lint "$chart" -f "$chart/values.openshift-production.yaml" \
  "${required[@]}"
render_profile >"$manifest"

grep -qF 'kind: StatefulSet' "$manifest"
grep -qF 'replicas: 2' "$manifest"
grep -qF 'prometa.io/production-profile-id: "orchestra-ocp-4.20-amd64-v1"' "$manifest"
grep -qF 'image: registry.example.test/orchestra/inference-engine-ubi@sha256:aaaaaaaa' "$manifest"
grep -qF 'orchestra.prometa.ai/rollout-id: "release-2026-07"' "$manifest"
grep -qF 'automountServiceAccountToken: false' "$manifest"
grep -qF 'readOnlyRootFilesystem: true' "$manifest"
grep -qF 'allowPrivilegeEscalation: false' "$manifest"
grep -qF 'type: RuntimeDefault' "$manifest"
grep -qF 'MODEL_ROUTING_RATE_LIMIT_SENTINEL_CONFIG_FILE' "$manifest"
grep -qF 'MODEL_PLANE_OBSERVATION_VERSION' "$manifest"
grep -qF 'OTEL_EXPORTER_OTLP_ENDPOINT' "$manifest"
grep -qF 'secretName: engine-auth' "$manifest"
grep -qF 'secretName: engine-routing' "$manifest"
grep -qF 'secretName: engine-observer' "$manifest"
grep -qF 'secretName: engine-sentinel' "$manifest"
grep -qF 'configMap:' "$manifest"
grep -qF 'name: orchestra-model-plane-ca' "$manifest"
grep -qF 'storageClassName: "ocs-storagecluster-ceph-rbd"' "$manifest"
grep -qF 'whenDeleted: Retain' "$manifest"
grep -qF 'whenScaled: Retain' "$manifest"
grep -qF 'dns.operator.openshift.io/daemonset-dns: default' "$manifest"
grep -qF 'app.kubernetes.io/instance: orchestra-model-plane' "$manifest"
for port in 26379 6379 8443 443 4317; do
  grep -qF "port: $port" "$manifest"
done
for kind in ServiceAccount Service StatefulSet PodDisruptionBudget NetworkPolicy ServiceMonitor; do
  grep -qF "kind: $kind" "$manifest"
done
if grep -Eq '^kind: (Secret|Route|Ingress|Deployment)$' "$manifest"; then
  echo "Standalone model-plane chart rendered credentials or a public/control-plane workload" >&2
  exit 1
fi
if grep -Eq '(^|[[:space:]])(runAsUser|runAsGroup|fsGroup):' "$manifest"; then
  echo "OpenShift render pinned an identity managed by restricted-v2" >&2
  exit 1
fi
if grep -qF 'k8s-app: kube-dns' "$manifest"; then
  echo "OpenShift render retained the vanilla Kubernetes DNS selector" >&2
  exit 1
fi

expect_profile_failure "a mutable image tag" --set-string image.digest=
expect_profile_failure "one replica" --set replicaCount=1
expect_profile_failure "the direct shared-state backend" --set routing.sharedRateLimit.backend=direct
expect_profile_failure "insecure shared state" --set routing.sharedRateLimit.allowInsecureRedis=true
expect_profile_failure "disabled OTLP evidence" --set otel.enabled=false
expect_profile_failure "fixed UID mode" --set securityContextMode=fixed
expect_profile_failure "observation contract v1" --set observation.version=1
expect_profile_failure "a mismatched routing audience" \
  --set routing.expectedAudience=another-audience
expect_profile_failure "a rollout-deadlocking PDB" \
  --set podDisruptionBudget.minAvailable=2
expect_profile_failure "soft topology spread" \
  --set topologySpread.whenUnsatisfiable=ScheduleAnyway
expect_profile_failure "mounted backends without a model PVC" \
  --set modelBackends.mode=mounted
expect_profile_failure "unacknowledged namespace default deny" \
  --set productionProfile.namespaceDefaultDenyAcknowledged=false
expect_profile_failure "unowned backup responsibility" \
  --set persistence.externalBackupAcknowledged=false
expect_profile_failure "disabled NetworkPolicy" --set networkPolicy.enabled=false

echo "Standalone OpenShift model-plane profile render passed: $manifest"
