{{- define "orchestra-inference-engine.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "orchestra-inference-engine.fullname" -}}
{{- if .Values.fullnameOverride -}}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s-%s" .Release.Name (include "orchestra-inference-engine.name" .) | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}

{{- define "orchestra-inference-engine.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "orchestra-inference-engine.selectorLabels" -}}
app.kubernetes.io/name: {{ include "orchestra-inference-engine.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/component: model-plane
{{- end -}}

{{- define "orchestra-inference-engine.labels" -}}
helm.sh/chart: {{ include "orchestra-inference-engine.chart" . }}
{{ include "orchestra-inference-engine.selectorLabels" . }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end -}}

{{- define "orchestra-inference-engine.serviceAccountName" -}}
{{- if .Values.serviceAccount.create -}}
{{- default (include "orchestra-inference-engine.fullname" .) .Values.serviceAccount.name -}}
{{- else -}}
{{- required "serviceAccount.name is required when serviceAccount.create=false" .Values.serviceAccount.name -}}
{{- end -}}
{{- end -}}

{{- define "orchestra-inference-engine.image" -}}
{{- $repository := required "image.repository is required" .Values.image.repository -}}
{{- if .Values.image.digest -}}
{{- if not (regexMatch "^sha256:[a-f0-9]{64}$" .Values.image.digest) -}}
{{- fail "image.digest must be lowercase sha256:<64 hex>" -}}
{{- end -}}
{{- printf "%s@%s" $repository .Values.image.digest -}}
{{- else -}}
{{- if .Values.productionProfile.enabled -}}
{{- fail "image.digest is required by the OpenShift production profile" -}}
{{- end -}}
{{- printf "%s:%s" $repository (default .Chart.AppVersion .Values.image.tag) -}}
{{- end -}}
{{- end -}}

{{- define "orchestra-inference-engine.validate" -}}
{{- $environment := required "targetEnvironment is required" .Values.targetEnvironment -}}
{{- if not (has $environment (list "dev" "test" "staging" "prod")) -}}
{{- fail "targetEnvironment must be one of dev, test, staging, or prod" -}}
{{- end -}}
{{- $_ := required "deploymentId is required" .Values.deploymentId -}}
{{- $_ = required "rolloutId is required" .Values.rolloutId -}}
{{- if lt (int .Values.replicaCount) 1 -}}
{{- fail "replicaCount must be at least 1" -}}
{{- end -}}
{{- if not (has .Values.securityContextMode (list "fixed" "openshift")) -}}
{{- fail "securityContextMode must be fixed or openshift" -}}
{{- end -}}
{{- if and .Values.auth.enabled (not .Values.auth.existingSecretName) -}}
{{- fail "auth.existingSecretName is required when auth.enabled=true" -}}
{{- end -}}
{{- if and .Values.auth.enabled (not .Values.auth.keysKey) -}}
{{- fail "auth.keysKey is required when auth.enabled=true" -}}
{{- end -}}
{{- if and .Values.routing.enabled (not .Values.routing.artifactsSecretName) -}}
{{- fail "routing.artifactsSecretName is required when routing.enabled=true" -}}
{{- end -}}
{{- if and .Values.routing.enabled (or (not .Values.routing.policyKey) (not .Values.routing.trustStoreKey) (not .Values.routing.pricingKey)) -}}
{{- fail "routing policy, trust-store, and pricing Secret keys are required" -}}
{{- end -}}
{{- if and (not .Values.routing.enabled) .Values.routing.policyRequired -}}
{{- fail "routing.policyRequired cannot be true when routing.enabled=false" -}}
{{- end -}}
{{- $rateLimitScope := default "process-replica" .Values.routing.rateLimitScope -}}
{{- if not (has $rateLimitScope (list "process-replica" "deployment-shared")) -}}
{{- fail "routing.rateLimitScope must be process-replica or deployment-shared" -}}
{{- end -}}
{{- $shared := .Values.routing.sharedRateLimit -}}
{{- $sharedBackend := default "direct" $shared.backend -}}
{{- if eq $rateLimitScope "deployment-shared" -}}
{{- if not (has $sharedBackend (list "direct" "sentinel")) -}}
{{- fail "routing.sharedRateLimit.backend must be direct or sentinel" -}}
{{- end -}}
{{- if not $shared.existingSecretName -}}
{{- fail "routing.sharedRateLimit.existingSecretName is required for deployment-shared limits" -}}
{{- end -}}
{{- if and (eq $sharedBackend "direct") (not $shared.redisUrlKey) -}}
{{- fail "routing.sharedRateLimit.redisUrlKey is required for the direct backend" -}}
{{- end -}}
{{- if and (eq $sharedBackend "sentinel") (not $shared.sentinelConfigKey) -}}
{{- fail "routing.sharedRateLimit.sentinelConfigKey is required for the sentinel backend" -}}
{{- end -}}
{{- if and .Values.networkPolicy.enabled (not $shared.networkPolicyEgress) -}}
{{- fail "routing.sharedRateLimit.networkPolicyEgress is required with deployment-shared limits and NetworkPolicy" -}}
{{- end -}}
{{- else if $shared.existingSecretName -}}
{{- fail "routing.sharedRateLimit.existingSecretName requires deployment-shared limits" -}}
{{- end -}}
{{- if and .Values.observation.enabled (not .Values.observation.endpoint) -}}
{{- fail "observation.endpoint is required when observation.enabled=true" -}}
{{- end -}}
{{- if and .Values.observation.enabled (not .Values.observation.apiKeySecretName) -}}
{{- fail "observation.apiKeySecretName is required when observation.enabled=true" -}}
{{- end -}}
{{- if and .Values.observation.enabled (not .Values.observation.apiKeySecretKey) -}}
{{- fail "observation.apiKeySecretKey is required when observation.enabled=true" -}}
{{- end -}}
{{- if not (has (int .Values.observation.version) (list 1 2)) -}}
{{- fail "observation.version must be 1 or 2" -}}
{{- end -}}
{{- if and .Values.otel.enabled (not .Values.otel.endpoint) -}}
{{- fail "otel.endpoint is required when otel.enabled=true" -}}
{{- end -}}
{{- if and .Values.trustedCA.enabled (not .Values.trustedCA.configMapName) -}}
{{- fail "trustedCA.configMapName is required when trustedCA.enabled=true" -}}
{{- end -}}
{{- if and .Values.trustedCA.enabled (not .Values.trustedCA.key) -}}
{{- fail "trustedCA.key is required when trustedCA.enabled=true" -}}
{{- end -}}
{{- if and .Values.persistence.enabled (not .Values.persistence.accessModes) -}}
{{- fail "persistence.accessModes is required when persistence.enabled=true" -}}
{{- end -}}
{{- if and .Values.podDisruptionBudget.enabled (lt (int .Values.podDisruptionBudget.minAvailable) 1) -}}
{{- fail "podDisruptionBudget.minAvailable must be at least 1" -}}
{{- end -}}
{{- if le (int .Values.terminationGracePeriodSeconds) (int .Values.gracefulShutdown.preStopSleepSeconds) -}}
{{- fail "terminationGracePeriodSeconds must exceed preStopSleepSeconds" -}}
{{- end -}}
{{- if not (has .Values.networkPolicy.dns.provider (list "kubernetes" "openshift")) -}}
{{- fail "networkPolicy.dns.provider must be kubernetes or openshift" -}}
{{- end -}}
{{- if not (has .Values.modelBackends.mode (list "remote" "mounted" "hybrid")) -}}
{{- fail "modelBackends.mode must be remote, mounted, or hybrid" -}}
{{- end -}}
{{- if not (has .Values.workloadSurface.profileId (list "unrestricted" "orchestra-model-plane-workload-v1")) -}}
{{- fail "workloadSurface.profileId must be unrestricted or orchestra-model-plane-workload-v1" -}}
{{- end -}}
{{- $managedEnv := list "HOST" "PORT" "AUTH_ENABLED" "AUTH_KEYS_FILE" "OTEL_ENABLED" "OTEL_EXPORTER_OTLP_ENDPOINT" "OTEL_SERVICE_NAME" "OLLAMA_MODELS_DIR" "MLX_MODELS_DIR" "HF_VLM_MODELS_DIR" "SSL_CERT_FILE" "MODEL_PLANE_WORKLOAD_SURFACE" -}}
{{- range $key, $_ := .Values.extraEnv -}}
{{- if or (has $key $managedEnv) (hasPrefix "MODEL_ROUTING_" $key) (hasPrefix "MODEL_PLANE_OBSERVATION_" $key) -}}
{{- fail (printf "extraEnv cannot override chart-managed variable %s" $key) -}}
{{- end -}}
{{- end -}}
{{- $selectorLabels := list "app.kubernetes.io/name" "app.kubernetes.io/instance" "app.kubernetes.io/component" "prometa.io/production-profile-id" -}}
{{- range $key, $_ := .Values.podLabels -}}
{{- if has $key $selectorLabels -}}
{{- fail (printf "podLabels cannot override chart-managed label %s" $key) -}}
{{- end -}}
{{- end -}}
{{- if or (hasKey .Values.podAnnotations "orchestra.prometa.ai/rollout-id") (hasKey .Values.podAnnotations "orchestra.prometa.ai/deployment-id") -}}
{{- fail "podAnnotations cannot override rollout or deployment identity" -}}
{{- end -}}

{{- if eq $environment "prod" -}}
{{- if not .Values.image.digest -}}
{{- fail "prod requires an immutable image.digest" -}}
{{- end -}}
{{- if not .Values.auth.enabled -}}
{{- fail "prod requires auth.enabled=true" -}}
{{- end -}}
{{- if not (and .Values.routing.enabled .Values.routing.policyRequired) -}}
{{- fail "prod requires signed routing policy enforcement" -}}
{{- end -}}
{{- if not .Values.routing.expectedOrgId -}}
{{- fail "prod requires routing.expectedOrgId" -}}
{{- end -}}
{{- if not .Values.routing.expectedAudience -}}
{{- fail "prod requires routing.expectedAudience" -}}
{{- end -}}
{{- if not .Values.observation.enabled -}}
{{- fail "prod requires observation.enabled=true" -}}
{{- end -}}
{{- if not .Values.persistence.enabled -}}
{{- fail "prod requires persistent per-replica LKG state" -}}
{{- end -}}
{{- if and (gt (int .Values.replicaCount) 1) (ne $rateLimitScope "deployment-shared") -}}
{{- fail "prod replicas greater than 1 require deployment-shared limits" -}}
{{- end -}}
{{- if and (eq $rateLimitScope "deployment-shared") $shared.allowInsecureRedis -}}
{{- fail "prod forbids insecure shared rate-limit transport" -}}
{{- end -}}
{{- end -}}

{{- if .Values.productionProfile.enabled -}}
{{- if ne .Values.productionProfile.profileId "orchestra-ocp-4.20-amd64-v1" -}}
{{- fail "the OpenShift production profile ID must be orchestra-ocp-4.20-amd64-v1" -}}
{{- end -}}
{{- if ne .Values.productionProfile.imageFlavor "ubi9" -}}
{{- fail "the OpenShift production profile requires imageFlavor=ubi9" -}}
{{- end -}}
{{- if ne .Values.workloadSurface.profileId "orchestra-model-plane-workload-v1" -}}
{{- fail "the OpenShift production profile requires workloadSurface.profileId=orchestra-model-plane-workload-v1" -}}
{{- end -}}
{{- if not .Values.productionProfile.namespaceDefaultDenyAcknowledged -}}
{{- fail "the OpenShift production profile requires a pre-created namespace-wide default deny" -}}
{{- end -}}
{{- if ne $environment "prod" -}}
{{- fail "the OpenShift production profile requires targetEnvironment=prod" -}}
{{- end -}}
{{- if lt (int .Values.replicaCount) 2 -}}
{{- fail "the OpenShift production profile requires replicaCount >= 2" -}}
{{- end -}}
{{- if not .Values.image.pullSecrets -}}
{{- fail "the OpenShift production profile requires a customer registry pull Secret reference" -}}
{{- end -}}
{{- if ne .Values.service.type "ClusterIP" -}}
{{- fail "the OpenShift production profile exposes only an internal ClusterIP Service" -}}
{{- end -}}
{{- if .Values.serviceAccount.automountServiceAccountToken -}}
{{- fail "the OpenShift production profile forbids service-account token automount" -}}
{{- end -}}
{{- if ne .Values.securityContextMode "openshift" -}}
{{- fail "the OpenShift production profile delegates UID/GID allocation to restricted-v2" -}}
{{- end -}}
{{- if or (not .Values.containerSecurityContext.runAsNonRoot) .Values.containerSecurityContext.allowPrivilegeEscalation (not .Values.containerSecurityContext.readOnlyRootFilesystem) -}}
{{- fail "the OpenShift production profile requires the hardened container security context" -}}
{{- end -}}
{{- if not (has "ALL" .Values.containerSecurityContext.capabilities.drop) -}}
{{- fail "the OpenShift production profile requires dropping all Linux capabilities" -}}
{{- end -}}
{{- if ne .Values.containerSecurityContext.seccompProfile.type "RuntimeDefault" -}}
{{- fail "the OpenShift production profile requires RuntimeDefault seccomp" -}}
{{- end -}}
{{- if not .Values.podDisruptionBudget.enabled -}}
{{- fail "the OpenShift production profile requires a PodDisruptionBudget" -}}
{{- end -}}
{{- if ge (int .Values.podDisruptionBudget.minAvailable) (int .Values.replicaCount) -}}
{{- fail "the OpenShift production profile PDB must permit one replica to drain" -}}
{{- end -}}
{{- if not .Values.topologySpread.enabled -}}
{{- fail "the OpenShift production profile requires topology spread constraints" -}}
{{- end -}}
{{- if ne .Values.networkPolicy.dns.provider "openshift" -}}
{{- fail "the OpenShift production profile requires the OpenShift DNS NetworkPolicy selector" -}}
{{- end -}}
{{- if or (ne .Values.topologySpread.topologyKey "kubernetes.io/hostname") (ne .Values.topologySpread.whenUnsatisfiable "DoNotSchedule") -}}
{{- fail "the OpenShift production profile requires hard hostname topology spread" -}}
{{- end -}}
{{- if or (not .Values.resources.requests) (not .Values.resources.limits) -}}
{{- fail "the OpenShift production profile requires resource requests and limits" -}}
{{- end -}}
{{- if or (ne $rateLimitScope "deployment-shared") (ne $sharedBackend "sentinel") -}}
{{- fail "the OpenShift production profile requires external Sentinel shared state" -}}
{{- end -}}
{{- if not (hasPrefix "https://" .Values.observation.endpoint) -}}
{{- fail "the OpenShift production profile requires an HTTPS observation endpoint" -}}
{{- end -}}
{{- if ne (int .Values.observation.version) 2 -}}
{{- fail "the OpenShift production profile requires observation contract v2" -}}
{{- end -}}
{{- if ne .Values.routing.expectedAudience "orchestra-model-plane" -}}
{{- fail "the OpenShift production profile requires the orchestra-model-plane audience" -}}
{{- end -}}
{{- if or (not .Values.otel.enabled) (not (hasPrefix "https://" .Values.otel.endpoint)) -}}
{{- fail "the OpenShift production profile requires HTTPS OTLP export" -}}
{{- end -}}
{{- if not .Values.trustedCA.enabled -}}
{{- fail "the OpenShift production profile requires an existing trusted CA ConfigMap" -}}
{{- end -}}
{{- if or (not .Values.persistence.storageClassName) (not .Values.persistence.externalBackupAcknowledged) -}}
{{- fail "the OpenShift production profile requires explicit storage and external backup ownership" -}}
{{- end -}}
{{- if not (has "ReadWriteOnce" .Values.persistence.accessModes) -}}
{{- fail "the OpenShift production profile requires per-replica ReadWriteOnce LKG storage" -}}
{{- end -}}
{{- if not .Values.networkPolicy.enabled -}}
{{- fail "the OpenShift production profile requires NetworkPolicy" -}}
{{- end -}}
{{- if not .Values.networkPolicy.runtimeIngressFrom -}}
{{- fail "the OpenShift production profile requires explicit tenant-runtime ingress" -}}
{{- end -}}
{{- if not .Values.networkPolicy.monitoringIngressFrom -}}
{{- fail "the OpenShift production profile requires explicit monitoring ingress" -}}
{{- end -}}
{{- if not .Values.networkPolicy.observationEgress -}}
{{- fail "the OpenShift production profile requires explicit observation egress" -}}
{{- end -}}
{{- if not .Values.networkPolicy.otelEgress -}}
{{- fail "the OpenShift production profile requires explicit OTLP egress" -}}
{{- end -}}
{{- if and (has .Values.modelBackends.mode (list "remote" "hybrid")) (not .Values.modelBackends.networkPolicyEgress) -}}
{{- fail "remote or hybrid model backends require explicit egress" -}}
{{- end -}}
{{- $hasMountedModels := or .Values.modelStorage.ollamaExistingClaim .Values.modelStorage.mlxExistingClaim .Values.modelStorage.hfVlmExistingClaim -}}
{{- if and (has .Values.modelBackends.mode (list "mounted" "hybrid")) (not $hasMountedModels) -}}
{{- fail "mounted or hybrid model backends require a tenant-owned model PVC" -}}
{{- end -}}
{{- if not .Values.metrics.serviceMonitor.enabled -}}
{{- fail "the OpenShift production profile requires a ServiceMonitor" -}}
{{- end -}}
{{- end -}}
{{- end -}}
