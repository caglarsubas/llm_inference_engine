#!/usr/bin/env bash
# Relocate both signed engine variants, and optionally the signed Helm chart,
# while preserving cosign signatures and attestations. Authenticate cosign to
# the source and destination first.
set -euo pipefail

SOURCE="${ORCHESTRA_ENGINE_SOURCE:-ghcr.io/caglarsubas/llm_inference_engine}"
TAG="${ORCHESTRA_ENGINE_TAG:-v0.1.0}"
IMAGES=(inference-engine inference-engine-ubi)
INCLUDE_CHART="${ORCHESTRA_INCLUDE_CHART:-false}"
CHART_SOURCE="${ORCHESTRA_CHART_SOURCE:-${SOURCE}/charts}"
CHART_NAME="${ORCHESTRA_CHART_NAME:-orchestra-inference-engine}"
CHART_VERSION="${ORCHESTRA_CHART_VERSION:-0.1.0}"

usage() {
  cat <<'EOF'
usage: relocate_images.sh copy <destination-namespace>
       relocate_images.sh save [directory]
       relocate_images.sh load <destination-namespace> [directory]

Environment:
  ORCHESTRA_ENGINE_SOURCE  source namespace
  ORCHESTRA_ENGINE_TAG     source/destination tag
  ORCHESTRA_INCLUDE_CHART  true to include the signed Helm chart (default false)
  ORCHESTRA_CHART_SOURCE   chart source namespace (default <engine source>/charts)
  ORCHESTRA_CHART_NAME     chart OCI repository name
  ORCHESTRA_CHART_VERSION  chart source/destination version
EOF
}

need_cosign() {
  command -v cosign >/dev/null 2>&1 || {
    echo "ERROR: cosign is required" >&2
    exit 1
  }
}

case "${INCLUDE_CHART}" in
  true|false) ;;
  *) echo "ERROR: ORCHESTRA_INCLUDE_CHART must be true or false" >&2; exit 1 ;;
esac

copy_chart() {
  local destination="$1"
  if [ "${INCLUDE_CHART}" = "true" ]; then
    cosign copy -f \
      "${CHART_SOURCE}/${CHART_NAME}:${CHART_VERSION}" \
      "${destination}/charts/${CHART_NAME}:${CHART_VERSION}"
  fi
}

save_chart() {
  local directory="$1"
  if [ "${INCLUDE_CHART}" = "true" ]; then
    cosign save \
      "${CHART_SOURCE}/${CHART_NAME}:${CHART_VERSION}" \
      --dir "${directory}/${CHART_NAME}"
  fi
}

load_chart() {
  local destination="$1"
  local directory="$2"
  if [ "${INCLUDE_CHART}" = "true" ]; then
    cosign load \
      --dir "${directory}/${CHART_NAME}" \
      "${destination}/charts/${CHART_NAME}:${CHART_VERSION}"
  fi
}

case "${1:-}" in
  copy)
    destination="${2:-}"
    [ -n "${destination}" ] || { usage >&2; exit 1; }
    need_cosign
    for image in "${IMAGES[@]}"; do
      cosign copy -f \
        "${SOURCE}/${image}:${TAG}" \
        "${destination}/${image}:${TAG}"
    done
    copy_chart "${destination}"
    ;;
  save)
    directory="${2:-./orchestra-engine-images}"
    need_cosign
    mkdir -p "${directory}"
    for image in "${IMAGES[@]}"; do
      cosign save "${SOURCE}/${image}:${TAG}" --dir "${directory}/${image}"
    done
    save_chart "${directory}"
    ;;
  load)
    destination="${2:-}"
    directory="${3:-./orchestra-engine-images}"
    [ -n "${destination}" ] || { usage >&2; exit 1; }
    need_cosign
    for image in "${IMAGES[@]}"; do
      cosign load --dir "${directory}/${image}" "${destination}/${image}:${TAG}"
    done
    load_chart "${destination}" "${directory}"
    ;;
  *)
    usage >&2
    exit 1
    ;;
esac
