#!/usr/bin/env bash
# Relocate both signed engine variants while preserving cosign signatures and
# attestations. Authenticate cosign to the source and destination first.
set -euo pipefail

SOURCE="${ORCHESTRA_ENGINE_SOURCE:-ghcr.io/caglarsubas/llm_inference_engine}"
TAG="${ORCHESTRA_ENGINE_TAG:-v0.1.0}"
IMAGES=(inference-engine inference-engine-ubi)

usage() {
  cat <<'EOF'
usage: relocate_images.sh copy <destination-namespace>
       relocate_images.sh save [directory]
       relocate_images.sh load <destination-namespace> [directory]

Environment:
  ORCHESTRA_ENGINE_SOURCE  source namespace
  ORCHESTRA_ENGINE_TAG     source/destination tag
EOF
}

need_cosign() {
  command -v cosign >/dev/null 2>&1 || {
    echo "ERROR: cosign is required" >&2
    exit 1
  }
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
    ;;
  save)
    directory="${2:-./orchestra-engine-images}"
    need_cosign
    mkdir -p "${directory}"
    for image in "${IMAGES[@]}"; do
      cosign save "${SOURCE}/${image}:${TAG}" --dir "${directory}/${image}"
    done
    ;;
  load)
    destination="${2:-}"
    directory="${3:-./orchestra-engine-images}"
    [ -n "${destination}" ] || { usage >&2; exit 1; }
    need_cosign
    for image in "${IMAGES[@]}"; do
      cosign load --dir "${directory}/${image}" "${destination}/${image}:${TAG}"
    done
    ;;
  *)
    usage >&2
    exit 1
    ;;
esac
