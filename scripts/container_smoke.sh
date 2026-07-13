#!/usr/bin/env bash
# Prove the image boots under the OpenShift restricted-v2 identity shape:
# arbitrary high UID, GID 0, read-only root filesystem, writable tmp/state.
set -euo pipefail

IMAGE="${1:?usage: $0 <image> [arbitrary-uid]}"
ARBITRARY_UID="${2:-1001230000}"
NAME="inference-engine-smoke-${RANDOM}-${RANDOM}"
CID=""

cleanup() {
  if [ -n "${CID}" ]; then
    docker logs "${CID}" 2>&1 | tail -n 120 || true
    docker rm -f "${CID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

configured_user="$(docker image inspect --format '{{.Config.User}}' "${IMAGE}")"
case "${configured_user}" in
  ""|0|0:0|root|root:root)
    echo "ERROR: image default user is root: ${configured_user:-<empty>}" >&2
    exit 1
    ;;
esac

CID="$(docker run --detach \
  --name "${NAME}" \
  --read-only \
  --user "${ARBITRARY_UID}:0" \
  --tmpfs /tmp:rw,nosuid,nodev,size=64m,mode=1777 \
  --tmpfs /state:rw,nosuid,nodev,size=16m,mode=0770 \
  --env AUTH_ENABLED=false \
  --env MODEL_ROUTING_POLICY_REQUIRED=false \
  --env MODEL_PLANE_OBSERVATION_ENABLED=false \
  --env OLLAMA_HTTP_ENDPOINT= \
  "${IMAGE}")"

healthy=false
for _ in $(seq 1 60); do
  state="$(docker inspect --format '{{.State.Status}}' "${CID}")"
  health="$(docker inspect --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}' "${CID}")"
  if [ "${health}" = "healthy" ]; then
    healthy=true
    break
  fi
  if [ "${state}" != "running" ]; then
    echo "ERROR: container stopped before becoming healthy (state=${state}, health=${health})" >&2
    exit 1
  fi
  sleep 2
done

if [ "${healthy}" != "true" ]; then
  echo "ERROR: container did not become healthy" >&2
  exit 1
fi

actual_uid="$(docker exec "${CID}" id -u)"
actual_gid="$(docker exec "${CID}" id -g)"
[ "${actual_uid}" = "${ARBITRARY_UID}" ] || {
  echo "ERROR: expected UID ${ARBITRARY_UID}, got ${actual_uid}" >&2
  exit 1
}
[ "${actual_gid}" = "0" ] || {
  echo "ERROR: expected GID 0, got ${actual_gid}" >&2
  exit 1
}

docker exec "${CID}" sh -c 'printf smoke > /state/arbitrary-uid-smoke && test -s /state/arbitrary-uid-smoke'
docker exec "${CID}" sh -c '! touch /app/root-filesystem-must-stay-read-only 2>/dev/null'
docker exec "${CID}" python -c \
  "import urllib.request; assert urllib.request.urlopen('http://127.0.0.1:8080/v1/health', timeout=3).status == 200"

echo "PASS: ${IMAGE} (default_user=${configured_user}, runtime=${actual_uid}:${actual_gid}, read_only_root=true)"
