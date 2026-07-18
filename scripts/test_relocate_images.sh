#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
workdir="$(mktemp -d "${TMPDIR:-/tmp}/orchestra-relocate-test.XXXXXX")"
trap 'rm -rf "$workdir"' EXIT HUP INT TERM

cat > "$workdir/cosign" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$*" >> "${COSIGN_LOG:?}"
EOF
chmod +x "$workdir/cosign"

export PATH="$workdir:$PATH"
export COSIGN_LOG="$workdir/cosign.log"
export ORCHESTRA_ENGINE_SOURCE=registry.example.test/source
export ORCHESTRA_ENGINE_TAG=v0.1.8
export ORCHESTRA_CHART_VERSION=0.1.0

assert_line() {
  local expected="$1"
  grep -Fxq -- "$expected" "$COSIGN_LOG" || {
    echo "ERROR: missing cosign call: $expected" >&2
    cat "$COSIGN_LOG" >&2
    exit 1
  }
}

: > "$COSIGN_LOG"
ORCHESTRA_INCLUDE_CHART=false \
  "$repo_root/scripts/relocate_images.sh" copy registry.example.test/destination
[ "$(wc -l < "$COSIGN_LOG" | tr -d ' ')" = "2" ]
assert_line \
  "copy -f registry.example.test/source/inference-engine:v0.1.8 registry.example.test/destination/inference-engine:v0.1.8"
assert_line \
  "copy -f registry.example.test/source/inference-engine-ubi:v0.1.8 registry.example.test/destination/inference-engine-ubi:v0.1.8"

: > "$COSIGN_LOG"
ORCHESTRA_INCLUDE_CHART=true \
  "$repo_root/scripts/relocate_images.sh" copy registry.example.test/destination
[ "$(wc -l < "$COSIGN_LOG" | tr -d ' ')" = "3" ]
assert_line \
  "copy -f registry.example.test/source/charts/orchestra-inference-engine:0.1.0 registry.example.test/destination/charts/orchestra-inference-engine:0.1.0"

: > "$COSIGN_LOG"
ORCHESTRA_INCLUDE_CHART=true \
  "$repo_root/scripts/relocate_images.sh" save "$workdir/release"
assert_line \
  "save registry.example.test/source/charts/orchestra-inference-engine:0.1.0 --dir $workdir/release/orchestra-inference-engine"

: > "$COSIGN_LOG"
ORCHESTRA_INCLUDE_CHART=true \
  "$repo_root/scripts/relocate_images.sh" load \
    registry.example.test/destination "$workdir/release"
assert_line \
  "load --dir $workdir/release/orchestra-inference-engine registry.example.test/destination/charts/orchestra-inference-engine:0.1.0"

if ORCHESTRA_INCLUDE_CHART=invalid \
  "$repo_root/scripts/relocate_images.sh" copy registry.example.test/destination \
  >/dev/null 2>&1; then
  echo "ERROR: invalid ORCHESTRA_INCLUDE_CHART was accepted" >&2
  exit 1
fi

echo "Release artifact relocation contract passed"
