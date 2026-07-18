#!/usr/bin/env bash
# Keep the Python package, runtime, chart app version, and release tag aligned.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
release_tag="${1:-}"

read_toml_version() {
  sed -n 's/^version = "\([^"]*\)"$/\1/p' "$repo_root/pyproject.toml" | head -n 1
}

read_python_version() {
  sed -n 's/^__version__ = "\([^"]*\)"$/\1/p' \
    "$repo_root/src/inference_engine/__init__.py" | head -n 1
}

read_chart_value() {
  local key="$1"
  sed -n "s/^${key}: *\"\{0,1\}\([^\"]*\)\"\{0,1\}$/\\1/p" \
    "$repo_root/deploy/helm/inference-engine/Chart.yaml" | head -n 1
}

package_version="$(read_toml_version)"
runtime_version="$(read_python_version)"
chart_version="$(read_chart_value version)"
chart_app_version="$(read_chart_value appVersion)"

semver='^[0-9]+\.[0-9]+\.[0-9]+([+-][0-9A-Za-z.-]+)?$'
for entry in \
  "package:${package_version}" \
  "runtime:${runtime_version}" \
  "chart:${chart_version}" \
  "chart app:${chart_app_version}"; do
  label="${entry%%:*}"
  value="${entry#*:}"
  if ! printf '%s' "$value" | grep -Eq "$semver"; then
    echo "ERROR: ${label} version is not valid semver: ${value:-<empty>}" >&2
    exit 1
  fi
done

if [ "$package_version" != "$runtime_version" ] || \
   [ "$package_version" != "$chart_app_version" ]; then
  echo "ERROR: release versions differ" >&2
  echo "  pyproject.toml: ${package_version}" >&2
  echo "  runtime:        ${runtime_version}" >&2
  echo "  chart app:      ${chart_app_version}" >&2
  exit 1
fi

if [ -n "$release_tag" ] && [ "$release_tag" != "v${package_version}" ]; then
  echo "ERROR: release tag ${release_tag} does not match v${package_version}" >&2
  exit 1
fi

printf 'release_version=%s\n' "$package_version"
printf 'chart_version=%s\n' "$chart_version"
printf 'release_tag=%s\n' "${release_tag:-v${package_version}}"
