#!/usr/bin/env bash
#
# share_endpoint.sh — expose the local inference engine on a public HTTPS URL.
#
# Wraps ngrok (default) or cloudflared so the engine running on
# 127.0.0.1:<port> gets a stable, internet-reachable "Engine URL" you can
# paste into Prometa (Settings → Self-hosted (llm_inference_engine) → ENGINE
# URL) or hand to any OpenAI-compatible client. The tunnel terminates TLS and
# forwards to the loopback port — your laptop never needs an inbound firewall
# rule or a public IP.
#
# Usage:
#   scripts/share_endpoint.sh [--provider ngrok|cloudflared] [--port N]
#                             [--host H] [--domain example.ngrok.app] [-y]
#
# Examples:
#   scripts/share_endpoint.sh                      # ngrok against $PORT (.env)
#   scripts/share_endpoint.sh --port 8090          # tunnel the compose LB port
#   scripts/share_endpoint.sh --provider cloudflared
#   scripts/share_endpoint.sh --domain my.ngrok.app  # reserved ngrok domain
#
# Stop the tunnel with Ctrl-C; the public URL is revoked on exit.

set -euo pipefail

# --------------------------------------------------------------------------
# Defaults + .env loading
# --------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_FILE="$ROOT_DIR/.env"

PROVIDER="ngrok"
HOST_OVERRIDE=""
PORT_OVERRIDE=""
DOMAIN=""
ASSUME_YES="false"

# Pull HOST / PORT / AUTH_ENABLED out of .env without sourcing it (the file may
# contain values that aren't safe to eval). Default to the engine's own
# defaults when unset.
env_get() {
  local key="$1" default="$2" line
  if [[ -f "$ENV_FILE" ]]; then
    line="$(grep -E "^${key}=" "$ENV_FILE" | tail -n1 || true)"
    if [[ -n "$line" ]]; then
      printf '%s' "${line#*=}"
      return
    fi
  fi
  printf '%s' "$default"
}

# --------------------------------------------------------------------------
# Arg parsing
# --------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --provider) PROVIDER="${2:?--provider needs a value}"; shift 2 ;;
    --port)     PORT_OVERRIDE="${2:?--port needs a value}"; shift 2 ;;
    --host)     HOST_OVERRIDE="${2:?--host needs a value}"; shift 2 ;;
    --domain)   DOMAIN="${2:?--domain needs a value}"; shift 2 ;;
    -y|--yes)   ASSUME_YES="true"; shift ;;
    -h|--help)
      awk 'NR==1{next} /^#/{sub(/^# ?/,""); print; next} {exit}' "${BASH_SOURCE[0]}"
      exit 0 ;;
    *) echo "unknown flag: $1" >&2; exit 2 ;;
  esac
done

HOST="${HOST_OVERRIDE:-$(env_get HOST 127.0.0.1)}"
PORT="${PORT_OVERRIDE:-$(env_get PORT 8080)}"
AUTH_ENABLED="$(env_get AUTH_ENABLED false)"
LOCAL_URL="http://${HOST}:${PORT}"

# ANSI helpers (no-op when not a TTY).
if [[ -t 1 ]]; then
  BOLD=$'\033[1m'; DIM=$'\033[2m'; RED=$'\033[31m'; GRN=$'\033[32m'
  YEL=$'\033[33m'; CYN=$'\033[36m'; RST=$'\033[0m'
else
  BOLD=""; DIM=""; RED=""; GRN=""; YEL=""; CYN=""; RST=""
fi

note() { printf '%s\n' "$*"; }
err()  { printf '%s%s%s\n' "$RED" "$*" "$RST" >&2; }

# --------------------------------------------------------------------------
# Pre-flight: engine reachable?
# --------------------------------------------------------------------------
note "${BOLD}Inference engine → public endpoint${RST}"
note "${DIM}provider=${PROVIDER}  target=${LOCAL_URL}${RST}"
note ""

if ! curl -fsS --max-time 3 "${LOCAL_URL}/v1/health" >/dev/null 2>&1; then
  err "Engine is not answering on ${LOCAL_URL}/v1/health."
  err "Start it first, e.g.:"
  err "    make run            # native"
  err "    make compose-up     # docker (then tunnel LB_PORT, e.g. --port 8090)"
  exit 1
fi
note "${GRN}✓${RST} engine healthy on ${LOCAL_URL}"

# --------------------------------------------------------------------------
# Safety: a public URL with auth OFF is an open, unauthenticated LLM.
# --------------------------------------------------------------------------
if [[ "$(printf '%s' "$AUTH_ENABLED" | tr '[:upper:]' '[:lower:]')" != "true" ]]; then
  note ""
  err "WARNING: AUTH_ENABLED is not 'true'."
  err "Anyone with this URL can run inference on your machine for free."
  err "Turn on bearer-token auth before sharing widely:"
  err "    1) create .auth_keys.json  ->  [{\"key\":\"sk-...\",\"tenant\":\"prometa\"}]"
  err "    2) set AUTH_ENABLED=true in .env  ->  restart the engine"
  if [[ "$ASSUME_YES" != "true" ]]; then
    printf '%sContinue anyway? [y/N] %s' "$YEL" "$RST"
    read -r reply
    [[ "$reply" =~ ^[Yy]$ ]] || { note "Aborted."; exit 1; }
  fi
fi

# --------------------------------------------------------------------------
# Guidance printer — shown once the public URL is known.
# --------------------------------------------------------------------------
print_guidance() {
  local public_url="$1"
  local auth_hint=""
  if [[ "$(printf '%s' "$AUTH_ENABLED" | tr '[:upper:]' '[:lower:]')" == "true" ]]; then
    auth_hint=" -H 'Authorization: Bearer <your-key>'"
  fi
  note ""
  note "${GRN}${BOLD}Public endpoint is live${RST}"
  note "${BOLD}  ${public_url}${RST}"
  note ""
  note "${CYN}Prometa → Settings → Self-hosted (llm_inference_engine):${RST}"
  note "  ENGINE URL    ${public_url}"
  if [[ -n "$auth_hint" ]]; then
    note "  ENGINE TOKEN  <your bearer key from .auth_keys.json>"
  else
    note "  ENGINE TOKEN  (leave blank — auth is off)"
  fi
  note ""
  note "${CYN}List models:${RST}"
  note "  curl -s ${public_url}/v1/models${auth_hint} | jq '.data[].id'"
  note ""
  note "${CYN}Chat completion (swap \"model\" for any id from /v1/models):${RST}"
  note "  curl -s ${public_url}/v1/chat/completions \\"
  note "    -H 'content-type: application/json'${auth_hint} \\"
  note "    -d '{\"model\":\"llama3.2:3b\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":32}'"
  note ""
  note "${DIM}Full usage guide: docs/PUBLIC_ENDPOINT.md${RST}"
  note "${DIM}Press Ctrl-C to stop the tunnel (the URL is revoked on exit).${RST}"
  note ""
}

cleanup() {
  [[ -n "${TUNNEL_PID:-}" ]] && kill "$TUNNEL_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# --------------------------------------------------------------------------
# ngrok
# --------------------------------------------------------------------------
run_ngrok() {
  if ! command -v ngrok >/dev/null 2>&1; then
    err "ngrok not found. Install it:"
    err "    brew install ngrok            # macOS"
    err "    https://ngrok.com/download    # other platforms"
    err "Then authenticate once: ngrok config add-authtoken <token>"
    exit 1
  fi

  local args=(http "${HOST}:${PORT}" --log stdout)
  [[ -n "$DOMAIN" ]] && args+=(--domain "$DOMAIN")

  note "${DIM}starting: ngrok ${args[*]}${RST}"
  ngrok "${args[@]}" >/tmp/ngrok.share.log 2>&1 &
  TUNNEL_PID=$!

  # ngrok exposes a local inspection API; poll it for the assigned URL.
  local url="" i
  for i in $(seq 1 40); do
    if ! kill -0 "$TUNNEL_PID" 2>/dev/null; then
      err "ngrok exited early. Last log lines:"
      tail -n 20 /tmp/ngrok.share.log >&2 || true
      exit 1
    fi
    url="$(curl -fsS --max-time 2 http://127.0.0.1:4040/api/tunnels 2>/dev/null \
      | grep -oE 'https://[a-zA-Z0-9.-]+\.(ngrok\.app|ngrok-free\.app|ngrok\.io)' \
      | head -n1 || true)"
    [[ -n "$url" ]] && break
    sleep 0.5
  done

  if [[ -z "$url" ]]; then
    err "Could not determine the ngrok public URL (is :4040 in use?)."
    err "Check the ngrok dashboard at http://127.0.0.1:4040"
    exit 1
  fi

  print_guidance "$url"
  wait "$TUNNEL_PID"
}

# --------------------------------------------------------------------------
# cloudflared (no account needed for quick *.trycloudflare.com tunnels)
# --------------------------------------------------------------------------
run_cloudflared() {
  if ! command -v cloudflared >/dev/null 2>&1; then
    err "cloudflared not found. Install it:"
    err "    brew install cloudflared      # macOS"
    err "    https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/"
    exit 1
  fi

  note "${DIM}starting: cloudflared tunnel --url ${LOCAL_URL}${RST}"
  cloudflared tunnel --url "${LOCAL_URL}" --no-autoupdate >/tmp/cloudflared.share.log 2>&1 &
  TUNNEL_PID=$!

  local url="" i
  for i in $(seq 1 60); do
    if ! kill -0 "$TUNNEL_PID" 2>/dev/null; then
      err "cloudflared exited early. Last log lines:"
      tail -n 20 /tmp/cloudflared.share.log >&2 || true
      exit 1
    fi
    url="$(grep -oE 'https://[a-zA-Z0-9.-]+\.trycloudflare\.com' /tmp/cloudflared.share.log \
      | head -n1 || true)"
    [[ -n "$url" ]] && break
    sleep 0.5
  done

  if [[ -z "$url" ]]; then
    err "Could not determine the cloudflared public URL. Last log lines:"
    tail -n 20 /tmp/cloudflared.share.log >&2 || true
    exit 1
  fi

  print_guidance "$url"
  wait "$TUNNEL_PID"
}

case "$PROVIDER" in
  ngrok)       run_ngrok ;;
  cloudflared) run_cloudflared ;;
  *) err "unknown provider: $PROVIDER (use ngrok or cloudflared)"; exit 2 ;;
esac
