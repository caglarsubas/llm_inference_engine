#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# share-service.sh — run the public ngrok tunnel as a macOS launchd agent.
#
# Why this exists
# ---------------
# `make share` is great for ad-hoc sharing but dies with the terminal (and
# never survives a reboot).  For a "truly always-on" public endpoint we
# install ngrok as a launchd *user agent* with RunAtLoad + KeepAlive, so the
# tunnel boots on login and respawns on crash — the same pattern the engine
# itself uses (scripts/native-service.sh).  Pair the two and the engine plus
# its stable public URL always come back together.
#
# A reserved ngrok domain is mandatory here: a persistent agent on an
# ephemeral URL would hand out a different hostname on every respawn.  Claim
# the free static domain in the ngrok dashboard (Domains, *.ngrok-free.dev)
# first, then install with it.
#
# Subcommands
# -----------
#   install --domain <d> [--port N]  render plist, load + start the agent.
#                                     Idempotent: re-running re-renders.
#                                     --domain may be omitted if NGROK_DOMAIN
#                                     is set; --port defaults to PORT/.env/8080.
#   uninstall   bootout + remove the installed plist.
#   start       kickstart the agent.
#   stop        bootout (stops without uninstalling).
#   restart     stop + start (re-reads the plist).
#   status      print agent state, PID, last exit, the public URL, log paths.
#   logs        tail -F the tunnel logs.
#
# Examples
#   ./scripts/share-service.sh install --domain my-name.ngrok-free.dev
#   make share-install NGROK_DOMAIN=my-name.ngrok-free.dev
#   ./scripts/share-service.sh status
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TEMPLATE_DIR="$SCRIPT_DIR/launchd"
INSTALL_DIR="$HOME/Library/LaunchAgents"

LABEL="com.prometa.ngrok-tunnel"
PLIST="$INSTALL_DIR/$LABEL.plist"
USER_DOMAIN="gui/$(id -u)"

err()  { printf '\033[31m%s\033[0m\n' "$*" >&2; }
log()  { printf '\033[32m%s\033[0m\n' "$*"; }
note() { printf '\033[33m%s\033[0m\n' "$*"; }

require_macos() {
    if [[ "$(uname -s)" != "Darwin" ]]; then
        err "share-service.sh runs on macOS only — saw $(uname -s)."
        err "On Linux, run ngrok under systemd or your process manager."
        exit 1
    fi
}

resolve_ngrok_bin() {
    local found
    if command -v ngrok >/dev/null 2>&1; then
        found="$(command -v ngrok)"
    elif [[ -x /opt/homebrew/bin/ngrok ]]; then
        found="/opt/homebrew/bin/ngrok"
    elif [[ -x /usr/local/bin/ngrok ]]; then
        found="/usr/local/bin/ngrok"
    else
        err "ngrok not found.  Install with:"
        err "    brew install ngrok"
        err "    https://ngrok.com/download"
        exit 1
    fi
    printf '%s\n' "$found"
}

check_ngrok_auth() {
    # ngrok refuses to tunnel without a saved authtoken.  We can't read the
    # token value, but we can confirm a config file exists.
    if ! ngrok config check >/dev/null 2>&1; then
        err "ngrok has no valid config / authtoken."
        err "Authenticate once (free token at dashboard.ngrok.com):"
        err "    ngrok config add-authtoken <token>"
        exit 1
    fi
}

resolve_port() {
    # Precedence: --port flag (PORT var) > .env PORT > 8080.
    if [[ -n "${PORT:-}" ]]; then printf '%s\n' "$PORT"; return; fi
    local env_port=""
    [[ -f "$PROJECT_DIR/.env" ]] && env_port="$(grep -E '^PORT=' "$PROJECT_DIR/.env" 2>/dev/null | head -n1 | cut -d= -f2 | tr -d '[:space:]')"
    printf '%s\n' "${env_port:-8080}"
}

render_plist() {
    local domain="$1" port="$2" ngrok_bin
    ngrok_bin="$(resolve_ngrok_bin)"
    sed \
        -e "s|__NGROK_BIN__|$ngrok_bin|g" \
        -e "s|__PORT__|$port|g" \
        -e "s|__NGROK_DOMAIN__|$domain|g" \
        -e "s|__HOME__|$HOME|g" \
        "$TEMPLATE_DIR/$LABEL.plist" > "$PLIST"
}

bootout_if_loaded() {
    [[ -f "$PLIST" ]] || return 0
    if launchctl print "$USER_DOMAIN/$LABEL" >/dev/null 2>&1; then
        launchctl bootout "$USER_DOMAIN" "$PLIST" 2>/dev/null || true
    fi
}

bootstrap_one() {
    bootout_if_loaded
    launchctl bootstrap "$USER_DOMAIN" "$PLIST"
    launchctl kickstart -k "$USER_DOMAIN/$LABEL"
}

kill_adhoc_ngrok() {
    # Free ngrok allows a single simultaneous agent session.  A leftover
    # `make share` / detached ngrok would make the agent's tunnel fail to
    # claim the reserved domain, so clear any non-launchd ngrok first.
    if pgrep -f 'ngrok http' >/dev/null 2>&1; then
        note "Stopping existing ad-hoc ngrok session(s) so the agent can claim the domain..."
        pkill -f 'ngrok http' 2>/dev/null || true
        sleep 1
    fi
}

DOMAIN="${NGROK_DOMAIN:-}"
parse_install_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --domain) DOMAIN="${2:?--domain needs a value}"; shift 2 ;;
            --port)   PORT="${2:?--port needs a value}"; shift 2 ;;
            *) err "unknown install arg: $1"; exit 2 ;;
        esac
    done
}

cmd_install() {
    require_macos
    parse_install_args "$@"
    if [[ -z "$DOMAIN" ]]; then
        err "A reserved domain is required for an always-on tunnel."
        err "Pass it explicitly or via NGROK_DOMAIN:"
        err "    ./scripts/share-service.sh install --domain my-name.ngrok-free.dev"
        err "    make share-install NGROK_DOMAIN=my-name.ngrok-free.dev"
        err "Claim a free static domain at https://dashboard.ngrok.com/domains"
        exit 1
    fi
    resolve_ngrok_bin >/dev/null
    check_ngrok_auth
    local port; port="$(resolve_port)"
    mkdir -p "$INSTALL_DIR"
    kill_adhoc_ngrok
    log "Rendering $LABEL.plist (domain=$DOMAIN port=$port)"
    render_plist "$DOMAIN" "$port"
    log "Loading agent under $USER_DOMAIN"
    bootstrap_one
    log "Installed.  Public endpoint will be:"
    note "    https://$DOMAIN"
    note "    logs: /tmp/prometa-ngrok-tunnel.{out,err}.log"
    log "It now auto-starts on login and respawns on crash/reboot."
}

cmd_uninstall() {
    require_macos
    bootout_if_loaded
    rm -f "$PLIST"
    log "Uninstalled ngrok tunnel agent (the reserved domain stays on your account)."
}

cmd_start() {
    require_macos
    [[ -f "$PLIST" ]] || { err "Not installed — run '$0 install --domain <d>' first."; exit 1; }
    kill_adhoc_ngrok
    bootstrap_one
    log "Started."
}

cmd_stop() {
    require_macos
    bootout_if_loaded
    log "Stopped (plist remains installed; '$0 start' to resume)."
}

cmd_restart() { cmd_stop; cmd_start; }

cmd_status() {
    require_macos
    printf '\n=== %s ===\n' "$LABEL"
    if launchctl print "$USER_DOMAIN/$LABEL" 2>/dev/null \
        | grep -E '^\s+(state|pid|last exit code|program)\s*=' ; then
        :
    else
        note "    not loaded"
    fi
    # Surface the live public URL from ngrok's local inspection API.
    local url
    url="$(curl -fsS --max-time 2 http://127.0.0.1:4040/api/tunnels 2>/dev/null \
        | grep -oE 'https://[a-zA-Z0-9.-]+\.ngrok(-free)?\.(app|dev|io)' | head -n1 || true)"
    [[ -n "$url" ]] && printf '\nPublic URL: %s\n' "$url"
    printf '\nLog files:\n'
    ls -lh /tmp/prometa-ngrok-tunnel.*.log 2>/dev/null \
        | awk '{printf "    %s  %s  %s\n", $5, $6" "$7" "$8, $9}' || true
}

cmd_logs() { tail -F /tmp/prometa-ngrok-tunnel.{out,err}.log; }

usage() {
    sed -n '1,/^# ---*$/p' "$0" | sed 's/^# \{0,1\}//; s/^#//'
    exit 1
}

case "${1:-}" in
    install)   shift; cmd_install "$@" ;;
    uninstall) shift; cmd_uninstall "$@" ;;
    start)     shift; cmd_start "$@" ;;
    stop)      shift; cmd_stop "$@" ;;
    restart)   shift; cmd_restart "$@" ;;
    status)    shift; cmd_status "$@" ;;
    logs)      shift; cmd_logs "$@" ;;
    ""|-h|--help) usage ;;
    *) err "unknown subcommand: $1"; usage ;;
esac
