#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# native-service.sh — manage the engine + ollama as macOS launchd user agents.
#
# Why this exists
# ---------------
# The pragmatic on-prem topology runs the inference engine and Ollama
# *natively* (so they get Metal on Apple Silicon), and only the
# observability stack inside containers.  This script materialises the
# launchd plist templates under scripts/launchd/, substitutes absolute
# paths for the placeholders, drops them in ~/Library/LaunchAgents/, and
# bootstrap/bootout's them through ``launchctl``.
#
# Subcommands
# -----------
#   install     — render plists, copy to LaunchAgents/, load + start.
#                 Idempotent: re-running re-renders and reloads.
#   uninstall   — bootout + remove the installed plists.
#   start       — kickstart both agents.
#   stop        — bootout (stops without uninstalling).
#   restart     — stop + start (re-reads the plist).
#   status      — print agent state, PID, last exit, log paths.
#   logs ENGINE|OLLAMA   — tail -f the matching log stream.
#
# Design notes
# ------------
# * The plists use ``RunAtLoad=true`` + ``KeepAlive=SuccessfulExit:false``,
#   so the engine boots on login and respawns on crash — but launchd
#   throttles via ``ThrottleInterval=5`` so a misconfigured env doesn't
#   peg CPU.
# * We resolve the ollama binary path at install time so macOS users
#   without Homebrew (or with a non-default brew prefix) still get a
#   working plist.
# * Logs go to /tmp so they're readable without sudo and survive a
#   single reboot but don't accumulate forever.
# ---------------------------------------------------------------------------
set -euo pipefail

# Resolve project root from this script's location so the helper works
# regardless of the user's current directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TEMPLATE_DIR="$SCRIPT_DIR/launchd"
INSTALL_DIR="$HOME/Library/LaunchAgents"

ENGINE_LABEL="com.prometa.inference-engine"
OLLAMA_LABEL="com.prometa.ollama"
ENGINE_PLIST="$INSTALL_DIR/$ENGINE_LABEL.plist"
OLLAMA_PLIST="$INSTALL_DIR/$OLLAMA_LABEL.plist"

# launchctl in modern macOS prefers the GUI domain for user agents.
USER_DOMAIN="gui/$(id -u)"

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

err() { printf '\033[31m%s\033[0m\n' "$*" >&2; }
log() { printf '\033[32m%s\033[0m\n' "$*"; }
note() { printf '\033[33m%s\033[0m\n' "$*"; }

require_macos() {
    if [[ "$(uname -s)" != "Darwin" ]]; then
        err "native-service.sh runs on macOS only — saw $(uname -s)."
        err "On Linux, use 'docker compose up' (containerised topology)."
        exit 1
    fi
}

check_tcc_protected_path() {
    # macOS TCC ("Files and Folders" permission) blocks launchd-spawned
    # processes from reading these paths until the binary has explicit
    # Full Disk Access — and TCC under launchd shows no UI prompt, so the
    # request hangs indefinitely.  We refuse to install when the project
    # lives in one of these locations *unless* the operator has confirmed
    # that the venv Python has been granted Full Disk Access (we can't
    # detect FDA from the CLI, so it has to be an explicit opt-in).
    #
    # Bypass for an already-granted setup:
    #     PROMETA_SKIP_TCC_CHECK=1 ./scripts/native-service.sh install
    if [[ "${PROMETA_SKIP_TCC_CHECK:-0}" == "1" ]]; then
        note "PROMETA_SKIP_TCC_CHECK=1 — assuming the venv Python has Full Disk Access."
        return 0
    fi
    local resolved
    resolved="$(cd "$PROJECT_DIR" && pwd -P)"  # follow symlinks
    case "$resolved" in
        "$HOME/Desktop"/*|"$HOME/Documents"/*|"$HOME/Downloads"/*)
            err ""
            err "The project lives under a TCC-protected directory:"
            err "    $resolved"
            err ""
            err "macOS does not let launchd-spawned services read files from"
            err "Desktop / Documents / Downloads without an explicit Full Disk"
            err "Access grant.  Without that grant the engine process hangs"
            err "forever inside Python's startup path-config code."
            err ""
            err "Pick one:"
            err ""
            err "  (a) Move the project to a non-protected directory.  Easiest:"
            err "         mv \"$resolved\" \"\$HOME/dev/$(basename "$resolved")\""
            err "         cd \"\$HOME/dev/$(basename "$resolved")\" && ./scripts/native-service.sh install"
            err ""
            err "  (b) Grant Full Disk Access to the venv Python in"
            err "      System Settings → Privacy & Security → Full Disk Access."
            err "      Add this exact binary:"
            err "         $(cd "$PROJECT_DIR" && readlink -f .venv/bin/python 2>/dev/null || echo "<run 'make install' first>")"
            err "      Then re-run with the bypass flag:"
            err "         PROMETA_SKIP_TCC_CHECK=1 ./scripts/native-service.sh install"
            err ""
            err "  (c) Run interactively for now:  'make run'"
            err "      (works because your Terminal already has the TCC grant;"
            err "      no auto-restart, but proves the topology end-to-end)."
            err ""
            exit 1
            ;;
    esac
}

resolve_ollama_bin() {
    # Prefer brew prefix if available so we pick up the same binary the
    # user invokes from a shell.  Fall back to PATH.
    local found
    if command -v ollama >/dev/null 2>&1; then
        found="$(command -v ollama)"
    elif [[ -x /opt/homebrew/bin/ollama ]]; then
        found="/opt/homebrew/bin/ollama"
    elif [[ -x /usr/local/bin/ollama ]]; then
        found="/usr/local/bin/ollama"
    elif [[ -x /Applications/Ollama.app/Contents/Resources/ollama ]]; then
        found="/Applications/Ollama.app/Contents/Resources/ollama"
    else
        err "ollama not found.  Install with one of:"
        err "    brew install ollama"
        err "    https://ollama.com/download (drag to /Applications)"
        exit 1
    fi
    printf '%s\n' "$found"
}

resolve_venv_python() {
    local venv_python="$PROJECT_DIR/.venv/bin/python"
    if [[ ! -x "$venv_python" ]]; then
        err ".venv not found at $PROJECT_DIR/.venv"
        err "Run 'make install-metal' first to build llama-cpp-python with Metal."
        exit 1
    fi
    printf '%s\n' "$venv_python"
}

render_plist() {
    # render_plist <template_basename> <dest_path>
    # Performs the placeholder substitutions and writes to dest.
    local template="$1"
    local dest="$2"
    local ollama_bin venv_python
    ollama_bin="$(resolve_ollama_bin)"
    venv_python="$(resolve_venv_python)"

    sed \
        -e "s|__PROJECT_DIR__|$PROJECT_DIR|g" \
        -e "s|__VENV_PYTHON__|$venv_python|g" \
        -e "s|__OLLAMA_BIN__|$ollama_bin|g" \
        -e "s|__HOME__|$HOME|g" \
        "$TEMPLATE_DIR/$template" > "$dest"
}

bootout_if_loaded() {
    # bootout_if_loaded <plist_path>
    # Only runs bootout if the agent is currently loaded — otherwise
    # macOS prints a misleading "Boot-out failed: 5: Input/output error".
    local plist="$1"
    [[ -f "$plist" ]] || return 0
    if launchctl print "$USER_DOMAIN/$(basename "$plist" .plist)" >/dev/null 2>&1; then
        launchctl bootout "$USER_DOMAIN" "$plist" 2>/dev/null || true
    fi
}

bootstrap_one() {
    local plist="$1"
    bootout_if_loaded "$plist"
    launchctl bootstrap "$USER_DOMAIN" "$plist"
    launchctl kickstart -k "$USER_DOMAIN/$(basename "$plist" .plist)"
}

# ---------------------------------------------------------------------------
# subcommands
# ---------------------------------------------------------------------------

cmd_install() {
    require_macos
    check_tcc_protected_path
    mkdir -p "$INSTALL_DIR"
    log "Rendering plists with project=$PROJECT_DIR"
    render_plist "$ENGINE_LABEL.plist" "$ENGINE_PLIST"
    render_plist "$OLLAMA_LABEL.plist" "$OLLAMA_PLIST"
    log "Loading agents under $USER_DOMAIN"
    bootstrap_one "$ENGINE_PLIST"
    bootstrap_one "$OLLAMA_PLIST"
    log "Installed.  Logs:"
    note "    /tmp/prometa-inference-engine.{out,err}.log"
    note "    /tmp/prometa-ollama.{out,err}.log"
    log "Endpoints:"
    note "    engine: http://127.0.0.1:8080/v1"
    note "    ollama: http://127.0.0.1:11434"
}

cmd_uninstall() {
    require_macos
    bootout_if_loaded "$ENGINE_PLIST"
    bootout_if_loaded "$OLLAMA_PLIST"
    rm -f "$ENGINE_PLIST" "$OLLAMA_PLIST"
    log "Uninstalled engine + ollama launchd agents."
}

cmd_start() {
    require_macos
    [[ -f "$ENGINE_PLIST" ]] || { err "Not installed — run '$0 install' first."; exit 1; }
    bootstrap_one "$ENGINE_PLIST"
    bootstrap_one "$OLLAMA_PLIST"
    log "Started."
}

cmd_stop() {
    require_macos
    bootout_if_loaded "$ENGINE_PLIST"
    bootout_if_loaded "$OLLAMA_PLIST"
    log "Stopped (plists remain installed; '$0 start' to resume)."
}

cmd_restart() {
    cmd_stop
    cmd_start
}

cmd_status() {
    require_macos
    for label in "$ENGINE_LABEL" "$OLLAMA_LABEL"; do
        printf '\n=== %s ===\n' "$label"
        if launchctl print "$USER_DOMAIN/$label" 2>/dev/null \
            | grep -E '^\s+(state|pid|last exit code|program)\s*=' ; then
            :
        else
            note "    not loaded"
        fi
    done
    printf '\nLog files:\n'
    ls -lh /tmp/prometa-inference-engine.*.log /tmp/prometa-ollama.*.log 2>/dev/null \
        | awk '{printf "    %s  %s  %s\n", $5, $6" "$7" "$8, $9}' \
        || true
}

cmd_logs() {
    local target="${1:-engine}"
    case "$target" in
        engine|ENGINE) tail -F /tmp/prometa-inference-engine.{out,err}.log ;;
        ollama|OLLAMA) tail -F /tmp/prometa-ollama.{out,err}.log ;;
        *) err "logs takes 'engine' or 'ollama'"; exit 2 ;;
    esac
}

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
