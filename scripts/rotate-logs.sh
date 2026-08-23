#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# rotate-logs.sh — size-triggered rotation for the launchd agent logs.
#
# Why this exists
# ---------------
# The agents write to /tmp with no rotation. On one eleven-day window the
# Ollama sidecar's stderr reached 262 MB (782k lines of `slot launch` and
# `all slots are idle` at llama-server's `--log-verbosity 4`, which ollama
# hardcodes and no env var lowers), and the engine's own log reached 72 MB.
#
# That is two problems at once. Unbounded growth on the boot volume is the
# obvious one. The quieter one is retention: /tmp survives exactly one reboot,
# so the history you want during an incident is the history most likely to be
# gone. Rotating into gzipped generations keeps far more history in far less
# space.
#
# Why copytruncate rather than rename
# -----------------------------------
# launchd opens StandardOutPath/StandardErrorPath once and holds the descriptor
# for the life of the agent. Renaming the file would leave every agent happily
# writing to the renamed inode, and the "fresh" log would stay empty until the
# next restart. Copying then truncating in place keeps the inode, so the open
# descriptors keep working. The trade-off is a small window between the copy
# and the truncate where writes can be lost; for these logs that is the right
# trade against restarting a healthy service to rotate a file.
#
# Usage
# -----
#   ./scripts/rotate-logs.sh            # rotate anything over the threshold
#   ./scripts/rotate-logs.sh --force    # rotate regardless of size
#   ./scripts/rotate-logs.sh --dry-run  # report what would happen
#
# Tunables (env):
#   PROMETA_LOG_DIR        directory to scan            (default /tmp)
#   PROMETA_LOG_PREFIX     basename prefix to match     (default prometa-)
#   PROMETA_LOG_MAX_BYTES  rotate above this size       (default 64 MiB)
#   PROMETA_LOG_KEEP       gzipped generations to keep  (default 5)
# ---------------------------------------------------------------------------
set -euo pipefail

LOG_DIR="${PROMETA_LOG_DIR:-/tmp}"
LOG_PREFIX="${PROMETA_LOG_PREFIX:-prometa-}"
MAX_BYTES="${PROMETA_LOG_MAX_BYTES:-$((64 * 1024 * 1024))}"
KEEP="${PROMETA_LOG_KEEP:-5}"

FORCE=0
DRY_RUN=0
for arg in "$@"; do
    case "$arg" in
        --force)   FORCE=1 ;;
        --dry-run) DRY_RUN=1 ;;
        -h|--help) sed -n '2,/^# ---*$/p' "$0" | sed 's/^# \{0,1\}//; s/^#//'; exit 0 ;;
        *) printf 'unknown argument: %s\n' "$arg" >&2; exit 2 ;;
    esac
done

stamp() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }

size_of() {
    # stat(1) is BSD here; fall back to 0 for a file that vanished mid-run.
    stat -f%z "$1" 2>/dev/null || printf '0'
}

human() {
    awk -v b="$1" 'BEGIN { printf (b >= 1048576) ? "%.1f MiB" : "%.1f KiB", (b >= 1048576) ? b/1048576 : b/1024 }'
}

rotate_one() {
    local log="$1" size
    size="$(size_of "$log")"

    if [[ "$FORCE" -ne 1 ]] && [[ "$size" -lt "$MAX_BYTES" ]]; then
        return 0
    fi
    if [[ "$size" -eq 0 ]]; then
        return 0
    fi

    if [[ "$DRY_RUN" -eq 1 ]]; then
        printf '%s would rotate %s (%s)\n' "$(stamp)" "$log" "$(human "$size")"
        return 0
    fi

    # Shift existing generations down; the oldest falls off the end.
    local i
    for (( i = KEEP - 1; i >= 1; i-- )); do
        if [[ -f "$log.$i.gz" ]]; then
            mv -f "$log.$i.gz" "$log.$((i + 1)).gz"
        fi
    done
    rm -f "$log.$((KEEP + 1)).gz"

    # copytruncate — see the header for why this cannot be a rename.
    cp "$log" "$log.1"
    : > "$log"
    gzip -f "$log.1"

    printf '%s rotated %s (%s -> %s.1.gz)\n' \
        "$(stamp)" "$log" "$(human "$size")" "$(basename "$log")"
}

shopt -s nullglob
found=0
for log in "$LOG_DIR/$LOG_PREFIX"*.log; do
    found=1
    rotate_one "$log"
done

if [[ "$found" -eq 0 ]]; then
    printf '%s no logs matching %s/%s*.log\n' "$(stamp)" "$LOG_DIR" "$LOG_PREFIX"
fi
