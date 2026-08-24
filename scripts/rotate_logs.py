#!/usr/bin/env python3
"""rotate_logs.py — size-triggered rotation for the launchd agent logs.

Why this exists
---------------
The agents write to /tmp with no rotation. Over one eleven-day window the
Ollama sidecar's stderr reached 262 MB — 782k lines of ``slot launch`` and
``all slots are idle`` at llama-server's ``--log-verbosity 4``, which ollama
hardcodes and no documented env var lowers — and the engine's own log reached
72 MB.

Unbounded growth on the boot volume is the obvious problem. The quieter one is
retention: /tmp survives exactly one reboot, so the history you most want
during an incident is the history most likely to be gone. Gzipped generations
keep far more history in far less space.

Why Python and not the shell
----------------------------
This started as ``rotate-logs.sh``, which worked when run by hand and failed
under launchd with ``Operation not permitted`` (exit 126). The cause is TCC,
not the script: when the repository lives under a protected directory
(~/Desktop, ~/Documents), ``/bin/bash`` spawned by launchd cannot even *read*
the script file. The venv interpreter already holds the Full Disk Access grant
the engine agent depends on, so running under it sidesteps the problem without
asking anyone to widen a system permission for a log rotator.

Why copytruncate and not rename
-------------------------------
launchd opens StandardOutPath/StandardErrorPath once and holds the descriptor
for the life of the agent. Renaming would leave every agent writing to the
renamed inode while the "fresh" log stayed empty until the next restart.
Copying then truncating in place keeps the inode, so open descriptors keep
working. The trade-off is a small window between copy and truncate where
writes can be lost; for these logs that beats restarting a healthy service to
rotate a file.

Usage
-----
    python3 scripts/rotate_logs.py             # rotate anything over the threshold
    python3 scripts/rotate_logs.py --force     # rotate regardless of size
    python3 scripts/rotate_logs.py --dry-run   # report what would happen

Tunables (env, all overridable by flags):
    PROMETA_LOG_DIR        directory to scan            (default /tmp)
    PROMETA_LOG_PREFIX     basename prefix to match     (default prometa-)
    PROMETA_LOG_MAX_BYTES  rotate above this size       (default 64 MiB)
    PROMETA_LOG_KEEP       gzipped generations to keep  (default 5)
"""

from __future__ import annotations

import argparse
import gzip
import os
import shutil
import sys
import time
from pathlib import Path

DEFAULT_MAX_BYTES = 64 * 1024 * 1024
DEFAULT_KEEP = 5


def stamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def human(size: int) -> str:
    if size >= 1024 * 1024:
        return f"{size / (1024 * 1024):.1f} MiB"
    return f"{size / 1024:.1f} KiB"


def shift_generations(log: Path, keep: int) -> None:
    """Move ``log.N.gz`` to ``log.N+1.gz``, oldest first, dropping the tail."""
    for index in range(keep - 1, 0, -1):
        current = Path(f"{log}.{index}.gz")
        if current.exists():
            current.replace(Path(f"{log}.{index + 1}.gz"))
    stale = Path(f"{log}.{keep + 1}.gz")
    if stale.exists():
        stale.unlink()


def rotate_one(log: Path, *, max_bytes: int, keep: int, force: bool, dry_run: bool) -> bool:
    try:
        size = log.stat().st_size
    except OSError:
        return False
    if size == 0:
        return False
    if not force and size < max_bytes:
        return False

    if dry_run:
        print(f"{stamp()} would rotate {log} ({human(size)})", flush=True)
        return True

    shift_generations(log, keep)

    archive = Path(f"{log}.1")
    shutil.copyfile(log, archive)
    # Truncate in place — see the module docstring for why this cannot be a
    # rename. Opening "r+b" keeps the inode the agents already hold open.
    with open(log, "r+b") as handle:
        handle.truncate(0)

    with open(archive, "rb") as src, gzip.open(f"{archive}.gz", "wb") as dst:
        shutil.copyfileobj(src, dst)
    archive.unlink()

    print(f"{stamp()} rotated {log} ({human(size)} -> {log.name}.1.gz)", flush=True)
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--log-dir", default=os.environ.get("PROMETA_LOG_DIR", "/tmp"))
    parser.add_argument("--prefix", default=os.environ.get("PROMETA_LOG_PREFIX", "prometa-"))
    parser.add_argument("--max-bytes", type=int,
                        default=int(os.environ.get("PROMETA_LOG_MAX_BYTES", DEFAULT_MAX_BYTES)))
    parser.add_argument("--keep", type=int,
                        default=int(os.environ.get("PROMETA_LOG_KEEP", DEFAULT_KEEP)))
    parser.add_argument("--force", action="store_true", help="rotate regardless of size")
    parser.add_argument("--dry-run", action="store_true", help="report without changing anything")
    args = parser.parse_args(argv)

    log_dir = Path(args.log_dir)
    candidates = sorted(log_dir.glob(f"{args.prefix}*.log"))
    if not candidates:
        print(f"{stamp()} no logs matching {log_dir}/{args.prefix}*.log", flush=True)
        return 0

    rotated = 0
    for log in candidates:
        try:
            if rotate_one(log, max_bytes=args.max_bytes, keep=args.keep,
                          force=args.force, dry_run=args.dry_run):
                rotated += 1
        except OSError as exc:
            # One unreadable log must not stop the others: a rotator that gives
            # up halfway is how the largest file stays the largest file.
            print(f"{stamp()} FAILED {log}: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)

    if rotated == 0 and not args.dry_run:
        print(f"{stamp()} nothing over {human(args.max_bytes)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
