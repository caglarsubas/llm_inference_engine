#!/usr/bin/env python3
"""tunnel_probe.py — measure the endpoint callers actually reach.

Why this exists
---------------
Every health signal the engine emits is measured from inside the host. The
tunnel in front of it is not, and over one eleven-day window it was the larger
availability risk by a wide margin: 402 session drops, 420 heartbeat timeouts
and 1,147 failed reconnect attempts, none of which appear anywhere in the
engine's telemetry. ``/v1/health`` reported ``ok`` throughout, because from
localhost it was.

So this probes the *public* URL end to end — DNS, ngrok edge, tunnel, engine —
and records what a caller on the internet would have seen.

Two signals, one pass
---------------------
The local ngrok agent API says whether a tunnel is *registered*; the public URL
says whether it *serves*. Recording both disambiguates the two failure modes
that otherwise look identical from outside:

* no tunnel registered  → the agent lost its session (the observed failure)
* tunnel registered but the request fails → the edge or the engine is the problem

Cost note: each run makes one real request through the tunnel, which counts
against the ngrok plan's request budget. At the default 60 s cadence that is
~43k/month. Raise ``StartInterval`` in the plist if that matters more than
resolution.

Output
------
* stdout — one line per run (launchd captures it; a state *change* is marked
  so ``grep CHANGE`` gives the outage history directly)
* ``--metrics-file`` — Prometheus textfile, scrapeable by node_exporter's
  textfile collector or any file-based scrape

Exit status is always 0 on a completed probe, including a probe that found the
tunnel down: a down tunnel is a measurement, not a failure of the measurement.
Non-zero means the probe itself could not run, which keeps launchd's
``last exit code`` meaningful.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_AGENT_API = "http://127.0.0.1:4040/api/tunnels"
DEFAULT_HEALTH_PATH = "/v1/health"
DEFAULT_STATE = "/tmp/planeon-tunnel-probe.state.json"
DEFAULT_METRICS = "/tmp/planeon-tunnel-probe.prom"


def _get_json(url: str, timeout: float) -> dict | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:  # noqa: S310 - fixed localhost URL
            return json.load(r)
    except Exception:
        return None


def resolve_public_url(agent_api: str, timeout: float) -> str | None:
    """Return the https public URL the agent is currently serving, if any."""
    payload = _get_json(agent_api, timeout)
    if not payload:
        return None
    tunnels = payload.get("tunnels") or []
    for tunnel in tunnels:
        if tunnel.get("proto") == "https" and tunnel.get("public_url"):
            return str(tunnel["public_url"])
    # An http-only tunnel still counts as registered.
    for tunnel in tunnels:
        if tunnel.get("public_url"):
            return str(tunnel["public_url"])
    return None


def probe(url: str, timeout: float) -> tuple[bool, int, float, str]:
    """GET ``url``; return (ok, http_status, elapsed_seconds, detail)."""
    started = time.perf_counter()
    request = urllib.request.Request(url, headers={"User-Agent": "planeon-tunnel-probe/1"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as r:  # noqa: S310
            r.read(2048)
            elapsed = time.perf_counter() - started
            return True, r.status, elapsed, "ok"
    except urllib.error.HTTPError as exc:
        # The tunnel delivered a response, so it is up even when the engine
        # answers 4xx/5xx. Keep those apart from a transport failure: an
        # unauthenticated probe legitimately gets a 401.
        elapsed = time.perf_counter() - started
        reachable = exc.code < 500
        return reachable, exc.code, elapsed, f"http_{exc.code}"
    except Exception as exc:
        elapsed = time.perf_counter() - started
        return False, 0, elapsed, f"{type(exc).__name__}: {exc}"[:120]


def load_state(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def write_state(path: Path, state: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True))
    tmp.replace(path)


def write_metrics(path: Path, state: dict, up: bool, elapsed: float, status: int) -> None:
    lines = [
        "# HELP planeon_tunnel_up Public endpoint answered a probe (1) or did not (0).",
        "# TYPE planeon_tunnel_up gauge",
        f"planeon_tunnel_up {1 if up else 0}",
        "# HELP planeon_tunnel_registered An ngrok tunnel is registered with the local agent.",
        "# TYPE planeon_tunnel_registered gauge",
        f"planeon_tunnel_registered {1 if state.get('public_url') else 0}",
        "# HELP planeon_tunnel_probe_duration_seconds Round-trip time of the last probe.",
        "# TYPE planeon_tunnel_probe_duration_seconds gauge",
        f"planeon_tunnel_probe_duration_seconds {elapsed:.4f}",
        "# HELP planeon_tunnel_probe_status_code HTTP status of the last probe (0 = no response).",
        "# TYPE planeon_tunnel_probe_status_code gauge",
        f"planeon_tunnel_probe_status_code {status}",
        "# HELP planeon_tunnel_probes_total Probes completed since the state file was created.",
        "# TYPE planeon_tunnel_probes_total counter",
        f"planeon_tunnel_probes_total {state.get('probes_total', 0)}",
        "# HELP planeon_tunnel_failures_total Probes that found the endpoint unreachable.",
        "# TYPE planeon_tunnel_failures_total counter",
        f"planeon_tunnel_failures_total {state.get('failures_total', 0)}",
        "# HELP planeon_tunnel_consecutive_failures Consecutive failing probes.",
        "# TYPE planeon_tunnel_consecutive_failures gauge",
        f"planeon_tunnel_consecutive_failures {state.get('consecutive_failures', 0)}",
        "# HELP planeon_tunnel_transitions_total Up/down state changes observed.",
        "# TYPE planeon_tunnel_transitions_total counter",
        f"planeon_tunnel_transitions_total {state.get('transitions_total', 0)}",
        "# HELP planeon_tunnel_last_transition_unixtime When the state last changed.",
        "# TYPE planeon_tunnel_last_transition_unixtime gauge",
        f"planeon_tunnel_last_transition_unixtime {state.get('last_transition_unixtime', 0)}",
    ]
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("\n".join(lines) + "\n")
    tmp.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--agent-api", default=os.environ.get("PLANEON_NGROK_API", DEFAULT_AGENT_API))
    parser.add_argument("--url", default=os.environ.get("PLANEON_TUNNEL_URL", ""),
                        help="probe this URL instead of discovering it from the agent API")
    parser.add_argument("--health-path", default=os.environ.get("PLANEON_TUNNEL_HEALTH_PATH",
                                                                DEFAULT_HEALTH_PATH))
    parser.add_argument("--timeout", type=float,
                        default=float(os.environ.get("PLANEON_TUNNEL_TIMEOUT", "10")))
    parser.add_argument("--state-file", default=os.environ.get("PLANEON_TUNNEL_STATE", DEFAULT_STATE))
    parser.add_argument("--metrics-file",
                        default=os.environ.get("PLANEON_TUNNEL_METRICS", DEFAULT_METRICS))
    args = parser.parse_args(argv)

    state_path = Path(args.state_file)
    state = load_state(state_path)

    public_url = args.url or resolve_public_url(args.agent_api, args.timeout)
    if not public_url:
        up, status, elapsed, detail = False, 0, 0.0, "no_tunnel_registered"
    else:
        target = public_url.rstrip("/") + args.health_path
        up, status, elapsed, detail = probe(target, args.timeout)

    now = time.time()
    previous = state.get("up")
    changed = previous is not None and bool(previous) != up

    state["public_url"] = public_url or ""
    state["up"] = up
    state["detail"] = detail
    state["probes_total"] = int(state.get("probes_total", 0)) + 1
    if up:
        state["consecutive_failures"] = 0
    else:
        state["failures_total"] = int(state.get("failures_total", 0)) + 1
        state["consecutive_failures"] = int(state.get("consecutive_failures", 0)) + 1
    if changed:
        state["transitions_total"] = int(state.get("transitions_total", 0)) + 1
        state["last_transition_unixtime"] = int(now)
    state.setdefault("transitions_total", 0)
    state.setdefault("last_transition_unixtime", 0)

    write_state(state_path, state)
    write_metrics(Path(args.metrics_file), state, up, elapsed, status)

    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now))
    marker = "CHANGE " if changed else ""
    total = state["probes_total"]
    failures = state.get("failures_total", 0)
    availability = 100.0 * (total - failures) / total if total else 0.0
    print(
        f"{stamp} {marker}{'up' if up else 'DOWN'} "
        f"url={public_url or '-'} status={status} rtt={elapsed:.3f}s "
        f"detail={detail} consecutive_failures={state['consecutive_failures']} "
        f"availability={availability:.2f}% ({total - failures}/{total})",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
