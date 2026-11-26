"""Compare two isolated Ollama daemons under matched bursts separated by real idle time.

Supply the PID and log of each dedicated daemon. For gateway TTFT also supply two
isolated OpenAI-compatible engine URLs. In gateway mode configure OLLAMA_KEEP_ALIVE
on the daemons before launching them. Start with both target models nonresident.
The secondary model, if supplied, receives the baseline TTL in both arms.
Each burst alternates which daemon receives the paired prompt first. No inference
runs concurrently. Reports use nearest-rank p95 and record every raw observation.
"""

import argparse
import ctypes
import json
import math
import os
import platform
import re
import subprocess
import threading
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

import httpx


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_line(file, row: dict) -> None:
    file.write(json.dumps(row) + "\n")
    file.flush()


def ttl_seconds(text: str) -> float:
    match = re.fullmatch(r"(\d+(?:\.\d+)?)(s|m|h)", text)
    if not match:
        raise ValueError(f"Expected a positive duration such as 30m, got {text!r}")
    return float(match[1]) * {"s": 1, "m": 60, "h": 3600}[match[2]]


@dataclass
class Arm:
    name: str
    daemon: httpx.Client
    gateway: httpx.Client | None
    pid: int
    log: Path
    ttl: str


def get(client: httpx.Client, path: str) -> dict:
    response = client.get(path)
    response.raise_for_status()
    return response.json()


def resident(arm: Arm) -> dict:
    models = {}
    for row in get(arm.daemon, "/api/ps")["models"]:
        model = row["name"]
        for field in ("size", "size_vram"):
            value = row.get(field)
            if type(value) is not int or value < 0:
                raise ValueError(
                    f"{arm.name} /api/ps returned invalid {field} for {model}: {value!r}"
                )
        models[model] = {k: row.get(k) for k in ("size", "size_vram", "digest", "expires_at")}
    return models


class RusageV0(ctypes.Structure):
    # Darwin SDK sys/resource.h, struct rusage_info_v0. Values are bytes.
    _fields_ = [("uuid", ctypes.c_ubyte * 16)] + [
        (name, ctypes.c_uint64)
        for name in (
            "user",
            "system",
            "pkg_idle",
            "interrupt",
            "pageins",
            "wired",
            "resident",
            "footprint",
            "start",
            "exit",
        )
    ]


def footprint_reader():
    if platform.system() != "Darwin":
        return lambda _pid: None
    rusage = ctypes.CDLL("/usr/lib/libproc.dylib").proc_pid_rusage
    rusage.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.c_void_p)
    rusage.restype = ctypes.c_int

    def read(pid):
        stats = RusageV0()
        return stats.footprint if rusage(pid, 0, ctypes.byref(stats)) == 0 else None

    return read


def processes() -> dict[int, tuple[int, int]]:
    data = subprocess.check_output(["ps", "-axo", "pid=,ppid=,rss="], text=True)
    return {
        pid: (parent, kib * 1024)
        for pid, parent, kib in (map(int, line.split()) for line in data.splitlines())
    }


def process_family(root: int, table: dict[int, tuple[int, int]]) -> set[int]:
    if root not in table:
        raise RuntimeError(f"Daemon pid {root} exited")
    pending = {root}
    members = set()
    while pending:
        members.update(pending)
        pending = {pid for pid, (parent, _) in table.items() if parent in pending} - members
    return members


def sample_memory(
    arms: list[Arm], output: Path, interval: float, stop: threading.Event, failures: list[str]
) -> None:
    with output.open("x") as stream:
        footprint = footprint_reader()
        while not stop.is_set():
            started = time.monotonic()
            try:
                table = processes()
                for arm in arms:
                    pids = process_family(arm.pid, table)
                    footprints = [footprint(pid) for pid in pids]
                    models = resident(arm)
                    write_line(
                        stream,
                        {
                            "utc": now(),
                            "monotonic_s": time.monotonic(),
                            "arm": arm.name,
                            "pids": sorted(pids),
                            "rss_bytes": sum(table[p][1] for p in pids),
                            "footprint_bytes": (
                                sum(footprints) if None not in footprints else None
                            ),
                            "ollama_bytes": sum(m["size"] for m in models.values()),
                            "ollama_vram_bytes": sum(m["size_vram"] for m in models.values()),
                            "models": models,
                        },
                    )
            except (
                OSError,
                ValueError,
                RuntimeError,
                subprocess.SubprocessError,
                httpx.HTTPError,
            ) as error:
                failures.append(f"{now()} {error!s}")
                write_line(stream, {"utc": now(), "sampling_error": str(error)})
            stop.wait(max(0, interval - (time.monotonic() - started)))


def stream_chat(
    arm: Arm,
    model: str,
    prompt: str,
    context: int,
    predict: int,
    secondary: bool,
    baseline_ttl: str,
    attempt: dict | None = None,
) -> dict:
    before = resident(arm)
    log_offset = arm.log.stat().st_size
    messages = [{"role": "user", "content": prompt}]
    gateway = arm.gateway if not secondary else None
    if gateway:
        path = "/v1/chat/completions"
        body = {
            "model": model,
            "messages": messages,
            "stream": True,
            "temperature": 0,
            "top_p": 1.0,
            "seed": 21,
            "max_tokens": predict,
        }
    else:
        path = "/api/chat"
        body = {
            "model": model,
            "messages": messages,
            "stream": True,
            "keep_alive": baseline_ttl if secondary else arm.ttl,
            "options": {
                "temperature": 0,
                "top_p": 1.0,
                "seed": 21,
                "num_ctx": context,
                "num_predict": predict,
            },
        }
    client = gateway or arm.daemon
    first = None
    output = ""
    final = {}
    finished = False
    started_utc = now()
    if attempt is not None:
        attempt.update(transport=path, utc=started_utc, ttft_s=None)
    started = time.monotonic()
    with client.stream("POST", path, json=body) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line or (gateway and not line.startswith("data:")):
                continue
            payload = line[5:].strip() if gateway else line
            if payload == "[DONE]":
                finished = True
                break
            chunk = json.loads(payload)
            if chunk.get("error"):
                raise RuntimeError(f"{arm.name} streaming error: {chunk['error']}")
            if chunk.get("model") and chunk["model"] != model:
                raise RuntimeError(f"Requested {model}, received {chunk['model']}")
            if gateway:
                choices = chunk.get("choices", [])
                content = choices[0].get("delta", {}).get("content") if choices else None
            else:
                content = chunk.get("message", {}).get("content")
                if chunk.get("done"):
                    finished = True
                    final = chunk
            if isinstance(content, str):
                output += content
                if first is None and content.strip():
                    first = time.monotonic() - started
                    if attempt is not None:
                        attempt["ttft_s"] = first
    elapsed = time.monotonic() - started
    if first is None or not finished:
        raise RuntimeError(f"{arm.name} incomplete stream or no visible answer for {model}")
    with arm.log.open("rb") as logs:
        logs.seek(log_offset)
        # Ollama can log "llama-server started" repeatedly for one physical load.
        # This marker names the weight blob once at the actual load.
        loads = [
            s.split(b"model=")[-1].split()[0].decode(errors="replace").split("/")[-1]
            for s in logs.read().splitlines()
            if b'msg="loading model via llama-server"' in s
        ]
    return {
        "arm": arm.name,
        "model": model,
        "transport": path,
        "utc": started_utc,
        "prompt": prompt,
        "answer": output,
        "ttft_s": first,
        "elapsed_s": elapsed,
        "resident_before": before,
        "resident_after": resident(arm),
        "physical_load_blobs": loads,
        "ollama_load_duration_ns": final.get("load_duration"),
        "prompt_eval_count": final.get("prompt_eval_count"),
        "eval_count": final.get("eval_count"),
    }


def percentile(values: Iterable[float]) -> dict:
    ordered = sorted(values)
    return {
        "n": len(ordered),
        "p50_s": median(ordered) if ordered else None,
        "p95_s": ordered[math.ceil(0.95 * len(ordered)) - 1] if ordered else None,
    }


def summarize(
    arms: list[Arm],
    output: Path,
    failures: list[str],
    start_s: float,
    end_s: float,
    interval: float,
    *,
    expected_requests: int | None = None,
    workload_error: str | None = None,
) -> dict:
    requests = [json.loads(line) for line in (output / "requests.jsonl").read_text().splitlines()]
    failed_trace = output / "failed-attempts.jsonl"
    failed_attempts = (
        [json.loads(line) for line in failed_trace.read_text().splitlines()]
        if failed_trace.exists()
        else []
    )
    trace = output / "memory.jsonl"
    samples = (
        [json.loads(line) for line in trace.read_text().splitlines()] if trace.exists() else []
    )
    windows = [json.loads(line) for line in (output / "idle.jsonl").read_text().splitlines()]
    results = {}
    # Allow one missed polling interval for scheduler and HTTP jitter. Longer
    # intervals have unknown residency and cannot contribute measured byte-seconds.
    gap_limit = interval * 2
    for arm in arms:
        rows = [r for r in requests if r["arm"] == arm.name]
        measured = [r for r in rows if r["phase"] == "burst"]
        prime = next((r for r in rows if r["phase"] == "prime"), {})
        memory = [r for r in samples if r.get("arm") == arm.name]
        times = [start_s, *(r["monotonic_s"] for r in memory), end_s]
        gaps = [b - a for a, b in zip(times, times[1:])]
        if not memory:
            failures.append(f"{arm.name} has no memory samples")
        long_gaps = sum(gap > gap_limit for gap in gaps)
        if long_gaps:
            failures.append(
                f"{arm.name} has {long_gaps} sampling gap(s) longer than {gap_limit:g}s"
            )
        footprints = sum(r["footprint_bytes"] is not None for r in memory)
        if platform.system() == "Darwin" and footprints < len(memory):
            failures.append(
                f"{arm.name} physical footprint available for {footprints}/{len(memory)} samples"
            )
        load_events = sum(len(r["physical_load_blobs"]) for r in measured)
        memory_fields = ["rss_bytes", "footprint_bytes", "ollama_bytes", "ollama_vram_bytes"]
        idle_statistics = []
        for window in windows:
            first, last = window["start_s"], window["end_s"]
            byte_seconds = 0.0
            span = 0.0
            for prev, nxt in zip(memory, memory[1:]):
                seconds = min(nxt["monotonic_s"], last) - max(prev["monotonic_s"], first)
                if seconds > 0 and nxt["monotonic_s"] - prev["monotonic_s"] <= gap_limit:
                    byte_seconds += seconds * prev["ollama_bytes"]
                    span += seconds
            if not span:
                failures.append(
                    f"{arm.name} has no usable memory sample intervals in idle cycle {window['cycle']}"
                )
            idle_statistics.append(
                {
                    "cycle": window["cycle"],
                    "sampled_s": span,
                    "elapsed_s": last - first,
                    "unsampled_s": max(0.0, last - first - span),
                    "ollama_byte_seconds": byte_seconds,
                    "mean_ollama_bytes": byte_seconds / span if span else None,
                }
            )
        results[arm.name] = {
            "primary_ttft": percentile(r["ttft_s"] for r in measured),
            "first_after_gap_ttft": percentile(
                r["ttft_s"] for r in measured if r["index"] == 0 and r["cycle"] > 0
            ),
            "within_burst_ttft": percentile(r["ttft_s"] for r in measured if r["index"] > 0),
            "prime_log_loads": len(prime.get("physical_load_blobs", [])),
            "primary_reloads_after_prime": load_events
            if prime.get("physical_load_blobs")
            else None,
            "primary_nonresident_starts_after_prime": sum(
                r["model"] not in r["resident_before"] for r in measured
            ),
            "secondary_loads": sum(
                len(r["physical_load_blobs"]) for r in rows if r["phase"] == "secondary"
            ),
            "idle_windows": idle_statistics,
            "returning_bursts": [
                {
                    "cycle": r["cycle"],
                    "resident_before": r["model"] in r["resident_before"],
                    "ttft_s": r["ttft_s"],
                    "physical_load_events": len(r["physical_load_blobs"]),
                }
                for r in measured
                if r["index"] == 0 and r["cycle"] > 0
            ],
            "sample_count": len(memory),
            "longest_sample_gap_s": max(gaps),
            "footprint_samples": footprints,
            "footprint_availability": (
                "unavailable"
                if footprints == 0
                else "complete"
                if footprints == len(memory)
                else "partial"
            ),
            "peak": {
                k: max((r[k] for r in memory if r[k] is not None), default=None)
                for k in memory_fields
            },
        }
    workload_status = "unverified" if expected_requests is None else "complete"
    if (
        workload_error
        or failed_attempts
        or (expected_requests is not None and len(requests) != expected_requests)
    ):
        workload_status = "incomplete"
    return {
        "arms": results,
        "workload_status": workload_status,
        "workload_error": workload_error,
        "expected_requests": expected_requests,
        "completed_requests": len(requests),
        "failed_attempts": failed_attempts,
        "sampling_status": "incomplete" if failures else "complete",
        "sampling_gap_limit_s": gap_limit,
        "sampling_errors": failures,
    }


def run(args) -> None:
    output = args.output_dir
    output.mkdir(exist_ok=True, parents=True)
    if any(output.iterdir()):
        raise ValueError(f"Use an empty output directory: {output}")
    if args.secondary_model == args.model:
        raise ValueError("The secondary model must differ from the primary model")
    arms = []
    for name in ("baseline", "candidate"):

        def field(suffix):
            return getattr(args, name + "_" + suffix)

        client = httpx.Client(base_url=field("url"), trust_env=False, timeout=180)
        gateway = (
            httpx.Client(base_url=field("gateway_url"), trust_env=False, timeout=180)
            if field("gateway_url")
            else None
        )
        arms.append(Arm(name, client, gateway, field("pid"), field("log"), field("keep_alive")))
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config["started_at"] = now()
    metadata = {}
    targets = [args.model] + ([args.secondary_model] if args.secondary_model else [])
    for arm in arms:
        tags = {t["name"]: t["digest"] for t in get(arm.daemon, "/api/tags")["models"]}
        metadata[arm.name] = {
            "version": get(arm.daemon, "/api/version")["version"],
            "digests": {t: tags[t] for t in targets},
        }
        if resident(arm):
            raise ValueError(f"Start {arm.name} with an empty dedicated daemon")
        os.kill(arm.pid, 0)
        arm.log.stat()
    if metadata["baseline"] != metadata["candidate"]:
        raise ValueError("The arms have different Ollama versions or model digests")
    if args.secondary_model and len(set(metadata["baseline"]["digests"].values())) < 2:
        raise ValueError("The primary and secondary tags resolve to the same model digest")
    config["daemons"] = metadata
    with (output / "config.json").open("x") as f:
        json.dump(config, f, indent=2)
    stop = threading.Event()
    failures = []
    start_s = time.monotonic()
    monitor = ThreadPoolExecutor(max_workers=1)
    sampling = monitor.submit(
        sample_memory, arms, output / "memory.jsonl", args.sample_interval, stop, failures
    )
    workload_failure = None
    try:
        with (output / "requests.jsonl").open("x") as f, (output / "idle.jsonl").open("x") as idle:
            pair_index = 0

            def pair(phase, cycle, index, prompt):
                nonlocal pair_index
                secondary = phase == "secondary"
                model = args.secondary_model if secondary else args.model
                for arm in arms if pair_index % 2 == 0 else reversed(arms):
                    attempt = dict(
                        arm=arm.name,
                        phase=phase,
                        cycle=cycle,
                        index=index,
                        model=model,
                        prompt=prompt,
                        ttft_s=None,
                    )
                    try:
                        row = stream_chat(
                            arm,
                            model,
                            prompt,
                            args.context,
                            args.predict,
                            secondary,
                            args.baseline_keep_alive,
                            attempt=attempt,
                        )
                    except (Exception, KeyboardInterrupt) as exc:
                        attempt.update(
                            failed_at=now(), error_type=type(exc).__name__, error=str(exc)
                        )
                        with (output / "failed-attempts.jsonl").open("x") as failed:
                            write_line(failed, attempt)
                        raise
                    row.update(phase=phase, cycle=cycle, index=index)
                    write_line(f, row)
                pair_index += 1

            pair("prime", 0, 0, "Return only the integer sum of 17 and 5.")
            for cycle in range(args.cycles + 1):
                if cycle:
                    # Both arms finish the preceding burst before the common idle clock
                    # starts. The matched 14B requests occur after that clock expires.
                    start = time.monotonic()
                    print(
                        f"{now()} cycle {cycle} target idle gap {args.idle_seconds:g}s", flush=True
                    )
                    time.sleep(max(0, start + args.idle_seconds - time.monotonic()))
                    write_line(
                        idle,
                        {
                            "cycle": cycle,
                            "start_s": start,
                            "end_s": time.monotonic(),
                            "resident": {a.name: resident(a) for a in arms},
                        },
                    )
                    if args.secondary_model:
                        pair("secondary", cycle, 0, "Give only the capital city of Spain.")
                for index in range(args.requests_per_burst):
                    a, b = 10 * cycle + index + 3, index + 4
                    pair("burst", cycle, index, f"Give only the integer sum of {a} and {b}.")
    except (Exception, KeyboardInterrupt) as exc:
        workload_failure = exc
    finally:
        stop.set()
        monitor.shutdown(wait=True)
        end_s = time.monotonic()
        for arm in arms:
            arm.daemon.close()
            if arm.gateway:
                arm.gateway.close()
    if exc := sampling.exception():
        failures.append(f"Memory sampler stopped: {type(exc).__name__}: {exc}")
    expected = len(arms) * (
        1
        + (1 + args.cycles) * args.requests_per_burst
        + (args.cycles if args.secondary_model else 0)
    )
    report = summarize(
        arms,
        output,
        failures,
        start_s,
        end_s,
        args.sample_interval,
        expected_requests=expected,
        workload_error=(
            f"{type(workload_failure).__name__}: {workload_failure}"
            if workload_failure is not None
            else None
        ),
    )
    with (output / "summary.json").open("x") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    if workload_failure is not None:
        raise workload_failure
    if report["workload_status"] != "complete":
        raise SystemExit("Workload was incomplete. See summary.json.")
    if report["sampling_status"] != "complete":
        raise SystemExit("Memory sampling was incomplete. See summary.json and memory.jsonl.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for arm, default_ttl in (("baseline", "30m"), ("candidate", "60m")):
        p.add_argument(f"--{arm}-url", required=True)
        p.add_argument(f"--{arm}-pid", type=int, required=True)
        p.add_argument(f"--{arm}-log", type=Path, required=True)
        p.add_argument(f"--{arm}-gateway-url")
        p.add_argument(f"--{arm}-keep-alive", default=default_ttl)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--model", default="ministral-3:8b")
    p.add_argument("--secondary-model")
    p.add_argument("--idle-seconds", type=float, default=1860)
    p.add_argument("--cycles", type=int, default=2)
    p.add_argument("--requests-per-burst", type=int, default=10)
    p.add_argument("--context", type=int, default=32768)
    p.add_argument("--predict", type=int, default=24)
    p.add_argument("--sample-interval", type=float, default=1.0)
    args = p.parse_args()
    if not (
        0
        < ttl_seconds(args.baseline_keep_alive)
        < args.idle_seconds
        < ttl_seconds(args.candidate_keep_alive)
    ):
        p.error("Require baseline keep-alive < target idle gap < candidate keep-alive")
    if (
        not math.isfinite(args.sample_interval)
        or min(
            args.sample_interval, args.cycles, args.requests_per_burst, args.context, args.predict
        )
        <= 0
    ):
        p.error("Sample interval, cycle count, burst size and inference limits must be positive")
    if bool(args.baseline_gateway_url) != bool(args.candidate_gateway_url):
        p.error("Supply both gateways or neither")
    if args.baseline_url == args.candidate_url or args.baseline_pid == args.candidate_pid:
        p.error("The two arms need distinct daemons")
    run(args)


if __name__ == "__main__":
    main()
