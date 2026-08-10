"""Concurrent-traffic stress harness for the inference engine.

Drives N concurrent chat completions against a running server and reports
per-model latency percentiles + aggregate throughput. Built for two specific
investigations the unit tests can't cover:

1. **Same-model serialization** (the per-adapter lock).
2. **Cross-model parallelism** (the per-key load lock + ModelManager dispatch).

Usage:

    # one-shot: 20 requests to a single model, concurrency=8
    uv run python scripts/stress_test.py --requests 20 --concurrency 8 \\
        --models llama3.2:1b

    # cross-model: 20 requests split across two models, concurrency=8
    uv run python scripts/stress_test.py --requests 20 --concurrency 8 \\
        --models llama3.2:1b,Llama-3.2-1B-Instruct-4bit:mlx

    # streaming (verifies cancellation paths don't deadlock)
    uv run python scripts/stress_test.py --requests 16 --concurrency 8 \\
        --models llama3.2:1b --stream

The server must be running on ``--base-url`` (default ``http://127.0.0.1:8080``).
If the server has ``AUTH_ENABLED=true``, pass ``--api-key`` (or set
``ENGINE_API_KEY`` in the env).

Proxy-overhead mode
-------------------

The blueprint puts a budget on what this engine may ADD to a call: p50 < 5 ms
and p99 < 20 ms on the cache-miss path, and < 10 ms of extra time-to-first-token
when streaming. Total request latency cannot answer that on its own — almost
all of it is the model generating.

``--baseline-url`` / ``--baseline-model`` turn on the measurement. The harness
then fires the SAME workload twice: once through the engine and once straight
at the OpenAI-compatible upstream the engine proxies to, **interleaved
request-by-request** so upstream drift (a warming cache, a busy GPU, thermal
throttle) lands on both series rather than on one:

    uv run python scripts/stress_test.py --requests 200 --concurrency 1 \\
        --max-tokens 1 --models llama3.2:1b \\
        --baseline-url http://127.0.0.1:11434 --baseline-model llama3.2:1b

Added latency is reported as the difference between the two series AT EACH
PERCENTILE. That is a distribution-level statement, not a per-request one: the
p99 row says "the engine's p99 minus the upstream's p99", not "the worst
request paid this". Every request uses a distinct prompt, so neither series is
served from a prefix cache the other warmed.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field

import httpx


ENGINE_SERIES = "engine"
BASELINE_SERIES = "baseline"

# Blueprint budget for what the proxy may add on the cache-miss path.
TARGET_ADDED_P50_MS = 5.0
TARGET_ADDED_P99_MS = 20.0
TARGET_ADDED_TTFT_MS = 10.0


@dataclass
class _Outcome:
    model: str
    started_at: float
    finished_at: float
    status: int
    error: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    chunks: int = 0  # for streaming
    # Which leg of an overhead run this came from. Plain runs are all "engine".
    series: str = ENGINE_SERIES
    # Time to the first chunk carrying content, streaming only.
    ttft_s: float | None = None

    @property
    def latency_s(self) -> float:
        return self.finished_at - self.started_at

    @property
    def ok(self) -> bool:
        return self.status == 200 and self.error is None


@dataclass
class _Report:
    outcomes: list[_Outcome] = field(default_factory=list)

    def add(self, o: _Outcome) -> None:
        self.outcomes.append(o)


def _percentiles(values: list[float], pcts: tuple[float, ...] = (50, 95, 99)) -> dict[float, float]:
    if not values:
        return {p: float("nan") for p in pcts}
    sorted_v = sorted(values)
    out: dict[float, float] = {}
    for p in pcts:
        # nearest-rank percentile (good enough for a stress harness)
        rank = max(0, min(len(sorted_v) - 1, int(round(p / 100 * (len(sorted_v) - 1)))))
        out[p] = sorted_v[rank]
    return out


def _chunk_has_content(line: str) -> bool:
    """Does this SSE line carry generated text? Role-only frames do not.

    TTFT must be measured against the first frame a caller could render. The
    first frame of an OpenAI stream is ``delta: {"role": "assistant"}``, which
    carries nothing, and counting it would understate the number on whichever
    side emits it sooner.
    """
    payload = line[len("data: ") :].strip()
    if not payload or payload == "[DONE]":
        return False
    try:
        event = json.loads(payload)
    except json.JSONDecodeError:
        return False
    for choice in event.get("choices") or []:
        delta = choice.get("delta") or {}
        if delta.get("content") or delta.get("tool_calls") or delta.get("reasoning_content"):
            return True
    return False


async def _fire_one(
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    model: str,
    prompt: str,
    *,
    max_tokens: int,
    stream: bool,
    headers: dict[str, str],
    series: str = ENGINE_SERIES,
) -> _Outcome:
    async with sem:
        started = time.perf_counter()
        body = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.2,
            "stream": stream,
        }
        try:
            if stream:
                chunks = 0
                ttft_s: float | None = None
                async with client.stream("POST", "/v1/chat/completions", json=body, headers=headers) as r:
                    if r.status_code != 200:
                        await r.aread()
                        return _Outcome(
                            model, started, time.perf_counter(), r.status_code, r.text,
                            series=series,
                        )
                    async for line in r.aiter_lines():
                        if line.startswith("data: ") and "[DONE]" not in line:
                            chunks += 1
                            if ttft_s is None and _chunk_has_content(line):
                                ttft_s = time.perf_counter() - started
                return _Outcome(
                    model, started, time.perf_counter(), 200,
                    chunks=chunks, series=series, ttft_s=ttft_s,
                )
            else:
                r = await client.post("/v1/chat/completions", json=body, headers=headers)
                if r.status_code != 200:
                    return _Outcome(
                        model, started, time.perf_counter(), r.status_code, r.text,
                        series=series,
                    )
                data = r.json()
                u = data.get("usage", {}) or {}
                return _Outcome(
                    model,
                    started,
                    time.perf_counter(),
                    200,
                    prompt_tokens=u.get("prompt_tokens", 0),
                    completion_tokens=u.get("completion_tokens", 0),
                    series=series,
                )
        except Exception as exc:  # noqa: BLE001
            return _Outcome(model, started, time.perf_counter(), 0, str(exc), series=series)


async def _run(
    *,
    base_url: str,
    api_key: str | None,
    models: list[str],
    n_requests: int,
    concurrency: int,
    max_tokens: int,
    stream: bool,
    prompt: str,
    baseline_url: str | None = None,
    baseline_model: str | None = None,
    baseline_api_key: str | None = None,
    quiet: bool = False,
) -> _Report:
    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    baseline_headers: dict[str, str] = {}
    if baseline_api_key:
        baseline_headers["Authorization"] = f"Bearer {baseline_api_key}"

    sem = asyncio.Semaphore(concurrency)
    report = _Report()
    measure_overhead = bool(baseline_url and baseline_model)

    async with httpx.AsyncClient(base_url=base_url, timeout=600.0) as client:
        baseline_client = (
            httpx.AsyncClient(base_url=baseline_url, timeout=600.0)
            if measure_overhead
            else None
        )
        try:
            tasks = []
            # One prompt index per FIRED REQUEST, not per pair, so the two
            # series never share a prompt and neither is served from a prefix
            # cache the other warmed.
            index = 0
            for i in range(n_requests):
                tasks.append(
                    _fire_one(
                        client, sem,
                        models[i % len(models)],
                        f"{prompt} (request {index})",
                        max_tokens=max_tokens,
                        stream=stream,
                        headers=headers,
                    )
                )
                index += 1
                if baseline_client is not None:
                    tasks.append(
                        _fire_one(
                            baseline_client, sem,
                            baseline_model or "",
                            f"{prompt} (request {index})",
                            max_tokens=max_tokens,
                            stream=stream,
                            headers=baseline_headers,
                            series=BASELINE_SERIES,
                        )
                    )
                    index += 1

            wall_start = time.perf_counter()
            if measure_overhead:
                # Submission order is engine, baseline, engine, … and the
                # semaphore admits waiters FIFO, so the two series stay spread
                # evenly across the run instead of one following the other.
                # That is what makes upstream drift common-mode between them
                # rather than a bias on whichever series ran second.
                running = [asyncio.create_task(coro) for coro in tasks]
                for task in running:
                    _record(report, await task, stream=stream, quiet=quiet)
            else:
                for coro in asyncio.as_completed(tasks):
                    _record(report, await coro, stream=stream, quiet=quiet)
            report.wall_seconds = time.perf_counter() - wall_start  # type: ignore[attr-defined]
        finally:
            if baseline_client is not None:
                await baseline_client.aclose()
    return report


def _record(report: _Report, o: _Outcome, *, stream: bool, quiet: bool) -> None:
    report.add(o)
    if quiet:
        return
    mark = "OK " if o.ok else f"ERR {o.status}"
    extra = f"chunks={o.chunks}" if stream else f"out_tok={o.completion_tokens}"
    print(f"  [{mark}] {o.series:<9} {o.model:<32} {o.latency_s * 1000:7.0f} ms  {extra}")


def _print_report(report: _Report, *, stream: bool, concurrency: int) -> None:
    by_model: dict[tuple[str, str], list[_Outcome]] = defaultdict(list)
    for o in report.outcomes:
        by_model[(o.series, o.model)].append(o)

    wall = getattr(report, "wall_seconds", 0.0)
    total = len(report.outcomes)
    ok_count = sum(1 for o in report.outcomes if o.ok)
    print()
    print("=== summary ===")
    print(f"  concurrency={concurrency}  total={total}  ok={ok_count}  err={total - ok_count}  wall={wall:.2f}s")

    for (series, model), items in by_model.items():
        oks = [o for o in items if o.ok]
        latencies = [o.latency_s * 1000 for o in oks]
        pcts = _percentiles(latencies)
        completion_tokens = sum(o.completion_tokens for o in oks)
        chunks = sum(o.chunks for o in oks)
        if stream:
            unit = f"{chunks} chunks  ({chunks / wall:.1f} chunks/s aggregate)"
        else:
            unit = (
                f"{completion_tokens} out tokens  "
                f"({completion_tokens / wall:.1f} tok/s aggregate)"
                if wall > 0
                else "no wall time"
            )
        label = model if series == ENGINE_SERIES else f"{model}  [{series}]"
        print(f"  {label}")
        print(f"    n={len(oks)}  p50={pcts[50]:.0f} ms  p95={pcts[95]:.0f} ms  p99={pcts[99]:.0f} ms")
        print(f"    {unit}")

    _print_overhead(report, stream=stream)

    if any(not o.ok for o in report.outcomes):
        print()
        print("=== errors ===")
        for o in report.outcomes:
            if not o.ok:
                print(f"  [{o.status}] {o.series} {o.model}: {o.error}")


def _verdict(added_ms: float, target_ms: float) -> str:
    return "OK" if added_ms <= target_ms else "MISS"


def _print_overhead(report: _Report, *, stream: bool) -> None:
    """Report what the proxy ADDS, separately from what the upstream costs."""
    engine = [o for o in report.outcomes if o.ok and o.series == ENGINE_SERIES]
    baseline = [o for o in report.outcomes if o.ok and o.series == BASELINE_SERIES]
    if not engine or not baseline:
        return

    print()
    print("=== proxy overhead (engine minus direct upstream, per percentile) ===")

    def _rows(name: str, engine_ms: list[float], baseline_ms: list[float], targets: dict) -> None:
        e = _percentiles(engine_ms)
        b = _percentiles(baseline_ms)
        print(f"  {name}")
        print(
            f"    upstream direct  n={len(baseline_ms):<5} "
            f"p50={b[50]:8.2f} ms  p95={b[95]:8.2f} ms  p99={b[99]:8.2f} ms"
        )
        print(
            f"    through engine   n={len(engine_ms):<5} "
            f"p50={e[50]:8.2f} ms  p95={e[95]:8.2f} ms  p99={e[99]:8.2f} ms"
        )
        added = {p: e[p] - b[p] for p in (50, 95, 99)}
        print(
            f"    ADDED                  "
            f"    p50={added[50]:8.2f} ms  p95={added[95]:8.2f} ms  p99={added[99]:8.2f} ms"
        )
        for pct, target in targets.items():
            print(
                f"    target p{pct} < {target:.0f} ms → {_verdict(added[pct], target)} "
                f"(measured {added[pct]:.2f} ms)"
            )

    _rows(
        "total request latency",
        [o.latency_s * 1000 for o in engine],
        [o.latency_s * 1000 for o in baseline],
        {50: TARGET_ADDED_P50_MS, 99: TARGET_ADDED_P99_MS},
    )

    if stream:
        engine_ttft = [o.ttft_s * 1000 for o in engine if o.ttft_s is not None]
        baseline_ttft = [o.ttft_s * 1000 for o in baseline if o.ttft_s is not None]
        if engine_ttft and baseline_ttft:
            _rows(
                "time to first content chunk",
                engine_ttft,
                baseline_ttft,
                {50: TARGET_ADDED_TTFT_MS},
            )

    print(
        "  note: each row is a difference of PERCENTILES across two interleaved\n"
        "        series, not a per-request delta. It includes the client's own\n"
        "        loopback hop to the engine, which a caller pays too."
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", default=os.environ.get("ENGINE_URL", "http://127.0.0.1:8080"))
    p.add_argument("--api-key", default=os.environ.get("ENGINE_API_KEY"))
    p.add_argument("--models", required=True, help="comma-separated model ids")
    p.add_argument("--requests", type=int, default=20)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--stream", action="store_true")
    p.add_argument(
        "--prompt",
        default="Respond with a single short sentence.",
        help="prompt text shared by all requests",
    )
    p.add_argument(
        "--baseline-url",
        default=os.environ.get("BASELINE_URL"),
        help=(
            "OpenAI-compatible base URL of the upstream the engine proxies to. "
            "Setting this (with --baseline-model) turns on proxy-overhead mode."
        ),
    )
    p.add_argument(
        "--baseline-model",
        default=os.environ.get("BASELINE_MODEL"),
        help="the upstream's OWN model id, as it names it (not the engine's id)",
    )
    p.add_argument("--baseline-api-key", default=os.environ.get("BASELINE_API_KEY"))
    p.add_argument(
        "--quiet",
        action="store_true",
        help="suppress the per-request lines; print only the summary",
    )
    args = p.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    overhead = bool(args.baseline_url and args.baseline_model)
    print(
        f"firing {args.requests} requests at concurrency={args.concurrency} across {models}"
        f" (stream={args.stream}, max_tokens={args.max_tokens})"
    )
    if overhead:
        print(
            f"proxy-overhead mode: interleaving with direct calls to "
            f"{args.baseline_url} model={args.baseline_model}"
        )
    report = asyncio.run(
        _run(
            base_url=args.base_url,
            api_key=args.api_key,
            models=models,
            n_requests=args.requests,
            concurrency=args.concurrency,
            max_tokens=args.max_tokens,
            stream=args.stream,
            prompt=args.prompt,
            baseline_url=args.baseline_url,
            baseline_model=args.baseline_model,
            baseline_api_key=args.baseline_api_key,
            quiet=args.quiet,
        )
    )
    _print_report(report, stream=args.stream, concurrency=args.concurrency)
    return 0 if all(o.ok for o in report.outcomes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
