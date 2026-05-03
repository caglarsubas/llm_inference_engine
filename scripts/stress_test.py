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
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass, field

import httpx


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


async def _fire_one(
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    model: str,
    prompt: str,
    *,
    max_tokens: int,
    stream: bool,
    headers: dict[str, str],
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
                async with client.stream("POST", "/v1/chat/completions", json=body, headers=headers) as r:
                    if r.status_code != 200:
                        await r.aread()
                        return _Outcome(model, started, time.perf_counter(), r.status_code, r.text)
                    async for line in r.aiter_lines():
                        if line.startswith("data: ") and "[DONE]" not in line:
                            chunks += 1
                return _Outcome(model, started, time.perf_counter(), 200, chunks=chunks)
            else:
                r = await client.post("/v1/chat/completions", json=body, headers=headers)
                if r.status_code != 200:
                    return _Outcome(model, started, time.perf_counter(), r.status_code, r.text)
                data = r.json()
                u = data.get("usage", {}) or {}
                return _Outcome(
                    model,
                    started,
                    time.perf_counter(),
                    200,
                    prompt_tokens=u.get("prompt_tokens", 0),
                    completion_tokens=u.get("completion_tokens", 0),
                )
        except Exception as exc:  # noqa: BLE001
            return _Outcome(model, started, time.perf_counter(), 0, str(exc))


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
) -> _Report:
    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    sem = asyncio.Semaphore(concurrency)
    report = _Report()

    async with httpx.AsyncClient(base_url=base_url, timeout=600.0) as client:
        tasks = [
            _fire_one(
                client, sem,
                models[i % len(models)],
                f"{prompt} (request {i})",
                max_tokens=max_tokens,
                stream=stream,
                headers=headers,
            )
            for i in range(n_requests)
        ]
        wall_start = time.perf_counter()
        for coro in asyncio.as_completed(tasks):
            o = await coro
            report.add(o)
            mark = "OK " if o.ok else f"ERR {o.status}"
            extra = f"chunks={o.chunks}" if stream else f"out_tok={o.completion_tokens}"
            print(f"  [{mark}] {o.model:<32} {o.latency_s * 1000:7.0f} ms  {extra}")
        report.wall_seconds = time.perf_counter() - wall_start  # type: ignore[attr-defined]
    return report


def _print_report(report: _Report, *, stream: bool, concurrency: int) -> None:
    by_model: dict[str, list[_Outcome]] = defaultdict(list)
    for o in report.outcomes:
        by_model[o.model].append(o)

    wall = getattr(report, "wall_seconds", 0.0)
    total = len(report.outcomes)
    ok_count = sum(1 for o in report.outcomes if o.ok)
    print()
    print(f"=== summary ===")
    print(f"  concurrency={concurrency}  total={total}  ok={ok_count}  err={total - ok_count}  wall={wall:.2f}s")

    for model, items in by_model.items():
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
        print(f"  {model}")
        print(f"    n={len(oks)}  p50={pcts[50]:.0f} ms  p95={pcts[95]:.0f} ms  p99={pcts[99]:.0f} ms")
        print(f"    {unit}")

    if any(not o.ok for o in report.outcomes):
        print()
        print("=== errors ===")
        for o in report.outcomes:
            if not o.ok:
                print(f"  [{o.status}] {o.model}: {o.error}")


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
    args = p.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    print(
        f"firing {args.requests} requests at concurrency={args.concurrency} across {models}"
        f" (stream={args.stream}, max_tokens={args.max_tokens})"
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
        )
    )
    _print_report(report, stream=args.stream, concurrency=args.concurrency)
    return 0 if all(o.ok for o in report.outcomes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
