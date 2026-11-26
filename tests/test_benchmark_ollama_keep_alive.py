import json
import os
import sys

import httpx
import pytest

from scripts import benchmark_ollama_keep_alive as bench


@pytest.mark.parametrize("visible", [True, False])
def test_timeout_retains_observed_ttft_and_stops_the_workload(tmp_path, monkeypatch, visible):
    class InterruptedStream(httpx.SyncByteStream):
        def __iter__(self):
            yield b'data: {"choices":[{"delta":{"reasoning_content":"thinking"}}]}\n\n'
            yield b'data: {"choices":[{"delta":{"content":" "}}]}\n\n'
            if visible:
                yield b'data: {"choices":[{"delta":{"content":"7"}}]}\n\n'
            raise httpx.ReadTimeout("terminal SSE timed out")

    generation_hosts = []

    def handler(request):
        if request.url.path == "/api/tags":
            return httpx.Response(
                200, json={"models": [{"name": "ministral-3:8b", "digest": "8b"}]}
            )
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": "mock"})
        if request.url.path == "/api/ps":
            return httpx.Response(200, json={"models": []})
        assert request.url.path == "/v1/chat/completions"
        generation_hosts.append(request.url.host)
        if len(generation_hosts) == 4:
            return httpx.Response(200, stream=InterruptedStream())
        return httpx.Response(
            200, text='data: {"choices":[{"delta":{"content":"7"}}]}\n\ndata: [DONE]\n\n'
        )

    client = httpx.Client
    monkeypatch.setattr(
        httpx, "Client", lambda **kw: client(transport=httpx.MockTransport(handler), **kw)
    )
    pids = [os.getpid(), os.getppid()]
    monkeypatch.setattr(bench, "processes", lambda: {pid: (-1, 1024) for pid in pids})
    monkeypatch.setattr(bench, "footprint_reader", lambda: lambda pid: 1024)
    output = tmp_path / "result"
    argv = ["benchmark", "--output-dir", str(output), "--requests-per-burst", "1"]
    for arm, pid in zip(("baseline", "candidate"), pids, strict=True):
        log = tmp_path / (arm + ".log")
        log.touch()
        argv.extend(
            [
                f"--{arm}-url",
                f"http://{arm}",
                f"--{arm}-gateway-url",
                f"http://gw-{arm}",
                f"--{arm}-pid",
                str(pid),
                f"--{arm}-log",
                str(log),
            ]
        )
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(httpx.ReadTimeout, match="terminal SSE timed out"):
        bench.main()

    assert generation_hosts == ["gw-baseline", "gw-candidate", "gw-candidate", "gw-baseline"]
    attempts = [
        json.loads(line) for line in (output / "failed-attempts.jsonl").read_text().splitlines()
    ]
    (failed,) = attempts
    assert (failed["arm"], failed["phase"], failed["cycle"], failed["index"]) == (
        "baseline",
        "burst",
        0,
        0,
    )
    assert failed["error_type"] == "ReadTimeout" and failed["error"] == "terminal SSE timed out"
    if visible:
        assert failed["ttft_s"] > 0
    else:
        assert failed["ttft_s"] is None
    report = json.loads((output / "summary.json").read_text())
    assert report["workload_status"] == "incomplete"
    assert report["failed_attempts"] == attempts
    assert report["workload_error"] == "ReadTimeout: terminal SSE timed out"
    assert report["expected_requests"] == 8
    assert (
        report["completed_requests"]
        == len((output / "requests.jsonl").read_text().splitlines())
        == 3
    )
    assert report["arms"]["baseline"]["primary_ttft"]["n"] == 0
    assert report["arms"]["candidate"]["primary_ttft"]["n"] == 1
