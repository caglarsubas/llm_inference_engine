"""A judge backend that never answers costs /v1/evals/run the deadline, no more.

The existing 504 test (``test_eval_run_judging``) drives a scripted adapter
that sleeps. These drive the real ``OllamaHttpAdapter`` over a loopback socket
to a server that takes ``POST /api/chat`` and never finishes it, which is what a
stuck llama-server slot looks like from the engine. Three things are pinned:

* the 504 ``generation_timeout`` arrives within the configured timeout;
* the engine hangs up on the upstream, which is what makes Ollama abort the
  generation and free its slot (a real Ollama logs that as a 500);
* the next judge call is served.

Two ways of never answering, because they fail differently. ``silent`` sends
nothing, so httpx's own read timeout would catch it too. ``drip`` sends headers
and then one body byte at a time forever. That resets httpx's per-read timeout
on every byte, so only the total-elapsed deadline ends it.

Prompted by the 2026-09-29 report of judge calls that 504'd 15–17 minutes
after they started. That one was the host sleeping mid-request, not the
engine; see ``generation_deadline``.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import httpx
import pytest

from inference_engine.adapters.ollama_http import OllamaHttpAdapter
from inference_engine.config import settings
from inference_engine.evals import EvalRunner, RubricRegistry, TenantRubricStore
from inference_engine.main import app
from inference_engine.manager import ModelManager
from inference_engine.registry import ModelDescriptor
from inference_engine.scheduler import TenantScheduler

JUDGE = "qwen3.6:27b"
DEADLINE = 0.5
# Unwinding the cancellation and rendering the 504 take real time; a second of
# slack keeps a loaded CI runner from flaking without hiding a missed deadline.
SLACK = 1.0

_VERDICT = json.dumps(
    {
        "model": JUDGE,
        "message": {"role": "assistant", "content": '{"score": 4, "justification": "ok"}'},
        "done": True,
        "done_reason": "stop",
        "prompt_eval_count": 10,
        "eval_count": 5,
    }
).encode()


class _HungOllama:
    """Loopback ``/api/chat`` that hangs on the first ``hang`` requests, then answers."""

    def __init__(self, mode: str, *, hang: int = 1) -> None:
        self.mode = mode
        self.hang = hang
        self.accepted = 0
        self.hung_up = asyncio.Event()  # the engine closed a hung request
        self._writers: list[asyncio.StreamWriter] = []
        self._server: asyncio.Server | None = None

    async def __aenter__(self) -> str:
        self._server = await asyncio.start_server(self._handle, "127.0.0.1", 0)
        port = self._server.sockets[0].getsockname()[1]
        return f"http://127.0.0.1:{port}"

    async def __aexit__(self, *exc_info) -> None:
        assert self._server is not None
        self._server.close()
        for writer in self._writers:
            writer.close()
        await self._server.wait_closed()

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self._writers.append(writer)
        self.accepted += 1
        hang = self.accepted <= self.hang
        try:
            head = await reader.readuntil(b"\r\n\r\n")
            length = next(
                int(line.split(b":", 1)[1])
                for line in head.split(b"\r\n")
                if line.lower().startswith(b"content-length:")
            )
            await reader.readexactly(length)
            if not hang:
                writer.write(
                    b"HTTP/1.1 200 OK\r\ncontent-type: application/json\r\n"
                    b"content-length: %d\r\n\r\n%s" % (len(_VERDICT), _VERDICT)
                )
                await writer.drain()
                return
            if self.mode == "silent":
                while await reader.read(1024):  # b"" is the engine hanging up
                    pass
            else:
                writer.write(
                    b"HTTP/1.1 200 OK\r\ncontent-type: application/json\r\n"
                    b"transfer-encoding: chunked\r\n\r\n"
                )
                while not reader.at_eof():
                    writer.write(b"1\r\n \r\n")
                    await writer.drain()
                    await asyncio.sleep(0.02)
            self.hung_up.set()
        except (ConnectionError, asyncio.IncompleteReadError):
            if hang:
                self.hung_up.set()
        finally:
            writer.close()


@pytest.fixture
async def judge_at(monkeypatch):
    """Point the eval route at a real Ollama adapter on ``endpoint``."""
    from inference_engine.api.state import app_state  # noqa: PLC0415

    monkeypatch.setattr(settings, "chat_completion_timeout_seconds", DEADLINE)
    monkeypatch.setattr(settings, "scheduler_enabled", True)
    monkeypatch.setattr(app_state, "rubric_registry", RubricRegistry.with_builtins())
    monkeypatch.setattr(app_state, "tenant_rubrics", TenantRubricStore())
    monkeypatch.setattr(app_state, "scheduler", TenantScheduler())
    managers: list[ModelManager] = []

    def _install(endpoint: str) -> None:
        name, tag = JUDGE.split(":")
        desc = ModelDescriptor(
            name=name,
            tag=tag,
            namespace="library",
            registry="registry.ollama.ai",
            model_path=Path(f"ollama_http://{endpoint}/{JUDGE}"),
            format="ollama_http",
            params={"model_id": JUDGE},
            size_bytes=0,
            endpoint=endpoint,
        )

        class _Reg:
            def get(self, model: str) -> ModelDescriptor | None:
                return desc if model == JUDGE else None

            def list_models(self) -> list[ModelDescriptor]:
                return [desc]

        manager = ModelManager(
            _Reg(), adapter_factory=lambda d: OllamaHttpAdapter(), memory_budget_bytes=100
        )
        managers.append(manager)
        monkeypatch.setattr(app_state, "eval_runner", EvalRunner(manager))

    yield _install
    for manager in managers:
        await manager.shutdown()


async def _post() -> tuple[httpx.Response, float]:
    body = {
        "rubric": "helpfulness",
        "prompt": "How do I reset my password?",
        "response": "Use the 'Forgot password' link.",
        "judge_model": JUDGE,
    }
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://engine"
    ) as client:
        start = time.perf_counter()
        # ASGITransport ignores httpx timeouts, so a missed deadline would hang
        # the test rather than fail it without this bound.
        try:
            response = await asyncio.wait_for(
                client.post("/v1/evals/run", json=body), DEADLINE + 5 * SLACK
            )
        except TimeoutError:
            pytest.fail(f"no response after {time.perf_counter() - start:.1f}s")
        return response, time.perf_counter() - start


@pytest.mark.parametrize("mode", ["silent", "drip"])
async def test_a_hung_judge_backend_is_a_504_within_the_deadline(judge_at, mode) -> None:
    from inference_engine.api.state import app_state  # noqa: PLC0415

    backend = _HungOllama(mode)
    async with backend as endpoint:
        judge_at(endpoint)

        r, elapsed = await _post()

        assert r.status_code == 504, r.text
        assert r.json()["detail"]["type"] == "generation_timeout"
        assert r.json()["detail"]["model"] == JUDGE
        assert DEADLINE * 0.9 <= elapsed < DEADLINE + SLACK, elapsed
        # The engine hung up rather than leaving the request open upstream.
        await asyncio.wait_for(backend.hung_up.wait(), SLACK)
        assert app_state.scheduler.snapshot().in_flight_by_resource == {}

        # Nothing is left stuck on either side: the next call is served.
        r, elapsed = await _post()

        assert r.status_code == 200, r.text
        assert r.json()["verdict"]["parse_status"] == "clean"
        assert elapsed < DEADLINE, elapsed
        assert backend.accepted == 2
