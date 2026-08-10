"""Bounded upstream retries + the per-deployment cooldown / breaker (WS-D).

The behaviour under test is a *precedence*: a transient upstream failure must
cost the caller a retry on the model they asked for, and only a failure that
survives that retry may change which model answers. The adapter tests use
``httpx.MockTransport`` so the real HTTP path runs; the route tests drive the
ASGI app so the reservation and ledger interactions are in scope too.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import httpx
import pytest

from inference_engine.adapters import UpstreamGenerationError
from inference_engine.adapters._upstream import is_health_signal
from inference_engine.adapters._upstream_retry import (
    RetryPlan,
    UpstreamDeadline,
    deadline_scope,
    is_retryable,
    plan_retry,
    retry_after_seconds,
    retry_counters,
)
from inference_engine.adapters.base import GenerationParams, GenerationTimeoutError
from inference_engine.adapters.ollama_http import OllamaHttpAdapter
from inference_engine.adapters.openrouter_adapter import OpenRouterAdapter
from inference_engine.adapters.vllm_adapter import VLLMAdapter
from inference_engine.config import settings
from inference_engine.registry import ModelDescriptor
from inference_engine.registry.breaker import (
    CLOSED,
    HALF_OPEN,
    OPEN,
    UpstreamBreaker,
    breaker_span_attrs,
    descriptor_allows,
    descriptor_deployment_key,
    get_upstream_breaker,
)
from inference_engine.registry.vllm_probe import VLLMUpstreamProbe
from inference_engine.schemas import ChatMessage

_MESSAGES = [ChatMessage(role="user", content="hi")]


@pytest.fixture(autouse=True)
def _clean_resilience_state(monkeypatch):
    """Every test starts with a closed breaker, no retry tally, and no sleeps."""
    get_upstream_breaker().reset()
    retry_counters.reset()
    monkeypatch.setattr(settings, "upstream_retry_enabled", True)
    monkeypatch.setattr(settings, "upstream_retry_max_attempts", 2)
    monkeypatch.setattr(settings, "upstream_retry_base_delay_seconds", 0.001)
    monkeypatch.setattr(settings, "upstream_retry_max_delay_seconds", 0.002)
    monkeypatch.setattr(settings, "upstream_breaker_enabled", True)
    monkeypatch.setattr(settings, "upstream_breaker_failure_threshold", 3)
    monkeypatch.setattr(settings, "upstream_breaker_cooldown_seconds", 15.0)
    monkeypatch.setattr(settings, "upstream_breaker_max_cooldown_seconds", 120.0)
    yield
    get_upstream_breaker().reset()
    retry_counters.reset()


# ---------------------------------------------------------------------------
# Classification — what a retry is allowed to spend the deadline on
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("status", [408, 425, 429, 500, 502, 503, 504, 529])
def test_transient_status_codes_are_retryable(status: int) -> None:
    exc = httpx.HTTPStatusError(
        "boom",
        request=httpx.Request("POST", "http://x/v1/chat/completions"),
        response=httpx.Response(status),
    )
    assert is_retryable(exc) is True


@pytest.mark.parametrize("status", [400, 401, 403, 404, 409, 413, 422, 501])
def test_client_errors_are_not_retryable(status: int) -> None:
    """A 400 is the request being wrong; resending it just burns the deadline."""
    exc = httpx.HTTPStatusError(
        "boom",
        request=httpx.Request("POST", "http://x/v1/chat/completions"),
        response=httpx.Response(status),
    )
    assert is_retryable(exc) is False


def test_connect_failures_retry_but_read_timeouts_do_not() -> None:
    request = httpx.Request("POST", "http://x/v1/chat/completions")
    assert is_retryable(httpx.ConnectError("refused", request=request)) is True
    assert is_retryable(httpx.ConnectTimeout("no route", request=request)) is True
    assert is_retryable(httpx.RemoteProtocolError("reset", request=request)) is True
    # The upstream had the request when the clock ran out; a second attempt
    # re-pays the whole wait rather than recovering from a blip.
    assert is_retryable(httpx.ReadTimeout("slow", request=request)) is False


def test_a_deadline_expiry_is_never_a_health_signal() -> None:
    """The breaker's classifier, stated as the matrix its docstring claims.

    Timeouts split on evidence, not on type: a connect timeout proves the
    upstream never took the work, every other timeout only proves our own
    budget ran out — which a healthy deployment generating a long answer
    produces just as readily as a sick one.
    """
    request = httpx.Request("POST", "http://x/v1/chat/completions")

    def _status(code: int) -> httpx.HTTPStatusError:
        return httpx.HTTPStatusError(
            "boom", request=request, response=httpx.Response(code)
        )

    counts_as_health = {
        # The deadline this branch added to the streaming path, and the
        # per-operation timeouts underneath it.
        TimeoutError("upstream deadline exhausted"): False,
        httpx.ReadTimeout("slow", request=request): False,
        httpx.WriteTimeout("slow", request=request): False,
        httpx.PoolTimeout("saturated", request=request): False,
        # Evidence the upstream never accepted the call.
        httpx.ConnectTimeout("no route", request=request): True,
        httpx.ConnectError("refused", request=request): True,
        httpx.RemoteProtocolError("reset", request=request): True,
        # The upstream saying "later" or "I broke".
        _status(429): True,
        _status(503): True,
        # The request being wrong.
        _status(400): False,
        _status(404): False,
        _status(422): False,
    }
    for exc, expected in counts_as_health.items():
        assert is_health_signal(exc) is expected, type(exc).__name__
        # One predicate serves both questions; pin that so neither can drift
        # away from the other silently.
        assert is_retryable(exc) is expected, type(exc).__name__


def test_retry_after_is_parsed_from_seconds_and_dates() -> None:
    def _status(headers: dict) -> httpx.HTTPStatusError:
        return httpx.HTTPStatusError(
            "boom",
            request=httpx.Request("POST", "http://x/"),
            response=httpx.Response(429, headers=headers),
        )

    assert retry_after_seconds(_status({"retry-after": "7"})) == 7.0
    # A date in the past means "now", never a negative delay.
    past = _status({"retry-after": "Wed, 21 Oct 2015 07:28:00 GMT"})
    assert retry_after_seconds(past) == 0.0
    assert retry_after_seconds(_status({})) is None


def test_retry_after_is_honoured_above_the_backoff_ceiling(monkeypatch) -> None:
    monkeypatch.setattr(settings, "upstream_retry_max_delay_seconds", 0.5)
    exc = httpx.HTTPStatusError(
        "slow down",
        request=httpx.Request("POST", "http://x/"),
        response=httpx.Response(429, headers={"retry-after": "3"}),
    )
    plan = plan_retry(attempt=1, exc=exc, remaining_seconds=60.0)
    assert plan.retry is True
    assert plan.delay_seconds == pytest.approx(3.0)


def test_a_retry_that_does_not_fit_the_deadline_is_refused() -> None:
    exc = httpx.HTTPStatusError(
        "slow down",
        request=httpx.Request("POST", "http://x/"),
        response=httpx.Response(503, headers={"retry-after": "30"}),
    )
    plan = plan_retry(attempt=1, exc=exc, remaining_seconds=2.0)
    assert plan == RetryPlan(retry=False, reason="deadline_too_short")


def test_attempts_are_capped_by_configuration(monkeypatch) -> None:
    monkeypatch.setattr(settings, "upstream_retry_max_attempts", 2)
    exc = httpx.HTTPStatusError(
        "boom",
        request=httpx.Request("POST", "http://x/"),
        response=httpx.Response(503),
    )
    assert plan_retry(attempt=1, exc=exc, remaining_seconds=60).retry is True
    assert plan_retry(attempt=2, exc=exc, remaining_seconds=60) == RetryPlan(
        retry=False, reason="attempts_exhausted"
    )


async def test_a_disabled_timeout_leaves_every_upstream_await_unbounded() -> None:
    """``CHAT_COMPLETION_TIMEOUT_SECONDS=0`` must not smuggle in a cap."""
    deadline = UpstreamDeadline(None)
    assert deadline.remaining is None
    async with deadline_scope(deadline):
        await asyncio.sleep(0.05)


async def test_an_upstream_await_is_bounded_by_what_is_left_of_the_budget() -> None:
    deadline = UpstreamDeadline(0.05)
    started = time.monotonic()
    with pytest.raises(TimeoutError):
        async with deadline_scope(deadline):
            await asyncio.sleep(5)
    assert time.monotonic() - started < 1.0


async def test_a_spent_budget_refuses_to_open_another_upstream_operation() -> None:
    """The check that stops a stream handing on another chunk past the deadline."""
    deadline = UpstreamDeadline(0.0)
    with pytest.raises(TimeoutError):
        async with deadline_scope(deadline):
            await asyncio.sleep(0)


# ---------------------------------------------------------------------------
# Adapter behaviour — vLLM (and, by inheritance, OpenRouter)
# ---------------------------------------------------------------------------


def _vllm_descriptor(
    endpoint: str = "http://vllm:8000",
    model_id: str = "test-model",
) -> ModelDescriptor:
    return ModelDescriptor(
        name="test",
        tag="vllm",
        namespace="vllm",
        registry="local",
        model_path=Path(f"vllm://{endpoint}/{model_id}"),
        format="vllm",
        params={"model_id": model_id},
        size_bytes=0,
        endpoint=endpoint,
    )


def _ollama_descriptor(
    endpoint: str = "http://ollama:11434",
    model_id: str = "llama3.2:1b",
) -> ModelDescriptor:
    return ModelDescriptor(
        name="llama3.2",
        tag="1b",
        namespace="library",
        registry="registry.ollama.ai",
        model_path=Path(f"ollama_http://{endpoint}/{model_id}"),
        format="ollama_http",
        params={"model_id": model_id},
        size_bytes=1,
        endpoint=endpoint,
    )


def _install_transport(adapter, handler) -> None:
    assert adapter._client is not None  # noqa: SLF001 — test scaffolding
    adapter._client = httpx.AsyncClient(  # noqa: SLF001
        base_url=adapter._client.base_url,
        transport=httpx.MockTransport(handler),
        timeout=30.0,
    )


def _chat_body(content: str = "hi") -> dict:
    return {
        "id": "chatcmpl-x",
        "object": "chat.completion",
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
    }


async def _loaded_vllm(handler) -> VLLMAdapter:
    adapter = VLLMAdapter()
    await adapter.load(_vllm_descriptor())
    _install_transport(adapter, handler)
    return adapter


def test_a_transient_503_is_retried_on_the_same_deployment() -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(503, json={"error": "overloaded"})
        return httpx.Response(200, json=_chat_body("recovered"))

    async def run():
        adapter = await _loaded_vllm(handler)
        result = await adapter.generate(_MESSAGES, GenerationParams(max_tokens=4))
        await adapter.unload()
        return result

    result = asyncio.run(run())
    assert result.text == "recovered"
    assert len(calls) == 2
    assert retry_counters.snapshot() == {"vllm": 1}


def test_a_400_is_not_retried() -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(400, json={"error": "bad tool schema"})

    async def run():
        adapter = await _loaded_vllm(handler)
        with pytest.raises(UpstreamGenerationError) as excinfo:
            await adapter.generate(_MESSAGES, GenerationParams(max_tokens=4))
        await adapter.unload()
        return excinfo.value

    exc = asyncio.run(run())
    assert exc.upstream_status_code == 400
    assert len(calls) == 1
    assert retry_counters.snapshot() == {}


def test_retries_stop_at_the_configured_attempt_cap(monkeypatch) -> None:
    monkeypatch.setattr(settings, "upstream_retry_max_attempts", 2)
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(503, json={"error": "still down"})

    async def run():
        adapter = await _loaded_vllm(handler)
        with pytest.raises(UpstreamGenerationError):
            await adapter.generate(_MESSAGES, GenerationParams(max_tokens=4))
        await adapter.unload()

    asyncio.run(run())
    assert len(calls) == 2


def test_retries_are_disabled_by_configuration(monkeypatch) -> None:
    monkeypatch.setattr(settings, "upstream_retry_enabled", False)
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(503)

    async def run():
        adapter = await _loaded_vllm(handler)
        with pytest.raises(UpstreamGenerationError):
            await adapter.generate(_MESSAGES, GenerationParams(max_tokens=4))
        await adapter.unload()

    asyncio.run(run())
    assert len(calls) == 1


def test_a_retry_never_outlives_the_chat_completion_deadline(monkeypatch) -> None:
    """The retry budget is the request budget, not a new one."""
    monkeypatch.setattr(settings, "chat_completion_timeout_seconds", 0.2)
    monkeypatch.setattr(settings, "upstream_retry_base_delay_seconds", 5.0)
    monkeypatch.setattr(settings, "upstream_retry_max_delay_seconds", 5.0)
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(503)

    async def run():
        adapter = await _loaded_vllm(handler)
        with pytest.raises(UpstreamGenerationError):
            await adapter.generate(_MESSAGES, GenerationParams(max_tokens=4))
        await adapter.unload()

    asyncio.run(run())
    # The 5 s backoff cannot fit a 0.2 s budget, so the 503 surfaces instead of
    # a timeout produced by our own waiting.
    assert len(calls) == 1


async def test_cancelling_inside_the_backoff_gives_the_half_open_trial_back(
    monkeypatch,
) -> None:
    """A cancelled request must not wedge a healthy deployment OPEN forever.

    ``begin_attempt`` holds the single half-open trial until the caller reports
    an outcome. A cancellation that landed in the retry *sleep* used to unwind
    without reporting one, so the trial stayed held and every later
    ``begin_attempt`` answered OPEN for that key — the deployment silently gone
    from ``/v1/models`` and from candidate selection until a process restart.
    """
    monkeypatch.setattr(settings, "upstream_breaker_failure_threshold", 1)
    monkeypatch.setattr(settings, "upstream_breaker_cooldown_seconds", 0.0)
    monkeypatch.setattr(settings, "upstream_retry_base_delay_seconds", 30.0)
    monkeypatch.setattr(settings, "upstream_retry_max_delay_seconds", 30.0)
    monkeypatch.setattr(settings, "chat_completion_timeout_seconds", 0.0)

    breaker = get_upstream_breaker()
    adapter = await _loaded_vllm(lambda request: httpx.Response(503))
    key = adapter.upstream_deployment_key()
    # Open it, with a zero cooldown so the next call is admitted as the trial.
    breaker.record_failure(key, reason="upstream_http_503")
    assert breaker.state(key) == HALF_OPEN

    task = asyncio.create_task(adapter.generate(_MESSAGES, GenerationParams(max_tokens=4)))
    # Let the first attempt fail and the 30 s backoff start, then walk away.
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await adapter.unload()

    assert breaker._states[key].half_open_in_flight is False  # noqa: SLF001
    # The trial is free for the next caller, and the cooldown still stands:
    # nothing was proven about the deployment either way.
    assert breaker.begin_attempt(key) == HALF_OPEN


def _sse(lines: list[str]) -> bytes:
    return "".join(f"data: {line}\n\n" for line in lines).encode()


def _sse_chunk(text: str, finish: str | None = None) -> str:
    return json.dumps(
        {
            "choices": [
                {"index": 0, "delta": {"content": text}, "finish_reason": finish}
            ]
        }
    )


def test_a_stream_is_retried_when_it_fails_before_the_first_chunk() -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(503, json={"error": "no capacity"})
        return httpx.Response(
            200,
            content=_sse([_sse_chunk("re"), _sse_chunk("try", "stop"), "[DONE]"]),
            headers={"content-type": "text/event-stream"},
        )

    async def run():
        adapter = await _loaded_vllm(handler)
        chunks = [
            piece.text
            async for piece in adapter.stream(_MESSAGES, GenerationParams(max_tokens=4))
        ]
        await adapter.unload()
        return chunks

    assert "".join(asyncio.run(run())) == "retry"
    assert len(calls) == 2


def test_a_stream_is_never_retried_after_a_chunk_has_been_delivered() -> None:
    """A partial stream cannot be restarted and concatenated — it terminates."""
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)

        async def _dies_after_one_chunk():
            yield _sse([_sse_chunk("par")])
            raise httpx.RemoteProtocolError("peer closed connection", request=request)

        return httpx.Response(
            200,
            content=_dies_after_one_chunk(),
            headers={"content-type": "text/event-stream"},
        )

    async def run():
        adapter = await _loaded_vllm(handler)
        seen: list[str] = []
        with pytest.raises(UpstreamGenerationError) as excinfo:
            async for piece in adapter.stream(_MESSAGES, GenerationParams(max_tokens=4)):
                seen.append(piece.text)
        await adapter.unload()
        return seen, excinfo.value

    seen, exc = asyncio.run(run())
    assert seen == ["par"]
    assert exc.error_type == "upstream_request_error"
    # One attempt only: the consumer already saw "par", and a RemoteProtocolError
    # is otherwise squarely retryable — the emitted chunk is what stops it.
    assert len(calls) == 1
    assert retry_counters.snapshot() == {}


def _dripping_sse(request: httpx.Request, *, chunks: int, gap: float) -> httpx.Response:
    """An SSE response that keeps arriving — the shape httpx's read timeout misses.

    httpx's read timeout is per read operation, so a stream that produces a
    chunk before every deadline resets it forever. Only a TOTAL-elapsed budget
    stops this one.
    """

    async def _body():
        for _ in range(chunks):
            await asyncio.sleep(gap)
            yield _sse([_sse_chunk("x")])

    return httpx.Response(
        200, content=_body(), headers={"content-type": "text/event-stream"}
    )


async def test_a_stream_never_outlives_the_chat_completion_deadline(monkeypatch) -> None:
    """The budget bounds the whole stream, not just its first byte.

    The scheduler lease is held for the duration of a stream, so an unbounded
    one takes the deployment's dispatch slot with it — the pathology #91 fixed
    for the blocking path and left standing here.
    """
    monkeypatch.setattr(settings, "chat_completion_timeout_seconds", 0.3)
    adapter = await _loaded_vllm(lambda r: _dripping_sse(r, chunks=200, gap=0.05))

    seen: list[str] = []
    started = time.monotonic()
    with pytest.raises(GenerationTimeoutError):
        async for piece in adapter.stream(_MESSAGES, GenerationParams(max_tokens=64)):
            seen.append(piece.text)
    elapsed = time.monotonic() - started
    await adapter.unload()

    # The upstream would have dripped for 10 s. The budget was 0.3 s, and the
    # bound is tight rather than merely present: five local runs landed at
    # 301-303 ms, so a ceiling of 0.8 s is slack for a loaded CI box, not room
    # for the bound to have gone soft.
    assert elapsed < 0.8
    # …and it was a deadline, not a failure to connect: chunks did arrive first.
    assert len(seen) >= 1
    assert len(seen) < 200


async def test_the_ollama_stream_is_bounded_by_the_same_budget(monkeypatch) -> None:
    monkeypatch.setattr(settings, "chat_completion_timeout_seconds", 0.3)
    adapter = OllamaHttpAdapter()
    await adapter.load(_ollama_descriptor())
    _install_transport(adapter, lambda r: _dripping_sse(r, chunks=200, gap=0.05))

    started = time.monotonic()
    with pytest.raises(GenerationTimeoutError):
        async for _ in adapter.stream(_MESSAGES, GenerationParams(max_tokens=64)):
            pass
    elapsed = time.monotonic() - started
    await adapter.unload()
    assert elapsed < 0.8


async def test_a_bounded_stream_still_delivers_a_short_one_intact(monkeypatch) -> None:
    """The bound must not truncate a stream that fits inside the budget."""
    monkeypatch.setattr(settings, "chat_completion_timeout_seconds", 5.0)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=_sse([_sse_chunk("in "), _sse_chunk("budget", "stop"), "[DONE]"]),
            headers={"content-type": "text/event-stream"},
        )

    adapter = await _loaded_vllm(handler)
    text = "".join(
        [p.text async for p in adapter.stream(_MESSAGES, GenerationParams(max_tokens=8))]
    )
    await adapter.unload()
    assert text == "in budget"


def test_a_stream_rejected_with_a_400_surfaces_typed_without_retrying() -> None:
    """The stream error path must survive a response body it never read."""
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(400, json={"error": "bad tool schema"})

    async def run():
        adapter = await _loaded_vllm(handler)
        with pytest.raises(UpstreamGenerationError) as excinfo:
            async for _ in adapter.stream(_MESSAGES, GenerationParams(max_tokens=4)):
                pass
        await adapter.unload()
        return excinfo.value

    exc = asyncio.run(run())
    assert exc.error_type == "upstream_http_error"
    assert exc.upstream_status_code == 400
    assert len(calls) == 1


def test_ollama_http_retries_a_transient_upstream_failure() -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(502, json={"error": "gateway"})
        return httpx.Response(200, json=_chat_body("ollama"))

    async def run():
        adapter = OllamaHttpAdapter()
        await adapter.load(_ollama_descriptor())
        _install_transport(adapter, handler)
        result = await adapter.generate(_MESSAGES, GenerationParams(max_tokens=4))
        await adapter.unload()
        return result

    assert asyncio.run(run()).text == "ollama"
    assert len(calls) == 2
    assert retry_counters.snapshot() == {"ollama_http": 1}


def test_openrouter_inherits_the_same_retry_path(monkeypatch) -> None:
    monkeypatch.setattr(settings, "openrouter_api_key", "sk-test")
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(429, headers={"retry-after": "0"})
        return httpx.Response(200, json=_chat_body("openrouter"))

    async def run():
        adapter = OpenRouterAdapter()
        descriptor = _vllm_descriptor(endpoint="https://openrouter.ai/api")
        adapter_descriptor = ModelDescriptor(
            **{**vars(descriptor), "format": "openrouter"}
        )
        await adapter.load(adapter_descriptor)
        _install_transport(adapter, handler)
        result = await adapter.generate(_MESSAGES, GenerationParams(max_tokens=4))
        await adapter.unload()
        return result

    assert asyncio.run(run()).text == "openrouter"
    assert len(calls) == 2
    assert retry_counters.snapshot() == {"openrouter": 1}


def test_the_deployment_key_separates_models_on_one_host() -> None:
    """One sick model must not take its co-tenants on the same host down."""
    first = descriptor_deployment_key(_ollama_descriptor(model_id="gemma4:31b"))
    second = descriptor_deployment_key(_ollama_descriptor(model_id="llama3.2:1b"))
    assert first != second
    assert descriptor_deployment_key(
        ModelDescriptor(
            name="local",
            tag="gguf",
            namespace="library",
            registry="local",
            model_path=Path("/tmp/local.gguf"),
            format="gguf",
            size_bytes=1,
        )
    ) is None


# ---------------------------------------------------------------------------
# Breaker — opening, cooldown, half-open, and candidate selection
# ---------------------------------------------------------------------------


def test_consecutive_transient_failures_open_the_deployment(monkeypatch) -> None:
    monkeypatch.setattr(settings, "upstream_breaker_failure_threshold", 2)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    async def run():
        adapter = OllamaHttpAdapter()
        await adapter.load(_ollama_descriptor())
        _install_transport(adapter, handler)
        for _ in range(2):
            with pytest.raises(UpstreamGenerationError):
                await adapter.generate(_MESSAGES, GenerationParams(max_tokens=4))
        return adapter

    adapter = asyncio.run(run())
    key = adapter.upstream_deployment_key()
    assert get_upstream_breaker().state(key) == OPEN
    # …and candidate selection now skips the descriptor entirely.
    assert descriptor_allows(_ollama_descriptor()) is False


def test_client_errors_never_open_a_deployment(monkeypatch) -> None:
    """One malformed caller must not take a healthy upstream from everyone."""
    monkeypatch.setattr(settings, "upstream_breaker_failure_threshold", 2)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": "bad request"})

    async def run():
        adapter = OllamaHttpAdapter()
        await adapter.load(_ollama_descriptor())
        _install_transport(adapter, handler)
        for _ in range(5):
            with pytest.raises(UpstreamGenerationError):
                await adapter.generate(_MESSAGES, GenerationParams(max_tokens=4))
        return adapter

    adapter = asyncio.run(run())
    assert get_upstream_breaker().state(adapter.upstream_deployment_key()) == CLOSED
    assert descriptor_allows(_ollama_descriptor()) is True


def test_an_open_deployment_refuses_the_call_without_touching_the_upstream(
    monkeypatch,
) -> None:
    monkeypatch.setattr(settings, "upstream_breaker_failure_threshold", 1)
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(503)

    async def run():
        adapter = OllamaHttpAdapter()
        await adapter.load(_ollama_descriptor())
        _install_transport(adapter, handler)
        with pytest.raises(UpstreamGenerationError):
            await adapter.generate(_MESSAGES, GenerationParams(max_tokens=4))
        calls_before = len(calls)
        with pytest.raises(UpstreamGenerationError) as excinfo:
            await adapter.generate(_MESSAGES, GenerationParams(max_tokens=4))
        return excinfo.value, calls_before

    exc, calls_before = asyncio.run(run())
    assert exc.error_type == "upstream_cooldown"
    # The first call spent its attempt and its one retry; the second never
    # reached the transport at all.
    assert calls_before == 2
    assert len(calls) == 2


def test_a_half_open_probe_admits_exactly_one_caller_and_closes_on_success() -> None:
    breaker = UpstreamBreaker(failure_threshold=1, cooldown_seconds=0.0)
    key = "vllm|http://x:8000|m"

    breaker.record_failure(key, reason="upstream_http_503")
    # Cooldown of 0 s means the very next attempt is the half-open trial.
    assert breaker.begin_attempt(key) == HALF_OPEN
    # …and it is the only one: a concurrent caller is still turned away.
    assert breaker.begin_attempt(key) == OPEN

    breaker.record_success(key)
    assert breaker.state(key) == CLOSED
    assert breaker.begin_attempt(key) == CLOSED


def test_a_failed_half_open_probe_reopens_with_a_longer_cooldown() -> None:
    breaker = UpstreamBreaker(
        failure_threshold=1, cooldown_seconds=10.0, max_cooldown_seconds=1000.0
    )
    key = "vllm|http://x:8000|m"

    breaker.record_failure(key, reason="upstream_http_503")
    first = breaker.cooldown_remaining(key)
    assert first == pytest.approx(10.0, abs=0.5)

    # Force the cooldown to have elapsed, then fail the trial.
    breaker._states[key].open_until = 0.0  # noqa: SLF001 — no clock injection seam
    assert breaker.begin_attempt(key) == HALF_OPEN
    breaker.record_failure(key, reason="upstream_http_503")
    assert breaker.cooldown_remaining(key) == pytest.approx(20.0, abs=0.5)


def test_a_failure_that_says_nothing_gives_the_half_open_trial_back() -> None:
    breaker = UpstreamBreaker(failure_threshold=1, cooldown_seconds=0.0)
    key = "vllm|http://x:8000|m"
    breaker.record_failure(key, reason="upstream_http_503")

    assert breaker.begin_attempt(key) == HALF_OPEN
    breaker.abandon_attempt(key)
    # The cooldown still stands — nothing was proven — but the trial is free.
    assert breaker.begin_attempt(key) == HALF_OPEN


def test_the_disabled_breaker_never_opens(monkeypatch) -> None:
    monkeypatch.setattr(settings, "upstream_breaker_enabled", False)
    breaker = get_upstream_breaker()
    key = "vllm|http://x:8000|m"
    for _ in range(10):
        breaker.record_failure(key, reason="upstream_http_503")
    assert breaker.state(key) == CLOSED
    assert breaker.begin_attempt(key) == CLOSED


async def test_slow_but_successful_streams_never_open_the_deployment(monkeypatch) -> None:
    """A long generation is not an unhealthy deployment.

    The regression this pins: bounding a stream by the request deadline made
    ``TimeoutError`` the ordinary end of a stream that simply ran long, and
    while the classifier counted every timeout as a health signal, three such
    streams on a *healthy* upstream opened the breaker and withdrew the model
    from every tenant. The upstream here answers immediately and keeps
    answering — nothing about it is sick except that it is slower than the
    caller's budget.
    """
    monkeypatch.setattr(settings, "upstream_breaker_failure_threshold", 3)
    monkeypatch.setattr(settings, "chat_completion_timeout_seconds", 0.2)

    adapter = OllamaHttpAdapter()
    await adapter.load(_ollama_descriptor())
    _install_transport(adapter, lambda r: _dripping_sse(r, chunks=200, gap=0.02))
    key = adapter.upstream_deployment_key()

    delivered = 0
    for _ in range(3):
        with pytest.raises(GenerationTimeoutError):
            async for piece in adapter.stream(_MESSAGES, GenerationParams(max_tokens=64)):
                delivered += len(piece.text)
    await adapter.unload()

    # Every one of the three did reach the deadline, and did so with tokens in
    # hand — this is the slow-generation shape, not a dead upstream.
    assert delivered >= 3
    assert get_upstream_breaker().state(key) == CLOSED
    assert descriptor_allows(_ollama_descriptor()) is True


async def test_an_unreachable_upstream_still_opens_the_deployment(monkeypatch) -> None:
    """The other half of the rule: a failure to connect is still a verdict.

    Sits next to the test above so the pair reads as one decision — a timeout
    measured on a call the upstream accepted is discounted, a failure that
    proves it never accepted one is not.
    """
    monkeypatch.setattr(settings, "upstream_breaker_failure_threshold", 3)
    monkeypatch.setattr(settings, "chat_completion_timeout_seconds", 30.0)

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    adapter = OllamaHttpAdapter()
    await adapter.load(_ollama_descriptor())
    _install_transport(adapter, handler)
    key = adapter.upstream_deployment_key()

    for _ in range(3):
        with pytest.raises(UpstreamGenerationError):
            async for _ in adapter.stream(_MESSAGES, GenerationParams(max_tokens=8)):
                pass
    await adapter.unload()

    assert get_upstream_breaker().state(key) == OPEN
    assert descriptor_allows(_ollama_descriptor()) is False


def test_the_vllm_probe_reports_cooldown_without_calling_the_upstream() -> None:
    breaker = get_upstream_breaker()
    descriptor = _vllm_descriptor()
    breaker.record_failure(
        descriptor_deployment_key(descriptor), reason="upstream_http_503"
    )
    # Threshold is 3 by default, so one failure is not enough yet.
    assert get_upstream_breaker().state(descriptor_deployment_key(descriptor)) == CLOSED
    for _ in range(2):
        breaker.record_failure(
            descriptor_deployment_key(descriptor), reason="upstream_http_503"
        )

    calls: list[str] = []

    def factory(base_url: str, timeout) -> httpx.Client:
        calls.append(base_url)
        return httpx.Client(
            base_url=base_url,
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, json={"data": [{"id": "test-model"}]})
            ),
        )

    probe = VLLMUpstreamProbe(ttl_seconds=0.0, client_factory=factory)
    result = probe.probe(descriptor)
    assert result.loadable is False
    assert result.reason == "upstream_cooldown"
    assert calls == []


def test_the_vllm_probe_is_the_half_open_trial_and_closes_the_breaker() -> None:
    breaker = get_upstream_breaker()
    descriptor = _vllm_descriptor()
    key = descriptor_deployment_key(descriptor)
    for _ in range(3):
        breaker.record_failure(key, reason="upstream_http_503")
    assert breaker.state(key) == OPEN
    breaker._states[key].open_until = 0.0  # noqa: SLF001 — no clock injection seam

    def factory(base_url: str, timeout) -> httpx.Client:
        return httpx.Client(
            base_url=base_url,
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, json={"data": [{"id": "test-model"}]})
            ),
        )

    probe = VLLMUpstreamProbe(ttl_seconds=300.0, client_factory=factory)
    result = probe.probe(descriptor)
    assert result.loadable is True
    assert breaker.state(key) == CLOSED


def test_an_unreachable_vllm_probe_feeds_the_breaker(monkeypatch) -> None:
    monkeypatch.setattr(settings, "upstream_breaker_failure_threshold", 2)
    descriptor = _vllm_descriptor()

    def factory(base_url: str, timeout) -> httpx.Client:
        def _boom(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused", request=request)

        return httpx.Client(base_url=base_url, transport=httpx.MockTransport(_boom))

    probe = VLLMUpstreamProbe(ttl_seconds=0.0, client_factory=factory)
    assert probe.probe(descriptor).reason == "upstream_unreachable"
    assert probe.probe(descriptor).reason == "upstream_unreachable"
    assert probe.probe(descriptor).reason == "upstream_cooldown"


def test_a_silent_vllm_upstream_is_opened_by_the_probe_not_the_hot_path(
    monkeypatch,
) -> None:
    """Where a timeout IS allowed to be a health verdict.

    The hot path discounts timeouts because a generation deadline cannot tell
    a wedged deployment from a long answer. This probe can: it is a bare GET
    ``/v1/models`` under its own one-second bound, so a timeout on it is not
    "the answer was long", it is "a cheap call went unanswered". That is the
    one place a timeout still opens a deployment — and it exists for vLLM
    only, which is why a wedged ``ollama_http`` deployment stays selectable.
    """
    monkeypatch.setattr(settings, "upstream_breaker_failure_threshold", 2)
    descriptor = _vllm_descriptor()
    key = descriptor_deployment_key(descriptor)

    def factory(base_url: str, timeout) -> httpx.Client:
        def _silent(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("upstream never answered", request=request)

        return httpx.Client(base_url=base_url, transport=httpx.MockTransport(_silent))

    probe = VLLMUpstreamProbe(ttl_seconds=0.0, client_factory=factory)
    assert probe.probe(descriptor).reason == "upstream_timeout"
    assert get_upstream_breaker().state(key) == CLOSED
    assert probe.probe(descriptor).reason == "upstream_timeout"
    assert get_upstream_breaker().state(key) == OPEN

async def test_a_silent_ollama_upstream_stays_selectable(monkeypatch) -> None:
    """The uncovered case, pinned so it is a known limit and not a surprise.

    ``ollama_http`` has no reachability probe of its own, so the hot path is
    the only reporter it has — and the hot path discounts timeouts. A wedged
    Ollama deployment therefore keeps being handed requests, each of which
    spends its own full budget. Change that by giving the format a bounded
    liveness probe, not by counting generation timeouts again.
    """
    monkeypatch.setattr(settings, "upstream_breaker_failure_threshold", 2)
    monkeypatch.setattr(settings, "chat_completion_timeout_seconds", 0.2)

    def _silent(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("upstream never answered", request=request)

    adapter = OllamaHttpAdapter()
    await adapter.load(_ollama_descriptor())
    _install_transport(adapter, _silent)
    key = adapter.upstream_deployment_key()
    for _ in range(4):
        with pytest.raises(GenerationTimeoutError):
            await adapter.generate(_MESSAGES, GenerationParams(max_tokens=4))
    await adapter.unload()

    assert get_upstream_breaker().state(key) == CLOSED
    assert descriptor_allows(_ollama_descriptor()) is True


def test_a_missing_model_does_not_open_the_breaker() -> None:
    """A 200 catalog that omits the model is the config being wrong, not health."""
    descriptor = _vllm_descriptor()

    def factory(base_url: str, timeout) -> httpx.Client:
        return httpx.Client(
            base_url=base_url,
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, json={"data": [{"id": "other"}]})
            ),
        )

    probe = VLLMUpstreamProbe(ttl_seconds=0.0, client_factory=factory)
    for _ in range(5):
        assert probe.probe(descriptor).reason == "upstream_model_missing"
    assert get_upstream_breaker().state(descriptor_deployment_key(descriptor)) == CLOSED


def test_breaker_span_attributes_name_the_deployment_and_its_state() -> None:
    async def run():
        adapter = VLLMAdapter()
        await adapter.load(_vllm_descriptor())
        attrs = breaker_span_attrs(adapter)
        await adapter.unload()
        return attrs

    attrs = asyncio.run(run())
    assert attrs["llm.upstream.deployment"] == "vllm|http://vllm:8000|test-model"
    assert attrs["llm.upstream.breaker.state"] == CLOSED


def test_in_process_backends_carry_no_breaker_attributes() -> None:
    class _Local:
        backend_name = "llama_cpp"

    assert breaker_span_attrs(_Local()) == {}


def test_metrics_render_the_breaker_state_gauge() -> None:
    from inference_engine.api.metrics import metrics

    breaker = get_upstream_breaker()
    for _ in range(3):
        breaker.record_failure(
            "vllm|http://vllm:8000|test-model",
            reason="upstream_http_503",
            backend="vllm",
            endpoint="http://vllm:8000",
            model_id="test-model",
        )

    body = asyncio.run(metrics())
    assert (
        'inference_engine_upstream_breaker_state{backend="vllm",'
        'endpoint="http://vllm:8000",model="test-model"} 2' in body
    )
    assert "inference_engine_upstream_breaker_opened_total 1" in body
    assert "inference_engine_upstream_retries_total" in body
