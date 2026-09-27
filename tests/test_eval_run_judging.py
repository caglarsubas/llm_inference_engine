"""/v1/evals/run as a scheduled workload — repeats, scheduler slots, typed errors,
and the safety rubric seeing the prompt it judges against."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterable
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from inference_engine.adapters import GenerationParams, InferenceAdapter, StreamChunk
from inference_engine.adapters.base import (
    ContextLengthExceededError,
    GenerationResult,
    UpstreamGenerationError,
)
from inference_engine.api import _scheduling
from inference_engine.cancellation import Cancellation
from inference_engine.config import settings
from inference_engine.evals import EvalRunner, RubricRegistry, TenantRubricStore, Verdict
from inference_engine.evals.runner import summarize_repeats
from inference_engine.main import app
from inference_engine.manager import ModelManager
from inference_engine.registry import ModelDescriptor
from inference_engine.scheduler import TenantScheduler

_RESOURCE = "scripted-judge:judge:1"


class _ScriptedJudge(InferenceAdapter):
    """Returns ``texts`` in turn; records every call's params and the scheduler
    state it ran under."""

    backend_name = "scripted-judge"

    def __init__(self) -> None:
        self.texts: list[str] = ['{"score": 4, "justification": "ok"}']
        self.params: list[GenerationParams] = []
        self.in_flight: list[dict] = []
        self.last_messages: list = []
        self.raise_on_generate: Exception | None = None
        self.sleep_seconds = 0.0
        self.generation_is_cancellable = False
        self._descriptor: ModelDescriptor | None = None

    @property
    def is_loaded(self) -> bool:
        return self._descriptor is not None

    @property
    def loaded_model(self) -> ModelDescriptor | None:
        return self._descriptor

    async def load(self, descriptor: ModelDescriptor) -> None:
        self._descriptor = descriptor

    async def unload(self) -> None:
        self._descriptor = None

    async def generate(
        self, messages: Iterable, params: GenerationParams, cancel: Cancellation | None = None
    ) -> GenerationResult:
        from inference_engine.api.state import app_state  # noqa: PLC0415

        self.last_messages = list(messages)
        self.in_flight.append(dict(app_state.scheduler.snapshot().in_flight_by_resource))
        if self.sleep_seconds:
            await asyncio.sleep(self.sleep_seconds)
        if self.raise_on_generate is not None:
            raise self.raise_on_generate
        text = self.texts[len(self.params) % len(self.texts)]
        self.params.append(params)
        return GenerationResult(
            text=text, finish_reason="stop", prompt_tokens=12, completion_tokens=8
        )

    async def stream(
        self, messages: Iterable, params: GenerationParams, cancel: Cancellation | None = None
    ) -> AsyncIterator[StreamChunk]:
        yield StreamChunk(text="", finish_reason="stop")


@pytest.fixture
def judge(monkeypatch) -> _ScriptedJudge:
    desc = ModelDescriptor(
        name="judge", tag="1", namespace="ns", registry="reg",
        model_path=Path("/tmp/judge"), format="gguf", size_bytes=1,
    )

    class _Reg:
        def get(self, name: str) -> ModelDescriptor | None:
            return desc if name == "judge:1" else None

        def list_models(self) -> list[ModelDescriptor]:
            return [desc]

    adapter = _ScriptedJudge()
    mgr = ModelManager(_Reg(), adapter_factory=lambda d: adapter, memory_budget_bytes=100)
    from inference_engine.api.state import app_state  # noqa: PLC0415

    monkeypatch.setattr(settings, "scheduler_enabled", True)
    monkeypatch.setattr(app_state, "eval_runner", EvalRunner(mgr))
    monkeypatch.setattr(app_state, "rubric_registry", RubricRegistry.with_builtins())
    monkeypatch.setattr(app_state, "tenant_rubrics", TenantRubricStore())
    monkeypatch.setattr(app_state, "scheduler", TenantScheduler())
    return adapter


def _body(**overrides) -> dict:
    body = {
        "rubric": "helpfulness",
        "prompt": "How do I reset my password?",
        "response": "Use the 'Forgot password' link.",
        "judge_model": "judge:1",
    }
    body.update(overrides)
    return body


# ---------------------------------------------------------------------------
# Safety rubric
# ---------------------------------------------------------------------------


def test_safety_judge_sees_the_prompt(judge) -> None:
    judge.texts = ['{"safe": true, "concerns": []}']
    r = TestClient(app).post(
        "/v1/evals/run",
        json=_body(rubric="safety", prompt="How do I pick a lock?", response="I can't help."),
    )
    assert r.status_code == 200, r.text
    user = next(m for m in judge.last_messages if m.role == "user").content
    assert "USER PROMPT:\nHow do I pick a lock?" in user
    assert "ASSISTANT RESPONSE:\nI can't help." in user


# ---------------------------------------------------------------------------
# Repeats
# ---------------------------------------------------------------------------


def test_single_run_keeps_greedy_judging(judge) -> None:
    r = TestClient(app).post("/v1/evals/run", json=_body(seed=7))
    assert r.status_code == 200, r.text
    body = r.json()
    assert [p.temperature for p in judge.params] == [0.0]
    assert [p.seed for p in judge.params] == [7]
    assert body["temperature"] == 0.0
    assert len(body["verdicts"]) == 1
    assert body["verdicts"][0] == body["verdict"]
    assert body["repeats"] == {
        "n": 1, "parsed": 1, "mean": 4.0, "stdev": 0.0, "min": 4.0, "max": 4.0,
        "agreement": 1.0,
    }


def test_repeats_vary_the_seed_and_report_agreement(judge, _session_exporter) -> None:
    _session_exporter.clear()
    judge.texts = [
        '{"score": 4, "justification": "a"}',
        '{"score": 4, "justification": "b"}',
        '{"score": 2, "justification": "c"}',
    ]
    r = TestClient(app).post("/v1/evals/run", json=_body(seed=7, n=3, temperature=0.7))
    assert r.status_code == 200, r.text
    body = r.json()

    assert [p.seed for p in judge.params] == [7, 8, 9]
    assert {p.temperature for p in judge.params} == {0.7}
    assert [v["score"] for v in body["verdicts"]] == [4.0, 4.0, 2.0]
    assert body["verdict"] == body["verdicts"][0]
    repeats = body["repeats"]
    assert repeats["n"] == repeats["parsed"] == 3
    assert repeats["mean"] == pytest.approx(10 / 3)
    assert (repeats["min"], repeats["max"]) == (2.0, 4.0)
    assert repeats["agreement"] == pytest.approx(2 / 3)
    assert repeats["stdev"] == pytest.approx(0.9428, abs=1e-4)

    spans = [s for s in _session_exporter.get_finished_spans() if s.name == "eval.run"]
    assert sorted(s.attributes["eval.repeat.index"] for s in spans) == [0, 1, 2]
    assert {s.attributes["eval.repeat.n"] for s in spans} == {3}
    assert {s.attributes["gen_ai.request.temperature"] for s in spans} == {0.7}


def test_repeats_without_a_seed_leave_it_unset(judge) -> None:
    r = TestClient(app).post("/v1/evals/run", json=_body(seed=None, n=2, temperature=0.5))
    assert r.status_code == 200, r.text
    assert [p.seed for p in judge.params] == [None, None]


def test_failed_verdicts_stay_out_of_the_statistics() -> None:
    verdicts = [
        Verdict(score=1.0, raw="", parse_status="clean"),
        Verdict(score=0.0, raw="", parse_status="failed"),
        Verdict(score=1.0, raw="", parse_status="repaired"),
    ]
    summary = summarize_repeats(verdicts)
    assert (summary.n, summary.parsed, summary.mean, summary.agreement) == (3, 2, 1.0, 1.0)

    none_parsed = summarize_repeats([Verdict(score=0.0, raw="", parse_status="failed")])
    assert (none_parsed.parsed, none_parsed.mean, none_parsed.agreement) == (0, None, None)


@pytest.mark.parametrize(
    "overrides",
    [
        {"n": 3},  # repeats at temperature 0 would all agree by construction
        {"n": 9, "temperature": 0.7},
        {"n": 0},
        {"temperature": 2.5},
    ],
)
def test_repeat_knobs_are_bounded(judge, overrides) -> None:
    r = TestClient(app).post("/v1/evals/run", json=_body(**overrides))
    assert r.status_code == 422, r.text
    assert judge.params == []


# ---------------------------------------------------------------------------
# Scheduler slot
# ---------------------------------------------------------------------------


async def _post(body: dict) -> httpx.Response:
    # ASGITransport rather than TestClient, so the request shares this test's
    # event loop with the scheduler lease the test holds.
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://engine"
    ) as client:
        return await client.post("/v1/evals/run", json=body)


@pytest.mark.asyncio
async def test_each_judge_call_holds_a_slot_and_releases_it(judge) -> None:
    from inference_engine.api.state import app_state  # noqa: PLC0415

    r = await _post(_body(n=2, temperature=0.3))
    assert r.status_code == 200, r.text
    assert judge.in_flight == [{_RESOURCE: 1}, {_RESOURCE: 1}]
    assert app_state.scheduler.snapshot().in_flight_by_resource == {}
    assert r.headers[_scheduling.RESOURCE_HEADER] == _RESOURCE
    assert int(r.headers[_scheduling.QUEUE_WAIT_MS_HEADER]) >= 0


@pytest.mark.asyncio
async def test_eval_span_carries_its_admission(judge, _session_exporter) -> None:
    _session_exporter.clear()
    r = await _post(_body())
    assert r.status_code == 200, r.text
    (s,) = [s for s in _session_exporter.get_finished_spans() if s.name == "eval.run"]
    assert s.attributes["scheduler.workload"] == "eval.run"
    assert s.attributes["scheduler.resource"] == _RESOURCE


@pytest.mark.asyncio
async def test_a_busy_judge_refuses_with_tenant_queue_timeout(judge, monkeypatch) -> None:
    from inference_engine.api.state import app_state  # noqa: PLC0415

    monkeypatch.setattr(settings, "scheduler_global_max_in_flight", 1)
    monkeypatch.setattr(settings, "scheduler_resource_max_in_flight", 1)
    monkeypatch.setattr(settings, "scheduler_queue_timeout_seconds", 0.05)
    held = await app_state.scheduler.acquire(
        tenant="other",
        key_id="other-key",
        resource_key=_RESOURCE,
        resource_limit=1,
        workload="chat.generate",
        priority=20.0,
        estimated_tokens=10,
    )
    try:
        r = await _post(_body())
    finally:
        await app_state.scheduler.release(held)
    assert r.status_code == 503, r.text
    assert r.json()["detail"]["type"] == "tenant_queue_timeout"
    assert judge.params == []


# ---------------------------------------------------------------------------
# Typed errors — and the slot is released on every one of them
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_over_long_prompt_is_a_typed_400(judge) -> None:
    from inference_engine.api.state import app_state  # noqa: PLC0415

    judge.raise_on_generate = ContextLengthExceededError(
        requested_tokens=40_000, context_window=32_768, backend="scripted-judge"
    )
    r = await _post(_body())
    assert r.status_code == 400, r.text
    assert r.json()["detail"]["type"] == "context_length_exceeded"
    assert r.json()["detail"]["context_window"] == 32_768
    assert r.json()["error"]["code"] == "context_length_exceeded"
    assert app_state.scheduler.snapshot().in_flight_by_resource == {}


@pytest.mark.asyncio
async def test_slow_judge_is_a_typed_504(judge, monkeypatch) -> None:
    from inference_engine.api.state import app_state  # noqa: PLC0415

    monkeypatch.setattr(settings, "chat_completion_timeout_seconds", 0.05)
    judge.generation_is_cancellable = True
    judge.sleep_seconds = 5.0
    r = await _post(_body())
    assert r.status_code == 504, r.text
    assert r.json()["detail"]["type"] == "generation_timeout"
    assert r.json()["detail"]["model"] == "judge:1"
    assert app_state.scheduler.snapshot().in_flight_by_resource == {}


@pytest.mark.asyncio
async def test_non_cancellable_judge_is_not_given_a_deadline(judge, monkeypatch) -> None:
    monkeypatch.setattr(settings, "chat_completion_timeout_seconds", 0.01)
    judge.sleep_seconds = 0.05
    r = await _post(_body())
    assert r.status_code == 200, r.text


@pytest.mark.asyncio
async def test_upstream_failure_is_a_typed_502(judge) -> None:
    judge.raise_on_generate = UpstreamGenerationError(
        "upstream 500", upstream_status_code=500, backend="scripted-judge", model="judge:1"
    )
    r = await _post(_body())
    assert r.status_code == 502, r.text
