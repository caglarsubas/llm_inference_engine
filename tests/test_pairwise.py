"""Pairwise rubric — runner contract, score extraction, route 400 / 200 paths."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterable
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from inference_engine.adapters import GenerationParams, InferenceAdapter, StreamChunk
from inference_engine.adapters.base import GenerationResult
from inference_engine.cancellation import Cancellation
from inference_engine.evals import EvalRunner, RubricRegistry
from inference_engine.evals.rubrics import PAIRWISE_QUALITY
from inference_engine.main import app
from inference_engine.manager import ModelManager
from inference_engine.registry import ModelDescriptor


# ---------------------------------------------------------------------------
# Spec sanity — score_extractor mapping
# ---------------------------------------------------------------------------


def test_pairwise_score_a_wins() -> None:
    assert PAIRWISE_QUALITY.score_extractor({"winner": "A", "reason": "..."}) == 1.0


def test_pairwise_score_b_wins() -> None:
    assert PAIRWISE_QUALITY.score_extractor({"winner": "B", "reason": "..."}) == 0.0


def test_pairwise_score_tie() -> None:
    assert PAIRWISE_QUALITY.score_extractor({"winner": "tie", "reason": "..."}) == 0.5


def test_pairwise_score_unknown_collapses_to_zero() -> None:
    """Unrecognised winner labels (model misbehaved) collapse to 0.0 so we
    don't silently inherit a default value upstream."""
    assert PAIRWISE_QUALITY.score_extractor({"winner": "AAAA", "reason": "..."}) == 0.0


def test_pairwise_flag_is_true_on_builtin() -> None:
    assert PAIRWISE_QUALITY.pairwise is True


# ---------------------------------------------------------------------------
# Runner — pairwise enforcement + template rendering
# ---------------------------------------------------------------------------


class _ScriptedJudge(InferenceAdapter):
    """Pairwise-aware judge: returns whatever JSON has been set on next_text."""

    backend_name = "scripted-judge"

    def __init__(self) -> None:
        self.next_text: str = ""
        self.last_messages: list = []
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
        self.last_messages = list(messages)
        return GenerationResult(
            text=self.next_text, finish_reason="stop", prompt_tokens=12, completion_tokens=8
        )

    async def stream(
        self, messages: Iterable, params: GenerationParams, cancel: Cancellation | None = None
    ) -> AsyncIterator[StreamChunk]:
        yield StreamChunk(text="", finish_reason="stop")


@pytest.fixture
def runner_kit():
    desc = ModelDescriptor(
        name="judge", tag="1", namespace="ns", registry="reg",
        model_path=Path("/tmp/judge"), format="gguf", size_bytes=1,
    )

    class _Reg:
        def get(self, name: str) -> ModelDescriptor | None:
            return desc if name == "judge:1" else None

        def list_models(self) -> list[ModelDescriptor]:
            return [desc]

    judge = _ScriptedJudge()
    mgr = ModelManager(_Reg(), adapter_factory=lambda d: judge, memory_budget_bytes=100)
    return EvalRunner(mgr), RubricRegistry.with_builtins(), judge


@pytest.mark.asyncio
async def test_pairwise_rubric_renders_both_responses_to_judge(runner_kit) -> None:
    runner, registry, judge = runner_kit
    judge.next_text = '{"winner": "A", "reason": "more concise"}'
    rubric = registry.get("pairwise_quality")

    verdict, _ = await runner.run(
        rubric,
        prompt="Pick a number",
        response="Seven, because it is prime",
        response_b="Forty-two",
        expected=None,
        judge_model="judge:1",
    )

    assert verdict.parsed["winner"] == "A"
    assert verdict.score == 1.0
    # Judge saw both responses in its user message.
    user_msg = next(m for m in judge.last_messages if m.role == "user").content
    assert "Seven, because it is prime" in user_msg
    assert "Forty-two" in user_msg
    assert "RESPONSE A" in user_msg
    assert "RESPONSE B" in user_msg


@pytest.mark.asyncio
async def test_pairwise_runner_rejects_missing_response_b(runner_kit) -> None:
    runner, registry, _ = runner_kit
    rubric = registry.get("pairwise_quality")
    with pytest.raises(ValueError, match="pairwise"):
        await runner.run(
            rubric,
            prompt="x",
            response="A",
            response_b=None,
            expected=None,
            judge_model="judge:1",
        )


@pytest.mark.asyncio
async def test_scalar_rubric_ignores_response_b(runner_kit) -> None:
    """Passing response_b to a non-pairwise rubric is fine — it just isn't used."""
    runner, registry, judge = runner_kit
    judge.next_text = '{"score": 4, "justification": "ok"}'
    rubric = registry.get("helpfulness")

    verdict, _ = await runner.run(
        rubric,
        prompt="x",
        response="hello",
        response_b="this should be ignored",
        expected=None,
        judge_model="judge:1",
    )
    assert verdict.score == 4.0


# ---------------------------------------------------------------------------
# Route — POST /v1/evals/run with pairwise
# ---------------------------------------------------------------------------


@pytest.fixture
def patched_eval_state(monkeypatch, runner_kit):
    """Swap the runtime EvalRunner / rubric_registry so the route uses our scripted judge."""
    runner, registry, judge = runner_kit
    from inference_engine.api.state import app_state  # noqa: PLC0415

    monkeypatch.setattr(app_state, "eval_runner", runner)
    monkeypatch.setattr(app_state, "rubric_registry", registry)
    return judge


def test_route_pairwise_returns_winner(patched_eval_state) -> None:
    patched_eval_state.next_text = '{"winner": "B", "reason": "more accurate"}'
    client = TestClient(app)
    r = client.post(
        "/v1/evals/run",
        json={
            "rubric": "pairwise_quality",
            "prompt": "What is 2+2?",
            "response": "4 (computed)",
            "response_b": "4",
            "judge_model": "judge:1",
            "candidate_completion_id": "chatcmpl-A",
            "candidate_b_completion_id": "chatcmpl-B",
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["rubric"] == "pairwise_quality"
    assert body["verdict"]["score"] == 0.0  # B wins → 0.0
    assert body["verdict"]["parsed"]["winner"] == "B"
    assert body["candidate_completion_id"] == "chatcmpl-A"


def test_route_pairwise_400_without_response_b(patched_eval_state) -> None:
    client = TestClient(app)
    r = client.post(
        "/v1/evals/run",
        json={
            "rubric": "pairwise_quality",
            "prompt": "x",
            "response": "A",
            "judge_model": "judge:1",
        },
    )
    assert r.status_code == 400
    assert "pairwise" in r.json()["detail"]
    assert "response_b" in r.json()["detail"]


def test_route_lists_pairwise_flag_on_rubrics(patched_eval_state) -> None:
    client = TestClient(app)
    r = client.get("/v1/evals/rubrics")
    assert r.status_code == 200
    by_name = {item["name"]: item for item in r.json()["data"]}
    assert by_name["pairwise_quality"]["pairwise"] is True
    assert by_name["helpfulness"]["pairwise"] is False
    assert by_name["correctness"]["pairwise"] is False


def test_route_pairwise_emits_eval_pairwise_attr_on_span(
    patched_eval_state, _session_exporter
) -> None:
    _session_exporter.clear()
    patched_eval_state.next_text = '{"winner": "A", "reason": "ok"}'
    client = TestClient(app)
    r = client.post(
        "/v1/evals/run",
        json={
            "rubric": "pairwise_quality",
            "prompt": "x",
            "response": "A",
            "response_b": "B",
            "judge_model": "judge:1",
            "candidate_b_completion_id": "chatcmpl-B",
        },
    )
    assert r.status_code == 200

    eval_spans = [s for s in _session_exporter.get_finished_spans() if s.name == "eval.run"]
    assert len(eval_spans) == 1
    s = eval_spans[0]
    assert s.attributes["eval.pairwise"] is True
    assert s.attributes["eval.candidate_b.completion_id"] == "chatcmpl-B"
