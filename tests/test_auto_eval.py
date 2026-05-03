"""Auto-eval — schemas, helper batching + isolation, blocking/background semantics."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterable
from pathlib import Path

import pytest

from inference_engine.adapters import GenerationParams, InferenceAdapter, StreamChunk
from inference_engine.adapters.base import GenerationResult
from inference_engine.api import _auto_eval
from inference_engine.auth import Identity
from inference_engine.cancellation import Cancellation
from inference_engine.evals import EvalRunner, RubricRegistry
from inference_engine.manager import ModelManager
from inference_engine.registry import ModelDescriptor
from inference_engine.schemas import AutoEvalResult, AutoEvalSpec


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


def test_auto_eval_spec_rejects_empty_rubrics() -> None:
    with pytest.raises(ValueError):
        AutoEvalSpec(rubrics=[])


def test_auto_eval_spec_default_mode_is_background() -> None:
    spec = AutoEvalSpec(rubrics=["helpfulness"])
    assert spec.mode == "background"


def test_auto_eval_spec_accepts_blocking_mode() -> None:
    spec = AutoEvalSpec(rubrics=["safety"], mode="blocking")
    assert spec.mode == "blocking"


def test_auto_eval_spec_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError):
        AutoEvalSpec(rubrics=["safety"], mode="dispatch")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Helper — fake judge adapter, real EvalRunner + RubricRegistry
# ---------------------------------------------------------------------------


class _ScriptedJudge(InferenceAdapter):
    """Adapter whose ``generate`` returns a per-rubric scripted response.

    Simulates a real judge by inspecting the system prompt the runner builds
    and returning a tailored JSON verdict for whichever rubric is in play.
    """

    backend_name = "scripted-judge"

    def __init__(self) -> None:
        self._descriptor: ModelDescriptor | None = None
        self.calls = 0
        self.fail_on_rubric: str | None = None  # raise from generate when this rubric is judged

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
        self.calls += 1
        msgs = list(messages)
        system = msgs[0].content if msgs else ""

        if "helpfulness" in system or "1-5 scale" in system:
            rubric = "helpfulness"
            text = '{"score": 4, "justification": "concise and on-topic"}'
        elif "fact-checking" in system:
            rubric = "correctness"
            text = '{"correct": true, "reason": "matches the expected reference"}'
        elif "safety classifier" in system:
            rubric = "safety"
            text = '{"safe": true, "concerns": []}'
        else:  # pragma: no cover — should not happen
            rubric = "unknown"
            text = "{}"

        if self.fail_on_rubric == rubric:
            raise RuntimeError(f"simulated judge failure for {rubric}")

        return GenerationResult(text=text, finish_reason="stop", prompt_tokens=10, completion_tokens=10)

    async def stream(
        self, messages: Iterable, params: GenerationParams, cancel: Cancellation | None = None
    ) -> AsyncIterator[StreamChunk]:
        yield StreamChunk(text="x", finish_reason="stop")


@pytest.fixture
def runner_kit() -> tuple[EvalRunner, RubricRegistry, _ScriptedJudge]:
    """Build (runner, rubric_registry, judge) wired through a real ModelManager."""
    desc = ModelDescriptor(
        name="judge",
        tag="1",
        namespace="ns",
        registry="reg",
        model_path=Path("/tmp/judge"),
        format="gguf",
        size_bytes=1,
    )

    class _Reg:
        def get(self, name: str) -> ModelDescriptor | None:
            return desc if name == "judge:1" else None

        def list_models(self) -> list[ModelDescriptor]:
            return [desc]

    judge = _ScriptedJudge()
    mgr = ModelManager(_Reg(), adapter_factory=lambda d: judge, memory_budget_bytes=100)
    runner = EvalRunner(mgr)
    rubric_registry = RubricRegistry.with_builtins()
    return runner, rubric_registry, judge


_IDENTITY = Identity(tenant="dev", key_id="sk-x")


# ---------------------------------------------------------------------------
# run_blocking — concurrency + isolation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_blocking_runs_all_rubrics_in_request_order(runner_kit) -> None:
    runner, registry, judge = runner_kit
    spec = AutoEvalSpec(rubrics=["helpfulness", "safety"], mode="blocking")

    results = await _auto_eval.run_blocking(
        runner,
        registry,
        spec,
        default_judge_model="judge:1",
        prompt="What is 2+2?",
        response="4",
        candidate_model="llama:1",
        candidate_completion_id="chatcmpl-1",
        identity=_IDENTITY,
    )

    assert [r.rubric for r in results] == ["helpfulness", "safety"]
    assert results[0].verdict["score"] == 4.0
    assert results[1].verdict["score"] == 1.0
    assert all(r.error is None for r in results)
    assert judge.calls == 2


@pytest.mark.asyncio
async def test_blocking_isolates_per_rubric_failures(runner_kit) -> None:
    """One rubric's exception must not poison the others' verdicts."""
    runner, registry, judge = runner_kit
    judge.fail_on_rubric = "safety"
    spec = AutoEvalSpec(rubrics=["helpfulness", "safety"], mode="blocking")

    results = await _auto_eval.run_blocking(
        runner,
        registry,
        spec,
        default_judge_model="judge:1",
        prompt="x",
        response="y",
        candidate_model="m",
        candidate_completion_id="c",
        identity=_IDENTITY,
    )

    assert results[0].error is None
    assert results[0].verdict["score"] == 4.0
    assert results[1].error is not None
    assert "simulated judge failure" in results[1].error
    assert results[1].verdict["parse_status"] == "failed"
    assert results[1].verdict["score"] == 0.0


@pytest.mark.asyncio
async def test_unknown_rubric_yields_failed_result_not_exception(runner_kit) -> None:
    runner, registry, _ = runner_kit
    spec = AutoEvalSpec(rubrics=["ghost"], mode="blocking")

    results = await _auto_eval.run_blocking(
        runner,
        registry,
        spec,
        default_judge_model="judge:1",
        prompt="x",
        response="y",
        candidate_model="m",
        candidate_completion_id="c",
        identity=_IDENTITY,
    )

    assert len(results) == 1
    assert results[0].error == "unknown rubric: 'ghost'"
    assert results[0].verdict["parse_status"] == "failed"


@pytest.mark.asyncio
async def test_correctness_without_expected_isolated_to_one_rubric(runner_kit) -> None:
    """``correctness`` requires expected; the runner ValueError must not break sibling rubrics."""
    runner, registry, _ = runner_kit
    spec = AutoEvalSpec(rubrics=["correctness", "safety"], mode="blocking", expected=None)

    results = await _auto_eval.run_blocking(
        runner,
        registry,
        spec,
        default_judge_model="judge:1",
        prompt="x",
        response="y",
        candidate_model="m",
        candidate_completion_id="c",
        identity=_IDENTITY,
    )

    by_rubric = {r.rubric: r for r in results}
    assert by_rubric["correctness"].error is not None
    assert "requires an 'expected'" in by_rubric["correctness"].error
    assert by_rubric["safety"].error is None
    assert by_rubric["safety"].verdict["score"] == 1.0


# ---------------------------------------------------------------------------
# run_background — fire-and-forget semantics
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_background_returns_a_task_that_finishes_with_results(runner_kit) -> None:
    runner, registry, judge = runner_kit
    spec = AutoEvalSpec(rubrics=["helpfulness"])  # default mode = background

    task = _auto_eval.run_background(
        runner,
        registry,
        spec,
        default_judge_model="judge:1",
        prompt="x",
        response="y",
        candidate_model="m",
        candidate_completion_id="c",
        identity=_IDENTITY,
    )

    assert isinstance(task, asyncio.Task)
    results = await task
    assert isinstance(results, list)
    assert len(results) == 1
    assert isinstance(results[0], AutoEvalResult)
    assert results[0].verdict["score"] == 4.0
    assert judge.calls == 1


@pytest.mark.asyncio
async def test_background_task_survives_caller_returning(runner_kit) -> None:
    """Spawn the task inside an inner coroutine that returns immediately;
    the task should still complete on the running event loop."""
    runner, registry, judge = runner_kit
    spec = AutoEvalSpec(rubrics=["helpfulness", "safety"])

    async def caller() -> asyncio.Task:
        return _auto_eval.run_background(
            runner,
            registry,
            spec,
            default_judge_model="judge:1",
            prompt="x",
            response="y",
            candidate_model="m",
            candidate_completion_id="c",
            identity=_IDENTITY,
        )

    task = await caller()
    # caller has returned; task should still be alive and complete cleanly.
    results = await task
    assert len(results) == 2
    assert judge.calls == 2
