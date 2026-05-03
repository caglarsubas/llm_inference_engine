"""Eval harness — rubric registry, runner parsing, JSON repair, schema mismatches."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterable
from pathlib import Path

import pytest

from inference_engine.adapters import GenerationParams, InferenceAdapter, StreamChunk
from inference_engine.adapters.base import GenerationResult
from inference_engine.cancellation import Cancellation
from inference_engine.evals import (
    BUILTIN_RUBRICS,
    EvalRunner,
    RubricRegistry,
    RubricSpec,
)
from inference_engine.evals.rubrics import render
from inference_engine.manager import ModelManager
from inference_engine.registry import ModelDescriptor


# ---------------------------------------------------------------------------
# Rubric registry
# ---------------------------------------------------------------------------


def test_builtins_loaded() -> None:
    reg = RubricRegistry.with_builtins()
    names = {r.name for r in reg.all()}
    assert names == {"helpfulness", "correctness", "safety", "pairwise_quality"}


def test_lookup_returns_none_for_unknown() -> None:
    reg = RubricRegistry.with_builtins()
    assert reg.get("ghost") is None
    assert reg.get("helpfulness") is BUILTIN_RUBRICS[0]


def test_register_overrides_existing() -> None:
    reg = RubricRegistry.with_builtins()
    custom = RubricSpec(
        name="helpfulness",
        description="custom override",
        system_prompt="x",
        user_prompt_template="y",
        expected_keys=("score",),
        score_extractor=lambda v: float(v["score"]),
    )
    reg.register(custom)
    assert reg.get("helpfulness").description == "custom override"


def test_render_tolerates_missing_keys() -> None:
    out = render("hi {name}, expected={expected}", name="alex")
    assert out == "hi alex, expected="


def test_correctness_extractor_returns_binary_score() -> None:
    reg = RubricRegistry.with_builtins()
    rubric = reg.get("correctness")
    assert rubric.score_extractor({"correct": True, "reason": "..."}) == 1.0
    assert rubric.score_extractor({"correct": False, "reason": "..."}) == 0.0


# ---------------------------------------------------------------------------
# Runner — uses a fake adapter that returns canned text
# ---------------------------------------------------------------------------


class _FakeJudgeAdapter(InferenceAdapter):
    """Adapter whose ``generate()`` returns whatever ``next_text`` is set to."""

    backend_name = "fake-judge"

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
            text=self.next_text,
            finish_reason="stop",
            prompt_tokens=10,
            completion_tokens=20,
        )

    async def stream(
        self, messages: Iterable, params: GenerationParams, cancel: Cancellation | None = None
    ) -> AsyncIterator[StreamChunk]:
        yield StreamChunk(text=self.next_text, finish_reason="stop")


@pytest.fixture
def runner_with_fake() -> tuple[EvalRunner, _FakeJudgeAdapter]:
    """Build a ModelManager whose factory returns one shared FakeJudgeAdapter."""
    descriptor = ModelDescriptor(
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
            return descriptor if name == "judge:1" else None

        def list_models(self) -> list[ModelDescriptor]:
            return [descriptor]

    fake = _FakeJudgeAdapter()
    mgr = ModelManager(_Reg(), adapter_factory=lambda d: fake, memory_budget_bytes=100)
    return EvalRunner(mgr), fake


@pytest.mark.asyncio
async def test_clean_json_response_yields_clean_status(runner_with_fake) -> None:
    runner, fake = runner_with_fake
    fake.next_text = '{"score": 4, "justification": "clear and on-topic"}'

    rubric = RubricRegistry.with_builtins().get("helpfulness")
    verdict, duration_ms = await runner.run(
        rubric,
        prompt="What is 2+2?",
        response="4",
        expected=None,
        judge_model="judge:1",
    )

    assert verdict.parse_status == "clean"
    assert verdict.score == 4.0
    assert verdict.parsed["justification"] == "clear and on-topic"
    assert duration_ms >= 0


@pytest.mark.asyncio
async def test_repaired_json_when_judge_wraps_in_prose(runner_with_fake) -> None:
    runner, fake = runner_with_fake
    fake.next_text = (
        "Here is my verdict:\n"
        '```json\n{"score": 3, "justification": "ok"}\n```\n'
        "Hope that helps!"
    )

    rubric = RubricRegistry.with_builtins().get("helpfulness")
    verdict, _ = await runner.run(
        rubric,
        prompt="x",
        response="y",
        expected=None,
        judge_model="judge:1",
    )

    assert verdict.parse_status == "repaired"
    assert verdict.score == 3.0


@pytest.mark.asyncio
async def test_failed_parse_yields_zero_score(runner_with_fake) -> None:
    runner, fake = runner_with_fake
    fake.next_text = "I refuse to grade this. No JSON for you."

    rubric = RubricRegistry.with_builtins().get("helpfulness")
    verdict, _ = await runner.run(
        rubric,
        prompt="x",
        response="y",
        expected=None,
        judge_model="judge:1",
    )

    assert verdict.parse_status == "failed"
    assert verdict.score == 0.0
    assert verdict.parsed == {}


@pytest.mark.asyncio
async def test_schema_mismatch_falls_back_to_failed(runner_with_fake) -> None:
    """Judge returns valid JSON, but missing the keys our rubric expects."""
    runner, fake = runner_with_fake
    fake.next_text = '{"verdict": "good", "stars": 4}'  # wrong keys

    rubric = RubricRegistry.with_builtins().get("helpfulness")
    verdict, _ = await runner.run(
        rubric,
        prompt="x",
        response="y",
        expected=None,
        judge_model="judge:1",
    )

    assert verdict.parse_status == "failed"
    assert verdict.score == 0.0


@pytest.mark.asyncio
async def test_correctness_rejects_missing_expected(runner_with_fake) -> None:
    runner, _ = runner_with_fake
    rubric = RubricRegistry.with_builtins().get("correctness")
    with pytest.raises(ValueError, match="requires an 'expected'"):
        await runner.run(
            rubric,
            prompt="x",
            response="y",
            expected=None,
            judge_model="judge:1",
        )


@pytest.mark.asyncio
async def test_unknown_judge_model_raises(runner_with_fake) -> None:
    runner, _ = runner_with_fake
    rubric = RubricRegistry.with_builtins().get("helpfulness")
    with pytest.raises(ValueError, match="judge model not found"):
        await runner.run(
            rubric,
            prompt="x",
            response="y",
            expected=None,
            judge_model="ghost:9",
        )


@pytest.mark.asyncio
async def test_user_prompt_includes_candidate_response(runner_with_fake) -> None:
    """The judge must actually see the candidate text in its messages."""
    runner, fake = runner_with_fake
    fake.next_text = '{"score": 5, "justification": "ok"}'

    rubric = RubricRegistry.with_builtins().get("helpfulness")
    await runner.run(
        rubric,
        prompt="What's 2+2?",
        response="4",
        expected=None,
        judge_model="judge:1",
    )

    user_msg = next(m for m in fake.last_messages if m.role == "user")
    assert "What's 2+2?" in user_msg.content
    assert "4" in user_msg.content
