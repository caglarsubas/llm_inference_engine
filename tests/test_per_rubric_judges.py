"""Per-rubric judge model overrides — schema, resolver, end-to-end batch dispatch."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterable
from pathlib import Path

import pytest

from inference_engine.adapters import GenerationParams, InferenceAdapter, StreamChunk
from inference_engine.adapters.base import GenerationResult
from inference_engine.api._auto_eval import _resolve_judge_model, run_blocking
from inference_engine.auth import Identity
from inference_engine.cancellation import Cancellation
from inference_engine.evals import EvalRunner, RubricRegistry
from inference_engine.manager import ModelManager
from inference_engine.registry import ModelDescriptor
from inference_engine.schemas import AutoEvalSpec


# ---------------------------------------------------------------------------
# Schema acceptance
# ---------------------------------------------------------------------------


def test_spec_accepts_judge_models_map() -> None:
    spec = AutoEvalSpec(
        rubrics=["safety", "correctness"],
        judge_model="llama3.2:1b",
        judge_models={"correctness": "llama3.2:3b"},
    )
    assert spec.judge_models == {"correctness": "llama3.2:3b"}
    assert spec.judge_model == "llama3.2:1b"


def test_spec_judge_models_optional() -> None:
    spec = AutoEvalSpec(rubrics=["safety"], judge_model="x")
    assert spec.judge_models is None


# ---------------------------------------------------------------------------
# Resolver — precedence is per-rubric > spec-default > settings-default
# ---------------------------------------------------------------------------


def test_resolver_per_rubric_override_wins() -> None:
    spec = AutoEvalSpec(
        rubrics=["safety", "correctness"],
        judge_model="default-judge",
        judge_models={"correctness": "premium-judge"},
    )
    assert _resolve_judge_model(spec, "correctness", "settings-default") == "premium-judge"


def test_resolver_falls_back_to_spec_default() -> None:
    spec = AutoEvalSpec(
        rubrics=["safety", "correctness"],
        judge_model="default-judge",
        judge_models={"correctness": "premium-judge"},
    )
    # safety isn't in judge_models → spec.judge_model
    assert _resolve_judge_model(spec, "safety", "settings-default") == "default-judge"


def test_resolver_falls_back_to_settings_default_when_no_spec() -> None:
    spec = AutoEvalSpec(rubrics=["safety"])  # no judge_model, no overrides
    assert _resolve_judge_model(spec, "safety", "settings-default") == "settings-default"


def test_resolver_handles_empty_overrides_map() -> None:
    spec = AutoEvalSpec(rubrics=["safety"], judge_model="x", judge_models={})
    assert _resolve_judge_model(spec, "safety", "fallback") == "x"


# ---------------------------------------------------------------------------
# End-to-end — running two rubrics dispatches to two different judges
# ---------------------------------------------------------------------------


class _JudgeRouter(InferenceAdapter):
    """Adapter that records which judge_model it was loaded as.

    For this test we use a single adapter instance; ManagerBox below routes
    each judge_model id to its own ``_JudgeRouter`` so we can verify they
    were called with the expected payloads.
    """

    backend_name = "router"

    def __init__(self, label: str) -> None:
        self.label = label
        self.calls: list[str] = []
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
        # System prompt drives whether this is helpfulness/correctness/safety.
        msgs = list(messages)
        system = msgs[0].content if msgs else ""
        self.calls.append(system[:60])
        if "fact-checking" in system:
            text = '{"correct": true, "reason": "ok"}'
        elif "1-5 scale" in system or "helpfulness" in system:
            text = '{"score": 5, "justification": "ok"}'
        elif "safety classifier" in system:
            text = '{"safe": true, "concerns": []}'
        else:  # pragma: no cover
            text = "{}"
        return GenerationResult(text=text, finish_reason="stop", prompt_tokens=10, completion_tokens=10)

    async def stream(
        self, messages: Iterable, params: GenerationParams, cancel: Cancellation | None = None
    ) -> AsyncIterator[StreamChunk]:
        yield StreamChunk(text="", finish_reason="stop")


@pytest.fixture
def two_judge_runner():
    """Build a registry that knows about two judge models, each backed by its own adapter."""
    fast = ModelDescriptor(
        name="fast", tag="judge", namespace="ns", registry="reg",
        model_path=Path("/tmp/fast"), format="gguf", size_bytes=1,
    )
    premium = ModelDescriptor(
        name="premium", tag="judge", namespace="ns", registry="reg",
        model_path=Path("/tmp/premium"), format="gguf", size_bytes=1,
    )
    by_name = {"fast:judge": fast, "premium:judge": premium}

    class _Reg:
        def get(self, name: str) -> ModelDescriptor | None:
            return by_name.get(name)

        def list_models(self) -> list[ModelDescriptor]:
            return list(by_name.values())

    fast_adapter = _JudgeRouter("fast")
    premium_adapter = _JudgeRouter("premium")
    by_descriptor = {"fast:judge": fast_adapter, "premium:judge": premium_adapter}

    def _factory(desc: ModelDescriptor) -> _JudgeRouter:
        return by_descriptor[desc.qualified_name]

    mgr = ModelManager(_Reg(), adapter_factory=_factory, memory_budget_bytes=100)
    return EvalRunner(mgr), RubricRegistry.with_builtins(), fast_adapter, premium_adapter


@pytest.mark.asyncio
async def test_each_rubric_routes_to_its_overridden_judge(two_judge_runner) -> None:
    """When the spec maps correctness → premium and others → fast, the
    correctness rubric hits the premium adapter; safety hits fast."""
    runner, registry, fast, premium = two_judge_runner
    spec = AutoEvalSpec(
        rubrics=["safety", "correctness"],
        judge_model="fast:judge",
        judge_models={"correctness": "premium:judge"},
        expected="reference text",
        mode="blocking",
    )

    results = await run_blocking(
        runner,
        registry,
        spec,
        default_judge_model="fast:judge",
        prompt="What is 2+2?",
        response="4",
        candidate_model="cand:1",
        candidate_completion_id="chatcmpl-x",
        identity=Identity(tenant="dev", key_id="sk-x"),
    )

    assert {r.rubric for r in results} == {"safety", "correctness"}
    by_rubric = {r.rubric: r for r in results}

    assert by_rubric["safety"].judge_model == "fast:judge"
    assert by_rubric["correctness"].judge_model == "premium:judge"

    # The fast adapter saw the safety rubric system prompt; premium saw correctness.
    assert any("safety classifier" in c for c in fast.calls)
    assert any("fact-checking" in c for c in premium.calls)
    # Crucially: premium never saw the safety prompt, fast never saw correctness.
    assert not any("fact-checking" in c for c in fast.calls)
    assert not any("safety classifier" in c for c in premium.calls)
