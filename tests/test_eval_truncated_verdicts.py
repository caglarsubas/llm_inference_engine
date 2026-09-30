"""Judge verdicts cut off at the token cap — salvaged when the score was already
written, failed when the cut could have clipped it — and the scheduler estimate
that has to cover the cap."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterable
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

from inference_engine.adapters import GenerationParams, InferenceAdapter, StreamChunk
from inference_engine.adapters.base import GenerationResult
from inference_engine.cancellation import Cancellation
from inference_engine.evals import EvalRunner, RubricRegistry, Verdict
from inference_engine.evals.runner import summarize_repeats
from inference_engine.evals.schemas import RubricDefinition
from inference_engine.evals.tenant_rubrics import to_spec
from inference_engine.manager import ModelManager
from inference_engine.registry import ModelDescriptor

_RUBRICS = RubricRegistry.with_builtins()

# The shape qwen3.6:27b produced on 2026-09-30: a score, then a justification
# the 512-token cap cut off partway through.
_CUT_HELPFULNESS = (
    '{\n  "score": 5,\n  "justification": "The response walks through the reset '
    "flow step by step, names the exact menu entries, and anticipates the"
)


class _CappedJudge(InferenceAdapter):
    """Returns ``text`` with ``finish_reason``; records the last call."""

    backend_name = "capped-judge"

    def __init__(self) -> None:
        self.text = ""
        self.finish_reason = "length"
        self.last_messages: list = []
        self.last_params: GenerationParams | None = None
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
        self.last_params = params
        return GenerationResult(
            text=self.text,
            finish_reason=self.finish_reason,
            prompt_tokens=900,
            completion_tokens=params.max_tokens if self.finish_reason == "length" else 40,
        )

    async def stream(
        self, messages: Iterable, params: GenerationParams, cancel: Cancellation | None = None
    ) -> AsyncIterator[StreamChunk]:
        yield StreamChunk(text=self.text, finish_reason=self.finish_reason)


@pytest.fixture
def capped() -> tuple[EvalRunner, _CappedJudge]:
    desc = ModelDescriptor(
        name="judge", tag="1", namespace="ns", registry="reg",
        model_path=Path("/tmp/judge"), format="gguf", size_bytes=1,
    )

    class _Reg:
        def get(self, name: str) -> ModelDescriptor | None:
            return desc if name == "judge:1" else None

        def list_models(self) -> list[ModelDescriptor]:
            return [desc]

    adapter = _CappedJudge()
    mgr = ModelManager(_Reg(), adapter_factory=lambda d: adapter, memory_budget_bytes=100)
    return EvalRunner(mgr), adapter


async def _judge(runner: EvalRunner, rubric_name: str, **kwargs) -> Verdict:
    rubric = kwargs.pop("rubric", None) or _RUBRICS.get(rubric_name)
    kwargs.setdefault("prompt", "How do I reset my password?")
    kwargs.setdefault("response", "Use the 'Forgot password' link.")
    kwargs.setdefault("expected", "Use the reset link." if rubric.requires_expected else None)
    if rubric.pairwise:
        kwargs.setdefault("response_b", "Call support.")
    verdict, _ = await runner.run(rubric, judge_model="judge:1", **kwargs)
    return verdict


# ---------------------------------------------------------------------------
# Salvaged
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_score_before_a_cut_off_justification_is_kept(capped) -> None:
    runner, judge = capped
    judge.text = _CUT_HELPFULNESS
    verdict = await _judge(runner, "helpfulness")

    assert verdict.parse_status == "truncated"
    assert verdict.score == 5.0
    assert verdict.parsed["justification"].endswith("and anticipates the")
    assert verdict.raw == _CUT_HELPFULNESS


@pytest.mark.parametrize(
    ("rubric", "text", "score"),
    [
        ("pairwise_quality", '{\n"winner": "A",\n"reason": "A names the menu entries', 1.0),
        ("pairwise_quality", '{"winner": "tie", "reason": "Both are', 0.5),
        ("correctness", '{"correct": false, "reason": "The reference says', 0.0),
        # The cut lands inside a list: the list and the object are both closed.
        ("safety", '{"safe": false, "concerns": ["harmful instructions", "step-by-st', 0.0),
    ],
)
@pytest.mark.asyncio
async def test_each_builtin_shape_is_salvaged(capped, rubric, text, score) -> None:
    runner, judge = capped
    judge.text = text
    verdict = await _judge(runner, rubric)
    assert (verdict.parse_status, verdict.score) == ("truncated", score)


@pytest.mark.parametrize(
    ("tail", "kept"),
    [
        ("quotes \\", "quotes "),  # a lone backslash: the escape was cut
        ("quotes \\\\", "quotes \\"),  # an escaped backslash is complete
        ("says \\u00", "says "),  # a unicode escape short of four hex digits
        ('says \\"reset', 'says "reset'),
    ],
)
@pytest.mark.asyncio
async def test_a_cut_escape_is_dropped(capped, tail, kept) -> None:
    runner, judge = capped
    judge.text = '{"score": 3, "justification": "' + tail
    verdict = await _judge(runner, "helpfulness")
    assert verdict.parse_status == "truncated"
    assert verdict.parsed["justification"] == kept


@pytest.mark.asyncio
async def test_a_fenced_cut_off_verdict_is_salvaged(capped) -> None:
    runner, judge = capped
    judge.text = "```json\n" + _CUT_HELPFULNESS
    verdict = await _judge(runner, "helpfulness")
    assert (verdict.parse_status, verdict.score) == ("truncated", 5.0)


def test_truncated_verdicts_count_in_repeat_statistics() -> None:
    summary = summarize_repeats(
        [
            Verdict(score=5.0, raw="", parse_status="truncated"),
            Verdict(score=5.0, raw="", parse_status="clean"),
        ]
    )
    assert (summary.parsed, summary.mean, summary.agreement) == (2, 5.0, 1.0)


# ---------------------------------------------------------------------------
# Not salvaged
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_only_a_length_stop_is_salvaged(capped) -> None:
    runner, judge = capped
    judge.text = _CUT_HELPFULNESS
    judge.finish_reason = "stop"
    verdict = await _judge(runner, "helpfulness")
    assert (verdict.parse_status, verdict.score) == ("failed", 0.0)


@pytest.mark.parametrize(
    ("rubric", "text"),
    [
        # The score comes last and the cut is in it: "A" may be "A" or not.
        ("pairwise_quality", '{"reason": "A is clearer", "winner": "A'),
        # A number with nothing after it may itself be clipped (4 of 45).
        ("helpfulness", '{"justification": "Clear.", "score": 4'),
        # Cut inside a key: the justification never started.
        ("helpfulness", '{"score": 5, "justif'),
        # Cut between members: a key the rubric needs is missing.
        ("helpfulness", '{"score": 5, '),
        # Nothing but the opening brace (a thinking judge spent the budget).
        ("helpfulness", "{"),
        ("helpfulness", ""),
    ],
)
@pytest.mark.asyncio
async def test_a_cut_that_could_clip_the_score_fails(capped, rubric, text) -> None:
    runner, judge = capped
    judge.text = text
    verdict = await _judge(runner, rubric)
    assert (verdict.parse_status, verdict.score) == ("failed", 0.0)


@pytest.mark.asyncio
async def test_a_clipped_tenant_label_fails(capped) -> None:
    rubric = to_spec(
        RubricDefinition.model_validate(
            {
                "name": "process_conformance",
                "system_prompt": "Judge conformance. Output JSON.",
                "user_prompt_template": "{response}",
                "expected_keys": ["reason", "verdict"],
                "score": {
                    "kind": "choice",
                    "key": "verdict",
                    "values": {"conformant": 1, "partially_conformant": 0.5, "nonconformant": 0},
                },
            }
        )
    )
    runner, judge = capped
    judge.text = '{"reason": "Skips the approval step.", "verdict": "partially_confor'
    verdict = await _judge(runner, "", rubric=rubric)
    assert (verdict.parse_status, verdict.score) == ("failed", 0.0)

    judge.text = '{"verdict": "partially_conformant", "reason": "Skips the approval'
    verdict = await _judge(runner, "", rubric=rubric)
    assert (verdict.parse_status, verdict.score) == ("truncated", 0.5)


# ---------------------------------------------------------------------------
# Span and scheduler estimate
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_span_says_the_judge_hit_the_cap(capped, _session_exporter) -> None:
    _session_exporter.clear()
    runner, judge = capped
    judge.text = _CUT_HELPFULNESS
    await _judge(runner, "helpfulness")

    (s,) = [s for s in _session_exporter.get_finished_spans() if s.name == "eval.run"]
    assert s.attributes["eval.parse_status"] == "truncated"
    assert s.attributes["eval.score"] == 5.0
    assert s.attributes["gen_ai.response.finish_reasons"] == "length"
    assert s.attributes["gen_ai.usage.output_tokens"] == judge.last_params.max_tokens


@pytest.mark.parametrize("rubric", ["helpfulness", "correctness", "safety", "pairwise_quality"])
@pytest.mark.asyncio
async def test_scheduler_estimate_covers_the_judge_cap(capped, rubric) -> None:
    runner, judge = capped
    judge.text = '{"score": 4, "justification": "ok"}'
    estimates: list[int] = []

    @asynccontextmanager
    async def admission(adapter, model, estimated_tokens):
        estimates.append(estimated_tokens)
        yield {}

    await _judge(runner, rubric, n=2, temperature=0.5, admission=admission)

    prompt_chars = sum(len(m.content) for m in judge.last_messages)
    assert estimates == [prompt_chars // 4 + judge.last_params.max_tokens] * 2
    assert judge.last_params.max_tokens == 512
    assert judge.last_params.think is False
