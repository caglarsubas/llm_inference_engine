"""EvalRunner — candidate output + rubric → judge model → Verdict.

Flow:

  1. Render the rubric's prompt template with the candidate prompt/response/expected.
  2. Acquire the judge adapter from the ``ModelManager`` (loads if necessary).
  3. Generate with ``json_mode=True`` so the judge is constrained to return JSON,
     and ``think=False`` so a reasoning judge answers instead of spending the
     token budget on chain of thought. Each call runs under the engine's
     total-elapsed generation deadline and, when the caller passes an
     ``admission``, inside a scheduler slot. ``n`` repeats make ``n`` calls,
     repeat ``i`` with ``seed + i``.
  4. Parse + validate against the rubric's ``expected_keys``. If the judge
     wrapped the JSON in surrounding prose, salvage the first balanced
     ``{...}`` block (``parse_status="repaired"``). If the judge ran out of
     tokens (``finish_reason="length"``) inside a string, close the object
     and keep the score when it does not come from the cut string
     (``parse_status="truncated"``). If we still can't get a dict matching the
     schema, return ``parse_status="failed"`` with score=0 so the downstream
     aggregator can decide what to do.
"""

from __future__ import annotations

import json
import re
import statistics
import time
import uuid
from collections import Counter
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager, nullcontext
from dataclasses import dataclass

from .. import model_substitution
from ..adapters import GenerationParams, InferenceAdapter
from ..adapters.base import ContextLengthExceededError, GenerationTimeoutError
from ..generation_deadline import generate_within_deadline
from ..manager import ModelManager, ModelNotFoundError
from ..model_substitution import ModelSubstituter, ModelSubstitution
from ..observability import get_logger, span
from ..schemas import ChatMessage
from .rubrics import RubricSpec, render
from .schemas import RepeatSummary, Verdict

log = get_logger("evals")


# Greedy enough to match a JSON object in the middle of a Markdown / commentary
# wrapper. Not a full JSON parser — we still json.loads the captured block.
_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)

# A verdict is a score plus a short reason; see ``think=False`` below. Up to
# 2026-09-30, 49 of 1357 qwen3.6:27b verdicts ran into this cap mid-
# justification. It stays at 512 because those calls already took up to 99 s,
# and doubling it walks them toward the generation deadline. The built-in
# rubrics ask for a short reason instead, and a cut-off verdict whose score was
# already written is kept (``parse_status="truncated"``).
_JUDGE_MAX_TOKENS = 512

# Appended to the string the judge was cut off in, to check the score does not
# depend on it.
_CUT_MARK = "\u2026"

# Opens a scheduler slot around one judge call: given the resolved adapter, the
# judge model and an estimated token count, it yields span attributes for the
# admission. The route supplies it; background auto-eval runs without one.
Admission = Callable[[InferenceAdapter, str, int], AbstractAsyncContextManager[dict]]


@dataclass(frozen=True)
class EvalOutcome:
    verdict: Verdict
    duration_ms: float
    judge_model: str  # the judge that actually ran
    substitution: ModelSubstitution | None = None
    # Every repeat's verdict, in seed order; ``verdict`` is the first.
    verdicts: tuple[Verdict, ...] = ()


def summarize_repeats(verdicts: list[Verdict] | tuple[Verdict, ...]) -> RepeatSummary:
    """Agreement across repeated verdicts, leaving out the ones that failed."""
    scores = [v.score for v in verdicts if v.parse_status != "failed"]
    if not scores:
        return RepeatSummary(n=len(verdicts), parsed=0)
    modal = Counter(scores).most_common(1)[0][1]
    return RepeatSummary(
        n=len(verdicts),
        parsed=len(scores),
        mean=statistics.fmean(scores),
        stdev=statistics.pstdev(scores),
        min=min(scores),
        max=max(scores),
        agreement=modal / len(scores),
    )


class EvalRunner:
    def __init__(
        self,
        manager: ModelManager,
        substituter: ModelSubstituter | None = None,
    ) -> None:
        self._manager = manager
        self._substituter = substituter

    async def run(
        self,
        rubric: RubricSpec,
        **kwargs,
    ) -> tuple[Verdict, float]:
        """Run a single evaluation. Returns (verdict, duration_ms).

        Takes the same keywords as :meth:`evaluate`, which also reports the
        judge that actually ran.
        """
        outcome = await self.evaluate(rubric, **kwargs)
        return outcome.verdict, outcome.duration_ms

    async def evaluate(
        self,
        rubric: RubricSpec,
        *,
        prompt: str,
        response: str,
        expected: str | None,
        judge_model: str,
        seed: int | None = 0,
        response_b: str | None = None,
        # provenance fields purely for span attribution
        candidate_model: str | None = None,
        candidate_completion_id: str | None = None,
        candidate_b_completion_id: str | None = None,
        tenant: str | None = None,
        temperature: float = 0.0,
        n: int = 1,
        admission: Admission | None = None,
    ) -> EvalOutcome:
        """Run one evaluation ``n`` times, reporting which judge actually ran.

        Raises ``ContextLengthExceededError`` when the rendered prompt does not
        fit the judge, and ``GenerationTimeoutError`` when a judge call passes
        the generation deadline; one failed repeat fails the evaluation.
        """
        if n < 1:
            raise ValueError("n must be at least 1")
        if rubric.requires_expected and not expected:
            raise ValueError(f"rubric {rubric.name!r} requires an 'expected' reference")
        if rubric.pairwise and not response_b:
            raise ValueError(
                f"rubric {rubric.name!r} is pairwise — 'response_b' is required"
            )

        requested_judge = judge_model
        substitution: ModelSubstitution | None = None
        if self._substituter is not None:
            substitution = await self._substituter.choose(judge_model, self._manager)
        try:
            if substitution is not None:
                try:
                    adapter, _desc = await self._manager.get(substitution.served_model)
                    judge_model = substitution.served_model
                except ModelNotFoundError:
                    substitution = None
            if substitution is None:
                adapter, _desc = await self._manager.get(judge_model)
        except ModelNotFoundError as exc:
            raise ValueError(f"judge model not found: {judge_model!r}") from exc

        rendered_user = render(
            rubric.user_prompt_template,
            prompt=prompt,
            response=response,
            response_b=response_b or "",
            expected=expected or "",
        )
        messages = [
            ChatMessage(role="system", content=rubric.system_prompt),
            ChatMessage(role="user", content=rendered_user),
        ]
        estimated_tokens = max(
            1, (len(rubric.system_prompt) + len(rendered_user)) // 4 + _JUDGE_MAX_TOKENS
        )

        start = time.perf_counter()
        verdicts: list[Verdict] = []
        for index in range(n):
            params = GenerationParams(
                # 0.0 by default: deterministic-ish judging. Repeats raise it so
                # the spread across verdicts measures the judge, not the seed.
                temperature=temperature,
                top_p=1.0,
                top_k=0,
                max_tokens=_JUDGE_MAX_TOKENS,
                seed=None if seed is None else seed + index,
                json_mode=True,
                # A verdict is a few dozen tokens of JSON. Left to its default a
                # reasoning judge (qwen3.8) thinks first and spends all 512 tokens
                # there, returning empty content: 7 of 10 judge calls on
                # 2026-09-26 parsed as ``failed`` with ``raw_head=''`` this way.
                think=False,
            )
            slot = (
                admission(adapter, judge_model, estimated_tokens)
                if admission is not None
                else nullcontext({})
            )
            async with slot as admission_attrs:
                verdicts.append(
                    await self._judge_once(
                        adapter,
                        rubric,
                        messages,
                        params,
                        judge_model=judge_model,
                        requested_judge=requested_judge,
                        substitution=substitution,
                        candidate_model=candidate_model,
                        candidate_completion_id=candidate_completion_id,
                        candidate_b_completion_id=candidate_b_completion_id,
                        tenant=tenant,
                        repeat=(index, n) if n > 1 else None,
                        extra_attrs=admission_attrs or {},
                    )
                )

        duration_ms = (time.perf_counter() - start) * 1000
        return EvalOutcome(
            verdict=verdicts[0],
            duration_ms=duration_ms,
            judge_model=judge_model,
            substitution=substitution,
            verdicts=tuple(verdicts),
        )

    async def _judge_once(
        self,
        adapter: InferenceAdapter,
        rubric: RubricSpec,
        messages: list[ChatMessage],
        params: GenerationParams,
        *,
        judge_model: str,
        requested_judge: str,
        substitution: ModelSubstitution | None,
        candidate_model: str | None,
        candidate_completion_id: str | None,
        candidate_b_completion_id: str | None,
        tenant: str | None,
        repeat: tuple[int, int] | None,
        extra_attrs: dict,
    ) -> Verdict:
        attrs: dict[str, object] = {
            "eval.rubric.name": rubric.name,
            "eval.judge.model": judge_model,
            "gen_ai.system": adapter.backend_name,
            "gen_ai.request.model": judge_model,
            "gen_ai.request.temperature": params.temperature,
            "llm.request.key_source": getattr(
                adapter, "request_key_source", "local-inference"
            ),
            **model_substitution.span_attrs(substitution),
        }
        if substitution is not None:
            attrs["eval.judge.requested_model"] = requested_judge
        if candidate_model:
            attrs["eval.candidate.model"] = candidate_model
        if candidate_completion_id:
            attrs["eval.candidate.completion_id"] = candidate_completion_id
        if candidate_b_completion_id:
            attrs["eval.candidate_b.completion_id"] = candidate_b_completion_id
        if rubric.pairwise:
            attrs["eval.pairwise"] = True
        if tenant:
            attrs["planeon.tenant"] = tenant
        if repeat is not None:
            attrs["eval.repeat.index"], attrs["eval.repeat.n"] = repeat
        attrs.update(extra_attrs)

        with span("eval.run", **attrs) as s:
            try:
                result = await generate_within_deadline(adapter, messages, params, judge_model)
            except ContextLengthExceededError:
                s.bind(**{"error.type": "context_length_exceeded"})
                raise
            except GenerationTimeoutError:
                s.bind(**{"error.type": "generation_timeout"})
                raise
            verdict = self._parse(
                result.text, rubric, truncated=result.finish_reason == "length"
            )

            s.bind(
                **{
                    "eval.score": verdict.score,
                    "eval.parse_status": verdict.parse_status,
                    "gen_ai.response.finish_reasons": result.finish_reason,
                    "gen_ai.usage.input_tokens": result.prompt_tokens,
                    "gen_ai.usage.output_tokens": result.completion_tokens,
                }
            )
        return verdict

    def _parse(self, raw: str, rubric: RubricSpec, *, truncated: bool = False) -> Verdict:
        """Extract structured verdict from the judge's response, attempting repair.

        ``truncated`` says the judge stopped at the token cap rather than
        finishing its answer.
        """
        # 1) Try clean: the whole response IS valid JSON.
        try:
            parsed = json.loads(raw)
            if self._matches_schema(parsed, rubric):
                return Verdict(
                    score=rubric.score_extractor(parsed),
                    parsed=parsed,
                    raw=raw,
                    parse_status="clean",
                )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            pass

        # 2) Repair: pull the first {...} block out of surrounding prose.
        match = _JSON_OBJECT.search(raw)
        if match:
            try:
                parsed = json.loads(match.group(0))
                if self._matches_schema(parsed, rubric):
                    return Verdict(
                        score=rubric.score_extractor(parsed),
                        parsed=parsed,
                        raw=raw,
                        parse_status="repaired",
                    )
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                pass

        # 3) Truncated: the judge hit the token cap partway through a string,
        #    typically the justification after an already-written score.
        if truncated and (closed := _close_truncated(raw)) is not None:
            as_cut, marked = closed
            try:
                parsed = json.loads(as_cut)
                if self._matches_schema(parsed, rubric):
                    score = rubric.score_extractor(parsed)
                    # A score read from the cut string could come from a
                    # clipped label; keep it only if marking the cut string
                    # leaves it unchanged.
                    if rubric.score_extractor(json.loads(marked)) == score:
                        return Verdict(
                            score=score,
                            parsed=parsed,
                            raw=raw,
                            parse_status="truncated",
                        )
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                pass

        # 4) Give up. Surface a 0 score so downstream aggregation isn't poisoned.
        log.warning("eval.parse_failed", rubric=rubric.name, raw_head=raw[:200])
        return Verdict(score=0.0, parsed={}, raw=raw, parse_status="failed")

    @staticmethod
    def _matches_schema(parsed: object, rubric: RubricSpec) -> bool:
        return isinstance(parsed, dict) and all(k in parsed for k in rubric.expected_keys)


def _close_truncated(raw: str) -> tuple[str, str] | None:
    """Close the JSON object a judge was cut off inside a string of.

    Returns the object closed as written and closed with ``_CUT_MARK`` added to
    the cut string, or None unless ``raw`` holds one unfinished object whose
    text ends inside a string. A cut anywhere else, such as after a number that
    may itself be clipped, is not salvaged.
    """
    start = raw.find("{")
    if start < 0:
        return None
    body = raw[start:]
    closers: list[str] = []
    in_string = escaped = False
    for ch in body:
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch in "{[":
            closers.append("}" if ch == "{" else "]")
        elif ch in "}]":
            # A mismatched bracket is malformed; a closed outer object is not
            # truncated.
            if not closers or closers.pop() != ch or not closers:
                return None
    if not in_string:
        return None
    # Drop an escape the cut split: a lone backslash, or ``\u`` short of 4 hex.
    tail = re.search(r"(\\+)(u[0-9a-fA-F]{0,3})?$", body)
    if tail and len(tail.group(1)) % 2:
        body = body[: tail.start()] + tail.group(1)[:-1]
    closing = '"' + "".join(reversed(closers))
    return body + closing, body + _CUT_MARK + closing


# Convenience helper for routes / tests that want a stable id.
def make_eval_id() -> str:
    return f"eval-{uuid.uuid4().hex}"
