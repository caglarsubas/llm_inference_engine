"""EvalRunner — candidate output + rubric → judge model → Verdict.

Flow:

  1. Render the rubric's prompt template with the candidate prompt/response/expected.
  2. Acquire the judge adapter from the ``ModelManager`` (loads if necessary).
  3. Generate with ``json_mode=True`` so the judge is constrained to return JSON.
  4. Parse + validate against the rubric's ``expected_keys``. If the judge
     wrapped the JSON in surrounding prose, salvage the first balanced
     ``{...}`` block (``parse_status="repaired"``). If we still can't get a
     dict matching the schema, return ``parse_status="failed"`` with score=0
     so the downstream aggregator can decide what to do.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass

from .. import model_substitution
from ..adapters import GenerationParams
from ..manager import ModelManager, ModelNotFoundError
from ..model_substitution import ModelSubstituter, ModelSubstitution
from ..observability import get_logger, span
from ..schemas import ChatMessage
from .rubrics import RubricSpec, render
from .schemas import Verdict

log = get_logger("evals")


# Greedy enough to match a JSON object in the middle of a Markdown / commentary
# wrapper. Not a full JSON parser — we still json.loads the captured block.
_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


@dataclass(frozen=True)
class EvalOutcome:
    verdict: Verdict
    duration_ms: float
    judge_model: str  # the judge that actually ran
    substitution: ModelSubstitution | None = None


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
    ) -> EvalOutcome:
        """Run a single evaluation, reporting which judge actually ran."""
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
        params = GenerationParams(
            temperature=0.0,  # deterministic-ish judging
            top_p=1.0,
            top_k=0,
            max_tokens=512,
            seed=seed,
            json_mode=True,
        )

        start = time.perf_counter()
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

        with span("eval.run", **attrs) as s:
            result = await adapter.generate(messages, params)
            verdict = self._parse(result.text, rubric)

            s.bind(
                **{
                    "eval.score": verdict.score,
                    "eval.parse_status": verdict.parse_status,
                    "gen_ai.usage.input_tokens": result.prompt_tokens,
                    "gen_ai.usage.output_tokens": result.completion_tokens,
                }
            )

        duration_ms = (time.perf_counter() - start) * 1000
        return EvalOutcome(
            verdict=verdict,
            duration_ms=duration_ms,
            judge_model=judge_model,
            substitution=substitution,
        )

    def _parse(self, raw: str, rubric: RubricSpec) -> Verdict:
        """Extract structured verdict from the judge's response, attempting repair."""
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

        # 3) Give up. Surface a 0 score so downstream aggregation isn't poisoned.
        log.warning("eval.parse_failed", rubric=rubric.name, raw_head=raw[:200])
        return Verdict(score=0.0, parsed={}, raw=raw, parse_status="failed")

    @staticmethod
    def _matches_schema(parsed: object, rubric: RubricSpec) -> bool:
        return isinstance(parsed, dict) and all(k in parsed for k in rubric.expected_keys)


# Convenience helper for routes / tests that want a stable id.
def make_eval_id() -> str:
    return f"eval-{uuid.uuid4().hex}"
