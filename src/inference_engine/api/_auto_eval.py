"""Auto-eval helper — drive the EvalRunner from chat completion outputs.

Two entry points:

* ``run_blocking()`` — awaitable; returns ``list[AutoEvalResult]`` once every
  requested rubric has produced a verdict. Used in synchronous chat responses.

* ``run_background()`` — schedules the evals on the running event loop and
  returns immediately. The task is **shielded** so the request handler exiting
  doesn't cancel it. Failures are surfaced as structured logs + ``eval.run``
  spans (errors marked) rather than swallowed silently.

Per-rubric isolation: one rubric raising a ValueError (e.g. correctness without
``expected``) doesn't cascade into the others. Each verdict is independent.
"""

from __future__ import annotations

import asyncio
from typing import Any

from ..auth import Identity
from ..evals.runner import EvalRunner
from ..evals.schemas import Verdict
from ..observability import get_logger
from ..schemas import AutoEvalResult, AutoEvalSpec

log = get_logger("auto_eval")


def _resolve_judge_model(spec: AutoEvalSpec, rubric_name: str, default_judge_model: str) -> str:
    """Pick the judge model for one rubric.

    Precedence (highest first):
      1. ``spec.judge_models[rubric_name]`` — per-rubric override
      2. ``spec.judge_model`` — spec-level default
      3. ``settings.default_judge_model`` (passed in as ``default_judge_model``)
    """
    if spec.judge_models and rubric_name in spec.judge_models:
        return spec.judge_models[rubric_name]
    return spec.judge_model or default_judge_model


async def _run_one(
    runner: EvalRunner,
    rubric_registry: Any,
    rubric_name: str,
    *,
    spec: AutoEvalSpec,
    default_judge_model: str,
    prompt: str,
    response: str,
    candidate_model: str,
    candidate_completion_id: str,
    tenant: str,
) -> AutoEvalResult:
    rubric = rubric_registry.get(rubric_name)
    judge_model = _resolve_judge_model(spec, rubric_name, default_judge_model)

    if rubric is None:
        return AutoEvalResult(
            rubric=rubric_name,
            judge_model=judge_model,
            verdict=Verdict(score=0.0, raw="", parse_status="failed").model_dump(),
            duration_ms=0.0,
            error=f"unknown rubric: {rubric_name!r}",
        )

    try:
        verdict, duration_ms = await runner.run(
            rubric,
            prompt=prompt,
            response=response,
            expected=spec.expected,
            judge_model=judge_model,
            candidate_model=candidate_model,
            candidate_completion_id=candidate_completion_id,
            tenant=tenant,
        )
        return AutoEvalResult(
            rubric=rubric.name,
            judge_model=judge_model,
            verdict=verdict.model_dump(),
            duration_ms=round(duration_ms, 2),
        )
    except Exception as exc:  # noqa: BLE001 — isolate per-rubric failures
        log.warning(
            "auto_eval.rubric_failed",
            rubric=rubric_name,
            error=str(exc),
            error_type=type(exc).__name__,
        )
        return AutoEvalResult(
            rubric=rubric_name,
            judge_model=judge_model,
            verdict=Verdict(score=0.0, raw="", parse_status="failed").model_dump(),
            duration_ms=0.0,
            error=str(exc),
        )


async def run_blocking(
    runner: EvalRunner,
    rubric_registry: Any,
    spec: AutoEvalSpec,
    *,
    default_judge_model: str,
    prompt: str,
    response: str,
    candidate_model: str,
    candidate_completion_id: str,
    identity: Identity,
) -> list[AutoEvalResult]:
    """Run every rubric concurrently; collect verdicts in request order."""
    tasks = [
        _run_one(
            runner,
            rubric_registry,
            name,
            spec=spec,
            default_judge_model=default_judge_model,
            prompt=prompt,
            response=response,
            candidate_model=candidate_model,
            candidate_completion_id=candidate_completion_id,
            tenant=identity.tenant,
        )
        for name in spec.rubrics
    ]
    return list(await asyncio.gather(*tasks))


def run_background(
    runner: EvalRunner,
    rubric_registry: Any,
    spec: AutoEvalSpec,
    *,
    default_judge_model: str,
    prompt: str,
    response: str,
    candidate_model: str,
    candidate_completion_id: str,
    identity: Identity,
) -> asyncio.Task:
    """Schedule background evals; return the task handle for tests / shutdown.

    The task is intentionally NOT awaited by the caller. ``asyncio.create_task``
    already detaches the task from the caller's cancellation scope, so the
    request handler returning won't cancel mid-flight evals. The warm prefix
    cache means each rubric typically finishes in <100 ms anyway.
    """
    return asyncio.create_task(
        run_blocking(
            runner,
            rubric_registry,
            spec,
            default_judge_model=default_judge_model,
            prompt=prompt,
            response=response,
            candidate_model=candidate_model,
            candidate_completion_id=candidate_completion_id,
            identity=identity,
        )
    )
