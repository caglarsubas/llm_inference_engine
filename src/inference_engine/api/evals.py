"""LLM-as-a-Judge endpoints — list rubrics, run a single eval."""

from __future__ import annotations

import time

from fastapi import APIRouter, Depends, HTTPException

from ..auth import Identity, require_identity
from ..config import settings
from ..evals.runner import make_eval_id
from ..evals.schemas import (
    EvalRequest,
    EvalResponse,
    PolicyEntryInfo,
    PolicyList,
    PolicyMatchInfo,
    RubricInfo,
    RubricList,
)
from .state import app_state

router = APIRouter()


@router.get("/v1/evals/rubrics", response_model=RubricList)
async def list_rubrics(_=Depends(require_identity)) -> RubricList:
    return RubricList(
        data=[
            RubricInfo(
                name=r.name,
                description=r.description,
                requires_expected=r.requires_expected,
                expected_keys=list(r.expected_keys),
                pairwise=r.pairwise,
            )
            for r in app_state.rubric_registry.all()
        ]
    )


@router.get("/v1/evals/policy", response_model=PolicyList)
async def list_policy(_=Depends(require_identity)) -> PolicyList:
    """List the active server-side auto-eval policy entries (in match-priority
    order). Empty list = no policy installed; per-request ``auto_eval`` only."""
    return PolicyList(
        data=[
            PolicyEntryInfo(
                name=e.name,
                match=PolicyMatchInfo(tenant=e.match.tenant, model=e.match.model),
                rubrics=list(e.spec.rubrics),
                mode=e.spec.mode,
                judge_model=e.spec.judge_model,
            )
            for e in app_state.policy_registry.all()
        ]
    )


@router.post("/v1/evals/run", response_model=EvalResponse)
async def run_eval(
    req: EvalRequest,
    identity: Identity = Depends(require_identity),
) -> EvalResponse:
    rubric = app_state.rubric_registry.get(req.rubric)
    if rubric is None:
        raise HTTPException(status_code=404, detail=f"unknown rubric: {req.rubric!r}")

    if rubric.requires_expected and not req.expected:
        raise HTTPException(
            status_code=400,
            detail=f"rubric {rubric.name!r} requires 'expected' reference text",
        )
    if rubric.pairwise and not req.response_b:
        raise HTTPException(
            status_code=400,
            detail=f"rubric {rubric.name!r} is pairwise — 'response_b' is required",
        )

    judge_model = req.judge_model or settings.default_judge_model

    try:
        verdict, duration_ms = await app_state.eval_runner.run(
            rubric,
            prompt=req.prompt,
            response=req.response,
            response_b=req.response_b,
            expected=req.expected,
            judge_model=judge_model,
            seed=req.seed,
            candidate_model=req.candidate_model,
            candidate_completion_id=req.candidate_completion_id,
            candidate_b_completion_id=req.candidate_b_completion_id,
            tenant=identity.tenant,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return EvalResponse(
        id=make_eval_id(),
        created=int(time.time()),
        rubric=rubric.name,
        judge_model=judge_model,
        candidate_model=req.candidate_model,
        candidate_completion_id=req.candidate_completion_id,
        verdict=verdict,
        duration_ms=round(duration_ms, 2),
    )
