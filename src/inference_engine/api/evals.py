"""LLM-as-a-Judge endpoints — rubrics (built-in and per tenant), policy, runs."""

from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import APIRouter, Depends, HTTPException, Response

from .. import model_substitution
from ..adapters import InferenceAdapter
from ..adapters.base import (
    ContextLengthExceededError,
    GenerationTimeoutError,
    UpstreamGenerationError,
)
from ..auth import Identity, require_identity
from ..config import settings
from ..evals.rubrics import RubricSpec
from ..evals.runner import make_eval_id, summarize_repeats
from ..evals.schemas import (
    EvalRequest,
    EvalResponse,
    PolicyEntryInfo,
    PolicyList,
    PolicyMatchInfo,
    RubricDefinition,
    RubricDetail,
    RubricInfo,
    RubricList,
)
from ..evals.tenant_rubrics import RubricLimitError, StoredRubric
from . import _model_routing
from ._scheduling import acquire_slot, scheduler_span_attrs
from .chat import _raise_generation_http_error
from .state import app_state

router = APIRouter()

# Below interactive chat (20) and single completions (5), above rerank (-5):
# judge calls are usually batch work that a person is not watching token by
# token.
_EVAL_PRIORITY = 0.0


def _builtin_info(r: RubricSpec) -> dict:
    return {
        "name": r.name,
        "description": r.description,
        "requires_expected": r.requires_expected,
        "expected_keys": list(r.expected_keys),
        "pairwise": r.pairwise,
    }


def _tenant_info(stored: StoredRubric) -> dict:
    d = stored.definition
    return {
        "name": d.name,
        "description": d.description,
        "requires_expected": d.requires_expected,
        "expected_keys": list(d.expected_keys),
        "pairwise": d.pairwise,
        "source": "tenant",
        "digest": stored.digest,
    }


def _tenant_detail(stored: StoredRubric) -> RubricDetail:
    d = stored.definition
    return RubricDetail(
        **_tenant_info(stored),
        system_prompt=d.system_prompt,
        user_prompt_template=d.user_prompt_template,
        score=d.score,
        created_at=stored.created_at,
        updated_at=stored.updated_at,
    )


def _reserved(name: str) -> HTTPException:
    return HTTPException(
        status_code=409,
        detail={
            "message": f"{name!r} is a built-in rubric; choose another name",
            "type": "rubric_name_reserved",
            "code": "rubric_name_reserved",
            "param": "name",
        },
    )


def _unknown(name: str) -> HTTPException:
    return HTTPException(status_code=404, detail=f"unknown rubric: {name!r}")


@router.get("/v1/evals/rubrics", response_model=RubricList)
async def list_rubrics(identity: Identity = Depends(require_identity)) -> RubricList:
    """Built-in rubrics plus the ones the caller's tenant registered."""
    tenant = app_state.tenant_rubrics.list(identity.tenant)
    own = {stored.definition.name for stored in tenant}
    return RubricList(
        data=[
            RubricInfo(**_builtin_info(r))
            for r in app_state.rubric_registry.all()
            if r.name not in own
        ]
        + [RubricInfo(**_tenant_info(stored)) for stored in tenant]
    )


@router.get("/v1/evals/rubrics/{name}", response_model=RubricDetail)
async def get_rubric(name: str, identity: Identity = Depends(require_identity)) -> RubricDetail:
    stored = app_state.tenant_rubrics.get(identity.tenant, name)
    if stored is not None:
        return _tenant_detail(stored)
    builtin = app_state.rubric_registry.get(name)
    if builtin is None:
        raise _unknown(name)
    return RubricDetail(
        **_builtin_info(builtin),
        system_prompt=builtin.system_prompt,
        user_prompt_template=builtin.user_prompt_template,
    )


@router.post("/v1/evals/rubrics", response_model=RubricDetail)
async def register_rubric(
    definition: RubricDefinition,
    response: Response,
    identity: Identity = Depends(require_identity),
) -> RubricDetail:
    """Register a rubric for the caller's tenant, or replace one it owns.

    201 when the name is new, 200 when it replaced a rubric; the ``digest``
    changes whenever the definition does. Other tenants never see it.
    """
    if app_state.rubric_registry.get(definition.name) is not None:
        raise _reserved(definition.name)
    try:
        stored, created = app_state.tenant_rubrics.put(identity.tenant, definition)
    except RubricLimitError as exc:
        raise HTTPException(
            status_code=409,
            detail={
                "message": str(exc),
                "type": "rubric_limit_reached",
                "code": "rubric_limit_reached",
                "limit": exc.limit,
            },
        ) from exc
    response.status_code = 201 if created else 200
    return _tenant_detail(stored)


@router.delete("/v1/evals/rubrics/{name}", status_code=204)
async def delete_rubric(name: str, identity: Identity = Depends(require_identity)) -> Response:
    if app_state.tenant_rubrics.delete(identity.tenant, name):
        return Response(status_code=204)
    if app_state.rubric_registry.get(name) is not None:
        raise _reserved(name)
    raise _unknown(name)


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
    _model_routing.reject_unsupported_governed_workload(
        identity=identity,
        workload="eval.run",
    )
    # The tenant's own rubric first: registration refuses built-in names, so
    # this only matters if a later release adds a built-in a tenant already
    # uses, and then the tenant keeps the judge it registered.
    stored = app_state.tenant_rubrics.get(identity.tenant, req.rubric)
    rubric = stored.spec if stored is not None else app_state.rubric_registry.get(req.rubric)
    if rubric is None:
        raise _unknown(req.rubric)

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
        outcome = await app_state.eval_runner.evaluate(
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
            temperature=req.temperature,
            n=req.n,
            admission=_admission(identity),
        )
    except (ContextLengthExceededError, GenerationTimeoutError, UpstreamGenerationError) as exc:
        # 400 context_length_exceeded / 504 generation_timeout / 502, the same
        # typed payloads chat completions answer with.
        _raise_generation_http_error(exc)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # The judge is this route's primary model, so its substitution is the one
    # the response headers report.
    if outcome.substitution is not None:
        model_substitution.bind(outcome.substitution)
    return EvalResponse(
        id=make_eval_id(),
        created=int(time.time()),
        rubric=rubric.name,
        rubric_source="tenant" if stored is not None else "builtin",
        rubric_digest=stored.digest if stored is not None else None,
        judge_model=outcome.judge_model,
        **model_substitution.fields(outcome.substitution),
        candidate_model=req.candidate_model,
        candidate_completion_id=req.candidate_completion_id,
        temperature=req.temperature,
        verdict=outcome.verdict,
        verdicts=list(outcome.verdicts),
        repeats=summarize_repeats(outcome.verdicts),
        duration_ms=round(outcome.duration_ms, 2),
    )


def _admission(identity: Identity):
    """One scheduler slot per judge call, so repeats interleave with other
    tenants' work instead of holding the model for all of them."""

    @asynccontextmanager
    async def admit(adapter: InferenceAdapter, model_name: str, estimated_tokens: int):
        lease = await acquire_slot(
            identity=identity,
            adapter=adapter,
            model_name=model_name,
            workload="eval.run",
            priority=_EVAL_PRIORITY,
            estimated_tokens=estimated_tokens,
        )
        try:
            yield scheduler_span_attrs(lease)
        finally:
            await app_state.scheduler.release(lease)

    return admit
