"""API integration for locally enforced signed model-routing policy."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass

from fastapi import HTTPException

from ..adapters import InferenceAdapter
from ..auth import Identity
from ..config import CERTIFIED_MODEL_WORKLOAD_SURFACE, settings
from ..manager import ModelNotFoundError
from ..model_plane_control import QUARANTINED_ERROR_CODE, RuntimeControlRefusal
from ..model_routing_runtime import (
    ModelRoutingDecision,
    ModelRoutingEnforcementError,
    enforce_model_routing_request,
    model_routing_policy_identity_attrs,
    model_routing_span_attrs,
    observe_model_routing_usage,
    retain_model_routing_reservation,
    settle_model_routing_reservation,
    usage_cost_micros,
)
from ..observability import get_logger, span
from ..schemas import (
    ChatCompletionRequest,
    ChatImageUrlContentPart,
    CompletionRequest,
)
from . import _fallback, _usage
from .state import app_state


log = get_logger("api.model_routing")


@dataclass(frozen=True)
class ResolvedRoutingCandidate:
    adapter: InferenceAdapter
    model_name: str
    candidate_index: int | None
    fallback_info: _fallback.FallbackInfo | None = None


def _identity_attrs(identity: Identity) -> dict:
    attrs = {
        "prometa.tenant": identity.tenant,
        "prometa.key_id": identity.key_id,
    }
    if identity.org_id is not None:
        attrs["prometa.org_id"] = identity.org_id
    return attrs


def chat_input_token_upper_bound(req: ChatCompletionRequest) -> int | None:
    for message in req.messages:
        if not isinstance(message.content, list):
            continue
        for part in message.content:
            if isinstance(part, ChatImageUrlContentPart) and not part.image_url.url.startswith(
                "data:"
            ):
                return None

    model_bound = {
        "messages": [message.model_dump(exclude_none=True) for message in req.messages],
        "tools": [tool.model_dump(exclude_none=True) for tool in req.tools or []],
        "tool_choice": req.tool_choice,
        "response_format": (
            req.response_format.model_dump(exclude_none=True) if req.response_format else None
        ),
        "chat_template_kwargs": req.chat_template_kwargs,
        "stop": req.stop,
    }
    encoded = json.dumps(
        model_bound,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return len(encoded) + settings.model_routing_input_token_reserve


def completion_input_token_upper_bound(req: CompletionRequest) -> int:
    prompts = [req.prompt] if isinstance(req.prompt, str) else list(req.prompt)
    encoded = json.dumps(
        {"prompt": prompts, "stop": req.stop},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return len(encoded) + settings.model_routing_input_token_reserve


def embedding_input_token_upper_bound(inputs: list[str]) -> int:
    encoded = json.dumps(
        {"input": inputs},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return len(encoded) + settings.model_routing_input_token_reserve


def _enforcement_http_error(exc: ModelRoutingEnforcementError) -> HTTPException:
    if exc.code in {
        "rate_limit_exceeded",
        "token_rate_limit_exceeded",
        "budget_exceeded",
    }:
        status_code = 429
    elif exc.code in {
        "org_identity_missing",
        "org_identity_mismatch",
        "route_not_allowed",
    }:
        status_code = 403
    elif exc.code in {
        "policy_not_yet_valid",
        "policy_expired",
        "policy_offline_lease_expired",
        "pricing_catalog_unavailable",
        "rate_limit_backend_unavailable",
        "rate_limit_state_capacity",
    }:
        status_code = 503
    else:
        status_code = 400

    detail = {
        "message": "model routing policy denied request",
        "type": exc.code,
        "code": exc.code,
        "policy_id": exc.policy_id,
        "route_id": exc.route_id,
    }
    headers: dict[str, str] = {}
    if exc.retry_after_seconds is not None:
        detail["retry_after_seconds"] = exc.retry_after_seconds
        headers["Retry-After"] = str(exc.retry_after_seconds)
    # OpenAI's SDKs read these to schedule their own backoff instead of
    # retrying blind. ``remaining`` is 0 by construction — we only get here
    # because that window is full. Each trio is emitted only for a denial on
    # the window it describes: the spend window has no OpenAI header dimension,
    # so a budget denial carries Retry-After and nothing else, rather than
    # telling an SDK that a request window it never exhausted has closed.
    if exc.code == "rate_limit_exceeded":
        headers["x-ratelimit-remaining-requests"] = "0"
        if exc.limit_requests is not None:
            headers["x-ratelimit-limit-requests"] = str(exc.limit_requests)
        if exc.retry_after_seconds is not None:
            headers["x-ratelimit-reset-requests"] = f"{exc.retry_after_seconds}s"
    elif exc.code == "token_rate_limit_exceeded":
        headers["x-ratelimit-remaining-tokens"] = "0"
        if exc.limit_tokens is not None:
            headers["x-ratelimit-limit-tokens"] = str(exc.limit_tokens)
        if exc.retry_after_seconds is not None:
            headers["x-ratelimit-reset-tokens"] = f"{exc.retry_after_seconds}s"
    return HTTPException(
        status_code=status_code, detail=detail, headers=headers or None
    )


def _emit_denial_span(
    *,
    identity: Identity,
    code: str,
    route_id: str | None = None,
    workload: str | None = None,
) -> None:
    active = app_state.model_routing_runtime.policy
    attrs: dict = {
        "model_routing.enforced": active is not None,
        "model_routing.decision": "deny",
        "model_routing.denial.code": code,
        "model_routing.rate_limit.scope": app_state.model_routing_rate_limiter.scope,
        **_identity_attrs(identity),
    }
    if active is not None:
        attrs.update(model_routing_policy_identity_attrs(active))
    if route_id is not None:
        attrs["model_routing.route.id"] = route_id
    if workload is not None:
        attrs["model_routing.denial.workload"] = workload
    _usage.bind_denial(
        code=code,
        policy_id=active.policy_id if active is not None else None,
        route_id=route_id,
    )
    with span("model.routing.decision", **attrs):
        pass


def _runtime_control_http_error(refusal: RuntimeControlRefusal) -> HTTPException:
    """Say that a control applies and at which scope, and nothing more.

    The lease id and its revision identify control-plane state to a caller who
    may only be one of many tenants behind this replica, so they stay on the
    authenticated admin status and in the local log.
    """
    detail = {
        "message": (
            "runtime control has quarantined this subject"
            if refusal.code == QUARANTINED_ERROR_CODE
            else "a matched runtime control is stale and this deployment stops rather than serve it"
        ),
        "type": refusal.code,
        "code": refusal.code,
        "scope": refusal.scope,
    }
    return HTTPException(status_code=503, detail=detail)


def enforce_runtime_control(*, identity: Identity) -> None:
    """Refuse a quarantined subject from cache, before any admission work.

    Runs ahead of routing enforcement so a refused request never takes a
    rate-limit reservation it will not use. The check is a few comparisons
    against the cached lease projection; the lease's signature was verified
    when it arrived on the observer's reply, not here.
    """
    control = app_state.model_plane_runtime_control
    if control is None:
        return
    refusal = control.evaluate(tenant=identity.tenant)
    if refusal is None:
        return
    _emit_denial_span(identity=identity, code=refusal.code)
    raise _runtime_control_http_error(refusal)


async def enforce_generation_request(
    *,
    identity: Identity,
    requested_model: str,
    input_token_upper_bound: int | None,
    output_token_budget: int,
) -> ModelRoutingDecision | None:
    enforce_runtime_control(identity=identity)
    try:
        decision = await asyncio.to_thread(
            enforce_model_routing_request,
            app_state.model_routing_runtime,
            identity=identity,
            requested_model=requested_model,
            input_token_upper_bound=input_token_upper_bound,
            output_token_budget=output_token_budget,
            rate_limiter=app_state.model_routing_rate_limiter,
            clock_skew_seconds=settings.model_routing_clock_skew_seconds,
        )
    except ModelRoutingEnforcementError as exc:
        _emit_denial_span(
            identity=identity,
            code=exc.code,
            route_id=exc.route_id,
        )
        raise _enforcement_http_error(exc) from exc
    _usage.bind_decision(decision)
    return decision


def observe_usage(
    decision: ModelRoutingDecision | None,
    *,
    model: str | None,
    input_tokens: int,
    output_tokens: int,
) -> None:
    """Hand the request's real usage to its admission reservation.

    Purely local — no store round trip — so it is safe to call from inside a
    generation span. ``settle_reservation`` is what commits it.
    """
    if decision is None or decision.reservation is None:
        return
    observe_model_routing_usage(
        decision.reservation,
        tokens=input_tokens + output_tokens,
        cost_micros=usage_cost_micros(
            app_state.model_routing_pricing,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        ),
    )


def retain_reservation(decision: ModelRoutingDecision | None) -> None:
    """Keep the admission reserve for a request that vanished mid-flight.

    Also purely local; ``settle_reservation`` is still what commits it.
    """
    if decision is None or decision.reservation is None:
        return
    retain_model_routing_reservation(decision.reservation)


async def settle_reservation(decision: ModelRoutingDecision | None) -> None:
    """Close out the admission reservation. Idempotent, and never raises.

    Call it from a ``finally`` on every path that reached enforcement. A store
    failure here is logged and swallowed: the request has already been served,
    and the unsettled reservation stands at its conservative admission value
    until its window slides past it — restrictive, never permissive.
    """
    if decision is None or decision.reservation is None:
        return
    try:
        await asyncio.to_thread(
            settle_model_routing_reservation,
            app_state.model_routing_rate_limiter,
            decision.reservation,
        )
    except Exception as exc:  # noqa: BLE001 - settling must not fail a served request
        log.warning(
            "model_routing.reservation_settle_failed",
            policy_id=decision.active.policy_id,
            route_id=decision.route.route_id,
            error_type=type(exc).__name__,
        )


def reject_unsupported_governed_workload(
    *,
    identity: Identity,
    workload: str,
) -> None:
    active = app_state.model_routing_runtime.policy
    certified_surface = (
        settings.model_plane_workload_surface == CERTIFIED_MODEL_WORKLOAD_SURFACE
    )
    if active is None and not certified_surface:
        return
    code = (
        "model_plane_workload_not_certified"
        if certified_surface
        else "model_routing_workload_not_integrated"
    )
    _emit_denial_span(identity=identity, code=code, workload=workload)
    detail = {
        "message": (
            "workload is outside the configured model-plane workload surface"
            if certified_surface
            else "workload is not integrated with governed model routing"
        ),
        "type": code,
        "code": code,
        "workload": workload,
    }
    if active is not None:
        detail["policy_id"] = active.policy_id
    if certified_surface:
        detail["workload_surface"] = CERTIFIED_MODEL_WORKLOAD_SURFACE
    raise HTTPException(
        status_code=503,
        detail=detail,
    )


async def resolve_initial_candidate(
    *,
    requested_model: str,
    decision: ModelRoutingDecision | None,
    identity: Identity,
    extra_span_attrs: dict | None = None,
) -> ResolvedRoutingCandidate:
    candidates = decision.candidate_models if decision is not None else (requested_model,)
    first_failed_model: str | None = None
    first_failure_reason: str | None = None
    first_error_type: str | None = None

    with span(
        "model.routing.decision",
        **_identity_attrs(identity),
        **model_routing_span_attrs(decision),
        **(extra_span_attrs or {}),
    ) as routing_span:
        for index, candidate in enumerate(candidates):
            try:
                with span(
                    "model.acquire",
                    model=candidate,
                    **_identity_attrs(identity),
                    **model_routing_span_attrs(
                        decision,
                        candidate_model=candidate,
                        candidate_index=(index if decision is not None else None),
                    ),
                    **(extra_span_attrs or {}),
                ) as acquire_span:
                    adapter, descriptor = await app_state.manager.get(candidate)
                    acquire_span.bind(
                        **{
                            "llm.request.key_source": _fallback.request_key_source(adapter),
                        }
                    )
            except ModelNotFoundError:
                if first_failed_model is None:
                    first_failed_model = candidate
                    first_failure_reason = "model_unavailable"
                    first_error_type = "ModelNotFoundError"
                continue
            except Exception as exc:
                if decision is None:
                    raise
                if first_failed_model is None:
                    first_failed_model = candidate
                    first_failure_reason = "model_acquire_error"
                    first_error_type = exc.__class__.__name__
                continue

            fallback_info = None
            if index > 0 and first_failed_model is not None:
                fallback_info = _fallback.FallbackInfo(
                    from_model=first_failed_model,
                    from_backend="unavailable",
                    reason=first_failure_reason or "model_unavailable",
                    error_type=first_error_type or "ModelNotFoundError",
                )
            routing_span.bind(
                **model_routing_span_attrs(
                    decision,
                    candidate_model=descriptor.qualified_name,
                    candidate_index=(index if decision is not None else None),
                ),
                **{"model_routing.route.initial_fallback": index > 0},
            )
            return ResolvedRoutingCandidate(
                adapter=adapter,
                model_name=descriptor.qualified_name,
                candidate_index=(index if decision is not None else None),
                fallback_info=fallback_info,
            )

        routing_span.bind(**{"error.type": "model_route_unavailable"})

    if decision is None:
        _usage.bind_error_type("model_not_found")
        raise HTTPException(
            status_code=404,
            detail=f"model not found: {requested_model!r}",
        )
    _usage.bind_error_type("model_route_unavailable")
    raise HTTPException(
        status_code=503,
        detail={
            "message": "no model in the signed route is available",
            "type": "model_route_unavailable",
            "code": "model_route_unavailable",
            "policy_id": decision.active.policy_id,
            "route_id": decision.route.route_id,
        },
    )


async def resolve_next_fallback(
    *,
    decision: ModelRoutingDecision | None,
    current_candidate_index: int | None,
    adapter: InferenceAdapter,
    model_name: str,
    exc: Exception,
    identity: Identity,
    extra_span_attrs: dict | None = None,
) -> ResolvedRoutingCandidate | None:
    if decision is None:
        fallback = await _fallback.resolve_openrouter_fallback(
            adapter=adapter,
            model_name=model_name,
            exc=exc,
            identity=identity,
            intent_attrs=extra_span_attrs,
        )
        if fallback is None:
            return None
        fallback_adapter, fallback_model_name, fallback_info = fallback
        return ResolvedRoutingCandidate(
            adapter=fallback_adapter,
            model_name=fallback_model_name,
            candidate_index=None,
            fallback_info=fallback_info,
        )

    start = (current_candidate_index if current_candidate_index is not None else -1) + 1
    reason, error_type = _fallback.classify_error(exc)
    fallback_info = _fallback.FallbackInfo(
        from_model=model_name,
        from_backend=adapter.backend_name,
        reason=reason,
        error_type=error_type,
    )
    for index in range(start, len(decision.candidate_models)):
        candidate = decision.candidate_models[index]
        try:
            with span(
                "model.fallback.acquire",
                model=candidate,
                **_fallback.span_attrs(fallback_info),
                **_identity_attrs(identity),
                **model_routing_span_attrs(
                    decision,
                    candidate_model=candidate,
                    candidate_index=index,
                ),
                **(extra_span_attrs or {}),
            ) as acquire_span:
                fallback_adapter, descriptor = await app_state.manager.get(candidate)
                acquire_span.bind(
                    **{
                        "llm.request.key_source": _fallback.request_key_source(fallback_adapter),
                        "llm.fallback.to_model": descriptor.qualified_name,
                        "llm.fallback.to_backend": fallback_adapter.backend_name,
                    }
                )
        except ModelNotFoundError:
            fallback_info = _fallback.FallbackInfo(
                from_model=candidate,
                from_backend="unavailable",
                reason="model_unavailable",
                error_type="ModelNotFoundError",
            )
            continue
        except Exception as acquire_exc:
            fallback_info = _fallback.FallbackInfo(
                from_model=candidate,
                from_backend="unavailable",
                reason="model_acquire_error",
                error_type=acquire_exc.__class__.__name__,
            )
            continue
        return ResolvedRoutingCandidate(
            adapter=fallback_adapter,
            model_name=descriptor.qualified_name,
            candidate_index=index,
            fallback_info=fallback_info,
        )
    return None


__all__ = [
    "ModelRoutingDecision",
    "ResolvedRoutingCandidate",
    "chat_input_token_upper_bound",
    "completion_input_token_upper_bound",
    "embedding_input_token_upper_bound",
    "enforce_generation_request",
    "enforce_runtime_control",
    "model_routing_span_attrs",
    "observe_usage",
    "reject_unsupported_governed_workload",
    "retain_reservation",
    "resolve_initial_candidate",
    "resolve_next_fallback",
    "settle_reservation",
]
