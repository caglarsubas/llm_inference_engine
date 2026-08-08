"""Bind API-layer objects onto the request's ``prometa.model-usage.v1`` record.

Mirrors the ``_fallback.span_attrs()`` idiom: routes hand over the objects they
already hold and this module owns the mapping onto ledger fields. Keeping it
here is what lets ``usage_ledger`` stay free of ``Identity``,
``ModelRoutingDecision``, ``InferenceAdapter``, and ``app_state``.

Every mapping step is wrapped in ``_never_raises``. The ledger is a side
channel: a bug in this mapping must degrade the invoice, never the completion.
Degrading means a record with fewer fields set — never no record, so a failing
step must not sit in the same swallowed body as the one that declares the
request billable.
"""

from __future__ import annotations

import functools
from collections.abc import Callable

from .. import usage_ledger
from ..adapters import InferenceAdapter
from ..auth import Identity
from ..model_routing_runtime import ModelRoutingDecision, usage_cost_micros
from ..observability import get_logger
from . import _fallback
from .state import app_state

log = get_logger("api.usage")

# ``requested_model`` is whatever the caller typed. The record is a persisted,
# deliberately un-sampled billing artefact, so it gets the same bound the
# request-id header applies to a caller-controlled string.
_MAX_REQUESTED_MODEL_CHARS = 200


def _never_raises(fn: Callable) -> Callable:
    @functools.wraps(fn)
    def wrapper(*args, **kwargs) -> None:
        try:
            fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - the ledger must not fail a request
            log.warning(
                "usage_ledger.bind_failed",
                binder=fn.__name__,
                error_type=type(exc).__name__,
            )

    return wrapper


def bind_request(
    *,
    identity: Identity,
    requested_model: str,
    operation: str,
    stream: bool,
    input_token_upper_bound: int | None,
    output_token_budget: int,
) -> None:
    """Attribute the record and declare it billable.

    This is the one seam a request reaches only with a resolved identity and a
    body the schema accepted, which is exactly the pair ``mark_billable``
    requires. It runs before enforcement, so a denial is still billed — a
    denial is an answered request, unlike everything rejected ahead of here.

    The two steps are separately wrapped, and billability is declared first, so
    a failure while mapping the attribution costs those fields and nothing
    else. Sharing one wrapped body would let such a failure skip
    ``mark_billable`` and make ``flush`` discard the whole invoice line.
    """
    _mark_billable()
    _bind_attribution(
        identity=identity,
        requested_model=requested_model,
        operation=operation,
        stream=stream,
        input_token_upper_bound=input_token_upper_bound,
        output_token_budget=output_token_budget,
    )


@_never_raises
def _mark_billable() -> None:
    usage_ledger.mark_billable()


@_never_raises
def _bind_attribution(
    *,
    identity: Identity,
    requested_model: str,
    operation: str,
    stream: bool,
    input_token_upper_bound: int | None,
    output_token_budget: int,
) -> None:
    usage_ledger.bind(
        tenant=identity.tenant,
        key_id=identity.key_id,
        org_id=identity.org_id,
        requested_model=requested_model[:_MAX_REQUESTED_MODEL_CHARS],
        operation=operation,
        stream=stream,
        input_token_upper_bound=input_token_upper_bound,
        output_token_budget=output_token_budget,
    )


@_never_raises
def bind_decision(decision: ModelRoutingDecision | None) -> None:
    if decision is None:
        return
    usage_ledger.bind(
        policy_id=decision.active.policy_id,
        policy_digest=decision.active.digest,
        route_id=decision.route.route_id,
        pricing_digest=decision.pricing_digest,
    )


@_never_raises
def bind_denial(*, code: str, policy_id: str | None, route_id: str | None) -> None:
    usage_ledger.bind(denial_code=code)
    if policy_id is not None:
        usage_ledger.bind(policy_id=policy_id)
    if route_id is not None:
        usage_ledger.bind(route_id=route_id)


@_never_raises
def bind_serving(
    *,
    adapter: InferenceAdapter,
    model_name: str,
    fallback_info: _fallback.FallbackInfo | None,
) -> None:
    """Record which model is being attempted, plus its trace ids.

    Bound on entry rather than on success so a request that errors still says
    which model it hit. ``fallback_info=None`` leaves the fallback fields
    untouched — the structured-output retry re-enters the same seam without
    one, and it must not erase a fallback the first attempt established.
    """
    usage_ledger.bind(
        resolved_model=model_name,
        backend=adapter.backend_name,
        request_key_source=_fallback.request_key_source(adapter),
    )
    usage_ledger.capture_trace_context()
    if fallback_info is None:
        return
    usage_ledger.bind(
        fallback=True,
        fallback_from_model=fallback_info.from_model,
        fallback_from_backend=fallback_info.from_backend,
        fallback_reason=fallback_info.reason,
    )


@_never_raises
def bind_usage(
    *,
    input_tokens: int,
    output_tokens: int,
    cached_tokens: int = 0,
    ttft_ms: float | None = None,
    finish_reason: str | None = None,
    outcome: str | None = None,
) -> None:
    record = usage_ledger.current()
    if record is None:
        return
    usage_ledger.bind(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_tokens=cached_tokens,
        cost_micros=usage_cost_micros(
            app_state.model_routing_pricing,
            model=record.resolved_model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        ),
    )
    if ttft_ms is not None:
        usage_ledger.bind(ttft_ms=ttft_ms)
    if finish_reason is not None:
        usage_ledger.bind(finish_reason=finish_reason)
    if outcome is not None:
        usage_ledger.bind(outcome=outcome)


@_never_raises
def bind_error(exc: BaseException) -> None:
    usage_ledger.bind(error_type=_error_type(exc))


@_never_raises
def bind_error_type(error_type: str) -> None:
    usage_ledger.bind(error_type=error_type)


def _error_type(exc: BaseException) -> str:
    """Prefer the typed code a client would branch on over the class name."""
    detail = getattr(exc, "detail", None)
    if isinstance(detail, dict) and isinstance(detail.get("type"), str):
        return detail["type"]
    typed = getattr(exc, "error_detail", None)
    if callable(typed):
        payload = typed()
        if isinstance(payload, dict) and isinstance(payload.get("type"), str):
            return payload["type"]
    return type(exc).__name__


__all__ = [
    "bind_decision",
    "bind_denial",
    "bind_error",
    "bind_error_type",
    "bind_request",
    "bind_serving",
    "bind_usage",
]
