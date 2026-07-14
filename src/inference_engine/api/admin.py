"""``/v1/admin/...`` — operator-facing endpoints.

Anything that mutates operator-owned engine state belongs here (future:
``keys:reload``, ``models:warmup``, etc.). Prometa may issue desired artifacts,
but tenant automation remains responsible for mounting them and invoking these
endpoints.

All admin endpoints are gated by ``require_identity``. When ``AUTH_ENABLED``
is on, that means a valid bearer key; when it's off, callers resolve to
``Identity(tenant="anonymous")`` — fine for local dev, exposed in production
only behind the API gateway's allowlist.
"""

from __future__ import annotations

import time

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from ..auth import Identity, require_identity
from ..config import settings
from ..evals import load_policy
from ..model_routing import (
    ModelRoutingPolicyActivationError,
    activate_model_routing_policy_from_settings,
)
from ..model_routing_runtime import (
    ModelRoutingRuntimeConfigError,
    build_model_routing_runtime_state,
    load_model_routing_pricing_catalog,
)
from ..model_routing_status import ModelRoutingPolicyStatus, build_model_routing_status
from ..model_plane_observer import ModelPlaneObserverStatus
from ..observability import get_logger, span
from .state import app_state

router = APIRouter()
log = get_logger("admin")


class PolicyReloadResponse(BaseModel):
    object: str = "policy.reload"
    reloaded_at: int
    policies_loaded: int
    source: str


@router.post("/v1/admin/policies:reload", response_model=PolicyReloadResponse)
async def reload_policies(identity: Identity = Depends(require_identity)) -> PolicyReloadResponse:
    """Re-read the auto-eval policy file and atomically swap the registry.

    The file is parsed strictly (same code path as startup) — a malformed
    file fails loudly with HTTP 400 and the existing policy stays in place.
    On success the new registry replaces the old one in a single attribute
    assignment; in-flight requests that already resolved through the old
    registry continue to use it (the resolver returns by value, not by
    reference to the registry object).
    """
    with span(
        "admin.policies.reload",
        **{
            "prometa.tenant": identity.tenant,
            "prometa.key_id": identity.key_id,
            "policy.source": str(settings.auto_eval_policies_file),
        },
    ) as s:
        try:
            new_registry = load_policy(settings.auto_eval_policies_file)
        except (ValueError, FileNotFoundError) as exc:
            # ValueError covers our own validation errors; FileNotFoundError
            # only happens when AUTH-style strict mode is configured (we
            # currently don't enforce file presence in load_policy, but if a
            # future tightening adds it the error type stays clean).
            log.error("admin.policies.reload_failed", error=str(exc))
            s.bind(**{"policy.reload.error": str(exc)})
            raise HTTPException(status_code=400, detail=f"policy reload failed: {exc}") from exc

        previous_count = len(app_state.policy_registry)
        app_state.policy_registry = new_registry
        loaded = len(new_registry)
        s.bind(
            **{
                "policy.previous_count": previous_count,
                "policy.loaded_count": loaded,
            }
        )

    log.info(
        "admin.policies.reloaded",
        previous=previous_count,
        loaded=loaded,
        tenant=identity.tenant,
    )

    return PolicyReloadResponse(
        reloaded_at=int(time.time()),
        policies_loaded=loaded,
        source=str(settings.auto_eval_policies_file),
    )


@router.get(
    "/v1/admin/model-routing-policy",
    response_model=ModelRoutingPolicyStatus,
)
async def model_routing_policy_status(
    identity: Identity = Depends(require_identity),
) -> ModelRoutingPolicyStatus:
    del identity
    return build_model_routing_status(
        app_state.model_routing_runtime,
        auth_enabled=settings.auth_enabled,
    )


@router.get(
    "/v1/admin/model-plane-observer",
    response_model=ModelPlaneObserverStatus,
)
async def model_plane_observer_status(
    identity: Identity = Depends(require_identity),
) -> ModelPlaneObserverStatus:
    del identity
    observer = app_state.model_plane_observer
    if observer is None:
        return ModelPlaneObserverStatus(enabled=False)
    return observer.status()


@router.post(
    "/v1/admin/model-routing-policy:reload",
    response_model=ModelRoutingPolicyStatus,
)
async def reload_model_routing_policy(
    identity: Identity = Depends(require_identity),
) -> ModelRoutingPolicyStatus:
    """Verify and atomically activate candidate or last-known-good policy."""

    previous_state = app_state.model_routing_runtime
    previous = previous_state.policy
    with span(
        "admin.model_routing_policy.reload",
        **{
            "prometa.tenant": identity.tenant,
            "prometa.key_id": identity.key_id,
            "model_routing.policy.previous_digest": (
                previous.digest if previous is not None else ""
            ),
            "model_routing.pricing.previous_digest": (
                previous_state.pricing.digest if previous_state.pricing is not None else ""
            ),
        },
    ) as s:
        try:
            activated = activate_model_routing_policy_from_settings()
        except ModelRoutingPolicyActivationError as exc:
            log.error(
                "admin.model_routing_policy.reload_failed",
                error_code=exc.code,
                candidate_error_code=exc.candidate_error_code,
                last_known_good_error_code=exc.last_known_good_error_code,
            )
            s.bind(
                **{
                    "model_routing.policy.reload.error_code": exc.code,
                    "model_routing.policy.reload.candidate_error_code": (
                        exc.candidate_error_code or ""
                    ),
                    "model_routing.policy.reload.last_known_good_error_code": (
                        exc.last_known_good_error_code or ""
                    ),
                }
            )
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "model routing policy reload failed",
                    "type": exc.code,
                    "candidate_error_code": exc.candidate_error_code,
                    "last_known_good_error_code": exc.last_known_good_error_code,
                },
            ) from exc

        try:
            pricing = load_model_routing_pricing_catalog(
                settings.model_routing_pricing_file,
                max_bytes=settings.model_routing_max_file_bytes,
            )
            next_state = build_model_routing_runtime_state(
                activated,
                pricing,
                auth_enabled=settings.auth_enabled,
                expected_org_id=settings.model_routing_expected_org_id,
            )
        except ModelRoutingRuntimeConfigError as exc:
            log.error(
                "admin.model_routing_policy.runtime_config_failed",
                error_code=exc.code,
            )
            s.bind(**{"model_routing.policy.reload.error_code": exc.code})
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "model routing policy runtime configuration failed",
                    "type": exc.code,
                },
            ) from exc

        app_state.model_routing_runtime = next_state
        s.bind(
            **{
                "model_routing.policy.active": activated is not None,
                "model_routing.policy.digest": (activated.digest if activated is not None else ""),
                "model_routing.policy.revision": (
                    activated.revision if activated is not None else 0
                ),
                "model_routing.policy.source": (
                    activated.source if activated is not None else "disabled"
                ),
                "model_routing.pricing.digest": (
                    next_state.pricing.digest if next_state.pricing is not None else ""
                ),
            }
        )

    return build_model_routing_status(
        next_state,
        auth_enabled=settings.auth_enabled,
    )


@router.post(
    "/v1/admin/model-routing-pricing:reload",
    response_model=ModelRoutingPolicyStatus,
)
async def reload_model_routing_pricing(
    identity: Identity = Depends(require_identity),
) -> ModelRoutingPolicyStatus:
    """Atomically replace pricing while preserving the active policy."""

    previous_state = app_state.model_routing_runtime
    previous_policy = previous_state.policy
    previous_pricing = previous_state.pricing
    with span(
        "admin.model_routing_pricing.reload",
        **{
            "prometa.tenant": identity.tenant,
            "prometa.key_id": identity.key_id,
            "model_routing.policy.digest": (
                previous_policy.digest if previous_policy is not None else ""
            ),
            "model_routing.pricing.previous_digest": (
                previous_pricing.digest if previous_pricing is not None else ""
            ),
        },
    ) as s:
        try:
            if previous_policy is None:
                raise ModelRoutingRuntimeConfigError("model_routing_policy_required")
            pricing = load_model_routing_pricing_catalog(
                settings.model_routing_pricing_file,
                max_bytes=settings.model_routing_max_file_bytes,
            )
            if pricing is None:
                raise ModelRoutingRuntimeConfigError("pricing_catalog_required")
            next_state = build_model_routing_runtime_state(
                previous_policy,
                pricing,
                auth_enabled=settings.auth_enabled,
                expected_org_id=settings.model_routing_expected_org_id,
            )
        except ModelRoutingRuntimeConfigError as exc:
            log.error(
                "admin.model_routing_pricing.reload_failed",
                error_code=exc.code,
            )
            s.bind(**{"model_routing.pricing.reload.error_code": exc.code})
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "model routing pricing reload failed",
                    "type": exc.code,
                },
            ) from exc

        app_state.model_routing_runtime = next_state
        s.bind(
            **{
                "model_routing.pricing.digest": pricing.digest,
                "model_routing.pricing.model_count": len(pricing.by_model),
            }
        )

    log.info(
        "admin.model_routing_pricing.reloaded",
        tenant=identity.tenant,
        policy_digest=previous_policy.digest,
        pricing_digest=pricing.digest,
        model_count=len(pricing.by_model),
    )
    return build_model_routing_status(
        next_state,
        auth_enabled=settings.auth_enabled,
    )
