"""``/v1/admin/...`` — operator-facing endpoints.

Right now the only entry is ``policies:reload``, but anything Prometa's
control plane calls to mutate engine state belongs here (future: ``keys:reload``,
``models:warmup``, etc.).

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
    ModelRoutingRuntimeState,
    build_model_routing_runtime_state,
    load_model_routing_pricing_catalog,
)
from ..observability import get_logger, span
from .state import app_state

router = APIRouter()
log = get_logger("admin")


class PolicyReloadResponse(BaseModel):
    object: str = "policy.reload"
    reloaded_at: int
    policies_loaded: int
    source: str


class ModelRoutingPolicyStatusResponse(BaseModel):
    object: str = "model_routing_policy.status"
    active: bool
    policy_id: str | None = None
    revision: int | None = None
    digest: str | None = None
    source: str | None = None
    org_id: str | None = None
    environment: str | None = None
    release_id: str | None = None
    deployment_id: str | None = None
    offline_lease_expires_at: str | None = None
    candidate_error_code: str | None = None
    request_time_enforcement: bool = False
    route_count: int = 0
    rate_limit_scope: str | None = None
    pricing_catalog_digest: str | None = None
    pricing_model_count: int = 0
    org_binding_mode: str | None = None


def _model_routing_status(
    state: ModelRoutingRuntimeState,
) -> ModelRoutingPolicyStatusResponse:
    active = state.policy
    if active is None:
        return ModelRoutingPolicyStatusResponse(
            active=False,
            pricing_catalog_digest=(state.pricing.digest if state.pricing else None),
            pricing_model_count=(len(state.pricing.by_model) if state.pricing else 0),
        )
    claims = active.verified.claims
    return ModelRoutingPolicyStatusResponse(
        active=True,
        policy_id=claims.policy_id,
        revision=claims.revision,
        digest=active.digest,
        source=active.source,
        org_id=claims.org_id,
        environment=claims.target_environment,
        release_id=claims.release_id,
        deployment_id=claims.deployment_id,
        offline_lease_expires_at=claims.offline_lease_expires_at,
        candidate_error_code=active.candidate_error_code,
        request_time_enforcement=True,
        route_count=len(claims.routes),
        rate_limit_scope="process-replica",
        pricing_catalog_digest=(state.pricing.digest if state.pricing else None),
        pricing_model_count=(len(state.pricing.by_model) if state.pricing else 0),
        org_binding_mode=("auth-key-org" if settings.auth_enabled else "deployment-org"),
    )


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
    response_model=ModelRoutingPolicyStatusResponse,
)
async def model_routing_policy_status(
    identity: Identity = Depends(require_identity),
) -> ModelRoutingPolicyStatusResponse:
    del identity
    return _model_routing_status(app_state.model_routing_runtime)


@router.post(
    "/v1/admin/model-routing-policy:reload",
    response_model=ModelRoutingPolicyStatusResponse,
)
async def reload_model_routing_policy(
    identity: Identity = Depends(require_identity),
) -> ModelRoutingPolicyStatusResponse:
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

    return _model_routing_status(next_state)
