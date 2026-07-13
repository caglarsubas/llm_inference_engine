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
    ActivatedModelRoutingPolicy,
    ModelRoutingPolicyActivationError,
    activate_model_routing_policy_from_settings,
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


def _model_routing_status(
    active: ActivatedModelRoutingPolicy | None,
) -> ModelRoutingPolicyStatusResponse:
    if active is None:
        return ModelRoutingPolicyStatusResponse(active=False)
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
    return _model_routing_status(app_state.model_routing_policy)


@router.post(
    "/v1/admin/model-routing-policy:reload",
    response_model=ModelRoutingPolicyStatusResponse,
)
async def reload_model_routing_policy(
    identity: Identity = Depends(require_identity),
) -> ModelRoutingPolicyStatusResponse:
    """Verify and atomically activate candidate or last-known-good policy."""

    previous = app_state.model_routing_policy
    with span(
        "admin.model_routing_policy.reload",
        **{
            "prometa.tenant": identity.tenant,
            "prometa.key_id": identity.key_id,
            "model_routing.policy.previous_digest": (
                previous.digest if previous is not None else ""
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

        app_state.model_routing_policy = activated
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
            }
        )

    return _model_routing_status(activated)
