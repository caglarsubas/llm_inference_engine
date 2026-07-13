"""Payload-free status for the active signed model-routing policy."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from .model_routing_runtime import ModelRoutingRuntimeState


class ModelRoutingPolicyStatus(BaseModel):
    object: Literal["model_routing_policy.status"] = "model_routing_policy.status"
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


def build_model_routing_status(
    state: ModelRoutingRuntimeState,
    *,
    auth_enabled: bool,
) -> ModelRoutingPolicyStatus:
    active = state.policy
    if active is None:
        return ModelRoutingPolicyStatus(
            active=False,
            pricing_catalog_digest=(state.pricing.digest if state.pricing else None),
            pricing_model_count=(len(state.pricing.by_model) if state.pricing else 0),
        )

    claims = active.verified.claims
    return ModelRoutingPolicyStatus(
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
        org_binding_mode=("auth-key-org" if auth_enabled else "deployment-org"),
    )
