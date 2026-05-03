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
