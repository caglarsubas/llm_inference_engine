from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from inference_engine import auth as auth_mod
from inference_engine.api import admin as admin_mod
from inference_engine.api.state import app_state
from inference_engine.config import settings
from inference_engine.main import app
from inference_engine.model_routing import (
    ActivatedModelRoutingPolicy,
    ModelRoutingPolicyActivationError,
    ModelRoutingPolicyEnvelope,
    ModelRoutingTrustStore,
    verify_model_routing_policy,
)


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "model-routing-policy-v1.json"


@pytest.fixture(autouse=True)
def _explicit_auth_baseline(monkeypatch):
    monkeypatch.setattr(settings, "auth_enabled", False)


def _active_policy(
    *, source: str = "candidate", candidate_error_code: str | None = None
) -> ActivatedModelRoutingPolicy:
    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    envelope = ModelRoutingPolicyEnvelope.model_validate(fixture["policy"], strict=True)
    trust = ModelRoutingTrustStore.model_validate(
        {
            "trustVersion": 1,
            "entries": [fixture["trust"]],
            "revokedKeyIds": [],
            "revokedJtis": [],
        },
        strict=True,
    )
    verified = verify_model_routing_policy(
        envelope,
        trust,
        now=datetime(2026, 7, 13, 0, 10, tzinfo=UTC),
        expected_environment="staging",
        expected_org_id="org-golden",
    )
    return ActivatedModelRoutingPolicy(
        verified=verified,
        source=source,
        candidate_error_code=candidate_error_code,
    )


def test_status_is_payload_free_and_reports_active_identity() -> None:
    app_state.model_routing_policy = _active_policy(
        source="last-known-good",
        candidate_error_code="invalid_signature",
    )

    response = TestClient(app).get("/v1/admin/model-routing-policy")

    assert response.status_code == 200
    assert response.json() == {
        "object": "model_routing_policy.status",
        "active": True,
        "policy_id": "routing-golden-v1",
        "revision": 1,
        "digest": "sha256:b320a77f8c2a14916c0776a051eca6be614fbdb52ac6854783e651680c6973be",
        "source": "last-known-good",
        "org_id": "org-golden",
        "environment": "staging",
        "release_id": "release-golden-model-v1",
        "deployment_id": "model-plane-golden-v1",
        "offline_lease_expires_at": "2026-07-13T00:30:00.000Z",
        "candidate_error_code": "invalid_signature",
    }


def test_status_reports_disabled_without_claiming_policy() -> None:
    app_state.model_routing_policy = None
    response = TestClient(app).get("/v1/admin/model-routing-policy")
    assert response.status_code == 200
    assert response.json()["active"] is False
    assert response.json()["digest"] is None


def test_reload_atomically_replaces_active_policy(monkeypatch) -> None:
    app_state.model_routing_policy = None
    activated = _active_policy()
    monkeypatch.setattr(
        admin_mod,
        "activate_model_routing_policy_from_settings",
        lambda: activated,
    )

    response = TestClient(app).post("/v1/admin/model-routing-policy:reload")

    assert response.status_code == 200
    assert response.json()["digest"] == activated.digest
    assert app_state.model_routing_policy is activated


def test_failed_reload_preserves_previous_policy(monkeypatch) -> None:
    previous = _active_policy()
    app_state.model_routing_policy = previous

    def fail():
        raise ModelRoutingPolicyActivationError(
            "no_valid_policy",
            candidate_error_code="invalid_signature",
            last_known_good_error_code="offline_lease_expired",
        )

    monkeypatch.setattr(admin_mod, "activate_model_routing_policy_from_settings", fail)
    response = TestClient(app).post("/v1/admin/model-routing-policy:reload")

    assert response.status_code == 400
    assert response.json()["detail"] == {
        "message": "model routing policy reload failed",
        "type": "no_valid_policy",
        "candidate_error_code": "invalid_signature",
        "last_known_good_error_code": "offline_lease_expired",
    }
    assert app_state.model_routing_policy is previous


def test_model_routing_admin_routes_require_bearer_when_auth_enabled(
    monkeypatch,
) -> None:
    monkeypatch.setattr(settings, "auth_enabled", True)
    auth_mod._set_keys_for_tests([("sk-model-admin", "ops")])
    app_state.model_routing_policy = _active_policy()
    client = TestClient(app)

    assert client.get("/v1/admin/model-routing-policy").status_code == 401
    response = client.get(
        "/v1/admin/model-routing-policy",
        headers={"Authorization": "Bearer sk-model-admin"},
    )
    assert response.status_code == 200
    assert response.json()["active"] is True
