from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType

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
from inference_engine.model_routing_runtime import (
    LoadedModelRoutingPricingCatalog,
    ModelRoutingModelPrice,
    ModelRoutingPricingCatalog,
    ModelRoutingRuntimeConfigError,
    ModelRoutingRuntimeState,
)


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "model-routing-policy-v1.json"


@pytest.fixture(autouse=True)
def _explicit_auth_baseline(monkeypatch):
    previous = app_state.model_routing_runtime
    monkeypatch.setattr(settings, "auth_enabled", False)
    app_state.model_routing_runtime = ModelRoutingRuntimeState()
    yield
    app_state.model_routing_runtime = previous


def _pricing() -> LoadedModelRoutingPricingCatalog:
    prices = [
        ModelRoutingModelPrice(
            model="qwen3:32b",
            input_cost_micros_per_million_tokens=0,
            output_cost_micros_per_million_tokens=0,
        ),
        ModelRoutingModelPrice(
            model="llama3.3:70b:openrouter",
            input_cost_micros_per_million_tokens=100_000,
            output_cost_micros_per_million_tokens=100_000,
        ),
    ]
    catalog = ModelRoutingPricingCatalog(pricing_version=1, models=prices)
    return LoadedModelRoutingPricingCatalog(
        catalog=catalog,
        digest="sha256:pricing",
        by_model=MappingProxyType({price.model: price for price in prices}),
    )


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
        "request_time_enforcement": True,
        "route_count": 2,
        "rate_limit_scope": "process-replica",
        "pricing_catalog_digest": None,
        "pricing_model_count": 0,
        "org_binding_mode": "deployment-org",
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
    monkeypatch.setattr(
        admin_mod,
        "load_model_routing_pricing_catalog",
        lambda *args, **kwargs: _pricing(),
    )
    monkeypatch.setattr(settings, "model_routing_expected_org_id", "org-golden")

    response = TestClient(app).post("/v1/admin/model-routing-policy:reload")

    assert response.status_code == 200
    assert response.json()["digest"] == activated.digest
    assert response.json()["pricing_catalog_digest"] == "sha256:pricing"
    assert app_state.model_routing_policy is activated
    assert app_state.model_routing_pricing is not None


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


def test_runtime_config_failure_preserves_previous_atomic_state(monkeypatch) -> None:
    previous = ModelRoutingRuntimeState(policy=_active_policy(), pricing=_pricing())
    app_state.model_routing_runtime = previous
    monkeypatch.setattr(
        admin_mod,
        "activate_model_routing_policy_from_settings",
        _active_policy,
    )

    def fail_pricing(*args, **kwargs):
        raise ModelRoutingRuntimeConfigError("pricing_catalog_invalid")

    monkeypatch.setattr(admin_mod, "load_model_routing_pricing_catalog", fail_pricing)
    response = TestClient(app).post("/v1/admin/model-routing-policy:reload")

    assert response.status_code == 400
    assert response.json()["detail"]["type"] == "pricing_catalog_invalid"
    assert app_state.model_routing_runtime is previous


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
