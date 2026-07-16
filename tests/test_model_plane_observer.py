from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from inference_engine.api.state import app_state
from inference_engine.config import Settings, settings
from inference_engine.main import _run_observer_after_startup, app
from inference_engine.model_plane_observer import (
    ModelPlaneObservationConfig,
    ModelPlaneObservationConfigError,
    ModelPlaneObservationReporter,
    build_model_plane_observation,
    load_model_plane_observation_config,
    model_plane_observation_span_attrs,
    model_inventory_summary,
    model_routing_inventory_summary,
    read_model_plane_observation_api_key,
)
from inference_engine.model_routing import (
    ActivatedModelRoutingPolicy,
    ModelRoutingPolicyEnvelope,
    ModelRoutingTrustStore,
    verify_model_routing_policy,
)
from inference_engine.model_routing_runtime import ModelRoutingRateLimiter, ModelRoutingRuntimeState
from inference_engine.model_routing_status import ModelRoutingPolicyStatus
from inference_engine.schemas import ModelInfo, ModelList, UnavailableModel


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "model-routing-policy-v1.json"


def active_policy() -> ActivatedModelRoutingPolicy:
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
    return ActivatedModelRoutingPolicy(verified=verified, source="candidate")


class FakeState:
    def __init__(
        self,
        *,
        ready: bool = True,
        policy: ActivatedModelRoutingPolicy | None = None,
    ) -> None:
        self.model_routing_runtime = ModelRoutingRuntimeState(policy=policy)
        self.model_routing_rate_limiter = ModelRoutingRateLimiter()
        self._ready = ready

    def readiness(self) -> dict:
        return {"ready": self._ready, "status": "ready" if self._ready else "error"}


def settings_for(**overrides):
    values = {
        "model_plane_observation_enabled": True,
        "model_plane_observation_endpoint": (
            "https://orchestra.example/api/model-routing-observations"
        ),
        "model_plane_observation_api_key": "pk_model_plane_test",
        "model_plane_observation_api_key_file": "",
        "model_plane_observation_deployment_id": "model-plane-staging-a",
        "model_plane_observation_target_environment": "staging",
        "model_plane_observation_engine_instance_id": "engine-pod-0",
        "model_plane_observation_version": 1,
        "model_plane_observation_interval_seconds": 60.0,
        "model_plane_observation_timeout_seconds": 5.0,
        "model_plane_observation_jitter_ratio": 0.1,
        "auth_enabled": True,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def config_for(**overrides) -> ModelPlaneObservationConfig:
    loaded = load_model_plane_observation_config(settings_for(**overrides))
    assert loaded is not None
    return loaded


@pytest.mark.parametrize(("value", "expected"), [("1", 1), ("2", 2)])
def test_settings_parses_observation_version_from_environment(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
    expected: int,
) -> None:
    monkeypatch.setenv("MODEL_PLANE_OBSERVATION_VERSION", value)

    loaded = Settings(_env_file=None)

    assert loaded.model_plane_observation_version == expected


def test_settings_rejects_unknown_observation_version_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MODEL_PLANE_OBSERVATION_VERSION", "3")

    with pytest.raises(ValidationError, match="model_plane_observation_version"):
        Settings(_env_file=None)


def inventory() -> ModelList:
    return ModelList(
        data=[ModelInfo(id="zeta:model"), ModelInfo(id="alpha:model")],
        unavailable=[UnavailableModel(id="offline:model", reason="upstream_timeout")],
    )


def test_disabled_observer_ignores_unconfigured_fields() -> None:
    assert (
        load_model_plane_observation_config(
            settings_for(
                model_plane_observation_enabled=False,
                model_plane_observation_endpoint="",
                model_plane_observation_api_key="",
            )
        )
        is None
    )


@pytest.mark.parametrize(
    ("overrides", "code"),
    [
        ({"model_plane_observation_endpoint": "http://orchestra.example/api/model-routing-observations"}, "insecure_endpoint"),
        ({"model_plane_observation_endpoint": "https://orchestra.example/wrong"}, "invalid_endpoint"),
        ({"model_plane_observation_endpoint": "https://user:pass@orchestra.example/api/model-routing-observations"}, "invalid_endpoint"),
        ({"model_plane_observation_target_environment": "production"}, "invalid_environment"),
        ({"model_plane_observation_engine_instance_id": "bad instance"}, "invalid_identifier"),
        ({"model_plane_observation_version": 3}, "invalid_observation_version"),
        ({"model_plane_observation_api_key": ""}, "ambiguous_api_key"),
        ({"model_plane_observation_api_key": "pk_non_ascii_\u00e9"}, "invalid_api_key"),
        ({"model_plane_observation_api_key_file": "/tmp/key"}, "ambiguous_api_key"),
    ],
)
def test_enabled_observer_rejects_incomplete_or_unsafe_configuration(
    overrides: dict,
    code: str,
) -> None:
    with pytest.raises(ModelPlaneObservationConfigError) as exc_info:
        load_model_plane_observation_config(settings_for(**overrides))
    assert exc_info.value.code == code


def test_loopback_http_is_allowed_for_local_integration() -> None:
    config = config_for(
        model_plane_observation_endpoint=(
            "http://127.0.0.1:3000/api/model-routing-observations"
        )
    )
    assert config.endpoint.startswith("http://127.0.0.1:3000/")


def test_api_key_file_is_re_read_for_rotation(tmp_path: Path) -> None:
    key_file = tmp_path / "model-plane-api-key"
    key_file.write_text("pk_before\n", encoding="utf-8")
    config = config_for(
        model_plane_observation_api_key="",
        model_plane_observation_api_key_file=str(key_file),
    )

    assert read_model_plane_observation_api_key(config) == "pk_before"
    key_file.write_text("pk_after\n", encoding="utf-8")
    assert read_model_plane_observation_api_key(config) == "pk_after"


def test_enabled_observer_requires_initial_key_file(tmp_path: Path) -> None:
    with pytest.raises(ModelPlaneObservationConfigError) as exc_info:
        config_for(
            model_plane_observation_api_key="",
            model_plane_observation_api_key_file=str(tmp_path / "missing"),
        )
    assert exc_info.value.code == "api_key_unavailable"


def test_inventory_digest_is_deterministic_and_payload_free() -> None:
    first = inventory()
    second = ModelList(data=list(reversed(first.data)), unavailable=first.unavailable)

    digest, available, unavailable = model_inventory_summary(first)

    assert model_inventory_summary(second) == (digest, available, unavailable)
    assert digest == "sha256:ab07e0564578866c954565b0b596ea79d61468f54078a505723103fac22d6035"
    assert available == 2
    assert unavailable == 1


def test_observation_matches_exact_platform_shape_without_inventory_names() -> None:
    payload = build_model_plane_observation(
        config_for(),
        FakeState(),
        inventory,
        observation_id="observation-fixed-1",
        now=datetime(2026, 7, 13, 12, 30, tzinfo=UTC),
    )

    assert set(payload) == {
        "artifactType",
        "observationVersion",
        "observationId",
        "deploymentId",
        "targetEnvironment",
        "engineInstanceId",
        "engineVersion",
        "healthStatus",
        "inventoryDigest",
        "availableModelCount",
        "unavailableModelCount",
        "observedAt",
        "routingPolicy",
    }
    assert payload["artifactType"] == "orchestra.model-plane-observation"
    assert payload["observationVersion"] == 1
    assert payload["healthStatus"] == "ready"
    assert payload["observedAt"] == "2026-07-13T12:30:00.000Z"
    assert payload["routingPolicy"] == {
        "object": "model_routing_policy.status",
        "active": False,
        "policy_id": None,
        "revision": None,
        "digest": None,
        "source": None,
        "org_id": None,
        "environment": None,
        "release_id": None,
        "deployment_id": None,
        "offline_lease_expires_at": None,
        "candidate_error_code": None,
        "request_time_enforcement": False,
        "route_count": 0,
        "rate_limit_scope": None,
        "pricing_catalog_digest": None,
        "pricing_model_count": 0,
        "org_binding_mode": None,
    }
    serialized = json.dumps(payload)
    assert "alpha:model" not in serialized
    assert "zeta:model" not in serialized
    assert "offline:model" not in serialized
    assert "pk_model_plane_test" not in serialized


def test_observation_v2_reports_inactive_routing_inventory_without_names() -> None:
    payload = build_model_plane_observation(
        config_for(model_plane_observation_version=2),
        FakeState(),
        inventory,
        candidate_availability=lambda _candidate: False,
        observation_id="observation-v2-inactive",
    )

    assert payload["observationVersion"] == 2
    assert payload["routingInventory"] == {
        "object": "model_routing_inventory.status",
        "status": "not_applicable",
        "policy_digest": None,
        "candidate_count": 0,
        "available_candidate_count": 0,
        "unavailable_candidate_count": 0,
        "ready_route_count": 0,
        "unavailable_route_count": 0,
    }
    serialized = json.dumps(payload)
    assert "alpha:model" not in serialized
    assert "zeta:model" not in serialized


def test_observation_v2_binds_payload_free_route_readiness_to_active_policy() -> None:
    policy = active_policy()
    available = {"qwen3:32b"}
    payload = build_model_plane_observation(
        config_for(
            model_plane_observation_version=2,
            model_plane_observation_deployment_id="model-plane-golden-v1",
        ),
        FakeState(policy=policy),
        inventory,
        candidate_availability=lambda candidate: candidate in available,
        observation_id="observation-v2-active",
    )

    assert payload["routingInventory"] == {
        "object": "model_routing_inventory.status",
        "status": "degraded",
        "policy_digest": policy.digest,
        "candidate_count": 3,
        "available_candidate_count": 1,
        "unavailable_candidate_count": 2,
        "ready_route_count": 1,
        "unavailable_route_count": 1,
    }
    attrs = model_plane_observation_span_attrs(payload)
    assert attrs["prometa.artifact.type"] == "model-routing-policy"
    assert attrs["prometa.artifact.digest"] == policy.digest
    assert attrs["prometa.policy.digest"] == policy.digest
    assert attrs["prometa.release.id"] == "release-golden-model-v1"
    assert attrs["prometa.deployment.id"] == "model-plane-golden-v1"
    assert attrs["prometa.environment"] == "staging"
    serialized = json.dumps(payload)
    assert "qwen3:32b" not in serialized
    assert "llama3.3:70b:openrouter" not in serialized
    assert "llama3.2:3b" not in serialized


@pytest.mark.parametrize(
    ("available", "status", "ready_routes", "unavailable_routes"),
    [
        ({"qwen3:32b", "llama3.2:3b"}, "ready", 2, 0),
        ({"qwen3:32b"}, "degraded", 1, 1),
        (set(), "unavailable", 0, 2),
    ],
)
def test_routing_inventory_status_is_derived_from_route_coverage(
    available: set[str],
    status: str,
    ready_routes: int,
    unavailable_routes: int,
) -> None:
    summary = model_routing_inventory_summary(
        FakeState(policy=active_policy()),
        lambda candidate: candidate in available,
    )

    assert summary["status"] == status
    assert summary["ready_route_count"] == ready_routes
    assert summary["unavailable_route_count"] == unavailable_routes


def test_observation_v2_requires_local_availability_resolver() -> None:
    with pytest.raises(ModelPlaneObservationConfigError) as exc_info:
        build_model_plane_observation(
            config_for(model_plane_observation_version=2),
            FakeState(),
            inventory,
        )
    assert exc_info.value.code == "routing_inventory_resolver_missing"


def test_not_ready_state_is_reported_without_raising() -> None:
    payload = build_model_plane_observation(
        config_for(),
        FakeState(ready=False),
        inventory,
    )
    assert payload["healthStatus"] == "not_ready"
    attrs = model_plane_observation_span_attrs(payload)
    assert attrs["prometa.deployment.id"] == "model-plane-staging-a"
    assert "prometa.artifact.digest" not in attrs


def test_active_policy_scope_mismatch_fails_closed(monkeypatch) -> None:
    from inference_engine import model_plane_observer as observer_module

    monkeypatch.setattr(
        observer_module,
        "build_model_routing_status",
        lambda *args, **kwargs: ModelRoutingPolicyStatus(
            active=True,
            policy_id="routing-1",
            revision=1,
            digest=f"sha256:{'a' * 64}",
            source="candidate",
            org_id="org-1",
            environment="prod",
            release_id="release-1",
            deployment_id="other-deployment",
            offline_lease_expires_at="2026-07-14T00:00:00.000Z",
            request_time_enforcement=True,
            route_count=1,
            rate_limit_scope="process-replica",
            org_binding_mode="auth-key-org",
        ),
    )

    with pytest.raises(ModelPlaneObservationConfigError) as exc_info:
        build_model_plane_observation(config_for(), FakeState(), inventory)
    assert exc_info.value.code == "routing_scope_mismatch"


@pytest.mark.asyncio
async def test_transient_failure_retries_same_observation_id() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(500 if len(requests) == 1 else 201)

    state = FakeState()
    reporter = ModelPlaneObservationReporter(config_for(), state, inventory)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        assert await reporter.report_once(client) is False
        assert await reporter.report_once(client) is True

    bodies = [json.loads(request.content) for request in requests]
    assert bodies[0]["observationId"] == bodies[1]["observationId"]
    assert requests[0].headers["x-api-key"] == "pk_model_plane_test"
    assert reporter.status().successes_total == 1
    assert reporter.status().failures_total == 1
    assert reporter.status().pending_observation_id is None
    assert state.readiness()["ready"] is True


def test_admin_status_is_payload_free_and_disabled_by_default(monkeypatch) -> None:
    monkeypatch.setattr(settings, "auth_enabled", False)
    monkeypatch.setattr(app_state, "model_plane_observer", None)

    response = TestClient(app).get("/v1/admin/model-plane-observer")

    assert response.status_code == 200
    assert response.json() == {
        "object": "model_plane_observer.status",
        "enabled": False,
        "running": False,
        "attempts_total": 0,
        "successes_total": 0,
        "failures_total": 0,
        "consecutive_failures": 0,
        "pending_observation_id": None,
        "last_attempt_at": None,
        "last_success_at": None,
        "last_error_code": None,
    }


@pytest.mark.asyncio
async def test_permanent_rejection_drops_payload_before_next_cycle() -> None:
    observation_ids: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        observation_ids.append(json.loads(request.content)["observationId"])
        return httpx.Response(400 if len(observation_ids) == 1 else 201)

    reporter = ModelPlaneObservationReporter(config_for(), FakeState(), inventory)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        assert await reporter.report_once(client) is False
        assert reporter.status().pending_observation_id is None
        assert await reporter.report_once(client) is True

    assert observation_ids[0] != observation_ids[1]


@pytest.mark.asyncio
async def test_auth_rejection_retains_payload_and_rotated_file_recovers(
    tmp_path: Path,
) -> None:
    key_file = tmp_path / "model-plane-api-key"
    key_file.write_text("pk_old", encoding="utf-8")
    config = config_for(
        model_plane_observation_api_key="",
        model_plane_observation_api_key_file=str(key_file),
    )
    calls: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        calls.append((request.headers["x-api-key"], body["observationId"]))
        return httpx.Response(401 if request.headers["x-api-key"] == "pk_old" else 201)

    reporter = ModelPlaneObservationReporter(config, FakeState(), inventory)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        assert await reporter.report_once(client) is False
        key_file.write_text("pk_new", encoding="utf-8")
        assert await reporter.report_once(client) is True

    assert calls[0][0] == "pk_old"
    assert calls[1][0] == "pk_new"
    assert calls[0][1] == calls[1][1]


@pytest.mark.asyncio
async def test_redirect_is_not_followed_or_retried_as_same_payload() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(307, headers={"location": "https://attacker.example/steal"})

    reporter = ModelPlaneObservationReporter(config_for(), FakeState(), inventory)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        assert await reporter.report_once(client) is False

    assert calls == ["https://orchestra.example/api/model-routing-observations"]
    assert reporter.status().pending_observation_id is None


@pytest.mark.asyncio
async def test_lifecycle_waits_for_startup_without_owning_startup_task() -> None:
    started = asyncio.Event()

    class FakeReporter:
        async def run(self) -> None:
            started.set()

    startup = asyncio.create_task(asyncio.sleep(60))
    observer = asyncio.create_task(
        _run_observer_after_startup(startup, FakeReporter()),
    )
    await asyncio.sleep(0)
    assert started.is_set() is False

    observer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await observer
    assert startup.cancelled() is False
    startup.cancel()
    with pytest.raises(asyncio.CancelledError):
        await startup
