"""``POST /v1/admin/policies:reload`` — atomic swap, malformed-file rejection, auth gate."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from inference_engine import auth as auth_mod
from inference_engine.config import settings
from inference_engine.evals import PolicyEntry, PolicyMatch, PolicyRegistry
from inference_engine.main import app
from inference_engine.schemas import AutoEvalSpec


def _write_policies(path: Path, names: list[str]) -> None:
    path.write_text(
        json.dumps(
            [
                {
                    "name": name,
                    "match": {"tenant": "*", "model": "*"},
                    "auto_eval": {"rubrics": ["safety"]},
                }
                for name in names
            ]
        )
    )


@pytest.fixture
def policies_file(tmp_path: Path, monkeypatch):
    path = tmp_path / "policies.json"
    monkeypatch.setattr(settings, "auto_eval_policies_file", path)
    return path


def test_reload_swaps_registry_on_success(policies_file) -> None:
    """Hitting reload with a fresh file count atomically replaces the registry."""
    from inference_engine.api.state import app_state  # noqa: PLC0415

    # Start with one policy; reload should take us to three.
    _write_policies(policies_file, ["p1"])
    app_state.policy_registry = PolicyRegistry(
        [
            PolicyEntry(
                name="p1",
                match=PolicyMatch(),
                spec=AutoEvalSpec(rubrics=["safety"]),
            )
        ]
    )
    assert len(app_state.policy_registry) == 1

    _write_policies(policies_file, ["a", "b", "c"])
    client = TestClient(app)
    r = client.post("/v1/admin/policies:reload")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["policies_loaded"] == 3
    assert body["source"].endswith("policies.json")
    assert body["object"] == "policy.reload"

    assert len(app_state.policy_registry) == 3
    assert {e.name for e in app_state.policy_registry.all()} == {"a", "b", "c"}


def test_reload_400_on_malformed_file_keeps_old_registry(policies_file) -> None:
    """A bad file must NOT clobber the in-memory registry."""
    from inference_engine.api.state import app_state  # noqa: PLC0415

    # Seed a known-good registry.
    app_state.policy_registry = PolicyRegistry(
        [
            PolicyEntry(
                name="kept",
                match=PolicyMatch(),
                spec=AutoEvalSpec(rubrics=["safety"]),
            )
        ]
    )

    policies_file.write_text("not-an-array")  # malformed JSON
    client = TestClient(app)
    r = client.post("/v1/admin/policies:reload")
    assert r.status_code == 400
    assert "reload failed" in r.json()["detail"]

    # Old registry preserved.
    assert len(app_state.policy_registry) == 1
    assert app_state.policy_registry.all()[0].name == "kept"


def test_reload_400_on_invalid_schema_keeps_old_registry(policies_file) -> None:
    """JSON parses but auto_eval is missing — same atomicity guarantee."""
    from inference_engine.api.state import app_state  # noqa: PLC0415

    app_state.policy_registry = PolicyRegistry([])
    # Plant a non-empty registry to verify we don't fall through to it.
    app_state.policy_registry = PolicyRegistry(
        [PolicyEntry(name="kept", match=PolicyMatch(), spec=AutoEvalSpec(rubrics=["safety"]))]
    )

    policies_file.write_text(json.dumps([{"match": {}}]))  # missing auto_eval
    client = TestClient(app)
    r = client.post("/v1/admin/policies:reload")
    assert r.status_code == 400
    assert len(app_state.policy_registry) == 1


def test_reload_with_missing_file_yields_empty_registry(tmp_path: Path, monkeypatch) -> None:
    """Missing file is non-fatal (matches startup behaviour) — registry empties."""
    from inference_engine.api.state import app_state  # noqa: PLC0415

    monkeypatch.setattr(settings, "auto_eval_policies_file", tmp_path / "ghost.json")
    app_state.policy_registry = PolicyRegistry(
        [PolicyEntry(name="x", match=PolicyMatch(), spec=AutoEvalSpec(rubrics=["safety"]))]
    )
    assert len(app_state.policy_registry) == 1

    client = TestClient(app)
    r = client.post("/v1/admin/policies:reload")
    assert r.status_code == 200
    assert r.json()["policies_loaded"] == 0
    assert len(app_state.policy_registry) == 0


def test_reload_requires_bearer_when_auth_enabled(policies_file, monkeypatch) -> None:
    """With AUTH_ENABLED=true, the admin endpoint refuses unauthenticated calls."""
    monkeypatch.setattr(settings, "auth_enabled", True)
    auth_mod._set_keys_for_tests([("sk-admin-key", "ops")])
    _write_policies(policies_file, ["x"])

    client = TestClient(app)
    r = client.post("/v1/admin/policies:reload")
    assert r.status_code == 401

    r2 = client.post(
        "/v1/admin/policies:reload",
        headers={"Authorization": "Bearer sk-admin-key"},
    )
    assert r2.status_code == 200, r2.text


def test_reload_emits_admin_span_with_counts(policies_file, _session_exporter) -> None:
    from inference_engine.api.state import app_state  # noqa: PLC0415

    app_state.policy_registry = PolicyRegistry([])
    _write_policies(policies_file, ["a", "b"])
    _session_exporter.clear()

    client = TestClient(app)
    r = client.post("/v1/admin/policies:reload")
    assert r.status_code == 200

    spans = [s for s in _session_exporter.get_finished_spans() if s.name == "admin.policies.reload"]
    assert len(spans) == 1
    s = spans[0]
    assert s.attributes["policy.previous_count"] == 0
    assert s.attributes["policy.loaded_count"] == 2
    assert "policies.json" in s.attributes["policy.source"]
