"""Managed auth-key status and atomic reload endpoints."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from inference_engine import auth as auth_mod
from inference_engine.auth import load_keys, require_identity
from inference_engine.config import settings
from inference_engine.main import app


def _write_keys(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(json.dumps(records), encoding="utf-8")


def _request_identity(secret: str):
    class Request:
        headers = {"authorization": f"Bearer {secret}"}

        class state:
            pass

    return require_identity(Request())


@pytest.fixture(autouse=True)
def _reset_auth(monkeypatch):
    auth_mod._reset_for_tests()
    monkeypatch.setattr(settings, "auth_enabled", False)
    yield
    auth_mod._reset_for_tests()


@pytest.fixture
def active_key_file(tmp_path: Path, monkeypatch) -> Path:
    path = tmp_path / "auth_keys.json"
    _write_keys(
        path,
        [
            {
                "key": "mpk-old-secret",
                "key_id": "mpk_old",
                "tenant": "tenant-runtime",
                "org_id": "org-acme",
            }
        ],
    )
    monkeypatch.setattr(settings, "auth_keys_file", path)
    monkeypatch.setattr(settings, "auth_enabled", True)
    load_keys()
    return path


def test_status_requires_auth_and_never_returns_secret(active_key_file: Path) -> None:
    client = TestClient(app)

    assert client.get("/v1/admin/auth-keys").status_code == 401
    response = client.get(
        "/v1/admin/auth-keys",
        headers={"Authorization": "Bearer mpk-old-secret"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "auth_keys.status"
    assert body["keys_loaded"] == 1
    assert body["active_keys"] == 1
    assert body["keys"][0]["key_id"] == "mpk_old"
    assert body["keys"][0]["org_id"] == "org-acme"
    assert "mpk-old-secret" not in response.text


def test_reload_activates_overlap_and_retains_calling_key(active_key_file: Path) -> None:
    _write_keys(
        active_key_file,
        [
            {
                "key": "mpk-old-secret",
                "key_id": "mpk_old",
                "tenant": "tenant-runtime",
                "org_id": "org-acme",
                "expires_at": "2099-01-01T00:00:00Z",
            },
            {
                "key": "mpk-new-secret",
                "key_id": "mpk_new",
                "tenant": "tenant-runtime",
                "org_id": "org-acme",
                "not_before": "2026-01-01T00:00:00Z",
                "expires_at": "2099-01-01T00:00:00Z",
            },
        ],
    )

    response = TestClient(app).post(
        "/v1/admin/auth-keys:reload",
        headers={"Authorization": "Bearer mpk-old-secret"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "auth_keys.reload"
    assert body["keys_loaded"] == 2
    assert body["active_keys"] == 2
    assert body["retained_caller"] is True
    assert body["digest"].startswith("sha256:")
    assert _request_identity("mpk-new-secret").key_id == "mpk_new"
    assert "mpk-old-secret" not in response.text
    assert "mpk-new-secret" not in response.text


def test_reload_rejects_lockout_and_preserves_previous_index(
    active_key_file: Path,
) -> None:
    _write_keys(
        active_key_file,
        [
            {
                "key": "mpk-new-secret",
                "key_id": "mpk_new",
                "tenant": "tenant-runtime",
                "org_id": "org-acme",
            }
        ],
    )

    response = TestClient(app).post(
        "/v1/admin/auth-keys:reload",
        headers={"Authorization": "Bearer mpk-old-secret"},
    )

    assert response.status_code == 400
    assert response.json()["detail"]["type"] == "auth_keys_caller_not_retained"
    assert _request_identity("mpk-old-secret").key_id == "mpk_old"


def test_reload_rejects_malformed_or_missing_candidate_without_mutation(
    active_key_file: Path,
) -> None:
    active_key_file.write_text("not-json", encoding="utf-8")
    client = TestClient(app)

    malformed = client.post(
        "/v1/admin/auth-keys:reload",
        headers={"Authorization": "Bearer mpk-old-secret"},
    )
    assert malformed.status_code == 400
    assert malformed.json()["detail"]["type"] == "auth_keys_invalid"
    assert _request_identity("mpk-old-secret").key_id == "mpk_old"

    active_key_file.unlink()
    missing = client.post(
        "/v1/admin/auth-keys:reload",
        headers={"Authorization": "Bearer mpk-old-secret"},
    )
    assert missing.status_code == 400
    assert missing.json()["detail"]["type"] == "auth_keys_file_missing"
    assert _request_identity("mpk-old-secret").key_id == "mpk_old"


def test_reload_when_auth_disabled_can_activate_an_empty_missing_file(
    tmp_path: Path,
    monkeypatch,
) -> None:
    auth_mod._reset_for_tests()
    monkeypatch.setattr(settings, "auth_enabled", False)
    monkeypatch.setattr(settings, "auth_keys_file", tmp_path / "missing.json")

    response = TestClient(app).post("/v1/admin/auth-keys:reload")

    assert response.status_code == 200
    assert response.json()["keys_loaded"] == 0
    assert response.json()["active_keys"] == 0
    assert response.json()["retained_caller"] is True
