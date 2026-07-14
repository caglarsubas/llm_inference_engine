"""Auth — bearer token resolution, anonymous fallback, key file loading."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from inference_engine import auth as auth_mod
from inference_engine.auth import (
    Identity,
    _redact,
    auth_key_status,
    load_keys,
    reload_keys,
    require_identity,
)
from inference_engine.config import settings


def _fake_request(headers: dict[str, str] | None = None):
    """Construct an object that quacks like a FastAPI Request for require_identity."""
    return SimpleNamespace(
        headers=headers or {},
        state=SimpleNamespace(),
    )


@pytest.fixture(autouse=True)
def _reset_auth_state(monkeypatch):
    """Each test starts with a fresh in-memory key index and AUTH_ENABLED=false."""
    auth_mod._reset_for_tests()
    monkeypatch.setattr(settings, "auth_enabled", False)
    monkeypatch.setattr(settings, "model_routing_expected_org_id", "")
    yield
    auth_mod._reset_for_tests()


def test_anonymous_when_disabled() -> None:
    identity = require_identity(_fake_request())
    assert identity == Identity(tenant="anonymous", key_id="anon")


def test_anonymous_identity_uses_explicit_deployment_org_binding(monkeypatch) -> None:
    monkeypatch.setattr(settings, "model_routing_expected_org_id", "org-local")
    identity = require_identity(_fake_request())
    assert identity.org_id == "org-local"


def test_anonymous_path_caches_on_request_state() -> None:
    req = _fake_request()
    first = require_identity(req)
    second = require_identity(req)
    assert first is second  # state cache hit
    assert req.state.identity is first


def test_resolves_valid_bearer_when_enabled(monkeypatch) -> None:
    monkeypatch.setattr(settings, "auth_enabled", True)
    auth_mod._set_keys_for_tests([("sk-alpha-1234567890", "alpha")])

    identity = require_identity(_fake_request({"authorization": "Bearer sk-alpha-1234567890"}))
    assert identity.tenant == "alpha"
    assert "sk-alp" in identity.key_id and "7890" in identity.key_id


def test_resolves_org_bound_bearer_when_enabled(monkeypatch) -> None:
    monkeypatch.setattr(settings, "auth_enabled", True)
    auth_mod._set_keys_for_tests([("sk-org-key", "runtime", "org-acme")])

    identity = require_identity(_fake_request({"authorization": "Bearer sk-org-key"}))

    assert identity.org_id == "org-acme"


def test_case_insensitive_bearer_scheme(monkeypatch) -> None:
    monkeypatch.setattr(settings, "auth_enabled", True)
    auth_mod._set_keys_for_tests([("sk-x-12345", "x")])
    identity = require_identity(_fake_request({"authorization": "bearer sk-x-12345"}))
    assert identity.tenant == "x"


def test_missing_header_raises_401(monkeypatch) -> None:
    monkeypatch.setattr(settings, "auth_enabled", True)
    auth_mod._set_keys_for_tests([("sk-x", "x")])
    with pytest.raises(HTTPException) as exc:
        require_identity(_fake_request())
    assert exc.value.status_code == 401


def test_unknown_key_raises_401(monkeypatch) -> None:
    monkeypatch.setattr(settings, "auth_enabled", True)
    auth_mod._set_keys_for_tests([("sk-known", "k")])
    with pytest.raises(HTTPException) as exc:
        require_identity(_fake_request({"authorization": "Bearer sk-bogus-9999"}))
    assert exc.value.status_code == 401


def test_redaction_keeps_short_keys_terse() -> None:
    assert _redact("abc") == "abc***"
    assert _redact("sk-supersecret-9999") == "sk-sup...9999"


def test_load_keys_reads_json_file(tmp_path: Path, monkeypatch) -> None:
    keys_path = tmp_path / "keys.json"
    keys_path.write_text(
        json.dumps(
            [
                {"key": "sk-foo", "tenant": "team-foo", "org_id": "org-acme"},
                {"key": "sk-bar", "tenant": "bar"},
            ]
        )
    )
    monkeypatch.setattr(settings, "auth_keys_file", keys_path)
    monkeypatch.setattr(settings, "auth_enabled", True)

    count = load_keys()
    assert count == 2

    identity = require_identity(_fake_request({"authorization": "Bearer sk-foo"}))
    assert identity.tenant == "team-foo"
    assert identity.org_id == "org-acme"


def test_load_keys_uses_explicit_key_id_and_validity_window(
    tmp_path: Path,
    monkeypatch,
) -> None:
    keys_path = tmp_path / "keys.json"
    keys_path.write_text(
        json.dumps(
            [
                {
                    "key": "mpk-runtime-secret",
                    "key_id": "mpk_runtime_r2",
                    "tenant": "runtime",
                    "org_id": "org-acme",
                    "not_before": "2026-01-01T00:00:00Z",
                    "expires_at": "2027-01-01T00:00:00Z",
                }
            ]
        )
    )
    monkeypatch.setattr(settings, "auth_keys_file", keys_path)
    monkeypatch.setattr(settings, "auth_enabled", True)
    monkeypatch.setattr(
        auth_mod,
        "_utc_now",
        lambda: datetime(2026, 7, 14, tzinfo=timezone.utc),
    )

    assert load_keys() == 1
    identity = require_identity(_fake_request({"authorization": "Bearer mpk-runtime-secret"}))
    assert identity.key_id == "mpk_runtime_r2"


@pytest.mark.parametrize(
    ("not_before", "expires_at"),
    [
        ("2027-01-01T00:00:00Z", "2028-01-01T00:00:00Z"),
        ("2025-01-01T00:00:00Z", "2026-01-01T00:00:00Z"),
    ],
)
def test_require_identity_rejects_keys_outside_validity_window(
    tmp_path: Path,
    monkeypatch,
    not_before: str,
    expires_at: str,
) -> None:
    keys_path = tmp_path / "keys.json"
    keys_path.write_text(
        json.dumps(
            [
                {
                    "key": "mpk-windowed-secret",
                    "key_id": "mpk_windowed",
                    "tenant": "runtime",
                    "org_id": "org-acme",
                    "not_before": not_before,
                    "expires_at": expires_at,
                }
            ]
        )
    )
    monkeypatch.setattr(settings, "auth_keys_file", keys_path)
    monkeypatch.setattr(settings, "auth_enabled", True)
    monkeypatch.setattr(
        auth_mod,
        "_utc_now",
        lambda: datetime(2026, 7, 14, tzinfo=timezone.utc),
    )
    load_keys()

    with pytest.raises(HTTPException) as raised:
        require_identity(_fake_request({"authorization": "Bearer mpk-windowed-secret"}))
    assert raised.value.status_code == 401
    assert raised.value.detail == "invalid api key"


def test_load_keys_rejects_unknown_fields_and_naive_timestamps(
    tmp_path: Path,
    monkeypatch,
) -> None:
    keys_path = tmp_path / "keys.json"
    monkeypatch.setattr(settings, "auth_keys_file", keys_path)
    monkeypatch.setattr(settings, "auth_enabled", True)
    keys_path.write_text(json.dumps([{"key": "sk-foo", "tenant": "runtime", "typo": True}]))
    with pytest.raises(ValueError, match="unsupported fields"):
        load_keys()

    keys_path.write_text(
        json.dumps(
            [
                {
                    "key": "sk-foo",
                    "tenant": "runtime",
                    "not_before": "2026-07-14T00:00:00",
                }
            ]
        )
    )
    with pytest.raises(ValueError, match="must include a timezone"):
        load_keys()


def test_reload_is_atomic_and_requires_the_calling_key_in_candidate(
    tmp_path: Path,
    monkeypatch,
) -> None:
    keys_path = tmp_path / "keys.json"
    monkeypatch.setattr(settings, "auth_keys_file", keys_path)
    monkeypatch.setattr(settings, "auth_enabled", True)
    keys_path.write_text(
        json.dumps(
            [
                {
                    "key": "mpk-old-secret",
                    "key_id": "mpk_old",
                    "tenant": "runtime",
                    "org_id": "org-acme",
                }
            ]
        )
    )
    load_keys()

    keys_path.write_text("not-json")
    with pytest.raises(ValueError, match="valid UTF-8 JSON"):
        reload_keys(required_key="mpk-old-secret")
    assert (
        require_identity(_fake_request({"authorization": "Bearer mpk-old-secret"})).key_id
        == "mpk_old"
    )

    keys_path.write_text(
        json.dumps(
            [
                {
                    "key": "mpk-new-secret",
                    "key_id": "mpk_new",
                    "tenant": "runtime",
                    "org_id": "org-acme",
                }
            ]
        )
    )
    with pytest.raises(ValueError, match="retain the active credential"):
        reload_keys(required_key="mpk-old-secret")
    assert (
        require_identity(_fake_request({"authorization": "Bearer mpk-old-secret"})).key_id
        == "mpk_old"
    )


def test_reload_accepts_bounded_overlap_and_status_never_exposes_secrets(
    tmp_path: Path,
    monkeypatch,
) -> None:
    keys_path = tmp_path / "keys.json"
    monkeypatch.setattr(settings, "auth_keys_file", keys_path)
    monkeypatch.setattr(settings, "auth_enabled", True)
    monkeypatch.setattr(
        auth_mod,
        "_utc_now",
        lambda: datetime(2026, 7, 14, tzinfo=timezone.utc),
    )
    keys_path.write_text(
        json.dumps(
            [
                {
                    "key": "mpk-old-secret",
                    "key_id": "mpk_old",
                    "tenant": "runtime",
                    "org_id": "org-acme",
                }
            ]
        )
    )
    load_keys()
    keys_path.write_text(
        json.dumps(
            [
                {
                    "key": "mpk-old-secret",
                    "key_id": "mpk_old",
                    "tenant": "runtime",
                    "org_id": "org-acme",
                    "expires_at": "2026-07-14T01:00:00Z",
                },
                {
                    "key": "mpk-new-secret",
                    "key_id": "mpk_new",
                    "tenant": "runtime",
                    "org_id": "org-acme",
                    "not_before": "2026-07-14T00:00:00Z",
                    "expires_at": "2026-10-12T00:00:00Z",
                },
            ]
        )
    )

    result = reload_keys(required_key="mpk-old-secret")

    assert result.keys_loaded == 2
    assert result.active_keys == 2
    assert result.retained_caller is True
    assert (
        require_identity(_fake_request({"authorization": "Bearer mpk-new-secret"})).key_id
        == "mpk_new"
    )
    rendered_status = json.dumps(auth_key_status(), default=str)
    assert "mpk_old" in rendered_status
    assert "mpk_new" in rendered_status
    assert "mpk-old-secret" not in rendered_status
    assert "mpk-new-secret" not in rendered_status


def test_status_fingerprints_legacy_keys_without_exposing_secret_fragments(
    tmp_path: Path,
    monkeypatch,
) -> None:
    keys_path = tmp_path / "keys.json"
    keys_path.write_text(
        json.dumps([{"key": "sk-legacy-secret-1234", "tenant": "runtime"}]),
        encoding="utf-8",
    )
    monkeypatch.setattr(settings, "auth_keys_file", keys_path)
    monkeypatch.setattr(settings, "auth_enabled", True)
    load_keys()

    status = auth_key_status()

    assert status["keys"][0]["key_id"].startswith("legacy-sha256:")
    assert "sk-leg" not in status["keys"][0]["key_id"]
    assert "1234" not in status["keys"][0]["key_id"]


def test_load_keys_rejects_blank_org_binding(tmp_path: Path, monkeypatch) -> None:
    keys_path = tmp_path / "keys.json"
    keys_path.write_text(json.dumps([{"key": "sk-foo", "tenant": "team-foo", "org_id": ""}]))
    monkeypatch.setattr(settings, "auth_keys_file", keys_path)
    monkeypatch.setattr(settings, "auth_enabled", True)

    with pytest.raises(ValueError, match="invalid 'org_id'"):
        load_keys()


def test_load_keys_rejects_duplicates_without_echoing_secret(
    tmp_path: Path,
    monkeypatch,
) -> None:
    keys_path = tmp_path / "keys.json"
    secret = "sk-secret-duplicate"
    keys_path.write_text(
        json.dumps(
            [
                {"key": secret, "tenant": "one", "org_id": "org-acme"},
                {"key": secret, "tenant": "two", "org_id": "org-acme"},
            ]
        )
    )
    monkeypatch.setattr(settings, "auth_keys_file", keys_path)
    monkeypatch.setattr(settings, "auth_enabled", True)

    with pytest.raises(ValueError, match="duplicates an earlier key") as raised:
        load_keys()
    assert secret not in str(raised.value)


def test_missing_keys_file_is_fatal_when_enabled(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(settings, "auth_keys_file", tmp_path / "nope.json")
    monkeypatch.setattr(settings, "auth_enabled", True)
    with pytest.raises(FileNotFoundError):
        load_keys()


def test_missing_keys_file_is_ok_when_disabled(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(settings, "auth_keys_file", tmp_path / "nope.json")
    # auth_enabled stays False from the fixture
    assert load_keys() == 0
