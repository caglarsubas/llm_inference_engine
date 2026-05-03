"""Auth — bearer token resolution, anonymous fallback, key file loading."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from inference_engine import auth as auth_mod
from inference_engine.auth import Identity, _redact, load_keys, require_identity
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
    yield
    auth_mod._reset_for_tests()


def test_anonymous_when_disabled() -> None:
    identity = require_identity(_fake_request())
    assert identity == Identity(tenant="anonymous", key_id="anon")


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
        json.dumps([{"key": "sk-foo", "tenant": "team-foo"}, {"key": "sk-bar", "tenant": "bar"}])
    )
    monkeypatch.setattr(settings, "auth_keys_file", keys_path)
    monkeypatch.setattr(settings, "auth_enabled", True)

    count = load_keys()
    assert count == 2

    identity = require_identity(_fake_request({"authorization": "Bearer sk-foo"}))
    assert identity.tenant == "team-foo"


def test_missing_keys_file_is_fatal_when_enabled(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(settings, "auth_keys_file", tmp_path / "nope.json")
    monkeypatch.setattr(settings, "auth_enabled", True)
    with pytest.raises(FileNotFoundError):
        load_keys()


def test_missing_keys_file_is_ok_when_disabled(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(settings, "auth_keys_file", tmp_path / "nope.json")
    # auth_enabled stays False from the fixture
    assert load_keys() == 0
