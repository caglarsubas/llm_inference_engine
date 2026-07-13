"""Per-key bearer-token auth + tenant attribution.

Keys live in a JSON file pointed to by ``AUTH_KEYS_FILE``. The file holds an
array of records:

    [
      {"key": "sk-dev-12345",  "tenant": "dev", "org_id": "org-acme"},
      {"key": "sk-prod-67890", "tenant": "production", "org_id": "org-acme"}
    ]

When ``AUTH_ENABLED=false`` (default for local dev), every request resolves to
``Identity(tenant="anonymous", key_id="anon")`` so routes don't need branching.

Why a JSON file rather than env vars or a database:
- Multiple keys per tenant + multiple tenants per process are the common shape.
- Env vars don't scale past a couple of keys without ugly serialisation.
- A real DB belongs in Prometa's control plane, not in this engine. The keys
  file is the seam where the control plane drops a generated set.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from fastapi import HTTPException, Request

from .config import settings
from .observability import get_logger

log = get_logger("auth")


@dataclass(frozen=True)
class Identity:
    """Resolved caller identity. Attached to ``request.state.identity``."""

    tenant: str
    key_id: str  # redacted key suitable for logs / span attributes
    org_id: str | None = None


@dataclass(frozen=True)
class _KeyRecord:
    key: str
    tenant: str
    org_id: str | None = None


_keys_by_value: dict[str, _KeyRecord] = {}
_keys_loaded = False


def _redact(key: str) -> str:
    """Return a short, safe-to-log identifier for a key."""
    if len(key) <= 12:
        return key[:3] + "***"
    return f"{key[:6]}...{key[-4:]}"


def load_keys() -> int:
    """Populate the in-memory key index from disk. Returns count loaded.

    Idempotent — calling again replaces the index. Callers that want a hot
    reload simply re-invoke this. Missing file is non-fatal when auth is off,
    fatal when on.
    """
    global _keys_by_value, _keys_loaded

    path = Path(settings.auth_keys_file)
    if not path.exists():
        if settings.auth_enabled:
            raise FileNotFoundError(f"AUTH_ENABLED=true but keys file missing: {path}")
        _keys_by_value = {}
        _keys_loaded = True
        log.info("auth.keys_missing_but_disabled", path=str(path))
        return 0

    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"keys file must be a JSON array, got {type(raw).__name__}")

    index: dict[str, _KeyRecord] = {}
    for position, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise ValueError(f"key record {position} must be an object")
        if "key" not in entry or "tenant" not in entry:
            raise ValueError(f"key record {position} missing 'key' or 'tenant'")
        key = entry["key"]
        tenant = entry["tenant"]
        if not isinstance(key, str) or not key or key != key.strip():
            raise ValueError(f"key record {position} has invalid 'key'")
        if not isinstance(tenant, str) or not tenant or tenant != tenant.strip():
            raise ValueError(f"key record {position} has invalid 'tenant'")
        org_id = entry.get("org_id")
        if org_id is not None and (
            not isinstance(org_id, str) or not org_id or org_id != org_id.strip()
        ):
            raise ValueError(f"key record {position} has invalid 'org_id'")
        record = _KeyRecord(
            key=key,
            tenant=tenant,
            org_id=org_id,
        )
        if record.key in index:
            raise ValueError(f"key record {position} duplicates an earlier key")
        index[record.key] = record

    _keys_by_value = index
    _keys_loaded = True
    log.info("auth.keys_loaded", path=str(path), count=len(index))
    return len(index)


def _ensure_loaded() -> None:
    if not _keys_loaded:
        load_keys()


def require_identity(request: Request) -> Identity:
    """FastAPI dependency. Resolves the caller's identity or raises 401.

    Uses ``request.state.identity`` as a per-request cache so it can be looked
    up multiple times in a single request without re-parsing the header.
    """
    if hasattr(request.state, "identity"):
        return request.state.identity

    if not settings.auth_enabled:
        identity = Identity(
            tenant="anonymous",
            key_id="anon",
            org_id=settings.model_routing_expected_org_id or None,
        )
        request.state.identity = identity
        return identity

    _ensure_loaded()

    header = request.headers.get("authorization", "")
    if not header.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="missing bearer token")

    key = header[7:].strip()
    record = _keys_by_value.get(key)
    if record is None:
        log.warning("auth.invalid_key", key_id=_redact(key))
        raise HTTPException(status_code=401, detail="invalid api key")

    identity = Identity(
        tenant=record.tenant,
        key_id=_redact(key),
        org_id=record.org_id,
    )
    request.state.identity = identity
    return identity


# ---------------------------------------------------------------------------
# Test helpers — keep production code from importing test paths.
# ---------------------------------------------------------------------------


def _set_keys_for_tests(records: list[tuple[str, str] | tuple[str, str, str]]) -> None:
    """Install a key index directly. Used by the auth tests."""
    global _keys_by_value, _keys_loaded
    index: dict[str, _KeyRecord] = {}
    for record in records:
        key, tenant, *org_id = record
        index[key] = _KeyRecord(
            key=key,
            tenant=tenant,
            org_id=(org_id[0] if org_id else None),
        )
    _keys_by_value = index
    _keys_loaded = True


def _reset_for_tests() -> None:
    global _keys_by_value, _keys_loaded
    _keys_by_value = {}
    _keys_loaded = False
