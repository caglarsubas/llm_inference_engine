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

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
    key_id: str | None = None
    not_before: datetime | None = None
    expires_at: datetime | None = None

    def active_at(self, now: datetime) -> bool:
        if self.not_before is not None and now < self.not_before:
            return False
        return self.expires_at is None or now < self.expires_at

    @property
    def safe_key_id(self) -> str:
        return self.key_id or _redact(self.key)

    @property
    def status_key_id(self) -> str:
        if self.key_id is not None:
            return self.key_id
        fingerprint = hashlib.sha256(self.key.encode("utf-8")).hexdigest()[:16]
        return f"legacy-sha256:{fingerprint}"


@dataclass(frozen=True)
class AuthKeySnapshot:
    """Strictly parsed key file ready for one atomic in-memory swap."""

    by_value: dict[str, _KeyRecord]
    digest: str
    source: str


@dataclass(frozen=True)
class AuthKeyReloadResult:
    """Secret-free result returned by the authenticated reload endpoint."""

    digest: str
    source: str
    keys_loaded: int
    active_keys: int
    retained_caller: bool


_keys_by_value: dict[str, _KeyRecord] = {}
_keys_loaded = False
_keys_digest: str | None = None

_KEY_RECORD_FIELDS = {
    "key",
    "tenant",
    "org_id",
    "key_id",
    "not_before",
    "expires_at",
}
_MAX_AUTH_KEYS_FILE_BYTES = 1024 * 1024


def _redact(key: str) -> str:
    """Return a short, safe-to-log identifier for a key."""
    if len(key) <= 12:
        return key[:3] + "***"
    return f"{key[:6]}...{key[-4:]}"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_timestamp(value: Any, *, field: str, position: int) -> datetime | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"key record {position} has invalid '{field}'")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"key record {position} has invalid '{field}'") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"key record {position} '{field}' must include a timezone")
    return parsed.astimezone(timezone.utc)


def _parse_key_snapshot(path: Path) -> AuthKeySnapshot:
    raw_bytes = path.read_bytes()
    if len(raw_bytes) > _MAX_AUTH_KEYS_FILE_BYTES:
        raise ValueError("keys file exceeds 1 MiB")
    try:
        raw = json.loads(raw_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("keys file must contain valid UTF-8 JSON") from exc
    if not isinstance(raw, list):
        raise ValueError(f"keys file must be a JSON array, got {type(raw).__name__}")

    index: dict[str, _KeyRecord] = {}
    key_ids: set[str] = set()
    for position, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise ValueError(f"key record {position} must be an object")
        unsupported = set(entry) - _KEY_RECORD_FIELDS
        if unsupported:
            raise ValueError(f"key record {position} contains unsupported fields")
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
        key_id = entry.get("key_id")
        if key_id is not None and (
            not isinstance(key_id, str) or not key_id or key_id != key_id.strip()
        ):
            raise ValueError(f"key record {position} has invalid 'key_id'")
        not_before = _parse_timestamp(
            entry.get("not_before"), field="not_before", position=position
        )
        expires_at = _parse_timestamp(
            entry.get("expires_at"), field="expires_at", position=position
        )
        if not_before is not None and expires_at is not None and expires_at <= not_before:
            raise ValueError(f"key record {position} expires before it becomes active")

        record = _KeyRecord(
            key=key,
            tenant=tenant,
            org_id=org_id,
            key_id=key_id,
            not_before=not_before,
            expires_at=expires_at,
        )
        if record.key in index:
            raise ValueError(f"key record {position} duplicates an earlier key")
        if record.key_id is not None and record.key_id in key_ids:
            raise ValueError(f"key record {position} duplicates an earlier key_id")
        index[record.key] = record
        if record.key_id is not None:
            key_ids.add(record.key_id)

    return AuthKeySnapshot(
        by_value=index,
        digest=f"sha256:{hashlib.sha256(raw_bytes).hexdigest()}",
        source=str(path),
    )


def _activate_snapshot(snapshot: AuthKeySnapshot) -> None:
    global _keys_by_value, _keys_digest, _keys_loaded
    _keys_by_value = snapshot.by_value
    _keys_digest = snapshot.digest
    _keys_loaded = True


def load_keys() -> int:
    """Populate the in-memory key index from disk. Returns count loaded.

    Idempotent — calling again replaces the index. Callers that want a hot
    reload simply re-invoke this. Missing file is non-fatal when auth is off,
    fatal when on.
    """
    global _keys_by_value, _keys_digest, _keys_loaded

    path = Path(settings.auth_keys_file)
    if not path.exists():
        if settings.auth_enabled:
            raise FileNotFoundError(f"AUTH_ENABLED=true but keys file missing: {path}")
        _keys_by_value = {}
        _keys_digest = None
        _keys_loaded = True
        log.info("auth.keys_missing_but_disabled", path=str(path))
        return 0

    snapshot = _parse_key_snapshot(path)
    _activate_snapshot(snapshot)
    log.info("auth.keys_loaded", path=str(path), count=len(snapshot.by_value))
    return len(snapshot.by_value)


def reload_keys(*, required_key: str | None = None) -> AuthKeyReloadResult:
    """Validate the complete candidate and swap it atomically.

    When auth is enabled the credential used to invoke reload must remain
    active in the candidate. This makes a two-key overlap safe: add the new
    key and reload with the old one, then let the old validity window expire.
    """

    path = Path(settings.auth_keys_file)
    if not path.exists():
        if settings.auth_enabled:
            raise FileNotFoundError(f"AUTH_ENABLED=true but keys file missing: {path}")
        snapshot = AuthKeySnapshot(by_value={}, digest="", source=str(path))
    else:
        snapshot = _parse_key_snapshot(path)

    now = _utc_now()
    retained_caller = required_key is None
    if required_key is not None:
        candidate_caller = snapshot.by_value.get(required_key)
        retained_caller = candidate_caller is not None and candidate_caller.active_at(now)
        if not retained_caller:
            raise ValueError("candidate must retain the active credential invoking reload")

    _activate_snapshot(snapshot)
    active_keys = sum(record.active_at(now) for record in snapshot.by_value.values())
    return AuthKeyReloadResult(
        digest=snapshot.digest,
        source=snapshot.source,
        keys_loaded=len(snapshot.by_value),
        active_keys=active_keys,
        retained_caller=retained_caller,
    )


def auth_key_status(*, now: datetime | None = None) -> dict[str, Any]:
    """Return payload-free key metadata for the operator status endpoint."""

    _ensure_loaded()
    checked_at = (now or _utc_now()).astimezone(timezone.utc)
    records = sorted(_keys_by_value.values(), key=lambda record: record.status_key_id)
    return {
        "source": str(settings.auth_keys_file),
        "digest": _keys_digest,
        "keys_loaded": len(records),
        "active_keys": sum(record.active_at(checked_at) for record in records),
        "checked_at": checked_at,
        "keys": [
            {
                "key_id": record.status_key_id,
                "tenant": record.tenant,
                "org_id": record.org_id,
                "not_before": record.not_before,
                "expires_at": record.expires_at,
                "active": record.active_at(checked_at),
            }
            for record in records
        ],
    }


def presented_bearer_token(request: Request) -> str | None:
    header = request.headers.get("authorization", "")
    if not header.lower().startswith("bearer "):
        return None
    token = header[7:].strip()
    return token or None


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
    if record is None or not record.active_at(_utc_now()):
        log.warning("auth.invalid_key", key_id=_redact(key))
        raise HTTPException(status_code=401, detail="invalid api key")

    identity = Identity(
        tenant=record.tenant,
        key_id=record.safe_key_id,
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
    global _keys_by_value, _keys_digest, _keys_loaded
    _keys_by_value = {}
    _keys_digest = None
    _keys_loaded = False
