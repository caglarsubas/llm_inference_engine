"""Verified desired-state policy for the tenant-deployed model plane.

The Prometa control plane signs policy bytes out of band. This module verifies
and activates those bytes locally; it never calls the control plane.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from cryptography.hazmat.primitives.serialization import load_der_public_key
from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, ValidationError
from pydantic.alias_generators import to_camel

from .config import settings

MODEL_ROUTING_POLICY_TYPE = "orchestra.model-routing-policy"
MODEL_ROUTING_POLICY_VERSION = 1
MODEL_ROUTING_POLICY_CANONICALIZATION = "signed-payload-json-v1"
MODEL_ROUTING_POLICY_AUDIENCE = "orchestra-model-plane"
MAX_MODEL_ROUTING_ROUTES = 128
MAX_MODEL_ROUTING_FALLBACKS = 8
MAX_SAFE_INTEGER = 9_007_199_254_740_991

_CANONICAL_TIMESTAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$")


class _ContractModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        extra="forbid",
        frozen=True,
    )


class ModelRoutingLimits(_ContractModel):
    max_input_tokens: StrictInt | None
    max_output_tokens: StrictInt | None
    max_requests_per_minute: StrictInt | None
    max_cost_micros_per_request: StrictInt | None


class ModelRoutingRoute(_ContractModel):
    route_id: StrictStr
    requested_model: StrictStr
    primary_model: StrictStr
    fallback_models: list[StrictStr]
    limits: ModelRoutingLimits


class ModelRoutingPolicyClaims(_ContractModel):
    artifact_type: Literal[MODEL_ROUTING_POLICY_TYPE]
    policy_version: Literal[MODEL_ROUTING_POLICY_VERSION]
    issuer: StrictStr
    key_id: StrictStr
    subject: StrictStr
    org_id: StrictStr
    audience: StrictStr
    target_environment: Literal["dev", "test", "staging", "prod"]
    policy_id: StrictStr
    revision: StrictInt
    release_id: StrictStr
    deployment_id: StrictStr
    routes: list[ModelRoutingRoute]
    issued_at: StrictStr
    not_before: StrictStr
    expires_at: StrictStr
    offline_lease_expires_at: StrictStr
    jti: StrictStr
    revocation_ref: StrictStr


class ModelRoutingPolicyEnvelope(_ContractModel):
    policy_id: StrictStr
    policy_version: Literal[MODEL_ROUTING_POLICY_VERSION]
    algorithm: Literal["ed25519"]
    canonicalization: Literal[MODEL_ROUTING_POLICY_CANONICALIZATION]
    issuer: StrictStr
    key_id: StrictStr
    signed_payload: StrictStr
    signature: StrictStr
    signed: Literal[True]


class ModelRoutingTrustEntry(_ContractModel):
    issuer: StrictStr
    key_id: StrictStr
    public_key_spki_der_base64: StrictStr
    allowed_org_ids: list[StrictStr]
    allowed_environments: list[Literal["dev", "test", "staging", "prod"]]


class ModelRoutingTrustStore(_ContractModel):
    trust_version: Literal[1]
    entries: list[ModelRoutingTrustEntry]
    revoked_key_ids: list[StrictStr] = Field(default_factory=list)
    revoked_jtis: list[StrictStr] = Field(default_factory=list)


class ModelRoutingPolicyError(ValueError):
    """Stable, payload-free verification failure."""

    def __init__(self, code: str, detail: str | None = None) -> None:
        self.code = code
        self.detail = detail
        super().__init__(code if detail is None else f"{code}: {detail}")


class ModelRoutingPolicyActivationError(ModelRoutingPolicyError):
    def __init__(
        self,
        code: str,
        *,
        candidate_error_code: str | None = None,
        last_known_good_error_code: str | None = None,
    ) -> None:
        self.candidate_error_code = candidate_error_code
        self.last_known_good_error_code = last_known_good_error_code
        detail = ", ".join(
            item
            for item in (
                f"candidate={candidate_error_code}" if candidate_error_code else "",
                (
                    f"last_known_good={last_known_good_error_code}"
                    if last_known_good_error_code
                    else ""
                ),
            )
            if item
        )
        super().__init__(code, detail or None)


@dataclass(frozen=True)
class VerifiedModelRoutingPolicy:
    envelope: ModelRoutingPolicyEnvelope
    claims: ModelRoutingPolicyClaims
    digest: str


@dataclass(frozen=True)
class ActivatedModelRoutingPolicy:
    verified: VerifiedModelRoutingPolicy
    source: Literal["candidate", "last-known-good"]
    candidate_error_code: str | None = None

    @property
    def policy_id(self) -> str:
        return self.verified.claims.policy_id

    @property
    def revision(self) -> int:
        return self.verified.claims.revision

    @property
    def digest(self) -> str:
        return self.verified.digest


def canonical_json(value: Any) -> str:
    """Match the platform's recursively sorted JSON.stringify contract."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def model_routing_policy_digest(signed_payload: str) -> str:
    return f"sha256:{hashlib.sha256(signed_payload.encode('utf-8')).hexdigest()}"


def _non_empty(value: str, *, code: str) -> str:
    if not value or value != value.strip():
        raise ModelRoutingPolicyError(code)
    return value


def _parse_timestamp(value: str) -> datetime:
    if not _CANONICAL_TIMESTAMP.fullmatch(value):
        raise ModelRoutingPolicyError("invalid_validity_window")
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=UTC)
    except ValueError as exc:
        raise ModelRoutingPolicyError("invalid_validity_window") from exc


def _validate_routes(routes: list[ModelRoutingRoute]) -> None:
    if not 1 <= len(routes) <= MAX_MODEL_ROUTING_ROUTES:
        raise ModelRoutingPolicyError("invalid_routes")

    route_ids: set[str] = set()
    selectors: set[str] = set()
    wildcard_seen = False
    for index, route in enumerate(routes):
        route_id = _non_empty(route.route_id, code="invalid_routes")
        requested_model = _non_empty(route.requested_model, code="invalid_routes")
        primary_model = _non_empty(route.primary_model, code="invalid_routes")
        if route_id in route_ids or requested_model in selectors or wildcard_seen:
            raise ModelRoutingPolicyError("invalid_routes")
        route_ids.add(route_id)
        selectors.add(requested_model)
        wildcard_seen = requested_model == "*"
        if wildcard_seen and index != len(routes) - 1:
            raise ModelRoutingPolicyError("invalid_routes")
        if len(route.fallback_models) > MAX_MODEL_ROUTING_FALLBACKS:
            raise ModelRoutingPolicyError("invalid_routes")
        fallbacks = [
            _non_empty(fallback, code="invalid_routes") for fallback in route.fallback_models
        ]
        if len(set(fallbacks)) != len(fallbacks) or primary_model in fallbacks:
            raise ModelRoutingPolicyError("invalid_routes")
        for limit in (
            route.limits.max_input_tokens,
            route.limits.max_output_tokens,
            route.limits.max_requests_per_minute,
            route.limits.max_cost_micros_per_request,
        ):
            if limit is not None and (limit <= 0 or limit > MAX_SAFE_INTEGER):
                raise ModelRoutingPolicyError("invalid_routes")


def _validate_trust_store(trust_store: ModelRoutingTrustStore) -> None:
    identities: set[tuple[str, str]] = set()
    for entry in trust_store.entries:
        identity = (
            _non_empty(entry.issuer, code="malformed_trust_store"),
            _non_empty(entry.key_id, code="malformed_trust_store"),
        )
        if identity in identities:
            raise ModelRoutingPolicyError("malformed_trust_store")
        identities.add(identity)
        if (
            not entry.allowed_org_ids
            or not entry.allowed_environments
            or len(set(entry.allowed_org_ids)) != len(entry.allowed_org_ids)
            or len(set(entry.allowed_environments)) != len(entry.allowed_environments)
        ):
            raise ModelRoutingPolicyError("malformed_trust_store")
        for org_id in entry.allowed_org_ids:
            _non_empty(org_id, code="malformed_trust_store")
    if len(set(trust_store.revoked_key_ids)) != len(trust_store.revoked_key_ids):
        raise ModelRoutingPolicyError("malformed_trust_store")
    if len(set(trust_store.revoked_jtis)) != len(trust_store.revoked_jtis):
        raise ModelRoutingPolicyError("malformed_trust_store")
    for value in (*trust_store.revoked_key_ids, *trust_store.revoked_jtis):
        _non_empty(value, code="malformed_trust_store")


def _load_json_model(
    path: Path,
    model_type: type[_ContractModel],
    *,
    max_bytes: int,
    missing_code: str,
    malformed_code: str,
) -> _ContractModel:
    if not path.exists():
        raise ModelRoutingPolicyError(missing_code)
    try:
        with path.open("rb") as handle:
            encoded = handle.read(max_bytes + 1)
        if not encoded or len(encoded) > max_bytes:
            raise ModelRoutingPolicyError(malformed_code)
        raw = json.loads(encoded.decode("utf-8"))
        return model_type.model_validate(raw, strict=True)
    except ModelRoutingPolicyError:
        raise
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        RecursionError,
        ValidationError,
    ) as exc:
        raise ModelRoutingPolicyError(malformed_code) from exc


def load_model_routing_trust_store(
    path: Path | str,
    *,
    max_bytes: int = 1_048_576,
) -> ModelRoutingTrustStore:
    model = _load_json_model(
        Path(path),
        ModelRoutingTrustStore,
        max_bytes=max_bytes,
        missing_code="trust_store_missing",
        malformed_code="malformed_trust_store",
    )
    assert isinstance(model, ModelRoutingTrustStore)
    _validate_trust_store(model)
    return model


def load_model_routing_envelope(
    path: Path | str,
    *,
    max_bytes: int = 1_048_576,
) -> ModelRoutingPolicyEnvelope:
    model = _load_json_model(
        Path(path),
        ModelRoutingPolicyEnvelope,
        max_bytes=max_bytes,
        missing_code="policy_missing",
        malformed_code="malformed_envelope",
    )
    assert isinstance(model, ModelRoutingPolicyEnvelope)
    return model


def _resolve_trust_entry(
    envelope: ModelRoutingPolicyEnvelope,
    trust_store: ModelRoutingTrustStore,
) -> ModelRoutingTrustEntry:
    if envelope.key_id in trust_store.revoked_key_ids:
        raise ModelRoutingPolicyError("revoked_key")
    for entry in trust_store.entries:
        if entry.issuer == envelope.issuer and entry.key_id == envelope.key_id:
            return entry
    raise ModelRoutingPolicyError("untrusted_key")


def _verify_signature(
    envelope: ModelRoutingPolicyEnvelope,
    entry: ModelRoutingTrustEntry,
) -> None:
    try:
        key_der = base64.b64decode(entry.public_key_spki_der_base64, validate=True)
        key = load_der_public_key(key_der)
        if not isinstance(key, Ed25519PublicKey):
            raise TypeError("not an Ed25519 public key")
    except (binascii.Error, TypeError, ValueError) as exc:
        raise ModelRoutingPolicyError("malformed_trust_store") from exc
    try:
        signature = base64.b64decode(envelope.signature, validate=True)
        key.verify(signature, envelope.signed_payload.encode("utf-8"))
    except (binascii.Error, InvalidSignature, ValueError) as exc:
        raise ModelRoutingPolicyError("invalid_signature") from exc


def verify_model_routing_policy(
    envelope: ModelRoutingPolicyEnvelope,
    trust_store: ModelRoutingTrustStore,
    *,
    now: datetime | None = None,
    expected_audience: str = MODEL_ROUTING_POLICY_AUDIENCE,
    expected_environment: str | None = None,
    expected_org_id: str | None = None,
    clock_skew_seconds: int = 0,
) -> VerifiedModelRoutingPolicy:
    if clock_skew_seconds < 0:
        raise ModelRoutingPolicyError("invalid_clock_skew")
    entry = _resolve_trust_entry(envelope, trust_store)
    _verify_signature(envelope, entry)

    try:
        raw_claims = json.loads(envelope.signed_payload)
        if not isinstance(raw_claims, dict):
            raise ValueError("claims are not an object")
    except (json.JSONDecodeError, ValueError) as exc:
        raise ModelRoutingPolicyError("malformed_claims") from exc
    try:
        if canonical_json(raw_claims) != envelope.signed_payload:
            raise ModelRoutingPolicyError("non_canonical_payload")
    except (TypeError, ValueError) as exc:
        if isinstance(exc, ModelRoutingPolicyError):
            raise
        raise ModelRoutingPolicyError("malformed_claims") from exc

    try:
        claims = ModelRoutingPolicyClaims.model_validate(raw_claims, strict=True)
    except ValidationError as exc:
        raise ModelRoutingPolicyError("malformed_claims") from exc

    for value in (
        claims.issuer,
        claims.key_id,
        claims.subject,
        claims.org_id,
        claims.audience,
        claims.policy_id,
        claims.release_id,
        claims.deployment_id,
        claims.jti,
        claims.revocation_ref,
    ):
        _non_empty(value, code="malformed_claims")
    if (
        envelope.policy_id != claims.policy_id
        or envelope.policy_version != claims.policy_version
        or envelope.issuer != claims.issuer
        or envelope.key_id != claims.key_id
        or claims.subject != f"model-routing-policy:{claims.policy_id}"
    ):
        raise ModelRoutingPolicyError("envelope_claim_mismatch")
    if claims.audience != expected_audience:
        raise ModelRoutingPolicyError("audience_mismatch")
    if claims.org_id not in entry.allowed_org_ids:
        raise ModelRoutingPolicyError("org_not_allowed")
    if claims.target_environment not in entry.allowed_environments:
        raise ModelRoutingPolicyError("environment_not_allowed")
    if expected_org_id and claims.org_id != expected_org_id:
        raise ModelRoutingPolicyError("org_mismatch")
    if expected_environment and claims.target_environment != expected_environment:
        raise ModelRoutingPolicyError("environment_mismatch")
    if claims.jti in trust_store.revoked_jtis:
        raise ModelRoutingPolicyError("revoked_policy")
    if claims.revision < 1 or claims.revision > MAX_SAFE_INTEGER:
        raise ModelRoutingPolicyError("invalid_revision")
    _validate_routes(claims.routes)

    issued_at = _parse_timestamp(claims.issued_at)
    not_before = _parse_timestamp(claims.not_before)
    expires_at = _parse_timestamp(claims.expires_at)
    offline_lease_expires_at = _parse_timestamp(claims.offline_lease_expires_at)
    if (
        not_before < issued_at
        or expires_at <= not_before
        or offline_lease_expires_at < not_before
        or offline_lease_expires_at > expires_at
    ):
        raise ModelRoutingPolicyError("invalid_validity_window")

    checked_at = now or datetime.now(UTC)
    if checked_at.tzinfo is None:
        raise ModelRoutingPolicyError("invalid_verification_time")
    checked_at = checked_at.astimezone(UTC)
    skew = timedelta(seconds=clock_skew_seconds)
    if checked_at + skew < not_before:
        raise ModelRoutingPolicyError("not_yet_valid")
    if checked_at - skew > expires_at:
        raise ModelRoutingPolicyError("expired")
    if checked_at - skew > offline_lease_expires_at:
        raise ModelRoutingPolicyError("offline_lease_expired")

    return VerifiedModelRoutingPolicy(
        envelope=envelope,
        claims=claims,
        digest=model_routing_policy_digest(envelope.signed_payload),
    )


def _persist_last_known_good(
    path: Path,
    envelope: ModelRoutingPolicyEnvelope,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json(envelope.model_dump(by_alias=True)).encode("utf-8") + b"\n"
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


class ModelRoutingPolicyStore:
    def __init__(
        self,
        *,
        candidate_path: Path | str,
        last_known_good_path: Path | str,
        trust_store_path: Path | str,
        required: bool,
        expected_audience: str = MODEL_ROUTING_POLICY_AUDIENCE,
        expected_environment: str | None = None,
        expected_org_id: str | None = None,
        clock_skew_seconds: int = 0,
        max_file_bytes: int = 1_048_576,
    ) -> None:
        self.candidate_path = Path(candidate_path)
        self.last_known_good_path = Path(last_known_good_path)
        self.trust_store_path = Path(trust_store_path)
        self.required = required
        self.expected_audience = expected_audience
        self.expected_environment = expected_environment or None
        self.expected_org_id = expected_org_id or None
        self.clock_skew_seconds = clock_skew_seconds
        self.max_file_bytes = max_file_bytes

    def _verify_path(
        self,
        path: Path,
        trust_store: ModelRoutingTrustStore,
        *,
        now: datetime | None,
    ) -> VerifiedModelRoutingPolicy:
        envelope = load_model_routing_envelope(path, max_bytes=self.max_file_bytes)
        return verify_model_routing_policy(
            envelope,
            trust_store,
            now=now,
            expected_audience=self.expected_audience,
            expected_environment=self.expected_environment,
            expected_org_id=self.expected_org_id,
            clock_skew_seconds=self.clock_skew_seconds,
        )

    @staticmethod
    def _validate_progression(
        candidate: VerifiedModelRoutingPolicy,
        previous: VerifiedModelRoutingPolicy,
    ) -> None:
        if candidate.claims.policy_id != previous.claims.policy_id:
            raise ModelRoutingPolicyError("policy_identity_mismatch")
        if candidate.claims.revision < previous.claims.revision:
            raise ModelRoutingPolicyError("revision_rollback")
        if (
            candidate.claims.revision == previous.claims.revision
            and candidate.digest != previous.digest
        ):
            raise ModelRoutingPolicyError("revision_conflict")

    def activate(self, *, now: datetime | None = None) -> ActivatedModelRoutingPolicy | None:
        candidate_exists = self.candidate_path.exists()
        last_known_good_exists = self.last_known_good_path.exists()
        if not candidate_exists and not last_known_good_exists:
            if self.required:
                raise ModelRoutingPolicyActivationError("policy_required")
            return None

        try:
            trust_store = load_model_routing_trust_store(
                self.trust_store_path,
                max_bytes=self.max_file_bytes,
            )
        except ModelRoutingPolicyError as exc:
            raise ModelRoutingPolicyActivationError(
                "no_valid_policy",
                candidate_error_code=exc.code,
            ) from exc

        candidate_error: ModelRoutingPolicyError | None = None
        if candidate_exists:
            try:
                verified = self._verify_path(self.candidate_path, trust_store, now=now)
                if last_known_good_exists:
                    try:
                        previous = self._verify_path(
                            self.last_known_good_path,
                            trust_store,
                            now=now,
                        )
                    except ModelRoutingPolicyError:
                        previous = None
                    if previous is not None:
                        self._validate_progression(verified, previous)
                _persist_last_known_good(self.last_known_good_path, verified.envelope)
                return ActivatedModelRoutingPolicy(verified=verified, source="candidate")
            except ModelRoutingPolicyError as exc:
                candidate_error = exc
            except OSError as exc:
                candidate_error = ModelRoutingPolicyError("last_known_good_write_failed")
                candidate_error.__cause__ = exc

        last_known_good_error: ModelRoutingPolicyError | None = None
        if last_known_good_exists:
            try:
                verified = self._verify_path(
                    self.last_known_good_path,
                    trust_store,
                    now=now,
                )
                return ActivatedModelRoutingPolicy(
                    verified=verified,
                    source="last-known-good",
                    candidate_error_code=(candidate_error.code if candidate_error else None),
                )
            except ModelRoutingPolicyError as exc:
                last_known_good_error = exc

        raise ModelRoutingPolicyActivationError(
            "no_valid_policy",
            candidate_error_code=(candidate_error.code if candidate_error else "policy_missing"),
            last_known_good_error_code=(
                last_known_good_error.code if last_known_good_error else "policy_missing"
            ),
        )


def activate_model_routing_policy_from_settings(
    *,
    now: datetime | None = None,
) -> ActivatedModelRoutingPolicy | None:
    return ModelRoutingPolicyStore(
        candidate_path=settings.model_routing_policy_file,
        last_known_good_path=settings.model_routing_last_known_good_file,
        trust_store_path=settings.model_routing_trust_store_file,
        required=settings.model_routing_policy_required,
        expected_audience=settings.model_routing_expected_audience,
        expected_environment=settings.model_routing_expected_environment,
        expected_org_id=settings.model_routing_expected_org_id,
        clock_skew_seconds=settings.model_routing_clock_skew_seconds,
        max_file_bytes=settings.model_routing_max_file_bytes,
    ).activate(now=now)
