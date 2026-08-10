"""Signed, short-TTL runtime-control leases carried on the observation reply.

The engine has no inbound control channel and does not grow one here. The
model-plane observer already POSTs a payload-free observation on a timer; the
control plane answers that POST with a ``runtimeControls`` delivery block, and
this module verifies the lease inside it locally against the same Ed25519 trust
store the routing policy uses (``model_routing.resolve_trust_entry`` /
``verify_signed_envelope``). Nothing on the inference request path calls out.

Scope is typed and this runtime declares what it can enforce. A lease names
subjects at one of five scopes; the only identities this engine holds are the
org and deployment it is configured as and the tenant a request authenticated
as, so ``org``, ``deployment`` and ``tenant`` are enforced and ``solution`` and
``agent`` are ignored — the engine has no way to resolve either and would be
guessing. The acknowledgement declares that enforceable set and counts only the
matched controls this replica is actually refusing for — never the controls the
lease names — so a control the fleet cannot enforce renders as unenforced
rather than as a green tick.

Enforcement precedence is fixed: ``mode`` is decided first, so an advisory
lease can never refuse a request under any staleness condition; then an
unmatched request serves; then a matched quarantine refuses while the lease is
live; and only a matched quarantine on an expired lease consults the stale
action. A lease this replica matches nothing in stops nothing, expired or not.

``staleAction`` decides what happens to enforcement, never to traffic
directly. ``continue`` continues *enforcing* the last verified lease, so a
quarantine that was in force stays in force past expiry; it never means resume
serving, because a quarantine that lapsed when the issuer became unreachable
would make anyone who can partition the network a kill-switch override.
``stop`` is the wider opt-in: while the lease is stale it also refuses the
subjects the lease named ``serving``, so every subject that lease governed here
stops. Neither reaches a request the lease matched no control for.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    StrictStr,
    ValidationError,
    model_validator,
)
from pydantic.alias_generators import to_camel

from .model_routing import (
    MODEL_ROUTING_POLICY_CANONICALIZATION,
    ModelRoutingPolicyError,
    ModelRoutingTrustStore,
    SignedContractModel,
    canonical_json,
    load_model_routing_trust_store,
    model_routing_policy_digest,
    parse_canonical_timestamp,
    require_non_empty,
    resolve_trust_entry,
    verify_signed_envelope,
)
from .observability import get_logger

RUNTIME_CONTROL_LEASE_TYPE = "orchestra.runtime-control-lease"
RUNTIME_CONTROL_LEASE_VERSION_V1 = 1
#: Key the control plane puts its lease under, on the reply to an observation.
RUNTIME_CONTROL_DELIVERY_KEY = "runtimeControls"
#: Key the engine puts its acknowledgement under, on the observation itself. A
#: separate name from the delivery block because they are different objects
#: travelling in opposite directions; intake ignores it where it is unknown.
RUNTIME_CONTROL_ACK_KEY = "runtimeControlAck"
MAX_RUNTIME_CONTROLS = 256
MAX_LEASE_REVISION = 2_147_483_647

RuntimeControlScopeName = Literal["org", "tenant", "deployment", "solution", "agent"]
RuntimeControlMode = Literal["advisory", "enforcing"]
RuntimeControlStaleAction = Literal["continue", "stop"]

#: Scopes whose subject this engine can resolve, and therefore the only scopes
#: it reports itself able to enforce. ``org`` and ``deployment`` resolve against
#: this replica's own configured identity; ``tenant`` resolves against the
#: identity a request authenticated as.
ENFORCEABLE_SCOPES: tuple[RuntimeControlScopeName, ...] = ("deployment", "org", "tenant")
#: Scopes this engine ignores. It authenticates tenants and holds no agent or
#: solution identity, so a control at either scope is reported as ignored.
IGNORED_SCOPES: tuple[RuntimeControlScopeName, ...] = ("agent", "solution")

QUARANTINED_ERROR_CODE = "model_plane_quarantined"
STALE_CONTROL_ERROR_CODE = "model_plane_control_lease_stale"

#: Refusal codes that mean the delivered lease could not be read at all, as
#: opposed to being read and then rejected. Only these raise ``leaseParseFailed``
#: on the acknowledgement.
_LEASE_PARSE_FAILURE_CODES = frozenset(
    {"lease_parse_failed", "malformed_claims", "non_canonical_payload"}
)

log = get_logger("model_plane.runtime_control")


class ModelPlaneRuntimeControlError(ValueError):
    """Stable, payload-free runtime-control failure."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class ModelPlaneRuntimeControlConfigError(ModelPlaneRuntimeControlError):
    """An enabled runtime-control seam is incomplete or unsafe to run."""


class RuntimeControlEntry(SignedContractModel):
    scope: RuntimeControlScopeName
    subject_id: StrictStr
    state: Literal["quarantined", "serving"]
    # The control plane omits it on controls it has no operator text for, so
    # requiring it here would refuse leases it legitimately issues.
    reason_code: StrictStr | None = None


class RuntimeControlLeaseClaims(SignedContractModel):
    artifact_type: Literal[RUNTIME_CONTROL_LEASE_TYPE]
    lease_version: Literal[RUNTIME_CONTROL_LEASE_VERSION_V1]
    issuer: StrictStr
    key_id: StrictStr
    org_id: StrictStr
    target_environment: Literal["dev", "test", "staging", "prod"]
    lease_id: StrictStr
    revision: StrictInt
    issued_at: StrictStr
    not_before: StrictStr
    expires_at: StrictStr
    mode: RuntimeControlMode
    stale_action: RuntimeControlStaleAction
    controls: list[RuntimeControlEntry]
    jti: StrictStr


class RuntimeControlLeaseEnvelope(SignedContractModel):
    artifact_type: Literal[RUNTIME_CONTROL_LEASE_TYPE]
    lease_version: Literal[RUNTIME_CONTROL_LEASE_VERSION_V1]
    algorithm: Literal["ed25519"]
    canonicalization: Literal[MODEL_ROUTING_POLICY_CANONICALIZATION]
    issuer: StrictStr
    key_id: StrictStr
    signed_payload: StrictStr
    signature: StrictStr
    signed: Literal[True]


@dataclass(frozen=True)
class VerifiedRuntimeControlLease:
    envelope: RuntimeControlLeaseEnvelope
    claims: RuntimeControlLeaseClaims
    digest: str
    not_before: datetime
    expires_at: datetime

    @property
    def lease_id(self) -> str:
        return self.claims.lease_id

    @property
    def revision(self) -> int:
        return self.claims.revision


class RuntimeControlAcknowledgement(BaseModel):
    """The acknowledgement direction of the wire contract, field for field.

    The nine fields and their camelCase spellings are pinned by contract §7b;
    the intake reads exactly these, so nothing is added to or renamed in this
    object without changing that document first. It travels as an additive key
    on the observation, and this replica's own operator detail — the lease jti,
    when it was applied — stays on the admin status rather than riding here.

    ``lease_id`` null means this replica holds no lease at all, which is a
    different operator situation from holding an expired one. ``enforcement``
    is what this replica *did*, not what the lease asked for: a lease naming
    controls this replica matched none of reports ``advisory``, because
    reporting ``enforcing`` would let the control plane count a replica towards
    a quarantine it does not apply.

    Every member is required and none defaults. A missing one would let a
    caller build an object that says less than it appears to — the intake
    refuses those rather than filling them in, so this side cannot produce
    one. For the same reason ``enforcing`` must name the lease it is
    enforcing: §1 is that a surface may only claim enforcement it can prove.
    """

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    lease_id: str | None
    revision: int | None
    enforcement: Literal["advisory", "enforcing"]
    enforceable_scopes: list[str]
    enforced_control_count: int
    ignored_control_count: int
    stale: bool
    lease_expires_at: str | None
    lease_parse_failed: bool

    @model_validator(mode="after")
    def _enforcement_names_its_lease(self) -> RuntimeControlAcknowledgement:
        if self.enforcement == "enforcing" and (
            self.lease_id is None or self.revision is None
        ):
            raise ValueError("an enforcing acknowledgement must name its lease")
        return self


def unheld_acknowledgement(
    *,
    lease_parse_failed: bool = False,
) -> RuntimeControlAcknowledgement:
    """The §7b object for a replica holding no lease.

    It still declares the scopes this runtime could resolve, because that
    describes the runtime rather than the lease.
    """
    return RuntimeControlAcknowledgement(
        lease_id=None,
        revision=None,
        enforcement="advisory",
        enforceable_scopes=list(ENFORCEABLE_SCOPES),
        enforced_control_count=0,
        ignored_control_count=0,
        stale=False,
        lease_expires_at=None,
        lease_parse_failed=lease_parse_failed,
    )


class ModelPlaneRuntimeControlStatus(BaseModel):
    """Operator-facing status. Never sent on the observation wire."""

    object: Literal["model_plane_runtime_control.status"] = (
        "model_plane_runtime_control.status"
    )
    enabled: bool
    acknowledgement: RuntimeControlAcknowledgement = Field(
        default_factory=unheld_acknowledgement
    )
    enforceable_scopes: list[str] = list(ENFORCEABLE_SCOPES)
    ignored_scopes: list[str] = list(IGNORED_SCOPES)
    stale_action_policy: str | None = None
    lease_stale_action: str | None = None
    effective_stale_action: str | None = None
    lease_mode: str | None = None
    # Whether the held lease quarantines anything this replica matched, which
    # is a different question from whether that quarantine is being enforced:
    # an advisory or expired lease still has a desired state.
    matched_state: Literal["serving", "quarantined"] = "serving"
    control_count: int = 0
    matched_control_count: int = 0
    ignored_control_count: int = 0
    lease_digest: str | None = None
    # Operator detail the wire acknowledgement deliberately does not carry.
    lease_jti: str | None = None
    applied_at: str | None = None
    last_delivery_status: str | None = None
    last_delivery_code: str | None = None
    leases_accepted_total: int = 0
    leases_rejected_total: int = 0
    refusals_total: int = 0
    last_refresh_error_code: str | None = None


@dataclass(frozen=True)
class RuntimeControlRefusal:
    """Typed, payload-free reason this replica is refusing a request.

    ``scope`` names the scope that matched, never the subject that was matched:
    the caller learns that a control applies, not who else one applies to.
    """

    code: str
    scope: RuntimeControlScopeName | None


@dataclass(frozen=True)
class RuntimeControlConfig:
    trust_store_path: Path
    expected_environment: str
    expected_org_id: str
    expected_deployment_id: str
    max_lease_seconds: int
    stale_action_policy: Literal["lease", "continue", "stop"]
    clock_skew_seconds: int
    max_response_bytes: int
    max_trust_store_bytes: int


@dataclass(frozen=True)
class _Snapshot:
    """Immutable projection read by the request path without a lock.

    ``lease_deadline`` is the lease's own ``expiresAt`` as a Unix timestamp, so
    the request path compares two floats on the same clock the lease was
    written against rather than against an interval measured at apply time.
    """

    lease_deadline: float = 0.0
    # True only when this replica matched at least one quarantine it can
    # enforce, in a lease whose mode is ``enforcing``. Everything else — no
    # lease, an advisory lease, a lease this replica matched nothing in, a
    # lease naming only ``serving`` — is enforcement of nothing, and the
    # request path can stop at this one boolean.
    enforcing: bool = False
    lease_mode: RuntimeControlMode | None = None
    stale_action: RuntimeControlStaleAction | None = None
    # Set by an org- or deployment-scoped quarantine whose subject is this
    # replica's own identity: it matches every request, not one tenant. The
    # value is the scope that matched, which is all a refusal ever discloses.
    replica_quarantine_scope: Literal["org", "deployment"] | None = None
    quarantined_tenants: frozenset[str] = frozenset()
    # Every subject this replica matched, whatever state the control named.
    # ``stop`` stops the lease's ``serving`` subjects too once it is stale;
    # ``continue`` never touches them.
    replica_governed_scope: Literal["org", "deployment"] | None = None
    governed_tenants: frozenset[str] = frozenset()
    control_count: int = 0
    matched_control_count: int = 0
    # Quarantines whose subject is this replica's own org or deployment id.
    # Those are matched the moment the lease is applied; a tenant-scoped one
    # is only matched once a request from that tenant arrives.
    replica_quarantine_count: int = 0
    ignored_control_count: int = 0
    lease_id: str | None = None
    revision: int | None = None
    jti: str | None = None
    digest: str | None = None
    expires_at: datetime | None = None
    applied_at: datetime | None = None


@dataclass(frozen=True)
class _Projection:
    """One lease's controls, reduced to this replica's enforceable subset."""

    replica_quarantine_scope: Literal["org", "deployment"] | None
    quarantined_tenants: frozenset[str]
    replica_governed_scope: Literal["org", "deployment"] | None
    governed_tenants: frozenset[str]
    matched_control_count: int
    matched_quarantine_count: int
    replica_quarantine_count: int
    ignored_control_count: int


def canonical_instant(value: datetime) -> str:
    """Render the exact ``Date.toISOString()`` spelling the intake round-trips."""
    return value.astimezone(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def load_model_plane_runtime_control_config(settings) -> RuntimeControlConfig | None:
    """Validate an enabled seam and prove the trust store is readable.

    Returns ``None`` when the deployment has not opted in, which is the state
    in which every other code path in this module is never reached.
    """
    if not settings.model_plane_runtime_control_enabled:
        return None
    if not settings.model_plane_observation_enabled:
        raise ModelPlaneRuntimeControlConfigError("observation_reporter_required")

    expected_environment = settings.model_plane_observation_target_environment.strip()
    expected_deployment_id = settings.model_plane_observation_deployment_id.strip()
    expected_org_id = settings.model_routing_expected_org_id.strip()
    if not expected_environment or not expected_deployment_id:
        raise ModelPlaneRuntimeControlConfigError("observation_scope_required")
    if not expected_org_id:
        # Without it an org-scoped control has no subject to match against, and
        # the replica would silently ignore the broadest control the contract has.
        raise ModelPlaneRuntimeControlConfigError("expected_org_id_required")

    config = RuntimeControlConfig(
        trust_store_path=Path(settings.model_plane_runtime_control_trust_store_file),
        expected_environment=expected_environment,
        expected_org_id=expected_org_id,
        expected_deployment_id=expected_deployment_id,
        max_lease_seconds=settings.model_plane_runtime_control_max_lease_seconds,
        stale_action_policy=settings.model_plane_runtime_control_stale_action,
        clock_skew_seconds=settings.model_routing_clock_skew_seconds,
        max_response_bytes=settings.model_plane_runtime_control_max_response_bytes,
        max_trust_store_bytes=settings.model_routing_max_file_bytes,
    )
    try:
        load_model_routing_trust_store(
            config.trust_store_path,
            max_bytes=config.max_trust_store_bytes,
        )
    except ModelRoutingPolicyError as exc:
        raise ModelPlaneRuntimeControlConfigError(exc.code) from exc
    return config


def _validate_controls(claims: RuntimeControlLeaseClaims) -> None:
    if len(claims.controls) > MAX_RUNTIME_CONTROLS:
        raise ModelPlaneRuntimeControlError("invalid_controls")
    seen: set[tuple[str, str]] = set()
    for control in claims.controls:
        try:
            subject_id = require_non_empty(control.subject_id, code="invalid_controls")
            if control.reason_code is not None:
                require_non_empty(control.reason_code, code="invalid_controls")
        except ModelRoutingPolicyError as exc:
            raise ModelPlaneRuntimeControlError(exc.code) from exc
        identity = (control.scope, subject_id)
        if identity in seen:
            raise ModelPlaneRuntimeControlError("invalid_controls")
        seen.add(identity)


def verify_runtime_control_lease(
    envelope: RuntimeControlLeaseEnvelope,
    trust_store: ModelRoutingTrustStore,
    config: RuntimeControlConfig,
    *,
    now: datetime | None = None,
) -> VerifiedRuntimeControlLease:
    """Verify a lease exactly as the routing policy is verified, plus its own rules.

    Key purpose is checked here and not only where leases are signed: a check
    only the issuer performs is not a check, because a compromised or
    misconfigured issuer is the case it exists for. The trust entry must name
    this artifact type in ``allowedArtifactTypes``, so a routing, bundle or
    promotion key is refused even though its signature is perfectly valid.
    """
    try:
        entry = resolve_trust_entry(envelope, trust_store)
        verify_signed_envelope(envelope, entry)
    except ModelRoutingPolicyError as exc:
        raise ModelPlaneRuntimeControlError(exc.code) from exc

    if (
        entry.allowed_artifact_types is None
        or RUNTIME_CONTROL_LEASE_TYPE not in entry.allowed_artifact_types
    ):
        raise ModelPlaneRuntimeControlError("signing_key_purpose_denied")

    try:
        raw_claims = json.loads(envelope.signed_payload)
        if not isinstance(raw_claims, dict):
            raise ValueError("claims are not an object")
    except (json.JSONDecodeError, ValueError) as exc:
        raise ModelPlaneRuntimeControlError("malformed_claims") from exc
    try:
        if canonical_json(raw_claims) != envelope.signed_payload:
            raise ModelPlaneRuntimeControlError("non_canonical_payload")
    except (TypeError, ValueError) as exc:
        if isinstance(exc, ModelPlaneRuntimeControlError):
            raise
        raise ModelPlaneRuntimeControlError("malformed_claims") from exc

    try:
        claims = RuntimeControlLeaseClaims.model_validate(raw_claims, strict=True)
    except ValidationError as exc:
        raise ModelPlaneRuntimeControlError("malformed_claims") from exc

    try:
        for value in (
            claims.issuer,
            claims.key_id,
            claims.org_id,
            claims.lease_id,
            claims.jti,
        ):
            require_non_empty(value, code="malformed_claims")
    except ModelRoutingPolicyError as exc:
        raise ModelPlaneRuntimeControlError(exc.code) from exc

    if (
        envelope.artifact_type != claims.artifact_type
        or envelope.lease_version != claims.lease_version
        or envelope.issuer != claims.issuer
        or envelope.key_id != claims.key_id
    ):
        raise ModelPlaneRuntimeControlError("envelope_claim_mismatch")
    if claims.org_id not in entry.allowed_org_ids:
        raise ModelPlaneRuntimeControlError("org_not_allowed")
    if claims.target_environment not in entry.allowed_environments:
        raise ModelPlaneRuntimeControlError("environment_not_allowed")
    if claims.org_id != config.expected_org_id:
        raise ModelPlaneRuntimeControlError("org_mismatch")
    if claims.target_environment != config.expected_environment:
        raise ModelPlaneRuntimeControlError("environment_mismatch")
    if claims.jti in trust_store.revoked_jtis:
        raise ModelPlaneRuntimeControlError("revoked_lease")
    if not 0 <= claims.revision <= MAX_LEASE_REVISION:
        raise ModelPlaneRuntimeControlError("invalid_revision")
    _validate_controls(claims)

    try:
        issued_at = parse_canonical_timestamp(claims.issued_at)
        not_before = parse_canonical_timestamp(claims.not_before)
        expires_at = parse_canonical_timestamp(claims.expires_at)
    except ModelRoutingPolicyError as exc:
        raise ModelPlaneRuntimeControlError(exc.code) from exc
    if not_before < issued_at or expires_at <= not_before:
        raise ModelPlaneRuntimeControlError("invalid_lease_window")
    if expires_at - not_before > timedelta(seconds=config.max_lease_seconds):
        raise ModelPlaneRuntimeControlError("lease_ttl_too_long")

    checked_at = (now or datetime.now(UTC)).astimezone(UTC)
    skew = timedelta(seconds=config.clock_skew_seconds)
    if checked_at + skew < not_before:
        raise ModelPlaneRuntimeControlError("not_yet_valid")
    if checked_at - skew > expires_at:
        raise ModelPlaneRuntimeControlError("lease_expired")

    return VerifiedRuntimeControlLease(
        envelope=envelope,
        claims=claims,
        digest=model_routing_policy_digest(envelope.signed_payload),
        not_before=not_before,
        expires_at=expires_at,
    )


class ModelPlaneRuntimeControl:
    """Holds the last verified lease and answers the request path from cache.

    ``evaluate`` does a handful of comparisons and at most two set lookups; no
    signature is verified on the request path. Refreshes arrive on the
    observer's cadence and swap an immutable snapshot in one assignment, so a
    reader never sees a half-applied lease and never takes a lock.
    """

    def __init__(self, config: RuntimeControlConfig) -> None:
        self._config = config
        self._snapshot = _Snapshot()
        self._refresh_lock = threading.Lock()
        # High-water mark of every revision accepted for this deployment's
        # (orgId, targetEnvironment) pair, which verification has already
        # pinned. Held apart from the snapshot so ordering is a property of
        # the whole series rather than of whichever lease is current.
        self._highest_revision: int | None = None
        # Tenants named by the lease this replica currently holds that a
        # request has actually arrived for. Cleared whenever the control set
        # changes, so it never outlives the lease that named them.
        self._matched_tenants: frozenset[str] = frozenset()
        self._leases_accepted_total = 0
        self._leases_rejected_total = 0
        self._refusals_total = 0
        self._lease_parse_failed = False
        self._last_refresh_error_code: str | None = None
        self._last_delivery_status: str | None = None
        self._last_delivery_code: str | None = None

    @property
    def config(self) -> RuntimeControlConfig:
        return self._config

    # --- request path -----------------------------------------------------

    def evaluate(
        self,
        *,
        tenant: str,
        now: float | None = None,
    ) -> RuntimeControlRefusal | None:
        """Decide one request in the contract's fixed precedence order.

        1. advisory mode never refuses, whatever the lease's staleness;
        2. an unmatched request serves, including every request when no lease
           is held and every request whose only controls are at a scope this
           replica ignores;
        3. a matched quarantine on a live lease refuses;
        4. a matched quarantine on an expired lease consults the stale action,
           which decides how much is refused and never whether the quarantine
           survives: ``continue`` keeps refusing exactly the quarantined
           subjects, ``stop`` also refuses the subjects the same lease named
           ``serving``;
        5. a matched control in ``serving`` state serves while the lease is
           live, which is what not adding it to the quarantined sets already
           produces.
        """
        snapshot = self._snapshot
        if snapshot.lease_id is None:
            return None
        if tenant in snapshot.governed_tenants:
            self._record_tenant_match(tenant)
        if not snapshot.enforcing:
            return None
        moment = time.time() if now is None else now
        if moment < snapshot.lease_deadline or not self._stop_when_stale(snapshot):
            code = QUARANTINED_ERROR_CODE
            scope = self._quarantined_scope(snapshot, tenant)
        else:
            code = STALE_CONTROL_ERROR_CODE
            scope = self._governed_scope(snapshot, tenant)
        if scope is None:
            return None
        self._refusals_total += 1
        return RuntimeControlRefusal(code=code, scope=scope)

    def _record_tenant_match(self, tenant: str) -> None:
        """Remember a tenant the held lease names that a request has arrived for.

        A tenant is not one of this replica's own identities: unlike its org
        and deployment ids, it is resolved from a request, so until one arrives
        the replica has matched nobody at that scope. Recording it here is what
        lets the acknowledgement count it (§8b) without the replica claiming to
        have matched a subject it has never seen.
        """
        if tenant not in self._matched_tenants:
            self._matched_tenants = self._matched_tenants | {tenant}

    def _matched_quarantined_tenants(self, snapshot: _Snapshot) -> frozenset[str]:
        return self._matched_tenants & snapshot.quarantined_tenants

    def _enforced_control_count(self, snapshot: _Snapshot) -> int:
        """Contract §8b: what this replica is actually refusing for.

        Only controls at a scope it declares enforceable, whose subject it
        matched, and whose state is ``quarantined``: a matched ``serving``
        control is governed, not enforced. Expiry does not reduce it —
        ``continue`` keeps those quarantines in force and ``stop`` widens what
        is refused around them, so reporting zero while still refusing would
        tell the control plane this replica had released subjects it has not.
        """
        if not snapshot.enforcing:
            return 0
        return snapshot.replica_quarantine_count + len(
            self._matched_quarantined_tenants(snapshot)
        )

    @staticmethod
    def _quarantined_scope(
        snapshot: _Snapshot,
        tenant: str,
    ) -> RuntimeControlScopeName | None:
        if snapshot.replica_quarantine_scope is not None:
            return snapshot.replica_quarantine_scope
        if tenant in snapshot.quarantined_tenants:
            return "tenant"
        return None

    @staticmethod
    def _governed_scope(
        snapshot: _Snapshot,
        tenant: str,
    ) -> RuntimeControlScopeName | None:
        """The scope this request matched at, whichever state it was named in."""
        if snapshot.replica_governed_scope is not None:
            return snapshot.replica_governed_scope
        if tenant in snapshot.governed_tenants:
            return "tenant"
        return None

    def _stop_when_stale(self, snapshot: _Snapshot) -> bool:
        policy = self._config.stale_action_policy
        if policy == "lease":
            return snapshot.stale_action == "stop"
        return policy == "stop"

    # --- refresh ----------------------------------------------------------

    def apply_observation_response(
        self,
        body: bytes,
        *,
        now: datetime | None = None,
    ) -> None:
        """Read a runtime-control lease off one accepted observation reply.

        Never raises: a control plane that does not implement this contract, or
        that answers with something unverifiable, leaves the current lease to
        expire on its own clock rather than lifting or inventing one.
        """
        try:
            self._apply(body, now=now)
        except ModelPlaneRuntimeControlError as exc:
            self._reject(exc.code)
        except Exception:  # noqa: BLE001 - a control refresh cannot break reporting
            self._reject("control_refresh_failed")

    def _apply(self, body: bytes, *, now: datetime | None = None) -> None:
        if len(body) > self._config.max_response_bytes:
            raise ModelPlaneRuntimeControlError("control_response_too_large")
        try:
            document = json.loads(body.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ModelPlaneRuntimeControlError("malformed_control_response") from exc
        if not isinstance(document, dict):
            raise ModelPlaneRuntimeControlError("malformed_control_response")
        delivery = document.get(RUNTIME_CONTROL_DELIVERY_KEY)
        if delivery is None:
            # A reply with no delivery block says nothing: this control plane
            # predates the contract. The lease already held keeps governing.
            return
        if not isinstance(delivery, dict):
            raise ModelPlaneRuntimeControlError("malformed_control_delivery")

        status = delivery.get("status")
        code = delivery.get("code")
        self._last_delivery_status = status if isinstance(status, str) else "unknown"
        self._last_delivery_code = code if isinstance(code, str) else None
        if status != "issued":
            # "disabled" and "unavailable" are the control plane declining to
            # deliver, not a release. Nothing is lifted; the held lease runs out.
            return
        try:
            envelope = RuntimeControlLeaseEnvelope.model_validate(
                delivery.get("lease"),
                strict=True,
            )
        except ValidationError as exc:
            raise ModelPlaneRuntimeControlError("lease_parse_failed") from exc

        try:
            trust_store = load_model_routing_trust_store(
                self._config.trust_store_path,
                max_bytes=self._config.max_trust_store_bytes,
            )
        except ModelRoutingPolicyError as exc:
            raise ModelPlaneRuntimeControlError(exc.code) from exc

        lease = verify_runtime_control_lease(envelope, trust_store, self._config, now=now)
        with self._refresh_lock:
            self._validate_progression(lease)
            self._accept(lease, now=now)

    def _validate_progression(self, lease: VerifiedRuntimeControlLease) -> None:
        """Order refreshes globally by revision, never within one ``leaseId``.

        Verification has already pinned ``orgId`` and ``targetEnvironment`` to
        this deployment's own, and the contract makes ``revision`` monotonic for
        that pair, so the highest revision ever accepted here is the whole
        series' watermark. Ordering against it rather than against the current
        lease's own id is what stops a genuine, unexpired, correctly signed
        *older* lease from a different stream lifting a live quarantine — an
        attacker who can replay one captured reply would otherwise only need it
        to carry a different ``leaseId``.
        """
        current = self._snapshot
        watermark = self._highest_revision
        if watermark is None or current.revision is None or current.lease_id is None:
            return
        if lease.revision < watermark or (
            lease.revision == watermark and lease.lease_id != current.lease_id
        ):
            raise ModelPlaneRuntimeControlError("control_lease_out_of_order")
        if (
            lease.revision == current.revision
            and current.expires_at is not None
            and lease.expires_at < current.expires_at
        ):
            # An older instance of the revision this replica already holds:
            # applying it would pull the deadline backwards and shorten a
            # window the issuer has since extended. An identical redelivery —
            # the ordinary case, because the control plane re-attaches the
            # current lease to every observation reply — carries the same
            # expiry and is accepted rather than counted as a rejection.
            raise ModelPlaneRuntimeControlError("lease_replayed")

    def _project(
        self,
        claims: RuntimeControlLeaseClaims,
    ) -> _Projection:
        """Reduce the lease's controls to what this replica can and does match.

        A control at ``solution`` or ``agent`` scope is counted as ignored and
        never reaches the matched sets. A control at ``org`` or ``deployment``
        scope matches only when its subject is this replica's own identity, so
        a quarantine aimed at a sibling deployment counts for nothing here.
        ``matched_quarantine_count`` is the contract's enforced count: a
        matched control in ``serving`` state is governed, not enforced.
        """
        replica_scope: Literal["org", "deployment"] | None = None
        governed_scope: Literal["org", "deployment"] | None = None
        tenants: set[str] = set()
        governed_tenants: set[str] = set()
        matched = 0
        quarantined = 0
        replica_quarantines = 0
        ignored = 0
        for control in claims.controls:
            if control.scope not in ENFORCEABLE_SCOPES:
                ignored += 1
                continue
            if control.scope == "org" and control.subject_id != self._config.expected_org_id:
                continue
            if (
                control.scope == "deployment"
                and control.subject_id != self._config.expected_deployment_id
            ):
                continue
            matched += 1
            if control.scope == "tenant":
                governed_tenants.add(control.subject_id)
            elif control.scope == "deployment":
                governed_scope = "deployment"
            elif governed_scope is None:
                governed_scope = "org"
            if control.state != "quarantined":
                continue
            quarantined += 1
            if control.scope == "tenant":
                tenants.add(control.subject_id)
                continue
            replica_quarantines += 1
            if control.scope == "deployment":
                replica_scope = "deployment"
            elif replica_scope is None:
                replica_scope = "org"
        return _Projection(
            replica_quarantine_scope=replica_scope,
            quarantined_tenants=frozenset(tenants),
            replica_governed_scope=governed_scope,
            governed_tenants=frozenset(governed_tenants),
            matched_control_count=matched,
            matched_quarantine_count=quarantined,
            replica_quarantine_count=replica_quarantines,
            ignored_control_count=ignored,
        )

    def _accept(
        self,
        lease: VerifiedRuntimeControlLease,
        *,
        now: datetime | None = None,
    ) -> None:
        claims = lease.claims
        applied_at = (now or datetime.now(UTC)).astimezone(UTC)
        previous = self._snapshot
        projection = self._project(claims)
        # A lease that quarantines nothing this replica matched enforces
        # nothing here, whatever it asked for, so it is reported and behaves as
        # advisory (contract §8c).
        enforcing = (
            claims.mode == "enforcing" and projection.matched_quarantine_count > 0
        )
        self._snapshot = _Snapshot(
            lease_deadline=lease.expires_at.timestamp(),
            enforcing=enforcing,
            lease_mode=claims.mode,
            stale_action=claims.stale_action,
            replica_quarantine_scope=projection.replica_quarantine_scope,
            quarantined_tenants=projection.quarantined_tenants,
            replica_governed_scope=projection.replica_governed_scope,
            governed_tenants=projection.governed_tenants,
            control_count=len(claims.controls),
            matched_control_count=projection.matched_control_count,
            replica_quarantine_count=projection.replica_quarantine_count,
            ignored_control_count=projection.ignored_control_count,
            lease_id=claims.lease_id,
            revision=claims.revision,
            jti=claims.jti,
            digest=lease.digest,
            expires_at=lease.expires_at,
            applied_at=applied_at,
        )
        # A refresh that carries the same control set — the redelivery every
        # observation tick produces, or a window extension at the same revision
        # — leaves the subjects this replica has already matched matched. Any
        # other lease names its own subjects, so the record starts empty.
        if (previous.lease_id, previous.revision) != (claims.lease_id, claims.revision):
            self._matched_tenants = frozenset()
        log.info(
            "model_plane_runtime_control_lease_accepted",
            lease_id=claims.lease_id,
            lease_jti=claims.jti,
            revision=claims.revision,
            lease_mode=claims.mode,
            control_count=len(claims.controls),
            matched_control_count=projection.matched_control_count,
            ignored_control_count=projection.ignored_control_count,
        )
        self._highest_revision = max(claims.revision, self._highest_revision or 0)
        self._leases_accepted_total += 1
        self._lease_parse_failed = False
        self._last_refresh_error_code = None

    def _reject(self, code: str) -> None:
        self._leases_rejected_total += 1
        # §5.11: a lease this replica could not read is reported as such and
        # never as "no quarantine". Refusals that are not parse failures — a
        # bad signature, a wrong key purpose, an out-of-order revision — leave
        # the flag alone, because the lease was read and then rejected.
        if code in _LEASE_PARSE_FAILURE_CODES:
            self._lease_parse_failed = True
        self._last_refresh_error_code = code
        log.warning(
            "model_plane_runtime_control_lease_rejected",
            error_code=code,
            held_lease_id=self._snapshot.lease_id,
        )

    # --- reporting --------------------------------------------------------

    def acknowledgement(self, *, observed_at: datetime) -> RuntimeControlAcknowledgement:
        """State what this replica can enforce and is enforcing, at one instant.

        ``stale`` is computed against ``observed_at`` because the intake
        cross-checks exactly that: a replica whose staleness disagrees with the
        expiry it reports is refused rather than stored.

        Staleness does not reduce the count. A stale lease is still being
        enforced here — ``continue`` keeps its quarantines in force and
        ``stop`` widens them — so reporting zero would tell the control plane
        this replica had released subjects it is still refusing.
        """
        snapshot = self._snapshot
        parse_failed = self._lease_parse_failed
        if snapshot.lease_id is None or snapshot.expires_at is None:
            return unheld_acknowledgement(lease_parse_failed=parse_failed)
        enforced = self._enforced_control_count(snapshot)
        moment = observed_at.astimezone(UTC)
        return RuntimeControlAcknowledgement(
            lease_id=snapshot.lease_id,
            revision=snapshot.revision,
            # What this replica did, not what the lease asked for: zero
            # enforced controls is reported as advisory however the lease is
            # addressed (contract §8c).
            enforcement=("enforcing" if enforced else "advisory"),
            enforceable_scopes=list(ENFORCEABLE_SCOPES),
            enforced_control_count=enforced,
            ignored_control_count=snapshot.ignored_control_count,
            stale=snapshot.expires_at <= moment,
            lease_expires_at=canonical_instant(snapshot.expires_at),
            lease_parse_failed=parse_failed,
        )

    def status(self, *, now: float | None = None) -> ModelPlaneRuntimeControlStatus:
        snapshot = self._snapshot
        moment = time.time() if now is None else now
        observed_at = datetime.fromtimestamp(moment, tz=UTC)
        return ModelPlaneRuntimeControlStatus(
            enabled=True,
            acknowledgement=self.acknowledgement(observed_at=observed_at),
            enforceable_scopes=list(ENFORCEABLE_SCOPES),
            ignored_scopes=list(IGNORED_SCOPES),
            stale_action_policy=self._config.stale_action_policy,
            lease_stale_action=snapshot.stale_action,
            effective_stale_action=("stop" if self._stop_when_stale(snapshot) else "continue"),
            lease_mode=snapshot.lease_mode,
            # A tenant-scoped control is only matched once a request from that
            # tenant arrives, so a quarantine naming somebody else's tenant
            # leaves this replica reading ``serving``.
            matched_state=(
                "quarantined"
                if snapshot.replica_quarantine_scope is not None
                or self._matched_quarantined_tenants(snapshot)
                else "serving"
            ),
            control_count=snapshot.control_count,
            matched_control_count=snapshot.matched_control_count,
            ignored_control_count=snapshot.ignored_control_count,
            lease_digest=snapshot.digest,
            lease_jti=snapshot.jti,
            applied_at=(
                canonical_instant(snapshot.applied_at)
                if snapshot.applied_at is not None
                else None
            ),
            last_delivery_status=self._last_delivery_status,
            last_delivery_code=self._last_delivery_code,
            leases_accepted_total=self._leases_accepted_total,
            leases_rejected_total=self._leases_rejected_total,
            refusals_total=self._refusals_total,
            last_refresh_error_code=self._last_refresh_error_code,
        )

    def metrics_snapshot(self, *, now: float | None = None) -> dict[str, int | float]:
        snapshot = self._snapshot
        moment = time.time() if now is None else now
        held = snapshot.lease_id is not None
        stale = held and moment >= snapshot.lease_deadline
        return {
            "lease_held": 1 if held else 0,
            # Expiry does not end enforcement, so this stays 1 while a stale
            # quarantine is still being refused.
            "enforcing": 1 if snapshot.enforcing else 0,
            "stale": 1 if stale else 0,
            "stopping_while_stale": (
                1 if (stale and snapshot.enforcing and self._stop_when_stale(snapshot)) else 0
            ),
            "controls": snapshot.control_count,
            "matched_controls": snapshot.matched_control_count,
            "ignored_controls": snapshot.ignored_control_count,
            "enforced_controls": self._enforced_control_count(snapshot),
            "revision": snapshot.revision or 0,
            "lease_seconds_remaining": max(snapshot.lease_deadline - moment, 0.0),
            "leases_accepted_total": self._leases_accepted_total,
            "leases_rejected_total": self._leases_rejected_total,
            "refusals_total": self._refusals_total,
        }


__all__ = [
    "ENFORCEABLE_SCOPES",
    "IGNORED_SCOPES",
    "MAX_RUNTIME_CONTROLS",
    "QUARANTINED_ERROR_CODE",
    "RUNTIME_CONTROL_ACK_KEY",
    "RUNTIME_CONTROL_DELIVERY_KEY",
    "RUNTIME_CONTROL_LEASE_TYPE",
    "RUNTIME_CONTROL_LEASE_VERSION_V1",
    "STALE_CONTROL_ERROR_CODE",
    "ModelPlaneRuntimeControl",
    "ModelPlaneRuntimeControlConfigError",
    "ModelPlaneRuntimeControlError",
    "ModelPlaneRuntimeControlStatus",
    "RuntimeControlAcknowledgement",
    "RuntimeControlConfig",
    "RuntimeControlEntry",
    "RuntimeControlLeaseClaims",
    "RuntimeControlLeaseEnvelope",
    "RuntimeControlRefusal",
    "VerifiedRuntimeControlLease",
    "canonical_instant",
    "load_model_plane_runtime_control_config",
    "unheld_acknowledgement",
    "verify_runtime_control_lease",
]
