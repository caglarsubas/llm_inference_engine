from __future__ import annotations

import base64
import json
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
from fastapi.testclient import TestClient
from pydantic import ValidationError

from inference_engine import auth as auth_mod
from inference_engine.api.state import app_state
from inference_engine.auth import Identity
from inference_engine.config import Settings, settings
from inference_engine.main import app
from inference_engine.model_plane_control import (
    ENFORCEABLE_SCOPES,
    IGNORED_SCOPES,
    RUNTIME_CONTROL_ACK_KEY,
    RUNTIME_CONTROL_LEASE_TYPE,
    ModelPlaneRuntimeControl,
    ModelPlaneRuntimeControlConfigError,
    ModelPlaneRuntimeControlError,
    RuntimeControlAcknowledgement,
    RuntimeControlConfig,
    RuntimeControlLeaseEnvelope,
    canonical_instant,
    load_model_plane_runtime_control_config,
    verify_runtime_control_lease,
)
from inference_engine.model_plane_observer import (
    ModelPlaneObservationReporter,
    build_model_plane_observation,
    load_model_plane_observation_config,
)
from inference_engine.model_routing import canonical_json, load_model_routing_trust_store
from inference_engine.model_routing_runtime import (
    ModelRoutingRateLimiter,
    ModelRoutingRuntimeState,
)
from inference_engine.schemas import ModelInfo, ModelList, UnavailableModel


ISSUER = "prometa-platform/runtime-control-lease"
KEY_ID = "sha256:" + "ab" * 32
DEPLOYMENT_ID = "model-plane-staging-a"
ORG_ID = "org-golden"
ENVIRONMENT = "staging"
TENANT = "orders-runtime"


def _claims(
    *,
    revision: int = 1,
    controls: list[dict] | None = None,
    mode: str = "enforcing",
    stale_action: str = "continue",
    issued_at: datetime | None = None,
    ttl_seconds: int = 300,
    lease_id: str = "lease-under-test",
    org_id: str = ORG_ID,
    environment: str = ENVIRONMENT,
    jti: str | None = None,
) -> dict:
    start = issued_at or datetime.now(UTC)
    return {
        "artifactType": RUNTIME_CONTROL_LEASE_TYPE,
        "leaseVersion": 1,
        "issuer": ISSUER,
        "keyId": KEY_ID,
        "orgId": org_id,
        "targetEnvironment": environment,
        "leaseId": lease_id,
        "revision": revision,
        "issuedAt": canonical_instant(start),
        "notBefore": canonical_instant(start),
        "expiresAt": canonical_instant(start + timedelta(seconds=ttl_seconds)),
        "mode": mode,
        "staleAction": stale_action,
        "controls": controls if controls is not None else [],
        "jti": jti or f"{lease_id}-{revision}-{int(start.timestamp() * 1000)}",
    }


def control(
    subject_id: str = TENANT,
    *,
    scope: str = "tenant",
    state: str = "quarantined",
) -> dict:
    return {
        "scope": scope,
        "subjectId": subject_id,
        "state": state,
        "reasonCode": "operator_quarantine",
    }


_KEEP_DEFAULT_ARTIFACT_TYPES = object()


class _Signer:
    def __init__(self, allowed_artifact_types=_KEEP_DEFAULT_ARTIFACT_TYPES) -> None:
        self.private_key = Ed25519PrivateKey.generate()
        self.public_der = self.private_key.public_key().public_bytes(
            Encoding.DER,
            PublicFormat.SubjectPublicKeyInfo,
        )
        self._allowed_artifact_types = (
            [RUNTIME_CONTROL_LEASE_TYPE]
            if allowed_artifact_types is _KEEP_DEFAULT_ARTIFACT_TYPES
            else allowed_artifact_types
        )

    def envelope(self, claims: dict) -> dict:
        payload = canonical_json(claims)
        return {
            "artifactType": claims["artifactType"],
            "leaseVersion": claims["leaseVersion"],
            "algorithm": "ed25519",
            "canonicalization": "signed-payload-json-v1",
            "issuer": claims["issuer"],
            "keyId": claims["keyId"],
            "signedPayload": payload,
            "signature": base64.b64encode(
                self.private_key.sign(payload.encode("utf-8"))
            ).decode("ascii"),
            "signed": True,
        }

    def trust_document(self, *, revoked_jtis: list[str] | None = None) -> dict:
        entry = {
            "issuer": ISSUER,
            "keyId": KEY_ID,
            "publicKeySpkiDerBase64": base64.b64encode(self.public_der).decode("ascii"),
            "allowedOrgIds": [ORG_ID],
            "allowedEnvironments": [ENVIRONMENT],
        }
        if self._allowed_artifact_types is not None:
            entry["allowedArtifactTypes"] = self._allowed_artifact_types
        return {
            "trustVersion": 1,
            "entries": [entry],
            "revokedKeyIds": [],
            "revokedJtis": revoked_jtis or [],
        }


@pytest.fixture
def signer() -> _Signer:
    return _Signer()


@pytest.fixture
def trust_store_path(tmp_path: Path, signer: _Signer) -> Path:
    path = tmp_path / "runtime-control-trust.json"
    path.write_text(json.dumps(signer.trust_document()), encoding="utf-8")
    return path


def config_for(trust_store_path: Path, **overrides) -> RuntimeControlConfig:
    values = {
        "trust_store_path": trust_store_path,
        "expected_environment": ENVIRONMENT,
        "expected_org_id": ORG_ID,
        "expected_deployment_id": DEPLOYMENT_ID,
        "max_lease_seconds": 900,
        "stale_action_policy": "lease",
        "clock_skew_seconds": 30,
        "max_response_bytes": 65_536,
        "max_trust_store_bytes": 1_048_576,
    }
    values.update(overrides)
    return RuntimeControlConfig(**values)


def reply(
    envelope: dict | None,
    *,
    status: str = "issued",
    code: str | None = None,
    delivery: bool = True,
) -> bytes:
    """The exact reply shape the control plane's observation intake returns."""
    body: dict = {
        "observationId": "obs-1",
        "status": "recorded",
        "ordering": "in_order",
        "idempotent": False,
        "authority": "tenant_model_plane_assertion",
        "credentialBinding": "managed",
    }
    if delivery:
        block: dict = {"contractVersion": 1, "status": status}
        if status == "unavailable":
            block["code"] = code or "missing_keypair"
        elif status == "issued":
            block["lease"] = envelope
        body["runtimeControls"] = block
    return json.dumps(body).encode("utf-8")


def settings_ns(**overrides) -> SimpleNamespace:
    values = {
        "model_plane_runtime_control_enabled": True,
        "model_plane_observation_enabled": True,
        "model_plane_observation_target_environment": ENVIRONMENT,
        "model_plane_observation_deployment_id": DEPLOYMENT_ID,
        "model_plane_runtime_control_trust_store_file": Path(".model_routing_trust.json"),
        "model_plane_runtime_control_max_lease_seconds": 900,
        "model_plane_runtime_control_stale_action": "lease",
        "model_plane_runtime_control_max_response_bytes": 65_536,
        "model_routing_expected_org_id": ORG_ID,
        "model_routing_clock_skew_seconds": 30,
        "model_routing_max_file_bytes": 1_048_576,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


# --- off unless configured ------------------------------------------------


def test_default_settings_build_no_runtime_control_seam() -> None:
    loaded = Settings(_env_file=None)

    assert loaded.model_plane_runtime_control_enabled is False
    assert load_model_plane_runtime_control_config(loaded) is None


def test_disabled_seam_ignores_every_other_runtime_control_field(
    trust_store_path: Path,
) -> None:
    assert (
        load_model_plane_runtime_control_config(
            settings_ns(
                model_plane_runtime_control_enabled=False,
                model_plane_observation_enabled=False,
                model_plane_runtime_control_trust_store_file=Path("/nonexistent"),
                model_routing_expected_org_id="",
            )
        )
        is None
    )
    assert trust_store_path.exists()


@pytest.mark.parametrize(
    ("overrides", "expected_code"),
    [
        ({"model_plane_observation_enabled": False}, "observation_reporter_required"),
        ({"model_plane_observation_deployment_id": ""}, "observation_scope_required"),
        ({"model_routing_expected_org_id": "  "}, "expected_org_id_required"),
    ],
)
def test_an_incomplete_seam_fails_startup_rather_than_looking_controlled(
    trust_store_path: Path,
    overrides: dict,
    expected_code: str,
) -> None:
    with pytest.raises(ModelPlaneRuntimeControlConfigError) as excinfo:
        load_model_plane_runtime_control_config(
            settings_ns(
                model_plane_runtime_control_trust_store_file=trust_store_path,
                **overrides,
            )
        )

    assert excinfo.value.code == expected_code


def test_enabling_without_a_readable_trust_store_fails_startup(tmp_path: Path) -> None:
    with pytest.raises(ModelPlaneRuntimeControlConfigError) as excinfo:
        load_model_plane_runtime_control_config(
            settings_ns(
                model_plane_runtime_control_trust_store_file=tmp_path / "absent.json"
            )
        )

    assert excinfo.value.code == "trust_store_missing"


def test_the_seam_needs_no_particular_observation_version(
    trust_store_path: Path,
) -> None:
    """The acknowledgement is additive, so v1 and v2 both carry it."""
    for version in (1, 2):
        loaded = load_model_plane_runtime_control_config(
            settings_ns(
                model_plane_runtime_control_trust_store_file=trust_store_path,
                model_plane_observation_version=version,
            )
        )
        assert loaded is not None
        assert loaded.expected_deployment_id == DEPLOYMENT_ID
        assert loaded.expected_org_id == ORG_ID
        assert loaded.stale_action_policy == "lease"


# --- verification ---------------------------------------------------------


def _verify(signer: _Signer, trust_store_path: Path, claims: dict, **overrides):
    envelope = RuntimeControlLeaseEnvelope.model_validate(
        signer.envelope(claims),
        strict=True,
    )
    trust = load_model_routing_trust_store(trust_store_path)
    return verify_runtime_control_lease(
        envelope,
        trust,
        config_for(trust_store_path, **overrides),
    )


def test_lease_verifies_against_the_routing_trust_store(
    signer: _Signer,
    trust_store_path: Path,
) -> None:
    lease = _verify(
        signer,
        trust_store_path,
        _claims(revision=4, controls=[control()]),
    )

    assert lease.revision == 4
    assert lease.lease_id == "lease-under-test"
    assert lease.digest.startswith("sha256:")
    assert [entry.subject_id for entry in lease.claims.controls] == [TENANT]
    assert lease.claims.mode == "enforcing"


@pytest.mark.parametrize(
    ("claims_overrides", "expected_code"),
    [
        ({"org_id": "org-other"}, "org_not_allowed"),
        ({"ttl_seconds": 100_000}, "lease_ttl_too_long"),
        ({"revision": -1}, "invalid_revision"),
        ({"mode": "off"}, "malformed_claims"),
        ({"stale_action": "halt"}, "malformed_claims"),
        ({"environment": "prod"}, "environment_not_allowed"),
    ],
)
def test_lease_scope_and_lifetime_are_enforced(
    signer: _Signer,
    trust_store_path: Path,
    claims_overrides: dict,
    expected_code: str,
) -> None:
    with pytest.raises(ModelPlaneRuntimeControlError) as excinfo:
        _verify(signer, trust_store_path, _claims(**claims_overrides))

    assert excinfo.value.code == expected_code


def test_a_lease_for_another_org_is_refused_even_when_the_key_allows_it(
    signer: _Signer,
    tmp_path: Path,
) -> None:
    path = tmp_path / "two-org-trust.json"
    document = signer.trust_document()
    document["entries"][0]["allowedOrgIds"] = [ORG_ID, "org-other"]
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ModelPlaneRuntimeControlError) as excinfo:
        _verify(signer, path, _claims(org_id="org-other"))

    assert excinfo.value.code == "org_mismatch"


def test_an_envelope_that_disagrees_with_its_own_claims_is_refused(
    signer: _Signer,
    trust_store_path: Path,
) -> None:
    claims = _claims()
    envelope = signer.envelope(claims)
    envelope["artifactType"] = "orchestra.model-routing-policy"

    with pytest.raises(ValueError):
        RuntimeControlLeaseEnvelope.model_validate(envelope, strict=True)


@pytest.mark.parametrize(
    "allowed_artifact_types",
    [["orchestra.routing-policy"], ["orchestra.bundle", "orchestra.promotion"], None],
)
def test_a_key_that_is_not_a_runtime_control_key_cannot_sign_a_lease(
    tmp_path: Path,
    allowed_artifact_types: list[str] | None,
) -> None:
    """A check only the issuer performs is not a check (contract §3.4).

    The signature below is valid; the lease is refused on the key's declared
    purpose alone, and an entry that declares none at all fails closed.
    """
    wrong_purpose = _Signer(allowed_artifact_types=allowed_artifact_types)
    path = tmp_path / "wrong-purpose-trust.json"
    path.write_text(json.dumps(wrong_purpose.trust_document()), encoding="utf-8")

    with pytest.raises(ModelPlaneRuntimeControlError) as excinfo:
        _verify(wrong_purpose, path, _claims(controls=[control()]))

    assert excinfo.value.code == "signing_key_purpose_denied"


def test_tampered_payload_and_untrusted_signer_are_refused(
    signer: _Signer,
    trust_store_path: Path,
) -> None:
    claims = _claims(controls=[control()])
    tampered = signer.envelope(claims)
    tampered["signedPayload"] = canonical_json(_claims(controls=[]))
    trust = load_model_routing_trust_store(trust_store_path)

    with pytest.raises(ModelPlaneRuntimeControlError) as signature_failure:
        verify_runtime_control_lease(
            RuntimeControlLeaseEnvelope.model_validate(tampered, strict=True),
            trust,
            config_for(trust_store_path),
        )
    assert signature_failure.value.code == "invalid_signature"

    stranger = _Signer()
    with pytest.raises(ModelPlaneRuntimeControlError) as untrusted:
        verify_runtime_control_lease(
            RuntimeControlLeaseEnvelope.model_validate(
                stranger.envelope(claims),
                strict=True,
            ),
            trust,
            config_for(trust_store_path),
        )
    assert untrusted.value.code == "invalid_signature"


def test_revoked_lease_identifier_is_refused(signer: _Signer, tmp_path: Path) -> None:
    claims = _claims()
    path = tmp_path / "trust-with-revocation.json"
    path.write_text(
        json.dumps(signer.trust_document(revoked_jtis=[claims["jti"]])),
        encoding="utf-8",
    )

    with pytest.raises(ModelPlaneRuntimeControlError) as excinfo:
        _verify(signer, path, claims)

    assert excinfo.value.code == "revoked_lease"


# --- shared conformance vectors -------------------------------------------

VECTOR_DIR = Path(__file__).parent / "fixtures" / "lease-vectors"
VECTOR_MANIFEST = json.loads(
    (VECTOR_DIR / "manifest.json").read_text(encoding="utf-8")
)
VECTOR_IDS = [entry["id"] for entry in VECTOR_MANIFEST["vectors"]]
VECTOR_FILES = {entry["id"]: entry["file"] for entry in VECTOR_MANIFEST["vectors"]}
ACK_VECTOR_IDS = [
    entry["id"] for entry in VECTOR_MANIFEST["acknowledgementVectors"]
]
ACK_VECTOR_FILES = {
    entry["id"]: entry["file"] for entry in VECTOR_MANIFEST["acknowledgementVectors"]
}


def _vector_instant(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=UTC)


@pytest.mark.parametrize("vector_id", VECTOR_IDS)
def test_shared_conformance_vector(tmp_path: Path, vector_id: str) -> None:
    """Run the cross-implementation vector set (contract §8).

    The fixtures under ``fixtures/lease-vectors`` are the shared gate: the same
    trust document, the same envelopes and the same clock are replayed by the
    SDK host and the control plane. A divergence in claim names,
    canonicalization, key-purpose handling or scope typing fails here rather
    than in a deployment. ``expectByRuntime["engine"]`` is this side's row;
    the ``sdk-host`` row deliberately disagrees on ``lease-agent-scope``.

    ``evaluate`` runs before ``status`` on every evaluate step, as the shared
    runner does, because a tenant-scoped control is not matched here until a
    request from that tenant arrives.
    """
    vector = json.loads(
        (VECTOR_DIR / VECTOR_FILES[vector_id]).read_text(encoding="utf-8")
    )
    context = vector["context"]
    engine = vector["runtimes"]["engine"]
    trust_path = tmp_path / "vector-trust.json"
    trust_path.write_text(
        json.dumps(
            {
                "trustVersion": 1,
                "entries": vector["trust"],
                "revokedKeyIds": [],
                "revokedJtis": [],
            }
        ),
        encoding="utf-8",
    )
    seam = ModelPlaneRuntimeControl(
        config_for(
            trust_path,
            expected_org_id=context["expectedOrgId"],
            expected_environment=context["expectedEnvironment"],
            expected_deployment_id=engine["subjects"]["deployment"],
            max_lease_seconds=context["maxLeaseSeconds"],
            clock_skew_seconds=context["maxClockSkewSeconds"],
            stale_action_policy="lease",
        )
    )
    tenant = engine["subjects"]["tenant"]
    assert list(ENFORCEABLE_SCOPES) == engine["enforceableScopes"]

    accepted_total = 0
    rejected_total = 0
    for step in vector["steps"]:
        at = _vector_instant(step["at"])
        if step["action"] == "apply":
            expect = step["expect"]
            seam.apply_observation_response(
                json.dumps(
                    {"runtimeControls": {"status": "issued", "lease": step["lease"]}}
                ).encode("utf-8"),
                now=at,
            )
            reported = seam.status(now=at.timestamp())
            if expect["outcome"] == "verified":
                accepted_total += 1
                assert reported.leases_accepted_total == accepted_total
                assert reported.lease_digest == expect["digest"]
                assert reported.acknowledgement.lease_id == expect["leaseId"]
                assert reported.acknowledgement.revision == expect["revision"]
                assert reported.lease_mode == expect["mode"]
                assert reported.lease_stale_action == expect["staleAction"]
                assert reported.control_count == expect["controlCount"]
            else:
                rejected_total += 1
                assert reported.leases_rejected_total == rejected_total
                assert reported.last_refresh_error_code == expect["errorCode"]
            continue

        expected = step["expectByRuntime"]["engine"]
        admitting = seam.evaluate(tenant=tenant, now=at.timestamp()) is None
        reported = seam.status(now=at.timestamp())
        assert admitting is expected["admitting"]
        assert reported.matched_state == expected["state"]
        assert reported.acknowledgement.stale == expected["stale"]
        assert reported.acknowledgement.enforcement == expected["enforcement"]
        assert (
            reported.acknowledgement.enforced_control_count
            == expected["enforcedControlCount"]
        )


@pytest.mark.parametrize("vector_id", ACK_VECTOR_IDS)
def test_acknowledgement_vector_shape(vector_id: str) -> None:
    """Contract §7b.19, from the producing side.

    The lease direction had vectors and this one did not, which is how the
    engine came to emit a shape the control plane does not read. The intake
    runs these forwards; here every ``accepted`` case is a shape this engine
    must be able to emit — the nine pinned camelCase keys, no more and no
    fewer — and every ``rejected`` case is one its model cannot express.
    """
    vector = json.loads(
        (VECTOR_DIR / ACK_VECTOR_FILES[vector_id]).read_text(encoding="utf-8")
    )
    expected = vector["expect"]["outcome"]

    if expected == "absent":
        # An observation carrying no acknowledgement at all. The engine emits
        # the key only when a seam is configured, which is the same statement.
        assert vector["acknowledgement"] is None
        assert RUNTIME_CONTROL_ACK_KEY not in observation(1, seam=None)
        return

    if expected == "rejected":
        with pytest.raises(ValidationError):
            RuntimeControlAcknowledgement.model_validate(
                vector["acknowledgement"], strict=True
            )
        return

    emitted = RuntimeControlAcknowledgement.model_validate(
        vector["acknowledgement"]
    ).model_dump(mode="json", by_alias=True)
    assert emitted == vector["expect"]["normalized"]


def test_the_engine_never_emits_a_rejected_acknowledgement_shape(
    signer: _Signer,
    trust_store_path: Path,
) -> None:
    """No ``rejected`` case may leave this runtime, on any lease it holds.

    The two the engine could plausibly produce are an ``enforcing`` claim with
    no lease to name (§1: enforcement that cannot be proved) and a member left
    out. Both are checked against the object it actually builds, live and
    stale, held and unheld.
    """
    pinned = set(
        json.loads(
            (VECTOR_DIR / ACK_VECTOR_FILES["ack-enforcing-quarantine"]).read_text(
                encoding="utf-8"
            )
        )["expect"]["normalized"]
    )
    unheld = ModelPlaneRuntimeControl(config_for(trust_store_path))
    held = control_holding(
        signer,
        trust_store_path,
        _claims(
            ttl_seconds=300,
            controls=[control(), control("audit-agent", scope="agent")],
        ),
    )
    assert held.evaluate(tenant=TENANT) is not None

    for seam, at in (
        (unheld, datetime.now(UTC)),
        (held, datetime.now(UTC)),
        (held, datetime.now(UTC) + timedelta(seconds=400)),
    ):
        wire = seam.acknowledgement(observed_at=at).model_dump(mode="json", by_alias=True)
        assert set(wire) == pinned
        assert wire["enforcement"] in {"advisory", "enforcing"}
        if wire["enforcement"] == "enforcing":
            assert wire["leaseId"] is not None
            assert wire["revision"] is not None


def test_the_vector_set_pins_the_key_purpose_field() -> None:
    """The shared trust document is what stops the three sides diverging again."""
    vector = json.loads(
        (VECTOR_DIR / VECTOR_FILES["lease-wrong-key-purpose"]).read_text(encoding="utf-8")
    )
    declared = {
        entry["keyId"]: entry["allowedArtifactTypes"] for entry in vector["trust"]
    }
    signing_key = vector["steps"][0]["lease"]["keyId"]

    assert RUNTIME_CONTROL_LEASE_TYPE not in declared[signing_key]
    assert any(RUNTIME_CONTROL_LEASE_TYPE in value for value in declared.values())


# --- scope typing ---------------------------------------------------------


def test_the_engine_declares_the_scopes_it_can_resolve() -> None:
    assert ENFORCEABLE_SCOPES == ("deployment", "org", "tenant")
    assert IGNORED_SCOPES == ("agent", "solution")


def control_holding(
    signer: _Signer,
    trust_store_path: Path,
    claims: dict,
    **overrides,
) -> ModelPlaneRuntimeControl:
    seam = ModelPlaneRuntimeControl(config_for(trust_store_path, **overrides))
    seam.apply_observation_response(reply(signer.envelope(claims)))
    assert seam.status().leases_accepted_total == 1
    return seam


def test_agent_and_solution_controls_are_ignored_and_reported_as_ignored(
    signer: _Signer,
    trust_store_path: Path,
) -> None:
    """The control plane's own vocabulary, arriving at a runtime without it."""
    seam = control_holding(
        signer,
        trust_store_path,
        _claims(
            controls=[
                control("audit-agent", scope="agent"),
                control("orders-solution", scope="solution"),
            ]
        ),
    )

    assert seam.evaluate(tenant=TENANT) is None
    assert seam.evaluate(tenant="audit-agent") is None
    assert seam.evaluate(tenant="orders-solution") is None
    ack = seam.acknowledgement(observed_at=datetime.now(UTC))
    # A lease this replica enforces nothing in reports 0 and advisory, so the
    # control plane cannot count it towards a quarantine it does not apply.
    assert ack.enforcement == "advisory"
    assert ack.enforced_control_count == 0
    assert ack.ignored_control_count == 2
    assert ack.enforceable_scopes == ["deployment", "org", "tenant"]


def test_tenant_scope_refuses_only_the_named_tenant(
    signer: _Signer,
    trust_store_path: Path,
) -> None:
    seam = control_holding(
        signer,
        trust_store_path,
        _claims(controls=[control(TENANT), control("audit-agent", scope="agent")]),
    )

    refusal = seam.evaluate(tenant=TENANT)

    assert refusal is not None
    assert refusal.code == "model_plane_quarantined"
    assert refusal.scope == "tenant"
    assert seam.evaluate(tenant="unrelated-runtime") is None
    ack = seam.acknowledgement(observed_at=datetime.now(UTC))
    assert ack.enforced_control_count == 1
    assert ack.ignored_control_count == 1


@pytest.mark.parametrize(
    ("scope", "subject_id"),
    [("org", ORG_ID), ("deployment", DEPLOYMENT_ID)],
)
def test_org_and_deployment_scope_quarantine_this_whole_replica(
    signer: _Signer,
    trust_store_path: Path,
    scope: str,
    subject_id: str,
) -> None:
    seam = control_holding(
        signer,
        trust_store_path,
        _claims(controls=[control(subject_id, scope=scope)]),
    )

    for tenant in (TENANT, "unrelated-runtime"):
        refusal = seam.evaluate(tenant=tenant)
        assert refusal is not None
        assert refusal.scope == scope
    assert seam.acknowledgement(observed_at=datetime.now(UTC)).enforced_control_count == 1


@pytest.mark.parametrize(
    ("scope", "subject_id"),
    [("org", "org-someone-else"), ("deployment", "someone-elses-engine")],
)
def test_a_replica_scoped_control_aimed_elsewhere_counts_for_nothing(
    signer: _Signer,
    trust_store_path: Path,
    scope: str,
    subject_id: str,
) -> None:
    """§2.2: the count is what this replica matched, not what the lease names."""
    seam = control_holding(
        signer,
        trust_store_path,
        _claims(controls=[control(subject_id, scope=scope)]),
    )

    assert seam.evaluate(tenant=TENANT) is None
    ack = seam.acknowledgement(observed_at=datetime.now(UTC))
    assert ack.enforced_control_count == 0
    assert ack.ignored_control_count == 0
    assert seam.status().control_count == 1


def test_a_serving_control_is_not_a_quarantine(
    signer: _Signer,
    trust_store_path: Path,
) -> None:
    seam = control_holding(
        signer,
        trust_store_path,
        _claims(controls=[control(TENANT, state="serving")]),
    )

    assert seam.evaluate(tenant=TENANT) is None
    # §8b: a matched control in ``serving`` state is governed, not enforced, so
    # the count is 0 and §8c makes an enforcing lease report ``advisory`` here.
    ack = seam.acknowledgement(observed_at=datetime.now(UTC))
    assert ack.enforcement == "advisory"
    assert ack.enforced_control_count == 0
    assert seam.status().matched_state == "serving"


# --- enforcement precedence ------------------------------------------------


def test_an_advisory_lease_reports_without_refusing(
    signer: _Signer,
    trust_store_path: Path,
) -> None:
    seam = control_holding(
        signer,
        trust_store_path,
        _claims(mode="advisory", controls=[control()]),
    )

    assert seam.evaluate(tenant=TENANT) is None
    ack = seam.acknowledgement(observed_at=datetime.now(UTC))
    assert ack.enforcement == "advisory"
    assert ack.enforced_control_count == 0
    assert seam.status().control_count == 1


@pytest.mark.parametrize("policy", ["lease", "continue", "stop"])
def test_an_advisory_lease_can_never_refuse_however_stale(
    signer: _Signer,
    trust_store_path: Path,
    policy: str,
) -> None:
    """Contract §4.7: mode is decided first, before staleness is considered."""
    seam = control_holding(
        signer,
        trust_store_path,
        _claims(mode="advisory", stale_action="stop", ttl_seconds=300,
                controls=[control(), control(ORG_ID, scope="org")]),
        stale_action_policy=policy,
    )

    assert seam.evaluate(tenant=TENANT, now=time.time() + 400) is None
    assert seam.evaluate(tenant="anyone", now=time.time() + 400) is None


def test_a_lease_with_no_controls_lifts_a_previous_quarantine(
    signer: _Signer,
    trust_store_path: Path,
) -> None:
    seam = control_holding(
        signer,
        trust_store_path,
        _claims(revision=1, controls=[control()]),
    )
    assert seam.evaluate(tenant=TENANT) is not None

    seam.apply_observation_response(reply(signer.envelope(_claims(revision=2))))

    assert seam.evaluate(tenant=TENANT) is None
    assert seam.status().control_count == 0


@pytest.mark.parametrize(
    ("policy", "lease_stale_action", "effective", "expected_code"),
    [
        ("lease", "continue", "continue", "model_plane_quarantined"),
        ("lease", "stop", "stop", "model_plane_control_lease_stale"),
        ("continue", "stop", "continue", "model_plane_quarantined"),
        ("stop", "continue", "stop", "model_plane_control_lease_stale"),
    ],
)
def test_a_matched_quarantine_outlives_its_lease_whichever_stale_action_applies(
    signer: _Signer,
    trust_store_path: Path,
    policy: str,
    lease_stale_action: str,
    effective: str,
    expected_code: str,
) -> None:
    """Contract §7a: the stale action decides enforcement, never traffic.

    ``continue`` continues *enforcing* the last verified lease, so the matched
    quarantine survives expiry and still refuses with the quarantine code.
    ``stop`` is the wider branch and refuses under the stale code. Neither
    resumes serving a quarantined subject, because a quarantine that lapsed on
    expiry could be lifted by cutting this replica off from the issuer.
    """
    seam = control_holding(
        signer,
        trust_store_path,
        _claims(ttl_seconds=300, stale_action=lease_stale_action, controls=[control()]),
        stale_action_policy=policy,
    )
    after_expiry = time.time() + 400

    assert seam.evaluate(tenant=TENANT) is not None
    refusal = seam.evaluate(tenant=TENANT, now=after_expiry)

    assert refusal is not None
    assert refusal.code == expected_code
    assert refusal.scope == "tenant"
    assert seam.status(now=after_expiry).effective_stale_action == effective
    # A tenant no control named is untouched by staleness, whatever the policy.
    assert seam.evaluate(tenant="unrelated-runtime", now=after_expiry) is None


@pytest.mark.parametrize("policy", ["lease", "continue", "stop"])
def test_no_lease_stops_nothing(trust_store_path: Path, policy: str) -> None:
    """Nothing is matched before a lease arrives, so nothing can be refused."""
    seam = ModelPlaneRuntimeControl(
        config_for(trust_store_path, stale_action_policy=policy)
    )

    assert seam.evaluate(tenant=TENANT) is None
    assert seam.acknowledgement(observed_at=datetime.now(UTC)).lease_id is None


def test_an_expired_lease_reports_itself_stale_and_still_enforcing(
    signer: _Signer,
    trust_store_path: Path,
) -> None:
    """Contract §8b: report what is actually being enforced, stale or not.

    Zeroing the count while still refusing would show the control plane a
    quarantine this replica had released, which is the failure §1 exists to
    prevent wearing the acknowledgement's clothes.
    """
    seam = control_holding(
        signer,
        trust_store_path,
        _claims(revision=4, ttl_seconds=300, controls=[control()]),
    )
    after_expiry = time.time() + 400

    assert seam.evaluate(tenant=TENANT, now=after_expiry) is not None
    ack = seam.acknowledgement(observed_at=datetime.fromtimestamp(after_expiry, tz=UTC))
    assert ack.stale is True
    assert ack.enforcement == "enforcing"
    assert ack.enforced_control_count == 1
    assert ack.revision == 4


def test_request_path_reads_a_cached_decision_rather_than_verifying(
    signer: _Signer,
    trust_store_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import inference_engine.model_plane_control as control_module

    verifications = 0
    real_verify = control_module.verify_runtime_control_lease

    def _counting_verify(*args, **kwargs):
        nonlocal verifications
        verifications += 1
        return real_verify(*args, **kwargs)

    monkeypatch.setattr(control_module, "verify_runtime_control_lease", _counting_verify)
    seam = control_holding(signer, trust_store_path, _claims(controls=[control()]))

    for _ in range(500):
        seam.evaluate(tenant=TENANT)

    assert verifications == 1


# --- refresh behaviour ----------------------------------------------------


def test_a_reply_without_a_delivery_block_leaves_the_held_lease_alone(
    signer: _Signer,
    trust_store_path: Path,
) -> None:
    seam = control_holding(signer, trust_store_path, _claims(controls=[control()]))

    seam.apply_observation_response(reply(None, delivery=False))

    assert seam.evaluate(tenant=TENANT) is not None
    assert seam.status().leases_rejected_total == 0


@pytest.mark.parametrize(
    ("status", "code"),
    [("disabled", None), ("unavailable", "missing_keypair")],
)
def test_a_control_plane_declining_to_deliver_does_not_release_anyone(
    signer: _Signer,
    trust_store_path: Path,
    status: str,
    code: str | None,
) -> None:
    seam = control_holding(signer, trust_store_path, _claims(controls=[control()]))

    seam.apply_observation_response(reply(None, status=status, code=code))

    assert seam.evaluate(tenant=TENANT) is not None
    reported = seam.status()
    assert reported.last_delivery_status == status
    assert reported.last_delivery_code == code
    assert reported.leases_rejected_total == 0


@pytest.mark.parametrize(
    "body",
    [
        b"not json",
        b"[]",
        b'{"runtimeControls": 7}',
        b'{"runtimeControls": {"status": "issued", "lease": {"leaseVersion": 1}}}',
    ],
)
def test_an_unusable_refresh_keeps_the_lease_it_could_not_replace(
    signer: _Signer,
    trust_store_path: Path,
    body: bytes,
) -> None:
    seam = control_holding(signer, trust_store_path, _claims(controls=[control()]))

    seam.apply_observation_response(body)

    assert seam.evaluate(tenant=TENANT) is not None
    reported = seam.status()
    assert reported.leases_rejected_total == 1
    assert reported.last_refresh_error_code is not None


def test_an_unparseable_lease_is_never_read_as_an_absent_quarantine(
    signer: _Signer,
    trust_store_path: Path,
) -> None:
    seam = control_holding(signer, trust_store_path, _claims(controls=[control()]))

    seam.apply_observation_response(
        b'{"runtimeControls": {"status": "issued", "lease": {"leaseVersion": 1}}}'
    )

    assert seam.evaluate(tenant=TENANT) is not None
    assert seam.status().last_refresh_error_code == "lease_parse_failed"


def test_oversized_reply_is_refused_without_parsing(
    signer: _Signer,
    trust_store_path: Path,
) -> None:
    seam = ModelPlaneRuntimeControl(config_for(trust_store_path, max_response_bytes=32))

    seam.apply_observation_response(reply(signer.envelope(_claims())))

    reported = seam.status()
    assert reported.last_refresh_error_code == "control_response_too_large"
    assert reported.acknowledgement.lease_id is None


def test_older_revision_and_replayed_lease_cannot_lift_a_live_quarantine(
    signer: _Signer,
    trust_store_path: Path,
) -> None:
    seam = control_holding(
        signer,
        trust_store_path,
        _claims(revision=5, controls=[control()]),
    )

    seam.apply_observation_response(reply(signer.envelope(_claims(revision=4))))
    assert seam.status().last_refresh_error_code == "control_lease_out_of_order"

    replayed = _claims(
        revision=5,
        issued_at=datetime.now(UTC) - timedelta(seconds=120),
        controls=[],
    )
    seam.apply_observation_response(reply(signer.envelope(replayed)))
    assert seam.status().last_refresh_error_code == "lease_replayed"

    reused = _claims(revision=5, lease_id="a-different-control-set", controls=[])
    seam.apply_observation_response(reply(signer.envelope(reused)))
    assert seam.status().last_refresh_error_code == "control_lease_out_of_order"

    assert seam.evaluate(tenant=TENANT) is not None
    assert seam.status().acknowledgement.revision == 5


def test_same_revision_refresh_extends_the_lease(
    signer: _Signer,
    trust_store_path: Path,
) -> None:
    seam = control_holding(
        signer,
        trust_store_path,
        _claims(revision=3, controls=[control()], ttl_seconds=120),
    )
    first_expiry = seam.status().acknowledgement.lease_expires_at

    seam.apply_observation_response(
        reply(
            signer.envelope(
                _claims(
                    revision=3,
                    controls=[control()],
                    ttl_seconds=120,
                    issued_at=datetime.now(UTC) + timedelta(seconds=30),
                )
            )
        )
    )

    reported = seam.status()
    assert reported.leases_accepted_total == 2
    assert reported.acknowledgement.lease_expires_at > first_expiry


# --- observation payload --------------------------------------------------


class FakeState:
    def __init__(self) -> None:
        self.model_routing_runtime = ModelRoutingRuntimeState()
        self.model_routing_rate_limiter = ModelRoutingRateLimiter()

    def readiness(self) -> dict:
        return {"ready": True, "status": "ready"}


def inventory() -> ModelList:
    return ModelList(
        data=[ModelInfo(id="zeta:model")],
        unavailable=[UnavailableModel(id="offline:model", reason="upstream_timeout")],
    )


def available(_model_id: str) -> bool:
    return True


def observation_config(version: int = 1):
    loaded = load_model_plane_observation_config(
        SimpleNamespace(
            model_plane_observation_enabled=True,
            model_plane_observation_endpoint=(
                "https://orchestra.example/api/model-routing-observations"
            ),
            model_plane_observation_api_key="pk_model_plane_test",
            model_plane_observation_api_key_file="",
            model_plane_observation_deployment_id=DEPLOYMENT_ID,
            model_plane_observation_target_environment=ENVIRONMENT,
            model_plane_observation_engine_instance_id="engine-pod-0",
            model_plane_observation_version=version,
            model_plane_observation_interval_seconds=60.0,
            model_plane_observation_timeout_seconds=5.0,
            model_plane_observation_jitter_ratio=0.1,
            auth_enabled=True,
        )
    )
    assert loaded is not None
    return loaded


OBSERVATION_V1_KEYS = {
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


def observation(
    version: int = 1,
    seam: ModelPlaneRuntimeControl | None = None,
) -> dict:
    return build_model_plane_observation(
        observation_config(version),
        FakeState(),
        inventory,
        candidate_availability=available,
        runtime_control=seam,
        observation_id="observation-fixed-1",
        now=datetime(2026, 7, 13, 12, 30, tzinfo=UTC),
    )


def test_versions_1_and_2_are_byte_identical_with_the_seam_off() -> None:
    for version, extra in ((1, set()), (2, {"routingInventory"})):
        payload = observation(version)
        assert set(payload) == OBSERVATION_V1_KEYS | extra
        assert payload == observation(version, seam=None)


@pytest.mark.parametrize("version", [1, 2])
def test_the_acknowledgement_is_additive_and_bumps_no_version(
    signer: _Signer,
    trust_store_path: Path,
    version: int,
) -> None:
    """§5: bumping observationVersion would make a pre-v3 intake reject it all."""
    seam = control_holding(
        signer,
        trust_store_path,
        _claims(
            revision=6,
            ttl_seconds=300,
            controls=[
                control(DEPLOYMENT_ID, scope="deployment"),
                control("audit-agent", scope="agent"),
            ],
        ),
    )

    payload = observation(version, seam=seam)

    assert payload["observationVersion"] == version
    assert set(payload) - {RUNTIME_CONTROL_ACK_KEY} == set(observation(version))
    ack = payload[RUNTIME_CONTROL_ACK_KEY]
    # §7b.17: exactly these nine camelCase keys, and no ``mode`` — that is the
    # issuer's instruction, while this object says what the replica did.
    assert set(ack) == {
        "leaseId",
        "revision",
        "enforcement",
        "enforceableScopes",
        "enforcedControlCount",
        "ignoredControlCount",
        "stale",
        "leaseExpiresAt",
        "leaseParseFailed",
    }
    assert ack["revision"] == 6
    assert ack["enforcement"] == "enforcing"
    assert ack["enforcedControlCount"] == 1
    assert ack["ignoredControlCount"] == 1
    assert ack["enforceableScopes"] == ["deployment", "org", "tenant"]
    # Payload-free: counts and the lease identity travel; the subject does not.
    assert "audit-agent" not in json.dumps(payload)
    assert "operator_quarantine" not in json.dumps(payload)


def test_staleness_is_reported_against_the_observation_instant(
    signer: _Signer,
    trust_store_path: Path,
) -> None:
    """The intake cross-checks ``stale`` against the expiry in the same payload."""
    seam = control_holding(
        signer,
        trust_store_path,
        _claims(ttl_seconds=300, controls=[control()]),
    )

    ack = seam.acknowledgement(observed_at=datetime.now(UTC) + timedelta(seconds=400))

    assert ack.stale is True
    assert ack.lease_expires_at is not None


async def test_reporter_reads_the_lease_off_its_own_accepted_reply(
    signer: _Signer,
    trust_store_path: Path,
) -> None:
    seam = ModelPlaneRuntimeControl(config_for(trust_store_path))
    envelope = signer.envelope(_claims(controls=[control()]))
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        return httpx.Response(201, content=reply(envelope))

    reporter = ModelPlaneObservationReporter(
        observation_config(1),
        FakeState(),
        inventory,
        available,
        seam,
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        assert await reporter.report_once(client) is True

    assert seen == ["/api/model-routing-observations"]
    assert seam.evaluate(tenant=TENANT) is not None
    assert seam.status().leases_accepted_total == 1


async def test_a_rejected_observation_delivers_no_lease(
    signer: _Signer,
    trust_store_path: Path,
) -> None:
    seam = ModelPlaneRuntimeControl(config_for(trust_store_path))

    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(400, content=reply(signer.envelope(_claims())))

    reporter = ModelPlaneObservationReporter(
        observation_config(1),
        FakeState(),
        inventory,
        available,
        seam,
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        assert await reporter.report_once(client) is False

    assert seam.status().leases_accepted_total == 0


# --- request path and operator surfaces -----------------------------------


@pytest.fixture
def _installed_control(monkeypatch: pytest.MonkeyPatch):
    previous = app_state.model_plane_runtime_control
    monkeypatch.setattr(settings, "auth_enabled", False)
    auth_mod._reset_for_tests()

    def install(seam: ModelPlaneRuntimeControl | None) -> None:
        app_state.model_plane_runtime_control = seam

    yield install
    auth_mod._reset_for_tests()
    app_state.model_plane_runtime_control = previous


@pytest.mark.parametrize(
    ("path", "payload"),
    [
        (
            "/v1/chat/completions",
            {
                "model": "reasoning",
                "messages": [{"role": "user", "content": "confidential-prompt-body"}],
            },
        ),
        ("/v1/completions", {"model": "reasoning", "prompt": "confidential-prompt-body"}),
        ("/v1/embeddings", {"model": "reasoning", "input": "confidential-prompt-body"}),
        (
            "/v1/rerank",
            {
                "model": "reasoning",
                "query": "confidential-prompt-body",
                "documents": ["a", "b"],
            },
        ),
    ],
)
def test_every_governed_route_is_inside_the_quarantine(
    signer: _Signer,
    trust_store_path: Path,
    _installed_control,
    path: str,
    payload: dict,
) -> None:
    _installed_control(
        control_holding(
            signer,
            trust_store_path,
            _claims(revision=7, controls=[control("anonymous")]),
        )
    )

    response = TestClient(app).post(path, json=payload)

    assert response.status_code == 503
    error = response.json()["error"]
    assert error["type"] == "model_plane_quarantined"
    assert error["code"] == "model_plane_quarantined"
    assert error["scope"] == "tenant"
    assert "confidential-prompt-body" not in json.dumps(response.json())


def test_the_refusal_names_no_control_plane_state(
    signer: _Signer,
    trust_store_path: Path,
    _installed_control,
) -> None:
    claims = _claims(revision=7, lease_id="lease-secret", controls=[control("anonymous")])
    _installed_control(control_holding(signer, trust_store_path, claims))

    body = TestClient(app).post(
        "/v1/chat/completions",
        json={"model": "reasoning", "messages": [{"role": "user", "content": "hi"}]},
    ).text

    assert "lease-secret" not in body
    assert claims["jti"] not in body
    assert "operator_quarantine" not in body


def test_an_unquarantined_tenant_is_untouched_by_a_live_lease(
    signer: _Signer,
    trust_store_path: Path,
    _installed_control,
) -> None:
    _installed_control(
        control_holding(
            signer,
            trust_store_path,
            _claims(controls=[control("some-other-tenant")]),
        )
    )

    response = TestClient(app).post(
        "/v1/chat/completions",
        json={
            "model": "definitely-not-a-model",
            "messages": [{"role": "user", "content": "hello"}],
        },
    )

    assert response.json()["error"]["type"] != "model_plane_quarantined"


def test_unconfigured_seam_leaves_the_request_path_untouched(_installed_control) -> None:
    from inference_engine.api import _model_routing

    _installed_control(None)

    assert (
        _model_routing.enforce_runtime_control(identity=Identity(tenant="t", key_id="k"))
        is None
    )


def test_metrics_report_enforcement_state_without_naming_subjects(
    signer: _Signer,
    trust_store_path: Path,
    _installed_control,
) -> None:
    claims = _claims(
        revision=9,
        lease_id="lease-secret",
        controls=[
            control(TENANT),
            control(DEPLOYMENT_ID, scope="deployment"),
            control("audit-agent", scope="agent"),
        ],
    )
    _installed_control(control_holding(signer, trust_store_path, claims))

    body = TestClient(app).get("/v1/metrics").text

    assert "inference_engine_model_plane_runtime_control_enabled 1" in body
    assert "inference_engine_model_plane_runtime_control_lease_held 1" in body
    assert "inference_engine_model_plane_runtime_control_enforcing 1" in body
    assert "inference_engine_model_plane_runtime_control_stale 0" in body
    assert "inference_engine_model_plane_runtime_control_controls 3" in body
    assert "inference_engine_model_plane_runtime_control_matched_controls 2" in body
    assert "inference_engine_model_plane_runtime_control_ignored_controls 1" in body
    # §8b: the tenant-scoped quarantine is matched by the lease but not yet by
    # a request, so only the deployment-scoped one is being enforced.
    assert "inference_engine_model_plane_runtime_control_enforced_controls 1" in body
    assert "inference_engine_model_plane_runtime_control_revision 9" in body
    for secret in (TENANT, "audit-agent", "lease-secret", claims["jti"], "operator_quarantine"):
        assert secret not in body


def test_metrics_report_the_off_seam_as_off(_installed_control) -> None:
    _installed_control(None)

    body = TestClient(app).get("/v1/metrics").text

    assert "inference_engine_model_plane_runtime_control_enabled 0" in body
    assert "inference_engine_model_plane_runtime_control_enforcing 0" in body


def test_admin_status_separates_acknowledged_state_from_enforcement(
    signer: _Signer,
    trust_store_path: Path,
    _installed_control,
) -> None:
    _installed_control(None)
    client = TestClient(app)

    off = client.get("/v1/admin/model-plane-runtime-control").json()
    assert off["enabled"] is False
    # §8c: nothing enforced reads as advisory, however it got there.
    assert off["acknowledgement"]["enforcement"] == "advisory"
    assert off["acknowledgement"]["leaseId"] is None

    _installed_control(
        control_holding(
            signer,
            trust_store_path,
            _claims(
                revision=11,
                controls=[
                    control(DEPLOYMENT_ID, scope="deployment"),
                    control("audit-agent", scope="agent"),
                ],
            ),
        )
    )
    body = client.get("/v1/admin/model-plane-runtime-control").json()

    assert body["enabled"] is True
    # The acknowledgement renders in its own pinned camelCase spelling here,
    # because it is the same object the observation carries (§7b).
    assert body["acknowledgement"]["revision"] == 11
    assert body["acknowledgement"]["enforcement"] == "enforcing"
    assert body["acknowledgement"]["enforcedControlCount"] == 1
    assert body["acknowledgement"]["ignoredControlCount"] == 1
    assert body["acknowledgement"]["stale"] is False
    assert body["enforceable_scopes"] == ["deployment", "org", "tenant"]
    assert body["ignored_scopes"] == ["agent", "solution"]
    assert body["control_count"] == 2
    assert body["matched_control_count"] == 1
    assert body["stale_action_policy"] == "lease"
    assert body["lease_stale_action"] == "continue"
    assert body["effective_stale_action"] == "continue"
    assert body["lease_mode"] == "enforcing"
