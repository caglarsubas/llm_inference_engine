from __future__ import annotations

import base64
import copy
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from inference_engine.model_routing import (
    ModelRoutingPolicyActivationError,
    ModelRoutingPolicyEnvelope,
    ModelRoutingPolicyError,
    ModelRoutingPolicyStore,
    ModelRoutingTrustStore,
    canonical_json,
    load_model_routing_envelope,
    model_routing_policy_digest,
    verify_model_routing_policy,
)


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "model-routing-policy-v1.json"


def _fixture() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def _now(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


def _trust_document(fixture: dict | None = None) -> dict:
    source = fixture or _fixture()
    return {
        "trustVersion": 1,
        "entries": [source["trust"]],
        "revokedKeyIds": [],
        "revokedJtis": [],
    }


def _models(fixture: dict | None = None):
    source = fixture or _fixture()
    envelope = ModelRoutingPolicyEnvelope.model_validate(source["policy"], strict=True)
    trust = ModelRoutingTrustStore.model_validate(_trust_document(source), strict=True)
    return envelope, trust


def _signed_envelope(
    claims: dict,
    *,
    canonical: bool = True,
) -> tuple[dict, dict]:
    private_key = Ed25519PrivateKey.generate()
    public_der = private_key.public_key().public_bytes(
        Encoding.DER,
        PublicFormat.SubjectPublicKeyInfo,
    )
    payload = canonical_json(claims) if canonical else json.dumps(claims, indent=2)
    signature = private_key.sign(payload.encode("utf-8"))
    envelope = {
        "policyId": claims["policyId"],
        "policyVersion": 1,
        "algorithm": "ed25519",
        "canonicalization": "signed-payload-json-v1",
        "issuer": claims["issuer"],
        "keyId": claims["keyId"],
        "signedPayload": payload,
        "signature": base64.b64encode(signature).decode("ascii"),
        "signed": True,
    }
    trust = {
        "trustVersion": 1,
        "entries": [
            {
                "issuer": claims["issuer"],
                "keyId": claims["keyId"],
                "publicKeySpkiDerBase64": base64.b64encode(public_der).decode("ascii"),
                "allowedOrgIds": [claims["orgId"]],
                "allowedEnvironments": [claims["targetEnvironment"]],
            }
        ],
        "revokedKeyIds": [],
        "revokedJtis": [],
    }
    return envelope, trust


def _verify_fixture(*, now: str = "2026-07-13T00:10:00.000Z"):
    fixture = _fixture()
    envelope, trust = _models(fixture)
    return verify_model_routing_policy(
        envelope,
        trust,
        now=_now(now),
        expected_audience=fixture["verification"]["expectedAudience"],
        expected_environment=fixture["verification"]["expectedEnvironment"],
        expected_org_id=fixture["verification"]["expectedOrgId"],
    )


def test_cross_language_golden_vector_verifies_exact_bytes() -> None:
    fixture = _fixture()
    verified = _verify_fixture()

    assert verified.digest == fixture["verification"]["expectedDigest"]
    assert verified.digest == model_routing_policy_digest(fixture["policy"]["signedPayload"])
    assert (
        canonical_json(json.loads(fixture["policy"]["signedPayload"]))
        == (fixture["policy"]["signedPayload"])
    )
    assert verified.claims.policy_id == "routing-golden-v1"
    assert verified.claims.revision == 1
    assert [route.route_id for route in verified.claims.routes] == [
        "reasoning",
        "default",
    ]


def test_tampered_payload_fails_signature() -> None:
    fixture = _fixture()
    fixture["policy"]["signedPayload"] = fixture["policy"]["signedPayload"].replace(
        "qwen3:32b", "qwen3:72b"
    )
    envelope, trust = _models(fixture)
    with pytest.raises(ModelRoutingPolicyError, match="invalid_signature"):
        verify_model_routing_policy(envelope, trust, now=_now("2026-07-13T00:10:00.000Z"))


def test_malformed_signature_is_a_stable_verification_error() -> None:
    fixture = _fixture()
    fixture["policy"]["signature"] = "not-base64!"
    envelope, trust = _models(fixture)
    with pytest.raises(ModelRoutingPolicyError, match="invalid_signature"):
        verify_model_routing_policy(envelope, trust, now=_now("2026-07-13T00:10:00.000Z"))


@pytest.mark.parametrize(
    ("mutation", "error_code"),
    [
        (lambda trust, claims: trust["revokedKeyIds"].append(claims["keyId"]), "revoked_key"),
        (lambda trust, claims: trust["revokedJtis"].append(claims["jti"]), "revoked_policy"),
        (
            lambda trust, claims: trust["entries"][0].update({"allowedOrgIds": ["other-org"]}),
            "org_not_allowed",
        ),
        (
            lambda trust, claims: trust["entries"][0].update({"allowedEnvironments": ["prod"]}),
            "environment_not_allowed",
        ),
    ],
)
def test_trust_constraints_and_revocation_fail_closed(mutation, error_code) -> None:
    fixture = _fixture()
    claims = json.loads(fixture["policy"]["signedPayload"])
    trust_raw = _trust_document(fixture)
    mutation(trust_raw, claims)
    envelope = ModelRoutingPolicyEnvelope.model_validate(fixture["policy"], strict=True)
    trust = ModelRoutingTrustStore.model_validate(trust_raw, strict=True)
    with pytest.raises(ModelRoutingPolicyError, match=error_code):
        verify_model_routing_policy(envelope, trust, now=_now("2026-07-13T00:10:00.000Z"))


def test_expected_org_and_environment_are_local_bindings() -> None:
    envelope, trust = _models()
    with pytest.raises(ModelRoutingPolicyError, match="org_mismatch"):
        verify_model_routing_policy(
            envelope,
            trust,
            now=_now("2026-07-13T00:10:00.000Z"),
            expected_org_id="other-org",
        )
    with pytest.raises(ModelRoutingPolicyError, match="environment_mismatch"):
        verify_model_routing_policy(
            envelope,
            trust,
            now=_now("2026-07-13T00:10:00.000Z"),
            expected_environment="prod",
        )


@pytest.mark.parametrize(
    ("now", "error_code"),
    [
        ("2026-07-13T00:00:00.000Z", "not_yet_valid"),
        ("2026-07-13T00:31:00.000Z", "offline_lease_expired"),
        ("2026-07-13T01:01:00.000Z", "expired"),
    ],
)
def test_validity_and_offline_lease_are_enforced(now, error_code) -> None:
    with pytest.raises(ModelRoutingPolicyError, match=error_code):
        _verify_fixture(now=now)


def test_clock_skew_is_explicit_and_bounded_by_caller() -> None:
    fixture = _fixture()
    envelope, trust = _models(fixture)
    verified = verify_model_routing_policy(
        envelope,
        trust,
        now=_now("2026-07-13T00:00:45.000Z"),
        clock_skew_seconds=15,
    )
    assert verified.claims.policy_id == "routing-golden-v1"

    with pytest.raises(ModelRoutingPolicyError, match="invalid_verification_time"):
        verify_model_routing_policy(
            envelope,
            trust,
            now=datetime(2026, 7, 13, 0, 10),
        )


def test_valid_signature_over_noncanonical_bytes_is_rejected() -> None:
    claims = json.loads(_fixture()["policy"]["signedPayload"])
    envelope_raw, trust_raw = _signed_envelope(claims, canonical=False)
    envelope = ModelRoutingPolicyEnvelope.model_validate(envelope_raw, strict=True)
    trust = ModelRoutingTrustStore.model_validate(trust_raw, strict=True)
    with pytest.raises(ModelRoutingPolicyError, match="non_canonical_payload"):
        verify_model_routing_policy(envelope, trust, now=_now("2026-07-13T00:10:00.000Z"))


def test_strict_claim_schema_and_route_validation_reject_signed_bad_state() -> None:
    claims = json.loads(_fixture()["policy"]["signedPayload"])
    claims["unexpected"] = True
    envelope_raw, trust_raw = _signed_envelope(claims)
    with pytest.raises(ModelRoutingPolicyError, match="malformed_claims"):
        verify_model_routing_policy(
            ModelRoutingPolicyEnvelope.model_validate(envelope_raw, strict=True),
            ModelRoutingTrustStore.model_validate(trust_raw, strict=True),
            now=_now("2026-07-13T00:10:00.000Z"),
        )

    claims.pop("unexpected")
    claims["routes"][1]["requestedModel"] = "reasoning"
    envelope_raw, trust_raw = _signed_envelope(claims)
    with pytest.raises(ModelRoutingPolicyError, match="invalid_routes"):
        verify_model_routing_policy(
            ModelRoutingPolicyEnvelope.model_validate(envelope_raw, strict=True),
            ModelRoutingTrustStore.model_validate(trust_raw, strict=True),
            now=_now("2026-07-13T00:10:00.000Z"),
        )

    claims["routes"][1]["requestedModel"] = "*"
    claims["routes"][0]["limits"]["maxOutputTokens"] = 9_007_199_254_740_992
    envelope_raw, trust_raw = _signed_envelope(claims)
    with pytest.raises(ModelRoutingPolicyError, match="invalid_routes"):
        verify_model_routing_policy(
            ModelRoutingPolicyEnvelope.model_validate(envelope_raw, strict=True),
            ModelRoutingTrustStore.model_validate(trust_raw, strict=True),
            now=_now("2026-07-13T00:10:00.000Z"),
        )


def test_envelope_read_is_bounded_during_the_read(tmp_path: Path) -> None:
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(_fixture()["policy"]), encoding="utf-8")
    with pytest.raises(ModelRoutingPolicyError, match="malformed_envelope"):
        load_model_routing_envelope(path, max_bytes=64)


def _write_store_files(root: Path, fixture: dict | None = None) -> tuple[Path, Path, Path]:
    source = fixture or _fixture()
    candidate = root / "candidate.json"
    last_known_good = root / "last-known-good.json"
    trust = root / "trust.json"
    candidate.write_text(json.dumps(source["policy"], indent=2), encoding="utf-8")
    trust.write_text(json.dumps(_trust_document(source), indent=2), encoding="utf-8")
    return candidate, last_known_good, trust


def _store(root: Path, *, required: bool = True) -> ModelRoutingPolicyStore:
    candidate, last_known_good, trust = _write_store_files(root)
    return ModelRoutingPolicyStore(
        candidate_path=candidate,
        last_known_good_path=last_known_good,
        trust_store_path=trust,
        required=required,
        expected_environment="staging",
        expected_org_id="org-golden",
    )


def test_candidate_activation_atomically_persists_last_known_good(tmp_path: Path) -> None:
    store = _store(tmp_path)
    active = store.activate(now=_now("2026-07-13T00:10:00.000Z"))

    assert active is not None
    assert active.source == "candidate"
    assert active.policy_id == "routing-golden-v1"
    assert store.last_known_good_path.exists()
    assert store.last_known_good_path.stat().st_mode & 0o777 == 0o600
    persisted = json.loads(store.last_known_good_path.read_text(encoding="utf-8"))
    assert persisted["signedPayload"] == _fixture()["policy"]["signedPayload"]


def test_invalid_candidate_uses_still_valid_last_known_good(tmp_path: Path) -> None:
    store = _store(tmp_path)
    first = store.activate(now=_now("2026-07-13T00:10:00.000Z"))
    assert first is not None

    candidate = json.loads(store.candidate_path.read_text(encoding="utf-8"))
    candidate["signature"] = "AAAA"
    store.candidate_path.write_text(json.dumps(candidate), encoding="utf-8")
    recovered = store.activate(now=_now("2026-07-13T00:11:00.000Z"))

    assert recovered is not None
    assert recovered.source == "last-known-good"
    assert recovered.candidate_error_code == "invalid_signature"
    assert recovered.digest == first.digest


def test_key_rotation_advances_revision_and_older_candidate_cannot_roll_back(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    initial = store.activate(now=_now("2026-07-13T00:10:00.000Z"))
    assert initial is not None

    claims = json.loads(_fixture()["policy"]["signedPayload"])
    claims.update(
        {
            "revision": 2,
            "jti": "routing-golden-v1-r2",
            "keyId": "routing-test-key-v2",
        }
    )
    claims["routes"][0]["primaryModel"] = "qwen3:72b"
    rotated_envelope, rotated_trust = _signed_envelope(claims)
    trust = _trust_document()
    trust["entries"].extend(rotated_trust["entries"])
    store.trust_store_path.write_text(json.dumps(trust), encoding="utf-8")
    store.candidate_path.write_text(json.dumps(rotated_envelope), encoding="utf-8")

    rotated = store.activate(now=_now("2026-07-13T00:11:00.000Z"))
    assert rotated is not None
    assert rotated.source == "candidate"
    assert rotated.revision == 2
    assert rotated.digest != initial.digest

    store.candidate_path.write_text(
        json.dumps(_fixture()["policy"]),
        encoding="utf-8",
    )
    recovered = store.activate(now=_now("2026-07-13T00:12:00.000Z"))
    assert recovered is not None
    assert recovered.source == "last-known-good"
    assert recovered.revision == 2
    assert recovered.candidate_error_code == "revision_rollback"


def test_same_revision_with_different_bytes_cannot_replace_last_known_good(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    initial = store.activate(now=_now("2026-07-13T00:10:00.000Z"))
    assert initial is not None

    claims = json.loads(_fixture()["policy"]["signedPayload"])
    claims["keyId"] = "routing-conflict-key"
    claims["routes"][0]["primaryModel"] = "qwen3:72b"
    conflict_envelope, conflict_trust = _signed_envelope(claims)
    trust = _trust_document()
    trust["entries"].extend(conflict_trust["entries"])
    store.trust_store_path.write_text(json.dumps(trust), encoding="utf-8")
    store.candidate_path.write_text(json.dumps(conflict_envelope), encoding="utf-8")

    recovered = store.activate(now=_now("2026-07-13T00:11:00.000Z"))
    assert recovered is not None
    assert recovered.source == "last-known-good"
    assert recovered.digest == initial.digest
    assert recovered.candidate_error_code == "revision_conflict"


def test_revocation_and_offline_expiry_also_invalidate_last_known_good(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    store.activate(now=_now("2026-07-13T00:10:00.000Z"))
    store.candidate_path.unlink()

    trust = json.loads(store.trust_store_path.read_text(encoding="utf-8"))
    trust["revokedJtis"] = ["routing-golden-v1-r1"]
    store.trust_store_path.write_text(json.dumps(trust), encoding="utf-8")
    with pytest.raises(ModelRoutingPolicyActivationError) as revoked:
        store.activate(now=_now("2026-07-13T00:11:00.000Z"))
    assert revoked.value.last_known_good_error_code == "revoked_policy"

    trust["revokedJtis"] = []
    store.trust_store_path.write_text(json.dumps(trust), encoding="utf-8")
    with pytest.raises(ModelRoutingPolicyActivationError) as expired:
        store.activate(now=_now("2026-07-13T00:31:00.000Z"))
    assert expired.value.last_known_good_error_code == "offline_lease_expired"


def test_missing_policy_is_optional_only_when_no_policy_state_exists(tmp_path: Path) -> None:
    optional = ModelRoutingPolicyStore(
        candidate_path=tmp_path / "candidate.json",
        last_known_good_path=tmp_path / "lkg.json",
        trust_store_path=tmp_path / "trust.json",
        required=False,
    )
    assert optional.activate(now=_now("2026-07-13T00:10:00.000Z")) is None

    required = ModelRoutingPolicyStore(
        candidate_path=tmp_path / "candidate.json",
        last_known_good_path=tmp_path / "lkg.json",
        trust_store_path=tmp_path / "trust.json",
        required=True,
    )
    with pytest.raises(ModelRoutingPolicyActivationError, match="policy_required"):
        required.activate(now=_now("2026-07-13T00:10:00.000Z"))


def test_invalid_candidate_without_last_known_good_never_disables_silently(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path, required=False)
    candidate = copy.deepcopy(_fixture()["policy"])
    candidate["signature"] = "AAAA"
    store.candidate_path.write_text(json.dumps(candidate), encoding="utf-8")

    with pytest.raises(ModelRoutingPolicyActivationError) as failure:
        store.activate(now=_now("2026-07-13T00:10:00.000Z"))
    assert failure.value.candidate_error_code == "invalid_signature"
    assert failure.value.last_known_good_error_code == "policy_missing"
