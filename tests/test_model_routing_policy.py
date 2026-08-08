from __future__ import annotations

import base64
import copy
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import get_args

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
from pydantic.alias_generators import to_camel

from inference_engine.model_routing import (
    MODEL_ROUTING_POLICY_VERSION_V1,
    MODEL_ROUTING_POLICY_VERSION_V2,
    MODEL_ROUTING_POLICY_VERSIONS,
    _POLICY_VERSION_INTRODUCED_LIMIT_FIELDS,
    _POLICY_VERSION_INTRODUCED_ROUTE_FIELDS,
    _POLICY_VERSION_ROUTE_AND_LIMIT_FIELDS,
    _SHAPE_GATED_LIMIT_FIELDS,
    _SHAPE_GATED_ROUTE_FIELDS,
    ModelRoutingLimits,
    ModelRoutingPolicyActivationError,
    ModelRoutingPolicyEnvelope,
    ModelRoutingPolicyError,
    ModelRoutingPolicyStore,
    ModelRoutingPolicyVersion,
    ModelRoutingRoute,
    ModelRoutingTrustStore,
    _check_introduction_table,
    _cumulative_post_v1_route_and_limit_fields,
    canonical_json,
    load_model_routing_envelope,
    model_routing_policy_digest,
    verify_model_routing_policy,
)
from inference_engine.model_routing_status import ModelRoutingPolicyStatus


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "model-routing-policy-v1.json"
FIXTURE_V2_PATH = Path(__file__).parent / "fixtures" / "model-routing-policy-v2.json"


def _fixture(path: Path = FIXTURE_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


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
        "policyVersion": claims["policyVersion"],
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


def _verify_fixture(*, now: str = "2026-07-13T00:10:00.000Z", fixture: dict | None = None):
    fixture = fixture or _fixture()
    envelope, trust = _models(fixture)
    return verify_model_routing_policy(
        envelope,
        trust,
        now=_now(now),
        expected_audience=fixture["verification"]["expectedAudience"],
        expected_environment=fixture["verification"]["expectedEnvironment"],
        expected_org_id=fixture["verification"]["expectedOrgId"],
    )


V2_ROUTE_KEYS = ("candidateWeights", "shadowModel")
V2_LIMIT_KEYS = ("maxTokensPerMinute", "maxCostMicrosPerWindow", "budgetWindowSeconds")


def _verify_claims(
    claims: dict,
    *,
    now: str = "2026-07-13T00:10:00.000Z",
    expected_org_id: str | None = None,
):
    envelope_raw, trust_raw = _signed_envelope(claims)
    return verify_model_routing_policy(
        ModelRoutingPolicyEnvelope.model_validate(envelope_raw, strict=True),
        ModelRoutingTrustStore.model_validate(trust_raw, strict=True),
        now=_now(now),
        expected_org_id=expected_org_id,
    )


def _v2_container(route: dict, key: str) -> dict:
    return route if key in V2_ROUTE_KEYS else route["limits"]


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


def test_cross_language_golden_vector_v2_verifies_exact_bytes() -> None:
    fixture = _fixture(FIXTURE_V2_PATH)
    verified = _verify_fixture(fixture=fixture)

    assert verified.digest == fixture["verification"]["expectedDigest"]
    assert verified.digest == model_routing_policy_digest(fixture["policy"]["signedPayload"])
    assert (
        canonical_json(json.loads(fixture["policy"]["signedPayload"]))
        == (fixture["policy"]["signedPayload"])
    )
    assert verified.claims.policy_version == 2
    assert [route.route_id for route in verified.claims.routes] == [
        "reasoning",
        "default",
    ]

    reasoning, default = verified.claims.routes
    assert reasoning.limits.max_tokens_per_minute == 240_000
    assert reasoning.limits.max_cost_micros_per_window == 5_000_000
    assert reasoning.limits.budget_window_seconds == 3600
    assert reasoning.candidate_weights == [90, 10]
    assert reasoning.shadow_model == "llama3.2:3b"
    assert default.limits.max_tokens_per_minute is None
    assert default.limits.max_cost_micros_per_window is None
    assert default.limits.budget_window_seconds is None
    assert default.candidate_weights is None
    assert default.shadow_model is None


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("candidateWeights", None),
        ("shadowModel", None),
        ("maxTokensPerMinute", None),
        ("maxCostMicrosPerWindow", None),
        ("budgetWindowSeconds", None),
        ("shadowModel", "llama3.2:3b"),
    ],
)
def test_v1_policy_carrying_v2_fields_is_rejected(key: str, value) -> None:
    claims = json.loads(_fixture()["policy"]["signedPayload"])
    _v2_container(claims["routes"][0], key)[key] = value
    with pytest.raises(ModelRoutingPolicyError, match="malformed_claims"):
        _verify_claims(claims)


@pytest.mark.parametrize(
    ("claim", "value", "alone_code", "expected_org_id"),
    [
        ("revision", 0, "invalid_revision", None),
        ("audience", "somebody-else", "audience_mismatch", None),
        ("subject", "model-routing-policy:other", "envelope_claim_mismatch", None),
        ("orgId", "org-other", "org_mismatch", "org-golden"),
    ],
)
@pytest.mark.parametrize("v2_key", [*V2_ROUTE_KEYS, *V2_LIMIT_KEYS])
def test_v1_shape_violation_outranks_every_claim_level_error(
    v2_key: str,
    claim: str,
    value,
    alone_code: str,
    expected_org_id: str | None,
) -> None:
    claims = json.loads(_fixture()["policy"]["signedPayload"])
    claims[claim] = value
    with pytest.raises(ModelRoutingPolicyError, match=alone_code):
        _verify_claims(claims, expected_org_id=expected_org_id)

    _v2_container(claims["routes"][0], v2_key)[v2_key] = None
    with pytest.raises(ModelRoutingPolicyError, match="malformed_claims"):
        _verify_claims(claims, expected_org_id=expected_org_id)


def test_accepted_versions_enforced_advertised_and_shape_gated_are_one_set() -> None:
    assert get_args(ModelRoutingPolicyVersion) == (
        MODEL_ROUTING_POLICY_VERSION_V1,
        MODEL_ROUTING_POLICY_VERSION_V2,
    )
    assert ModelRoutingPolicyStatus(active=False).accepted_policy_versions == list(
        MODEL_ROUTING_POLICY_VERSIONS
    )
    assert set(_POLICY_VERSION_ROUTE_AND_LIMIT_FIELDS) == set(MODEL_ROUTING_POLICY_VERSIONS)
    assert set(_POLICY_VERSION_INTRODUCED_ROUTE_FIELDS) == set(MODEL_ROUTING_POLICY_VERSIONS)
    assert set(_POLICY_VERSION_INTRODUCED_LIMIT_FIELDS) == set(MODEL_ROUTING_POLICY_VERSIONS)


def test_v2_shape_gate_covers_every_non_v1_field_of_both_route_models() -> None:
    defaulted = {
        name
        for model in (ModelRoutingRoute, ModelRoutingLimits)
        for name, field in model.model_fields.items()
        if not field.is_required()
    }
    assert _POLICY_VERSION_ROUTE_AND_LIMIT_FIELDS[2] == defaulted
    assert _POLICY_VERSION_ROUTE_AND_LIMIT_FIELDS[1] == frozenset()
    assert _SHAPE_GATED_ROUTE_FIELDS | _SHAPE_GATED_LIMIT_FIELDS == defaulted
    assert {to_camel(name) for name in _POLICY_VERSION_INTRODUCED_ROUTE_FIELDS[2]} == set(
        V2_ROUTE_KEYS
    )
    assert {to_camel(name) for name in _POLICY_VERSION_INTRODUCED_LIMIT_FIELDS[2]} == set(
        V2_LIMIT_KEYS
    )


def test_introducing_a_later_version_field_cannot_change_an_earlier_versions_required_set() -> None:
    v3 = max(MODEL_ROUTING_POLICY_VERSIONS) + 1
    route_fields = {**_POLICY_VERSION_INTRODUCED_ROUTE_FIELDS, v3: frozenset({"canary_model"})}
    limit_fields = {
        **_POLICY_VERSION_INTRODUCED_LIMIT_FIELDS,
        v3: frozenset({"max_concurrent_requests"}),
    }
    extended = _cumulative_post_v1_route_and_limit_fields(
        (*MODEL_ROUTING_POLICY_VERSIONS, v3),
        route_fields,
        limit_fields,
    )

    for version in MODEL_ROUTING_POLICY_VERSIONS:
        assert extended[version] == _POLICY_VERSION_ROUTE_AND_LIMIT_FIELDS[version]
    v2_fields = _POLICY_VERSION_ROUTE_AND_LIMIT_FIELDS[MODEL_ROUTING_POLICY_VERSION_V2]
    assert extended[v3] == v2_fields | {"canary_model", "max_concurrent_requests"}


def test_introduction_table_self_check_rejects_a_version_it_does_not_cover() -> None:
    table = {
        version: names
        for version, names in _POLICY_VERSION_INTRODUCED_ROUTE_FIELDS.items()
        if version != MODEL_ROUTING_POLICY_VERSION_V2
    }
    with pytest.raises(RuntimeError, match="must cover exactly the accepted versions"):
        _check_introduction_table(table, ModelRoutingRoute, "route")

    assert set(_POLICY_VERSION_INTRODUCED_ROUTE_FIELDS) == set(MODEL_ROUTING_POLICY_VERSIONS)


@pytest.mark.parametrize(
    "route_fields_v2",
    [
        frozenset({"candidate_weights"}),
        frozenset({"candidate_weights", "shadow_model", "route_id"}),
        frozenset({"candidate_weights", "shadow_model", "canary_model"}),
    ],
    ids=["field-left-unassigned", "field-assigned-twice", "field-not-on-the-model"],
)
def test_introduction_table_self_check_rejects_a_broken_field_assignment(
    route_fields_v2: frozenset[str],
) -> None:
    table = {
        **_POLICY_VERSION_INTRODUCED_ROUTE_FIELDS,
        MODEL_ROUTING_POLICY_VERSION_V2: route_fields_v2,
    }
    with pytest.raises(RuntimeError, match="must assign every field of ModelRoutingRoute"):
        _check_introduction_table(table, ModelRoutingRoute, "route")

    assert _POLICY_VERSION_INTRODUCED_ROUTE_FIELDS[
        MODEL_ROUTING_POLICY_VERSION_V2
    ] == frozenset({"candidate_weights", "shadow_model"})


def test_introduction_table_self_check_rejects_a_requiredness_mismatch() -> None:
    table = {
        MODEL_ROUTING_POLICY_VERSION_V1: frozenset(
            {"requested_model", "primary_model", "fallback_models", "limits", "candidate_weights"}
        ),
        MODEL_ROUTING_POLICY_VERSION_V2: frozenset({"route_id", "shadow_model"}),
    }
    with pytest.raises(RuntimeError, match="wire requiredness does not match") as exc:
        _check_introduction_table(table, ModelRoutingRoute, "route")
    message = str(exc.value)
    assert "'route_id'" in message or "'candidate_weights'" in message

    assert ModelRoutingRoute.model_fields["route_id"].is_required()
    assert not ModelRoutingRoute.model_fields["candidate_weights"].is_required()


@pytest.mark.parametrize("key", [*V2_ROUTE_KEYS, *V2_LIMIT_KEYS])
def test_v2_policy_missing_v2_fields_is_rejected(key: str) -> None:
    claims = json.loads(_fixture(FIXTURE_V2_PATH)["policy"]["signedPayload"])
    _v2_container(claims["routes"][0], key).pop(key)
    with pytest.raises(ModelRoutingPolicyError, match="malformed_claims"):
        _verify_claims(claims)


@pytest.mark.parametrize(
    ("fixture_path", "envelope_version"),
    [(FIXTURE_PATH, 2), (FIXTURE_V2_PATH, 1)],
)
def test_envelope_and_claims_policy_versions_must_agree(
    fixture_path: Path,
    envelope_version: int,
) -> None:
    claims = json.loads(_fixture(fixture_path)["policy"]["signedPayload"])
    envelope_raw, trust_raw = _signed_envelope(claims)
    envelope_raw["policyVersion"] = envelope_version
    with pytest.raises(ModelRoutingPolicyError, match="envelope_claim_mismatch"):
        verify_model_routing_policy(
            ModelRoutingPolicyEnvelope.model_validate(envelope_raw, strict=True),
            ModelRoutingTrustStore.model_validate(trust_raw, strict=True),
            now=_now("2026-07-13T00:10:00.000Z"),
        )


def test_unknown_policy_version_is_rejected(tmp_path: Path) -> None:
    claims = json.loads(_fixture(FIXTURE_V2_PATH)["policy"]["signedPayload"])
    claims["policyVersion"] = 3
    envelope_raw, trust_raw = _signed_envelope(claims)
    envelope_raw["policyVersion"] = 2
    with pytest.raises(ModelRoutingPolicyError, match="malformed_claims"):
        verify_model_routing_policy(
            ModelRoutingPolicyEnvelope.model_validate(envelope_raw, strict=True),
            ModelRoutingTrustStore.model_validate(trust_raw, strict=True),
            now=_now("2026-07-13T00:10:00.000Z"),
        )

    envelope_raw["policyVersion"] = 3
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(envelope_raw), encoding="utf-8")
    with pytest.raises(ModelRoutingPolicyError, match="malformed_envelope"):
        load_model_routing_envelope(path)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("budgetWindowSeconds", None),
        ("maxCostMicrosPerWindow", None),
        ("budgetWindowSeconds", 0),
        ("budgetWindowSeconds", -1),
        ("budgetWindowSeconds", 86_401),
        ("maxTokensPerMinute", 0),
        ("maxTokensPerMinute", 9_007_199_254_740_992),
        ("maxCostMicrosPerWindow", 0),
    ],
)
def test_v2_route_limits_reject_nonsensical_values(key: str, value) -> None:
    claims = json.loads(_fixture(FIXTURE_V2_PATH)["policy"]["signedPayload"])
    claims["routes"][0]["limits"][key] = value
    with pytest.raises(ModelRoutingPolicyError, match="invalid_routes"):
        _verify_claims(claims)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("candidateWeights", [1]),
        ("candidateWeights", [1, 2, 3]),
        ("candidateWeights", []),
        ("candidateWeights", [1, -1]),
        ("candidateWeights", [0, 0]),
        ("candidateWeights", [9_007_199_254_740_991, 1]),
        ("shadowModel", ""),
        ("shadowModel", " x"),
        ("shadowModel", "qwen3:32b"),
    ],
)
def test_v2_candidate_weights_and_shadow_model_reject_nonsensical_values(key: str, value) -> None:
    claims = json.loads(_fixture(FIXTURE_V2_PATH)["policy"]["signedPayload"])
    claims["routes"][0][key] = value
    with pytest.raises(ModelRoutingPolicyError, match="invalid_routes"):
        _verify_claims(claims)


@pytest.mark.parametrize("route_index", [0, 1])
def test_v2_shadow_model_may_not_be_any_live_candidate_of_its_own_route(route_index: int) -> None:
    claims = json.loads(_fixture(FIXTURE_V2_PATH)["policy"]["signedPayload"])
    route = claims["routes"][route_index]
    for candidate in (route["primaryModel"], *route["fallbackModels"]):
        route["shadowModel"] = candidate
        with pytest.raises(ModelRoutingPolicyError, match="invalid_routes"):
            _verify_claims(claims)


def test_v2_shadow_model_may_name_another_routes_candidate() -> None:
    claims = json.loads(_fixture(FIXTURE_V2_PATH)["policy"]["signedPayload"])
    claims["routes"][1]["shadowModel"] = claims["routes"][0]["fallbackModels"][0]
    verified = _verify_claims(claims)
    assert verified.claims.routes[1].shadow_model == "llama3.3:70b:openrouter"


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("maxTokensPerMinute", "240000"),
        ("budgetWindowSeconds", True),
        ("candidateWeights", [1.5]),
        ("shadowModel", 5),
    ],
)
def test_v2_limits_reject_non_integer_types(key: str, value) -> None:
    claims = json.loads(_fixture(FIXTURE_V2_PATH)["policy"]["signedPayload"])
    _v2_container(claims["routes"][0], key)[key] = value
    with pytest.raises(ModelRoutingPolicyError, match="malformed_claims"):
        _verify_claims(claims)


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


def _store(
    root: Path,
    *,
    required: bool = True,
    fixture: dict | None = None,
) -> ModelRoutingPolicyStore:
    candidate, last_known_good, trust = _write_store_files(root, fixture)
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


def test_v2_policy_activates_and_persists_last_known_good_byte_identically(
    tmp_path: Path,
) -> None:
    fixture = _fixture(FIXTURE_V2_PATH)
    store = _store(tmp_path, fixture=fixture)

    active = store.activate(now=_now("2026-07-13T00:10:00.000Z"))

    assert active is not None
    assert active.source == "candidate"
    assert active.policy_id == "routing-golden-v2"
    assert store.last_known_good_path.stat().st_mode & 0o777 == 0o600
    persisted = json.loads(store.last_known_good_path.read_text(encoding="utf-8"))
    assert persisted["signedPayload"] == fixture["policy"]["signedPayload"]
