from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType

import pytest

from inference_engine.auth import Identity
from inference_engine.model_routing import (
    ActivatedModelRoutingPolicy,
    ModelRoutingPolicyEnvelope,
    ModelRoutingTrustStore,
    verify_model_routing_policy,
)
from inference_engine.model_routing_runtime import (
    LoadedModelRoutingPricingCatalog,
    ModelRoutingEnforcementError,
    ModelRoutingModelPrice,
    ModelRoutingPricingCatalog,
    ModelRoutingRateLimiter,
    ModelRoutingRuntimeConfigError,
    ModelRoutingRuntimeState,
    build_model_routing_runtime_state,
    enforce_model_routing_request,
    load_model_routing_pricing_catalog,
    model_routing_span_attrs,
    observe_model_routing_usage,
    settle_model_routing_reservation,
)


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "model-routing-policy-v1.json"
FIXTURE_V2_PATH = Path(__file__).parent / "fixtures" / "model-routing-policy-v2.json"
NOW = datetime(2026, 7, 13, 0, 10, tzinfo=UTC)


def _active_policy(path: Path = FIXTURE_PATH) -> ActivatedModelRoutingPolicy:
    fixture = json.loads(path.read_text(encoding="utf-8"))
    envelope = ModelRoutingPolicyEnvelope.model_validate(fixture["policy"], strict=True)
    trust = ModelRoutingTrustStore.model_validate(
        {
            "trustVersion": 1,
            "entries": [fixture["trust"]],
            "revokedKeyIds": [],
            "revokedJtis": [],
        },
        strict=True,
    )
    verified = verify_model_routing_policy(
        envelope,
        trust,
        now=NOW,
        expected_environment="staging",
        expected_org_id="org-golden",
    )
    return ActivatedModelRoutingPolicy(verified=verified, source="candidate")


def _pricing() -> LoadedModelRoutingPricingCatalog:
    prices = [
        ModelRoutingModelPrice(
            model="qwen3:32b",
            input_cost_micros_per_million_tokens=1_000_000,
            output_cost_micros_per_million_tokens=2_000_000,
        ),
        ModelRoutingModelPrice(
            model="llama3.3:70b:openrouter",
            input_cost_micros_per_million_tokens=3_000_000,
            output_cost_micros_per_million_tokens=4_000_000,
        ),
    ]
    catalog = ModelRoutingPricingCatalog(pricing_version=1, models=prices)
    return LoadedModelRoutingPricingCatalog(
        catalog=catalog,
        digest="sha256:pricing",
        by_model=MappingProxyType({price.model: price for price in prices}),
    )


def _state(
    active: ActivatedModelRoutingPolicy | None = None,
    pricing: LoadedModelRoutingPricingCatalog | None = None,
) -> ModelRoutingRuntimeState:
    return ModelRoutingRuntimeState(
        policy=active if active is not None else _active_policy(),
        pricing=pricing if pricing is not None else _pricing(),
    )


def _identity(tenant: str = "runtime") -> Identity:
    return Identity(tenant=tenant, key_id="sk-test", org_id="org-golden")


def _enforce(
    *,
    state: ModelRoutingRuntimeState | None = None,
    requested_model: str = "reasoning",
    input_tokens: int | None = 100,
    output_tokens: int = 10,
    limiter: ModelRoutingRateLimiter | None = None,
    identity: Identity | None = None,
    now: datetime = NOW,
):
    return enforce_model_routing_request(
        state or _state(),
        identity=identity or _identity(),
        requested_model=requested_model,
        input_token_upper_bound=input_tokens,
        output_token_budget=output_tokens,
        rate_limiter=limiter or ModelRoutingRateLimiter(),
        now=now,
    )


def _replace_reasoning_route(active: ActivatedModelRoutingPolicy, **changes):
    claims = active.verified.claims
    route = claims.routes[0].model_copy(update=changes)
    next_claims = claims.model_copy(update={"routes": [route, *claims.routes[1:]]})
    return replace(active, verified=replace(active.verified, claims=next_claims))


def test_pricing_catalog_loads_strictly_and_has_stable_digest(tmp_path: Path) -> None:
    path = tmp_path / "pricing.json"
    path.write_text(
        json.dumps(
            {
                "pricingVersion": 1,
                "models": [
                    {
                        "model": "qwen3:32b",
                        "inputCostMicrosPerMillionTokens": 0,
                        "outputCostMicrosPerMillionTokens": 100_000,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    loaded = load_model_routing_pricing_catalog(path)

    assert loaded is not None
    assert loaded.by_model["qwen3:32b"].output_cost_micros_per_million_tokens == 100_000
    assert loaded.digest.startswith("sha256:")
    assert load_model_routing_pricing_catalog(tmp_path / "missing.json") is None


@pytest.mark.parametrize(
    "payload",
    [
        {"pricingVersion": 1, "models": []},
        {
            "pricingVersion": 1,
            "models": [
                {
                    "model": "x",
                    "inputCostMicrosPerMillionTokens": 0,
                    "outputCostMicrosPerMillionTokens": 0,
                },
                {
                    "model": "x",
                    "inputCostMicrosPerMillionTokens": 0,
                    "outputCostMicrosPerMillionTokens": 0,
                },
            ],
        },
        {
            "pricingVersion": 1,
            "models": [
                {
                    "model": "x",
                    "inputCostMicrosPerMillionTokens": -1,
                    "outputCostMicrosPerMillionTokens": 0,
                }
            ],
        },
    ],
)
def test_pricing_catalog_rejects_unsafe_shapes(tmp_path: Path, payload: dict) -> None:
    path = tmp_path / "pricing.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ModelRoutingRuntimeConfigError):
        load_model_routing_pricing_catalog(path)


def test_runtime_state_requires_deployment_org_when_auth_is_disabled() -> None:
    with pytest.raises(ModelRoutingRuntimeConfigError, match="org_binding_required"):
        build_model_routing_runtime_state(
            _active_policy(),
            _pricing(),
            auth_enabled=False,
            expected_org_id=None,
        )


def test_runtime_state_requires_prices_for_every_costed_candidate() -> None:
    with pytest.raises(ModelRoutingRuntimeConfigError, match="pricing_catalog_required"):
        build_model_routing_runtime_state(
            _active_policy(),
            None,
            auth_enabled=True,
            expected_org_id=None,
        )

    incomplete = _pricing()
    incomplete = replace(
        incomplete,
        by_model=MappingProxyType({"qwen3:32b": incomplete.by_model["qwen3:32b"]}),
    )
    with pytest.raises(ModelRoutingRuntimeConfigError, match="pricing_model_missing"):
        build_model_routing_runtime_state(
            _active_policy(),
            incomplete,
            auth_enabled=True,
            expected_org_id=None,
        )


def test_exact_alias_and_wildcard_resolve_to_signed_candidates() -> None:
    exact = _enforce()
    wildcard = _enforce(requested_model="unlisted", output_tokens=100)

    assert exact is not None
    assert exact.route.route_id == "reasoning"
    assert exact.candidate_models == (
        "qwen3:32b",
        "llama3.3:70b:openrouter",
    )
    assert wildcard is not None
    assert wildcard.route.route_id == "default"
    assert wildcard.candidate_models == ("llama3.2:3b",)


def test_request_without_exact_or_wildcard_route_is_denied() -> None:
    active = _active_policy()
    claims = active.verified.claims.model_copy(
        update={"routes": [active.verified.claims.routes[0]]}
    )
    active = replace(active, verified=replace(active.verified, claims=claims))

    with pytest.raises(ModelRoutingEnforcementError, match="route_not_allowed"):
        _enforce(state=_state(active=active), requested_model="unlisted")


def test_missing_or_wrong_org_identity_is_denied() -> None:
    with pytest.raises(ModelRoutingEnforcementError, match="org_identity_missing"):
        _enforce(identity=Identity(tenant="runtime", key_id="sk-test"))
    with pytest.raises(ModelRoutingEnforcementError, match="org_identity_mismatch"):
        _enforce(identity=Identity(tenant="runtime", key_id="sk-test", org_id="other"))


def test_invalid_alias_and_negative_bounds_cannot_reach_wildcard_route() -> None:
    with pytest.raises(ModelRoutingEnforcementError, match="invalid_requested_model"):
        _enforce(requested_model=" ")
    with pytest.raises(ModelRoutingEnforcementError, match="invalid_request_bounds"):
        _enforce(output_tokens=-1)
    with pytest.raises(ModelRoutingEnforcementError, match="invalid_request_bounds"):
        _enforce(input_tokens=-1)


def test_input_and_output_limits_are_enforced_before_dispatch() -> None:
    with pytest.raises(ModelRoutingEnforcementError, match="input_token_limit_exceeded"):
        _enforce(input_tokens=32_769)
    with pytest.raises(ModelRoutingEnforcementError, match="output_token_limit_exceeded"):
        _enforce(output_tokens=4_097)
    with pytest.raises(ModelRoutingEnforcementError, match="input_token_estimate_unavailable"):
        _enforce(input_tokens=None)


def test_cost_limit_is_worst_case_across_primary_and_all_fallbacks() -> None:
    decision = _enforce(input_tokens=100, output_tokens=10)
    assert decision is not None
    assert decision.estimated_max_cost_micros == 460

    active = _active_policy()
    limits = active.verified.claims.routes[0].limits.model_copy(
        update={"max_cost_micros_per_request": 459}
    )
    active = _replace_reasoning_route(active, limits=limits)
    with pytest.raises(ModelRoutingEnforcementError, match="cost_limit_exceeded"):
        _enforce(state=_state(active=active), input_tokens=100, output_tokens=10)


def test_rate_limit_is_per_policy_route_org_and_tenant() -> None:
    active = _active_policy()
    limits = active.verified.claims.routes[0].limits.model_copy(
        update={"max_requests_per_minute": 2}
    )
    state = _state(active=_replace_reasoning_route(active, limits=limits))
    clock = [10.0]
    limiter = ModelRoutingRateLimiter(clock=lambda: clock[0])

    _enforce(state=state, limiter=limiter)
    _enforce(state=state, limiter=limiter)
    with pytest.raises(ModelRoutingEnforcementError) as raised:
        _enforce(state=state, limiter=limiter)
    assert raised.value.code == "rate_limit_exceeded"
    assert raised.value.retry_after_seconds == 60

    _enforce(state=state, limiter=limiter, identity=_identity("other-tenant"))
    clock[0] = 70.1
    _enforce(state=state, limiter=limiter)


def test_request_time_freshness_rejects_offline_lease_and_expiry() -> None:
    with pytest.raises(ModelRoutingEnforcementError, match="policy_offline_lease_expired"):
        _enforce(now=datetime(2026, 7, 13, 0, 31, tzinfo=UTC))
    with pytest.raises(ModelRoutingEnforcementError, match="policy_expired"):
        _enforce(now=datetime(2026, 7, 13, 1, 1, tzinfo=UTC))


def test_policy_evidence_attributes_are_complete_and_payload_free() -> None:
    decision = _enforce()
    attrs = model_routing_span_attrs(
        decision,
        candidate_model="qwen3:32b",
        candidate_index=0,
    )

    assert attrs["model_routing.policy.id"] == "routing-golden-v1"
    assert attrs["model_routing.policy.revision"] == 1
    assert attrs["model_routing.policy.release_id"] == "release-golden-model-v1"
    assert attrs["model_routing.policy.deployment_id"] == "model-plane-golden-v1"
    assert attrs["model_routing.policy.org_id"] == "org-golden"
    assert attrs["model_routing.policy.environment"] == "staging"
    assert attrs["prometa.artifact.type"] == "model-routing-policy"
    assert attrs["prometa.artifact.digest"] == attrs["model_routing.policy.digest"]
    assert attrs["prometa.policy.digest"] == attrs["model_routing.policy.digest"]
    assert attrs["prometa.release.id"] == "release-golden-model-v1"
    assert attrs["prometa.deployment.id"] == "model-plane-golden-v1"
    assert attrs["prometa.environment"] == "staging"
    assert attrs["model_routing.route.id"] == "reasoning"
    assert attrs["model_routing.route.selected_model"] == "qwen3:32b"
    assert attrs["model_routing.pricing.digest"] == "sha256:pricing"
    assert "signed_payload" not in attrs


def test_v2_policy_reserves_tokens_and_window_spend_on_admission() -> None:
    state = _state(active=_active_policy(FIXTURE_V2_PATH))

    decision = _enforce(state=state, input_tokens=100, output_tokens=10)

    assert decision is not None
    assert decision.route.candidate_weights == [90, 10]
    assert decision.route.shadow_model == "llama3.2:3b"
    assert decision.candidate_models == ("qwen3:32b", "llama3.3:70b:openrouter")
    assert decision.estimated_max_cost_micros == 460
    assert decision.reserved_tokens == 110
    assert decision.reserved_cost_micros == 460
    assert decision.reservation is not None
    # The token reserve is the input estimate plus this request's own output
    # budget, not the route's 4096-token maxOutputTokens ceiling.
    assert decision.route.limits.max_output_tokens == 4_096
    assert decision.reservation.tokens is not None
    assert decision.reservation.tokens.amount == 110
    assert decision.reservation.spend is not None
    assert decision.reservation.spend.amount == 460

    attrs = model_routing_span_attrs(decision)
    assert attrs["model_routing.limit.max_requests_per_minute"] == 120
    assert attrs["model_routing.limit.max_tokens_per_minute"] == 240_000
    assert attrs["model_routing.limit.max_cost_micros_per_window"] == 5_000_000
    assert attrs["model_routing.limit.budget_window_seconds"] == 3_600
    assert attrs["model_routing.reserved_tokens"] == 110
    assert attrs["model_routing.reserved_cost_micros"] == 460


def test_token_rate_limit_denies_before_the_request_ceiling_is_reached() -> None:
    state = _state(active=_active_policy(FIXTURE_V2_PATH))
    limiter = ModelRoutingRateLimiter(clock=lambda: 10.0)

    # 240_000 tokens per minute against a 36_864-token reservation admits six.
    for _ in range(6):
        decision = _enforce(
            state=state,
            limiter=limiter,
            input_tokens=32_768,
            output_tokens=4_096,
        )
        assert decision is not None

    with pytest.raises(ModelRoutingEnforcementError) as raised:
        _enforce(state=state, limiter=limiter, input_tokens=32_768, output_tokens=4_096)
    assert raised.value.code == "token_rate_limit_exceeded"
    assert raised.value.limit_tokens == 240_000
    assert raised.value.limit_requests is None
    assert raised.value.retry_after_seconds == 60


def test_settling_real_usage_frees_the_over_reserved_token_budget() -> None:
    state = _state(active=_active_policy(FIXTURE_V2_PATH))
    limiter = ModelRoutingRateLimiter(clock=lambda: 10.0)

    for _ in range(6):
        decision = _enforce(
            state=state,
            limiter=limiter,
            input_tokens=32_768,
            output_tokens=4_096,
        )
        assert decision is not None
        observe_model_routing_usage(decision.reservation, tokens=64, cost_micros=7)
        settle_model_routing_reservation(limiter, decision.reservation)

    # 6 * 64 real tokens, not 6 * 36_864 reserved ones, so the window is open.
    assert _enforce(state=state, limiter=limiter, input_tokens=32_768, output_tokens=4_096)


def test_settling_is_idempotent_so_a_double_exit_cannot_double_credit() -> None:
    state = _state(active=_active_policy(FIXTURE_V2_PATH))
    limiter = ModelRoutingRateLimiter(clock=lambda: 10.0)

    decision = _enforce(state=state, limiter=limiter, input_tokens=32_768, output_tokens=4_096)
    assert decision is not None
    observe_model_routing_usage(decision.reservation, tokens=100, cost_micros=10)
    settle_model_routing_reservation(limiter, decision.reservation)
    observe_model_routing_usage(decision.reservation, tokens=1, cost_micros=1)
    settle_model_routing_reservation(limiter, decision.reservation)

    window = limiter._buckets[("tpm", decision.active.digest, "reasoning", "org-golden", "runtime")]
    assert window.total == 100


def test_a_request_that_never_reported_usage_releases_its_whole_reservation() -> None:
    state = _state(active=_active_policy(FIXTURE_V2_PATH))
    limiter = ModelRoutingRateLimiter(clock=lambda: 10.0)

    for _ in range(6):
        decision = _enforce(
            state=state,
            limiter=limiter,
            input_tokens=32_768,
            output_tokens=4_096,
        )
        assert decision is not None
        settle_model_routing_reservation(limiter, decision.reservation)

    assert _enforce(state=state, limiter=limiter, input_tokens=32_768, output_tokens=4_096)


def test_served_but_unpriced_usage_keeps_its_window_spend_hold() -> None:
    state = _state(active=_active_policy(FIXTURE_V2_PATH))
    limiter = ModelRoutingRateLimiter(clock=lambda: 10.0)

    decision = _enforce(state=state, limiter=limiter, input_tokens=100, output_tokens=10)
    assert decision is not None
    observe_model_routing_usage(decision.reservation, tokens=64, cost_micros=None)
    settle_model_routing_reservation(limiter, decision.reservation)

    key = (decision.active.digest, "reasoning", "org-golden", "runtime")
    assert limiter._buckets[("tpm", *key)].total == 64
    assert limiter._buckets[("spend", *key)].total == 460


def test_window_spend_budget_denies_once_the_window_is_exhausted() -> None:
    active = _active_policy(FIXTURE_V2_PATH)
    limits = active.verified.claims.routes[0].limits.model_copy(
        update={
            "max_requests_per_minute": None,
            "max_tokens_per_minute": None,
            "max_cost_micros_per_request": None,
            "max_cost_micros_per_window": 1_000,
            "budget_window_seconds": 3_600,
        }
    )
    state = _state(active=_replace_reasoning_route(active, limits=limits))
    clock = [10.0]
    limiter = ModelRoutingRateLimiter(clock=lambda: clock[0])

    for _ in range(2):
        assert _enforce(state=state, limiter=limiter, input_tokens=100, output_tokens=10)
    with pytest.raises(ModelRoutingEnforcementError) as raised:
        _enforce(state=state, limiter=limiter, input_tokens=100, output_tokens=10)

    assert raised.value.code == "budget_exceeded"
    assert raised.value.retry_after_seconds == 3_600
    assert raised.value.limit_tokens is None
    assert raised.value.limit_requests is None

    # The budget window is an hour, not the RPM minute.
    clock[0] = 100.0
    with pytest.raises(ModelRoutingEnforcementError, match="budget_exceeded"):
        _enforce(state=state, limiter=limiter, input_tokens=100, output_tokens=10)
    clock[0] = 3_610.1
    assert _enforce(state=state, limiter=limiter, input_tokens=100, output_tokens=10)


def test_window_budget_needs_pricing_and_an_input_estimate() -> None:
    active = _active_policy(FIXTURE_V2_PATH)
    unpriced = ModelRoutingRuntimeState(policy=active, pricing=None)
    with pytest.raises(ModelRoutingEnforcementError, match="pricing_catalog_unavailable"):
        _enforce(state=unpriced)
    with pytest.raises(ModelRoutingEnforcementError, match="input_token_estimate_unavailable"):
        _enforce(state=_state(active=active), input_tokens=None)


def test_a_costed_window_route_cannot_activate_without_pricing() -> None:
    active = _active_policy(FIXTURE_V2_PATH)
    limits = active.verified.claims.routes[0].limits.model_copy(
        update={"max_cost_micros_per_request": None}
    )
    with pytest.raises(ModelRoutingRuntimeConfigError) as raised:
        build_model_routing_runtime_state(
            _replace_reasoning_route(active, limits=limits),
            None,
            auth_enabled=True,
            expected_org_id="org-golden",
        )
    assert raised.value.code == "pricing_catalog_required"


def test_denial_on_one_dimension_leaves_the_others_uncharged() -> None:
    active = _active_policy(FIXTURE_V2_PATH)
    limits = active.verified.claims.routes[0].limits.model_copy(
        update={"max_requests_per_minute": 10, "max_tokens_per_minute": 200}
    )
    state = _state(active=_replace_reasoning_route(active, limits=limits))
    limiter = ModelRoutingRateLimiter(clock=lambda: 10.0)

    with pytest.raises(ModelRoutingEnforcementError, match="token_rate_limit_exceeded"):
        _enforce(state=state, limiter=limiter, input_tokens=300, output_tokens=0)

    key = (state.policy.digest, "reasoning", "org-golden", "runtime")
    # Windows are evaluated in order and the first denial ends the admission, so
    # spend is never reached. The two that were reached carry no charge.
    for dimension in ("rpm", "tpm"):
        assert limiter._buckets[(dimension, *key)].total == 0
    assert ("spend", *key) not in limiter._buckets


def test_v1_policies_and_null_v2_fields_reserve_nothing() -> None:
    decision = _enforce()

    assert decision is not None
    assert decision.reservation is None
    assert decision.reserved_tokens is None
    assert decision.reserved_cost_micros is None

    wildcard = _enforce(state=_state(active=_active_policy(FIXTURE_V2_PATH)), requested_model="chat")
    assert wildcard is not None
    assert wildcard.reservation is None
    assert wildcard.reserved_tokens is None


def test_parallel_admissions_never_overshoot_the_token_window() -> None:
    active = _active_policy(FIXTURE_V2_PATH)
    limits = active.verified.claims.routes[0].limits.model_copy(
        update={
            "max_requests_per_minute": None,
            "max_cost_micros_per_request": None,
            "max_cost_micros_per_window": None,
            "budget_window_seconds": None,
            "max_tokens_per_minute": 1_000,
        }
    )
    state = _state(active=_replace_reasoning_route(active, limits=limits))
    limiter = ModelRoutingRateLimiter(clock=lambda: 10.0)
    barrier = threading.Barrier(16)

    def attempt(_: int) -> str:
        barrier.wait()
        try:
            _enforce(state=state, limiter=limiter, input_tokens=90, output_tokens=10)
            return "accepted"
        except ModelRoutingEnforcementError as exc:
            assert exc.code == "token_rate_limit_exceeded"
            return "denied"

    with ThreadPoolExecutor(max_workers=16) as pool:
        results = list(pool.map(attempt, range(16)))

    assert results.count("accepted") == 10
    key = ("tpm", state.policy.digest, "reasoning", "org-golden", "runtime")
    assert limiter._buckets[key].total == 1_000


def test_policy_evidence_reports_deployment_shared_rate_limit_scope() -> None:
    class SharedScopeLimiter(ModelRoutingRateLimiter):
        scope = "deployment-shared"

    decision = _enforce(limiter=SharedScopeLimiter())
    attrs = model_routing_span_attrs(decision)

    assert attrs["model_routing.rate_limit.scope"] == "deployment-shared"
