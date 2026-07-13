"""Request-time enforcement for a locally activated model-routing policy."""

from __future__ import annotations

import hashlib
import json
import math
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import MappingProxyType
from typing import Literal, Mapping

from pydantic import BaseModel, ConfigDict, StrictInt, StrictStr, ValidationError
from pydantic.alias_generators import to_camel

from .auth import Identity
from .model_routing import (
    MAX_SAFE_INTEGER,
    ActivatedModelRoutingPolicy,
    ModelRoutingRoute,
    canonical_json,
)

MODEL_ROUTING_PRICING_VERSION = 1
MODEL_ROUTING_RATE_LIMIT_WINDOW_SECONDS = 60.0


class _RuntimeConfigModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        extra="forbid",
        frozen=True,
    )


class ModelRoutingModelPrice(_RuntimeConfigModel):
    model: StrictStr
    input_cost_micros_per_million_tokens: StrictInt
    output_cost_micros_per_million_tokens: StrictInt


class ModelRoutingPricingCatalog(_RuntimeConfigModel):
    pricing_version: Literal[MODEL_ROUTING_PRICING_VERSION]
    models: list[ModelRoutingModelPrice]


@dataclass(frozen=True)
class LoadedModelRoutingPricingCatalog:
    catalog: ModelRoutingPricingCatalog
    digest: str
    by_model: Mapping[str, ModelRoutingModelPrice]


@dataclass(frozen=True)
class ModelRoutingRuntimeState:
    policy: ActivatedModelRoutingPolicy | None = None
    pricing: LoadedModelRoutingPricingCatalog | None = None


@dataclass(frozen=True)
class ModelRoutingDecision:
    active: ActivatedModelRoutingPolicy
    route: ModelRoutingRoute
    requested_model: str
    candidate_models: tuple[str, ...]
    input_token_upper_bound: int | None
    output_token_budget: int
    estimated_max_cost_micros: int | None
    pricing_digest: str | None


class ModelRoutingRuntimeConfigError(ValueError):
    """Stable failure for deployment state that cannot enforce a policy."""

    def __init__(self, code: str, detail: str | None = None) -> None:
        self.code = code
        self.detail = detail
        super().__init__(code if detail is None else f"{code}: {detail}")


class ModelRoutingEnforcementError(ValueError):
    """Stable payload-free request denial."""

    def __init__(
        self,
        code: str,
        *,
        policy_id: str,
        route_id: str | None = None,
        retry_after_seconds: int | None = None,
    ) -> None:
        self.code = code
        self.policy_id = policy_id
        self.route_id = route_id
        self.retry_after_seconds = retry_after_seconds
        super().__init__(code)


def _validate_pricing_catalog(catalog: ModelRoutingPricingCatalog) -> None:
    if not catalog.models:
        raise ModelRoutingRuntimeConfigError("pricing_catalog_empty")
    seen: set[str] = set()
    for price in catalog.models:
        model = price.model
        if not model or model != model.strip() or model in seen:
            raise ModelRoutingRuntimeConfigError("pricing_catalog_invalid")
        seen.add(model)
        for value in (
            price.input_cost_micros_per_million_tokens,
            price.output_cost_micros_per_million_tokens,
        ):
            if value < 0 or value > MAX_SAFE_INTEGER:
                raise ModelRoutingRuntimeConfigError("pricing_catalog_invalid")


def load_model_routing_pricing_catalog(
    path: Path | str,
    *,
    max_bytes: int = 1_048_576,
) -> LoadedModelRoutingPricingCatalog | None:
    pricing_path = Path(path)
    if not pricing_path.exists():
        return None
    try:
        with pricing_path.open("rb") as handle:
            encoded = handle.read(max_bytes + 1)
        if not encoded or len(encoded) > max_bytes:
            raise ModelRoutingRuntimeConfigError("pricing_catalog_invalid")
        raw = json.loads(encoded.decode("utf-8"))
        catalog = ModelRoutingPricingCatalog.model_validate(raw, strict=True)
    except ModelRoutingRuntimeConfigError:
        raise
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        RecursionError,
        ValidationError,
    ) as exc:
        raise ModelRoutingRuntimeConfigError("pricing_catalog_invalid") from exc

    _validate_pricing_catalog(catalog)
    canonical = canonical_json(catalog.model_dump(by_alias=True))
    digest = f"sha256:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"
    by_model = MappingProxyType({price.model: price for price in catalog.models})
    return LoadedModelRoutingPricingCatalog(catalog=catalog, digest=digest, by_model=by_model)


def validate_model_routing_runtime_state(
    state: ModelRoutingRuntimeState,
    *,
    auth_enabled: bool,
    expected_org_id: str | None,
) -> None:
    active = state.policy
    if active is None:
        return

    claims = active.verified.claims
    if not auth_enabled and expected_org_id != claims.org_id:
        raise ModelRoutingRuntimeConfigError("org_binding_required")

    for route in claims.routes:
        if route.limits.max_cost_micros_per_request is None:
            continue
        if state.pricing is None:
            raise ModelRoutingRuntimeConfigError("pricing_catalog_required")
        missing = [
            model
            for model in (route.primary_model, *route.fallback_models)
            if model not in state.pricing.by_model
        ]
        if missing:
            raise ModelRoutingRuntimeConfigError(
                "pricing_model_missing",
                ",".join(missing),
            )


def build_model_routing_runtime_state(
    policy: ActivatedModelRoutingPolicy | None,
    pricing: LoadedModelRoutingPricingCatalog | None,
    *,
    auth_enabled: bool,
    expected_org_id: str | None,
) -> ModelRoutingRuntimeState:
    state = ModelRoutingRuntimeState(policy=policy, pricing=pricing)
    validate_model_routing_runtime_state(
        state,
        auth_enabled=auth_enabled,
        expected_org_id=expected_org_id or None,
    )
    return state


class ModelRoutingRateLimiter:
    """Process-local sliding-window limiter keyed by policy, route, and tenant."""

    def __init__(
        self,
        *,
        max_buckets: int = 10_000,
        clock=time.monotonic,
    ) -> None:
        if max_buckets < 1:
            raise ValueError("max_buckets must be positive")
        self._max_buckets = max_buckets
        self._clock = clock
        self._lock = threading.Lock()
        self._buckets: OrderedDict[tuple[str, str, str, str], deque[float]] = OrderedDict()

    def consume(
        self,
        *,
        digest: str,
        route_id: str,
        org_id: str,
        tenant: str,
        limit: int,
        policy_id: str,
    ) -> None:
        key = (digest, route_id, org_id, tenant)
        now = self._clock()
        cutoff = now - MODEL_ROUTING_RATE_LIMIT_WINDOW_SECONDS
        with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None:
                self._prune_empty_locked(cutoff)
                if len(self._buckets) >= self._max_buckets:
                    raise ModelRoutingEnforcementError(
                        "rate_limit_state_capacity",
                        policy_id=policy_id,
                        route_id=route_id,
                    )
                bucket = deque()
                self._buckets[key] = bucket
            else:
                self._buckets.move_to_end(key)

            while bucket and bucket[0] <= cutoff:
                bucket.popleft()
            if len(bucket) >= limit:
                retry_after = max(
                    1,
                    math.ceil(MODEL_ROUTING_RATE_LIMIT_WINDOW_SECONDS - (now - bucket[0])),
                )
                raise ModelRoutingEnforcementError(
                    "rate_limit_exceeded",
                    policy_id=policy_id,
                    route_id=route_id,
                    retry_after_seconds=retry_after,
                )
            bucket.append(now)

    def _prune_empty_locked(self, cutoff: float) -> None:
        stale: list[tuple[str, str, str, str]] = []
        for key, bucket in self._buckets.items():
            while bucket and bucket[0] <= cutoff:
                bucket.popleft()
            if not bucket:
                stale.append(key)
        for key in stale:
            self._buckets.pop(key, None)

    def reset(self) -> None:
        with self._lock:
            self._buckets.clear()


def _parse_timestamp(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=UTC)


def _select_route(
    active: ActivatedModelRoutingPolicy,
    requested_model: str,
) -> ModelRoutingRoute | None:
    wildcard: ModelRoutingRoute | None = None
    for route in active.verified.claims.routes:
        if route.requested_model == requested_model:
            return route
        if route.requested_model == "*":
            wildcard = route
    return wildcard


def _estimate_max_cost_micros(
    *,
    route: ModelRoutingRoute,
    pricing: LoadedModelRoutingPricingCatalog,
    input_token_upper_bound: int,
    output_token_budget: int,
) -> int:
    total = 0
    for model in (route.primary_model, *route.fallback_models):
        price = pricing.by_model[model]
        input_product = input_token_upper_bound * price.input_cost_micros_per_million_tokens
        output_product = output_token_budget * price.output_cost_micros_per_million_tokens
        input_cost = (input_product + 999_999) // 1_000_000
        output_cost = (output_product + 999_999) // 1_000_000
        total += input_cost + output_cost
    return total


def enforce_model_routing_request(
    state: ModelRoutingRuntimeState,
    *,
    identity: Identity,
    requested_model: str,
    input_token_upper_bound: int | None,
    output_token_budget: int,
    rate_limiter: ModelRoutingRateLimiter,
    now: datetime | None = None,
    clock_skew_seconds: int = 0,
) -> ModelRoutingDecision | None:
    active = state.policy
    if active is None:
        return None

    claims = active.verified.claims
    if not requested_model or requested_model != requested_model.strip():
        raise ModelRoutingEnforcementError(
            "invalid_requested_model",
            policy_id=claims.policy_id,
        )
    if (
        output_token_budget < 0
        or (input_token_upper_bound is not None and input_token_upper_bound < 0)
        or clock_skew_seconds < 0
    ):
        raise ModelRoutingEnforcementError(
            "invalid_request_bounds",
            policy_id=claims.policy_id,
        )
    checked_at = now or datetime.now(UTC)
    if checked_at.tzinfo is None:
        raise ModelRoutingEnforcementError(
            "invalid_request_time",
            policy_id=claims.policy_id,
        )
    checked_at = checked_at.astimezone(UTC)
    skew = timedelta(seconds=clock_skew_seconds)
    if checked_at + skew < _parse_timestamp(claims.not_before):
        raise ModelRoutingEnforcementError(
            "policy_not_yet_valid",
            policy_id=claims.policy_id,
        )
    if checked_at - skew > _parse_timestamp(claims.expires_at):
        raise ModelRoutingEnforcementError("policy_expired", policy_id=claims.policy_id)
    if checked_at - skew > _parse_timestamp(claims.offline_lease_expires_at):
        raise ModelRoutingEnforcementError(
            "policy_offline_lease_expired",
            policy_id=claims.policy_id,
        )
    if identity.org_id is None:
        raise ModelRoutingEnforcementError(
            "org_identity_missing",
            policy_id=claims.policy_id,
        )
    if identity.org_id != claims.org_id:
        raise ModelRoutingEnforcementError(
            "org_identity_mismatch",
            policy_id=claims.policy_id,
        )

    route = _select_route(active, requested_model)
    if route is None:
        raise ModelRoutingEnforcementError(
            "route_not_allowed",
            policy_id=claims.policy_id,
        )

    limits = route.limits
    if limits.max_input_tokens is not None:
        if input_token_upper_bound is None:
            raise ModelRoutingEnforcementError(
                "input_token_estimate_unavailable",
                policy_id=claims.policy_id,
                route_id=route.route_id,
            )
        if input_token_upper_bound > limits.max_input_tokens:
            raise ModelRoutingEnforcementError(
                "input_token_limit_exceeded",
                policy_id=claims.policy_id,
                route_id=route.route_id,
            )
    if limits.max_output_tokens is not None and output_token_budget > limits.max_output_tokens:
        raise ModelRoutingEnforcementError(
            "output_token_limit_exceeded",
            policy_id=claims.policy_id,
            route_id=route.route_id,
        )

    estimated_cost: int | None = None
    if limits.max_cost_micros_per_request is not None:
        if input_token_upper_bound is None:
            raise ModelRoutingEnforcementError(
                "input_token_estimate_unavailable",
                policy_id=claims.policy_id,
                route_id=route.route_id,
            )
        if state.pricing is None:
            raise ModelRoutingEnforcementError(
                "pricing_catalog_unavailable",
                policy_id=claims.policy_id,
                route_id=route.route_id,
            )
        estimated_cost = _estimate_max_cost_micros(
            route=route,
            pricing=state.pricing,
            input_token_upper_bound=input_token_upper_bound,
            output_token_budget=output_token_budget,
        )
        if estimated_cost > limits.max_cost_micros_per_request:
            raise ModelRoutingEnforcementError(
                "cost_limit_exceeded",
                policy_id=claims.policy_id,
                route_id=route.route_id,
            )

    if limits.max_requests_per_minute is not None:
        rate_limiter.consume(
            digest=active.digest,
            route_id=route.route_id,
            org_id=claims.org_id,
            tenant=identity.tenant,
            limit=limits.max_requests_per_minute,
            policy_id=claims.policy_id,
        )

    return ModelRoutingDecision(
        active=active,
        route=route,
        requested_model=requested_model,
        candidate_models=(route.primary_model, *route.fallback_models),
        input_token_upper_bound=input_token_upper_bound,
        output_token_budget=output_token_budget,
        estimated_max_cost_micros=estimated_cost,
        pricing_digest=(state.pricing.digest if state.pricing is not None else None),
    )


def model_routing_span_attrs(
    decision: ModelRoutingDecision | None,
    *,
    candidate_model: str | None = None,
    candidate_index: int | None = None,
) -> dict:
    if decision is None:
        return {"model_routing.enforced": False}

    claims = decision.active.verified.claims
    limits = decision.route.limits
    attrs: dict = {
        "model_routing.enforced": True,
        "model_routing.policy.id": claims.policy_id,
        "model_routing.policy.revision": claims.revision,
        "model_routing.policy.digest": decision.active.digest,
        "model_routing.policy.release_id": claims.release_id,
        "model_routing.policy.deployment_id": claims.deployment_id,
        "model_routing.policy.org_id": claims.org_id,
        "model_routing.policy.environment": claims.target_environment,
        "model_routing.route.id": decision.route.route_id,
        "model_routing.route.requested_model": decision.requested_model,
        "model_routing.route.candidate_count": len(decision.candidate_models),
        "model_routing.output_token_budget": decision.output_token_budget,
        "model_routing.rate_limit.scope": "process-replica",
    }
    if decision.input_token_upper_bound is not None:
        attrs["model_routing.input_token_upper_bound"] = decision.input_token_upper_bound
    if decision.estimated_max_cost_micros is not None:
        attrs["model_routing.estimated_max_cost_micros"] = decision.estimated_max_cost_micros
    if decision.pricing_digest is not None:
        attrs["model_routing.pricing.digest"] = decision.pricing_digest
    if candidate_model is not None:
        attrs["model_routing.route.selected_model"] = candidate_model
    if candidate_index is not None:
        attrs["model_routing.route.candidate_index"] = candidate_index
    for key, value in (
        ("max_input_tokens", limits.max_input_tokens),
        ("max_output_tokens", limits.max_output_tokens),
        ("max_requests_per_minute", limits.max_requests_per_minute),
        ("max_cost_micros_per_request", limits.max_cost_micros_per_request),
    ):
        if value is not None:
            attrs[f"model_routing.limit.{key}"] = value
    return attrs
