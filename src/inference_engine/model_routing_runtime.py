"""Request-time enforcement for a locally activated model-routing policy."""

from __future__ import annotations

import hashlib
import ipaddress
import json
import math
import re
import secrets
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import MappingProxyType
from typing import Literal, Mapping, Protocol
from urllib.parse import urlsplit

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    StrictBool,
    StrictInt,
    StrictStr,
    ValidationError,
    field_validator,
    model_validator,
)
from pydantic.alias_generators import to_camel
from redis import Redis
from redis.exceptions import RedisError
from redis.sentinel import Sentinel

from .auth import Identity
from .model_routing import (
    MAX_SAFE_INTEGER,
    ActivatedModelRoutingPolicy,
    ModelRoutingRoute,
    canonical_json,
)

MODEL_ROUTING_PRICING_VERSION = 1
MODEL_ROUTING_RATE_LIMIT_WINDOW_MILLISECONDS = 60_000
# Entries one window may hold at once. ``count`` windows are already bounded by
# their own limit; an ``amount`` window is not, because settlement revises
# entries downward and a spend limit is denominated in micros.
MODEL_ROUTING_RATE_LIMIT_MAX_WINDOW_ENTRIES = 100_000
# Expired entries one batch of the shared-scope sweep may reclaim. Redis runs a
# script to completion on its only thread, so this bound is every other tenant's
# worst admission latency per batch. Must stay well under Lua's 8000-argument
# ``unpack`` ceiling: the batch is spliced into ``HDEL`` and ``ZREM``.
MODEL_ROUTING_RATE_LIMIT_EXPIRY_SWEEP_LIMIT = 512
MODEL_ROUTING_RATE_LIMIT_SCOPE_PROCESS = "process-replica"
MODEL_ROUTING_RATE_LIMIT_SCOPE_SHARED = "deployment-shared"
MODEL_ROUTING_RATE_LIMIT_REDIS_URL_MAX_BYTES = 4_096
MODEL_ROUTING_RATE_LIMIT_SENTINEL_CONFIG_VERSION = 1
MODEL_ROUTING_RATE_LIMIT_SENTINEL_CONFIG_MAX_BYTES = 65_536

ModelRoutingRateLimitScope = Literal["process-replica", "deployment-shared"]

_RATE_LIMIT_KEY_PREFIX = re.compile(r"^[A-Za-z0-9][A-Za-z0-9:_-]{0,63}$")
_RATE_LIMIT_SERVICE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_DNS_LABEL = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?$")

MODEL_ROUTING_BUDGET_DIMENSION_REQUESTS = "rpm"
MODEL_ROUTING_BUDGET_DIMENSION_TOKENS = "tpm"
MODEL_ROUTING_BUDGET_DIMENSION_SPEND = "spend"

# ``count`` windows hold one member per admitted request and never settle;
# ``amount`` windows hold a per-request reservation whose value is revised once
# the real usage is known.
_BUDGET_KIND_COUNT = "count"
_BUDGET_KIND_AMOUNT = "amount"

# One eval admits every configured dimension or none of them: each dimension is
# checked against its own sliding window first, and only a run in which all of
# them fit reaches the commit loop. Two loops rather than one because a partial
# commit would charge a request that was ultimately denied. ``#total`` is the
# hash field carrying the running window total; reservation fields are 32-char
# hex nonces, so the ``#`` prefix cannot collide with one. The second reply
# element is a retry hint only on a limit denial: on an admission and on a
# capacity denial it carries the fullest ``amount`` window's entry count.
_SHARED_RATE_LIMIT_SCRIPT = """
local nonce = ARGV[1]
local dimensions = tonumber(ARGV[2])
local max_entries = tonumber(ARGV[3])
local sweep_limit = tonumber(ARGV[4])
local server_time = redis.call('TIME')
local now_ms = (tonumber(server_time[1]) * 1000) + math.floor(tonumber(server_time[2]) / 1000)

local key_at = 1
local arg_at = 5
local occupancy = 0
local admitted = {}
for index = 1, dimensions do
    local kind = ARGV[arg_at]
    local limit = tonumber(ARGV[arg_at + 1])
    local window_ms = tonumber(ARGV[arg_at + 2])
    local amount = tonumber(ARGV[arg_at + 3])
    local code = ARGV[arg_at + 4]
    arg_at = arg_at + 5
    local window_key = KEYS[key_at]
    key_at = key_at + 1
    local amount_key = nil
    if kind == 'amount' then
        amount_key = KEYS[key_at]
        key_at = key_at + 1
    end

    local cutoff_ms = now_ms - window_ms
    local used
    local entries = 0
    if kind == 'amount' then
        used = tonumber(redis.call('HGET', amount_key, '#total') or '0')
        -- '#total' still carries the amounts of expired members no batch has
        -- reached, so it only ever reads high: a total that already fits will
        -- fit once reclaimed too. Only a denial needs the window swept clean.
        while true do
            local expired = redis.call(
                'ZRANGEBYSCORE', window_key, '-inf', cutoff_ms, 'LIMIT', 0, sweep_limit
            )
            if #expired == 0 then
                break
            end
            local reserved = redis.call('HMGET', amount_key, unpack(expired))
            local freed = 0
            for position = 1, #reserved do
                if reserved[position] then
                    freed = freed + tonumber(reserved[position])
                end
            end
            redis.call('HDEL', amount_key, unpack(expired))
            redis.call('ZREM', window_key, unpack(expired))
            if freed ~= 0 then
                redis.call('HINCRBY', amount_key, '#total', -freed)
                used = used - freed
            end
            if used + amount <= limit then
                break
            end
        end
        -- An admission can break out with expired members still in the zset,
        -- and those are not live reservations.
        entries = redis.call('ZCARD', window_key)
            - redis.call('ZCOUNT', window_key, '-inf', cutoff_ms)
        if entries >= max_entries then
            redis.call('PEXPIRE', window_key, window_ms)
            redis.call('PEXPIRE', amount_key, window_ms)
            return {0, entries, 'rate_limit_state_capacity'}
        end
    else
        redis.call('ZREMRANGEBYSCORE', window_key, '-inf', cutoff_ms)
        used = redis.call('ZCARD', window_key)
    end

    if used + amount > limit then
        local oldest = redis.call(
            'ZRANGEBYSCORE', window_key, '(' .. cutoff_ms, '+inf', 'WITHSCORES', 'LIMIT', 0, 1
        )
        local retry_ms = window_ms
        if oldest[2] then
            retry_ms = math.max(1, tonumber(oldest[2]) + window_ms - now_ms)
        end
        redis.call('PEXPIRE', window_key, window_ms)
        if amount_key then
            redis.call('PEXPIRE', amount_key, window_ms)
        end
        return {0, retry_ms, code}
    end
    admitted[index] = {kind, window_key, amount_key, window_ms, amount, entries}
end

for index = 1, dimensions do
    local plan = admitted[index]
    local kind = plan[1]
    local window_key = plan[2]
    local amount_key = plan[3]
    local window_ms = plan[4]
    local amount = plan[5]
    if kind == 'amount' then
        redis.call('ZADD', window_key, now_ms, nonce)
        redis.call('HSET', amount_key, nonce, amount)
        redis.call('HINCRBY', amount_key, '#total', amount)
        redis.call('PEXPIRE', amount_key, window_ms)
        if plan[6] + 1 > occupancy then
            occupancy = plan[6] + 1
        end
    else
        redis.call('ZADD', window_key, now_ms, tostring(now_ms) .. ':' .. nonce)
    end
    redis.call('PEXPIRE', window_key, window_ms)
end
return {1, occupancy, ''}
"""

# Revises reservations this same nonce placed. A reservation the window has
# already evicted is left alone: its request ran in a window that has closed,
# and re-adding the spend now would charge it to a window it did not happen in.
_SHARED_SETTLE_SCRIPT = """
local nonce = ARGV[1]
local dimensions = tonumber(ARGV[2])
local key_at = 1
local arg_at = 3
for index = 1, dimensions do
    local window_key = KEYS[key_at]
    local amount_key = KEYS[key_at + 1]
    key_at = key_at + 2
    local actual = tonumber(ARGV[arg_at])
    local window_ms = tonumber(ARGV[arg_at + 1])
    arg_at = arg_at + 2

    local reserved = redis.call('HGET', amount_key, nonce)
    if reserved then
        reserved = tonumber(reserved)
        if actual <= 0 then
            redis.call('ZREM', window_key, nonce)
            redis.call('HDEL', amount_key, nonce)
            if reserved ~= 0 then
                redis.call('HINCRBY', amount_key, '#total', -reserved)
            end
        elseif actual ~= reserved then
            redis.call('HSET', amount_key, nonce, actual)
            redis.call('HINCRBY', amount_key, '#total', actual - reserved)
        end
        redis.call('PEXPIRE', window_key, window_ms)
        redis.call('PEXPIRE', amount_key, window_ms)
    end
end
return 1
"""


class _RuntimeConfigModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        extra="forbid",
        frozen=True,
    )


def _as_text(value: object) -> str:
    """Read a script reply element as text under ``decode_responses=False``."""
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return value if isinstance(value, str) else ""


def _valid_rate_limit_hostname(value: str) -> bool:
    if not value or value != value.strip() or len(value) > 253:
        return False
    candidate = value[:-1] if value.endswith(".") else value
    try:
        ipaddress.ip_address(candidate)
        return True
    except ValueError:
        return bool(candidate) and all(
            _DNS_LABEL.fullmatch(label) for label in candidate.split(".")
        )


class ModelRoutingRateLimitSentinelEndpoint(_RuntimeConfigModel):
    host: StrictStr
    port: StrictInt = Field(ge=1, le=65_535)

    @field_validator("host")
    @classmethod
    def validate_host(cls, value: str) -> str:
        if not _valid_rate_limit_hostname(value):
            raise ValueError("invalid Sentinel host")
        return value


class ModelRoutingRateLimitSentinelConfig(_RuntimeConfigModel):
    config_version: Literal[MODEL_ROUTING_RATE_LIMIT_SENTINEL_CONFIG_VERSION]
    service_name: StrictStr
    sentinels: tuple[ModelRoutingRateLimitSentinelEndpoint, ...] = Field(
        min_length=3,
        max_length=16,
    )
    min_other_sentinels: StrictInt = Field(default=1, ge=1, le=15)
    database: StrictInt = Field(default=0, ge=0, le=65_535)
    username: StrictStr | None = Field(default=None, min_length=1, max_length=256)
    password: SecretStr = Field(min_length=1, max_length=4_096)
    sentinel_username: StrictStr | None = Field(default=None, min_length=1, max_length=256)
    sentinel_password: SecretStr = Field(min_length=1, max_length=4_096)
    tls: StrictBool = True
    ca_file: StrictStr | None = Field(default=None, min_length=1, max_length=4_096)
    required_replica_acks: StrictInt = Field(default=1, ge=1, le=16)
    replica_ack_timeout_milliseconds: StrictInt = Field(default=500, ge=1, le=30_000)

    @field_validator("service_name")
    @classmethod
    def validate_service_name(cls, value: str) -> str:
        if not _RATE_LIMIT_SERVICE_NAME.fullmatch(value):
            raise ValueError("invalid Sentinel service name")
        return value

    @field_validator("username", "sentinel_username")
    @classmethod
    def validate_username(cls, value: str | None) -> str | None:
        if value is not None and (value != value.strip() or any(char.isspace() for char in value)):
            raise ValueError("invalid Redis username")
        return value

    @field_validator("ca_file")
    @classmethod
    def validate_ca_file(cls, value: str | None) -> str | None:
        if value is not None and value != value.strip():
            raise ValueError("invalid CA file")
        return value

    @model_validator(mode="after")
    def validate_topology(self) -> ModelRoutingRateLimitSentinelConfig:
        endpoints = {(endpoint.host.lower(), endpoint.port) for endpoint in self.sentinels}
        if len(endpoints) != len(self.sentinels):
            raise ValueError("duplicate Sentinel endpoint")
        if self.min_other_sentinels >= len(self.sentinels):
            raise ValueError("invalid Sentinel peer threshold")
        if not self.tls and self.ca_file is not None:
            raise ValueError("CA file requires TLS")
        return self


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
class ModelRoutingBudgetDimension:
    """One sliding window a request is admitted against."""

    name: str
    kind: str
    denial_code: str
    limit: int
    window_milliseconds: int
    amount: int


@dataclass
class ModelRoutingReservation:
    """Budget an admitted request holds until its real usage is known.

    Mutable and request-scoped. ``settled`` makes the commit idempotent, which
    is what lets every exit path — success, error, cancellation, and the
    backstop that runs after a streamed body finishes — call it unconditionally.
    """

    nonce: str
    policy_id: str
    digest: str
    route_id: str
    org_id: str
    tenant: str
    tokens: ModelRoutingBudgetDimension | None
    spend: ModelRoutingBudgetDimension | None
    observed_tokens: int | None = None
    observed_cost_micros: int | None = None
    # Set when the request consumed an unknown amount rather than none: it
    # keeps the admission reserve instead of releasing it.
    retained: bool = False
    settled: bool = False


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
    rate_limit_scope: ModelRoutingRateLimitScope
    reserved_tokens: int | None = None
    reserved_cost_micros: int | None = None
    reservation: ModelRoutingReservation | None = None


def build_model_routing_budget_dimensions(
    *,
    limit: int | None,
    tokens: int | None,
    max_tokens_per_minute: int | None,
    cost_micros: int | None,
    max_cost_micros_per_window: int | None,
    budget_window_seconds: int | None,
) -> tuple[ModelRoutingBudgetDimension, ...]:
    """Describe the windows one request must fit into, in enforcement order."""

    dimensions: list[ModelRoutingBudgetDimension] = []
    if limit is not None:
        dimensions.append(
            ModelRoutingBudgetDimension(
                name=MODEL_ROUTING_BUDGET_DIMENSION_REQUESTS,
                kind=_BUDGET_KIND_COUNT,
                denial_code="rate_limit_exceeded",
                limit=limit,
                window_milliseconds=MODEL_ROUTING_RATE_LIMIT_WINDOW_MILLISECONDS,
                amount=1,
            )
        )
    if max_tokens_per_minute is not None:
        dimensions.append(
            ModelRoutingBudgetDimension(
                name=MODEL_ROUTING_BUDGET_DIMENSION_TOKENS,
                kind=_BUDGET_KIND_AMOUNT,
                denial_code="token_rate_limit_exceeded",
                limit=max_tokens_per_minute,
                window_milliseconds=MODEL_ROUTING_RATE_LIMIT_WINDOW_MILLISECONDS,
                amount=max(0, tokens or 0),
            )
        )
    if max_cost_micros_per_window is not None and budget_window_seconds is not None:
        dimensions.append(
            ModelRoutingBudgetDimension(
                name=MODEL_ROUTING_BUDGET_DIMENSION_SPEND,
                kind=_BUDGET_KIND_AMOUNT,
                denial_code="budget_exceeded",
                limit=max_cost_micros_per_window,
                window_milliseconds=budget_window_seconds * 1_000,
                amount=max(0, cost_micros or 0),
            )
        )
    return tuple(dimensions)


def _reservation_for(
    dimensions: tuple[ModelRoutingBudgetDimension, ...],
    *,
    nonce: str,
    policy_id: str,
    digest: str,
    route_id: str,
    org_id: str,
    tenant: str,
) -> ModelRoutingReservation | None:
    by_name = {dimension.name: dimension for dimension in dimensions}
    tokens = by_name.get(MODEL_ROUTING_BUDGET_DIMENSION_TOKENS)
    spend = by_name.get(MODEL_ROUTING_BUDGET_DIMENSION_SPEND)
    if tokens is None and spend is None:
        return None
    return ModelRoutingReservation(
        nonce=nonce,
        policy_id=policy_id,
        digest=digest,
        route_id=route_id,
        org_id=org_id,
        tenant=tenant,
        tokens=tokens,
        spend=spend,
    )


def _settlement_amounts(
    reservation: ModelRoutingReservation,
) -> tuple[tuple[ModelRoutingBudgetDimension, int], ...]:
    """Pair each settleable dimension with the amount to commit.

    A request that reported no usage at all — denied downstream, no candidate
    available, the upstream raised — settles to zero. Holding its reservation
    instead would let a burst of failures deny real traffic for a whole budget
    window with nothing served behind it.

    Two cases keep their hold instead, because their consumption is real and
    unknown rather than absent: a retained reservation, which is a request that
    reached the model and then vanished without a usage report (an abandoned
    stream), and a served request whose model is absent from the pricing
    catalog, whose tokens are known but whose spend is not.
    """
    if reservation.retained:
        return tuple(
            (dimension, dimension.amount)
            for dimension in (reservation.tokens, reservation.spend)
            if dimension is not None
        )
    settlements: list[tuple[ModelRoutingBudgetDimension, int]] = []
    if reservation.tokens is not None:
        settlements.append((reservation.tokens, reservation.observed_tokens or 0))
    if reservation.spend is not None:
        if reservation.observed_cost_micros is not None:
            spend = reservation.observed_cost_micros
        elif reservation.observed_tokens is None:
            spend = 0
        else:
            spend = reservation.spend.amount
        settlements.append((reservation.spend, spend))
    return tuple(settlements)


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
        limit_requests: int | None = None,
        limit_tokens: int | None = None,
    ) -> None:
        self.code = code
        self.policy_id = policy_id
        self.route_id = route_id
        self.retry_after_seconds = retry_after_seconds
        # The per-window request ceiling that was hit, when the denial was a
        # rate limit. Feeds the ``x-ratelimit-limit-requests`` response header
        # that OpenAI SDKs read for backoff.
        self.limit_requests = limit_requests
        # Same, for the per-minute token ceiling and ``x-ratelimit-limit-tokens``.
        self.limit_tokens = limit_tokens
        super().__init__(code)


class ModelRoutingRateLimiterProtocol(Protocol):
    scope: ModelRoutingRateLimitScope

    def consume(
        self,
        *,
        digest: str,
        route_id: str,
        org_id: str,
        tenant: str,
        limit: int | None,
        policy_id: str,
        tokens: int | None = None,
        max_tokens_per_minute: int | None = None,
        cost_micros: int | None = None,
        max_cost_micros_per_window: int | None = None,
        budget_window_seconds: int | None = None,
    ) -> ModelRoutingReservation | None: ...

    def settle(self, reservation: ModelRoutingReservation) -> None: ...

    def metrics_snapshot(self) -> dict[str, int]: ...

    def ping(self) -> None: ...

    def close(self) -> None: ...


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
        if (
            route.limits.max_cost_micros_per_request is None
            and route.limits.max_cost_micros_per_window is None
        ):
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


class _RateLimitWindowMetrics:
    """Occupancy of the per-window entry ceiling, for both limiter scopes.

    ``window_entries_peak`` is a high-water mark and never decays: a scrape
    samples it at an interval the burst that filled a window need not survive.
    """

    __slots__ = ("_lock", "_denials", "_peak")

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._denials = 0
        self._peak = 0

    def observe_entries(self, entries: int) -> None:
        with self._lock:
            if entries > self._peak:
                self._peak = entries

    def record_denial(self, entries: int = 0) -> None:
        with self._lock:
            self._denials += 1
            if entries > self._peak:
                self._peak = entries

    def snapshot(self, max_window_entries: int) -> dict[str, int]:
        with self._lock:
            return {
                "state_capacity_denials_total": self._denials,
                "window_entries_peak": self._peak,
                "max_window_entries": max_window_entries,
            }


class _LocalWindow:
    """One process-local sliding window: ordered entries plus a running total.

    A settlement to zero drops the entry outright, the way the shared script's
    ``ZREM`` does, so a released reservation cannot go on dating the window or
    keeping it from being reclaimed. The deque cannot be seeked into cheaply,
    so it keeps a placeholder until the next prune or compaction removes it;
    ``amounts`` is the authority on which entries are live.
    """

    __slots__ = ("order", "amounts", "total", "window_seconds", "_released")

    def __init__(self, window_seconds: float) -> None:
        self.order: deque[tuple[float, str]] = deque()
        self.amounts: dict[str, int] = {}
        self.total = 0
        self.window_seconds = window_seconds
        self._released = 0

    def prune_at(self, now: float) -> None:
        self.prune(now - self.window_seconds)

    def prune(self, cutoff: float) -> None:
        while self.order:
            timestamp, nonce = self.order[0]
            if nonce not in self.amounts:
                self.order.popleft()
                self._released -= 1
                continue
            if timestamp > cutoff:
                return
            self.order.popleft()
            self.total -= self.amounts.pop(nonce, 0)

    def admit(self, now: float, nonce: str, amount: int) -> None:
        self.order.append((now, nonce))
        self.amounts[nonce] = amount
        self.total += amount

    def revise(self, nonce: str, amount: int) -> None:
        reserved = self.amounts.get(nonce)
        if reserved is None:
            return
        if amount > 0:
            self.amounts[nonce] = amount
            self.total += amount - reserved
            return
        del self.amounts[nonce]
        self.total -= reserved
        self._released += 1
        if self._released > len(self.amounts):
            self.order = deque(entry for entry in self.order if entry[1] in self.amounts)
            self._released = 0

    def oldest(self) -> float | None:
        return self.order[0][0] if self.order else None


class ModelRoutingRateLimiter:
    """Process-local sliding-window limiter keyed by policy, route, and tenant.

    **Per-replica, not a fleet budget.** Each replica keeps its own windows, so
    a deployment running N replicas admits up to N times every signed ceiling.
    Use ``deployment-shared`` scope when the numbers in the policy have to be
    the numbers the fleet actually honours.
    """

    scope: ModelRoutingRateLimitScope = MODEL_ROUTING_RATE_LIMIT_SCOPE_PROCESS

    def __init__(
        self,
        *,
        max_buckets: int = 10_000,
        max_window_entries: int = MODEL_ROUTING_RATE_LIMIT_MAX_WINDOW_ENTRIES,
        clock=time.monotonic,
    ) -> None:
        if max_buckets < 1:
            raise ValueError("max_buckets must be positive")
        if max_window_entries < 1:
            raise ValueError("max_window_entries must be positive")
        self._max_buckets = max_buckets
        self._max_window_entries = max_window_entries
        self._metrics = _RateLimitWindowMetrics()
        self._clock = clock
        self._lock = threading.Lock()
        self._buckets: OrderedDict[tuple[str, str, str, str, str], _LocalWindow] = OrderedDict()

    def consume(
        self,
        *,
        digest: str,
        route_id: str,
        org_id: str,
        tenant: str,
        limit: int | None,
        policy_id: str,
        tokens: int | None = None,
        max_tokens_per_minute: int | None = None,
        cost_micros: int | None = None,
        max_cost_micros_per_window: int | None = None,
        budget_window_seconds: int | None = None,
    ) -> ModelRoutingReservation | None:
        dimensions = build_model_routing_budget_dimensions(
            limit=limit,
            tokens=tokens,
            max_tokens_per_minute=max_tokens_per_minute,
            cost_micros=cost_micros,
            max_cost_micros_per_window=max_cost_micros_per_window,
            budget_window_seconds=budget_window_seconds,
        )
        if not dimensions:
            return None
        nonce = secrets.token_hex(16)
        now = self._clock()
        keys = [(dimension.name, digest, route_id, org_id, tenant) for dimension in dimensions]
        with self._lock:
            # Reclaiming before any window is created, never between them: a
            # sweep partway through the loop would collect the still-empty
            # windows this same admission had just installed.
            if any(key not in self._buckets for key in keys):
                self._prune_empty_locked(now)
                missing = sum(1 for key in keys if key not in self._buckets)
                if len(self._buckets) + missing > self._max_buckets:
                    self._metrics.record_denial()
                    raise ModelRoutingEnforcementError(
                        "rate_limit_state_capacity",
                        policy_id=policy_id,
                        route_id=route_id,
                    )

            # Denial precedence is the one ``_SHARED_RATE_LIMIT_SCRIPT``
            # applies: dimensions in enforcement order, and within a dimension
            # the entry ceiling ahead of the limit.
            windows: list[tuple[ModelRoutingBudgetDimension, _LocalWindow]] = []
            for dimension, key in zip(dimensions, keys):
                window = self._buckets.get(key)
                if window is None:
                    window = _LocalWindow(dimension.window_milliseconds / 1000)
                    self._buckets[key] = window
                else:
                    self._buckets.move_to_end(key)
                window.prune_at(now)
                if dimension.kind == _BUDGET_KIND_AMOUNT:
                    entries = len(window.amounts)
                    if entries >= self._max_window_entries:
                        self._metrics.record_denial(entries)
                        raise ModelRoutingEnforcementError(
                            "rate_limit_state_capacity",
                            policy_id=policy_id,
                            route_id=route_id,
                        )
                if window.total + dimension.amount > dimension.limit:
                    oldest = window.oldest()
                    window_seconds = dimension.window_milliseconds / 1000
                    retry_after = max(
                        1,
                        math.ceil(
                            window_seconds if oldest is None else window_seconds - (now - oldest)
                        ),
                    )
                    raise ModelRoutingEnforcementError(
                        dimension.denial_code,
                        policy_id=policy_id,
                        route_id=route_id,
                        retry_after_seconds=retry_after,
                        limit_requests=(
                            dimension.limit
                            if dimension.name == MODEL_ROUTING_BUDGET_DIMENSION_REQUESTS
                            else None
                        ),
                        limit_tokens=(
                            dimension.limit
                            if dimension.name == MODEL_ROUTING_BUDGET_DIMENSION_TOKENS
                            else None
                        ),
                    )
                windows.append((dimension, window))

            for dimension, window in windows:
                window.admit(now, nonce, dimension.amount)
                if dimension.kind == _BUDGET_KIND_AMOUNT:
                    self._metrics.observe_entries(len(window.amounts))

        return _reservation_for(
            dimensions,
            nonce=nonce,
            policy_id=policy_id,
            digest=digest,
            route_id=route_id,
            org_id=org_id,
            tenant=tenant,
        )

    def settle(self, reservation: ModelRoutingReservation) -> None:
        settlements = _settlement_amounts(reservation)
        if not settlements:
            return
        now = self._clock()
        with self._lock:
            for dimension, amount in settlements:
                key = (
                    dimension.name,
                    reservation.digest,
                    reservation.route_id,
                    reservation.org_id,
                    reservation.tenant,
                )
                window = self._buckets.get(key)
                if window is None:
                    continue
                window.prune_at(now)
                window.revise(reservation.nonce, amount)

    def metrics_snapshot(self) -> dict[str, int]:
        return self._metrics.snapshot(self._max_window_entries)

    def _prune_empty_locked(self, now: float) -> None:
        stale: list[tuple[str, str, str, str, str]] = []
        for key, window in self._buckets.items():
            window.prune_at(now)
            if not window.amounts:
                stale.append(key)
        for key in stale:
            self._buckets.pop(key, None)

    def reset(self) -> None:
        with self._lock:
            self._buckets.clear()

    def ping(self) -> None:
        return None

    def close(self) -> None:
        return None


class RedisModelRoutingRateLimiter:
    """Deployment-wide sliding window using one atomic Redis-protocol script."""

    scope: ModelRoutingRateLimitScope = MODEL_ROUTING_RATE_LIMIT_SCOPE_SHARED

    def __init__(
        self,
        client: Redis,
        *,
        key_prefix: str,
        required_replica_acks: int = 0,
        replica_ack_timeout_milliseconds: int = 0,
        max_window_entries: int = MODEL_ROUTING_RATE_LIMIT_MAX_WINDOW_ENTRIES,
        auxiliary_clients: tuple[Redis, ...] = (),
    ) -> None:
        if not _RATE_LIMIT_KEY_PREFIX.fullmatch(key_prefix):
            raise ModelRoutingRuntimeConfigError("rate_limit_key_prefix_invalid")
        if required_replica_acks < 0:
            raise ValueError("required_replica_acks must not be negative")
        if required_replica_acks > 0 and replica_ack_timeout_milliseconds < 1:
            raise ValueError("replica_ack_timeout_milliseconds must be positive")
        if max_window_entries < 1:
            raise ValueError("max_window_entries must be positive")
        self._client = client
        self._key_prefix = key_prefix
        self._max_window_entries = max_window_entries
        self._metrics = _RateLimitWindowMetrics()
        self._required_replica_acks = required_replica_acks
        self._replica_ack_timeout_milliseconds = replica_ack_timeout_milliseconds
        self._auxiliary_clients = auxiliary_clients

    def _validate_replica_acknowledgements(self, acknowledged: object) -> None:
        if isinstance(acknowledged, bool) or not isinstance(acknowledged, int):
            raise ValueError("invalid replication acknowledgement response")
        if acknowledged < self._required_replica_acks:
            raise ValueError("insufficient replication acknowledgements")

    def metrics_snapshot(self) -> dict[str, int]:
        return self._metrics.snapshot(self._max_window_entries)

    def _identity_digest(self, *, digest: str, route_id: str, org_id: str, tenant: str) -> str:
        canonical = json.dumps(
            [digest, route_id, org_id, tenant],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _dimension_keys(
        self,
        dimension: ModelRoutingBudgetDimension,
        identity_digest: str,
    ) -> tuple[str, ...]:
        window_key = f"{self._key_prefix}:{dimension.name}:{identity_digest}"
        if dimension.kind == _BUDGET_KIND_COUNT:
            return (window_key,)
        return (window_key, f"{window_key}:amounts")

    def ping(self) -> None:
        try:
            if self._client.ping() is not True:
                raise ModelRoutingRuntimeConfigError("rate_limit_backend_unavailable")
        except ModelRoutingRuntimeConfigError:
            raise
        except (RedisError, OSError, ValueError, TypeError) as exc:
            raise ModelRoutingRuntimeConfigError("rate_limit_backend_unavailable") from exc

    def consume(
        self,
        *,
        digest: str,
        route_id: str,
        org_id: str,
        tenant: str,
        limit: int | None,
        policy_id: str,
        tokens: int | None = None,
        max_tokens_per_minute: int | None = None,
        cost_micros: int | None = None,
        max_cost_micros_per_window: int | None = None,
        budget_window_seconds: int | None = None,
    ) -> ModelRoutingReservation | None:
        dimensions = build_model_routing_budget_dimensions(
            limit=limit,
            tokens=tokens,
            max_tokens_per_minute=max_tokens_per_minute,
            cost_micros=cost_micros,
            max_cost_micros_per_window=max_cost_micros_per_window,
            budget_window_seconds=budget_window_seconds,
        )
        if not dimensions:
            return None
        identity_digest = self._identity_digest(
            digest=digest,
            route_id=route_id,
            org_id=org_id,
            tenant=tenant,
        )
        nonce = secrets.token_hex(16)
        keys: list[str] = []
        script_args: list[object] = [
            nonce,
            len(dimensions),
            self._max_window_entries,
            MODEL_ROUTING_RATE_LIMIT_EXPIRY_SWEEP_LIMIT,
        ]
        for dimension in dimensions:
            keys.extend(self._dimension_keys(dimension, identity_digest))
            script_args.extend(
                (
                    dimension.kind,
                    dimension.limit,
                    dimension.window_milliseconds,
                    dimension.amount,
                    dimension.denial_code,
                )
            )
        by_code = {dimension.denial_code: dimension for dimension in dimensions}
        try:
            arguments = (
                _SHARED_RATE_LIMIT_SCRIPT,
                len(keys),
                *keys,
                *script_args,
            )
            acknowledged: object | None = None
            if self._required_replica_acks > 0:
                pipeline = self._client.pipeline(transaction=False)
                pipeline.eval(*arguments)
                pipeline.wait(
                    self._required_replica_acks,
                    self._replica_ack_timeout_milliseconds,
                )
                pipeline_result = pipeline.execute()
                if not isinstance(pipeline_result, (list, tuple)) or len(pipeline_result) != 2:
                    raise ValueError("invalid replicated rate-limit response")
                result, acknowledged = pipeline_result
            else:
                result = self._client.eval(*arguments)
            if not isinstance(result, (list, tuple)) or len(result) != 3:
                raise ValueError("invalid rate-limit response")
            accepted = int(result[0])
            if accepted == 1:
                if acknowledged is not None:
                    self._validate_replica_acknowledgements(acknowledged)
                self._metrics.observe_entries(int(result[1]))
                return _reservation_for(
                    dimensions,
                    nonce=nonce,
                    policy_id=policy_id,
                    digest=digest,
                    route_id=route_id,
                    org_id=org_id,
                    tenant=tenant,
                )
            if accepted != 0:
                raise ValueError("invalid rate-limit decision")
            code = _as_text(result[2])
            if code == "rate_limit_state_capacity":
                self._metrics.record_denial(int(result[1]))
                raise ModelRoutingEnforcementError(
                    "rate_limit_state_capacity",
                    policy_id=policy_id,
                    route_id=route_id,
                )
            denied = by_code.get(code)
            if denied is None:
                raise ValueError("invalid rate-limit dimension")
            raise ModelRoutingEnforcementError(
                denied.denial_code,
                policy_id=policy_id,
                route_id=route_id,
                retry_after_seconds=max(1, math.ceil(int(result[1]) / 1000)),
                limit_requests=(
                    denied.limit if denied.name == MODEL_ROUTING_BUDGET_DIMENSION_REQUESTS else None
                ),
                limit_tokens=(
                    denied.limit if denied.name == MODEL_ROUTING_BUDGET_DIMENSION_TOKENS else None
                ),
            )
        except ModelRoutingEnforcementError:
            raise
        except (RedisError, OSError, ValueError, TypeError) as exc:
            raise ModelRoutingEnforcementError(
                "rate_limit_backend_unavailable",
                policy_id=policy_id,
                route_id=route_id,
                retry_after_seconds=1,
            ) from exc

    def settle(self, reservation: ModelRoutingReservation) -> None:
        settlements = _settlement_amounts(reservation)
        if not settlements:
            return
        identity_digest = self._identity_digest(
            digest=reservation.digest,
            route_id=reservation.route_id,
            org_id=reservation.org_id,
            tenant=reservation.tenant,
        )
        keys: list[str] = []
        script_args: list[object] = [reservation.nonce, len(settlements)]
        for dimension, amount in settlements:
            keys.extend(self._dimension_keys(dimension, identity_digest))
            script_args.extend((amount, dimension.window_milliseconds))
        self._client.eval(_SHARED_SETTLE_SCRIPT, len(keys), *keys, *script_args)

    def close(self) -> None:
        for client in (self._client, *self._auxiliary_clients):
            try:
                client.close()
            except (RedisError, OSError):
                continue


def _is_loopback(hostname: str) -> bool:
    if hostname.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def _validate_rate_limit_redis_url(value: str, *, allow_insecure: bool) -> str:
    if not value or any(character.isspace() for character in value):
        raise ModelRoutingRuntimeConfigError("rate_limit_backend_url_invalid")
    try:
        parsed = urlsplit(value)
        hostname = parsed.hostname or ""
        parsed.port
    except ValueError as exc:
        raise ModelRoutingRuntimeConfigError("rate_limit_backend_url_invalid") from exc
    if not hostname or parsed.fragment or parsed.scheme not in {"redis", "rediss"}:
        raise ModelRoutingRuntimeConfigError("rate_limit_backend_url_invalid")
    if parsed.scheme == "redis" and not (_is_loopback(hostname) or allow_insecure):
        raise ModelRoutingRuntimeConfigError("rate_limit_backend_tls_required")
    return value


def _read_rate_limit_redis_url(path: Path) -> str:
    try:
        with path.open("rb") as stream:
            raw = stream.read(MODEL_ROUTING_RATE_LIMIT_REDIS_URL_MAX_BYTES + 1)
        if not raw or len(raw) > MODEL_ROUTING_RATE_LIMIT_REDIS_URL_MAX_BYTES:
            raise ModelRoutingRuntimeConfigError("rate_limit_backend_url_invalid")
        return raw.decode("utf-8").strip()
    except ModelRoutingRuntimeConfigError:
        raise
    except (OSError, UnicodeError) as exc:
        raise ModelRoutingRuntimeConfigError("rate_limit_backend_url_unavailable") from exc


def _read_rate_limit_sentinel_config(path: Path) -> ModelRoutingRateLimitSentinelConfig:
    try:
        with path.open("rb") as stream:
            raw = stream.read(MODEL_ROUTING_RATE_LIMIT_SENTINEL_CONFIG_MAX_BYTES + 1)
        if not raw or len(raw) > MODEL_ROUTING_RATE_LIMIT_SENTINEL_CONFIG_MAX_BYTES:
            raise ModelRoutingRuntimeConfigError("rate_limit_sentinel_config_invalid")
        return ModelRoutingRateLimitSentinelConfig.model_validate_json(raw)
    except ModelRoutingRuntimeConfigError:
        raise
    except (ValidationError, UnicodeError, ValueError) as exc:
        raise ModelRoutingRuntimeConfigError("rate_limit_sentinel_config_invalid") from exc
    except OSError as exc:
        raise ModelRoutingRuntimeConfigError("rate_limit_sentinel_config_unavailable") from exc


def _build_sentinel_rate_limit_client(
    config: ModelRoutingRateLimitSentinelConfig,
    *,
    allow_insecure: bool,
    connect_timeout_seconds: float,
    operation_timeout_seconds: float,
) -> tuple[Redis, tuple[Redis, ...]]:
    if not config.tls and not allow_insecure:
        if any(not _is_loopback(endpoint.host) for endpoint in config.sentinels):
            raise ModelRoutingRuntimeConfigError("rate_limit_backend_tls_required")
    if config.replica_ack_timeout_milliseconds >= operation_timeout_seconds * 1_000:
        raise ModelRoutingRuntimeConfigError("rate_limit_sentinel_config_invalid")

    common: dict[str, object] = {
        "decode_responses": False,
        "health_check_interval": 30,
        "retry_on_timeout": False,
        "socket_connect_timeout": connect_timeout_seconds,
        "socket_keepalive": True,
        "socket_timeout": operation_timeout_seconds,
    }
    if config.tls:
        common.update(
            {
                "ssl": True,
                "ssl_cert_reqs": "required",
                "ssl_check_hostname": True,
            }
        )
        if config.ca_file is not None:
            common["ssl_ca_certs"] = config.ca_file

    data_connection = dict(common)
    data_connection["db"] = config.database
    data_connection["password"] = config.password.get_secret_value()
    if config.username is not None:
        data_connection["username"] = config.username

    sentinel_connection = dict(common)
    sentinel_connection["password"] = config.sentinel_password.get_secret_value()
    if config.sentinel_username is not None:
        sentinel_connection["username"] = config.sentinel_username

    try:
        manager = Sentinel(
            [(endpoint.host, endpoint.port) for endpoint in config.sentinels],
            min_other_sentinels=config.min_other_sentinels,
            sentinel_kwargs=sentinel_connection,
            **data_connection,
        )
        client = manager.master_for(config.service_name, check_connection=True)
    except (RedisError, OSError, ValueError, TypeError) as exc:
        raise ModelRoutingRuntimeConfigError("rate_limit_sentinel_config_invalid") from exc
    return client, tuple(manager.sentinels)


def build_model_routing_rate_limiter(settings) -> ModelRoutingRateLimiterProtocol:
    """Build a strict local or tenant-owned shared limiter without connecting yet."""

    scope = settings.model_routing_rate_limit_scope
    direct_url = settings.model_routing_rate_limit_redis_url.strip()
    url_file_value = settings.model_routing_rate_limit_redis_url_file.strip()
    sentinel_file_value = settings.model_routing_rate_limit_sentinel_config_file.strip()

    if scope == MODEL_ROUTING_RATE_LIMIT_SCOPE_PROCESS:
        if direct_url or url_file_value or sentinel_file_value:
            raise ModelRoutingRuntimeConfigError("rate_limit_backend_config_unused")
        return ModelRoutingRateLimiter(
            max_buckets=settings.model_routing_rate_limit_max_buckets,
            max_window_entries=settings.model_routing_rate_limit_max_window_entries,
        )
    if scope != MODEL_ROUTING_RATE_LIMIT_SCOPE_SHARED:
        raise ModelRoutingRuntimeConfigError("rate_limit_scope_invalid")
    if sum(bool(value) for value in (direct_url, url_file_value, sentinel_file_value)) != 1:
        raise ModelRoutingRuntimeConfigError("rate_limit_backend_source_invalid")

    key_prefix = settings.model_routing_rate_limit_key_prefix.strip()
    if not _RATE_LIMIT_KEY_PREFIX.fullmatch(key_prefix):
        raise ModelRoutingRuntimeConfigError("rate_limit_key_prefix_invalid")

    if sentinel_file_value:
        sentinel_config = _read_rate_limit_sentinel_config(Path(sentinel_file_value).expanduser())
        client, sentinel_clients = _build_sentinel_rate_limit_client(
            sentinel_config,
            allow_insecure=settings.model_routing_rate_limit_allow_insecure_redis,
            connect_timeout_seconds=(settings.model_routing_rate_limit_connect_timeout_seconds),
            operation_timeout_seconds=(settings.model_routing_rate_limit_operation_timeout_seconds),
        )
        return RedisModelRoutingRateLimiter(
            client,
            key_prefix=key_prefix,
            required_replica_acks=sentinel_config.required_replica_acks,
            replica_ack_timeout_milliseconds=(sentinel_config.replica_ack_timeout_milliseconds),
            max_window_entries=settings.model_routing_rate_limit_max_window_entries,
            auxiliary_clients=sentinel_clients,
        )

    redis_url = (
        _read_rate_limit_redis_url(Path(url_file_value).expanduser())
        if url_file_value
        else direct_url
    )
    redis_url = _validate_rate_limit_redis_url(
        redis_url,
        allow_insecure=settings.model_routing_rate_limit_allow_insecure_redis,
    )
    try:
        client = Redis.from_url(
            redis_url,
            decode_responses=False,
            health_check_interval=30,
            retry_on_timeout=False,
            socket_connect_timeout=settings.model_routing_rate_limit_connect_timeout_seconds,
            socket_keepalive=True,
            socket_timeout=settings.model_routing_rate_limit_operation_timeout_seconds,
        )
    except (RedisError, OSError, ValueError, TypeError) as exc:
        raise ModelRoutingRuntimeConfigError("rate_limit_backend_url_invalid") from exc
    return RedisModelRoutingRateLimiter(
        client,
        key_prefix=key_prefix,
        max_window_entries=settings.model_routing_rate_limit_max_window_entries,
    )


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


def usage_cost_micros(
    pricing: LoadedModelRoutingPricingCatalog | None,
    *,
    model: str | None,
    input_tokens: int,
    output_tokens: int,
) -> int | None:
    """Charge for tokens actually served, or ``None`` when the model is unpriced.

    ``None`` rather than 0: a model absent from the catalog means "we do not
    know what this costs", and reporting it as free would silently under-bill.
    The per-component ceiling matches ``_estimate_max_cost_micros`` so the
    ledger and the pre-flight cost limit never disagree about the same call.
    """
    if pricing is None or model is None:
        return None
    price = pricing.by_model.get(model)
    if price is None:
        return None
    input_product = input_tokens * price.input_cost_micros_per_million_tokens
    output_product = output_tokens * price.output_cost_micros_per_million_tokens
    return (input_product + 999_999) // 1_000_000 + (output_product + 999_999) // 1_000_000


def enforce_model_routing_request(
    state: ModelRoutingRuntimeState,
    *,
    identity: Identity,
    requested_model: str,
    input_token_upper_bound: int | None,
    output_token_budget: int,
    rate_limiter: ModelRoutingRateLimiterProtocol,
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
    if (
        limits.max_cost_micros_per_request is not None
        or limits.max_cost_micros_per_window is not None
    ):
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
        if (
            limits.max_cost_micros_per_request is not None
            and estimated_cost > limits.max_cost_micros_per_request
        ):
            raise ModelRoutingEnforcementError(
                "cost_limit_exceeded",
                policy_id=claims.policy_id,
                route_id=route.route_id,
            )

    reserved_tokens: int | None = None
    if limits.max_tokens_per_minute is not None:
        if input_token_upper_bound is None:
            raise ModelRoutingEnforcementError(
                "input_token_estimate_unavailable",
                policy_id=claims.policy_id,
                route_id=route.route_id,
            )
        reserved_tokens = input_token_upper_bound + output_token_budget

    reserved_cost = estimated_cost if limits.max_cost_micros_per_window is not None else None

    reservation = rate_limiter.consume(
        digest=active.digest,
        route_id=route.route_id,
        org_id=claims.org_id,
        tenant=identity.tenant,
        limit=limits.max_requests_per_minute,
        policy_id=claims.policy_id,
        tokens=reserved_tokens,
        max_tokens_per_minute=limits.max_tokens_per_minute,
        cost_micros=reserved_cost,
        max_cost_micros_per_window=limits.max_cost_micros_per_window,
        budget_window_seconds=limits.budget_window_seconds,
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
        rate_limit_scope=rate_limiter.scope,
        reserved_tokens=reserved_tokens,
        reserved_cost_micros=reserved_cost,
        reservation=reservation,
    )


def observe_model_routing_usage(
    reservation: ModelRoutingReservation | None,
    *,
    tokens: int,
    cost_micros: int | None,
) -> None:
    """Record what the request really consumed, for the settle that follows.

    Replaces rather than accumulates: every call site reports the request
    total, including the structured-output retry that bills both attempts.
    """
    if reservation is None or reservation.settled:
        return
    reservation.observed_tokens = max(0, tokens)
    if cost_micros is not None:
        reservation.observed_cost_micros = max(0, cost_micros)


def retain_model_routing_reservation(
    reservation: ModelRoutingReservation | None,
) -> None:
    """Charge the admission reserve for a request whose usage never came back.

    An abandoned stream reaches its teardown with no token counts, because
    adapters report them on a terminal frame the client is no longer there to
    receive. That is not the same as a request that consumed nothing, and
    settling it to zero would make every window bypassable by disconnecting.
    """
    if reservation is None or reservation.settled:
        return
    reservation.retained = True


def settle_model_routing_reservation(
    rate_limiter: ModelRoutingRateLimiterProtocol,
    reservation: ModelRoutingReservation | None,
) -> None:
    """Commit the observed usage over the admission reservation. Idempotent.

    ``settled`` is set before the store is touched, so a failing settle cannot
    be retried into a double commit; the reservation then stands at its
    conservative admission value until its window slides past it.
    """
    if reservation is None or reservation.settled:
        return
    reservation.settled = True
    rate_limiter.settle(reservation)


def model_routing_policy_identity_attrs(
    active: ActivatedModelRoutingPolicy,
) -> dict:
    """Return legacy and canonical payload-free routing-policy identity."""

    claims = active.verified.claims
    return {
        "model_routing.policy.id": claims.policy_id,
        "model_routing.policy.revision": claims.revision,
        "model_routing.policy.digest": active.digest,
        "model_routing.policy.release_id": claims.release_id,
        "model_routing.policy.deployment_id": claims.deployment_id,
        "model_routing.policy.org_id": claims.org_id,
        "model_routing.policy.environment": claims.target_environment,
        "prometa.artifact.type": "model-routing-policy",
        "prometa.artifact.digest": active.digest,
        "prometa.policy.digest": active.digest,
        "prometa.release.id": claims.release_id,
        "prometa.deployment.id": claims.deployment_id,
        "prometa.environment": claims.target_environment,
    }


def model_routing_span_attrs(
    decision: ModelRoutingDecision | None,
    *,
    candidate_model: str | None = None,
    candidate_index: int | None = None,
) -> dict:
    if decision is None:
        return {"model_routing.enforced": False}

    limits = decision.route.limits
    attrs: dict = {
        "model_routing.enforced": True,
        **model_routing_policy_identity_attrs(decision.active),
        "model_routing.route.id": decision.route.route_id,
        "model_routing.route.requested_model": decision.requested_model,
        "model_routing.route.candidate_count": len(decision.candidate_models),
        "model_routing.output_token_budget": decision.output_token_budget,
        "model_routing.rate_limit.scope": decision.rate_limit_scope,
    }
    if decision.input_token_upper_bound is not None:
        attrs["model_routing.input_token_upper_bound"] = decision.input_token_upper_bound
    if decision.estimated_max_cost_micros is not None:
        attrs["model_routing.estimated_max_cost_micros"] = decision.estimated_max_cost_micros
    if decision.reserved_tokens is not None:
        attrs["model_routing.reserved_tokens"] = decision.reserved_tokens
    if decision.reserved_cost_micros is not None:
        attrs["model_routing.reserved_cost_micros"] = decision.reserved_cost_micros
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
        ("max_tokens_per_minute", limits.max_tokens_per_minute),
        ("max_cost_micros_per_window", limits.max_cost_micros_per_window),
        ("budget_window_seconds", limits.budget_window_seconds),
    ):
        if value is not None:
            attrs[f"model_routing.limit.{key}"] = value
    return attrs
