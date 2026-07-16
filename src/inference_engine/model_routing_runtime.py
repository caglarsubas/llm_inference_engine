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
MODEL_ROUTING_RATE_LIMIT_WINDOW_SECONDS = 60.0
MODEL_ROUTING_RATE_LIMIT_WINDOW_MILLISECONDS = 60_000
MODEL_ROUTING_RATE_LIMIT_SCOPE_PROCESS = "process-replica"
MODEL_ROUTING_RATE_LIMIT_SCOPE_SHARED = "deployment-shared"
MODEL_ROUTING_RATE_LIMIT_REDIS_URL_MAX_BYTES = 4_096
MODEL_ROUTING_RATE_LIMIT_SENTINEL_CONFIG_VERSION = 1
MODEL_ROUTING_RATE_LIMIT_SENTINEL_CONFIG_MAX_BYTES = 65_536

ModelRoutingRateLimitScope = Literal["process-replica", "deployment-shared"]

_RATE_LIMIT_KEY_PREFIX = re.compile(r"^[A-Za-z0-9][A-Za-z0-9:_-]{0,63}$")
_RATE_LIMIT_SERVICE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_DNS_LABEL = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?$")
_SHARED_RATE_LIMIT_SCRIPT = """
local key = KEYS[1]
local limit = tonumber(ARGV[1])
local window_ms = tonumber(ARGV[2])
local nonce = ARGV[3]
local server_time = redis.call('TIME')
local now_ms = (tonumber(server_time[1]) * 1000) + math.floor(tonumber(server_time[2]) / 1000)
local cutoff_ms = now_ms - window_ms

redis.call('ZREMRANGEBYSCORE', key, '-inf', cutoff_ms)
local count = redis.call('ZCARD', key)
if count >= limit then
    local oldest = redis.call('ZRANGE', key, 0, 0, 'WITHSCORES')
    local retry_ms = window_ms
    if oldest[2] then
        retry_ms = math.max(1, tonumber(oldest[2]) + window_ms - now_ms)
    end
    redis.call('PEXPIRE', key, window_ms)
    return {0, retry_ms, count}
end

redis.call('ZADD', key, now_ms, tostring(now_ms) .. ':' .. nonce)
redis.call('PEXPIRE', key, window_ms)
return {1, 0, count + 1}
"""


class _RuntimeConfigModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        extra="forbid",
        frozen=True,
    )


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


class ModelRoutingRateLimiterProtocol(Protocol):
    scope: ModelRoutingRateLimitScope

    def consume(
        self,
        *,
        digest: str,
        route_id: str,
        org_id: str,
        tenant: str,
        limit: int,
        policy_id: str,
    ) -> None: ...

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

    scope: ModelRoutingRateLimitScope = MODEL_ROUTING_RATE_LIMIT_SCOPE_PROCESS

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
        auxiliary_clients: tuple[Redis, ...] = (),
    ) -> None:
        if not _RATE_LIMIT_KEY_PREFIX.fullmatch(key_prefix):
            raise ModelRoutingRuntimeConfigError("rate_limit_key_prefix_invalid")
        if required_replica_acks < 0:
            raise ValueError("required_replica_acks must not be negative")
        if required_replica_acks > 0 and replica_ack_timeout_milliseconds < 1:
            raise ValueError("replica_ack_timeout_milliseconds must be positive")
        self._client = client
        self._key_prefix = key_prefix
        self._required_replica_acks = required_replica_acks
        self._replica_ack_timeout_milliseconds = replica_ack_timeout_milliseconds
        self._auxiliary_clients = auxiliary_clients

    def _validate_replica_acknowledgements(self, acknowledged: object) -> None:
        if isinstance(acknowledged, bool) or not isinstance(acknowledged, int):
            raise ValueError("invalid replication acknowledgement response")
        if acknowledged < self._required_replica_acks:
            raise ValueError("insufficient replication acknowledgements")

    def _key(self, *, digest: str, route_id: str, org_id: str, tenant: str) -> str:
        canonical = json.dumps(
            [digest, route_id, org_id, tenant],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        identity_digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return f"{self._key_prefix}:rpm:{identity_digest}"

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
        limit: int,
        policy_id: str,
    ) -> None:
        key = self._key(
            digest=digest,
            route_id=route_id,
            org_id=org_id,
            tenant=tenant,
        )
        try:
            arguments = (
                _SHARED_RATE_LIMIT_SCRIPT,
                1,
                key,
                limit,
                MODEL_ROUTING_RATE_LIMIT_WINDOW_MILLISECONDS,
                secrets.token_hex(16),
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
            retry_after_milliseconds = int(result[1])
            if accepted == 1:
                if acknowledged is not None:
                    self._validate_replica_acknowledgements(acknowledged)
                return
            if accepted != 0:
                raise ValueError("invalid rate-limit decision")
            raise ModelRoutingEnforcementError(
                "rate_limit_exceeded",
                policy_id=policy_id,
                route_id=route_id,
                retry_after_seconds=max(1, math.ceil(retry_after_milliseconds / 1000)),
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
    return RedisModelRoutingRateLimiter(client, key_prefix=key_prefix)


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
        rate_limit_scope=rate_limiter.scope,
    )


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
