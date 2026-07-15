"""Asynchronous, payload-free model-plane observation reporting.

The reporter is deliberately outside the inference request path. It reports
locally observed health, model inventory digests, and signed routing-policy
identity to an Orchestra control plane without making that plane a runtime
dependency.
"""

from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import json
import random
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Never, Protocol
from urllib.parse import urlsplit
from uuid import uuid4

import httpx
from pydantic import BaseModel

from . import __version__
from .model_routing_runtime import (
    ModelRoutingRateLimiterProtocol,
    ModelRoutingRuntimeState,
)
from .model_routing_status import build_model_routing_status
from .observability import get_logger, span
from .schemas import ModelList

MODEL_PLANE_OBSERVATION_TYPE = "orchestra.model-plane-observation"
MODEL_PLANE_OBSERVATION_VERSION_V1 = 1
MODEL_PLANE_OBSERVATION_VERSION_V2 = 2
_OBSERVATION_PATH = "/api/model-routing-observations"
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@+-]{0,199}$")
_TARGET_ENVIRONMENTS = frozenset({"dev", "test", "staging", "prod"})
_MAX_API_KEY_BYTES = 4_096
_RETRYABLE_STATUS_CODES = frozenset({401, 403, 408, 425, 429})

log = get_logger("model_plane.observer")


class ModelPlaneObservationConfigError(ValueError):
    """An enabled reporter is incomplete or unsafe to run."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class _ObservationState(Protocol):
    model_routing_runtime: ModelRoutingRuntimeState
    model_routing_rate_limiter: ModelRoutingRateLimiterProtocol

    def readiness(self) -> dict: ...


@dataclass(frozen=True)
class ModelPlaneObservationConfig:
    endpoint: str
    api_key: str
    api_key_file: Path | None
    deployment_id: str
    target_environment: str
    engine_instance_id: str
    auth_enabled: bool
    observation_version: Literal[1, 2]
    interval_seconds: float
    timeout_seconds: float
    jitter_ratio: float


class ModelPlaneObserverStatus(BaseModel):
    object: Literal["model_plane_observer.status"] = "model_plane_observer.status"
    enabled: bool
    running: bool = False
    attempts_total: int = 0
    successes_total: int = 0
    failures_total: int = 0
    consecutive_failures: int = 0
    pending_observation_id: str | None = None
    last_attempt_at: str | None = None
    last_success_at: str | None = None
    last_error_code: str | None = None


def _reject(code: str, message: str) -> Never:
    raise ModelPlaneObservationConfigError(code, message)


def _validate_identifier(value: str, field: str) -> str:
    if not _IDENTIFIER.fullmatch(value):
        _reject("invalid_identifier", f"{field} must be a bounded Orchestra identifier")
    return value


def _is_loopback(hostname: str) -> bool:
    if hostname.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def _validate_endpoint(value: str) -> str:
    try:
        parsed = urlsplit(value)
        hostname = parsed.hostname or ""
        parsed.port
    except ValueError:
        return _reject("invalid_endpoint", "Observation endpoint is not a valid URL")
    if (
        not hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path != _OBSERVATION_PATH
    ):
        return _reject(
            "invalid_endpoint",
            f"Observation endpoint must be the exact {_OBSERVATION_PATH} URL",
        )
    if parsed.scheme != "https" and not (parsed.scheme == "http" and _is_loopback(hostname)):
        return _reject(
            "insecure_endpoint",
            "Observation endpoint must use HTTPS except on loopback",
        )
    return value


def _validate_api_key(value: str) -> str:
    key = value.strip()
    try:
        encoded = key.encode("ascii")
    except UnicodeEncodeError:
        return _reject("invalid_api_key", "Observation API key must be an ASCII token")
    if not key or len(encoded) > _MAX_API_KEY_BYTES or any(char.isspace() for char in key):
        _reject("invalid_api_key", "Observation API key is empty, oversized, or malformed")
    return key


def _read_api_key_file(path: Path) -> str:
    try:
        with path.open("rb") as stream:
            raw = stream.read(_MAX_API_KEY_BYTES + 1)
        if len(raw) > _MAX_API_KEY_BYTES:
            _reject("invalid_api_key", "Observation API key file is oversized")
        return _validate_api_key(raw.decode("utf-8"))
    except ModelPlaneObservationConfigError:
        raise
    except (OSError, UnicodeError) as exc:
        raise ModelPlaneObservationConfigError(
            "api_key_unavailable",
            "Observation API key file cannot be read",
        ) from exc


def load_model_plane_observation_config(settings) -> ModelPlaneObservationConfig | None:
    """Validate enabled settings and prove the initial credential is readable."""
    if not settings.model_plane_observation_enabled:
        return None

    endpoint = _validate_endpoint(settings.model_plane_observation_endpoint.strip())
    deployment_id = _validate_identifier(
        settings.model_plane_observation_deployment_id.strip(),
        "deployment id",
    )
    target_environment = settings.model_plane_observation_target_environment.strip()
    if target_environment not in _TARGET_ENVIRONMENTS:
        _reject(
            "invalid_environment",
            "Observation target environment must be dev, test, staging, or prod",
        )
    engine_instance_id = _validate_identifier(
        settings.model_plane_observation_engine_instance_id.strip(),
        "engine instance id",
    )
    observation_version = settings.model_plane_observation_version
    if observation_version not in (
        MODEL_PLANE_OBSERVATION_VERSION_V1,
        MODEL_PLANE_OBSERVATION_VERSION_V2,
    ):
        _reject(
            "invalid_observation_version",
            "Observation version must be 1 or 2",
        )

    direct_key = settings.model_plane_observation_api_key
    key_file_value = settings.model_plane_observation_api_key_file.strip()
    if bool(direct_key) == bool(key_file_value):
        _reject(
            "ambiguous_api_key",
            "Configure exactly one observation API key source",
        )
    key_file = Path(key_file_value).expanduser() if key_file_value else None
    if key_file is not None:
        _read_api_key_file(key_file)
        direct_key = ""
    else:
        direct_key = _validate_api_key(direct_key)

    return ModelPlaneObservationConfig(
        endpoint=endpoint,
        api_key=direct_key,
        api_key_file=key_file,
        deployment_id=deployment_id,
        target_environment=target_environment,
        engine_instance_id=engine_instance_id,
        auth_enabled=settings.auth_enabled,
        observation_version=observation_version,
        interval_seconds=settings.model_plane_observation_interval_seconds,
        timeout_seconds=settings.model_plane_observation_timeout_seconds,
        jitter_ratio=settings.model_plane_observation_jitter_ratio,
    )


def read_model_plane_observation_api_key(config: ModelPlaneObservationConfig) -> str:
    """Read every dispatch so a mounted secret can rotate without restart."""
    if config.api_key_file is not None:
        return _read_api_key_file(config.api_key_file)
    return _validate_api_key(config.api_key)


def _canonical_observed_at(now: datetime) -> str:
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)
    return now.astimezone(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def model_inventory_summary(model_list: ModelList) -> tuple[str, int, int]:
    """Digest sorted unique available IDs without disclosing model names."""
    model_ids = sorted({item.id for item in model_list.data})
    canonical = json.dumps(model_ids, ensure_ascii=False, separators=(",", ":"))
    digest = f"sha256:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"
    return digest, len(model_ids), len(model_list.unavailable)


def model_routing_inventory_summary(
    state: _ObservationState,
    candidate_availability: Callable[[str], bool],
) -> dict[str, str | int | None]:
    """Summarize signed-route coverage without reporting model or route names."""
    active = state.model_routing_runtime.policy
    if active is None:
        return {
            "object": "model_routing_inventory.status",
            "status": "not_applicable",
            "policy_digest": None,
            "candidate_count": 0,
            "available_candidate_count": 0,
            "unavailable_candidate_count": 0,
            "ready_route_count": 0,
            "unavailable_route_count": 0,
        }

    routes = active.verified.claims.routes
    route_candidates = [
        (route.primary_model, *route.fallback_models) for route in routes
    ]
    candidates = sorted(
        {candidate for candidate_set in route_candidates for candidate in candidate_set}
    )
    available = {
        candidate for candidate in candidates if candidate_availability(candidate)
    }
    ready_route_count = sum(
        1
        for candidate_set in route_candidates
        if any(candidate in available for candidate in candidate_set)
    )
    unavailable_route_count = len(routes) - ready_route_count
    if unavailable_route_count == 0:
        status = "ready"
    elif ready_route_count == 0:
        status = "unavailable"
    else:
        status = "degraded"
    return {
        "object": "model_routing_inventory.status",
        "status": status,
        "policy_digest": active.digest,
        "candidate_count": len(candidates),
        "available_candidate_count": len(available),
        "unavailable_candidate_count": len(candidates) - len(available),
        "ready_route_count": ready_route_count,
        "unavailable_route_count": unavailable_route_count,
    }


def build_model_plane_observation(
    config: ModelPlaneObservationConfig,
    state: _ObservationState,
    inventory_provider: Callable[[], ModelList],
    *,
    candidate_availability: Callable[[str], bool] | None = None,
    observation_id: str | None = None,
    now: datetime | None = None,
) -> dict:
    """Build the platform's exact payload-free v1 observation shape."""
    routing = build_model_routing_status(
        state.model_routing_runtime,
        auth_enabled=config.auth_enabled,
        rate_limit_scope=state.model_routing_rate_limiter.scope,
    )
    if routing.active and (
        routing.deployment_id != config.deployment_id
        or routing.environment != config.target_environment
    ):
        _reject(
            "routing_scope_mismatch",
            "Active routing policy does not match observation deployment scope",
        )

    inventory_digest, available_count, unavailable_count = model_inventory_summary(
        inventory_provider()
    )
    readiness = state.readiness()
    if not readiness.get("ready"):
        health_status = "not_ready"
    elif routing.candidate_error_code:
        health_status = "degraded"
    else:
        health_status = "ready"

    payload = {
        "artifactType": MODEL_PLANE_OBSERVATION_TYPE,
        "observationVersion": config.observation_version,
        "observationId": _validate_identifier(
            observation_id or str(uuid4()),
            "observation id",
        ),
        "deploymentId": config.deployment_id,
        "targetEnvironment": config.target_environment,
        "engineInstanceId": config.engine_instance_id,
        "engineVersion": _validate_identifier(__version__, "engine version"),
        "healthStatus": health_status,
        "inventoryDigest": inventory_digest,
        "availableModelCount": available_count,
        "unavailableModelCount": unavailable_count,
        "observedAt": _canonical_observed_at(now or datetime.now(UTC)),
        "routingPolicy": routing.model_dump(mode="json"),
    }
    if config.observation_version == MODEL_PLANE_OBSERVATION_VERSION_V2:
        if candidate_availability is None:
            _reject(
                "routing_inventory_resolver_missing",
                "Observation v2 requires a local model-availability resolver",
            )
        payload["routingInventory"] = model_routing_inventory_summary(
            state,
            candidate_availability,
        )
    return payload


def _iso_timestamp(value: float | None) -> str | None:
    if value is None:
        return None
    return _canonical_observed_at(datetime.fromtimestamp(value, tz=UTC))


class ModelPlaneObservationReporter:
    def __init__(
        self,
        config: ModelPlaneObservationConfig,
        state: _ObservationState,
        inventory_provider: Callable[[], ModelList],
        candidate_availability: Callable[[str], bool] | None = None,
    ) -> None:
        self.config = config
        self._state = state
        self._inventory_provider = inventory_provider
        self._candidate_availability = candidate_availability
        self._running = False
        self._attempts_total = 0
        self._successes_total = 0
        self._failures_total = 0
        self._consecutive_failures = 0
        self._last_attempt_at: float | None = None
        self._last_success_at: float | None = None
        self._last_error_code: str | None = None
        self._pending: dict | None = None

    @property
    def metrics_snapshot(self) -> dict[str, int | float | None]:
        return {
            "running": 1 if self._running else 0,
            "attempts_total": self._attempts_total,
            "successes_total": self._successes_total,
            "failures_total": self._failures_total,
            "consecutive_failures": self._consecutive_failures,
            "pending": 1 if self._pending is not None else 0,
            "last_success_unixtime": self._last_success_at,
        }

    def status(self) -> ModelPlaneObserverStatus:
        return ModelPlaneObserverStatus(
            enabled=True,
            running=self._running,
            attempts_total=self._attempts_total,
            successes_total=self._successes_total,
            failures_total=self._failures_total,
            consecutive_failures=self._consecutive_failures,
            pending_observation_id=(
                str(self._pending["observationId"]) if self._pending is not None else None
            ),
            last_attempt_at=_iso_timestamp(self._last_attempt_at),
            last_success_at=_iso_timestamp(self._last_success_at),
            last_error_code=self._last_error_code,
        )

    def _mark_failure(self, code: str, *, retain_pending: bool) -> None:
        self._failures_total += 1
        self._consecutive_failures += 1
        self._last_error_code = code
        if not retain_pending:
            self._pending = None
        log.warning(
            "model_plane_observation_failed",
            error_code=code,
            consecutive_failures=self._consecutive_failures,
            retained_for_retry=retain_pending and self._pending is not None,
        )

    def _mark_success(self, status_code: int) -> None:
        observation_id = self._pending["observationId"] if self._pending else None
        self._successes_total += 1
        self._consecutive_failures = 0
        self._last_success_at = time.time()
        self._last_error_code = None
        self._pending = None
        log.debug(
            "model_plane_observation_recorded",
            observation_id=observation_id,
            status_code=status_code,
        )

    async def report_once(self, client: httpx.AsyncClient) -> bool:
        """Attempt one delivery; return true only when the platform accepts it."""
        self._attempts_total += 1
        self._last_attempt_at = time.time()
        if self._pending is None:
            try:
                self._pending = await asyncio.to_thread(
                    build_model_plane_observation,
                    self.config,
                    self._state,
                    self._inventory_provider,
                    candidate_availability=self._candidate_availability,
                )
            except ModelPlaneObservationConfigError as exc:
                self._mark_failure(exc.code, retain_pending=False)
                return False
            except Exception:  # noqa: BLE001 - report failure must not affect inference
                self._mark_failure("observation_build_failed", retain_pending=False)
                return False

        try:
            api_key = read_model_plane_observation_api_key(self.config)
        except ModelPlaneObservationConfigError as exc:
            self._mark_failure(exc.code, retain_pending=True)
            return False

        observation_id = str(self._pending["observationId"])
        try:
            with span(
                "model_plane.observation.report",
                **{
                    "model_plane.observation_id": observation_id,
                    "model_plane.deployment_id": self.config.deployment_id,
                    "model_plane.environment": self.config.target_environment,
                },
            ) as report_span:
                response = await client.post(
                    self.config.endpoint,
                    json=self._pending,
                    headers={"x-api-key": api_key},
                    follow_redirects=False,
                )
                report_span.bind(**{"http.response.status_code": response.status_code})
        except Exception:  # noqa: BLE001 - reporting never changes inference availability
            self._mark_failure("transport_error", retain_pending=True)
            return False

        if 200 <= response.status_code < 300:
            self._mark_success(response.status_code)
            return True

        retryable = (
            response.status_code in _RETRYABLE_STATUS_CODES or response.status_code >= 500
        )
        self._mark_failure(f"http_{response.status_code}", retain_pending=retryable)
        return False

    async def run(self) -> None:
        """Report immediately after startup, then at a bounded jittered cadence."""
        self._running = True
        try:
            timeout = httpx.Timeout(self.config.timeout_seconds)
            async with httpx.AsyncClient(timeout=timeout, follow_redirects=False) as client:
                while True:
                    await self.report_once(client)
                    jitter = random.uniform(-self.config.jitter_ratio, self.config.jitter_ratio)
                    await asyncio.sleep(self.config.interval_seconds * (1.0 + jitter))
        finally:
            self._running = False
