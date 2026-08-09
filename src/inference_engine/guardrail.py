"""Engine-side client for the ``orchestra-guardrail-evaluate-v1`` contract.

The engine is the one enforcement point a caller reaches without going through
the SDK runtime, so without this seam ``/v1/chat/completions`` is unguarded by
construction. This module owns the wire shape, the version window, the verdict
resolution, and the fail mode; ``api/_guardrail.py`` owns the route wiring.

Two invariants shape everything below.

**Inert unless configured.** ``load_guardrail_config`` returns ``None`` when
``guardrail_enabled`` is false, ``app_state.guardrail`` stays ``None``, and
every route helper returns before it builds a payload. A deployment that leaves
the switch off performs no work and takes no branch it did not take before.
Enabled-but-incomplete is a startup error rather than a degraded mode, because
a guardrail that silently does not run is worse than one that is absent.

**No usable verdict is not a verdict.** Timeouts, transport failures, 4xx, 5xx,
and version mismatches all resolve through the profile fail mode
(``closed`` by default, forced closed on the certified workload surface) and
never through a hardcoded allow or deny. A returned verdict is always obeyed.
"""

from __future__ import annotations

import ipaddress
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Never
from urllib.parse import urlsplit

import httpx

from .config import CERTIFIED_MODEL_WORKLOAD_SURFACE
from .observability import get_logger


# A window, never an equality check: two of the three implementations may be a
# version apart without the fleet breaking (contract 5.1).
GUARDRAIL_CONTRACT_VERSION = 1
GUARDRAIL_CONTRACT_MIN_SUPPORTED = 1
GUARDRAIL_CONTRACT_MAX_SUPPORTED = 1

EVALUATE_PATH = "/v1/guardrail:evaluate"

STAGE_LLM_INPUT = "llm_input"
STAGE_LLM_OUTPUT = "llm_output"
STAGE_TOOL_RESULT = "tool_result"

VERDICT_ALLOW = "allow"
VERDICT_TRANSFORM = "transform"
VERDICT_DENY = "deny"
# Not a wire verdict: the client's own marker for "the service produced no
# usable verdict and the fail mode resolved to closed".
VERDICT_UNAVAILABLE = "unavailable"

_WIRE_VERDICTS = frozenset({VERDICT_ALLOW, VERDICT_TRANSFORM, VERDICT_DENY})

# Per-stage in-band p99 server budgets (contract 2.9).
_STAGE_BUDGET_MS = {
    STAGE_LLM_INPUT: 25,
    STAGE_LLM_OUTPUT: 40,
    STAGE_TOOL_RESULT: 40,
}
_TRANSPORT_SLACK_MS = 10

_MAX_API_KEY_BYTES = 4_096
_PROFILE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_GUARDRAIL_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_MAX_GUARDRAIL_NAMES = 64
_REASON_CODE = re.compile(r"^[a-z][a-z0-9_]{0,127}$")
_DETECTOR_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")

_STATUS_REASON_CODES = {
    400: "guardrail_request_invalid",
    401: "guardrail_unauthenticated",
    403: "guardrail_tenant_mismatch",
    404: "guardrail_profile_unknown",
    413: "guardrail_payload_too_large",
    422: "guardrail_request_unsupported",
    429: "guardrail_overloaded",
    500: "guardrail_detector_failed",
}

_RESPONSE_KEYS = frozenset(
    {
        "contractVersion",
        "requestId",
        "verdict",
        "reason",
        "reasonCode",
        "evaluatedGuardrails",
        "transformedPayload",
        "assessments",
        "detectorPack",
        "latencyMs",
        "compat",
    }
)
_PAYLOAD_KEYS = frozenset({"kind", "text", "json"})
_DETECTOR_PACK_KEYS = frozenset({"id", "version", "digest"})

log = get_logger("guardrail")


class GuardrailConfigError(ValueError):
    """An enabled guardrail client is incomplete or unsafe to run."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class GuardrailConfig:
    """Validated operator configuration. Constructed only by ``load_guardrail_config``."""

    endpoint: str
    profile: str
    guardrails: tuple[str, ...]
    fail_mode: Literal["closed", "open"]
    timeout_seconds: float
    api_key: str
    api_key_file: Path | None
    fail_open_max_consecutive: int
    fail_open_window_seconds: float

    @property
    def evaluate_url(self) -> str:
        return f"{self.endpoint}{EVALUATE_PATH}"


@dataclass(frozen=True)
class GuardrailOutcome:
    """What the route acts on. Carries no checked content except ``transformed``."""

    stage: str
    verdict: str
    reason_code: str
    evaluated_count: int = 0
    findings_count: int = 0
    detector_digest: str | None = None
    latency_ms: float | None = None
    unknown_fields_dropped: int = 0
    # What the *server* dropped from this client's request (contract 5.2). The
    # count above is the mirror image: what this client dropped from the
    # response. Skew is one-directional, so both numbers are needed to see it.
    request_fields_dropped_by_server: int = 0
    failed_open: bool = False
    transformed: Any = None


def _reject(code: str, message: str) -> Never:
    raise GuardrailConfigError(code, message)


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
        # ``urlsplit`` defers port parsing to attribute access, so reading the
        # attribute is what rejects a malformed port.
        parsed.port
    except ValueError:
        return _reject("invalid_endpoint", "Guardrail endpoint is not a valid URL")
    if (
        not hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path not in ("", "/")
    ):
        return _reject(
            "invalid_endpoint",
            "Guardrail endpoint must be a bare origin; the evaluate path is contract-owned",
        )
    if parsed.scheme != "https" and not (parsed.scheme == "http" and _is_loopback(hostname)):
        return _reject(
            "insecure_endpoint",
            "Guardrail endpoint must use HTTPS except on loopback",
        )
    return value.rstrip("/")


def _validate_guardrails(value: str) -> tuple[str, ...]:
    """Parse the operator's guardrail list. The caller names its own coverage.

    An enabled deployment that named nothing would put the whole coverage
    decision on the far side of the wire, where this engine cannot see it and
    an empty result is indistinguishable from a clean scan.
    """
    names = [item.strip() for item in value.split(",") if item.strip()]
    if not names:
        _reject("missing_guardrails", "Guardrail is enabled with no guardrail names configured")
    if len(names) > _MAX_GUARDRAIL_NAMES:
        _reject("invalid_guardrails", "Too many guardrail names configured")
    if len(set(names)) != len(names):
        _reject("invalid_guardrails", "Guardrail names must be unique")
    for name in names:
        if not _GUARDRAIL_NAME.fullmatch(name):
            _reject("invalid_guardrails", "Guardrail names must be bounded identifiers")
    return tuple(names)


def _validate_api_key(value: str) -> str:
    key = value.strip()
    try:
        encoded = key.encode("ascii")
    except UnicodeEncodeError:
        return _reject("invalid_api_key", "Guardrail API key must be an ASCII token")
    if not key or len(encoded) > _MAX_API_KEY_BYTES or any(char.isspace() for char in key):
        _reject("invalid_api_key", "Guardrail API key is empty, oversized, or malformed")
    return key


def _read_api_key_file(path: Path) -> str:
    try:
        with path.open("rb") as stream:
            raw = stream.read(_MAX_API_KEY_BYTES + 1)
        if len(raw) > _MAX_API_KEY_BYTES:
            _reject("invalid_api_key", "Guardrail API key file is oversized")
        return _validate_api_key(raw.decode("utf-8"))
    except GuardrailConfigError:
        raise
    except (OSError, UnicodeError) as exc:
        raise GuardrailConfigError(
            "api_key_unavailable",
            "Guardrail API key file cannot be read",
        ) from exc


def load_guardrail_config(settings) -> GuardrailConfig | None:
    """Validate an operator-configured guardrail client, or return ``None``.

    ``guardrail_enabled`` is the off switch the contract names, and it is
    checked first so an unconfigured deployment reads no files and validates
    nothing. The two settings must agree in both directions: an endpoint
    configured while the switch is off would otherwise read as guarded to the
    operator who set it and be inert in production.
    """
    endpoint_value = settings.guardrail_endpoint.strip()
    if not settings.guardrail_enabled:
        if endpoint_value:
            _reject(
                "guardrail_disabled",
                "Guardrail endpoint is configured while GUARDRAIL_ENABLED is false",
            )
        return None
    if not endpoint_value:
        _reject("missing_endpoint", "Guardrail is enabled with no endpoint configured")

    endpoint = _validate_endpoint(endpoint_value)
    profile = settings.guardrail_profile.strip()
    if not _PROFILE.fullmatch(profile):
        _reject("invalid_profile", "Guardrail profile must be a bounded identifier")
    guardrails = _validate_guardrails(settings.guardrail_names)

    fail_mode = settings.guardrail_fail_mode
    certified = settings.model_plane_workload_surface == CERTIFIED_MODEL_WORKLOAD_SURFACE
    if certified and fail_mode != "closed":
        # The certified surface has no fail-open, consistent with every other
        # rule on it. A startup error, never a runtime surprise.
        _reject(
            "fail_open_not_permitted",
            "Guardrail fail mode must be closed under the certified workload surface",
        )

    direct_key = settings.guardrail_api_key
    key_file_value = settings.guardrail_api_key_file.strip()
    if direct_key and key_file_value:
        _reject("ambiguous_api_key", "Configure at most one guardrail API key source")
    if not direct_key and not key_file_value:
        # Bearer auth is mandatory on the evaluate route, so a keyless client
        # would 401 every request. Under fail-closed that is an outage dressed
        # as a policy decision; under fail-open it is a deployment that looks
        # configured and guards nothing.
        _reject("missing_api_key", "Guardrail is enabled with no API key configured")
    key_file = Path(key_file_value).expanduser() if key_file_value else None
    if key_file is not None:
        _read_api_key_file(key_file)
        direct_key = ""
    elif direct_key:
        direct_key = _validate_api_key(direct_key)

    return GuardrailConfig(
        endpoint=endpoint,
        profile=profile,
        guardrails=guardrails,
        fail_mode=fail_mode,
        timeout_seconds=settings.guardrail_timeout_seconds,
        api_key=direct_key,
        api_key_file=key_file,
        fail_open_max_consecutive=settings.guardrail_fail_open_max_consecutive,
        fail_open_window_seconds=settings.guardrail_fail_open_window_seconds,
    )


def read_guardrail_api_key(config: GuardrailConfig) -> str:
    """Read every dispatch so a mounted secret can rotate without restart."""
    if config.api_key_file is not None:
        return _read_api_key_file(config.api_key_file)
    return config.api_key


def stage_budget_ms(stage: str, timeout_seconds: float) -> int:
    """The server budget this client can actually wait for.

    The contract's client deadline is ``budgetMs + transportSlackMs``. Running
    that backwards from the configured deadline is what keeps the number the
    server is told honest when an operator tightens the timeout.
    """
    deadline_ms = int(timeout_seconds * 1000)
    return max(1, min(_STAGE_BUDGET_MS.get(stage, 40), deadline_ms - _TRANSPORT_SLACK_MS))


def _sanitize_reason_code(value: Any, fallback: str) -> str:
    """Reason codes reach spans and error bodies, so they are shape-checked.

    A server that stuffed checked content into this field would otherwise turn
    the payload-free surface into a leak.
    """
    if isinstance(value, str) and _REASON_CODE.fullmatch(value):
        return value
    return fallback


def _sanitize_detector_digest(value: Any) -> str | None:
    if isinstance(value, str) and _DETECTOR_DIGEST.fullmatch(value):
        return value
    return None


def _drop_unknown(obj: Any, allowed: frozenset[str]) -> tuple[dict, int]:
    if not isinstance(obj, dict):
        return {}, 0
    known = {k: v for k, v in obj.items() if k in allowed}
    return known, len(obj) - len(known)


def _payload_field(payload: Any) -> tuple[dict, str]:
    """Encode a payload for the wire, returning the object and its kind.

    Raises ``TypeError``/``ValueError`` for a payload httpx could not
    serialize; the caller resolves that through the fail mode rather than
    letting it surface as an unhandled 500 on the inference path.
    """
    if isinstance(payload, str):
        return {"kind": "text", "text": payload}, "text"
    json.dumps(payload, allow_nan=False)
    return {"kind": "json", "json": payload}, "json"


def _decode_transformed(payload: Any, expected_kind: str, stage: str) -> tuple[Any, bool]:
    """Return ``(value, ok)`` for a ``transformedPayload``.

    A transform normally comes back as the kind it was sent as. The one legal
    exception is ``json`` narrowing to ``text`` at ``tool_result``: defanging a
    tool result also wraps it in the untrusted-output framing, that framing is
    prose, and a JSON document has nowhere to carry prose — so the framing costs
    the shape. The narrowing is confined to ``tool_result`` because that is the
    only stage the framing is defined for; widening ``text`` to ``json`` has no
    such justification and stays a refusal.

    Accepting the narrowing here does not mean the caller can use it. A route
    that needs its own shape back — a message list, a batch of strings — refuses
    a bare string as an inapplicable transform, and an inapplicable transform is
    a deny, never a release of the original.
    """
    known, _ = _drop_unknown(payload, _PAYLOAD_KEYS)
    kind = known.get("kind")
    framed = expected_kind == "json" and kind == "text" and stage == STAGE_TOOL_RESULT
    if kind != expected_kind and not framed:
        return None, False
    if kind == "text":
        text = known.get("text")
        return (text, True) if isinstance(text, str) else (None, False)
    value = known.get("json")
    if value is None:
        return None, False
    return value, True


class GuardrailClient:
    """One long-lived HTTP client for the guardrail service.

    ``evaluate`` never raises: every failure path resolves through the fail
    mode, because an exception escaping here would put a guardrail outage on
    the inference request path as an unhandled 500.
    """

    def __init__(
        self,
        config: GuardrailConfig,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.config = config
        self._http = httpx.AsyncClient(
            timeout=httpx.Timeout(config.timeout_seconds),
            follow_redirects=False,
            transport=transport,
        )
        self._fail_open_streak = 0
        self._fail_open_window_started = 0.0

    async def aclose(self) -> None:
        await self._http.aclose()

    def _auth_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {read_guardrail_api_key(self.config)}",
            "Content-Type": "application/json",
        }

    async def evaluate(
        self,
        *,
        stage: str,
        payload: Any,
        tenant: str,
        request_id: str,
        org_id: str | None = None,
        tool: dict | None = None,
    ) -> GuardrailOutcome:
        budget_ms = stage_budget_ms(stage, self.config.timeout_seconds)
        try:
            payload_field, kind = _payload_field(payload)
        except (TypeError, ValueError):
            return self._unusable(stage, "guardrail_request_invalid")
        body = {
            "contractVersion": GUARDRAIL_CONTRACT_VERSION,
            "requestId": request_id[:256],
            "stage": stage,
            "profile": self.config.profile,
            "budgetMs": budget_ms,
            "payload": payload_field,
            # Name-only declarations: the engine serves raw OpenAI traffic and
            # holds no bundle, so the profile supplies each guardrail's type and
            # violation action. Same object shape a bundle-holding caller sends,
            # so the service parses one shape rather than a union.
            "guardrails": [
                {"name": name, "guardrailType": None, "onViolation": None}
                for name in self.config.guardrails
            ],
            "subject": {"tenant": tenant, "orgId": org_id},
            "tool": tool,
        }

        try:
            headers = self._auth_headers()
        except GuardrailConfigError:
            return self._unusable(stage, "guardrail_client_failed")

        response, failure = await self._post(body, headers, budget_ms)
        if response is None:
            return self._unusable(stage, failure or "guardrail_unavailable")
        return self._resolve(stage, kind, response)

    async def _post(
        self,
        body: dict,
        headers: dict[str, str],
        budget_ms: int,
    ) -> tuple[httpx.Response | None, str | None]:
        """One attempt, plus at most one retry for 429 and connection errors.

        Both attempts share one deadline: the contract's client deadline is
        ``budgetMs + transportSlackMs``, singular, so a retry may not re-arm it.

        A 500 is not retried in-band: a crashing detector crashes again, and
        the second crash costs the caller latency it does not have.
        """
        attempts = 0
        deadline = time.perf_counter() + self.config.timeout_seconds
        while True:
            attempts += 1
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                return None, "guardrail_timeout"
            try:
                response = await self._http.post(
                    self.config.evaluate_url,
                    json=body,
                    headers=headers,
                    timeout=remaining,
                )
            except httpx.TimeoutException:
                return None, "guardrail_timeout"
            except httpx.TransportError:
                if attempts == 1 and self._has_retry_budget(deadline, budget_ms):
                    continue
                return None, "guardrail_unavailable"
            except Exception:  # noqa: BLE001 - a client fault is never a verdict
                return None, "guardrail_unavailable"
            if (
                response.status_code == 429
                and attempts == 1
                and self._has_retry_budget(deadline, budget_ms)
            ):
                continue
            return response, None

    def _has_retry_budget(self, deadline: float, budget_ms: int) -> bool:
        """Contract 2.8: retry only while the remaining deadline beats ``budgetMs / 2``."""
        return (deadline - time.perf_counter()) > (budget_ms / 2000.0)

    def _resolve(self, stage: str, kind: str, response: httpx.Response) -> GuardrailOutcome:
        if response.status_code != 200:
            return self._unusable(stage, self._error_reason_code(response))

        try:
            body = response.json()
        except Exception:  # noqa: BLE001 - unparseable body is not a verdict
            return self._unusable(stage, "guardrail_request_invalid")
        if not isinstance(body, dict):
            return self._unusable(stage, "guardrail_request_invalid")

        known, dropped = _drop_unknown(body, _RESPONSE_KEYS)
        server_version = known.get("contractVersion")
        if not isinstance(server_version, int) or not (
            GUARDRAIL_CONTRACT_MIN_SUPPORTED <= server_version <= GUARDRAIL_CONTRACT_MAX_SUPPORTED
        ):
            return self._unusable(stage, "guardrail_contract_version_unsupported")

        evaluated = known.get("evaluatedGuardrails")
        if not isinstance(evaluated, list) or not evaluated:
            # "Nothing was evaluated" is not "I checked and found nothing". A
            # response with an empty coverage set carries no assurance, so it
            # resolves through the fail mode instead of reading as an allow.
            return self._unusable(stage, "guardrail_coverage_empty")

        self._record_success()
        verdict = known.get("verdict")
        reason_code = _sanitize_reason_code(known.get("reasonCode"), "guardrail_reason_unreported")
        assessments = known.get("assessments")
        pack, pack_dropped = _drop_unknown(known.get("detectorPack"), _DETECTOR_PACK_KEYS)
        latency = known.get("latencyMs")

        base = {
            "stage": stage,
            "evaluated_count": len(evaluated),
            "findings_count": _violated_count(assessments),
            "detector_digest": _sanitize_detector_digest(pack.get("digest")),
            "latency_ms": float(latency) if isinstance(latency, (int, float)) else None,
            "unknown_fields_dropped": dropped + pack_dropped,
            "request_fields_dropped_by_server": _dropped_count(known.get("compat")),
        }

        if verdict not in _WIRE_VERDICTS:
            # An unrecognized verdict is the exact class of bug this contract
            # exists to prevent: ignoring it would be a fail-open.
            return GuardrailOutcome(
                verdict=VERDICT_DENY, reason_code="guardrail_verdict_unrecognized", **base
            )

        transformed_payload = known.get("transformedPayload")
        if verdict != VERDICT_TRANSFORM:
            return GuardrailOutcome(verdict=verdict, reason_code=reason_code, **base)

        if transformed_payload is None:
            return GuardrailOutcome(
                verdict=VERDICT_DENY, reason_code="guardrail_transform_missing", **base
            )
        value, ok = _decode_transformed(transformed_payload, kind, stage)
        if not ok:
            return GuardrailOutcome(
                verdict=VERDICT_DENY, reason_code="guardrail_transform_invalid", **base
            )
        return GuardrailOutcome(
            verdict=VERDICT_TRANSFORM, reason_code=reason_code, transformed=value, **base
        )

    def _error_reason_code(self, response: httpx.Response) -> str:
        """Contract 2.8 maps status to reason code; the server body does not.

        Reading the code back off the body would let a server name a condition
        outside the table — and this string reaches the caller's error envelope
        and the span.
        """
        return _STATUS_REASON_CODES.get(
            response.status_code,
            "guardrail_detector_failed" if response.status_code >= 500 else "guardrail_unavailable",
        )

    def fault_outcome(self, stage: str) -> GuardrailOutcome:
        """Resolve a fault raised by this client object rather than by the service."""
        return self._unusable(stage, "guardrail_client_failed")

    def _record_success(self) -> None:
        self._fail_open_streak = 0
        self._fail_open_window_started = 0.0

    def _unusable(self, stage: str, reason_code: str) -> GuardrailOutcome:
        """Resolve "no usable verdict" through the fail mode, and only that."""
        if self.config.fail_mode != "open":
            return GuardrailOutcome(
                stage=stage, verdict=VERDICT_UNAVAILABLE, reason_code=reason_code
            )

        now = time.monotonic()
        if self._fail_open_streak == 0:
            self._fail_open_window_started = now
        # Nothing here resets the streak on a timer. Recovery runs through
        # ``_record_success`` alone (contract 3.3): a service that never
        # recovers must not periodically re-open the fleet. The window is the
        # second trip condition, not an expiry — an unbroken fail-open run
        # longer than the window trips even when its rate never reaches
        # ``fail_open_max_consecutive``.
        self._fail_open_streak += 1
        if (
            self._fail_open_streak > self.config.fail_open_max_consecutive
            or (now - self._fail_open_window_started) > self.config.fail_open_window_seconds
        ):
            return GuardrailOutcome(
                stage=stage,
                verdict=VERDICT_UNAVAILABLE,
                reason_code="guardrail_fail_open_budget_exhausted",
            )
        log.warning(
            "guardrail.fail_open",
            stage=stage,
            reason_code=reason_code,
            consecutive=self._fail_open_streak,
        )
        return GuardrailOutcome(
            stage=stage,
            verdict=VERDICT_ALLOW,
            reason_code="guardrail_unavailable_fail_open",
            failed_open=True,
        )


def client_fault_outcome(stage: str, client: Any) -> GuardrailOutcome:
    """Resolve a fault in the client object itself, not in the service.

    Routed through the client's own fail-open allowance, so a client-side fault
    costs exactly what a service-side one does: an allow that spends nothing is
    an unbounded fail-open. An object that cannot resolve it — a stub, or a
    client too broken to answer — resolves to ``closed``, on the rule that
    anything which could change whether content is released resolves toward
    denial.
    """
    resolve = getattr(client, "fault_outcome", None)
    if callable(resolve):
        return resolve(stage)
    return GuardrailOutcome(
        stage=stage, verdict=VERDICT_UNAVAILABLE, reason_code="guardrail_client_failed"
    )


def _dropped_count(compat: Any) -> int:
    if not isinstance(compat, dict):
        return 0
    value = compat.get("unknownFieldsDropped")
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _violated_count(assessments: Any) -> int:
    if not isinstance(assessments, list):
        return 0
    return sum(
        1
        for item in assessments
        if isinstance(item, dict) and item.get("violated") is True
    )


__all__ = [
    "EVALUATE_PATH",
    "GUARDRAIL_CONTRACT_MAX_SUPPORTED",
    "GUARDRAIL_CONTRACT_MIN_SUPPORTED",
    "GUARDRAIL_CONTRACT_VERSION",
    "STAGE_LLM_INPUT",
    "STAGE_LLM_OUTPUT",
    "STAGE_TOOL_RESULT",
    "VERDICT_ALLOW",
    "VERDICT_DENY",
    "VERDICT_TRANSFORM",
    "VERDICT_UNAVAILABLE",
    "GuardrailClient",
    "GuardrailConfig",
    "GuardrailConfigError",
    "GuardrailOutcome",
    "client_fault_outcome",
    "load_guardrail_config",
    "read_guardrail_api_key",
    "stage_budget_ms",
]
