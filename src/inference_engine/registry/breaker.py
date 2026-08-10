"""Per-deployment cooldown + circuit breaker, wired into the probe layer.

Where this sits
---------------

The registry probes (``vllm_probe``, ``openrouter_probe``) already answer one
question per upstream — "can this deployment serve right now?" — cache it with
a TTL, and feed ``ModelManager``'s probe-aware resolver, which is what decides
the candidate a request lands on. This module extends that same answer with
failure history instead of standing up a second state machine beside it:

* every probe checks the cooldown *before* its own TTL cache, so an open
  deployment reports ``upstream_cooldown`` and drops out of candidate
  selection without an HTTP round trip;
* ``VLLMUpstreamProbe`` goes further and claims the half-open trial with
  :meth:`UpstreamBreaker.begin_attempt`, feeding its own outcome back in — its
  per-model ``/v1/models`` check is a cheaper trial than a real generation.
  ``OpenRouterProbe`` only reads the cooldown (:meth:`UpstreamBreaker.allows`),
  because its catalog fetch is shared by every model on the endpoint and so
  says nothing about one deployment;
* the adapters feed generation outcomes in from the hot path, which is the
  only place that sees a 503 on a real completion, and claim the trial for the
  deployments no probe covers.

The half-open probe is therefore not a new mechanism: when the cooldown
expires, the next probe or the next request that reaches the adapter is
admitted as the single trial, and its result either closes the breaker or
re-opens it for a longer cooldown.

What a "deployment" is
----------------------

``(backend, endpoint, model_id)``. The endpoint alone is too coarse for a
multi-model host: one Ollama server serving ten models should not lose all ten
because one of them is repeatedly OOMing. The model id alone is too coarse the
other way — two hosts serving the same model id are two deployments, and one
being down says nothing about the other.

What counts as a failure
------------------------

From the hot path, exactly what the retry classifier calls transient
(``adapters._upstream.is_health_signal`` delegates to
``_upstream_retry.is_retryable``): connect / read / write / protocol transport
errors, and the status codes 408, 425, 429, 500, 502, 503, 504 and 529.

Two exclusions carry the weight, and both are deliberate:

* **A rejected request is never counted.** A 400, a 404 or a 422 is the request
  being wrong, so it leaves the deployment's history untouched — a caller
  sending malformed requests cannot cool a deployment down by being rejected.
  (That is a statement about rejections, not about callers in general: a
  request that makes the upstream itself fail with a 500 does count, because
  from here it is indistinguishable from any other 500.)
* **A timeout is our budget, not the upstream's verdict.** Elapsed time cannot
  tell a wedged deployment from a healthy one generating a long answer, so a
  deadline expiry and httpx's read/write/pool timeouts are all discounted.
  ``ConnectTimeout`` is the exception — it proves no connection was made.
  ``is_health_signal``'s docstring carries the full reasoning and the limit
  this leaves behind.

The vLLM probe feeds this breaker through a *different* classifier
(``vllm_probe.TRANSIENT_PROBE_REASONS``) and does count its own timeouts. That
is consistent rather than contradictory: it times a bare
``GET /v1/models`` against ``VLLM_UPSTREAM_PROBE_TIMEOUT_SECONDS``, where an
unanswered call is evidence about the server rather than about the length of
someone's completion.

What the threshold counts
-------------------------

LOGICAL CALLS, NOT ATTEMPTS. The adapters exhaust their own retries first and
report one verdict per request, so ``failure_threshold`` counts requests that
failed after every retry — not individual upstream round trips. With the
shipped defaults (``UPSTREAM_RETRY_MAX_ATTEMPTS=2``,
``UPSTREAM_BREAKER_FAILURE_THRESHOLD=3``) a deployment therefore absorbs up to
six failed upstream requests, across three callers, before it opens. That is
deliberate — a breaker that counted attempts would open twice as fast for a
blip the retry already absorbed — but an operator reading "threshold 3" will
not assume it, so it is written down here and in the README settings table.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

from ..config import settings
from ..observability import get_logger

log = get_logger("registry.breaker")

CLOSED = "closed"
OPEN = "open"
HALF_OPEN = "half_open"

# Prometheus gauge encoding. Ordered by severity so an alert reads `> 0`.
STATE_VALUES = {CLOSED: 0, HALF_OPEN: 1, OPEN: 2}

# Formats whose availability is a remote upstream's to lose. In-process
# backends (gguf, mlx) have no transport between us and the weights, so there
# is nothing here for them to describe.
HTTP_FORMATS = frozenset({"vllm", "openrouter", "ollama_http"})


def deployment_key(backend: str, endpoint: str | None, model_id: str | None) -> str:
    return f"{backend}|{(endpoint or '').rstrip('/')}|{model_id or ''}"


def descriptor_deployment_key(descriptor: Any) -> str | None:
    """Deployment key for a registry descriptor, or ``None`` when not remote."""
    fmt = getattr(descriptor, "format", "")
    if fmt not in HTTP_FORMATS:
        return None
    params = getattr(descriptor, "params", None) or {}
    model_id = params.get("model_id") or getattr(descriptor, "qualified_name", "")
    return deployment_key(fmt, getattr(descriptor, "endpoint", "") or "", str(model_id))


@dataclass
class DeploymentHealth:
    """Failure history for one deployment. All times are ``time.monotonic``."""

    consecutive_failures: int = 0
    open_until: float | None = None
    open_cycles: int = 0
    half_open_in_flight: bool = False
    opened_total: int = 0
    backend: str = ""
    endpoint: str = ""
    model_id: str = ""


@dataclass
class BreakerEntry:
    """A read-only view of one deployment, for metrics and diagnostics."""

    key: str
    backend: str
    endpoint: str
    model_id: str
    state: str
    consecutive_failures: int
    cooldown_remaining_seconds: float
    opened_total: int


@dataclass
class _Counters:
    opened_total: int = 0
    probes_admitted_total: int = 0
    skipped_total: int = 0
    by_backend_opened: dict[str, int] = field(default_factory=dict)


class UpstreamBreaker:
    """Consecutive-failure breaker with a cooldown and one half-open trial.

    Every method is safe to call from a worker thread (the probes run under
    ``asyncio.to_thread``) and from the event loop (the adapters), so the
    single lock guards all mutation.
    """

    def __init__(
        self,
        *,
        failure_threshold: int | None = None,
        cooldown_seconds: float | None = None,
        max_cooldown_seconds: float | None = None,
    ) -> None:
        self._failure_threshold = failure_threshold
        self._cooldown_seconds = cooldown_seconds
        self._max_cooldown_seconds = max_cooldown_seconds
        self._lock = threading.Lock()
        self._states: dict[str, DeploymentHealth] = {}
        self._counters = _Counters()

    # ------------------------------------------------------------------
    # configuration — read through settings so env changes take effect
    # ------------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        return settings.upstream_breaker_enabled

    @property
    def failure_threshold(self) -> int:
        if self._failure_threshold is not None:
            return max(1, self._failure_threshold)
        return max(1, settings.upstream_breaker_failure_threshold)

    @property
    def cooldown_seconds(self) -> float:
        if self._cooldown_seconds is not None:
            return max(0.0, self._cooldown_seconds)
        return max(0.0, settings.upstream_breaker_cooldown_seconds)

    @property
    def max_cooldown_seconds(self) -> float:
        if self._max_cooldown_seconds is not None:
            return max(0.0, self._max_cooldown_seconds)
        return max(0.0, settings.upstream_breaker_max_cooldown_seconds)

    # ------------------------------------------------------------------
    # admission
    # ------------------------------------------------------------------

    def begin_attempt(self, key: str | None) -> str:
        """Claim the right to call this deployment. Returns the state observed.

        ``closed`` and ``half_open`` mean go ahead; the caller MUST report the
        outcome via :meth:`record_success` / :meth:`record_failure`, because a
        granted half-open trial is held until it does. ``open`` means skip.
        """
        if key is None or not self.enabled:
            return CLOSED
        now = time.monotonic()
        with self._lock:
            state = self._states.get(key)
            if state is None or state.open_until is None:
                return CLOSED
            if now < state.open_until:
                self._counters.skipped_total += 1
                return OPEN
            if state.half_open_in_flight:
                # Another caller already holds the single trial.
                self._counters.skipped_total += 1
                return OPEN
            state.half_open_in_flight = True
            self._counters.probes_admitted_total += 1
            return HALF_OPEN

    def allows(self, key: str | None) -> bool:
        """Non-claiming check — ``True`` unless the deployment is in cooldown.

        Used by callers that only need to filter a candidate list and will not
        report an outcome. It never consumes the half-open trial.
        """
        return self.state(key) != OPEN

    def state(self, key: str | None) -> str:
        if key is None or not self.enabled:
            return CLOSED
        now = time.monotonic()
        with self._lock:
            state = self._states.get(key)
            if state is None or state.open_until is None:
                return CLOSED
            if now < state.open_until:
                return OPEN
            return HALF_OPEN

    def cooldown_remaining(self, key: str | None) -> float:
        if key is None:
            return 0.0
        now = time.monotonic()
        with self._lock:
            state = self._states.get(key)
            if state is None or state.open_until is None:
                return 0.0
            return max(0.0, state.open_until - now)

    # ------------------------------------------------------------------
    # outcomes
    # ------------------------------------------------------------------

    def abandon_attempt(self, key: str | None) -> None:
        """Hand a half-open trial back without judging the deployment.

        For outcomes that say nothing about upstream health — a 400, a
        misconfigured descriptor — the trial must be released or the slot stays
        held and the deployment never gets another probe. It is deliberately
        not a success: nothing was proven, so the cooldown stands.
        """
        if key is None:
            return
        with self._lock:
            state = self._states.get(key)
            if state is not None:
                state.half_open_in_flight = False

    def record_success(self, key: str | None) -> None:
        if key is None:
            return
        with self._lock:
            state = self._states.get(key)
            if state is None:
                return
            was_open = state.open_until is not None
            state.consecutive_failures = 0
            state.open_until = None
            state.open_cycles = 0
            state.half_open_in_flight = False
        if was_open:
            log.info("breaker.closed", deployment=key)

    def record_failure(
        self,
        key: str | None,
        *,
        reason: str = "",
        backend: str = "",
        endpoint: str = "",
        model_id: str = "",
    ) -> None:
        """Count one transient failure. May open, or re-open, the deployment."""
        if key is None or not self.enabled:
            return
        now = time.monotonic()
        with self._lock:
            state = self._states.get(key)
            if state is None:
                state = DeploymentHealth()
                self._states[key] = state
            state.backend = backend or state.backend
            state.endpoint = endpoint or state.endpoint
            state.model_id = model_id or state.model_id
            state.consecutive_failures += 1
            was_trialling = state.half_open_in_flight
            state.half_open_in_flight = False

            if not was_trialling and state.open_until is None:
                if state.consecutive_failures < self.failure_threshold:
                    return
                state.open_cycles = 1
            else:
                # The half-open trial failed: the deployment is still sick, so
                # back off further rather than probing at the same cadence.
                state.open_cycles += 1

            cooldown = min(
                self.max_cooldown_seconds,
                self.cooldown_seconds * (2 ** max(0, state.open_cycles - 1)),
            )
            state.open_until = now + cooldown
            state.opened_total += 1
            self._counters.opened_total += 1
            self._counters.by_backend_opened[state.backend] = (
                self._counters.by_backend_opened.get(state.backend, 0) + 1
            )
            opened_for = cooldown
            failures = state.consecutive_failures
        log.warning(
            "breaker.opened",
            deployment=key,
            reason=reason,
            cooldown_seconds=round(opened_for, 3),
            consecutive_failures=failures,
        )

    # ------------------------------------------------------------------
    # introspection
    # ------------------------------------------------------------------

    def entries(self) -> list[BreakerEntry]:
        now = time.monotonic()
        with self._lock:
            snapshot = [(key, DeploymentHealth(**vars(state))) for key, state in self._states.items()]
        out: list[BreakerEntry] = []
        for key, state in snapshot:
            if state.open_until is None:
                resolved, remaining = CLOSED, 0.0
            elif now < state.open_until:
                resolved, remaining = OPEN, state.open_until - now
            else:
                resolved, remaining = HALF_OPEN, 0.0
            out.append(
                BreakerEntry(
                    key=key,
                    backend=state.backend,
                    endpoint=state.endpoint,
                    model_id=state.model_id,
                    state=resolved,
                    consecutive_failures=state.consecutive_failures,
                    cooldown_remaining_seconds=remaining,
                    opened_total=state.opened_total,
                )
            )
        out.sort(key=lambda e: e.key)
        return out

    def counters(self) -> dict[str, int]:
        with self._lock:
            return {
                "opened_total": self._counters.opened_total,
                "half_open_probes_total": self._counters.probes_admitted_total,
                "skipped_total": self._counters.skipped_total,
            }

    def reset(self) -> None:
        with self._lock:
            self._states.clear()
            self._counters = _Counters()


_singleton: UpstreamBreaker | None = None


def get_upstream_breaker() -> UpstreamBreaker:
    """Process-wide singleton — sharing the failure history is the point."""
    global _singleton
    if _singleton is None:
        _singleton = UpstreamBreaker()
    return _singleton


def descriptor_allows(descriptor: Any) -> bool:
    """Candidate-selection gate for descriptors with no probe of their own.

    ``ollama_http`` has no reachability probe (Ollama's ``/api/tags`` is the
    registry's own discovery call, not a per-model health check), so this is
    the seam that keeps an open Ollama deployment out of candidate selection.
    """
    return get_upstream_breaker().allows(descriptor_deployment_key(descriptor))


def breaker_span_attrs(adapter: Any) -> dict:
    """Span attributes naming the deployment and its breaker state.

    Duck-typed so the registry layer does not import the adapter package.
    Returns ``{}`` for adapters with no remote deployment, which keeps the
    in-process backends' spans exactly as they were.
    """
    getter = getattr(adapter, "upstream_deployment_key", None)
    key = getter() if callable(getter) else None
    if not key:
        return {}
    breaker = get_upstream_breaker()
    return {
        "llm.upstream.deployment": key,
        "llm.upstream.breaker.state": breaker.state(key),
    }


__all__ = [
    "CLOSED",
    "HALF_OPEN",
    "HTTP_FORMATS",
    "OPEN",
    "STATE_VALUES",
    "BreakerEntry",
    "UpstreamBreaker",
    "breaker_span_attrs",
    "deployment_key",
    "descriptor_allows",
    "descriptor_deployment_key",
    "get_upstream_breaker",
]
