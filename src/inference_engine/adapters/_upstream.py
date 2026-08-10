"""Shared deployment plumbing for adapters that proxy an HTTP upstream.

Three adapters — vLLM, Ollama-HTTP and OpenRouter (a vLLM subclass) — all talk
OpenAI-shaped HTTP to a remote server. This mixin gives them one identity for
that server (the *deployment*), one place to claim it before a call, and one
place to report what happened, so retries (``_upstream_retry``) and the
breaker (``registry.breaker``) see the same events from all three.

The mixin owns admission and accounting only. Retry *loops* stay in the
adapters because their shapes differ in a way that matters: a blocking POST
can be reissued wholesale, while a stream can only be reissued before its
first chunk reaches the consumer, and only the adapter knows when that is.
"""

from __future__ import annotations

from ..registry.breaker import (
    OPEN,
    deployment_key,
    get_upstream_breaker,
)
from ._upstream_retry import is_retryable
from .base import UpstreamGenerationError


def is_health_signal(exc: BaseException) -> bool:
    """Does this failure say something about the DEPLOYMENT's health?

    Exactly the set :func:`_upstream_retry.is_retryable` accepts. That the two
    predicates coincide is a decision, not an accident, and the decision worth
    writing down is the one about **timeouts**.

    A TIMEOUT IS OUR BUDGET RUNNING OUT, NOT A VERDICT ON THE UPSTREAM.
    ``CHAT_COMPLETION_TIMEOUT_SECONDS`` bounds elapsed time, and elapsed time
    does not distinguish a wedged deployment from a healthy one generating a
    long answer — a 4k-token completion and a hung socket both end as
    ``TimeoutError``. An earlier revision of this function counted every
    timeout, and once the streaming path grew its own deadline that meant three
    slow-but-successful streams opened the breaker and withdrew a working model
    from every tenant. So a deadline expiry, an ``httpx.ReadTimeout``, a
    ``WriteTimeout`` and a ``PoolTimeout`` are all discounted here.

    ``httpx.ConnectTimeout`` is the exception, and for a reason that is about
    evidence rather than about clocks: it proves the upstream never accepted
    the work at all. Alongside it, what opens a deployment is the transport
    errors that left no response behind (``ConnectError``, ``ReadError``,
    ``WriteError``, ``RemoteProtocolError`` — the failure classes, not their
    ``*Timeout`` namesakes) and the transient status codes 408, 425, 429, 500,
    502, 503, 504 and 529.

    THE LIMIT THIS LEAVES, stated plainly because it is the cost of the rule
    above: a deployment that accepts connections and then goes silent is never
    opened *from the hot path*, because from here it is indistinguishable from
    a long generation. Telling the two apart needs a signal elapsed generation
    time cannot carry — a cheap call with a bound of its own — and one exists
    for vLLM: ``VLLMUpstreamProbe`` GETs ``/v1/models`` under
    ``VLLM_UPSTREAM_PROBE_TIMEOUT_SECONDS`` (1 s) and does feed its timeouts to
    the breaker. ``ollama_http`` has no such probe, so a silently wedged Ollama
    deployment stays in candidate selection and every request to it spends its
    own full budget.

    In the other direction the narrowing is unchanged: a rejected request — a
    400, a 404, a 422 — is never counted against the deployment, so the
    upstream's own rejections cannot cool a healthy deployment down.
    """
    return is_retryable(exc)


def _failure_reason(exc: BaseException) -> str:
    status = getattr(getattr(exc, "response", None), "status_code", None)
    if status is not None:
        return f"upstream_http_{status}"
    return type(exc).__name__


class HttpUpstreamMixin:
    """Deployment identity + breaker admission for HTTP-backed adapters."""

    backend_name: str = "http"

    _deployment_key: str | None = None
    _deployment_endpoint: str = ""
    _deployment_model_id: str = ""

    # ------------------------------------------------------------------
    # identity
    # ------------------------------------------------------------------

    def _bind_deployment(self, endpoint: str | None, model_id: str | None) -> None:
        self._deployment_endpoint = (endpoint or "").rstrip("/")
        self._deployment_model_id = model_id or ""
        self._deployment_key = deployment_key(
            self.backend_name, self._deployment_endpoint, self._deployment_model_id
        )

    def _clear_deployment(self) -> None:
        self._deployment_key = None
        self._deployment_endpoint = ""
        self._deployment_model_id = ""

    def upstream_deployment_key(self) -> str | None:
        return self._deployment_key

    # ------------------------------------------------------------------
    # admission + accounting
    # ------------------------------------------------------------------

    def _begin_upstream(self) -> None:
        """Claim the deployment for one logical call, or refuse to make it.

        Candidate selection already skips open deployments through the probe
        layer, so reaching here with an open breaker means the state changed
        between resolution and the call. Raising the same typed upstream error
        the route already maps keeps that race on the fallback path rather than
        letting it through to an upstream we know is failing.
        """
        if get_upstream_breaker().begin_attempt(self._deployment_key) != OPEN:
            return
        remaining = get_upstream_breaker().cooldown_remaining(self._deployment_key)
        raise UpstreamGenerationError(
            message=(
                "Upstream deployment is in cooldown after consecutive failures."
            ),
            error_type="upstream_cooldown",
            backend=self.backend_name,
            model=self._deployment_model_id,
            detail=f"cooldown has {remaining:.1f}s left",
        )

    def _finish_upstream(self, exc: BaseException | None) -> None:
        breaker = get_upstream_breaker()
        if exc is None:
            breaker.record_success(self._deployment_key)
            return
        if not is_health_signal(exc):
            # Says nothing about the deployment: release any half-open trial
            # without spending it, so the next caller can still probe.
            breaker.abandon_attempt(self._deployment_key)
            return
        breaker.record_failure(
            self._deployment_key,
            reason=_failure_reason(exc),
            backend=self.backend_name,
            endpoint=self._deployment_endpoint,
            model_id=self._deployment_model_id,
        )

    def _abandon_upstream(self) -> None:
        get_upstream_breaker().abandon_attempt(self._deployment_key)


__all__ = ["HttpUpstreamMixin", "is_health_signal"]
