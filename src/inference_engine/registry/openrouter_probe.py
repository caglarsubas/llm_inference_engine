"""Reachability probe for configured OpenRouter models.

Design
------

OpenRouter exposes a *single* ``/v1/models`` catalog that lists every model id
reachable for a given key. All configured OpenRouter descriptors share that one
upstream, so the original design — probe each descriptor with its own catalog
fetch — fanned N configured models into N identical ``GET /v1/models`` calls and
made availability flap whenever any one of them timed out (issues #36, #37).

This probe fetches the catalog **once per (endpoint, key)** and answers every
descriptor's ``loadable`` from that shared, cached id set. Two windows sit in
front of the upstream:

* a short **TTL** (``ttl_seconds``) so a burst of ``/v1/models`` calls / cold
  model resolutions reuses one fetch instead of one-per-descriptor;
* a longer **last-known-good** window (``last_known_good_seconds``) so a
  *transient* upstream failure (timeout, connection error, 5xx) keeps serving
  the previous successful catalog instead of pruning every configured model to
  ``model_not_found`` mid-benchmark (issue #37).

An *authoritative* upstream response is never masked by last-known-good: a 200
catalog that simply doesn't list a model yields ``upstream_model_missing``, and
a 4xx / bad-payload error surfaces as-is — the upstream answered, so we trust
it.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import httpx

from ..config import settings
from ..observability import get_logger
from .breaker import deployment_key, get_upstream_breaker
from .ollama import ModelDescriptor

log = get_logger("registry.openrouter_probe")


@dataclass(frozen=True)
class OpenRouterProbeResult:
    loadable: bool
    reason: str = ""
    detail: str = ""
    duration_ms: float = 0.0
    # True when the answer came from a last-known-good catalog after a transient
    # upstream failure rather than a fresh fetch. Purely observational — callers
    # treat ``loadable`` the same either way.
    stale: bool = False


ClientFactory = Callable[[str, httpx.Timeout, dict[str, str]], httpx.Client]


@dataclass(frozen=True)
class _CatalogFetch:
    """Outcome of one attempt to fetch an endpoint's ``/v1/models`` catalog."""

    ok: bool
    model_ids: frozenset[str]  # ids from this fetch (empty on failure)
    reason: str  # "" on success, else the failure reason
    detail: str
    duration_ms: float
    # Whether a failure is worth masking with last-known-good. Timeouts,
    # connection errors and 5xx are transient; 4xx / bad payloads are the
    # upstream authoritatively saying "no", so they are not.
    transient: bool = False


@dataclass
class _CatalogState:
    """Per-(endpoint, key) cache: the most recent attempt plus the last
    *successful* catalog, so a transient failure can fall back to it."""

    attempted_at: float  # monotonic time of the most recent fetch attempt
    last: _CatalogFetch  # the most recent attempt (ok or not)
    good_ids: frozenset[str] | None  # ids from the last successful fetch
    good_at: float | None  # monotonic time of that successful fetch


class OpenRouterProbe:
    """Cached, endpoint-deduped ``/v1/models`` probe for OpenRouter descriptors."""

    def __init__(
        self,
        *,
        timeout_seconds: float | None = None,
        ttl_seconds: float | None = None,
        last_known_good_seconds: float | None = None,
        client_factory: ClientFactory | None = None,
    ) -> None:
        self.timeout_seconds = (
            settings.openrouter_upstream_probe_timeout_seconds
            if timeout_seconds is None
            else timeout_seconds
        )
        self.ttl_seconds = (
            settings.openrouter_upstream_probe_ttl_seconds
            if ttl_seconds is None
            else ttl_seconds
        )
        self.last_known_good_seconds = (
            settings.openrouter_last_known_good_seconds
            if last_known_good_seconds is None
            else last_known_good_seconds
        )
        self._client_factory = client_factory or self._default_client_factory
        # One entry per (endpoint, key-length) — the shared catalog cache.
        self._catalog: dict[tuple[str, int], _CatalogState] = {}

    @staticmethod
    def _default_client_factory(
        base_url: str,
        timeout: httpx.Timeout,
        headers: dict[str, str],
    ) -> httpx.Client:
        return httpx.Client(base_url=base_url, timeout=timeout, headers=headers)

    def invalidate(self) -> None:
        self._catalog.clear()

    def probe(self, descriptor: ModelDescriptor) -> OpenRouterProbeResult:
        if descriptor.format != "openrouter":
            return OpenRouterProbeResult(loadable=True)

        endpoint = (descriptor.endpoint or settings.openrouter_endpoint).rstrip("/")
        model_id = str((descriptor.params or {}).get("model_id") or "")
        api_key = settings.openrouter_api_key.strip()
        if not api_key:
            return OpenRouterProbeResult(
                loadable=False,
                reason="openrouter_api_key_missing",
                detail="OPENROUTER_API_KEY is not set",
            )
        if not endpoint or not model_id:
            return OpenRouterProbeResult(
                loadable=False,
                reason="invalid_openrouter_descriptor",
                detail="OpenRouter descriptor must include endpoint and params['model_id']",
            )

        # Cooldown gate, checked without claiming the half-open trial. The
        # catalog fetch below is per-(endpoint, key) and shared by every
        # configured OpenRouter model, so it is not a health signal for one
        # deployment — it cannot serve as this breaker's half-open probe the
        # way the vLLM per-model probe can. The adapter's own generation
        # attempt is the trial, and it claims it on the hot path.
        breaker = get_upstream_breaker()
        deployment = deployment_key("openrouter", endpoint, model_id)
        if not breaker.allows(deployment):
            remaining = breaker.cooldown_remaining(deployment)
            return OpenRouterProbeResult(
                loadable=False,
                reason="upstream_cooldown",
                detail=(
                    f"deployment is in breaker cooldown for another "
                    f"{remaining:.1f}s after consecutive upstream failures"
                ),
            )

        state = self._catalog_for(endpoint, api_key)
        return self._result_for(model_id, state)

    # ------------------------------------------------------------------
    # catalog cache — one fetch per (endpoint, key), TTL + last-known-good
    # ------------------------------------------------------------------

    def _catalog_for(self, endpoint: str, api_key: str) -> _CatalogState:
        # ``len(api_key)`` stands in for "the key changed" without holding the
        # secret in a cache key.
        cache_key = (endpoint, len(api_key))
        now = time.monotonic()

        state = self._catalog.get(cache_key)
        if state is not None and (now - state.attempted_at) < self.ttl_seconds:
            return state  # still fresh — reuse without touching the upstream

        fetch = self._fetch_catalog(endpoint, api_key)
        if state is None:
            state = _CatalogState(
                attempted_at=now,
                last=fetch,
                good_ids=fetch.model_ids if fetch.ok else None,
                good_at=now if fetch.ok else None,
            )
        else:
            state.attempted_at = now
            state.last = fetch
            if fetch.ok:
                state.good_ids = fetch.model_ids
                state.good_at = now
        self._catalog[cache_key] = state

        self._log_fetch(endpoint, state, now)
        return state

    def _log_fetch(self, endpoint: str, state: _CatalogState, now: float) -> None:
        fetch = state.last
        base = {
            "endpoint": endpoint,
            "duration_ms": round(fetch.duration_ms, 2),
            "key_source": "openrouter-api-key",
        }
        if fetch.ok:
            log.info("openrouter_catalog.ok", n_models=len(fetch.model_ids), **base)
            return

        retained = (
            state.good_ids is not None
            and state.good_at is not None
            and fetch.transient
            and (now - state.good_at) < self.last_known_good_seconds
        )
        if retained:
            log.warning(
                "openrouter_catalog.stale_retained",
                reason=fetch.reason,
                detail=fetch.detail,
                age_s=round(now - (state.good_at or now), 1),
                **base,
            )
        else:
            log.warning(
                "openrouter_catalog.fail",
                reason=fetch.reason,
                detail=fetch.detail,
                **base,
            )

    def _result_for(self, model_id: str, state: _CatalogState) -> OpenRouterProbeResult:
        fetch = state.last
        duration = fetch.duration_ms

        if fetch.ok:
            return self._membership_result(model_id, fetch.model_ids, duration, stale=False)

        # Fetch failed. Fall back to the last successful catalog when the
        # failure is transient and still inside the grace window.
        now = time.monotonic()
        can_serve_lkg = (
            fetch.transient
            and state.good_ids is not None
            and state.good_at is not None
            and (now - state.good_at) < self.last_known_good_seconds
        )
        if can_serve_lkg:
            assert state.good_ids is not None
            return self._membership_result(model_id, state.good_ids, duration, stale=True)

        return OpenRouterProbeResult(
            loadable=False,
            reason=fetch.reason,
            detail=fetch.detail,
            duration_ms=duration,
        )

    @staticmethod
    def _membership_result(
        model_id: str,
        ids: frozenset[str],
        duration_ms: float,
        *,
        stale: bool,
    ) -> OpenRouterProbeResult:
        if model_id in ids:
            return OpenRouterProbeResult(
                loadable=True,
                detail="served from last-known-good catalog" if stale else "",
                duration_ms=duration_ms,
                stale=stale,
            )
        listed = ", ".join(sorted(ids)[:8]) if ids else "none"
        if len(ids) > 8:
            listed += ", ..."
        return OpenRouterProbeResult(
            loadable=False,
            reason="upstream_model_missing",
            detail=f"upstream did not list {model_id!r}; listed: {listed}",
            duration_ms=duration_ms,
            stale=stale,
        )

    def _fetch_catalog(self, endpoint: str, api_key: str) -> _CatalogFetch:
        timeout = httpx.Timeout(self.timeout_seconds)
        headers = {"Authorization": f"Bearer {api_key}"}
        t0 = time.perf_counter()
        try:
            with self._client_factory(endpoint, timeout, headers) as client:
                response = client.get("/v1/models")
                response.raise_for_status()
                payload = response.json()
            upstream_ids = self._extract_model_ids(payload)
        except httpx.TimeoutException as exc:
            return _CatalogFetch(
                ok=False,
                model_ids=frozenset(),
                reason="upstream_timeout",
                detail=str(exc).splitlines()[0][:240] if str(exc) else endpoint,
                duration_ms=(time.perf_counter() - t0) * 1000,
                transient=True,
            )
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            return _CatalogFetch(
                ok=False,
                model_ids=frozenset(),
                reason="upstream_http_error",
                detail=f"GET /v1/models returned HTTP {status}",
                duration_ms=(time.perf_counter() - t0) * 1000,
                # 5xx is a server-side blip worth masking; 4xx (auth, quota) is
                # authoritative and should prune.
                transient=status >= 500,
            )
        except httpx.RequestError as exc:
            return _CatalogFetch(
                ok=False,
                model_ids=frozenset(),
                reason="upstream_unreachable",
                detail=str(exc).splitlines()[0][:240] if str(exc) else endpoint,
                duration_ms=(time.perf_counter() - t0) * 1000,
                transient=True,
            )
        except ValueError as exc:
            return _CatalogFetch(
                ok=False,
                model_ids=frozenset(),
                reason="upstream_bad_models_response",
                detail=str(exc).splitlines()[0][:240] if str(exc) else "invalid JSON",
                duration_ms=(time.perf_counter() - t0) * 1000,
                transient=False,
            )

        return _CatalogFetch(
            ok=True,
            model_ids=frozenset(upstream_ids),
            reason="",
            detail="",
            duration_ms=(time.perf_counter() - t0) * 1000,
            transient=False,
        )

    @staticmethod
    def _extract_model_ids(payload: Any) -> list[str]:
        if not isinstance(payload, dict):
            raise ValueError("upstream /v1/models response must be an object")
        data = payload.get("data")
        if not isinstance(data, list):
            raise ValueError("upstream /v1/models response missing data[]")
        ids: list[str] = []
        for entry in data:
            if isinstance(entry, dict) and entry.get("id"):
                ids.append(str(entry["id"]))
        return ids


_singleton: OpenRouterProbe | None = None


def get_openrouter_probe() -> OpenRouterProbe:
    global _singleton
    if _singleton is None:
        _singleton = OpenRouterProbe()
    return _singleton


__all__ = ["OpenRouterProbe", "OpenRouterProbeResult", "get_openrouter_probe"]
