"""Reachability probe for OpenAI-compatible vLLM upstreams.

The vLLM registry is config-driven: an operator writes `.vllm_models.json`
after starting a separate model server. A config entry alone is not enough to
promise that the model is callable, so `/v1/models` checks the upstream's own
`/v1/models` endpoint before moving the descriptor into `data`.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import httpx

from ..config import settings
from ..observability import get_logger
from .ollama import ModelDescriptor

log = get_logger("registry.vllm_probe")


@dataclass(frozen=True)
class VLLMProbeResult:
    loadable: bool
    reason: str = ""
    detail: str = ""
    duration_ms: float = 0.0


ClientFactory = Callable[[str, httpx.Timeout], httpx.Client]


class VLLMUpstreamProbe:
    """Cached `/v1/models` probe for vLLM/OpenAI-compatible HTTP backends."""

    def __init__(
        self,
        *,
        timeout_seconds: float | None = None,
        ttl_seconds: float | None = None,
        client_factory: ClientFactory | None = None,
    ) -> None:
        self.timeout_seconds = (
            settings.vllm_upstream_probe_timeout_seconds
            if timeout_seconds is None
            else timeout_seconds
        )
        self.ttl_seconds = (
            settings.vllm_upstream_probe_ttl_seconds if ttl_seconds is None else ttl_seconds
        )
        self._client_factory = client_factory or self._default_client_factory
        self._cache: dict[tuple[str, str], tuple[float, VLLMProbeResult]] = {}

    @staticmethod
    def _default_client_factory(base_url: str, timeout: httpx.Timeout) -> httpx.Client:
        return httpx.Client(base_url=base_url, timeout=timeout)

    def invalidate(self) -> None:
        self._cache.clear()

    def probe(self, descriptor: ModelDescriptor) -> VLLMProbeResult:
        if descriptor.format != "vllm":
            return VLLMProbeResult(loadable=True)

        endpoint = (descriptor.endpoint or "").rstrip("/")
        model_id = str((descriptor.params or {}).get("model_id") or "")
        if not endpoint or not model_id:
            return VLLMProbeResult(
                loadable=False,
                reason="invalid_vllm_descriptor",
                detail="vLLM descriptor must include endpoint and params['model_id']",
            )

        key = (endpoint, model_id)
        now = time.monotonic()
        cached = self._cache.get(key)
        if cached is not None:
            created_at, result = cached
            if now - created_at < self.ttl_seconds:
                return result

        result = self._probe_upstream(endpoint, model_id)
        self._cache[key] = (now, result)

        log_kwargs = {
            "model": descriptor.qualified_name,
            "endpoint": endpoint,
            "model_id": model_id,
            "duration_ms": round(result.duration_ms, 2),
        }
        if result.loadable:
            log.info("vllm_probe.ok", **log_kwargs)
        else:
            log.warning(
                "vllm_probe.fail",
                reason=result.reason,
                detail=result.detail,
                **log_kwargs,
            )
        return result

    def _probe_upstream(self, endpoint: str, model_id: str) -> VLLMProbeResult:
        timeout = httpx.Timeout(self.timeout_seconds)
        t0 = time.perf_counter()
        try:
            with self._client_factory(endpoint, timeout) as client:
                response = client.get("/v1/models")
                response.raise_for_status()
                payload = response.json()
            upstream_ids = self._extract_model_ids(payload)
        except httpx.TimeoutException as exc:
            return VLLMProbeResult(
                loadable=False,
                reason="upstream_timeout",
                detail=str(exc).splitlines()[0][:240] if str(exc) else endpoint,
                duration_ms=(time.perf_counter() - t0) * 1000,
            )
        except httpx.HTTPStatusError as exc:
            return VLLMProbeResult(
                loadable=False,
                reason="upstream_http_error",
                detail=f"GET /v1/models returned HTTP {exc.response.status_code}",
                duration_ms=(time.perf_counter() - t0) * 1000,
            )
        except httpx.RequestError as exc:
            return VLLMProbeResult(
                loadable=False,
                reason="upstream_unreachable",
                detail=str(exc).splitlines()[0][:240] if str(exc) else endpoint,
                duration_ms=(time.perf_counter() - t0) * 1000,
            )
        except ValueError as exc:
            return VLLMProbeResult(
                loadable=False,
                reason="upstream_bad_models_response",
                detail=str(exc).splitlines()[0][:240] if str(exc) else "invalid JSON",
                duration_ms=(time.perf_counter() - t0) * 1000,
            )

        if model_id in upstream_ids:
            return VLLMProbeResult(
                loadable=True,
                duration_ms=(time.perf_counter() - t0) * 1000,
            )

        listed = ", ".join(upstream_ids[:8]) if upstream_ids else "none"
        if len(upstream_ids) > 8:
            listed += ", ..."
        return VLLMProbeResult(
            loadable=False,
            reason="upstream_model_missing",
            detail=f"upstream did not list {model_id!r}; listed: {listed}",
            duration_ms=(time.perf_counter() - t0) * 1000,
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


_singleton: VLLMUpstreamProbe | None = None


def get_vllm_probe() -> VLLMUpstreamProbe:
    global _singleton
    if _singleton is None:
        _singleton = VLLMUpstreamProbe()
    return _singleton


__all__ = ["VLLMProbeResult", "VLLMUpstreamProbe", "get_vllm_probe"]
