"""Discover models served by an Ollama HTTP server.

Why this exists alongside ``OllamaRegistry``
--------------------------------------------

``OllamaRegistry`` walks the on-disk Ollama blob store and emits
``format="gguf"`` descriptors that go through our local llama-cpp-python
adapter.  That's the fast path: in-process, no network hop, full prefix-cache
introspection.  Problem: llama-cpp-python's bundled llama.cpp lags new model
architectures by weeks.  In 2026 we ship a wheel that can't open
``gemma4`` / ``qwen3.6`` / ``ministral-3`` / ``nemotron3`` GGUFs, even though
Ollama's own ggml fork supports them today.

This registry is the "ask Ollama" fallback.  It lists what an Ollama HTTP
server reports via ``GET /api/tags`` and emits ``format="ollama_http"``
descriptors pointing at that server.  Combined with the load probe in
``probe.py`` and the ``CompositeRegistry`` fallback path, the manager picks
local llama.cpp when it works and Ollama when it doesn't — best of both
without operators having to know the seam.

The registry is lazy and cached
-------------------------------

``GET /api/tags`` is cheap (Ollama returns a static list off its own
on-disk index), but we still cache the response inside the process so a
client hammering ``/v1/models`` doesn't translate 1:1 into Ollama traffic.
The cache is invalidated on a TTL (``_TTL_SECONDS``) and on explicit
``invalidate()``; the fallback should reflect ``ollama pull`` events within
~30 s without a restart.

Reachability failures (Ollama not running, network blip) return an empty
list rather than raising.  The composite handles that as "fallback offered
nothing for that id" and the model goes to ``unavailable`` — consistent
with how the rest of the registry layer degrades.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path

from ..observability import get_logger
from .ollama import ModelDescriptor

log = get_logger("registry.ollama_http")

# Short TTL — discovery should react quickly to ``ollama pull`` without
# saturating the upstream with /api/tags requests in a hot loop.
_TTL_SECONDS = 30.0
_FETCH_TIMEOUT = 3.0


class OllamaHttpRegistry:
    def __init__(self, endpoint: str) -> None:
        self._endpoint = endpoint.rstrip("/") if endpoint else ""
        self._cache: dict[str, ModelDescriptor] = {}
        self._cache_expires_at: float = 0.0

    @property
    def endpoint(self) -> str:
        return self._endpoint

    def is_configured(self) -> bool:
        return bool(self._endpoint)

    # ------------------------------------------------------------------
    # discovery
    # ------------------------------------------------------------------

    def _fetch_tags(self) -> list[dict] | None:
        """Hit ``GET /api/tags`` and return the ``models`` list, or None on error.

        ``None`` (not ``[]``) communicates "couldn't ask" so callers can keep
        serving a stale cache rather than wiping the fallback's view of the
        world during a transient blip.
        """
        if not self._endpoint:
            return None
        url = self._endpoint + "/api/tags"
        try:
            with urllib.request.urlopen(url, timeout=_FETCH_TIMEOUT) as resp:
                body = resp.read().decode("utf-8")
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            log.warning("ollama_http.tags_fetch_failed", endpoint=self._endpoint, error=str(exc))
            return None
        try:
            payload = json.loads(body)
        except json.JSONDecodeError as exc:
            log.warning("ollama_http.tags_malformed", endpoint=self._endpoint, error=str(exc))
            return None
        return list(payload.get("models") or [])

    def _refresh(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now < self._cache_expires_at and self._cache:
            return
        models = self._fetch_tags()
        if models is None:
            # Back off briefly on failure so we don't hammer a flapping
            # ollama instance.  Keep the previous cache (possibly empty).
            self._cache_expires_at = now + min(_TTL_SECONDS, 5.0)
            return

        new_cache: dict[str, ModelDescriptor] = {}
        for entry in models:
            qname = str(entry.get("name") or entry.get("model") or "").strip()
            if not qname:
                continue
            # Ollama's ``name`` is already ``model:tag`` — split on the first
            # colon.  Tag-less entries fall back to "latest" the same way
            # Ollama itself does.
            if ":" in qname:
                model_name, tag = qname.split(":", 1)
            else:
                model_name, tag = qname, "latest"

            size = int(entry.get("size") or 0)
            digest = str(entry.get("digest") or "")

            # Skip Ollama Cloud aliases — manifests that resolve to hosted
            # inference rather than local weights.  Two heuristics, both
            # required for a robust skip:
            #   1. ``size == 0`` — Ollama reports zero bytes for cloud-only
            #      entries because there's no local blob.
            #   2. ``tag == "cloud"`` — the conventional Ollama tag for
            #      cloud-hosted variants (``minimax-m2.7:cloud``).
            # On-prem deployments don't have (and shouldn't need) the
            # OLLAMA_API_KEY required to reach those, and surfacing them in
            # /v1/models would give DeclarAI a phantom option that 500s on
            # call.  We log the skip so operators see why a manifest they
            # can see in ``ollama list`` is missing from the engine.
            if size <= 0 or tag == "cloud":
                log.info(
                    "ollama_http.skip_cloud_alias",
                    model=qname,
                    size_bytes=size,
                    reason="cloud_alias_no_local_weights",
                )
                continue
            desc = ModelDescriptor(
                name=model_name,
                tag=tag,
                namespace="library",
                registry="registry.ollama.ai",
                # Synthetic locator so callers reading ``model_path`` for
                # display don't trip on None.  Real lookup goes through
                # ``endpoint``.
                model_path=Path(f"ollama_http://{self._endpoint}/{qname}"),
                format="ollama_http",
                params={"model_id": qname, "digest": digest},
                size_bytes=size,
                endpoint=self._endpoint,
            )
            new_cache[desc.qualified_name] = desc
            new_cache[desc.fully_qualified_name] = desc

        self._cache = new_cache
        self._cache_expires_at = now + _TTL_SECONDS

    # ------------------------------------------------------------------
    # public surface — same shape as the other registries
    # ------------------------------------------------------------------

    def list_models(self) -> list[ModelDescriptor]:
        self._refresh()
        # De-dupe — qualified_name and fully_qualified_name both point at the
        # same descriptor in the cache.
        seen: dict[str, ModelDescriptor] = {}
        for desc in self._cache.values():
            seen.setdefault(desc.qualified_name, desc)
        return sorted(seen.values(), key=lambda d: d.qualified_name)

    def get(self, name_with_tag: str) -> ModelDescriptor | None:
        self._refresh()
        if name_with_tag in self._cache:
            return self._cache[name_with_tag]
        if ":" not in name_with_tag:
            for desc in self._cache.values():
                if desc.name == name_with_tag:
                    return desc
        return None

    def invalidate(self) -> None:
        self._cache_expires_at = 0.0
