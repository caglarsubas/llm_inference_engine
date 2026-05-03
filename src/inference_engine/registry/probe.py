"""GGUF load-probe — verify a manifest can actually be opened by llama.cpp.

Why this exists
---------------

``OllamaRegistry`` reports every manifest with a present blob, but llama.cpp
silently lags behind the model architectures ollama can pull (e.g. a 2026
``gemma4`` GGUF shipped before ``llama-cpp-python`` recognised the arch).
Surfacing those models in ``/v1/models`` only to 500 on first chat call
gives downstream callers (DeclarAI, agentic-hook-v2 dashboards) phantom
options that look broken.

The probe runs a **vocab-only** load (``Llama(vocab_only=True)``) which
opens the GGUF and parses the header / metadata / vocab without allocating
weight tensors.  Cost: ~50-200 ms per model on a laptop CPU; weight bytes
are mmapped, not read, so even a 30 GB model is cheap.  Failure surfaces
as a ``ValueError`` with the same wording chat would emit, so we capture
that as the unavailability reason.

Caching
-------

Results are keyed by ``(path, mtime_ns, size)`` so a re-pull of the same
qualified_name (which changes blob digest → path) re-probes automatically,
while repeated ``/v1/models`` calls on a stable cache do not.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..observability import get_logger
from .ollama import ModelDescriptor

log = get_logger("registry.probe")


@dataclass(frozen=True)
class ProbeResult:
    loadable: bool
    reason: str = ""
    detail: str = ""
    duration_ms: float = 0.0


class GGUFLoadProbe:
    """Lazy, cached vocab-only load probe for GGUF descriptors.

    Non-GGUF descriptors (mlx, vllm) are always reported as ``loadable=True``
    — their availability is checked elsewhere (MLX adapter import-time, vLLM
    upstream health check) and we don't want to drag llama-cpp into those
    code paths.
    """

    def __init__(self) -> None:
        self._cache: dict[tuple[str, int, int], ProbeResult] = {}

    @staticmethod
    def _cache_key(path: Path) -> tuple[str, int, int] | None:
        try:
            st = os.stat(path)
        except OSError:
            return None
        return (str(path), st.st_mtime_ns, st.st_size)

    def probe(self, descriptor: ModelDescriptor) -> ProbeResult:
        if descriptor.format != "gguf":
            return ProbeResult(loadable=True)

        key = self._cache_key(descriptor.model_path)
        if key is None:
            return ProbeResult(
                loadable=False,
                reason="missing_blob",
                detail=str(descriptor.model_path),
            )

        cached = self._cache.get(key)
        if cached is not None:
            return cached

        result = self._probe_load(descriptor)
        self._cache[key] = result

        log_kwargs = {
            "model": descriptor.qualified_name,
            "duration_ms": round(result.duration_ms, 2),
        }
        if result.loadable:
            log.info("probe.ok", **log_kwargs)
        else:
            log.warning(
                "probe.fail",
                reason=result.reason,
                detail=result.detail,
                **log_kwargs,
            )
        return result

    def _probe_load(self, descriptor: ModelDescriptor) -> ProbeResult:
        """Run a faithful load probe via the high-level ``Llama`` class.

        Why not ``vocab_only=True``?  Because architecture support in
        llama.cpp is checked *during weight-tensor materialisation*, not
        during vocab parsing — a 2026 ``gemma4`` GGUF parses its vocab fine
        but rejects on the unknown attention layout.  A vocab-only probe
        would happily report it ``loadable`` and we'd 500 on first chat.

        So the probe loads the model the same way the chat adapter does
        (mmap + arch graph build), but with a minimal ``n_ctx`` so the KV
        cache allocation is negligible.  Cost on cached pages: ~tens of ms
        for small models, low seconds for 30 GB models — paid once per
        ``(path, mtime)`` then cached.  We then drop the handle so the
        adapter's hot path is unaffected.

        ``n_ctx=128`` is the smallest value that survives every chat
        template's prompt-size lower bound; smaller values trip
        ``ggml_check_dims`` on some models.
        """
        from llama_cpp import Llama  # noqa: PLC0415 — heavy native module

        t0 = time.perf_counter()
        try:
            llm = Llama(
                model_path=str(descriptor.model_path),
                n_gpu_layers=0,
                n_ctx=128,
                n_batch=128,
                vocab_only=False,
                use_mmap=True,
                verbose=False,
            )
            del llm
            return ProbeResult(loadable=True, duration_ms=(time.perf_counter() - t0) * 1000)
        except Exception as exc:  # noqa: BLE001 — llama-cpp raises ValueError, RuntimeError, OSError
            return ProbeResult(
                loadable=False,
                reason=type(exc).__name__,
                detail=str(exc).splitlines()[0][:240] if str(exc) else "",
                duration_ms=(time.perf_counter() - t0) * 1000,
            )

    def probe_all(self, descriptors: list[ModelDescriptor]) -> dict[str, ProbeResult]:
        """Probe every descriptor. Returns ``{qualified_name: result}``."""
        return {d.qualified_name: self.probe(d) for d in descriptors}

    def invalidate(self) -> None:
        self._cache.clear()


_singleton: GGUFLoadProbe | None = None


def get_probe() -> GGUFLoadProbe:
    """Process-wide singleton — sharing the cache across requests is the point."""
    global _singleton
    if _singleton is None:
        _singleton = GGUFLoadProbe()
    return _singleton


# Re-export for convenience.
__all__ = ["GGUFLoadProbe", "ProbeResult", "get_probe"]


# A small helper used by Any-typed callers (the API layer) so they don't
# need to import the dataclass.
def as_dict(result: ProbeResult) -> dict[str, Any]:
    return {
        "loadable": result.loadable,
        "reason": result.reason,
        "detail": result.detail,
        "duration_ms": result.duration_ms,
    }
