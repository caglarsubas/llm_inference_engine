"""Runtime probe for MLX descriptors — can the in-process adapter actually load this?

Why this exists
---------------

``MLXRegistry`` reports every directory that looks HuggingFace-shaped
(``config.json`` + safetensors + tokenizer). That shape is necessary but not
sufficient: the in-process MLX adapter is a **text** adapter built on the
optional ``mlx-lm`` package, and two independent things can make a scanned
directory unservable by it.

1. **The runtime isn't installed.** ``mlx-lm`` lives in the ``mlx`` extra
   (``pyproject.toml``), so a deployment that ran a plain ``uv sync`` has no
   ``mlx_lm`` module at all. Every ``:mlx`` descriptor is then a phantom
   option — ``/v1/models`` advertises it, and the first chat call dies with
   ``ModuleNotFoundError`` inside ``adapter.load()``.

2. **The checkpoint is a vision-language model.** ``mlx-lm`` only loads
   causal-LM architectures. VLM checkpoints (``molmo``, ``glm4v``, …) carry a
   ``vision_config`` and a ``*ForConditionalGeneration`` architecture, and
   they are served here by the sidecar workers in ``scripts/`` —
   ``serve_mlx_vlm_openai.py`` and ``serve_molmo_mlx_openai.py`` — which
   expose an OpenAI-compatible surface that the engine consumes as a *vllm*
   descriptor via ``.vllm_models.json``. Advertising the raw weight directory
   as ``:mlx`` points callers at an adapter that would silently drop the image
   parts even if it could load the weights.

Installing the ``mlx`` extra fixes (1) and never fixes (2), so the two are
reported as distinct reasons rather than one "mlx broken" bucket.

This mirrors ``GGUFLoadProbe``: keep the phantom out of ``data`` and give the
caller a structured reason in ``unavailable[]`` instead of a 500 on first use.

Cost
----

No weights are read and ``mlx_lm`` is never imported — the probe parses
``config.json`` (a few KB) and asks ``importlib`` whether modules exist.
Architecture support is resolved the same way ``mlx-lm`` itself resolves it,
by looking for ``mlx_lm.models.<model_type>``.

Caching
-------

Results are keyed by ``(config path, mtime_ns, size)`` so re-downloading a
checkpoint re-probes automatically, while repeated ``/v1/models`` calls on a
stable directory do not.
"""

from __future__ import annotations

import importlib.util
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

from ..observability import get_logger
from .ollama import ModelDescriptor

log = get_logger("registry.mlx_probe")

# The module the in-process adapter imports in ``MLXAdapter.load``.
MLX_RUNTIME_MODULE = "mlx_lm"

# Config keys that mark a checkpoint as vision-language. ``vision_config`` is
# the reliable one across Molmo / GLM-4V / Qwen-VL / InternVL; the architecture
# suffix catches conditional-generation wrappers that nest their vision tower
# elsewhere.
_VISION_CONFIG_KEYS = ("vision_config", "vision_tower_config")
_VISION_ARCH_SUFFIXES = ("ForConditionalGeneration", "ForVisionText2Text")


@dataclass(frozen=True)
class MLXProbeResult:
    loadable: bool
    reason: str = ""
    detail: str = ""
    duration_ms: float = 0.0
    # Parsed ``model_type`` from config.json ("" when unreadable). Kept so
    # callers can report what the checkpoint actually is without re-reading.
    model_type: str = ""


def _is_vision_checkpoint(config: dict) -> bool:
    if any(key in config for key in _VISION_CONFIG_KEYS):
        return True
    architectures = config.get("architectures")
    if isinstance(architectures, list):
        for arch in architectures:
            if isinstance(arch, str) and arch.endswith(_VISION_ARCH_SUFFIXES):
                return True
    return False


class MLXRuntimeProbe:
    """Cached, import-free check that an MLX directory is servable in-process."""

    def __init__(self) -> None:
        self._cache: dict[tuple[str, int, int], MLXProbeResult] = {}

    @staticmethod
    def _cache_key(config_path: Path) -> tuple[str, int, int] | None:
        try:
            st = os.stat(config_path)
        except OSError:
            return None
        return (str(config_path), st.st_mtime_ns, st.st_size)

    @staticmethod
    def _runtime_installed() -> bool:
        # find_spec, not import: the probe runs on the /v1/models path and
        # importing mlx_lm drags in the whole Metal stack.
        try:
            return importlib.util.find_spec(MLX_RUNTIME_MODULE) is not None
        except (ImportError, ValueError):
            return False

    @staticmethod
    def _architecture_supported(model_type: str) -> bool:
        if not model_type:
            # No model_type to check. mlx-lm would fall back to its own
            # defaulting; don't invent a rejection the loader wouldn't make.
            return True
        try:
            return importlib.util.find_spec(f"{MLX_RUNTIME_MODULE}.models.{model_type}") is not None
        except (ImportError, ValueError):
            return False

    def probe(self, descriptor: ModelDescriptor) -> MLXProbeResult:
        if descriptor.format != "mlx":
            return MLXProbeResult(loadable=True)

        config_path = Path(descriptor.model_path) / "config.json"
        key = self._cache_key(config_path)
        if key is None:
            return MLXProbeResult(
                loadable=False,
                reason="mlx_config_unreadable",
                detail=str(config_path),
            )

        cached = self._cache.get(key)
        if cached is not None:
            return cached

        result = self._probe(descriptor, config_path)
        self._cache[key] = result

        log.info(
            "mlx_probe",
            model=descriptor.qualified_name,
            loadable=result.loadable,
            reason=result.reason,
            model_type=result.model_type,
            duration_ms=round(result.duration_ms, 2),
        )
        return result

    def _probe(self, descriptor: ModelDescriptor, config_path: Path) -> MLXProbeResult:
        started = time.perf_counter()

        def done(**kwargs) -> MLXProbeResult:
            return MLXProbeResult(duration_ms=(time.perf_counter() - started) * 1000.0, **kwargs)

        try:
            with config_path.open("r", encoding="utf-8") as fh:
                config = json.load(fh)
        except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            return done(
                loadable=False,
                reason="mlx_config_unreadable",
                detail=f"{config_path}: {exc}",
            )
        if not isinstance(config, dict):
            return done(
                loadable=False,
                reason="mlx_config_unreadable",
                detail=f"{config_path}: top level is not an object",
            )

        model_type = str(config.get("model_type") or "")

        # Vision first, and before the runtime check: for a VLM checkpoint
        # "install the mlx extra" is the wrong next step, so reporting
        # mlx_runtime_missing here would send the operator down a dead end.
        if _is_vision_checkpoint(config):
            return done(
                loadable=False,
                reason="mlx_vision_unsupported",
                detail=(
                    f"{model_type or 'checkpoint'} is a vision-language model; the in-process "
                    "mlx adapter is text-only. Serve it with scripts/serve_mlx_vlm_openai.py "
                    "(or scripts/serve_molmo_mlx_openai.py) and register that worker's "
                    "endpoint in .vllm_models.json."
                ),
                model_type=model_type,
            )

        if not self._runtime_installed():
            return done(
                loadable=False,
                reason="mlx_runtime_missing",
                detail=(
                    f"{MLX_RUNTIME_MODULE} is not installed in this environment; "
                    "install the 'mlx' extra (uv sync --extra mlx) to serve mlx models."
                ),
                model_type=model_type,
            )

        if not self._architecture_supported(model_type):
            return done(
                loadable=False,
                reason="mlx_architecture_unsupported",
                detail=(
                    f"{MLX_RUNTIME_MODULE} has no loader for model_type "
                    f"{model_type!r} ({MLX_RUNTIME_MODULE}.models.{model_type} not found)."
                ),
                model_type=model_type,
            )

        return done(loadable=True, model_type=model_type)


_singleton: MLXRuntimeProbe | None = None


def get_mlx_probe() -> MLXRuntimeProbe:
    global _singleton
    if _singleton is None:
        _singleton = MLXRuntimeProbe()
    return _singleton


__all__ = ["MLXProbeResult", "MLXRuntimeProbe", "get_mlx_probe"]
