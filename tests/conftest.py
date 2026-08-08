"""Shared pytest fixtures across the test suite.

The OTel SDK only allows one TracerProvider per process — if multiple test
files install their own in-memory exporter at session scope, the second wins
the TracerProvider but the first ends up with a dangling exporter that
captures nothing. Owning the fixture here at the test-package root means
every test module gets the same exporter instance.
"""

from __future__ import annotations

import atexit
import os
import shutil
import tempfile
from pathlib import Path

import pytest


def _isolate_model_stores() -> None:
    """Point the model-store settings at an empty scratch store for this run.

    ``inference_engine.config.settings`` is built at import time and
    ``inference_engine.api.state`` constructs ``OllamaRegistry`` at module
    scope, which raises when the store is missing — so this has to happen
    before the first ``inference_engine`` import, not in a fixture. An
    ``OLLAMA_MODELS_DIR`` / ``MLX_MODELS_DIR`` already in the environment
    always wins; anything in ``.env`` does not.
    """
    if "OLLAMA_MODELS_DIR" in os.environ and "MLX_MODELS_DIR" in os.environ:
        return
    root = Path(tempfile.mkdtemp(prefix="inference-engine-tests-"))
    atexit.register(shutil.rmtree, root, True)
    (root / "ollama" / "manifests").mkdir(parents=True)
    (root / "ollama" / "blobs").mkdir(parents=True)
    (root / "mlx").mkdir()
    os.environ.setdefault("OLLAMA_MODELS_DIR", str(root / "ollama"))
    os.environ.setdefault("MLX_MODELS_DIR", str(root / "mlx"))


_isolate_model_stores()

from inference_engine import otel as otel_mod  # noqa: E402 — see _isolate_model_stores()


@pytest.fixture(scope="session")
def _session_exporter():
    return otel_mod._install_in_memory_exporter()


@pytest.fixture(autouse=True)
def _reset_models_snapshot():
    """Keep the process-wide /v1/models snapshot cache from leaking across tests.

    The cache is a module global written by the background refresher (started in
    the ASGI lifespan). Tests that exercise the lifespan would otherwise leave a
    populated snapshot behind, and later tests that call the route functions
    directly expect a fresh probe pass. Clear it around every test.
    """
    from inference_engine.api import models as models_api

    models_api.snapshot_cache.clear()
    yield
    models_api.snapshot_cache.clear()
