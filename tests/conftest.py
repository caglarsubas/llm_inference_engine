"""Shared pytest fixtures across the test suite.

The OTel SDK only allows one TracerProvider per process — if multiple test
files install their own in-memory exporter at session scope, the second wins
the TracerProvider but the first ends up with a dangling exporter that
captures nothing. Owning the fixture here at the test-package root means
every test module gets the same exporter instance.
"""

from __future__ import annotations

import pytest

from inference_engine import otel as otel_mod


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
