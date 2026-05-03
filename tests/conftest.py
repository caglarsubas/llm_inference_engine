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
