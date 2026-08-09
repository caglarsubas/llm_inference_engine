"""The capability registry: what counts as evidence, and which way it flows.

Every test here guards one property — the registry may only ever become LESS
trusting, and only on proof. A registry that could promote, or that demoted on
ambiguous evidence, would replace a wrong static claim with a wrong dynamic one.
"""

from __future__ import annotations

import pytest

from inference_engine import structured_output_capability as capability
from inference_engine.adapters.base import InferenceAdapter


class _Adapter(InferenceAdapter):
    backend_name = "probe_backend"

    def __init__(self, *, declares: bool, host: str | None = None) -> None:
        self.supports_structured_outputs = declares
        self._host = host

    def deployment_id(self) -> str:
        return f"{self.backend_name}:{self._host}" if self._host else self.backend_name

    @property
    def is_loaded(self) -> bool:
        return True

    @property
    def loaded_model(self):
        return None

    async def load(self, descriptor) -> None: ...
    async def unload(self) -> None: ...
    async def generate(self, messages, params, cancel=None): ...
    async def stream(self, messages, params, cancel=None): ...


@pytest.fixture(autouse=True)
def _clean() -> None:
    capability.reset_for_tests()
    yield
    capability.reset_for_tests()


def test_declaration_is_the_starting_belief() -> None:
    assert capability.constrains_decoding(_Adapter(declares=True), "m") is True
    assert capability.constrains_decoding(_Adapter(declares=False), "m") is False


def test_proof_overrides_the_declaration() -> None:
    adapter = _Adapter(declares=True)
    assert capability.record_unenforced(adapter, "m", evidence="response_not_json")
    assert capability.constrains_decoding(adapter, "m") is False


def test_nothing_ever_promotes() -> None:
    """There is no API to raise trust, and that is deliberate.

    A conforming answer is not evidence of a grammar — an unconstrained model
    can produce one by luck — so the registry exposes no way to undo a
    demotion. Restarting the process is the only reset, which is the honest
    granularity: the operator changed something.
    """
    adapter = _Adapter(declares=True)
    capability.record_unenforced(adapter, "m", evidence="response_not_json")
    assert not [name for name in dir(capability) if "promote" in name or "trust_" in name]
    assert capability.constrains_decoding(adapter, "m") is False


def test_demotion_is_scoped_to_one_deployment() -> None:
    stale = _Adapter(declares=True, host="old")
    modern = _Adapter(declares=True, host="new")
    capability.record_unenforced(stale, "m", evidence="response_not_json")

    assert capability.constrains_decoding(stale, "m") is False
    assert capability.constrains_decoding(modern, "m") is True


def test_demotion_is_scoped_to_one_model() -> None:
    """Two models on one endpoint can differ — a schema-capable server still
    depends on the model honouring the grammar it is handed."""
    adapter = _Adapter(declares=True, host="one")
    capability.record_unenforced(adapter, "small", evidence="response_not_json")

    assert capability.constrains_decoding(adapter, "small") is False
    assert capability.constrains_decoding(adapter, "large") is True


def test_recording_twice_reports_only_the_first() -> None:
    """So the caller logs the transition once instead of on every request."""
    adapter = _Adapter(declares=True)
    assert capability.record_unenforced(adapter, "m", evidence="response_not_json") is True
    assert capability.record_unenforced(adapter, "m", evidence="response_not_json") is False
    assert len(capability.observed_unenforced()) == 1


def test_a_backend_that_never_claimed_anything_is_never_recorded() -> None:
    """`constrains_decoding` is already False for it, so there is nothing to
    learn and the set stays empty rather than filling with redundant entries."""
    adapter = _Adapter(declares=False)
    assert capability.constrains_decoding(adapter, "m") is False
    assert capability.observed_unenforced() == frozenset()


def test_default_deployment_id_is_the_backend_name() -> None:
    """In-process adapters serve one model, so the class name identifies the
    deployment; only remote adapters need to override."""
    assert _Adapter(declares=True).deployment_id() == "probe_backend"
