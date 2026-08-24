"""Registry source order — which backend gets first claim on a model id.

Order is the whole mechanism behind "primary vs fallback": ``resolve()`` walks
sources in order and takes the first descriptor its accept-predicate admits.
So a reordering here silently reroutes every request, which is exactly why it
is asserted rather than left to the comment above it.
"""

from __future__ import annotations

import pytest

from inference_engine.api.state import AppState
from inference_engine.config import settings


def _source_formats(state: AppState) -> list[str]:
    """The `format` each source claims, in consult order."""
    names = []
    for source in state.registry._sources:  # noqa: SLF001 - ordering is the contract
        names.append(type(source).__name__)
    return names


@pytest.fixture
def _ollama_http_configured(monkeypatch):
    monkeypatch.setattr(settings, "ollama_http_endpoint", "http://127.0.0.1:11434")
    return monkeypatch


def test_ollama_http_is_consulted_before_local_sources_by_default(
    _ollama_http_configured,
) -> None:
    _ollama_http_configured.setattr(settings, "prefer_ollama_http_over_gguf", True)
    order = _source_formats(AppState())

    http_at = order.index("OllamaHttpRegistry")
    assert http_at < order.index("OllamaRegistry")
    assert http_at < order.index("MLXRegistry")


def test_opting_out_restores_in_process_first_ordering(_ollama_http_configured) -> None:
    _ollama_http_configured.setattr(settings, "prefer_ollama_http_over_gguf", False)
    order = _source_formats(AppState())

    http_at = order.index("OllamaHttpRegistry")
    assert http_at > order.index("OllamaRegistry")
    assert http_at > order.index("MLXRegistry")


@pytest.mark.parametrize("primary", [True, False])
def test_explicitly_configured_upstreams_outrank_ollama_http_either_way(
    _ollama_http_configured, primary: bool
) -> None:
    """vLLM/OpenRouter entries are hand-listed in a config file.

    That is a stronger operator signal than "this name happens to exist on the
    Ollama server", so it wins regardless of the local-vs-HTTP toggle.
    """
    _ollama_http_configured.setattr(settings, "prefer_ollama_http_over_gguf", primary)
    order = _source_formats(AppState())

    http_at = order.index("OllamaHttpRegistry")
    assert order.index("VLLMRegistry") < http_at
    assert order.index("OpenRouterRegistry") < http_at


def test_unset_endpoint_leaves_only_the_local_and_configured_sources(monkeypatch) -> None:
    monkeypatch.setattr(settings, "ollama_http_endpoint", "")
    order = _source_formats(AppState())

    assert "OllamaHttpRegistry" not in order
    assert "OllamaRegistry" in order and "MLXRegistry" in order
