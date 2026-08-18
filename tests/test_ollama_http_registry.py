"""Discovery-time metadata for ollama-served models.

``/api/tags`` carries no context window, so without a second call every
ollama-served model reports a null window in ``/v1/models`` while a
GGUF-served one reports a real number — the same weights look different
depending on which backend happened to win resolution.
"""

from __future__ import annotations

import io
import json

import pytest

from inference_engine.registry import ollama_http as module
from inference_engine.registry.ollama_http import OllamaHttpRegistry

ENDPOINT = "http://ollama.test:11434"


def _tags_payload(*, digest: str = "sha256:abc") -> dict:
    return {"models": [{"name": "seer:27b", "size": 17_000_000_000, "digest": digest}]}


def _show_payload(*, arch: str = "qwen35", context_length: int = 262_144) -> dict:
    return {
        "model_info": {
            "general.architecture": arch,
            f"{arch}.context_length": context_length,
        }
    }


class _Transport:
    """Stands in for ``urlopen``, recording which endpoints were hit."""

    def __init__(self, *, show_result: dict | Exception, tags: dict | None = None) -> None:
        self.show_result = show_result
        self.tags = tags if tags is not None else _tags_payload()
        self.show_calls: list[str] = []
        self.tag_calls = 0

    def __call__(self, request, timeout=None):  # noqa: ANN001 - urlopen signature
        url = request if isinstance(request, str) else request.full_url
        if url.endswith("/api/tags"):
            self.tag_calls += 1
            return _response(self.tags)
        if url.endswith("/api/show"):
            body = json.loads(request.data.decode("utf-8"))
            self.show_calls.append(body["model"])
            if isinstance(self.show_result, Exception):
                raise self.show_result
            return _response(self.show_result)
        raise AssertionError(f"unexpected url {url}")


def _response(payload: dict):
    class _Ctx(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    return _Ctx(json.dumps(payload).encode("utf-8"))


def test_context_length_is_discovered_from_api_show(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = _Transport(show_result=_show_payload())
    monkeypatch.setattr(module.urllib.request, "urlopen", transport)

    desc = OllamaHttpRegistry(ENDPOINT).list_models()[0]

    assert desc.params["context_length"] == 262_144
    assert transport.show_calls == ["seer:27b"]


def test_context_length_is_cached_per_digest(monkeypatch: pytest.MonkeyPatch) -> None:
    """A TTL expiry must not re-ask for a number that cannot have changed."""
    transport = _Transport(show_result=_show_payload())
    monkeypatch.setattr(module.urllib.request, "urlopen", transport)
    registry = OllamaHttpRegistry(ENDPOINT)

    registry.list_models()
    registry._refresh(force=True)
    registry._refresh(force=True)

    assert transport.tag_calls == 3
    assert transport.show_calls == ["seer:27b"]


def test_re_pull_under_a_new_digest_re_asks(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = _Transport(show_result=_show_payload())
    monkeypatch.setattr(module.urllib.request, "urlopen", transport)
    registry = OllamaHttpRegistry(ENDPOINT)

    registry.list_models()
    transport.tags = _tags_payload(digest="sha256:def")
    registry._refresh(force=True)

    assert transport.show_calls == ["seer:27b", "seer:27b"]


def test_unprefixed_context_length_key_still_resolves(monkeypatch: pytest.MonkeyPatch) -> None:
    """The prefix is the GGUF arch string, which need not match the model name."""
    payload = {"model_info": {"something.context_length": 8192}}
    monkeypatch.setattr(
        module.urllib.request, "urlopen", _Transport(show_result=payload)
    )

    desc = OllamaHttpRegistry(ENDPOINT).list_models()[0]

    assert desc.params["context_length"] == 8192


def test_show_failure_leaves_the_model_discoverable(monkeypatch: pytest.MonkeyPatch) -> None:
    """An advisory field is not worth failing discovery over."""
    transport = _Transport(show_result=TimeoutError("upstream slow"))
    monkeypatch.setattr(module.urllib.request, "urlopen", transport)

    models = OllamaHttpRegistry(ENDPOINT).list_models()

    assert len(models) == 1
    assert models[0].qualified_name == "seer:27b"
    assert "context_length" not in models[0].params
    # The identity fields the adapter actually routes on must survive.
    assert models[0].params["model_id"] == "seer:27b"
