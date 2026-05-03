"""``/v1/rerank`` — schema, ranking correctness, top_n bound, error paths, span attrs."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterable
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from inference_engine.adapters import (
    EmbeddingResult,
    EmbeddingsNotSupportedError,
    GenerationParams,
    InferenceAdapter,
    StreamChunk,
)
from inference_engine.adapters.base import GenerationResult
from inference_engine.cancellation import Cancellation
from inference_engine.config import settings
from inference_engine.main import app
from inference_engine.registry import ModelDescriptor
from inference_engine.schemas import RerankRequest


# ---------------------------------------------------------------------------
# Schema acceptance
# ---------------------------------------------------------------------------


def test_request_accepts_query_and_documents() -> None:
    req = RerankRequest(model="x", query="q", documents=["a", "b"])
    assert req.query == "q"
    assert len(req.documents) == 2
    assert req.top_n is None
    assert req.return_documents is False


def test_request_rejects_empty_documents() -> None:
    with pytest.raises(ValueError):
        RerankRequest(model="x", query="q", documents=[])


def test_request_top_n_must_be_positive() -> None:
    with pytest.raises(ValueError):
        RerankRequest(model="x", query="q", documents=["a"], top_n=0)


# ---------------------------------------------------------------------------
# Cosine ranking — verified via TestClient with a controllable embedding adapter
# ---------------------------------------------------------------------------


class _ControlledEmbedAdapter(InferenceAdapter):
    """Emits a hand-picked embedding per input string.

    Maps each input to a deterministic vector via the ``vectors`` dict so tests
    can construct a known relevance ordering. Inputs not in the dict get a
    zero vector (cosine 0 against everything).
    """

    backend_name = "controlled"

    def __init__(self, vectors: dict[str, list[float]] | None = None, fail: bool = False) -> None:
        self._vectors = vectors or {}
        self._fail = fail
        self.last_embed_action: str = "batch"
        self._descriptor: ModelDescriptor | None = None

    @property
    def is_loaded(self) -> bool:
        return self._descriptor is not None

    @property
    def loaded_model(self) -> ModelDescriptor | None:
        return self._descriptor

    async def load(self, descriptor: ModelDescriptor) -> None:
        self._descriptor = descriptor

    async def unload(self) -> None:
        self._descriptor = None

    async def generate(
        self, messages: Iterable, params: GenerationParams, cancel: Cancellation | None = None
    ) -> GenerationResult:
        return GenerationResult(text="", finish_reason="stop", prompt_tokens=0, completion_tokens=0)

    async def stream(
        self, messages: Iterable, params: GenerationParams, cancel: Cancellation | None = None
    ) -> AsyncIterator[StreamChunk]:
        yield StreamChunk(text="", finish_reason="stop")

    async def embed(self, inputs: list[str]) -> EmbeddingResult:
        if self._fail:
            raise EmbeddingsNotSupportedError(self.backend_name)
        zero = [0.0, 0.0, 0.0]
        out = [self._vectors.get(s, zero) for s in inputs]
        return EmbeddingResult(embeddings=out, prompt_tokens=len(inputs))


def _stub_descriptor(name: str = "controlled:1") -> ModelDescriptor:
    return ModelDescriptor(
        name=name.split(":")[0],
        tag=name.split(":")[1] if ":" in name else "1",
        namespace="ns",
        registry="reg",
        model_path=Path(f"/tmp/{name}"),
        format="gguf",
        size_bytes=1024,
    )


@pytest.fixture
def patched_manager(monkeypatch):
    class _Box:
        adapter: InferenceAdapter | None = None

    box = _Box()

    async def _fake_get(model_id: str):
        if box.adapter is None:
            from inference_engine.manager import ModelNotFoundError  # noqa: PLC0415

            raise ModelNotFoundError(model_id)
        return box.adapter, _stub_descriptor(model_id)

    from inference_engine.api.state import app_state  # noqa: PLC0415

    monkeypatch.setattr(app_state.manager, "get", _fake_get)
    # Disable batching coalescer wait window so tests don't pay 10ms per request.
    monkeypatch.setattr(settings, "batch_max_wait_ms", 0.1)

    def _set(adapter: InferenceAdapter) -> None:
        box.adapter = adapter

    box.set_adapter = _set  # type: ignore[attr-defined]
    return box


def test_route_ranks_by_cosine_similarity(patched_manager) -> None:
    """Query vector aligned with doc 1 (perfectly), antiparallel with doc 0,
    orthogonal to doc 2. Expect order: 1 > 2 > 0."""
    adapter = _ControlledEmbedAdapter(
        vectors={
            "what is python?": [1.0, 0.0],
            "python is a programming language": [1.0, 0.0],   # same direction → cosine 1
            "rust is a programming language":   [0.0, 1.0],   # orthogonal     → cosine 0
            "i hate python":                     [-1.0, 0.0], # antiparallel   → cosine -1
        }
    )
    patched_manager.set_adapter(adapter)
    client = TestClient(app)

    r = client.post(
        "/v1/rerank",
        json={
            "model": "controlled:1",
            "query": "what is python?",
            "documents": [
                "i hate python",
                "python is a programming language",
                "rust is a programming language",
            ],
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    indices = [item["index"] for item in body["results"]]
    scores = [item["relevance_score"] for item in body["results"]]

    # Best match (cosine 1) → original index 1 (python doc)
    # Then orthogonal     → original index 2 (rust doc)
    # Then antiparallel   → original index 0 (i hate python)
    assert indices == [1, 2, 0]
    assert scores[0] == pytest.approx(1.0)
    assert scores[1] == pytest.approx(0.0)
    assert scores[2] == pytest.approx(-1.0)


def test_route_top_n_truncates_results(patched_manager) -> None:
    adapter = _ControlledEmbedAdapter(
        vectors={
            "q": [1.0, 0.0],
            "a": [1.0, 0.0],
            "b": [0.5, 0.5],
            "c": [0.0, 1.0],
        }
    )
    patched_manager.set_adapter(adapter)
    client = TestClient(app)

    r = client.post(
        "/v1/rerank",
        json={"model": "controlled:1", "query": "q", "documents": ["a", "b", "c"], "top_n": 2},
    )
    assert r.status_code == 200
    assert len(r.json()["results"]) == 2


def test_route_return_documents_echoes_text(patched_manager) -> None:
    adapter = _ControlledEmbedAdapter(
        vectors={"q": [1.0], "doc1": [1.0], "doc2": [0.0]}
    )
    patched_manager.set_adapter(adapter)
    client = TestClient(app)

    r = client.post(
        "/v1/rerank",
        json={"model": "controlled:1", "query": "q", "documents": ["doc1", "doc2"], "return_documents": True},
    )
    assert r.status_code == 200
    body = r.json()
    # First result's document text matches its original index.
    assert body["results"][0]["document"] in ("doc1", "doc2")
    assert all(item.get("document") for item in body["results"])


def test_route_default_omits_documents_for_compactness(patched_manager) -> None:
    adapter = _ControlledEmbedAdapter(
        vectors={"q": [1.0], "doc1": [1.0]}
    )
    patched_manager.set_adapter(adapter)
    client = TestClient(app)

    r = client.post("/v1/rerank", json={"model": "controlled:1", "query": "q", "documents": ["doc1"]})
    assert r.status_code == 200
    assert r.json()["results"][0]["document"] is None


def test_route_404_for_unknown_model(patched_manager) -> None:
    client = TestClient(app)
    r = client.post("/v1/rerank", json={"model": "ghost:1", "query": "q", "documents": ["a"]})
    assert r.status_code == 404


def test_route_400_for_empty_documents(patched_manager) -> None:
    patched_manager.set_adapter(_ControlledEmbedAdapter())
    client = TestClient(app)
    r = client.post("/v1/rerank", json={"model": "controlled:1", "query": "q", "documents": []})
    # Pydantic catches this at request validation time → 422.
    assert r.status_code == 422


def test_route_501_when_backend_lacks_embeddings(patched_manager) -> None:
    patched_manager.set_adapter(_ControlledEmbedAdapter(fail=True))
    client = TestClient(app)
    r = client.post("/v1/rerank", json={"model": "controlled:1", "query": "q", "documents": ["a"]})
    assert r.status_code == 501
    assert "rerank not supported" in r.json()["detail"]


def test_route_emits_rerank_run_span(patched_manager, _session_exporter) -> None:
    _session_exporter.clear()
    adapter = _ControlledEmbedAdapter(
        vectors={"q": [1.0, 0.0], "a": [1.0, 0.0], "b": [0.0, 1.0]}
    )
    patched_manager.set_adapter(adapter)
    client = TestClient(app)
    r = client.post(
        "/v1/rerank",
        json={"model": "controlled:1", "query": "q", "documents": ["a", "b"], "top_n": 1},
    )
    assert r.status_code == 200

    spans = [s for s in _session_exporter.get_finished_spans() if s.name == "rerank.run"]
    assert len(spans) == 1
    s = spans[0]
    assert s.attributes["gen_ai.system"] == "controlled"
    assert s.attributes["gen_ai.request.model"] == "controlled:1"
    assert s.attributes["rerank.documents_count"] == 2
    assert s.attributes["rerank.top_n"] == 1
    assert s.attributes["rerank.results_returned"] == 1
    assert s.attributes["embedding.dimensions"] == 2
    assert s.attributes["gen_ai.usage.input_tokens"] == 3  # query + 2 docs
