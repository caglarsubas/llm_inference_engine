"""Resident-model substitution: serve an interchangeable model Ollama already holds.

The rule is unit-tested against a scripted ``/api/ps``, then exercised through
the real ASGI app so the caller-facing contract — body fields, response headers
and the opt-out header — is checked with the middleware in the path.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterable
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from inference_engine import model_substitution as ms
from inference_engine.adapters import GenerationParams, InferenceAdapter, StreamChunk
from inference_engine.adapters.base import GenerationResult
from inference_engine.api.state import app_state
from inference_engine.cancellation import Cancellation
from inference_engine.evals import EvalRunner
from inference_engine.main import app
from inference_engine.manager import ModelNotFoundError
from inference_engine.registry import ModelDescriptor

GEMMA = "gemma4:26b"
QWEN = "qwen3.8:27b"
OTHER = "ministral-3:8b"
ENDPOINT = "http://ollama.test:11434"


def _descriptor(model_id: str, *, fmt: str = "ollama_http", endpoint: str = ENDPOINT):
    name, tag = model_id.rsplit(":", 1)
    return ModelDescriptor(
        name=name,
        tag=tag,
        namespace="library",
        registry="ollama",
        model_path=Path(f"/tmp/{model_id}"),
        format=fmt,
        params={"model_id": model_id},
        endpoint=endpoint if fmt == "ollama_http" else None,
    )


class _Resolver:
    def __init__(self, *descriptors: ModelDescriptor) -> None:
        self._by_name = {d.qualified_name: d for d in descriptors}
        self.calls = 0

    def resolve(self, model_id: str):
        self.calls += 1
        return self._by_name.get(model_id)


class _Probe:
    """Scripted ``/api/ps``: returns ``resident`` or raises ``error``."""

    def __init__(self, *resident: str, error: Exception | None = None) -> None:
        self.resident = set(resident)
        self.error = error
        self.calls = 0

    async def __call__(self, endpoint: str) -> frozenset[str]:
        self.calls += 1
        if self.error is not None:
            raise self.error
        return frozenset(self.resident)


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def _substituter(probe: _Probe, clock: _Clock | None = None, groups: str = f"{GEMMA},{QWEN}"):
    residency = ms.ResidencyCache(probe, clock=clock or _Clock())
    return ms.ModelSubstituter(ms.parse_groups(groups), residency)


_ALL = _Resolver(_descriptor(GEMMA), _descriptor(QWEN), _descriptor(OTHER))


# --- configuration ------------------------------------------------------------


def test_parse_groups_normalizes_and_drops_groups_that_cannot_substitute() -> None:
    assert ms.parse_groups(" gemma4:26b , qwen3.8:27b ; solo:1 ; a, b ,a ") == (
        (GEMMA, QWEN),
        ("a:latest", "b:latest"),
    )
    assert ms.parse_groups("") == ()


# --- the rule -----------------------------------------------------------------


async def test_serves_the_resident_member_when_the_requested_one_is_not_loaded() -> None:
    substitution = await _substituter(_Probe(GEMMA)).choose(QWEN, _ALL)

    assert substitution == ms.ModelSubstitution(
        requested_model=QWEN, served_model=GEMMA, reason=ms.REASON_NOT_RESIDENT
    )


async def test_the_requested_model_wins_whenever_it_is_resident() -> None:
    assert await _substituter(_Probe(GEMMA, QWEN)).choose(QWEN, _ALL) is None
    assert await _substituter(_Probe(QWEN)).choose(QWEN, _ALL) is None


async def test_models_outside_every_group_are_never_substituted() -> None:
    probe = _Probe(GEMMA)
    assert await _substituter(probe).choose(OTHER, _ALL) is None
    assert probe.calls == 0


async def test_disabled_without_groups_and_touches_nothing() -> None:
    probe = _Probe(GEMMA)
    resolver = _Resolver(_descriptor(GEMMA), _descriptor(QWEN))
    substituter = _substituter(probe, groups="")

    assert not substituter.enabled
    assert await substituter.choose(QWEN, resolver) is None
    assert probe.calls == 0 and resolver.calls == 0


async def test_unreadable_residency_serves_the_requested_model() -> None:
    probe = _Probe(error=RuntimeError("connection refused"))
    substituter = _substituter(probe)

    assert await substituter.choose(QWEN, _ALL) is None
    # The failure is cached, so a dead Ollama is asked once per TTL, not per request.
    assert await substituter.choose(QWEN, _ALL) is None
    assert probe.calls == 1


async def test_only_ollama_models_on_the_same_endpoint_are_interchangeable() -> None:
    elsewhere = _Resolver(_descriptor(QWEN), _descriptor(GEMMA, endpoint="http://other:11434"))
    assert await _substituter(_Probe(GEMMA)).choose(QWEN, elsewhere) is None

    local_gguf = _Resolver(_descriptor(QWEN, fmt="gguf"), _descriptor(GEMMA))
    assert await _substituter(_Probe(GEMMA)).choose(QWEN, local_gguf) is None


async def test_a_cold_load_counts_as_resident_until_ollama_lists_it() -> None:
    """``/api/ps`` omits a model until its load finishes.

    Without remembering the load, a request for the other member arriving
    mid-load would start a second load, and the pair would evict each other
    exactly as they did before this rule existed.
    """
    probe, clock = _Probe(), _Clock()
    substituter = _substituter(probe, clock)

    assert await substituter.choose(QWEN, _ALL) is None  # nothing resident: load qwen
    clock.now += 30  # well past the residency TTL, still mid-load
    joined = await substituter.choose(GEMMA, _ALL)
    assert joined == ms.ModelSubstitution(requested_model=GEMMA, served_model=QWEN)

    clock.now += ms.PENDING_LOAD_SECONDS
    assert await substituter.choose(GEMMA, _ALL) is None  # the note expired


async def test_residency_is_cached_briefly_per_endpoint() -> None:
    probe, clock = _Probe(GEMMA), _Clock()
    substituter = _substituter(probe, clock)

    await substituter.choose(QWEN, _ALL)
    await substituter.choose(QWEN, _ALL)
    assert probe.calls == 1
    clock.now += ms.RESIDENCY_TTL_SECONDS
    await substituter.choose(QWEN, _ALL)
    assert probe.calls == 2


async def test_the_caller_can_insist_on_the_exact_model() -> None:
    ms.begin_request([(b"x-engine-model-substitution", b"off")])
    assert await _substituter(_Probe(GEMMA)).choose(QWEN, _ALL) is None

    ms.begin_request([(b"x-engine-model-substitution", b"on")])
    assert await _substituter(_Probe(GEMMA)).choose(QWEN, _ALL) is not None


# --- through the app ------------------------------------------------------------


class _NamedAdapter(InferenceAdapter):
    backend_name = "ollama_http"

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id

    @property
    def is_loaded(self) -> bool:
        return True

    @property
    def loaded_model(self) -> ModelDescriptor | None:
        return None

    async def load(self, descriptor: ModelDescriptor) -> None: ...
    async def unload(self) -> None: ...

    async def generate(
        self, messages: Iterable, params: GenerationParams, cancel: Cancellation | None = None
    ) -> GenerationResult:
        text = (
            '{"score": 4, "justification": "fine"}' if params.json_mode else f"from {self.model_id}"
        )
        return GenerationResult(
            text=text, finish_reason="stop", prompt_tokens=3, completion_tokens=2
        )

    async def stream(
        self, messages: Iterable, params: GenerationParams, cancel: Cancellation | None = None
    ) -> AsyncIterator[StreamChunk]:
        yield StreamChunk(text=f"from {self.model_id}")
        yield StreamChunk(text="", finish_reason="stop")


@pytest.fixture(autouse=True)
def _ready():
    app_state.mark_ready()
    yield
    app_state.mark_ready()


@pytest.fixture
def served(monkeypatch) -> list[str]:
    """Ollama holds gemma4 only; record which model each acquire returns."""
    acquired: list[str] = []

    async def _get(model_id: str):
        descriptor = _ALL.resolve(model_id)
        if descriptor is None:
            raise ModelNotFoundError(model_id)
        acquired.append(model_id)
        return _NamedAdapter(model_id), descriptor

    substituter = _substituter(_Probe(GEMMA))
    monkeypatch.setattr(app_state.manager, "get", _get)
    monkeypatch.setattr(app_state.manager, "resolve", _ALL.resolve)
    monkeypatch.setattr(app_state, "model_substituter", substituter)
    monkeypatch.setattr(
        app_state, "eval_runner", EvalRunner(app_state.manager, substituter=substituter)
    )
    return acquired


def _chat(**extra) -> dict:
    return {"model": QWEN, "messages": [{"role": "user", "content": "hi"}], **extra}


def test_blocking_chat_tells_the_caller_which_model_served(served: list[str]) -> None:
    response = TestClient(app).post("/v1/chat/completions", json=_chat())

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["model"] == GEMMA
    assert body["substituted_from_model"] == QWEN
    assert body["substitution_reason"] == ms.REASON_NOT_RESIDENT
    assert body["choices"][0]["message"]["content"] == f"from {GEMMA}"
    assert response.headers[ms.SUBSTITUTED_FROM_HEADER] == QWEN
    assert response.headers[ms.SERVED_MODEL_HEADER] == GEMMA
    assert response.headers[ms.REASON_HEADER] == ms.REASON_NOT_RESIDENT
    assert served == [GEMMA]


def test_streaming_chat_reports_the_substitution_before_the_first_delta(
    served: list[str],
) -> None:
    with TestClient(app).stream("POST", "/v1/chat/completions", json=_chat(stream=True)) as r:
        assert r.status_code == 200
        assert r.headers[ms.SUBSTITUTED_FROM_HEADER] == QWEN
        body = "".join(r.iter_text())

    chunks = [
        json.loads(line[len("data: ") :])
        for line in body.splitlines()
        if line.startswith("data: {")
    ]
    assert chunks
    assert all(c["model"] == GEMMA and c["substituted_from_model"] == QWEN for c in chunks)


def test_a_resident_request_carries_no_substitution(served: list[str]) -> None:
    response = TestClient(app).post("/v1/chat/completions", json=_chat(model=GEMMA))

    assert response.status_code == 200, response.text
    assert response.json()["substituted_from_model"] is None
    assert ms.SUBSTITUTED_FROM_HEADER not in response.headers


def test_opt_out_header_serves_the_exact_model(served: list[str]) -> None:
    response = TestClient(app).post(
        "/v1/chat/completions",
        json=_chat(),
        headers={ms.OPT_OUT_HEADER: "off"},
    )

    assert response.status_code == 200, response.text
    assert response.json()["model"] == QWEN
    assert response.json()["substituted_from_model"] is None
    assert served == [QWEN]


def test_eval_judge_substitution_is_reported(served: list[str]) -> None:
    response = TestClient(app).post(
        "/v1/evals/run",
        json={"rubric": "helpfulness", "prompt": "p", "response": "r", "judge_model": QWEN},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["judge_model"] == GEMMA
    assert body["substituted_from_model"] == QWEN
    assert body["substitution_reason"] == ms.REASON_NOT_RESIDENT
    assert body["verdict"]["parse_status"] == "clean"
    assert response.headers[ms.SERVED_MODEL_HEADER] == GEMMA


def test_embeddings_are_never_substituted(served: list[str]) -> None:
    """A different model is a different vector space, so embeddings keep the ask."""
    response = TestClient(app).post("/v1/embeddings", json={"model": QWEN, "input": "hi"})

    assert ms.SUBSTITUTED_FROM_HEADER not in response.headers
    assert served == [QWEN]
