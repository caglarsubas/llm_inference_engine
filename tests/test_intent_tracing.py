"""Caller-supplied intent labels on chat traces."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass
from pathlib import Path

import pytest

from inference_engine.adapters import GenerationParams, InferenceAdapter, StreamChunk
from inference_engine.adapters.base import GenerationResult
from inference_engine.api.chat import _blocking_response, _intent_attrs, _resolve, _stream_response
from inference_engine.auth import Identity
from inference_engine.cancellation import Cancellation
from inference_engine.registry import ModelDescriptor
from inference_engine.schemas import ChatCompletionRequest


class _IntentAdapter(InferenceAdapter):
    backend_name = "intent-fake"

    @property
    def is_loaded(self) -> bool:
        return True

    @property
    def loaded_model(self) -> ModelDescriptor | None:
        return None

    async def load(self, descriptor: ModelDescriptor) -> None:
        return None

    async def unload(self) -> None:
        return None

    async def generate(
        self, messages: Iterable, params: GenerationParams, cancel: Cancellation | None = None
    ) -> GenerationResult:
        return GenerationResult(
            text="done",
            finish_reason="stop",
            prompt_tokens=3,
            completion_tokens=2,
        )

    async def stream(
        self, messages: Iterable, params: GenerationParams, cancel: Cancellation | None = None
    ) -> AsyncIterator[StreamChunk]:
        yield StreamChunk(text="done")
        yield StreamChunk(text="", finish_reason="stop")


@dataclass
class _FakeRequest:
    async def is_disconnected(self) -> bool:
        return False


def _base_request(**overrides) -> ChatCompletionRequest:
    payload = {
        "model": "intent-fake:1",
        "messages": [{"role": "user", "content": "configure and run it"}],
    }
    payload.update(overrides)
    return ChatCompletionRequest.model_validate(payload)


def _descriptor(name: str = "intent-fake:1") -> ModelDescriptor:
    return ModelDescriptor(
        name=name.split(":")[0],
        tag=name.split(":")[1] if ":" in name else "1",
        namespace="ns",
        registry="reg",
        model_path=Path(f"/tmp/{name}"),
        format="gguf",
        size_bytes=1024,
    )


def test_request_accepts_generic_dotted_intent_attrs() -> None:
    req = _base_request(
        **{
            "intent.labels": "configuration_edit,flow_execution",
            "intent.label_names": (
                "configuration_editing_execution,flow_process_execution"
            ),
            "intent.source": "client_classifier",
            "intent.preclassified": True,
            "intent.classifier_version": "intent-router-v1",
        }
    )

    assert req.intent_labels == ["configuration_edit", "flow_execution"]
    assert req.intent_label_names == [
        "configuration_editing_execution",
        "flow_process_execution",
    ]

    attrs = _intent_attrs(req)
    assert attrs["intent.labels"] == ["configuration_edit", "flow_execution"]
    assert attrs["intent.label_names"] == [
        "configuration_editing_execution",
        "flow_process_execution",
    ]
    assert attrs["intent.count"] == 2
    assert attrs["intent.source"] == "client_classifier"
    assert attrs["intent.preclassified"] is True
    assert attrs["intent.classifier_version"] == "intent-router-v1"


@pytest.mark.asyncio
async def test_model_acquire_span_stamps_intent_before_generation(
    monkeypatch,
    _session_exporter,
) -> None:
    _session_exporter.clear()
    req = _base_request(**{"intent.labels": "configuration_edit,flow_execution"})

    async def _fake_get(model_id: str):
        return _IntentAdapter(), _descriptor(model_id)

    from inference_engine.api.state import app_state  # noqa: PLC0415

    monkeypatch.setattr(app_state.manager, "get", _fake_get)
    await _resolve(
        req.model,
        Identity(tenant="dev", key_id="sk-x"),
        intent_attrs=_intent_attrs(req),
    )

    [acquire_span] = [
        span for span in _session_exporter.get_finished_spans() if span.name == "model.acquire"
    ]
    attrs = acquire_span.attributes
    assert list(attrs["intent.labels"]) == ["configuration_edit", "flow_execution"]


@pytest.mark.asyncio
async def test_blocking_chat_span_stamps_generic_intent_attrs(
    _session_exporter,
) -> None:
    _session_exporter.clear()
    req = _base_request(
        **{
            "intent.labels": ["configuration_edit", "flow_execution"],
            "intent.label_names": [
                "configuration_editing_execution",
                "flow_process_execution",
            ],
            "intent.source": "client_classifier",
            "intent.preclassified": True,
            "intent.classifier_version": "intent-router-v1",
        }
    )

    await _blocking_response(
        adapter=_IntentAdapter(),
        model_name=req.model,
        messages=req.messages,
        params=GenerationParams(),
        identity=Identity(tenant="dev", key_id="sk-x"),
        intent_attrs=_intent_attrs(req),
    )

    [chat_span] = [
        span for span in _session_exporter.get_finished_spans() if span.name == "chat.generate"
    ]
    attrs = chat_span.attributes
    assert list(attrs["intent.labels"]) == ["configuration_edit", "flow_execution"]
    assert list(attrs["intent.label_names"]) == [
        "configuration_editing_execution",
        "flow_process_execution",
    ]
    assert attrs["intent.count"] == 2
    assert attrs["intent.source"] == "client_classifier"
    assert attrs["intent.preclassified"] is True
    assert attrs["intent.classifier_version"] == "intent-router-v1"


@pytest.mark.asyncio
async def test_stream_chat_span_stamps_preclassified_button_intent(
    _session_exporter,
) -> None:
    _session_exporter.clear()
    req = _base_request(
        stream=True,
        **{
            "intent.labels": ["current_status"],
            "intent.label_names": ["current_status_information_gathering"],
            "intent.source": "support_button",
            "intent.preclassified": True,
        },
    )

    async for _ in _stream_response(
        adapter=_IntentAdapter(),
        model_name=req.model,
        messages=req.messages,
        params=GenerationParams(),
        identity=Identity(tenant="dev", key_id="sk-x"),
        request=_FakeRequest(),
        intent_attrs=_intent_attrs(req),
    ):
        pass

    [chat_span] = [
        span for span in _session_exporter.get_finished_spans() if span.name == "chat.stream"
    ]
    attrs = chat_span.attributes
    assert list(attrs["intent.labels"]) == ["current_status"]
    assert list(attrs["intent.label_names"]) == ["current_status_information_gathering"]
    assert attrs["intent.count"] == 1
    assert attrs["intent.source"] == "support_button"
    assert attrs["intent.preclassified"] is True
