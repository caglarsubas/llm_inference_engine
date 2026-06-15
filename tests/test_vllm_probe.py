from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from inference_engine.api import models as models_api
from inference_engine.registry import CompositeRegistry, ModelDescriptor, VLLMProbeResult
from inference_engine.registry.vllm_probe import VLLMUpstreamProbe


def _descriptor(
    *,
    endpoint: str = "http://vllm:8000",
    model_id: str = "Qwen/Qwen3-VL-8B-Instruct",
) -> ModelDescriptor:
    return ModelDescriptor(
        name="qwen3-vl-8b-instruct",
        tag="vllm",
        namespace="vllm",
        registry="local",
        model_path=Path(f"vllm://{endpoint}/{model_id}"),
        format="vllm",
        params={"model_id": model_id},
        endpoint=endpoint,
    )


def _probe(handler) -> VLLMUpstreamProbe:
    def factory(base_url: str, timeout: httpx.Timeout) -> httpx.Client:
        return httpx.Client(
            base_url=base_url,
            timeout=timeout,
            transport=httpx.MockTransport(handler),
        )

    return VLLMUpstreamProbe(timeout_seconds=0.1, ttl_seconds=0.0, client_factory=factory)


def test_vllm_probe_accepts_upstream_that_lists_model_id() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/models"
        return httpx.Response(
            200,
            json={"object": "list", "data": [{"id": "Qwen/Qwen3-VL-8B-Instruct"}]},
        )

    result = _probe(handler).probe(_descriptor())

    assert result.loadable is True
    assert result.reason == ""


def test_vllm_probe_rejects_upstream_serving_different_model() -> None:
    def handler(request: httpx.Request) -> httpx.Response:  # noqa: ARG001
        return httpx.Response(200, json={"object": "list", "data": [{"id": "other"}]})

    result = _probe(handler).probe(_descriptor())

    assert result.loadable is False
    assert result.reason == "upstream_model_missing"
    assert "Qwen/Qwen3-VL-8B-Instruct" in result.detail


def test_vllm_probe_reports_unreachable_upstream() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    result = _probe(handler).probe(_descriptor())

    assert result.loadable is False
    assert result.reason == "upstream_unreachable"
    assert "connection refused" in result.detail


def test_vllm_probe_reports_bad_models_payload() -> None:
    def handler(request: httpx.Request) -> httpx.Response:  # noqa: ARG001
        return httpx.Response(200, json={"models": []})

    result = _probe(handler).probe(_descriptor())

    assert result.loadable is False
    assert result.reason == "upstream_bad_models_response"


class _Source:
    def __init__(self, descriptor: ModelDescriptor) -> None:
        self.descriptor = descriptor

    def list_models(self) -> list[ModelDescriptor]:
        return [self.descriptor]

    def get(self, name_with_tag: str) -> ModelDescriptor | None:
        if name_with_tag in {self.descriptor.qualified_name, self.descriptor.name}:
            return self.descriptor
        return None


class _FakeProbe:
    def probe(self, descriptor: ModelDescriptor) -> VLLMProbeResult:  # noqa: ARG002
        return VLLMProbeResult(
            loadable=False,
            reason="upstream_unreachable",
            detail="connection refused",
        )


@pytest.mark.asyncio
async def test_models_api_keeps_unreachable_vllm_out_of_data(monkeypatch: pytest.MonkeyPatch) -> None:
    descriptor = _descriptor()
    monkeypatch.setattr(models_api.app_state, "registry", CompositeRegistry([_Source(descriptor)]))
    monkeypatch.setattr(models_api, "get_vllm_probe", lambda: _FakeProbe())

    result = await models_api.list_models(_=object())

    assert result.data == []
    assert len(result.unavailable) == 1
    assert result.unavailable[0].id == "qwen3-vl-8b-instruct:vllm"
    assert result.unavailable[0].reason == "upstream_unreachable"
    assert result.unavailable[0].backend == "vllm"
