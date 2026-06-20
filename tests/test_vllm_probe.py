from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from inference_engine.api import models as models_api
from inference_engine.registry import CompositeRegistry, ModelDescriptor, VLLMProbeResult
from inference_engine.registry.vllm import VLLMRegistry
from inference_engine.registry.vllm_probe import VLLMUpstreamProbe


def _descriptor(
    *,
    endpoint: str = "http://vllm:8000",
    model_id: str = "Qwen/Qwen3-VL-8B-Instruct",
    params: dict | None = None,
) -> ModelDescriptor:
    descriptor_params = {"model_id": model_id}
    if params:
        descriptor_params.update(params)
    return ModelDescriptor(
        name="qwen3-vl-8b-instruct",
        tag="vllm",
        namespace="vllm",
        registry="local",
        model_path=Path(f"vllm://{endpoint}/{model_id}"),
        format="vllm",
        params=descriptor_params,
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


class _FakePassingProbe:
    def probe(self, descriptor: ModelDescriptor) -> VLLMProbeResult:  # noqa: ARG002
        return VLLMProbeResult(loadable=True)


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
    assert result.unavailable[0].available is False
    assert result.unavailable[0].upstream_reachable is False
    assert result.unavailable[0].availability_status == "upstream_unreachable"


@pytest.mark.asyncio
async def test_models_data_returns_vllm_vlm_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    descriptor = _descriptor(
        params={
            "family": "Qwen3-VL",
            "profile": "vision",
            "modality": "text+image->text",
            "context_length": 256000,
            "supports_json_mode": True,
            "supports_strict_image_json": False,
            "strict_image_json_status": "pending_smoke",
            "strict_image_json_checked_at": "2026-06-19",
            "strict_image_json_detail": "not yet smoke validated",
            "commercial_use": "Apache-2.0",
            "benchmark_only": True,
            "parameter_count_b": 8,
            "open_weight": True,
            "proprietary": False,
        }
    )
    monkeypatch.setattr(models_api.app_state, "registry", CompositeRegistry([_Source(descriptor)]))
    monkeypatch.setattr(models_api, "get_vllm_probe", lambda: _FakePassingProbe())

    result = await models_api.list_model_catalog(_=object())

    assert result.unavailable == []
    assert len(result.data) == 1
    entry = result.data[0]
    assert entry.id == "qwen3-vl-8b-instruct:vllm"
    assert entry.available is True
    assert entry.upstream_reachable is True
    assert entry.availability_status == "available"
    assert entry.provider == "vllm"
    assert entry.backend == "vllm"
    assert entry.upstream_model_id == "Qwen/Qwen3-VL-8B-Instruct"
    assert entry.modality == "text+image->text"
    assert entry.supports_images is True
    assert entry.context_length == 256000
    assert entry.supports_json_mode is True
    assert entry.supports_strict_image_json is False
    assert entry.strict_image_json_status == "pending_smoke"
    assert entry.strict_image_json_checked_at == "2026-06-19"
    assert entry.strict_image_json_detail == "not yet smoke validated"
    assert entry.commercial_use == "Apache-2.0"
    assert entry.benchmark_only is True
    assert entry.parameter_count_b == 8
    assert entry.open_weight is True
    assert entry.proprietary is False


@pytest.mark.asyncio
async def test_models_data_returns_fakeshield_issue_43_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = VLLMRegistry(Path(__file__).resolve().parents[1] / ".vllm_models.fakeshield.example.json")
    monkeypatch.setattr(models_api.app_state, "registry", CompositeRegistry([registry]))
    monkeypatch.setattr(models_api, "get_vllm_probe", lambda: _FakePassingProbe())

    result = await models_api.list_model_catalog(_=object())

    assert result.unavailable == []
    assert len(result.data) == 1
    entry = result.data[0]
    assert entry.id == "fakeshield-22b:vllm"
    assert entry.provider == "vllm"
    assert entry.backend == "vllm"
    assert entry.upstream_model_id == "zhipeixu/fakeshield-v1-22b"
    assert entry.endpoint == "http://vllm-fakeshield-22b:8000"
    assert entry.model_path is not None
    assert "vllm-fakeshield-22b:8000" in entry.model_path
    assert entry.modality == "text+image->text"
    assert entry.supports_images is True
    assert entry.supports_json_mode is True
    assert entry.supports_strict_image_json is False
    assert entry.strict_image_json_status == "pending_smoke"
    assert entry.strict_image_json_checked_at == "2026-06-20"
    assert "Issue #43" in entry.strict_image_json_detail
    assert entry.family == "FakeShield"
    assert entry.profile == "forensics"
    assert entry.parameter_count_b == 22
    assert entry.open_weight is True
    assert entry.proprietary is False
    assert entry.commercial_use == "Apache-2.0; verify provider terms before production use"
    assert entry.benchmark_only is True


@pytest.mark.asyncio
async def test_models_data_keeps_unreachable_fakeshield_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = VLLMRegistry(Path(__file__).resolve().parents[1] / ".vllm_models.fakeshield.example.json")
    monkeypatch.setattr(models_api.app_state, "registry", CompositeRegistry([registry]))
    monkeypatch.setattr(models_api, "get_vllm_probe", lambda: _FakeProbe())

    result = await models_api.list_model_catalog(_=object())

    assert result.data == []
    assert len(result.unavailable) == 1
    unavailable = result.unavailable[0]
    assert unavailable.id == "fakeshield-22b:vllm"
    assert unavailable.reason == "upstream_unreachable"
    assert unavailable.available is False
    assert unavailable.upstream_reachable is False
    assert unavailable.availability_status == "upstream_unreachable"
    assert unavailable.availability_detail == "connection refused"
    assert unavailable.provider == "vllm"
    assert unavailable.backend == "vllm"
    assert unavailable.upstream_model_id == "zhipeixu/fakeshield-v1-22b"
    assert unavailable.endpoint == "http://vllm-fakeshield-22b:8000"
    assert unavailable.model_path is not None
    assert "vllm-fakeshield-22b:8000" in unavailable.model_path
    assert unavailable.modality == "text+image->text"
    assert unavailable.supports_images is True
    assert unavailable.supports_json_mode is True
    assert unavailable.supports_strict_image_json is False
    assert unavailable.strict_image_json_status == "pending_smoke"
    assert unavailable.strict_image_json_checked_at == "2026-06-20"
    assert "Issue #43" in unavailable.strict_image_json_detail
    assert unavailable.family == "FakeShield"
    assert unavailable.profile == "forensics"
    assert unavailable.parameter_count_b == 22
    assert unavailable.open_weight is True
    assert unavailable.proprietary is False
    assert unavailable.commercial_use == "Apache-2.0; verify provider terms before production use"
    assert unavailable.benchmark_only is True


@pytest.mark.asyncio
async def test_models_data_reports_demanded_vllm_candidates_as_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    live = tmp_path / "vllm.json"
    demanded = tmp_path / "demanded.json"
    demanded.write_text(
        """
        [
          {
            "name": "qwen3-vl-32b-instruct",
            "endpoint": "http://vllm-qwen3-vl-32b:8000",
            "model_id": "Qwen/Qwen3-VL-32B-Instruct",
            "modality": "text+image->text",
            "supports_json_mode": true,
            "supports_strict_image_json": false,
            "strict_image_json_status": "pending_smoke"
          }
        ]
        """,
        encoding="utf-8",
    )
    registry = VLLMRegistry(live, demanded_config_path=demanded)
    monkeypatch.setattr(models_api.app_state, "registry", CompositeRegistry([registry]))

    result = await models_api.list_model_catalog(_=object())

    assert result.data == []
    assert len(result.unavailable) == 1
    unavailable = result.unavailable[0]
    assert unavailable.id == "qwen3-vl-32b-instruct:vllm"
    assert unavailable.reason == "demanded_not_configured"
    assert unavailable.backend == "vllm"
    assert unavailable.format == "vllm"
    assert unavailable.available is False
    assert unavailable.upstream_reachable is False
    assert unavailable.availability_status == "demanded_not_configured"
    assert "Qwen/Qwen3-VL-32B-Instruct" in unavailable.detail
