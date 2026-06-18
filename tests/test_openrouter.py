from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from inference_engine.adapters.base import GenerationParams, UpstreamGenerationError
from inference_engine.adapters.openrouter_adapter import OpenRouterAdapter
from inference_engine.api import models as models_api
from inference_engine.config import settings
from inference_engine.registry import (
    CompositeRegistry,
    ModelDescriptor,
    OpenRouterProbe,
    OpenRouterProbeResult,
    OpenRouterRegistry,
)
from inference_engine.schemas import ChatMessage

ROOT = Path(__file__).resolve().parents[1]


def _write_models(path: Path, entries: list[dict]) -> None:
    path.write_text(json.dumps(entries), encoding="utf-8")


def _entry(**overrides) -> dict:
    data = {
        "name": "llama-3.1-70b-instruct",
        "model_id": "meta-llama/llama-3.1-70b-instruct",
        "parameter_count_b": 70,
        "open_weight": True,
        "open_source": True,
        "proprietary": False,
    }
    data.update(overrides)
    return data


def _descriptor(
    *,
    endpoint: str = "https://openrouter.ai/api",
    model_id: str = "meta-llama/llama-3.1-70b-instruct",
) -> ModelDescriptor:
    return ModelDescriptor(
        name="llama-3.1-70b-instruct",
        tag="openrouter",
        namespace="openrouter",
        registry="openrouter",
        model_path=Path(f"openrouter://{model_id}"),
        format="openrouter",
        params={
            "model_id": model_id,
            "parameter_count_b": 70,
            "open_weight": True,
            "open_source": True,
            "proprietary": False,
            "request_key_source": "openrouter-api-key",
            "provider": "openrouter",
            "modality": "text+image->text",
            "context_length": 131072,
            "max_image_side_px": 1024,
            "supports_json_mode": True,
            "supports_strict_image_json": False,
            "strict_image_json_status": "failed",
            "strict_image_json_checked_at": "2026-06-18",
            "strict_image_json_detail": "parse coverage 0/12",
            "family": "Llama",
            "profile": "vision",
        },
        endpoint=endpoint,
    )


def test_registry_parses_large_open_weight_entry(tmp_path: Path) -> None:
    path = tmp_path / "openrouter.json"
    _write_models(path, [_entry()])

    desc = OpenRouterRegistry(
        path,
        default_endpoint="https://openrouter.ai/api",
        min_parameter_count_b=50,
    ).list_models()[0]

    assert desc.format == "openrouter"
    assert desc.qualified_name == "llama-3.1-70b-instruct:openrouter"
    assert desc.endpoint == "https://openrouter.ai/api"
    assert desc.params["model_id"] == "meta-llama/llama-3.1-70b-instruct"
    assert desc.params["request_key_source"] == "openrouter-api-key"


def test_example_catalog_is_policy_compliant() -> None:
    reg = OpenRouterRegistry(
        ROOT / ".openrouter_models.example.json",
        default_endpoint="https://openrouter.ai/api",
        min_parameter_count_b=50,
    )

    descs = reg.list_models()
    assert len(descs) >= 20
    assert len({d.qualified_name for d in descs}) == len(descs)
    assert all(d.format == "openrouter" for d in descs)
    assert all(d.params["request_key_source"] == "openrouter-api-key" for d in descs)
    assert all(float(d.params["parameter_count_b"]) > 50 for d in descs)
    assert {"qwen3-vl-235b-a22b-instruct:openrouter", "hermes-4-405b:openrouter"} <= {
        d.qualified_name for d in descs
    }
    qwen3_vl = {
        d.qualified_name: d
        for d in descs
        if d.qualified_name.startswith("qwen3-vl-235b-a22b")
    }
    assert qwen3_vl
    assert all(d.params["supports_strict_image_json"] is False for d in qwen3_vl.values())
    assert (
        qwen3_vl["qwen3-vl-235b-a22b-instruct:openrouter"].params[
            "strict_image_json_status"
        ]
        == "unstable"
    )
    assert (
        qwen3_vl["qwen3-vl-235b-a22b-thinking:openrouter"].params[
            "strict_image_json_status"
        ]
        == "failed"
    )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"parameter_count_b": 50}, "must be > 50"),
        ({"open_weight": False}, "open_weight=true"),
        ({"open_source": False}, "open_source=false"),
        ({"proprietary": True}, "proprietary=false"),
    ],
)
def test_registry_rejects_models_outside_openrouter_policy(
    tmp_path: Path,
    overrides: dict,
    message: str,
) -> None:
    path = tmp_path / "openrouter.json"
    _write_models(path, [_entry(**overrides)])

    reg = OpenRouterRegistry(
        path,
        default_endpoint="https://openrouter.ai/api",
        min_parameter_count_b=50,
    )
    with pytest.raises(ValueError, match=message):
        reg.list_models()


def test_openrouter_probe_requires_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "openrouter_api_key", "")

    result = OpenRouterProbe(timeout_seconds=0.1, ttl_seconds=0.0).probe(_descriptor())

    assert result.loadable is False
    assert result.reason == "openrouter_api_key_missing"


def test_openrouter_probe_sends_authorization(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "openrouter_api_key", "sk-or-test")
    seen_headers: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_headers.append(request.headers["authorization"])
        assert request.url.path.endswith("/v1/models")
        return httpx.Response(
            200,
            json={
                "object": "list",
                "data": [{"id": "meta-llama/llama-3.1-70b-instruct"}],
            },
        )

    def factory(
        base_url: str,
        timeout: httpx.Timeout,
        headers: dict[str, str],
    ) -> httpx.Client:
        return httpx.Client(
            base_url=base_url,
            timeout=timeout,
            headers=headers,
            transport=httpx.MockTransport(handler),
        )

    result = OpenRouterProbe(
        timeout_seconds=0.1,
        ttl_seconds=0.0,
        client_factory=factory,
    ).probe(_descriptor())

    assert result.loadable is True
    assert seen_headers == ["Bearer sk-or-test"]


@pytest.mark.asyncio
async def test_openrouter_adapter_posts_with_key_source_and_bearer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "openrouter_api_key", "sk-or-test")
    captured_headers: list[str] = []
    captured_body: list[dict] = []

    def handler(req: httpx.Request) -> httpx.Response:
        captured_headers.append(req.headers["authorization"])
        captured_body.append(json.loads(req.content))
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-x",
                "object": "chat.completion",
                "model": "meta-llama/llama-3.1-70b-instruct",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "hello"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
            },
        )

    adapter = OpenRouterAdapter()
    await adapter.load(_descriptor())
    assert adapter.request_key_source == "openrouter-api-key"
    assert adapter._client is not None  # noqa: SLF001
    adapter._client = httpx.AsyncClient(  # noqa: SLF001
        base_url=adapter._client.base_url,  # noqa: SLF001
        headers=adapter._client.headers,  # noqa: SLF001
        transport=httpx.MockTransport(handler),
        timeout=30.0,
    )

    result = await adapter.generate(
        [ChatMessage(role="user", content="hi")],
        GenerationParams(max_tokens=16),
    )

    assert result.text == "hello"
    assert captured_headers == ["Bearer sk-or-test"]
    assert captured_body[0]["model"] == "meta-llama/llama-3.1-70b-instruct"


@pytest.mark.asyncio
async def test_openrouter_adapter_preserves_upstream_error_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "openrouter_api_key", "sk-or-test")

    def handler(req: httpx.Request) -> httpx.Response:  # noqa: ARG001
        return httpx.Response(
            429,
            json={
                "error": {
                    "message": "provider rejected image input",
                    "code": "provider_error",
                }
            },
        )

    adapter = OpenRouterAdapter()
    await adapter.load(_descriptor())
    assert adapter._client is not None  # noqa: SLF001
    adapter._client = httpx.AsyncClient(  # noqa: SLF001
        base_url=adapter._client.base_url,  # noqa: SLF001
        headers=adapter._client.headers,  # noqa: SLF001
        transport=httpx.MockTransport(handler),
        timeout=30.0,
    )

    with pytest.raises(UpstreamGenerationError) as ei:
        await adapter.generate(
            [ChatMessage(role="user", content="score this image")],
            GenerationParams(max_tokens=16),
        )

    assert ei.value.error_type == "upstream_http_error"
    assert ei.value.upstream_status_code == 429
    assert ei.value.backend == "openrouter"
    assert ei.value.model == "meta-llama/llama-3.1-70b-instruct"
    assert "provider rejected image input" in ei.value.detail
    assert "provider_error" in ei.value.detail


class _Source:
    def __init__(self, descriptor: ModelDescriptor) -> None:
        self.descriptor = descriptor

    def list_models(self) -> list[ModelDescriptor]:
        return [self.descriptor]

    def get(self, name_with_tag: str) -> ModelDescriptor | None:
        if name_with_tag in {self.descriptor.qualified_name, self.descriptor.name}:
            return self.descriptor
        return None


class _FakeOpenRouterProbe:
    def probe(self, descriptor: ModelDescriptor) -> OpenRouterProbeResult:  # noqa: ARG002
        return OpenRouterProbeResult(
            loadable=False,
            reason="openrouter_api_key_missing",
            detail="OPENROUTER_API_KEY is not set",
        )


class _FakePassingOpenRouterProbe:
    def probe(self, descriptor: ModelDescriptor) -> OpenRouterProbeResult:  # noqa: ARG002
        return OpenRouterProbeResult(loadable=True)


@pytest.mark.asyncio
async def test_models_api_keeps_unusable_openrouter_out_of_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(models_api.app_state, "registry", CompositeRegistry([_Source(_descriptor())]))
    monkeypatch.setattr(models_api, "get_openrouter_probe", lambda: _FakeOpenRouterProbe())

    result = await models_api.list_models(_=object())

    assert result.data == []
    assert len(result.unavailable) == 1
    assert result.unavailable[0].id == "llama-3.1-70b-instruct:openrouter"
    assert result.unavailable[0].reason == "openrouter_api_key_missing"
    assert result.unavailable[0].backend == "openrouter"


@pytest.mark.asyncio
async def test_models_data_returns_openrouter_catalog_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(models_api.app_state, "registry", CompositeRegistry([_Source(_descriptor())]))
    monkeypatch.setattr(models_api, "get_openrouter_probe", lambda: _FakePassingOpenRouterProbe())

    result = await models_api.list_model_catalog(_=object())

    assert result.object == "model_catalog"
    assert result.unavailable == []
    assert len(result.data) == 1
    entry = result.data[0]
    assert entry.id == "llama-3.1-70b-instruct:openrouter"
    assert entry.provider == "openrouter"
    assert entry.backend == "openrouter"
    assert entry.upstream_model_id == "meta-llama/llama-3.1-70b-instruct"
    assert entry.modality == "text+image->text"
    assert entry.supports_images is True
    assert entry.context_length == 131072
    assert entry.max_image_side_px == 1024
    assert entry.supports_json_mode is True
    assert entry.supports_strict_image_json is False
    assert entry.strict_image_json_status == "failed"
    assert entry.strict_image_json_checked_at == "2026-06-18"
    assert entry.strict_image_json_detail == "parse coverage 0/12"
    assert entry.request_key_source == "openrouter-api-key"
