from pathlib import Path

import httpx
import pytest

from inference_engine.api import models
from inference_engine.registry import ModelDescriptor
from inference_engine.registry.vllm_probe import VLLMUpstreamProbe


MODEL_ID = "docker.io/local/example:q4_k_m"
DIGEST = "3e" * 32


@pytest.mark.parametrize(
    "inventory", ["matched", "wrong_model", "conflict", "unavailable", "ordinary", "ambiguous"]
)
def test_catalog_reports_only_the_matching_docker_model(monkeypatch, inventory):
    requests = []
    quantization = "MOSTLY_Q4_K_M"

    def respond(request):
        requests.append(request.url.path)
        if request.url.path == "/proxy/engines/llama.cpp/v1/models":
            model = {"id": MODEL_ID}
            if inventory != "ordinary":
                model.update(owned_by="docker", dmr={"quantization": quantization})
            entries = [model, {"id": MODEL_ID}] if inventory == "ambiguous" else [model]
            return httpx.Response(200, json={"data": entries})
        assert request.url.path == "/proxy/api/tags"
        if inventory == "unavailable":
            return httpx.Response(503)
        tag = {
            "name": "unrelated:latest" if inventory == "wrong_model" else MODEL_ID,
            "digest": "sha256:" + DIGEST,
            "details": {"quantization_level": "F16" if inventory == "conflict" else quantization},
        }
        return httpx.Response(200, json={"models": [tag]})

    def client(url, timeout):
        return httpx.Client(base_url=url, timeout=timeout, transport=httpx.MockTransport(respond))

    probe = VLLMUpstreamProbe(client_factory=client)
    monkeypatch.setattr(models, "get_vllm_probe", lambda: probe)
    desc = ModelDescriptor(
        name="example",
        tag="server",
        registry="local",
        namespace="example",
        model_path=Path("example"),
        endpoint="http://test/proxy/engines/llama.cpp",
        format="vllm",
        params={"model_id": MODEL_ID},
    )
    assert probe.probe(desc).loadable
    entry = models._catalog_entry(desc)
    if inventory in {"ordinary", "ambiguous"}:
        assert entry.artifact_identity is None
        assert requests == ["/proxy/engines/llama.cpp/v1/models"]
    else:
        assert entry.artifact_identity.source == "docker_model_runner"
        assert entry.artifact_identity.quantization == quantization
        assert entry.artifact_identity.model_sha256 == (DIGEST if inventory == "matched" else None)
        assert requests == ["/proxy/engines/llama.cpp/v1/models", "/proxy/api/tags"]
    assert entry.workload_evaluations == []
