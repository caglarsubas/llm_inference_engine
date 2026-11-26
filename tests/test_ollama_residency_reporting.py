import asyncio

import httpx
import pytest
from fastapi.testclient import TestClient

from inference_engine import model_substitution
from inference_engine.api.state import app_state
from inference_engine.config import settings
from inference_engine.main import app
from inference_engine.model_substitution import ResidencyCache


@pytest.fixture
def install_probe(monkeypatch):
    """Serve /api/ps locally and count probes across health, metrics and routing."""
    original_client = httpx.AsyncClient
    monkeypatch.setattr(app_state, "readiness", lambda: {"ready": True, "status": "ready"})
    monkeypatch.setattr(app_state.registry, "list_models", lambda: [])

    def install(payload, status_code=200, endpoint="http://private-ollama.test"):
        requests = []

        def handle(request):
            requests.append(request.url.path)
            return httpx.Response(status_code, json=payload)

        transport = httpx.MockTransport(handle)
        monkeypatch.setattr(
            httpx, "AsyncClient", lambda **kw: original_client(transport=transport, **kw)
        )
        cache = ResidencyCache(clock=lambda: 0)
        monkeypatch.setattr(app_state, "ollama_residency", cache)
        monkeypatch.setattr(settings, "ollama_http_endpoint", endpoint)
        return requests

    return install


def test_health_and_metrics_share_observed_residency(install_probe, monkeypatch):
    requests = install_probe(
        {
            "models": [
                {"name": "ministral-3:8b", "model": "alias:8b", "size": 1000, "size_vram": 750},
                {"model": "cpu-only", "size_vram": 0},
            ]
        },
        endpoint="http://private-ollama.test/",
    )
    cached_clients = [{"model": "other:8b", "backend": "ollama_http"}]
    monkeypatch.setattr(app_state.manager, "cached_upstream_clients", lambda: cached_clients)
    endpoint = settings.ollama_http_endpoint.rstrip("/")
    app_state.ollama_residency.note_loading(endpoint, "pending:8b")
    client = TestClient(app)
    health = client.get("/v1/health").json()
    assert health["cached_upstream_clients"] == cached_clients
    assert health["ollama_residency"] == {
        "status": "observed",
        "models": [
            {"model": "ministral-3:8b", "size_bytes": 1000, "size_vram_bytes": 750},
            {"model": "cpu-only:latest", "size_bytes": None, "size_vram_bytes": 0},
        ],
    }

    metrics = client.get("/v1/metrics").text.splitlines()
    assert "inference_engine_cached_upstream_clients 1" in metrics
    assert 'inference_engine_ollama_residency_status{state="observed"} 1' in metrics
    assert "inference_engine_ollama_resident_models 2" in metrics
    assert 'inference_engine_ollama_resident_bytes{model="ministral-3:8b"} 1000' in metrics
    assert 'inference_engine_ollama_resident_vram_bytes{model="ministral-3:8b"} 750' in metrics
    assert 'inference_engine_ollama_resident_vram_bytes{model="cpu-only:latest"} 0' in metrics
    assert not any('resident_bytes{model="cpu-only:latest"}' in line for line in metrics)
    assert asyncio.run(app_state.ollama_residency.resident(endpoint)) == {
        "ministral-3:8b",
        "alias:8b",
        "cpu-only:latest",
        "pending:8b",
    }
    assert requests == ["/api/ps"]


@pytest.mark.parametrize(
    ("endpoint", "payload", "code", "status", "models"),
    [
        ("", {"models": []}, 200, "unconfigured", None),
        ("http://private-ollama.test", {"models": []}, 200, "observed", []),
        ("http://private-ollama.test", {}, 200, "unavailable", None),
        ("http://private-ollama.test", {"models": []}, 503, "unavailable", None),
    ],
)
def test_failed_unconfigured_and_empty_are_distinct(
    install_probe, endpoint, payload, code, status, models
):
    requests = install_probe(payload, code, endpoint)
    client = TestClient(app)
    assert client.get("/v1/ready").status_code == 200
    assert requests == []
    response = client.get("/v1/health")
    assert response.status_code == 200
    assert response.json()["ready"] is True
    assert response.json()["ollama_residency"] == {"status": status, "models": models}

    metrics = client.get("/v1/metrics").text
    assert f'inference_engine_ollama_residency_status{{state="{status}"}} 1' in metrics.splitlines()
    counts = [
        line
        for line in metrics.splitlines()
        if line.startswith("inference_engine_ollama_resident_models ")
    ]
    assert counts == (["inference_engine_ollama_resident_models 0"] if models == [] else [])
    assert "private-ollama.test" not in response.text
    assert "private-ollama.test" not in metrics
    assert requests == (["/api/ps"] if endpoint else [])


async def test_concurrent_scrapes_share_one_bounded_probe(monkeypatch):
    calls = 0

    async def stall(request):
        nonlocal calls
        calls += 1
        await asyncio.Event().wait()

    original_client = httpx.AsyncClient
    transport = httpx.MockTransport(stall)
    monkeypatch.setattr(
        httpx, "AsyncClient", lambda **kw: original_client(transport=transport, **kw)
    )
    monkeypatch.setattr(model_substitution, "PROBE_TIMEOUT_SECONDS", 0.01)
    cache = ResidencyCache()
    endpoint = "http://private-ollama.test"
    async with asyncio.timeout(1):
        results = await asyncio.gather(cache.report(endpoint), cache.report(endpoint))
    assert [r.status for r in results] == ["unavailable", "unavailable"]
    assert calls == 1
