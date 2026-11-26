import json

import httpx
import pytest

from scripts import qualify_local_model_routes as checks


DIGEST = "3a" * 32
GATEWAY = "http://127.0.0.1:9001"
OLLAMA = "http://127.0.0.1:9002"


@pytest.mark.parametrize("tags_prefix,ps_prefix", [("sha256:", ""), ("", "sha256:")])
def test_missing_route_is_incomplete_and_next_route_can_qualify(tags_prefix, ps_prefix):
    def handle(request):
        path = request.url.path
        if path == "/v1/models.data":
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "id": "toy:1b",
                            "backend": "ollama_http",
                            "endpoint": OLLAMA,
                            "artifact_identity": {
                                "source": "ollama_tags",
                                "manifest_sha256": DIGEST,
                            },
                        }
                    ]
                },
            )
        if path in {"/api/tags", "/api/ps"}:
            prefix = tags_prefix if path == "/api/tags" else ps_prefix
            return httpx.Response(
                200, json={"models": [{"name": "toy:1b", "digest": prefix + DIGEST}]}
            )
        if path == "/api/show":
            return httpx.Response(200, json={"capabilities": ["completion"]})
        assert path == "/v1/chat/completions"
        prompt = json.loads(request.content)["messages"][0]["content"]
        label = next(label for _, text, label in checks.CHAT_CASES if text == prompt)
        return httpx.Response(
            200,
            json={
                "model": "toy:1b",
                "choices": [{"message": {"content": label}, "finish_reason": "stop"}],
                "usage": {"completion_tokens": 5},
            },
        )

    transport = httpx.MockTransport(handle)
    with (
        httpx.Client(base_url=GATEWAY, transport=transport) as gateway,
        httpx.Client(base_url=OLLAMA, transport=transport) as ollama,
    ):
        assert checks.qualify(gateway, ollama, "missing")["error"] == "route_not_advertised"
        row = checks.qualify(gateway, ollama, "toy:1b")
    assert row["status"] == "passed" and row["summary"]["correct"] == 3
    assert row["digest"] == "sha256:" + DIGEST


def test_docker_tag_replaced_while_gateway_cache_still_has_a_match(monkeypatch):
    inventory_calls = 0
    remote_id = "docker.io/local/toy:q4"

    def handle(request):
        nonlocal inventory_calls
        if request.url.path == "/v1/models.data":
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "id": "toy:dmr",
                            "backend": "vllm",
                            "upstream_model_id": remote_id,
                            "endpoint": "http://127.0.0.1:9003/proxy/engines/llama.cpp",
                            "artifact_identity": {
                                "source": "docker_model_runner",
                                "model_sha256": DIGEST,
                            },
                        }
                    ]
                },
            )
        assert request.url == "http://127.0.0.1:9003/proxy/api/tags"
        inventory_calls += 1
        digest = DIGEST if inventory_calls == 1 else "f" * 64
        return httpx.Response(
            200, json={"models": [{"name": remote_id, "digest": "sha256:" + digest}]}
        )

    monkeypatch.setattr(checks, "chat_check", lambda *args: True)
    transport = httpx.MockTransport(handle)
    with (
        httpx.Client(base_url=GATEWAY, transport=transport) as gateway,
        httpx.Client(base_url=OLLAMA, transport=transport) as ollama,
    ):
        row = checks.qualify(gateway, ollama, "toy:dmr")
    assert inventory_calls == 2
    assert row["status"] == "incomplete" and row["error"] == "docker_model_id_changed"


def test_finite_vectors_with_a_top_score_tie_fail(monkeypatch):
    response = {
        "model": "toy:embed",
        "data": [{"index": i, "embedding": [1.0, 2.0]} for i in range(6)],
    }
    monkeypatch.setattr(checks, "request", lambda *args: response)
    row = {"rows": []}
    assert not checks.embedding_check(None, "toy:embed", 2, row)
    assert row["summary"]["valid_vectors"] == 6
    assert row["summary"]["correct"] == 0
