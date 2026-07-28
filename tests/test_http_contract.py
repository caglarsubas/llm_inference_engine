"""HTTP-level standardization contract.

The ``error`` envelope, ``x-request-id`` propagation, the root ``/metrics``
alias, and the tokenizer routes — all exercised through the real ASGI app so
middleware and exception handlers are in the path.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from inference_engine.api.state import app_state
from inference_engine.main import app


@pytest.fixture(autouse=True)
def _ready():
    app_state.mark_ready()
    yield
    app_state.mark_ready()


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


# --- error envelope ----------------------------------------------------------


def test_unknown_model_returns_both_detail_and_error(client: TestClient) -> None:
    response = client.post(
        "/v1/chat/completions",
        json={"model": "definitely-not-a-model:9b", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code >= 400
    body = response.json()
    # ``detail`` preserved for existing Prometha consumers...
    assert "detail" in body
    # ...``error`` added for OpenAI SDKs.
    assert "error" in body
    assert isinstance(body["error"]["message"], str)
    assert body["error"]["type"]
    assert "code" in body["error"]
    assert "param" in body["error"]


def test_validation_failure_carries_an_error_object(client: TestClient) -> None:
    response = client.post("/v1/chat/completions", json={"messages": []})
    assert response.status_code == 422
    error = response.json()["error"]
    assert error["type"] == "invalid_request_error"
    assert error["errors"], "field-level errors should survive"


def test_top_logprobs_without_logprobs_is_rejected(client: TestClient) -> None:
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "demo:1b",
            "messages": [{"role": "user", "content": "hi"}],
            "top_logprobs": 3,
        },
    )
    assert response.status_code == 422
    assert response.json()["error"]["type"] == "invalid_request_error"


def test_startup_503_also_carries_the_error_envelope(client: TestClient) -> None:
    app_state.mark_starting()
    response = client.get("/v1/models")

    assert response.status_code == 503
    body = response.json()
    assert body["detail"]["type"] == "engine_starting"
    assert body["error"]["type"] == "engine_starting"
    assert response.headers["retry-after"] == "5"


# --- request id --------------------------------------------------------------


def test_response_always_carries_a_request_id(client: TestClient) -> None:
    response = client.get("/v1/health")
    assert response.headers["x-request-id"].startswith("req_")


def test_inbound_request_id_is_echoed(client: TestClient) -> None:
    response = client.get("/v1/health", headers={"x-request-id": "caller-abc"})
    assert response.headers["x-request-id"] == "caller-abc"


def test_orchestra_runtime_request_id_is_honoured(client: TestClient) -> None:
    """The SDK's model gateway sends its correlation id under its own header."""
    response = client.get(
        "/v1/health", headers={"x-orchestra-runtime-request-id": "run-42"}
    )
    assert response.headers["x-request-id"] == "run-42"


def test_explicit_request_id_wins_over_the_orchestra_header(client: TestClient) -> None:
    response = client.get(
        "/v1/health",
        headers={"x-request-id": "primary", "x-orchestra-runtime-request-id": "secondary"},
    )
    assert response.headers["x-request-id"] == "primary"


def test_oversized_request_id_is_truncated(client: TestClient) -> None:
    response = client.get("/v1/health", headers={"x-request-id": "z" * 5000})
    assert len(response.headers["x-request-id"]) == 200


def test_error_responses_also_carry_a_request_id(client: TestClient) -> None:
    response = client.post("/v1/chat/completions", json={"messages": []})
    assert response.status_code == 422
    assert response.headers["x-request-id"]


# --- metrics -----------------------------------------------------------------


def test_metrics_available_at_the_conventional_root_path(client: TestClient) -> None:
    root = client.get("/metrics")
    versioned = client.get("/v1/metrics")

    assert root.status_code == 200
    assert versioned.status_code == 200
    assert "inference_engine_info" in root.text


def test_metrics_reachable_while_starting(client: TestClient) -> None:
    """Prometheus must be able to scrape a replica that is still warming up."""
    app_state.mark_starting()
    assert client.get("/v1/metrics").status_code == 200
    assert client.get("/metrics").status_code == 200


def test_genai_histograms_render_after_an_observation(client: TestClient) -> None:
    from inference_engine.genai_metrics import genai_metrics

    genai_metrics.record_operation(
        operation="chat", provider="llama_cpp", model="probe:1b", duration_seconds=0.2,
        input_tokens=10, output_tokens=5,
    )
    body = client.get("/metrics").text
    assert "gen_ai_client_operation_duration_seconds_bucket" in body
    assert "gen_ai_client_token_usage_bucket" in body


# --- tokenizer routes --------------------------------------------------------


def test_tokenize_rejects_both_prompt_and_messages(client: TestClient) -> None:
    response = client.post(
        "/tokenize",
        json={
            "model": "demo:1b",
            "prompt": "hi",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert response.status_code == 422


def test_tokenize_rejects_neither_prompt_nor_messages(client: TestClient) -> None:
    response = client.post("/tokenize", json={"model": "demo:1b"})
    assert response.status_code == 422


def test_tokenize_is_gated_by_startup_readiness(client: TestClient) -> None:
    """It lives outside /v1/ but still touches the model manager."""
    app_state.mark_starting()
    response = client.post("/tokenize", json={"model": "demo:1b", "prompt": "hi"})

    assert response.status_code == 503
    assert response.json()["error"]["type"] == "engine_starting"


def test_detokenize_rejects_an_oversized_token_array(client: TestClient) -> None:
    from inference_engine.api.tokenize import _MAX_DETOKENIZE_TOKENS

    response = client.post(
        "/detokenize",
        json={"model": "demo:1b", "tokens": [1] * (_MAX_DETOKENIZE_TOKENS + 1)},
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "tokens_too_many"


def test_tokenize_unknown_model_is_a_typed_404(client: TestClient) -> None:
    response = client.post(
        "/tokenize", json={"model": "definitely-not-a-model:9b", "prompt": "hi"}
    )
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "model_not_found"
