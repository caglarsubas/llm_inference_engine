from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from inference_engine.api.state import app_state
from inference_engine.main import app


@pytest.fixture(autouse=True)
def _reset_readiness():
    app_state.mark_ready()
    yield
    app_state.mark_ready()


def test_health_stays_reachable_while_starting() -> None:
    app_state.mark_starting()

    response = TestClient(app).get("/v1/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "starting"
    assert body["ready"] is False
    assert body["readiness"]["status"] == "starting"


def test_ready_endpoint_returns_retryable_503_while_starting() -> None:
    app_state.mark_starting()

    response = TestClient(app).get("/v1/ready")

    assert response.status_code == 503
    assert response.headers["retry-after"] == "5"
    body = response.json()
    assert body["status"] == "starting"
    assert body["ready"] is False


def test_openai_routes_return_typed_503_while_starting() -> None:
    app_state.mark_starting()

    response = TestClient(app).post(
        "/v1/chat/completions",
        json={"model": "demo:1b", "messages": [{"role": "user", "content": "hi"}]},
    )

    assert response.status_code == 503
    assert response.headers["retry-after"] == "5"
    detail = response.json()["detail"]
    assert detail["type"] == "engine_starting"
    assert detail["code"] == "engine_starting"
    assert detail["status"] == "starting"
    assert detail["ready"] is False


def test_startup_failure_returns_typed_503() -> None:
    app_state.mark_starting()
    app_state.mark_startup_failed(RuntimeError("probe boom"))

    response = TestClient(app).get("/v1/models")

    assert response.status_code == 503
    detail = response.json()["detail"]
    assert detail["type"] == "engine_startup_failed"
    assert detail["code"] == "engine_startup_failed"
    assert detail["status"] == "error"
    assert "RuntimeError: probe boom" in detail["startup_error"]
