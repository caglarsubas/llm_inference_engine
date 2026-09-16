"""HTTP-level standardization contract.

The ``error`` envelope, server-owned ``x-request-id``, the per-request
scheduler admission headers, the root ``/metrics`` alias, and the tokenizer
routes — all exercised through the real ASGI app so middleware and exception
handlers are in the path.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterable
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from inference_engine.adapters import GenerationParams, InferenceAdapter, StreamChunk
from inference_engine.adapters.base import GenerationResult
from inference_engine.api import _scheduling
from inference_engine.api.state import app_state
from inference_engine.cancellation import Cancellation
from inference_engine.main import app
from inference_engine.manager import ModelNotFoundError
from inference_engine.registry import ModelDescriptor
from inference_engine.scheduler import SchedulerLease


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
    assert response.headers["x-request-id"].startswith("req_")


# --- request id --------------------------------------------------------------


def test_response_always_carries_a_request_id(client: TestClient) -> None:
    response = client.get("/v1/health")
    assert response.headers["x-request-id"].startswith("req_")


def test_inbound_request_id_cannot_override_the_server_owned_id(client: TestClient) -> None:
    response = client.get("/v1/health", headers={"x-request-id": "caller-abc"})
    assert response.headers["x-request-id"].startswith("req_")
    assert response.headers["x-request-id"] != "caller-abc"


def test_onion_runtime_request_id_does_not_alias_the_engine_id(client: TestClient) -> None:
    response = client.get(
        "/v1/health", headers={"x-onion-runtime-request-id": "run-42"}
    )
    assert response.headers["x-request-id"].startswith("req_")
    assert response.headers["x-request-id"] != "run-42"


def test_repeated_caller_request_id_gets_a_distinct_engine_id_per_request(
    client: TestClient,
) -> None:
    first = client.get("/v1/health", headers={"x-request-id": "same-caller-id"})
    second = client.get("/v1/health", headers={"x-request-id": "same-caller-id"})

    assert first.headers["x-request-id"] != second.headers["x-request-id"]


def test_oversized_onion_identity_is_rejected_without_truncating(
    client: TestClient,
) -> None:
    response = client.get(
        "/v1/health",
        headers={"x-onion-model-attempt-id": "z" * 257},
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_model_invocation_identity"
    assert response.json()["error"]["param"] == "x-onion-model-attempt-id"
    assert response.headers["x-request-id"].startswith("req_")


def test_duplicate_onion_identity_header_is_rejected(client: TestClient) -> None:
    response = client.get(
        "/v1/health",
        headers=[
            ("x-onion-model-invocation-id", "invocation-1"),
            ("x-onion-model-invocation-id", "invocation-2"),
        ],
    )

    assert response.status_code == 400
    assert response.json()["error"]["param"] == "x-onion-model-invocation-id"


@pytest.mark.parametrize(
    "header",
    [
        "x-onion-runtime-request-id",
        "x-onion-model-invocation-id",
        "x-onion-model-attempt-id",
    ],
)
@pytest.mark.parametrize(
    "value",
    ["null", "NULL", "NoNe", "nil", "NIL", "undefined", "UnDeFiNeD"],
)
def test_flattened_null_identity_sentinels_fail_before_priced_execution(
    client: TestClient,
    header: str,
    value: str,
) -> None:
    response = client.post(
        "/v1/chat/completions",
        json={"model": "unused", "messages": [{"role": "user", "content": "hi"}]},
        headers={header: value},
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_model_invocation_identity"
    assert response.json()["error"]["param"] == header
    assert response.headers["x-request-id"].startswith("req_")
    assert "x-onion-usage-record-id" not in response.headers


@pytest.mark.parametrize(
    ("headers", "invalid_header"),
    [
        (
            {"x-onion-model-invocation-id": "invocation-1"},
            "x-onion-model-invocation-id",
        ),
        (
            {"x-onion-model-attempt-id": "attempt-1"},
            "x-onion-model-attempt-id",
        ),
        (
            {
                "x-onion-runtime-request-id": "runtime-1",
                "x-onion-model-attempt-id": "attempt-1",
            },
            "x-onion-model-attempt-id",
        ),
    ],
)
def test_onion_identity_hierarchy_fails_before_priced_execution(
    client: TestClient,
    headers: dict[str, str],
    invalid_header: str,
) -> None:
    response = client.post(
        "/v1/chat/completions",
        json={"model": "unused", "messages": [{"role": "user", "content": "hi"}]},
        headers=headers,
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_model_invocation_identity"
    assert response.json()["error"]["param"] == invalid_header
    assert response.headers["x-request-id"].startswith("req_")
    assert "x-onion-usage-record-id" not in response.headers


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


# --- per-request scheduler admission -----------------------------------------


class _AdmissionAdapter(InferenceAdapter):
    """Smallest adapter that can be admitted, generated from, and streamed."""

    backend_name = "header-test"

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
        return GenerationResult(
            text="ok", finish_reason="stop", prompt_tokens=3, completion_tokens=1
        )

    async def stream(
        self, messages: Iterable, params: GenerationParams, cancel: Cancellation | None = None
    ) -> AsyncIterator[StreamChunk]:
        yield StreamChunk(text="ok")
        yield StreamChunk(text="", finish_reason="stop")


_ADMISSION_MODEL = "fake-model:1b"
_ADMISSION_RESOURCE = f"{_AdmissionAdapter.backend_name}:{_ADMISSION_MODEL}"


@pytest.fixture
def admitted(monkeypatch) -> None:
    """Install one servable model so a chat request reaches the scheduler."""

    async def _get(model_id: str):
        if model_id != _ADMISSION_MODEL:
            raise ModelNotFoundError(model_id)
        name, tag = model_id.rsplit(":", 1)
        return _AdmissionAdapter(), ModelDescriptor(
            name=name,
            tag=tag,
            namespace="test",
            registry="test",
            model_path=Path(f"/tmp/{model_id}"),
            format="gguf",
            size_bytes=1,
        )

    monkeypatch.setattr(app_state.manager, "get", _get)


def _chat_body(**extra) -> dict:
    return {
        "model": _ADMISSION_MODEL,
        "messages": [{"role": "user", "content": "hi"}],
        **extra,
    }


def test_blocking_chat_reports_the_admission_that_served_it(
    client: TestClient, admitted: None
) -> None:
    response = client.post("/v1/chat/completions", json=_chat_body())

    assert response.status_code == 200, response.text
    assert int(response.headers[_scheduling.QUEUE_WAIT_MS_HEADER]) >= 0
    assert response.headers[_scheduling.RESOURCE_HEADER] == _ADMISSION_RESOURCE
    # Both depths count this request, so a lone caller sees 1 rather than 0.
    # A client rendering the raw value as "requests ahead of you" is off by
    # one, which is exactly the misleading progress this channel removes.
    assert response.headers[_scheduling.QUEUE_DEPTH_HEADER] == "1"
    assert response.headers[_scheduling.TENANT_QUEUE_DEPTH_HEADER] == "1"


def test_streaming_admission_headers_land_before_the_first_delta(
    client: TestClient, admitted: None
) -> None:
    """The reason these headers are worth more than a span on the stream path.

    Admission completes before the response is constructed, so the queue wait
    is already known when headers are flushed — the caller reads it at stream
    open, ahead of any content, rather than reconstructing it from a trace
    after the turn is over.
    """
    with client.stream("POST", "/v1/chat/completions", json=_chat_body(stream=True)) as response:
        assert response.status_code == 200
        # Read the headers with the body still unconsumed: this is the moment
        # a client can act on them.
        wait_ms = int(response.headers[_scheduling.QUEUE_WAIT_MS_HEADER])
        assert response.headers[_scheduling.RESOURCE_HEADER] == _ADMISSION_RESOURCE
        assert response.headers[_scheduling.QUEUE_DEPTH_HEADER] == "1"
        body = "".join(response.iter_text())

    assert wait_ms >= 0
    assert '"content":"ok"' in body


def test_a_request_that_never_reached_the_scheduler_reports_no_admission(
    client: TestClient, admitted: None
) -> None:
    """No admission, no headers — never a fabricated zero.

    A caller must be able to tell "I did not queue" from "I queued for 0 ms",
    because only the first means the number is absent.
    """
    response = client.post("/v1/chat/completions", json=_chat_body(model="nope:9b"))

    assert response.status_code >= 400
    assert _scheduling.QUEUE_WAIT_MS_HEADER not in response.headers
    assert _scheduling.RESOURCE_HEADER not in response.headers


def _lease(**overrides) -> SchedulerLease:
    fields = {
        "lease_id": 1,
        "tenant": "dev",
        "resource_key": _ADMISSION_RESOURCE,
        "workload": "chat.generate",
        "priority": 20.0,
        "estimated_tokens": 8,
        "wait_ms": 4120.4,
        "queue_depth_at_submit": 3,
        "tenant_queue_depth_at_submit": 1,
    }
    return SchedulerLease(**{**fields, **overrides})


def test_a_re_admitted_request_still_reports_the_admission_it_started_on() -> None:
    """Fallback and schema-repair retries queue again; the headers do not move.

    On a stream they could not: headers are flushed when the stream opens,
    before any re-admission exists. Holding the blocking path to the same rule
    keeps one meaning for the value instead of one per path.
    """
    telemetry = _scheduling.begin_admission()
    _scheduling.bind_admission(_lease())
    _scheduling.bind_admission(_lease(wait_ms=99.0, resource_key="other:model:7b"))

    headers = _scheduling.admission_headers(telemetry)
    assert headers[_scheduling.QUEUE_WAIT_MS_HEADER] == "4120"
    assert headers[_scheduling.RESOURCE_HEADER] == _ADMISSION_RESOURCE


def test_an_unwritable_resource_key_drops_only_its_own_header() -> None:
    """A model id the registry reported must not be able to fail a response.

    The numbers still go out; only the name that cannot be encoded is dropped.
    """
    telemetry = _scheduling.begin_admission()
    _scheduling.bind_admission(_lease(resource_key="ollama:modèle:7b"))

    headers = _scheduling.admission_headers(telemetry)
    assert _scheduling.RESOURCE_HEADER not in headers
    assert headers[_scheduling.QUEUE_WAIT_MS_HEADER] == "4120"
    assert headers[_scheduling.QUEUE_DEPTH_HEADER] == "3"


def test_binding_outside_a_request_scope_is_inert() -> None:
    """Route coroutines are called directly all over this suite; that must work."""
    _scheduling.bind_admission(_lease())
    assert _scheduling.admission_headers(None) == {}
