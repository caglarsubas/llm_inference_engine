"""Every request this engine sends Ollama must carry the configured ``num_ctx``.

Ollama sizes a runner by the ``options.num_ctx`` of whichever request loads it,
and a request that asks for a different window evicts the runner and reloads
it. So one path that sends the wrong value, or none, costs more than its own
request. It silently shrinks the window for every caller that shares the model,
and a long prompt then loses its head to front-truncation with nothing in the
response to say so.

That happened on 2026-09-26. The launchd template pinned ``N_CTX=8192`` in the
process environment, which pydantic-settings ranks above ``.env``, so the
deployment's ``N_CTX=32768`` never took effect. The pin was harmless while
chat went through the OpenAI shim, which ignores ``num_ctx``. Once #112 moved
chat to native ``/api/chat``, every request sent ``num_ctx=8192`` and
``ministral-3:8b`` reloaded at 8192 per slot: a ~20,000-token prompt arrived
as 1,983.

The tests below pin the value on the wire for each adapter entry point and
for each route that reaches one, and pin the deployment files so the template
cannot shadow ``.env`` again.
"""

from __future__ import annotations

import json
import plistlib
import re
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from inference_engine import model_substitution as ms
from inference_engine.adapters.base import GenerationParams
from inference_engine.adapters.ollama_http import OllamaHttpAdapter
from inference_engine.api.state import app_state
from inference_engine.config import Settings, settings
from inference_engine.evals import EvalRunner
from inference_engine.main import app
from inference_engine.manager import ModelNotFoundError
from inference_engine.registry import ModelDescriptor
from inference_engine.schemas import ChatMessage

REPO = Path(__file__).resolve().parents[1]
ENDPOINT = "http://ollama.test:11434"
MINISTRAL = "ministral-3:8b"
GEMMA = "gemma4:26b"
QWEN = "qwen3.8:27b"
N_CTX = 32768
# What ``POST /api/show`` reports for these models. Larger than N_CTX, so the
# ceiling rather than the trained window is what reaches the wire.
TRAINED = 262144

_JUDGE_JSON = '{"score": 4, "justification": "fine"}'


def _descriptor(model_id: str) -> ModelDescriptor:
    name, tag = model_id.rsplit(":", 1)
    return ModelDescriptor(
        name=name,
        tag=tag,
        namespace="library",
        registry="registry.ollama.ai",
        model_path=Path(f"ollama_http://{ENDPOINT}/{model_id}"),
        format="ollama_http",
        params={"model_id": model_id, "context_length": TRAINED},
        size_bytes=0,
        endpoint=ENDPOINT,
    )


class _Ollama:
    """A scripted Ollama that records every request body it receives."""

    def __init__(self) -> None:
        self.requests: list[tuple[str, dict]] = []

    def __call__(self, req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content) if req.content else {}
        self.requests.append((req.url.path, body))
        content = _JUDGE_JSON if body.get("format") else "ok"
        if body.get("stream"):
            lines = [
                {"message": {"role": "assistant", "content": content}, "done": False},
                {
                    "message": {"role": "assistant", "content": ""},
                    "done": True,
                    "done_reason": "stop",
                    "prompt_eval_count": 20000,
                    "eval_count": 1,
                },
            ]
            return httpx.Response(
                200,
                content="".join(json.dumps(line) + "\n" for line in lines).encode(),
                headers={"content-type": "application/x-ndjson"},
            )
        return httpx.Response(
            200,
            json={
                "model": body.get("model"),
                "message": {"role": "assistant", "content": content},
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 20000,
                "eval_count": 1,
            },
        )

    def assert_every_chat_carries(self, n_ctx: int) -> None:
        assert self.requests, "nothing reached Ollama"
        paths = {path for path, _ in self.requests}
        # The OpenAI shim cannot carry ``options`` at all; see the adapter's
        # module docstring. Chat that lands there loads at the host default.
        assert paths == {"/api/chat"}, paths
        assert [body["options"]["num_ctx"] for _, body in self.requests] == [n_ctx] * len(
            self.requests
        )


async def _loaded(model_id: str, ollama: _Ollama) -> OllamaHttpAdapter:
    adapter = OllamaHttpAdapter()
    await adapter.load(_descriptor(model_id))
    assert adapter._client is not None  # noqa: SLF001 - test scaffolding
    adapter._client = httpx.AsyncClient(  # noqa: SLF001
        base_url=adapter._client.base_url,
        transport=httpx.MockTransport(ollama),
        timeout=30.0,
    )
    return adapter


# --- adapter entry points ------------------------------------------------------


async def _generate(adapter: OllamaHttpAdapter) -> None:
    await adapter.generate([ChatMessage(role="user", content="hi")], GenerationParams())


async def _generate_structured(adapter: OllamaHttpAdapter) -> None:
    await adapter.generate(
        [ChatMessage(role="user", content="score it")],
        GenerationParams(
            json_mode=True,
            json_schema={"type": "object", "properties": {"score": {"type": "integer"}}},
        ),
    )


async def _generate_thinking_off(adapter: OllamaHttpAdapter) -> None:
    await adapter.generate(
        [ChatMessage(role="user", content="hi")], GenerationParams(think=False)
    )


async def _stream(adapter: OllamaHttpAdapter) -> None:
    async for _ in adapter.stream([ChatMessage(role="user", content="hi")], GenerationParams()):
        pass


async def _complete(adapter: OllamaHttpAdapter) -> None:
    await adapter.complete("hi", GenerationParams())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "call",
    [_generate, _generate_structured, _generate_thinking_off, _stream, _complete],
    ids=lambda f: f.__name__.lstrip("_"),
)
async def test_every_adapter_entry_point_sends_num_ctx(monkeypatch, call) -> None:
    monkeypatch.setattr(settings, "n_ctx", N_CTX)
    ollama = _Ollama()
    adapter = await _loaded(MINISTRAL, ollama)

    await call(adapter)

    ollama.assert_every_chat_carries(N_CTX)


@pytest.mark.asyncio
async def test_the_empty_multimodal_json_reprompt_keeps_num_ctx(monkeypatch) -> None:
    """The second POST rebuilds ``options``; it must not drop the window."""
    monkeypatch.setattr(settings, "n_ctx", N_CTX)
    seen: list[dict] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(json.loads(req.content))
        content = "" if len(seen) == 1 else _JUDGE_JSON
        return httpx.Response(
            200,
            json={"message": {"role": "assistant", "content": content}, "done": True},
        )

    adapter = OllamaHttpAdapter()
    await adapter.load(_descriptor(MINISTRAL))
    adapter._client = httpx.AsyncClient(  # noqa: SLF001
        base_url=ENDPOINT, transport=httpx.MockTransport(handler), timeout=30.0
    )

    await adapter.generate(
        [
            ChatMessage(
                role="user",
                content=[
                    {"type": "text", "text": "Return JSON."},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
                ],
            )
        ],
        GenerationParams(json_mode=True),
    )

    assert len(seen) == 2
    assert [body["options"]["num_ctx"] for body in seen] == [N_CTX, N_CTX]


# --- routes, through the real app ----------------------------------------------


@pytest.fixture
def ollama(monkeypatch) -> _Ollama:
    """Serve real Ollama adapters over a scripted transport; gemma4 is resident."""
    monkeypatch.setattr(settings, "n_ctx", N_CTX)
    upstream = _Ollama()
    descriptors = {m: _descriptor(m) for m in (MINISTRAL, GEMMA, QWEN)}
    adapters: dict[str, OllamaHttpAdapter] = {}

    async def _get(model_id: str):
        descriptor = descriptors.get(model_id)
        if descriptor is None:
            raise ModelNotFoundError(model_id)
        if model_id not in adapters:
            adapters[model_id] = await _loaded(model_id, upstream)
        return adapters[model_id], descriptor

    async def _resident(endpoint: str) -> frozenset[str]:
        return frozenset({GEMMA})

    substituter = ms.ModelSubstituter(
        ms.parse_groups(f"{GEMMA},{QWEN}"), ms.ResidencyCache(_resident)
    )
    monkeypatch.setattr(app_state.manager, "get", _get)
    monkeypatch.setattr(app_state.manager, "resolve", descriptors.get)
    monkeypatch.setattr(app_state, "model_substituter", substituter)
    monkeypatch.setattr(
        app_state, "eval_runner", EvalRunner(app_state.manager, substituter=substituter)
    )
    app_state.mark_ready()
    yield upstream
    app_state.mark_ready()


def _chat(model: str = MINISTRAL, **extra) -> dict:
    return {"model": model, "messages": [{"role": "user", "content": "hi"}], **extra}


def test_blocking_chat_sends_num_ctx(ollama: _Ollama) -> None:
    response = TestClient(app).post("/v1/chat/completions", json=_chat())

    assert response.status_code == 200, response.text
    assert response.json()["usage"]["prompt_tokens"] == 20000
    ollama.assert_every_chat_carries(N_CTX)


def test_streaming_chat_sends_num_ctx(ollama: _Ollama) -> None:
    with TestClient(app).stream(
        "POST", "/v1/chat/completions", json=_chat(stream=True)
    ) as response:
        assert response.status_code == 200
        body = "".join(response.iter_text())

    assert "[DONE]" in body
    ollama.assert_every_chat_carries(N_CTX)


def test_structured_chat_sends_num_ctx(ollama: _Ollama) -> None:
    schema = {
        "type": "object",
        "properties": {"score": {"type": "integer"}, "justification": {"type": "string"}},
        "required": ["score", "justification"],
    }
    response = TestClient(app).post(
        "/v1/chat/completions",
        json=_chat(
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "verdict", "schema": schema, "strict": True},
            }
        ),
    )

    assert response.status_code == 200, response.text
    assert json.loads(response.json()["choices"][0]["message"]["content"])["score"] == 4
    ollama.assert_every_chat_carries(N_CTX)


def test_substituted_chat_sends_num_ctx(ollama: _Ollama) -> None:
    """The resident member serves; it must not be reloaded at another size."""
    response = TestClient(app).post("/v1/chat/completions", json=_chat(model=QWEN))

    assert response.status_code == 200, response.text
    assert response.json()["substituted_from_model"] == QWEN
    assert {body["model"] for _, body in ollama.requests} == {GEMMA}
    ollama.assert_every_chat_carries(N_CTX)


def test_completions_route_sends_num_ctx(ollama: _Ollama) -> None:
    response = TestClient(app).post("/v1/completions", json={"model": MINISTRAL, "prompt": "hi"})

    assert response.status_code == 200, response.text
    ollama.assert_every_chat_carries(N_CTX)


def test_eval_judge_sends_num_ctx(ollama: _Ollama) -> None:
    """The eval runner calls the adapter directly, past the scheduler."""
    response = TestClient(app).post(
        "/v1/evals/run",
        json={"rubric": "helpfulness", "prompt": "p", "response": "r", "judge_model": MINISTRAL},
    )

    assert response.status_code == 200, response.text
    ollama.assert_every_chat_carries(N_CTX)


# --- deployment files ------------------------------------------------------------


def test_launchd_engine_template_does_not_shadow_dotenv() -> None:
    """A launchd ``EnvironmentVariables`` entry outranks ``.env``.

    The engine agent runs with ``WorkingDirectory`` at the checkout, so its
    ``.env`` is read, but pydantic-settings lets the process environment win.
    The template pinned ``N_CTX=8192`` there, which turned the deployment's
    ``N_CTX=32768`` into a no-op that no log line flagged as overridden.
    """
    template = REPO / "scripts" / "launchd" / "com.planeon.inference-engine.plist"
    # launchd accepts ``--`` inside comments; expat does not. Comments are not
    # what is under test.
    xml = re.sub(r"<!--.*?-->", "", template.read_text(), flags=re.DOTALL)
    env = plistlib.loads(xml.encode())["EnvironmentVariables"]

    # MEMORY_BUDGET_GB shadowed .env the same way (96.0 over 60).
    assert {"N_CTX", "MEMORY_BUDGET_GB"}.isdisjoint(env)


def test_env_example_n_ctx_matches_the_code_default() -> None:
    example = (REPO / ".env.example").read_text()
    match = re.search(r"^N_CTX=(\d+)\s*$", example, re.MULTILINE)

    assert match is not None
    assert int(match.group(1)) == Settings.model_fields["n_ctx"].default == N_CTX
