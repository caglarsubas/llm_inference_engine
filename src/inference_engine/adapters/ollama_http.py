"""Ollama HTTP adapter — proxy chat completions to an Ollama server.

Mirrors :class:`VLLMAdapter` (HTTP client to an OpenAI-compatible upstream)
but targets the Ollama-bundled endpoint.  Ollama exposes both a native
``/api/chat`` API and an OpenAI-compatible ``/v1/chat/completions`` shim;
we use the latter so the request/response shape matches what the rest of
this codebase already speaks (vLLM, OpenAI cloud, llama-cpp-python's
``create_chat_completion``).

Why this adapter exists at all
------------------------------

The local llama-cpp-python build can't open every GGUF Ollama can serve —
new architectures (gemma4, qwen3.6, ministral-3, nemotron3 in 2026) land
in Ollama's ggml fork weeks before they reach the python wheel.  Routing
those models through this adapter keeps them reachable end-to-end without
forcing operators to wait for a llama-cpp-python release or to maintain a
custom-built wheel.

Ownership of model lifecycle
----------------------------

Ollama owns model load / unload on its side; ``load()`` here only sets up
an HTTP client and ``unload()`` closes it.  Our ``ModelManager`` budget is
informational for these descriptors (the upstream's resident bytes are
the upstream's concern) — same shape as the vLLM adapter's caveat.

Limits documented honestly:

* **No embeddings** in this round.  Ollama supports them via ``/api/embed``
  but the surface differs from OpenAI's; ``embed()`` raises
  ``EmbeddingsNotSupportedError`` so the route maps to HTTP 501.
* **No prefix-cache introspection.**  Ollama runs its own KV cache on the
  upstream side and doesn't surface per-call hit counts; the
  ``prefix_cache_*`` properties report ``enabled=False``.
* **Streaming cancel** closes the HTTP connection — Ollama detects the
  drop and aborts decode the same way vLLM does.
* **Blocking generate cancel** is best-effort: closing the client doesn't
  abort an in-flight upstream request.  Same caveat as every other
  adapter's blocking path; agents that need fast cancel use ``stream=true``.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterable

import httpx

from ..cancellation import Cancellation
from ..observability import get_logger
from ..registry import ModelDescriptor
from ..schemas import ChatMessage
from .base import (
    EmbeddingResult,
    EmbeddingsNotSupportedError,
    GenerationParams,
    GenerationResult,
    InferenceAdapter,
    StreamChunk,
)

log = get_logger("adapter.ollama_http")


class OllamaHttpAdapter(InferenceAdapter):
    backend_name = "ollama_http"

    def __init__(self) -> None:
        self._descriptor: ModelDescriptor | None = None
        self._endpoint: str | None = None
        self._model_id: str | None = None
        self._client: httpx.AsyncClient | None = None

    @property
    def is_loaded(self) -> bool:
        return self._client is not None

    @property
    def loaded_model(self) -> ModelDescriptor | None:
        return self._descriptor

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def load(self, descriptor: ModelDescriptor) -> None:
        if descriptor.format != "ollama_http":
            raise ValueError(
                f"OllamaHttpAdapter only handles ollama_http, got {descriptor.format!r}"
            )
        if not descriptor.endpoint:
            raise ValueError(
                f"ollama_http descriptor {descriptor.qualified_name} missing endpoint"
            )
        model_id = (descriptor.params or {}).get("model_id") or descriptor.qualified_name

        # Idempotent reload.
        if (
            self._descriptor is not None
            and self._descriptor.endpoint == descriptor.endpoint
            and self._model_id == model_id
        ):
            return

        await self.unload()
        self._descriptor = descriptor
        self._endpoint = descriptor.endpoint
        self._model_id = str(model_id)
        # Generous timeout — first request after a cold ollama can wait on
        # mmap-warmup of a 30 GB model.  Streaming reads piggyback on this
        # client; per-chunk deadlines come from sse_starlette downstream.
        self._client = httpx.AsyncClient(base_url=self._endpoint, timeout=600.0)
        log.info(
            "loaded",
            model=descriptor.qualified_name,
            endpoint=self._endpoint,
            model_id=self._model_id,
        )

    async def unload(self) -> None:
        if self._client is not None:
            log.info(
                "unloaded",
                model=self._descriptor.qualified_name if self._descriptor else None,
            )
            await self._client.aclose()
            self._client = None
            self._descriptor = None
            self._endpoint = None
            self._model_id = None

    # ------------------------------------------------------------------
    # Request / response translation (OpenAI-compat shape)
    # ------------------------------------------------------------------

    @staticmethod
    def _to_messages(messages: Iterable[ChatMessage]) -> list[dict]:
        out: list[dict] = []
        for m in messages:
            entry: dict = {"role": m.role, "content": m.content}
            if m.tool_calls is not None:
                entry["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in m.tool_calls
                ]
            if m.tool_call_id is not None:
                entry["tool_call_id"] = m.tool_call_id
            if m.name is not None:
                entry["name"] = m.name
            out.append(entry)
        return out

    def _completion_kwargs(self, params: GenerationParams) -> dict:
        kw: dict = {
            "model": self._model_id,
            "temperature": params.temperature,
            "top_p": params.top_p,
            "max_tokens": params.max_tokens,
        }
        if params.top_k > 0:
            kw["top_k"] = params.top_k
        if params.stop:
            kw["stop"] = params.stop
        if params.seed is not None:
            kw["seed"] = params.seed
        if params.json_mode:
            kw["response_format"] = {"type": "json_object"}
        if params.tools:
            kw["tools"] = params.tools
        if params.tool_choice is not None:
            kw["tool_choice"] = params.tool_choice
        return kw

    # ------------------------------------------------------------------
    # generate / stream
    # ------------------------------------------------------------------

    async def generate(
        self,
        messages: Iterable[ChatMessage],
        params: GenerationParams,
        cancel: Cancellation | None = None,  # noqa: ARG002 — no mid-call hook
    ) -> GenerationResult:
        if not self.is_loaded:
            raise RuntimeError("model not loaded")

        body = {
            **self._completion_kwargs(params),
            "messages": self._to_messages(messages),
            "stream": False,
        }
        assert self._client is not None
        r = await self._client.post("/v1/chat/completions", json=body)
        r.raise_for_status()
        data = r.json()

        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        usage = data.get("usage") or {}
        tool_calls = message.get("tool_calls")

        return GenerationResult(
            text=message.get("content") or "",
            finish_reason=choice.get("finish_reason") or "stop",
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            tool_calls=list(tool_calls) if tool_calls else None,
        )

    async def stream(
        self,
        messages: Iterable[ChatMessage],
        params: GenerationParams,
        cancel: Cancellation | None = None,
    ) -> AsyncIterator[StreamChunk]:
        if not self.is_loaded:
            raise RuntimeError("model not loaded")

        body = {
            **self._completion_kwargs(params),
            "messages": self._to_messages(messages),
            "stream": True,
        }
        assert self._client is not None
        async with self._client.stream("POST", "/v1/chat/completions", json=body) as resp:
            resp.raise_for_status()
            async for raw_line in resp.aiter_lines():
                if cancel is not None and bool(cancel):
                    return
                if not raw_line or not raw_line.startswith("data:"):
                    continue
                payload = raw_line[5:].strip()
                if payload == "[DONE]":
                    return
                try:
                    event = json.loads(payload)
                except json.JSONDecodeError:
                    log.warning("ollama_http.stream.bad_json", payload=payload[:200])
                    continue
                choices = event.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                delta = choice.get("delta") or {}
                text = delta.get("content") or ""
                tool_call_deltas = delta.get("tool_calls")
                finish = choice.get("finish_reason")
                yield StreamChunk(
                    text=text,
                    finish_reason=finish,
                    tool_call_deltas=list(tool_call_deltas) if tool_call_deltas else None,
                )

    async def complete(
        self,
        prompt: str,
        params: GenerationParams,
        cancel: Cancellation | None = None,  # noqa: ARG002
    ) -> GenerationResult:
        # Ollama's OpenAI shim does expose /v1/completions but new agentic
        # workflows don't need raw completion; map it onto chat with a single
        # user turn so we don't carry a separate code path.
        msg = ChatMessage(role="user", content=prompt)
        return await self.generate([msg], params, cancel=cancel)

    async def embed(self, inputs: list[str]) -> EmbeddingResult:  # noqa: ARG002
        raise EmbeddingsNotSupportedError(self.backend_name)

    # ------------------------------------------------------------------
    # No prefix-cache introspection (Ollama doesn't surface it via HTTP).
    # Match the vLLM adapter's "report disabled" stance so the chat span
    # attrs are uniform across backends.
    # ------------------------------------------------------------------

    @property
    def prefix_cache_enabled(self) -> bool:
        return False

    @property
    def prefix_cache_last_action(self) -> str:
        return "disabled"

    @property
    def prefix_cache_last_overlap_tokens(self) -> int:
        return 0

    @property
    def prefix_cache_last_prompt_tokens(self) -> int:
        return 0


__all__ = ["OllamaHttpAdapter"]
