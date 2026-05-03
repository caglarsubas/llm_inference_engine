"""vLLM adapter — HTTP client to a vLLM OpenAI-compatible server.

vLLM runs its own continuous-batching inference loop on a CUDA host (it
doesn't run on Metal or CPU at production speed). This adapter is a thin
HTTP client that forwards our adapter contract to vLLM's
``/v1/chat/completions`` and ``/v1/completions`` endpoints, gets continuous
batching for free, and reuses every observability primitive we've built —
rounds 5 (cancellation), 14 (tool audit), 20 (streaming tool reassembly),
21 (tool timing) all keep working because they sit in the chat route layer
above the adapter.

Loading model state isn't an adapter concern here: vLLM owns its own model
lifecycle on the GPU and serves whatever was passed as ``--model`` at
startup. ``load()`` only sets up the HTTP client; ``unload()`` closes it.
The ModelManager's per-model memory budget is meaningless for vLLM-served
models (the GPU memory is the upstream's concern), but we still let the
manager track them so ``/v1/models`` and the auto-eval path see them
uniformly.

Limits documented honestly:

* **No embeddings.** vLLM has limited embedding support and we don't proxy
  it; ``embed()`` raises ``EmbeddingsNotSupportedError`` so the route maps
  to HTTP 501 (same shape as the MLX path).
* **No prefix-cache introspection.** vLLM does its own KV-cache management
  (PagedAttention) but the OpenAI-compatible HTTP surface doesn't expose
  per-call hit counts the way our llama.cpp + MLX integrations do.
  ``prefix_cache_*`` properties report ``enabled=False``.
* **Cancellation in streaming closes the HTTP connection;** vLLM detects
  the dropped connection and reaps the request from its batch — the same
  semantic our llama.cpp streaming path provides.
* **Cancellation in blocking generate is best-effort** (close client,
  request stays in flight upstream until it completes). Same caveat as
  every other adapter's blocking path.
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

log = get_logger("adapter.vllm")


class VLLMAdapter(InferenceAdapter):
    backend_name = "vllm"

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
        if descriptor.format != "vllm":
            raise ValueError(f"VLLMAdapter only handles vllm, got {descriptor.format!r}")
        if not descriptor.endpoint:
            raise ValueError(f"vLLM descriptor {descriptor.qualified_name} missing endpoint")
        model_id = descriptor.params.get("model_id") if descriptor.params else None
        if not model_id:
            raise ValueError(
                f"vLLM descriptor {descriptor.qualified_name} missing params['model_id']"
            )

        # Idempotent — re-loading the same descriptor is a no-op.
        if self._descriptor and self._descriptor.endpoint == descriptor.endpoint and self._model_id == model_id:
            return

        await self.unload()
        self._descriptor = descriptor
        self._endpoint = descriptor.endpoint
        self._model_id = str(model_id)
        # Generous timeout — first request after cold start can wait on
        # vLLM's CUDA graph compilation (~10-30s). Streaming reads use a
        # separate timeout via ``client.stream`` calls anyway.
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
    # Request / response translation
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
        # vLLM accepts top_k via ``extra_body`` in its OpenAI shim. Putting
        # it as a top-level field in the body is also accepted by current
        # versions; we send it bare and let vLLM route it.
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
        cancel: Cancellation | None = None,  # noqa: ARG002 — vLLM HTTP has no mid-call hook for blocking
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

        # vLLM emits standard OpenAI SSE: ``data: {json}`` per chunk, terminated
        # by ``data: [DONE]``. Cancellation closes the underlying connection
        # so vLLM reaps the request from its in-flight batch.
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
                    log.warning("vllm.stream.bad_json", payload=payload[:200])
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

    # ------------------------------------------------------------------
    # complete (legacy /v1/completions pass-through)
    # ------------------------------------------------------------------

    async def complete(
        self,
        prompt: str,
        params: GenerationParams,
        cancel: Cancellation | None = None,  # noqa: ARG002
    ) -> GenerationResult:
        if not self.is_loaded:
            raise RuntimeError("model not loaded")

        body = {
            "model": self._model_id,
            "prompt": prompt,
            "temperature": params.temperature,
            "top_p": params.top_p,
            "max_tokens": params.max_tokens,
        }
        if params.top_k > 0:
            body["top_k"] = params.top_k
        if params.stop:
            body["stop"] = params.stop
        if params.seed is not None:
            body["seed"] = params.seed

        assert self._client is not None
        r = await self._client.post("/v1/completions", json=body)
        r.raise_for_status()
        data = r.json()
        choice = (data.get("choices") or [{}])[0]
        usage = data.get("usage") or {}
        return GenerationResult(
            text=choice.get("text") or "",
            finish_reason=choice.get("finish_reason") or "stop",
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            tool_calls=None,
        )

    # ------------------------------------------------------------------
    # embed — not supported in this round
    # ------------------------------------------------------------------

    async def embed(self, inputs: list[str]) -> EmbeddingResult:
        raise EmbeddingsNotSupportedError(self.backend_name)

    # ------------------------------------------------------------------
    # vLLM doesn't expose per-call prefix-cache hit counts via the HTTP API;
    # we keep the introspection surface present but report disabled so the
    # chat span attrs stay uniform across backends.
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


# Helpful for callers that want to introspect this module's HTTP behaviour
# without importing httpx directly.
__all__ = ["VLLMAdapter"]
