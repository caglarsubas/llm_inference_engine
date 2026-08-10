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

import asyncio
import json
from collections.abc import AsyncIterator, Iterable
from contextlib import AsyncExitStack, aclosing

import httpx

from ..cancellation import Cancellation
from ..config import settings
from ..observability import get_logger
from ..registry import ModelDescriptor
from ..schemas import ChatMessage, dump_chat_content
from ._upstream import HttpUpstreamMixin
from ._upstream_retry import (
    RetryPlan,
    UpstreamDeadline,
    deadline_scope,
    log_retry,
    log_retry_refused,
    plan_retry,
)
from .base import (
    EmbeddingResult,
    EmbeddingsNotSupportedError,
    GenerationParams,
    GenerationResult,
    GenerationTimeoutError,
    InferenceAdapter,
    StreamChunk,
    UpstreamGenerationError,
)

log = get_logger("adapter.ollama_http")

_JSON_RETRY_MIN_TOKENS = 256
_JSON_RETRY_SYSTEM_PROMPT = (
    "Return only one compact valid JSON object. Follow the user requested "
    "schema and keys exactly. Do not include markdown, prose, or code fences."
)


def _chat_timeout() -> httpx.Timeout:
    seconds = settings.chat_completion_timeout_seconds
    if seconds <= 0:
        return httpx.Timeout(None)
    return httpx.Timeout(seconds)


def _deadline_seconds() -> float | None:
    seconds = settings.chat_completion_timeout_seconds
    return seconds if seconds > 0 else None


def _has_image_content(messages: list[dict]) -> bool:
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image_url":
                return True
    return False


def _prepend_json_retry_prompt(messages: list[dict]) -> list[dict]:
    return [{"role": "system", "content": _JSON_RETRY_SYSTEM_PROMPT}, *messages]


class OllamaHttpAdapter(HttpUpstreamMixin, InferenceAdapter):
    backend_name = "ollama_http"
    # DECLARED, then checked. Recent Ollama does implement structured outputs
    # in its own sampler, so True is the right PRIOR — but the OpenAI shim
    # silently ignores `response_format` on releases that predate it, and this
    # adapter cannot tell which release an endpoint is running.
    #
    # Measured on one such deployment (gemma4:26b): a strict json_schema
    # request for an object with a single integer key returned HTTP 200 and
    # "Mars has **two** moons: Phobos and Deimos." — prose, no error, no
    # repair. Under the old class-constant scheme that claim also switched OFF
    # the gateway's validate-and-repair net, so the caller received it.
    #
    # `structured_output_capability` now demotes a deployment the first time it
    # emits non-JSON, and the chat route validates trusted backends too rather
    # than skipping them, so the demotion happens on that same request instead
    # of a later one.
    supports_structured_outputs = True
    # httpx-backed: cancelling the await closes the upstream request, so a
    # deadline here really does end the work rather than just stop waiting.
    generation_is_cancellable = True

    def deployment_id(self) -> str:
        # The endpoint is load-bearing here: two Ollama hosts behind the same
        # adapter can be different releases with different capabilities, and a
        # demotion earned by one must not silence the other.
        return f"{self.backend_name}:{self._endpoint or 'unbound'}"

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
        self._bind_deployment(self._endpoint, self._model_id)
        self._client = httpx.AsyncClient(base_url=self._endpoint, timeout=_chat_timeout())
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
            self._clear_deployment()

    # ------------------------------------------------------------------
    # Request / response translation (OpenAI-compat shape)
    # ------------------------------------------------------------------

    @staticmethod
    def _to_messages(messages: Iterable[ChatMessage]) -> list[dict]:
        out: list[dict] = []
        for m in messages:
            entry: dict = {"role": m.role, "content": dump_chat_content(m.content)}
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
            # Ollama's OpenAI shim understands the json_schema form on recent
            # releases and ignores the extra key on older ones, degrading to
            # plain JSON mode rather than erroring.
            if params.json_schema:
                kw["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": params.json_schema_name,
                        "schema": params.json_schema,
                        "strict": params.json_schema_strict,
                    },
                }
            else:
                kw["response_format"] = {"type": "json_object"}
        if params.tools:
            kw["tools"] = params.tools
        if params.tool_choice is not None:
            kw["tool_choice"] = params.tool_choice
        if params.frequency_penalty is not None:
            kw["frequency_penalty"] = params.frequency_penalty
        if params.presence_penalty is not None:
            kw["presence_penalty"] = params.presence_penalty
        if params.repetition_penalty is not None:
            kw["repetition_penalty"] = params.repetition_penalty
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
        should_retry_empty_json = params.json_mode and _has_image_content(body["messages"])
        assert self._client is not None
        # One budget for the whole logical call: the transport retries below
        # and the empty-multimodal-JSON reprompt all draw from it, so neither
        # can push the request past CHAT_COMPLETION_TIMEOUT_SECONDS.
        deadline = UpstreamDeadline(_deadline_seconds())
        self._begin_upstream()
        reported = False
        try:
            try:
                r = await self._post_with_retries(body, deadline)
                data = r.json()

                if should_retry_empty_json and self._content_from_response(data) == "":
                    retry_body = dict(body)
                    retry_body.pop("response_format", None)
                    retry_body["max_tokens"] = max(
                        int(retry_body.get("max_tokens") or 0),
                        _JSON_RETRY_MIN_TOKENS,
                    )
                    retry_body["messages"] = _prepend_json_retry_prompt(body["messages"])
                    log.warning(
                        "ollama_http.retry_empty_multimodal_json",
                        model=self._model_id,
                        original_max_tokens=body.get("max_tokens"),
                        retry_max_tokens=retry_body["max_tokens"],
                    )
                    r = await self._post_with_retries(retry_body, deadline)
                    data = r.json()
            except Exception as exc:
                self._finish_upstream(exc)
                reported = True
                typed = self._as_typed_upstream_error(exc)
                if typed is exc:
                    raise
                raise typed from exc
            self._finish_upstream(None)
            reported = True
        finally:
            if not reported:
                self._abandon_upstream()

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

    @staticmethod
    def _content_from_response(data: dict) -> str:
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        return (message.get("content") or "").strip()

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
            # Ollama's OpenAI shim honours include_usage and emits the same
            # trailing usage-only chunk; older builds simply omit it and we
            # fall through with zero counts.
            "stream_options": {"include_usage": True},
        }
        assert self._client is not None
        deadline = UpstreamDeadline(_deadline_seconds())
        self._begin_upstream()
        attempt = 0
        reported = False
        try:
            while True:
                attempt += 1
                emitted = False
                try:
                    async with aclosing(self._stream_once(body, cancel, deadline)) as pieces:
                        async for piece in pieces:
                            emitted = True
                            yield piece
                except Exception as exc:
                    # HARD RULE: a stream that has already delivered a chunk
                    # cannot be reissued — there is no way to resume upstream,
                    # and rerunning the prompt would splice two completions.
                    plan = (
                        RetryPlan(retry=False, reason="stream_already_emitted")
                        if emitted
                        else plan_retry(
                            attempt=attempt,
                            exc=exc,
                            remaining_seconds=deadline.remaining,
                        )
                    )
                    if plan.retry:
                        log_retry(
                            backend=self.backend_name,
                            model=self._model_id or "",
                            attempt=attempt,
                            plan=plan,
                            exc=exc,
                            operation="/v1/chat/completions#stream",
                        )
                        await asyncio.sleep(plan.delay_seconds)
                        continue
                    log_retry_refused(
                        backend=self.backend_name,
                        model=self._model_id or "",
                        attempt=attempt,
                        plan=plan,
                        exc=exc,
                        operation="/v1/chat/completions#stream",
                    )
                    self._finish_upstream(exc)
                    reported = True
                    typed = self._as_typed_upstream_error(exc)
                    if typed is exc:
                        raise
                    raise typed from exc
                self._finish_upstream(None)
                reported = True
                return
        finally:
            if not reported:
                self._abandon_upstream()

    async def _stream_once(
        self,
        body: dict,
        cancel: Cancellation | None,
        deadline: UpstreamDeadline,
    ) -> AsyncIterator[StreamChunk]:
        """One SSE attempt, raising raw httpx errors for the caller to classify.

        THE BUDGET IS ENFORCED HERE, NOT BY ``httpx.Timeout``. httpx's read
        timeout is per read operation: a stream that drips a token every second
        resets it forever and never trips it, which is exactly what a slow
        upstream looks like. That is not merely a long request — the scheduler
        lease is held for the whole stream, so an unbounded one takes the
        deployment's dispatch slot with it. Every await on the upstream is
        therefore wrapped in :func:`deadline_scope`, and the loop re-checks the
        budget before each chunk it hands on, so the stream stops instead of
        outliving ``CHAT_COMPLETION_TIMEOUT_SECONDS``. The resulting
        ``TimeoutError`` reaches the caller as the route's typed
        ``generation_timeout`` — a terminal SSE ``error`` event, since the
        response line is already open — and is deliberately NOT counted
        against the deployment's health: a stream that merely ran longer than
        the caller's budget says nothing about the upstream. See
        :func:`_upstream.is_health_signal`.
        """
        assert self._client is not None
        async with AsyncExitStack() as stack:
            async with deadline_scope(deadline):
                resp = await stack.enter_async_context(
                    self._client.stream("POST", "/v1/chat/completions", json=body)
                )
                if resp.status_code >= 400:
                    # Read the body before raising: on a streamed response it is
                    # unread, and the error mapper reads it to build the 502
                    # detail.
                    await resp.aread()
            resp.raise_for_status()
            lines = resp.aiter_lines()
            while True:
                async with deadline_scope(deadline):
                    try:
                        raw_line = await anext(lines)
                    except StopAsyncIteration:
                        return
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
                    usage = event.get("usage")
                    if isinstance(usage, dict):
                        yield StreamChunk(
                            text="",
                            prompt_tokens=int(usage.get("prompt_tokens", 0)),
                            completion_tokens=int(usage.get("completion_tokens", 0)),
                        )
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
    # transport — one attempt, the retry loop over it, and error mapping
    # ------------------------------------------------------------------

    async def _post_once(self, body: dict, deadline: UpstreamDeadline) -> httpx.Response:
        """One attempt, bounded by what is LEFT of the deadline, not all of it."""
        assert self._client is not None
        async with deadline_scope(deadline):
            response = await self._client.post("/v1/chat/completions", json=body)
        response.raise_for_status()
        return response

    async def _post_with_retries(
        self,
        body: dict,
        deadline: UpstreamDeadline,
    ) -> httpx.Response:
        """Reissue this POST on the SAME deployment while that is honest.

        Sits below the route's fallback loop on purpose: a transient upstream
        failure must cost the caller a retry on the model they asked for, not
        a silent substitution of a different one.
        """
        attempt = 0
        while True:
            attempt += 1
            try:
                return await self._post_once(body, deadline)
            except Exception as exc:
                plan = plan_retry(
                    attempt=attempt,
                    exc=exc,
                    remaining_seconds=deadline.remaining,
                )
                if not plan.retry:
                    log_retry_refused(
                        backend=self.backend_name,
                        model=self._model_id or "",
                        attempt=attempt,
                        plan=plan,
                        exc=exc,
                        operation="/v1/chat/completions",
                    )
                    raise
                log_retry(
                    backend=self.backend_name,
                    model=self._model_id or "",
                    attempt=attempt,
                    plan=plan,
                    exc=exc,
                    operation="/v1/chat/completions",
                )
                await asyncio.sleep(plan.delay_seconds)

    def _as_typed_upstream_error(self, exc: Exception) -> Exception:
        if isinstance(exc, httpx.TimeoutException | TimeoutError):
            return self._timeout_error()
        if isinstance(exc, httpx.HTTPStatusError):
            return self._upstream_http_error(exc)
        if isinstance(exc, httpx.RequestError):
            return self._upstream_request_error(exc)
        return exc

    def _upstream_http_error(self, exc: httpx.HTTPStatusError) -> UpstreamGenerationError:
        try:
            payload = exc.response.json()
        except ValueError:
            detail = exc.response.text[:500]
        else:
            detail = json.dumps(payload, sort_keys=True)[:500]
        return UpstreamGenerationError(
            error_type="upstream_http_error",
            upstream_status_code=exc.response.status_code,
            backend=self.backend_name,
            model=self._model_id or "",
            detail=detail,
        )

    def _upstream_request_error(self, exc: httpx.RequestError) -> UpstreamGenerationError:
        return UpstreamGenerationError(
            error_type="upstream_request_error",
            backend=self.backend_name,
            model=self._model_id or "",
            detail=str(exc).splitlines()[0][:500] if str(exc) else exc.__class__.__name__,
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

    def _timeout_error(self) -> GenerationTimeoutError:
        return GenerationTimeoutError(
            timeout_seconds=settings.chat_completion_timeout_seconds,
            backend=self.backend_name,
            model=self._model_id or "",
        )


__all__ = ["OllamaHttpAdapter"]
