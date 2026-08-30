"""Ollama HTTP adapter — proxy chat completions to an Ollama server.

Mirrors :class:`VLLMAdapter` (HTTP client to a remote upstream) but targets
the Ollama-bundled endpoint.  Ollama exposes both a native ``/api/chat`` API
and an OpenAI-compatible ``/v1/chat/completions`` shim.  Chat goes through
the NATIVE endpoint; embeddings still go through the shim.

Why chat is not on the OpenAI shim
----------------------------------

It was, and ``settings.n_ctx`` was silently inert as a result (issue #111).
The shim accepts only the OpenAI request fields and builds the runtime
options itself, so there is no way to tell it what KV-cache size to load a
model at.  Measured against Ollama 0.32.14, all three spellings —
``options.num_ctx``, a top-level ``num_ctx``, a top-level ``context_length``
— returned HTTP 200 and loaded the model at the host default anyway; a
runner pre-loaded at 8192 through ``/api/chat`` was *reloaded* at the host
default by the very next shim request.  Only ``/api/chat`` carries
``options``, so only ``/api/chat`` can honour the setting.

That matters well beyond tidiness.  Ollama's default window scales with
available memory and reaches 262,144 on a large-memory host, and KV cache
scales linearly with it: ``ministral-3:8b`` measured 41 GB resident — 6 GB
of weights and 34 GiB of cache — where right-sizing to 32,768 brought the
same model to 9.8 GB.  Architectures without sliding-window attention pay
that on every layer, so an 8.9B model can outweigh a 25.8B one, and the
symptom reaching clients is scheduler starvation (``tenant_queue_timeout``),
not anything that looks like a context setting.

The cost of the native endpoint is that this adapter owns the translation
both ways (OpenAI shapes in, OpenAI shapes out) instead of getting it for
free from the shim.  ``_to_messages`` / ``_native_body`` and
``_result_from_payload`` / ``_chunk_from_event`` are that seam; everything
above the adapter still speaks OpenAI.

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

* **Embeddings are served** through the shim's ``/v1/embeddings``, which
  returns the OpenAI shape verbatim.  This matters because embedding-only
  models such as ``embeddinggemma:300m`` are among the GGUFs llama.cpp
  cannot open, so this backend is the only one that can serve them.
* **No prefix-cache introspection.**  Ollama runs its own KV cache on the
  upstream side and doesn't surface per-call hit counts; the
  ``prefix_cache_*`` properties report ``enabled=False``.
* **Streaming cancel** closes the HTTP connection — Ollama detects the
  drop and aborts decode the same way vLLM does.
* **Blocking generate cancel** is best-effort: closing the client doesn't
  abort an in-flight upstream request.  Same caveat as every other
  adapter's blocking path; agents that need fast cancel use ``stream=true``.
* **Context overflow is only sometimes an error.**  Ollama truncates an
  over-long prompt by default and says so nowhere in the response — the
  ``prompt_eval_count`` it reports is the count *after* truncation, so the
  condition is not detectable post-hoc, and there is no tokenizer endpoint
  to pre-count against.  A deployment run with context shift disabled does
  raise, and :meth:`_as_context_error` maps that to the same typed
  ``ContextLengthExceededError`` (HTTP 400 ``context_length_exceeded``) the
  llama.cpp path raises, rather than an opaque 502.  Operators who want the
  contract everywhere have to turn context shift off upstream.
* **Images must be inline.**  Native ``/api/chat`` takes raw base64 in
  ``images``; a ``data:`` URL is unwrapped here.  A remote ``http(s)``
  image URL is passed through and rejected upstream — the shim did not
  fetch those either.
"""

from __future__ import annotations

import asyncio
import json
import re
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
    ContextLengthExceededError,
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

# Chat is native (it is the only endpoint that accepts ``options``);
# embeddings stay on the OpenAI shim, which returns the OpenAI shape verbatim
# and has no context-window decision to make.
_CHAT_PATH = "/api/chat"
_EMBEDDINGS_PATH = "/v1/embeddings"

# Ollama's own overflow wording, taken from the server binary rather than
# guessed. Only the first carries both integers; the rest are matched as
# substrings so a phrasing change across releases degrades to "we still know
# it was a context overflow" instead of falling back to an opaque 502.
_CTX_OVERFLOW_RE = re.compile(
    r"input length\s*\((\d+)\s*tokens?\).*?context length\s*\((\d+)\s*tokens?\)",
    re.IGNORECASE | re.DOTALL,
)
_CTX_OVERFLOW_PHRASES = (
    "exceeds the context length",
    "exceeds maximum context length",
    "exceeds the available context",
    "cannot be truncated further",
)

# ``data:image/png;base64,<payload>`` — native ``/api/chat`` wants only the
# payload.
_DATA_URL_RE = re.compile(r"^data:[^;,]*;base64,", re.IGNORECASE)

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
    """True when any already-translated native message carries an image."""
    return any(message.get("images") for message in messages)


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
        self._last_embed_action: str = "none"
        # Resolved at load(); every request carries it as ``options.num_ctx``.
        self._num_ctx: int = settings.n_ctx

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
        self._num_ctx = self._resolve_num_ctx(descriptor)
        self._bind_deployment(self._endpoint, self._model_id)
        self._client = httpx.AsyncClient(base_url=self._endpoint, timeout=_chat_timeout())
        log.info(
            "loaded",
            model=descriptor.qualified_name,
            endpoint=self._endpoint,
            model_id=self._model_id,
            n_ctx=self._num_ctx,
            n_ctx_ceiling=settings.n_ctx,
        )

    def _resolve_num_ctx(self, descriptor: ModelDescriptor) -> int:
        """This model's effective context window, clamped to what it was trained for.

        The trained window comes from the registry, which already reads it off
        ``POST /api/show`` and caches it per blob digest — no extra upstream
        call here. An absent or unparsable value means "unknown" and leaves the
        configured ceiling alone, exactly as a missed GGUF probe does on the
        llama.cpp path.
        """
        declared = (descriptor.params or {}).get("context_length")
        try:
            n_ctx_train = int(declared) if declared is not None else 0
        except (TypeError, ValueError):
            n_ctx_train = 0
        return self._effective_n_ctx(settings.n_ctx, n_ctx_train)

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
            self._num_ctx = settings.n_ctx
            self._clear_deployment()

    # ------------------------------------------------------------------
    # Request / response translation (OpenAI-compat shape)
    # ------------------------------------------------------------------

    @staticmethod
    def _image_payload(url: str) -> str:
        """Native ``images`` entries are bare base64, not data URLs."""
        return _DATA_URL_RE.sub("", url, count=1)

    @classmethod
    def _split_content(cls, content) -> tuple[str, list[str]]:
        """OpenAI content parts -> ``(text, images)`` for one native message.

        Native ``/api/chat`` keeps text and images in separate fields, so the
        text parts are joined and the image parts collected. A plain string
        content passes through with no images.
        """
        if not isinstance(content, list):
            return ("" if content is None else str(content), [])
        texts: list[str] = []
        images: list[str] = []
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "text":
                texts.append(str(part.get("text") or ""))
            elif part.get("type") == "image_url":
                url = (part.get("image_url") or {}).get("url")
                if url:
                    images.append(cls._image_payload(str(url)))
        return ("\n".join(t for t in texts if t), images)

    @staticmethod
    def _tool_call_arguments(arguments: str | None):
        """Native tool calls carry an arguments OBJECT; OpenAI carries a string.

        A non-JSON string (a model that emitted junk, or a caller replaying a
        transcript) is wrapped rather than dropped: sending it through as-is
        would lose the turn, and Ollama tolerates a scalar-valued map.
        """
        if not arguments:
            return {}
        try:
            decoded = json.loads(arguments)
        except (TypeError, ValueError):
            return {"arguments": arguments}
        return decoded if isinstance(decoded, dict) else {"arguments": decoded}

    @classmethod
    def _to_messages(cls, messages: Iterable[ChatMessage]) -> list[dict]:
        out: list[dict] = []
        for m in messages:
            text, images = cls._split_content(dump_chat_content(m.content))
            entry: dict = {"role": m.role, "content": text}
            if images:
                entry["images"] = images
            if m.tool_calls is not None:
                entry["tool_calls"] = [
                    {
                        "id": tc.id,
                        "function": {
                            "name": tc.function.name,
                            "arguments": cls._tool_call_arguments(tc.function.arguments),
                        },
                    }
                    for tc in m.tool_calls
                ]
            # Native messages carry both of these, so a tool result still
            # points back at the call it answers.
            if m.tool_call_id is not None:
                entry["tool_call_id"] = m.tool_call_id
            if m.name is not None:
                entry["tool_name"] = m.name
            out.append(entry)
        return out

    def _options(self, params: GenerationParams) -> dict:
        """Native runtime options — including the one this endpoint exists for.

        ``num_ctx`` is the whole reason chat is not on the OpenAI shim: without
        it Ollama picks a window from available VRAM (up to 262,144) and the
        deployment's ``N_CTX`` means nothing. See the module docstring.
        """
        opts: dict = {
            "num_ctx": self._num_ctx,
            "temperature": params.temperature,
            "top_p": params.top_p,
            "num_predict": params.max_tokens,
        }
        if params.top_k > 0:
            opts["top_k"] = params.top_k
        if params.stop:
            opts["stop"] = list(params.stop)
        if params.seed is not None:
            opts["seed"] = params.seed
        if params.frequency_penalty is not None:
            opts["frequency_penalty"] = params.frequency_penalty
        if params.presence_penalty is not None:
            opts["presence_penalty"] = params.presence_penalty
        if params.repetition_penalty is not None:
            opts["repeat_penalty"] = params.repetition_penalty
        return opts

    def _native_body(
        self,
        messages: list[dict],
        params: GenerationParams,
        *,
        stream: bool,
    ) -> dict:
        body: dict = {
            "model": self._model_id,
            "messages": messages,
            "stream": stream,
            "options": self._options(params),
        }
        if params.json_mode:
            # Native structured outputs: ``format`` takes the bare schema, and
            # the string "json" is the schema-less JSON mode. This is the same
            # sampler the shim's ``response_format`` wraps, reached directly.
            body["format"] = params.json_schema if params.json_schema else "json"
        if params.tools:
            body["tools"] = params.tools
        # ``tool_choice`` / ``parallel_tool_calls`` have no native equivalent.
        # The shim dropped them too (Go ignores unknown request fields), so
        # this is the same behaviour, now visible rather than accidental.
        return body

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

        native_messages = self._to_messages(messages)
        body = self._native_body(native_messages, params, stream=False)
        should_retry_empty_json = params.json_mode and _has_image_content(native_messages)
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
                    retry_body.pop("format", None)
                    retry_options = dict(body["options"])
                    retry_options["num_predict"] = max(
                        int(retry_options.get("num_predict") or 0),
                        _JSON_RETRY_MIN_TOKENS,
                    )
                    retry_body["options"] = retry_options
                    retry_body["messages"] = _prepend_json_retry_prompt(native_messages)
                    log.warning(
                        "ollama_http.retry_empty_multimodal_json",
                        model=self._model_id,
                        original_max_tokens=body["options"].get("num_predict"),
                        retry_max_tokens=retry_options["num_predict"],
                    )
                    r = await self._post_with_retries(retry_body, deadline)
                    data = r.json()
            except Exception as exc:
                overflow = self._as_context_error(exc)
                if overflow is not None:
                    # A prompt that doesn't fit is a statement about the
                    # REQUEST, not the health of the deployment — same stance
                    # as the 501 on embeddings. Counting it would let one
                    # oversized caller open the breaker for everyone.
                    self._abandon_upstream()
                    reported = True
                    raise overflow from exc
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

        return self._result_from_payload(data)

    @classmethod
    def _result_from_payload(cls, data: dict) -> GenerationResult:
        """Native ``/api/chat`` response -> the OpenAI-shaped internal result."""
        message = data.get("message") or {}
        tool_calls = cls._tool_calls_from_native(message.get("tool_calls"))
        return GenerationResult(
            text=message.get("content") or "",
            # Native calls it ``done_reason``; "load" appears on a bare preload
            # response, which is not a completion this adapter ever asks for.
            finish_reason=data.get("done_reason") or "stop",
            prompt_tokens=int(data.get("prompt_eval_count") or 0),
            completion_tokens=int(data.get("eval_count") or 0),
            tool_calls=tool_calls,
            reasoning_content=message.get("thinking") or None,
        )

    @staticmethod
    def _tool_calls_from_native(raw, *, offset: int = 0) -> list[dict] | None:
        """Native tool calls -> the OpenAI shape the chat route reassembles.

        Two differences to close: ``arguments`` is an object upstream and a
        JSON string in the OpenAI contract, and ``type`` is implicit. Recent
        Ollama does emit an ``id``; older builds don't, and a synthesized one
        keeps a subsequent ``tool_call_id`` reply matchable.

        ``offset`` is how many calls this stream has already yielded. The chat
        route reassembles streamed calls BY INDEX, so two calls arriving in
        separate frames of one stream must not both fall back to index 0 — that
        would concatenate two different argument payloads into one call.
        """
        if not raw:
            return None
        out: list[dict] = []
        for position, call in enumerate(raw):
            if not isinstance(call, dict):
                continue
            fn = call.get("function") or {}
            index = int(fn.get("index", offset + position))
            arguments = fn.get("arguments")
            if not isinstance(arguments, str):
                arguments = json.dumps(arguments if arguments is not None else {})
            out.append(
                {
                    "id": call.get("id") or f"call_{index}",
                    "type": "function",
                    "index": index,
                    "function": {"name": fn.get("name") or "", "arguments": arguments},
                }
            )
        return out or None

    @staticmethod
    def _content_from_response(data: dict) -> str:
        return ((data.get("message") or {}).get("content") or "").strip()

    async def stream(
        self,
        messages: Iterable[ChatMessage],
        params: GenerationParams,
        cancel: Cancellation | None = None,
    ) -> AsyncIterator[StreamChunk]:
        if not self.is_loaded:
            raise RuntimeError("model not loaded")

        # Native streaming needs no usage opt-in: the final object always
        # carries ``prompt_eval_count`` / ``eval_count``, which is what the
        # shim's ``stream_options.include_usage`` was asking for.
        body = self._native_body(self._to_messages(messages), params, stream=True)
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
                            operation=f"{_CHAT_PATH}#stream",
                        )
                        await asyncio.sleep(plan.delay_seconds)
                        continue
                    log_retry_refused(
                        backend=self.backend_name,
                        model=self._model_id or "",
                        attempt=attempt,
                        plan=plan,
                        exc=exc,
                        operation=f"{_CHAT_PATH}#stream",
                    )
                    overflow = self._as_context_error(exc)
                    if overflow is not None:
                        self._abandon_upstream()
                        reported = True
                        raise overflow from exc
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
        """One streaming attempt, raising raw httpx errors for the caller to classify.

        Native ``/api/chat`` streams newline-delimited JSON objects rather than
        SSE ``data:`` frames — one object per token, the last with ``done``
        true and the token counts on it. There is no ``[DONE]`` sentinel and no
        usage opt-in to send.

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
                    self._client.stream("POST", _CHAT_PATH, json=body)
                )
                if resp.status_code >= 400:
                    # Read the body before raising: on a streamed response it is
                    # unread, and the error mapper reads it to build the 502
                    # detail.
                    await resp.aread()
            resp.raise_for_status()
            lines = resp.aiter_lines()
            tool_calls_seen = 0
            while True:
                async with deadline_scope(deadline):
                    try:
                        raw_line = await anext(lines)
                    except StopAsyncIteration:
                        return
                if cancel is not None and bool(cancel):
                    return
                payload = raw_line.strip()
                if not payload:
                    continue
                try:
                    event = json.loads(payload)
                except json.JSONDecodeError:
                    log.warning("ollama_http.stream.bad_json", payload=payload[:200])
                    continue
                if not isinstance(event, dict):
                    continue
                # An error can arrive mid-stream on a 200 response line; the
                # transport never sees it, so it is classified here.
                error = event.get("error")
                if error:
                    raise UpstreamGenerationError(
                        error_type="upstream_http_error",
                        backend=self.backend_name,
                        model=self._model_id or "",
                        detail=str(error)[:500],
                    )
                message = event.get("message") or {}
                tool_call_deltas = self._tool_calls_from_native(
                    message.get("tool_calls"), offset=tool_calls_seen
                )
                if tool_call_deltas:
                    tool_calls_seen += len(tool_call_deltas)
                done = bool(event.get("done"))
                yield StreamChunk(
                    text=message.get("content") or "",
                    finish_reason=(event.get("done_reason") or "stop") if done else None,
                    # The terminal object carries the counts; earlier ones
                    # don't, and None keeps them out of the route's accounting.
                    prompt_tokens=int(event.get("prompt_eval_count") or 0) if done else None,
                    completion_tokens=int(event.get("eval_count") or 0) if done else None,
                    tool_call_deltas=tool_call_deltas,
                )
                if done:
                    return

    # ------------------------------------------------------------------
    # transport — one attempt, the retry loop over it, and error mapping
    # ------------------------------------------------------------------

    async def _post_once(
        self,
        body: dict,
        deadline: UpstreamDeadline,
        path: str = _CHAT_PATH,
    ) -> httpx.Response:
        """One attempt, bounded by what is LEFT of the deadline, not all of it."""
        assert self._client is not None
        async with deadline_scope(deadline):
            response = await self._client.post(path, json=body)
        response.raise_for_status()
        return response

    async def _post_with_retries(
        self,
        body: dict,
        deadline: UpstreamDeadline,
        path: str = _CHAT_PATH,
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
                return await self._post_once(body, deadline, path)
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
                        operation=path,
                    )
                    raise
                log_retry(
                    backend=self.backend_name,
                    model=self._model_id or "",
                    attempt=attempt,
                    plan=plan,
                    exc=exc,
                    operation=path,
                )
                await asyncio.sleep(plan.delay_seconds)

    @staticmethod
    def _error_text(exc: Exception) -> str:
        """Whatever the upstream said, flattened to one searchable string."""
        if isinstance(exc, httpx.HTTPStatusError):
            try:
                payload = exc.response.json()
            except ValueError:
                return exc.response.text
            if isinstance(payload, dict):
                error = payload.get("error")
                if isinstance(error, dict):
                    return str(error.get("message") or error)
                if error is not None:
                    return str(error)
            return json.dumps(payload, sort_keys=True)
        return str(exc)

    def _as_context_error(self, exc: Exception) -> ContextLengthExceededError | None:
        """Translate Ollama's overflow error into the typed one; else ``None``.

        This is the same contract the llama.cpp path already offers — a
        deterministic ``400 context_length_exceeded`` rather than an opaque
        502 — but it can only fire where the upstream actually raises. With
        context shift enabled (Ollama's default) an over-long prompt is
        silently truncated instead, and nothing in the response says so: the
        ``prompt_eval_count`` reported is the count after truncation. See the
        module docstring.
        """
        if not isinstance(exc, httpx.HTTPStatusError | UpstreamGenerationError):
            return None
        text = (
            exc.detail
            if isinstance(exc, UpstreamGenerationError)
            else self._error_text(exc)
        ) or ""
        lowered = text.lower()
        match = _CTX_OVERFLOW_RE.search(text)
        if match is None and not any(p in lowered for p in _CTX_OVERFLOW_PHRASES):
            return None
        return ContextLengthExceededError(
            requested_tokens=int(match.group(1)) if match else None,
            context_window=int(match.group(2)) if match else self._num_ctx,
            backend=self.backend_name,
        )

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

    async def embed(self, inputs: list[str]) -> EmbeddingResult:
        """Embed via Ollama's OpenAI-compatible ``/v1/embeddings``.

        Ollama serves embeddings natively and its shim returns the OpenAI
        shape verbatim, so this is a straight passthrough rather than the
        capability probe the in-process llama.cpp adapter needs: there is no
        "decoder-only GGUF misused as an embedder" failure mode to detect
        here, because the upstream refuses those itself with a 400.

        This existed as an unconditional ``EmbeddingsNotSupportedError``
        while ollama_http was only a chat fallback. That stopped being
        harmless once it became the primary source: ``embeddinggemma:300m``
        is an embedding model that *only* this backend can open, so every
        ``/v1/embeddings`` call against it returned 501.

        ``data`` is sorted by ``index`` before the vectors are taken. The
        contract is "one vector per input, in request order" and the shim is
        not required to preserve order on the wire.
        """
        if not self.is_loaded:
            raise RuntimeError("model not loaded")
        if not inputs:
            return EmbeddingResult(embeddings=[], prompt_tokens=0)

        assert self._client is not None
        body = {"model": self._model_id, "input": list(inputs)}
        deadline = UpstreamDeadline(_deadline_seconds())
        self._begin_upstream()
        reported = False
        try:
            try:
                response = await self._post_with_retries(body, deadline, _EMBEDDINGS_PATH)
                payload = response.json()
            except httpx.HTTPStatusError as exc:
                # Ollama answers 501 when the model is not an embedding model
                # at all -- a decoder-only chat GGUF, say. That is a statement
                # about the MODEL, not the health of the deployment, so it must
                # not count as a breaker failure and must not surface as a 500.
                # It is exactly the condition the route already renders as HTTP
                # 501 "embeddings not supported by <backend>", which also tells
                # the caller what to do: load a real embedding model.
                #
                # in-process llama.cpp permits this same misuse (it has an
                # explicit decoder-only serial path), so a store that embedded
                # with a chat model under gguf-first ordering loses that here.
                # Honest 501 beats a silent quality regression on vectors no
                # encoder ever produced.
                reported = True
                if exc.response.status_code == 501:
                    self._abandon_upstream()
                    raise EmbeddingsNotSupportedError(self.backend_name) from exc
                self._finish_upstream(exc)
                raise self._upstream_http_error(exc) from exc
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

        data = payload.get("data")
        if not isinstance(data, list) or len(data) != len(inputs):
            raise UpstreamGenerationError(
                error_type="upstream_contract_error",
                backend=self.backend_name,
                model=self._model_id or "",
                detail=(
                    f"expected {len(inputs)} embeddings, got "
                    f"{len(data) if isinstance(data, list) else type(data).__name__}"
                ),
            )
        ordered = sorted(data, key=lambda item: item.get("index", 0))
        usage = payload.get("usage") or {}
        prompt_tokens = usage.get("prompt_tokens") or usage.get("total_tokens") or 0
        self._last_embed_action = "upstream"
        return EmbeddingResult(
            embeddings=[item["embedding"] for item in ordered],
            prompt_tokens=int(prompt_tokens),
        )

    @property
    def last_embed_action(self) -> str:
        """What the last ``embed()`` did — read by the embeddings coalescer."""
        return self._last_embed_action

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
