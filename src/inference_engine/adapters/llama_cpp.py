"""llama.cpp adapter via llama-cpp-python.

Loads GGUF blobs from the Ollama blob store directly. On Apple Silicon, build
llama-cpp-python with `CMAKE_ARGS="-DGGML_METAL=on"` to offload to the GPU.

Threading model: llama_cpp.Llama isn't safe for concurrent calls on a single
instance, so each request acquires a per-model lock. The actual generation
runs in a thread pool so the FastAPI event loop stays responsive.

Prompt cache (multi-slot, byte-keyed)
-------------------------------------

We install ``LlamaRAMCache`` — llama-cpp-python's content-addressed prefix
cache — but wrapped in a ``_TrackedLlamaRAMCache`` subclass that records
per-call hits/misses and the matched-prefix token length. ``LlamaRAMCache``
already does best-prefix lookup with LRU eviction internally, so we don't
re-implement slot management here the way we do on the MLX side; instead we
deeply instrument what's already there.

What this gives us in symmetry with ``MLXAdapter``:

* ``prefix_cache_last_overlap_tokens`` — exact token count reused on the most
  recent cache hit (read directly off the matched ``LlamaState.n_tokens``).
* ``prefix_cache_last_action`` — ``hit`` / ``miss`` / ``disabled`` /
  ``unconsulted`` (when llama.cpp's internal ``_input_ids`` already covered
  the prefix and the cache was never asked).

What it doesn't give us: within-conversation continuation reuse. When the
new prompt extends the exact previous prompt, llama.cpp matches it from
``_input_ids`` directly without consulting the cache — so the ``hit_count``
under-reports total reuse. The dominant case it does measure is the one
multi-slot was meant to solve: alternating prefixes / cross-conversation reuse.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import AsyncIterator, Iterable
from typing import Any

from ..cancellation import Cancellation
from ..config import settings
from ..observability import get_logger
from ..registry import ModelDescriptor
from ..schemas import ChatMessage, dump_chat_content
from .base import (
    ContextLengthExceededError,
    EmbeddingResult,
    GenerationParams,
    GenerationResult,
    InferenceAdapter,
    StreamChunk,
    TokenizationNotSupportedError,
)

log = get_logger("adapter.llama_cpp")

# Floor for the per-model context clamp. A model whose trained context is below
# this (rare for chat GGUFs) still loads at its own size; the floor only guards
# against a bogus/zero ``n_ctx_train`` collapsing the window to nothing.
_MIN_N_CTX = 512

# llama.cpp raises ``ValueError`` with this shape when the tokenized prompt does
# not fit ``n_ctx``: "Requested tokens (9001) exceed context window of 8192".
# We translate that into a typed ContextLengthExceededError so the API answers
# 400 instead of 500. Matched loosely (lower-cased substring) to survive minor
# wording drift across llama-cpp-python releases; the integers are best-effort.
_CTX_OVERFLOW_RE = re.compile(
    r"requested\s+tokens?\s*\((\d+)\).*context\s+window\s+of\s*(\d+)",
    re.IGNORECASE | re.DOTALL,
)


def _openai_logprob_content(logprobs: Any) -> list[dict] | None:
    """Extract the OpenAI ``logprobs.content`` list from a llama.cpp choice.

    llama-cpp-python already converts its native token-parallel-arrays shape
    into the chat shape (``{"content": [...], "refusal": None}``) before
    returning, so this is just the unwrap plus a defensive type check.
    """
    if not isinstance(logprobs, dict):
        return None
    content = logprobs.get("content")
    if not isinstance(content, list) or not content:
        return None
    return content


# ---------------------------------------------------------------------------
# Tracked LlamaRAMCache — built lazily on first use so the heavy llama_cpp
# import isn't paid at module-load time. The class is cached on the function
# object so subsequent calls reuse the same subclass.
# ---------------------------------------------------------------------------


def _tracked_cache(capacity_bytes: int) -> Any:
    cls = getattr(_tracked_cache, "_cls", None)
    if cls is None:
        from llama_cpp import LlamaRAMCache  # noqa: PLC0415

        class _TrackedLlamaRAMCache(LlamaRAMCache):
            """LlamaRAMCache + per-call hit/miss tracking.

            Reads ``state.n_tokens`` from the cache hit's ``LlamaState`` to
            report the matched-prefix length token-precise on every call.
            """

            def __init__(self, capacity_bytes: int) -> None:
                super().__init__(capacity_bytes)
                self.last_hit_prefix_len: int = 0
                self.last_action: str = "none"  # none | hit | miss
                self.hit_count: int = 0
                self.miss_count: int = 0

            def begin_call(self) -> None:
                """Reset per-call counters at the start of every generate/stream.

                Without this, ``last_*`` fields would persist across calls and
                report stale data on calls that didn't consult the cache (e.g.
                within-conversation continuations where llama.cpp's internal
                ``_input_ids`` covered the prefix).
                """
                self.last_hit_prefix_len = 0
                self.last_action = "unconsulted"

            def __getitem__(self, key):  # noqa: ANN001
                try:
                    state = super().__getitem__(key)
                except KeyError:
                    self.last_action = "miss"
                    self.miss_count += 1
                    raise
                self.last_hit_prefix_len = int(state.n_tokens)
                self.last_action = "hit"
                self.hit_count += 1
                return state

        _tracked_cache._cls = _TrackedLlamaRAMCache  # type: ignore[attr-defined]
        cls = _TrackedLlamaRAMCache
    return cls(capacity_bytes)


class LlamaCppAdapter(InferenceAdapter):
    backend_name = "llama_cpp"
    # llama-cpp-python compiles response_format["schema"] to a GBNF grammar.
    supports_structured_outputs = True

    def __init__(self) -> None:
        self._llm: Any = None  # llama_cpp.Llama (lazy import to keep startup quick)
        self._descriptor: ModelDescriptor | None = None
        self._lock = asyncio.Lock()
        self._cache: Any = None  # _TrackedLlamaRAMCache when enabled
        # Per-call stamp — the prompt-token count from the response.usage. Bound
        # alongside the cache's last_hit_prefix_len so spans can compute hit rate.
        self._last_prompt_tokens: int = 0
        # Embedding batch capability — None until first probed, then True/False.
        # Decoder-only chat GGUFs trip ``llama_decode returned -1`` on batched
        # embedding decode; encoder embedding GGUFs (bge, nomic, e5) handle it
        # fine. We probe once at the first batch call and cache the answer.
        self._supports_batched_embed: bool | None = None
        # Per-call introspection for the embed path: "batch" | "serial" |
        # "fallback" (tried batch, hit error, retried serial).
        self._last_embed_action: str = "none"

    @property
    def is_loaded(self) -> bool:
        return self._llm is not None

    @property
    def loaded_model(self) -> ModelDescriptor | None:
        return self._descriptor

    # ------------------------------------------------------------------
    # Prefix cache introspection
    # ------------------------------------------------------------------

    @property
    def prefix_cache_enabled(self) -> bool:
        return self._cache is not None

    @property
    def prefix_cache_capacity_bytes(self) -> int:
        return self._cache.capacity_bytes if self._cache is not None else 0

    @property
    def prefix_cache_size_bytes(self) -> int:
        """Bytes currently consumed by cached prefix states. 0 when disabled."""
        return self._cache.cache_size if self._cache is not None else 0

    # MLX-parity per-call properties. ``last_*`` reflect the most recent
    # generate/stream call's cache lookup. Read after the call completes.

    @property
    def prefix_cache_last_overlap_tokens(self) -> int:
        return self._cache.last_hit_prefix_len if self._cache is not None else 0

    @property
    def prefix_cache_last_prompt_tokens(self) -> int:
        return self._last_prompt_tokens

    @property
    def prefix_cache_last_action(self) -> str:
        if self._cache is None:
            return "disabled"
        return self._cache.last_action

    # Embedding-batch introspection.
    @property
    def supports_batched_embed(self) -> bool | None:
        """``None`` if not yet probed, else True/False from the first attempt."""
        return self._supports_batched_embed

    @property
    def last_embed_action(self) -> str:
        return self._last_embed_action

    @staticmethod
    def _effective_n_ctx(requested: int, n_ctx_train: int) -> int:
        """Clamp the configured context ceiling to what the model supports.

        ``settings.n_ctx`` is a *ceiling* (default 32768), not a fixed size.
        We size the actual KV cache to ``min(requested, n_ctx_train)`` so:

        * a long-context model (Nemotron, Qwen3) gets the full ceiling rather
          than the legacy 8192 that was truncating reasoning answers mid-table;
        * a short-context model (e.g. an 8192-trained GGUF) keeps its own size
          — no wasted KV memory, no RoPE-extrapolation past the trained window.

        ``n_ctx_train <= 0`` means "unknown" (probe missed / non-llama build),
        so we fall back to the requested ceiling unchanged. A small floor keeps
        a bogus zero from collapsing the window.
        """
        if n_ctx_train and n_ctx_train > 0:
            return max(min(requested, n_ctx_train), _MIN_N_CTX)
        return requested

    def _resolve_n_ctx(self, descriptor: ModelDescriptor) -> int:
        """Compute this model's effective context window.

        Reuses the startup load-probe's cached ``n_ctx_train`` (read off GGUF
        metadata) so we don't pay a second metadata load here. Best-effort:
        any probe failure falls back to the configured ceiling.
        """
        n_ctx_train = 0
        try:
            from ..registry.probe import get_probe  # noqa: PLC0415 — avoid import cycle at module load

            n_ctx_train = int(get_probe().probe(descriptor).n_ctx_train)
        except Exception:  # noqa: BLE001 — sizing is advisory; never block load
            n_ctx_train = 0
        return self._effective_n_ctx(settings.n_ctx, n_ctx_train)

    @staticmethod
    def _as_context_error(exc: Exception, *, backend: str) -> ContextLengthExceededError | None:
        """Translate llama.cpp's context-overflow ``ValueError`` into the typed
        ``ContextLengthExceededError``; return ``None`` for any other error so
        callers re-raise it untouched."""
        if not isinstance(exc, ValueError):
            return None
        m = _CTX_OVERFLOW_RE.search(str(exc))
        if m is None:
            return None
        requested, window = int(m.group(1)), int(m.group(2))
        return ContextLengthExceededError(
            requested_tokens=requested,
            context_window=window,
            backend=backend,
        )

    async def load(self, descriptor: ModelDescriptor) -> None:
        if descriptor.format != "gguf":
            raise ValueError(f"LlamaCppAdapter only handles gguf, got {descriptor.format!r}")
        if self._descriptor and self._descriptor.model_path == descriptor.model_path:
            return

        from llama_cpp import Llama  # noqa: PLC0415  (lazy import; heavy native module)

        effective_n_ctx = self._resolve_n_ctx(descriptor)

        log.info(
            "loading_model",
            model=descriptor.qualified_name,
            model_path=str(descriptor.model_path),
            size_bytes=descriptor.size_bytes,
            n_gpu_layers=settings.n_gpu_layers,
            n_ctx=effective_n_ctx,
            n_ctx_ceiling=settings.n_ctx,
        )

        # Free previous model first to keep RAM bounded.
        await self.unload()

        def _load() -> Any:
            return Llama(
                model_path=str(descriptor.model_path),
                n_gpu_layers=settings.n_gpu_layers,
                n_ctx=effective_n_ctx,
                n_threads=settings.n_threads or None,
                n_batch=settings.n_batch,
                verbose=False,
                # Allocate the embedding pooling layer at load time so the same
                # adapter can serve both /v1/chat/completions and /v1/embeddings.
                # For Llama-architecture models this doesn't degrade chat; for
                # purpose-built embedding GGUFs (bge, nomic, e5, …) this makes
                # create_embedding produce useful vectors.
                embedding=settings.llama_cpp_embedding_enabled,
                # llama-cpp-python auto-detects the chat format from GGUF metadata
                # when chat_format is not set; we rely on that for now.
            )

        self._llm = await asyncio.to_thread(_load)
        self._descriptor = descriptor

        # Install the prompt-prefix cache. llama-cpp-python keys cached states
        # by token-prefix hash, so subsequent requests sharing a prefix skip
        # prefill of those tokens. Big TTFT win for RAG / multi-turn workloads
        # where the system prompt + tools dominate the input. We use the
        # tracked subclass so per-call hit/overlap surfaces on spans.
        if settings.prefix_cache_bytes > 0:
            self._cache = _tracked_cache(settings.prefix_cache_bytes)
            self._llm.set_cache(self._cache)
            log.info(
                "prefix_cache_enabled",
                model=descriptor.qualified_name,
                capacity_bytes=self._cache.capacity_bytes,
            )

        log.info("model_loaded", model=descriptor.qualified_name)

    async def unload(self) -> None:
        if self._llm is not None:
            log.info("unloading_model", model=self._descriptor.qualified_name if self._descriptor else None)
            self._llm = None
            self._descriptor = None
            self._cache = None
            self._last_prompt_tokens = 0
            self._supports_batched_embed = None
            self._last_embed_action = "none"

    @staticmethod
    def _arguments_for_template(raw: str) -> Any:
        """Convert OpenAI-shaped ``tool_calls[].function.arguments`` (a JSON-encoded
        string) into the shape Jinja chat templates actually expect.

        Many HuggingFace-style GGUF templates (Nemotron, Qwen-coder, GLM-4-tool, …)
        iterate the arguments with filters like ``arguments | items`` — they assume
        a mapping. ``transformers.apply_chat_template`` quietly parses the JSON
        string into a dict before rendering; llama-cpp-python's Jinja path does
        not. Without this shim, the second turn of any tool-calling conversation
        (where the assistant's prior ``tool_calls`` come back from the client as
        OpenAI-spec strings) crashes with ``TypeError: Can only get item pairs
        from a mapping.`` on the template side.

        On parse failure (the value really is a free-form string, e.g.
        ``"hello"``), return the raw value so OpenAI-strict templates that index
        ``arguments`` as a string still work. Empty strings become ``{}`` for
        the same reason the HF helper does: a zero-argument call is a mapping
        with no keys, not a string of length zero.
        """
        if not raw:
            return {}
        try:
            return json.loads(raw)
        except (TypeError, ValueError):
            return raw

    @classmethod
    def _to_llama_messages(cls, messages: Iterable[ChatMessage]) -> list[dict]:
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
                            "arguments": cls._arguments_for_template(tc.function.arguments),
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

    @staticmethod
    def _sampling_kwargs(params: GenerationParams, *, chat: bool = True) -> dict:
        """Sampling knobs shared by the chat and raw-completion paths.

        Only forwards what the caller actually set — llama.cpp's own defaults
        (``repeat_penalty=1.0``, both OpenAI penalties at 0.0) are better left
        alone than overwritten with ours.

        ``chat`` selects the logprobs calling convention: the chat entry point
        takes ``logprobs: bool`` plus ``top_logprobs: int``, while the raw
        ``create_completion`` takes a single ``logprobs: int`` count.
        """
        kw: dict = {}
        if params.frequency_penalty is not None:
            kw["frequency_penalty"] = params.frequency_penalty
        if params.presence_penalty is not None:
            kw["presence_penalty"] = params.presence_penalty
        if params.repetition_penalty is not None:
            # llama.cpp calls this repeat_penalty; same multiplicative knob.
            kw["repeat_penalty"] = params.repetition_penalty
        if params.logit_bias:
            # llama.cpp keys logit_bias by int token id, OpenAI by string.
            bias: dict[int, float] = {}
            for token_id, value in params.logit_bias.items():
                try:
                    bias[int(token_id)] = float(value)
                except (TypeError, ValueError):
                    continue
            if bias:
                kw["logit_bias"] = bias
        if params.logprobs:
            # llama-cpp-python forwards ``top_logprobs`` as the underlying
            # completion's ``logprobs`` count, and passes ``None`` when it is
            # unset — which silently disables logprobs entirely. Default to 1
            # so ``logprobs=true`` alone still returns the chosen token.
            top_n = params.top_logprobs if params.top_logprobs is not None else 1
            if chat:
                kw["logprobs"] = True
                kw["top_logprobs"] = top_n
            else:
                kw["logprobs"] = top_n
        return kw

    @classmethod
    def _completion_kwargs(cls, params: GenerationParams) -> dict:
        kw: dict = {
            "temperature": params.temperature,
            "top_p": params.top_p,
            "top_k": params.top_k,
            "max_tokens": params.max_tokens,
        }
        if params.stop:
            kw["stop"] = params.stop
        if params.seed is not None:
            kw["seed"] = params.seed
        if params.json_mode:
            # llama-cpp-python compiles a GBNF grammar from ``schema`` and
            # constrains the sampler with it, so a Structured Outputs request
            # is genuinely enforced here rather than merely requested. Without
            # a schema this degrades to plain "must be valid JSON" mode.
            response_format: dict = {"type": "json_object"}
            if params.json_schema:
                response_format["schema"] = params.json_schema
            kw["response_format"] = response_format
        if params.tools:
            kw["tools"] = params.tools
        if params.tool_choice is not None:
            kw["tool_choice"] = params.tool_choice
        kw.update(cls._sampling_kwargs(params))
        return kw

    async def generate(
        self,
        messages: Iterable[ChatMessage],
        params: GenerationParams,
        cancel: Cancellation | None = None,  # noqa: ARG002 — not honored for blocking; see stream()
    ) -> GenerationResult:
        """Blocking inference. Cancellation is **not** honored mid-generation here.

        ``Llama.create_chat_completion(stream=False)`` runs the full C++ token
        loop before returning, and that high-level entry point does not accept
        a ``stopping_criteria`` argument. Honoring cancellation would require
        dropping to ``Llama.__call__``, reimplementing the chat templating, and
        forfeiting auto-detected chat formats. Streaming requests honor cancel
        correctly — agents that need fast cancel should use ``stream=true``.
        """
        if not self.is_loaded:
            raise RuntimeError("model not loaded")

        msgs = self._to_llama_messages(messages)
        kwargs = self._completion_kwargs(params)

        async with self._lock:
            if self._cache is not None:
                self._cache.begin_call()

            def _run() -> dict:
                return self._llm.create_chat_completion(messages=msgs, stream=False, **kwargs)

            try:
                result = await asyncio.to_thread(_run)
            except ValueError as exc:
                ctx_err = self._as_context_error(exc, backend=self.backend_name)
                if ctx_err is not None:
                    raise ctx_err from exc
                raise

            usage = result.get("usage", {}) or {}
            self._last_prompt_tokens = int(usage.get("prompt_tokens", 0))

        choice = result["choices"][0]
        message = choice.get("message") or {}
        usage = result.get("usage", {}) or {}
        tool_calls = message.get("tool_calls")
        # Vendor-XML repair (Nemotron etc.) is centralised in the chat layer
        # so MLX, vLLM, and Ollama-HTTP get the same treatment without each
        # adapter re-implementing it.
        return GenerationResult(
            text=message.get("content") or "",
            finish_reason=choice.get("finish_reason") or "stop",
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            tool_calls=list(tool_calls) if tool_calls else None,
            cached_tokens=self.last_cached_prompt_tokens,
            logprobs=_openai_logprob_content(choice.get("logprobs")),
        )

    async def stream(
        self,
        messages: Iterable[ChatMessage],
        params: GenerationParams,
        cancel: Cancellation | None = None,
    ) -> AsyncIterator[StreamChunk]:
        if not self.is_loaded:
            raise RuntimeError("model not loaded")

        msgs = self._to_llama_messages(messages)
        kwargs = self._completion_kwargs(params)

        # Streaming cancellation works by breaking out of the Python iterator
        # on the producer side. The next() call drives the C++ token loop; once
        # we stop iterating, the underlying generator goes out of scope and the
        # C++ inference halts. No stopping_criteria injection needed.

        # llama-cpp-python is sync; bridge its iterator to async via a queue.
        queue: asyncio.Queue[StreamChunk | None | Exception] = asyncio.Queue(maxsize=64)
        loop = asyncio.get_running_loop()

        def _producer() -> None:
            # llama.cpp's streaming path emits no usage block, so we account for
            # tokens ourselves: it yields exactly one event per sampled token
            # (plus a leading role-only event and a trailing finish-only one),
            # and ``Llama.n_tokens`` is the KV position — prompt + generated —
            # once the loop ends. That gives both halves without a second
            # tokenize pass over the templated prompt, which we cannot see from
            # out here anyway.
            completion_tokens = 0
            try:
                try:
                    iterator = self._llm.create_chat_completion(messages=msgs, stream=True, **kwargs)
                except ValueError as exc:
                    ctx_err = self._as_context_error(exc, backend=self.backend_name)
                    if ctx_err is not None:
                        raise ctx_err from exc
                    raise
                for event in iterator:
                    if cancel is not None and bool(cancel):
                        # Belt-and-braces: stopping_criteria already bailed out
                        # of the C++ loop, but we may have a buffered chunk.
                        break
                    choice = event["choices"][0]
                    delta = choice.get("delta") or {}
                    text = delta.get("content") or ""
                    tool_call_deltas = delta.get("tool_calls")
                    finish = choice.get("finish_reason")
                    if "content" in delta:
                        completion_tokens += 1
                    chunk = StreamChunk(
                        text=text,
                        finish_reason=finish,
                        tool_call_deltas=list(tool_call_deltas) if tool_call_deltas else None,
                        logprobs=_openai_logprob_content(choice.get("logprobs")),
                    )
                    asyncio.run_coroutine_threadsafe(queue.put(chunk), loop).result()
                # Terminal accounting chunk — no text, just the counts the
                # route layer needs for ``stream_options.include_usage``.
                total_tokens = int(getattr(self._llm, "n_tokens", 0) or 0)
                prompt_tokens = max(0, total_tokens - completion_tokens)
                self._last_prompt_tokens = prompt_tokens
                asyncio.run_coroutine_threadsafe(
                    queue.put(
                        StreamChunk(
                            text="",
                            prompt_tokens=prompt_tokens,
                            completion_tokens=completion_tokens,
                        )
                    ),
                    loop,
                ).result()
                asyncio.run_coroutine_threadsafe(queue.put(None), loop).result()
            except Exception as exc:  # noqa: BLE001 — surface to consumer
                asyncio.run_coroutine_threadsafe(queue.put(exc), loop).result()

        async with self._lock:
            if self._cache is not None:
                self._cache.begin_call()
            producer_task = asyncio.to_thread(_producer)
            producer_future = asyncio.ensure_future(producer_task)
            try:
                while True:
                    item = await queue.get()
                    if item is None:
                        return
                    if isinstance(item, Exception):
                        raise item
                    yield item
            finally:
                if not producer_future.done():
                    await producer_future

    async def complete(
        self,
        prompt: str,
        params: GenerationParams,
        cancel: Cancellation | None = None,  # noqa: ARG002 — same limit as generate()
    ) -> GenerationResult:
        """Raw text completion via ``Llama.create_completion`` — no chat template.

        Same blocking-mode cancellation caveat as ``generate()``: llama-cpp-python's
        high-level entry point doesn't expose a stopping-criteria hook on
        ``create_completion(stream=False)``, so cancellation is a no-op here.
        Streaming completions are deferred for a future round.
        """
        if not self.is_loaded:
            raise RuntimeError("model not loaded")

        kwargs: dict = {
            "temperature": params.temperature,
            "top_p": params.top_p,
            "top_k": params.top_k,
            "max_tokens": params.max_tokens,
        }
        if params.stop:
            kwargs["stop"] = params.stop
        if params.seed is not None:
            kwargs["seed"] = params.seed
        kwargs.update(self._sampling_kwargs(params, chat=False))

        async with self._lock:
            if self._cache is not None:
                self._cache.begin_call()

            def _run() -> dict:
                return self._llm.create_completion(prompt=prompt, stream=False, **kwargs)

            try:
                result = await asyncio.to_thread(_run)
            except ValueError as exc:
                ctx_err = self._as_context_error(exc, backend=self.backend_name)
                if ctx_err is not None:
                    raise ctx_err from exc
                raise
            usage = result.get("usage", {}) or {}
            self._last_prompt_tokens = int(usage.get("prompt_tokens", 0))

        choice = (result.get("choices") or [{}])[0]
        return GenerationResult(
            text=choice.get("text") or "",
            finish_reason=choice.get("finish_reason") or "stop",
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            tool_calls=None,  # raw completion never produces tool calls
            cached_tokens=self.last_cached_prompt_tokens,
        )

    # ------------------------------------------------------------------
    # Tokenizer surface — llama.cpp holds the vocabulary in-process, so we
    # can answer /tokenize and /detokenize exactly rather than estimating.
    # ------------------------------------------------------------------

    def tokenize(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
        if not self.is_loaded:
            raise RuntimeError("model not loaded")
        return [
            int(t)
            for t in self._llm.tokenize(text.encode("utf-8"), add_bos=add_special_tokens)
        ]

    def detokenize(self, tokens: list[int]) -> str:
        if not self.is_loaded:
            raise RuntimeError("model not loaded")
        return self._llm.detokenize([int(t) for t in tokens]).decode("utf-8", errors="replace")

    def format_chat_prompt(self, messages: Iterable[ChatMessage]) -> str:
        """Render the GGUF's own Jinja chat template.

        llama-cpp-python applies this internally on the generation path and
        doesn't expose the rendered string, so we rebuild the formatter from
        the same GGUF metadata it reads (``tokenizer.chat_template`` plus the
        BOS/EOS tokens). A GGUF with no embedded template — a base model, say —
        has nothing to render and raises instead of guessing a format.
        """
        if not self.is_loaded:
            raise RuntimeError("model not loaded")

        metadata = getattr(self._llm, "metadata", None) or {}
        template = metadata.get("tokenizer.chat_template")
        if not template:
            raise TokenizationNotSupportedError(
                f"{self.backend_name}: model has no embedded chat template; "
                "use 'prompt' instead of 'messages'"
            )

        from llama_cpp.llama_chat_format import Jinja2ChatFormatter  # noqa: PLC0415

        def _special(kind: str) -> str:
            token_id = metadata.get(f"tokenizer.ggml.{kind}_token_id")
            if token_id is None:
                return ""
            try:
                return self._llm.detokenize([int(token_id)]).decode("utf-8", errors="replace")
            except Exception:  # noqa: BLE001 — a missing special token is not fatal
                return ""

        formatter = Jinja2ChatFormatter(
            template=template,
            eos_token=_special("eos"),
            bos_token=_special("bos"),
        )
        return formatter(messages=self._to_llama_messages(messages)).prompt

    @property
    def max_model_len(self) -> int | None:
        if not self.is_loaded:
            return None
        try:
            return int(self._llm.n_ctx())
        except Exception:  # noqa: BLE001 — advisory only
            return None

    async def embed(self, inputs: list[str]) -> EmbeddingResult:
        """Compute embedding vectors via ``Llama.create_embedding``.

        Two-tier strategy with capability detection:

        1. **First call**: try the **batched** path (``create_embedding(input=list)``).
           Encoder embedding GGUFs (bge, nomic, e5) handle this fine and we get
           a true GPU batch.
        2. **If batch fails** with ``llama_decode returned …`` (typical for
           decoder-only chat GGUFs being misused as embedders), set
           ``_supports_batched_embed=False`` permanently for this adapter and
           fall back to **serial** per-input calls.
        3. **Subsequent calls** go straight to whichever path the probe
           settled on — no re-trying the failed batch path.

        The single-input case always uses ``create_embedding(input=str)``
        regardless; that path is uniform.
        """
        if not self.is_loaded:
            raise RuntimeError("model not loaded")
        if not settings.llama_cpp_embedding_enabled:
            from .base import EmbeddingsNotSupportedError  # noqa: PLC0415

            raise EmbeddingsNotSupportedError(
                "llama_cpp embedding mode disabled — set LLAMA_CPP_EMBEDDING_ENABLED=true"
            )

        async with self._lock:
            return await self._embed_locked(inputs)

    async def _embed_locked(self, inputs: list[str]) -> EmbeddingResult:
        # Single-input always uses the same call path; no decision needed.
        if len(inputs) == 1:
            self._last_embed_action = "serial"
            return await self._embed_serial(inputs)

        # Try batched first if we haven't decided yet — or known to work.
        if self._supports_batched_embed is None or self._supports_batched_embed:
            try:
                result = await self._embed_batched(inputs)
                if self._supports_batched_embed is None:
                    self._supports_batched_embed = True
                self._last_embed_action = "batch"
                return result
            except RuntimeError as exc:
                msg = str(exc)
                if "llama_decode" not in msg:
                    raise  # not the failure mode we expect — surface it
                log.warning(
                    "embed.batch_unsupported_fallback",
                    error=msg,
                    inputs=len(inputs),
                )
                self._supports_batched_embed = False
                # fall through to serial
                self._last_embed_action = "fallback"
        else:
            self._last_embed_action = "serial"

        return await self._embed_serial(inputs)

    async def _embed_batched(self, inputs: list[str]) -> EmbeddingResult:
        def _run() -> dict:
            return self._llm.create_embedding(input=inputs)

        result = await asyncio.to_thread(_run)
        records = sorted(
            result.get("data", []) or [], key=lambda d: int(d.get("index", 0))
        )
        embeddings: list[list[float]] = []
        for r in records:
            vec = r.get("embedding")
            if vec and isinstance(vec[0], list):
                vec = vec[0]
            embeddings.append([float(x) for x in vec])
        usage = result.get("usage", {}) or {}
        return EmbeddingResult(
            embeddings=embeddings, prompt_tokens=int(usage.get("prompt_tokens", 0))
        )

    async def _embed_serial(self, inputs: list[str]) -> EmbeddingResult:
        embeddings: list[list[float]] = []
        total_tokens = 0

        def _run_one(text: str) -> dict:
            return self._llm.create_embedding(input=text)

        for text in inputs:
            result = await asyncio.to_thread(_run_one, text)
            records = result.get("data", []) or []
            if not records:
                continue
            vec = records[0].get("embedding")
            if vec and isinstance(vec[0], list):
                vec = vec[0]
            embeddings.append([float(x) for x in vec])
            usage = result.get("usage", {}) or {}
            total_tokens += int(usage.get("prompt_tokens", 0))

        return EmbeddingResult(embeddings=embeddings, prompt_tokens=total_tokens)
