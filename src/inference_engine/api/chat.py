from __future__ import annotations

import json
import time
import uuid
from dataclasses import replace
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, Request
from sse_starlette.sse import EventSourceResponse

from ..adapters import (
    ContextLengthExceededError,
    GenerationParams,
    GenerationTimeoutError,
    InferenceAdapter,
    UpstreamGenerationError,
)
from ..response_normalize import (
    StreamDelta,
    StreamNormalizer,
    infer_model_capabilities,
    normalize_assistant_text,
)
from ..auth import Identity, require_identity
from ..cancellation import watch_disconnect
from ..config import settings
from ..evals import PolicyEntry
from ..observability import get_logger, span
from ..schemas import (
    AutoEvalSpec,
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionDelta,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    ToolCall,
    ToolCallDelta,
    ToolCallFunction,
    ToolCallFunctionDelta,
    Usage,
    chat_content_text,
)
from . import _auto_eval, _fallback, _model_routing, _tool_audit
from ._scheduling import acquire_slot, scheduler_span_attrs
from .state import app_state

router = APIRouter()
log = get_logger("api.chat")


def _params_from_request(req: ChatCompletionRequest) -> GenerationParams:
    stop: list[str] | None
    if req.stop is None:
        stop = None
    elif isinstance(req.stop, str):
        stop = [req.stop]
    else:
        stop = list(req.stop)

    json_mode = bool(req.response_format and req.response_format.get("type") == "json_object")

    # tools come in as ToolDefinition pydantic models; the backend wants the
    # plain OpenAI dict shape, so dump back to dicts here.
    tools = [t.model_dump() for t in req.tools] if req.tools else None

    return GenerationParams(
        temperature=req.temperature if req.temperature is not None else 0.7,
        top_p=req.top_p if req.top_p is not None else 0.95,
        top_k=req.top_k if req.top_k is not None else 40,
        max_tokens=req.max_tokens if req.max_tokens is not None else 512,
        stop=stop,
        seed=req.seed,
        json_mode=json_mode,
        tools=tools,
        tool_choice=req.tool_choice,
        chat_template_kwargs=req.chat_template_kwargs,
    )


def _identity_attrs(identity: Identity) -> dict:
    """Span attributes that flag the calling tenant on every inference span."""
    attrs = {
        "prometa.tenant": identity.tenant,
        "prometa.key_id": identity.key_id,
    }
    if identity.org_id is not None:
        attrs["prometa.org_id"] = identity.org_id
    return attrs


def _request_key_source(adapter: InferenceAdapter) -> str:
    return getattr(adapter, "request_key_source", "local-inference")


def _request_key_attrs(adapter: InferenceAdapter) -> dict:
    return {"llm.request.key_source": _request_key_source(adapter)}


def _estimated_chat_tokens(messages: list[ChatMessage], params: GenerationParams) -> int:
    chars = sum(len(chat_content_text(m.content)) for m in messages)
    return max(1, (chars // 4) + int(params.max_tokens or 0))


def _intent_attrs(req: ChatCompletionRequest) -> dict:
    """Return generic caller-supplied intent span attributes."""
    if not req.intent_labels:
        return {}

    labels = list(req.intent_labels)
    label_names = list(req.intent_label_names or [])
    source = req.intent_source or "request"
    preclassified = bool(req.intent_preclassified)

    attrs: dict = {
        "intent.labels": labels,
        "intent.label_names": label_names,
        "intent.count": len(labels),
        "intent.source": source,
        "intent.preclassified": preclassified,
    }
    if req.intent_classifier_version:
        attrs["intent.classifier_version"] = req.intent_classifier_version
    return attrs


async def _resolve(
    model_id: str,
    identity: Identity,
    intent_attrs: dict | None = None,
) -> tuple[InferenceAdapter, str]:
    """Compatibility wrapper for direct, non-governed model resolution tests."""
    resolved = await _model_routing.resolve_initial_candidate(
        requested_model=model_id,
        decision=None,
        identity=identity,
        extra_span_attrs=intent_attrs,
    )
    return resolved.adapter, resolved.model_name


@router.post("/v1/chat/completions")
async def chat_completions(
    req: ChatCompletionRequest,
    request: Request,
    identity: Identity = Depends(require_identity),
):
    intent_attrs = _intent_attrs(req)
    params = _params_from_request(req)
    decision = await _model_routing.enforce_generation_request(
        identity=identity,
        requested_model=req.model,
        input_token_upper_bound=_model_routing.chat_input_token_upper_bound(req),
        output_token_budget=int(params.max_tokens or 0),
    )
    active = await _model_routing.resolve_initial_candidate(
        requested_model=req.model,
        decision=decision,
        identity=identity,
        extra_span_attrs=intent_attrs,
    )

    # Resolve the effective auto-eval spec from server policy + request.
    auto_eval, policy = _resolve_auto_eval(
        req.auto_eval, tenant=identity.tenant, model_name=active.model_name
    )
    if decision is not None and auto_eval is not None:
        _model_routing.reject_unsupported_governed_workload(
            identity=identity,
            workload="chat.auto_eval",
        )

    if req.stream and auto_eval and auto_eval.mode == "blocking":
        # Blocking auto-eval needs the full response in hand — incompatible
        # with streaming by design. Reject before we start the stream.
        raise HTTPException(
            status_code=400,
            detail="auto_eval.mode='blocking' is incompatible with stream=true",
        )

    if req.stream:
        lease = await acquire_slot(
            identity=identity,
            adapter=active.adapter,
            model_name=active.model_name,
            workload="chat.stream",
            priority=30.0,
            estimated_tokens=_estimated_chat_tokens(req.messages, params),
        )
        return EventSourceResponse(
            _stream_response(
                active.adapter,
                active.model_name,
                req.messages,
                params,
                identity,
                request,
                auto_eval,
                policy,
                intent_attrs,
                lease,
                active.fallback_info,
                decision,
                active.candidate_index,
            )
        )

    return await _blocking_response(
        active.adapter,
        active.model_name,
        req.messages,
        params,
        identity,
        auto_eval,
        policy,
        intent_attrs,
        decision,
        active.candidate_index,
        active.fallback_info,
    )


def _prefix_cache_attrs(adapter: InferenceAdapter) -> dict:
    """Per-adapter prefix-cache attrs. Backend-specific keys depend on what
    each backend can introspect.

    * llama_cpp → ``prefix_cache.capacity_bytes`` (LlamaRAMCache is byte-keyed)
    * mlx       → ``prefix_cache.tokens`` (single-slot, token-indexed)
    """
    enabled = getattr(adapter, "prefix_cache_enabled", False)
    if not enabled:
        return {"prefix_cache.enabled": False}
    attrs: dict = {"prefix_cache.enabled": True}
    if hasattr(adapter, "prefix_cache_capacity_bytes"):
        attrs["prefix_cache.capacity_bytes"] = adapter.prefix_cache_capacity_bytes
    if hasattr(adapter, "prefix_cache_tokens"):
        attrs["prefix_cache.tokens"] = adapter.prefix_cache_tokens
    return attrs


def _prefix_cache_post_call_attrs(adapter: InferenceAdapter) -> dict:
    """Counters that change per-call. Bound after generate/stream completes."""
    if not getattr(adapter, "prefix_cache_enabled", False):
        return {}
    out: dict = {}
    if hasattr(adapter, "prefix_cache_size_bytes"):
        # llama.cpp: byte-level (cache is opaque, no token counts)
        out["prefix_cache.size_bytes"] = adapter.prefix_cache_size_bytes
    if hasattr(adapter, "prefix_cache_last_overlap_tokens"):
        # mlx: token-precise reuse from this call
        out["prefix_cache.tokens_reused"] = adapter.prefix_cache_last_overlap_tokens
        out["prefix_cache.tokens_total"] = adapter.prefix_cache_last_prompt_tokens
        out["prefix_cache.action"] = adapter.prefix_cache_last_action
    return out


def _resolve_auto_eval(
    request_spec: AutoEvalSpec | None,
    *,
    tenant: str,
    model_name: str,
) -> tuple[AutoEvalSpec | None, PolicyEntry | None]:
    """Pick the effective auto-eval spec.

    Server-side policy wins when it matches — the request's ``auto_eval`` is
    ignored in that case. This keeps Prometa's policy plane authoritative
    over compliance/safety rubrics. Returns ``(spec, policy_or_None)`` so
    callers can stamp provenance onto spans.
    """
    policy = app_state.policy_registry.resolve(tenant=tenant, model=model_name)
    if policy is not None:
        return policy.spec, policy
    return request_spec, None


def _auto_eval_attrs(spec: AutoEvalSpec | None, policy: PolicyEntry | None = None) -> dict:
    if spec is None:
        return {}
    attrs: dict = {
        "auto_eval.mode": spec.mode,
        "auto_eval.rubrics": list(spec.rubrics),
        "auto_eval.judge_model": spec.judge_model or settings.default_judge_model,
        "auto_eval.from_policy": policy is not None,
    }
    if policy is not None:
        attrs["auto_eval.policy.name"] = policy.name
        attrs["auto_eval.policy.match_tenant"] = policy.match.tenant
        attrs["auto_eval.policy.match_model"] = policy.match.model
    return attrs


def _timeout_span_attrs(exc: GenerationTimeoutError) -> dict:
    attrs: dict = {
        "error.type": "generation_timeout",
        "gen_ai.response.finish_reason": "timeout",
    }
    if exc.timeout_seconds is not None:
        attrs["generation.timeout_seconds"] = exc.timeout_seconds
    return attrs


def _normalize_blocking_result(
    result,
    params: GenerationParams,
    *,
    expects_reasoning_prelude: bool = False,
):
    """Apply the vendor-XML normalizer to an adapter result.

    Idempotent over backends that already returned structured ``tool_calls``
    (e.g. vLLM with a tool parser, llama.cpp with a Nemotron grammar): we
    only strip ``<think>`` blocks from ``content`` and leave the call ids
    intact so the agent's subsequent ``tool_call_id`` replies still match.

    ``expects_reasoning_prelude`` mirrors the streaming-path knob. Set for
    reasoning-family models so unanchored prose (max_tokens exhausted before
    ``</think>`` ever appears) is classified as reasoning instead of leaking
    the chain-of-thought into ``content``. See ``normalize_assistant_text``.
    """
    if not result.text and not result.tool_calls:
        return result
    normalized = normalize_assistant_text(
        result.text,
        existing_tool_calls=result.tool_calls,
        finish_reason=result.finish_reason,
        tools_requested=bool(params.tools),
        expects_reasoning_prelude=expects_reasoning_prelude,
    )
    content = normalized.content or ""
    if params.json_mode and content:
        content = _repair_json_mode_content(content)
    if (
        content == result.text
        and normalized.tool_calls == result.tool_calls
        and normalized.reasoning_content is None
        and normalized.finish_reason == result.finish_reason
    ):
        return result
    return replace(
        result,
        text=content,
        finish_reason=normalized.finish_reason,
        tool_calls=normalized.tool_calls,
        reasoning_content=normalized.reasoning_content or result.reasoning_content,
    )


def _repair_json_mode_content(text: str) -> str:
    """Return a parseable JSON payload when only fence residue surrounds it.

    Some local VLMs satisfy a JSON prompt but leave a stray closing code fence
    after the object. In JSON mode, keep the object and discard only that
    non-JSON residue. Do not invent JSON when the payload itself is malformed.
    """
    stripped = text.strip()
    if not stripped:
        return text
    try:
        json.loads(stripped)
    except json.JSONDecodeError:
        pass
    else:
        return stripped

    candidates = [stripped]
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].lstrip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        fenced = "\n".join(lines).strip()
        if fenced:
            candidates.append(fenced)

    decoder = json.JSONDecoder()
    for candidate in candidates:
        for marker in ("{", "["):
            start = candidate.find(marker)
            if start < 0:
                continue
            try:
                _, end = decoder.raw_decode(candidate[start:])
            except json.JSONDecodeError:
                continue
            suffix = candidate[start + end :].strip()
            if suffix in ("", "```"):
                return candidate[start : start + end].strip()
    return text


def _last_user_prompt(messages: list[ChatMessage]) -> str:
    """Pick the most recent user-turn content to use as the eval 'prompt'.

    Auto-eval treats the chat span as a single Q→A unit; the system prompt and
    earlier turns are assumed to be context, not the question being judged.
    """
    for m in reversed(messages):
        if m.role == "user":
            return chat_content_text(m.content)
    return ""


async def _generate_blocking_once(
    adapter: InferenceAdapter,
    model_name: str,
    messages: list[ChatMessage],
    params: GenerationParams,
    identity: Identity,
    auto_eval: AutoEvalSpec | None = None,
    policy: PolicyEntry | None = None,
    intent_attrs: dict | None = None,
    fallback_info: _fallback.FallbackInfo | None = None,
    routing_decision: _model_routing.ModelRoutingDecision | None = None,
    candidate_index: int | None = None,
):
    lease = await acquire_slot(
        identity=identity,
        adapter=adapter,
        model_name=model_name,
        workload="chat.generate",
        priority=20.0,
        estimated_tokens=_estimated_chat_tokens(messages, params),
    )
    cache_size_before = getattr(adapter, "prefix_cache_size_bytes", 0)
    # Capability hints feed the normalizer's ``expects_reasoning_prelude``
    # knob — reasoning-family models (Nemotron, DeepSeek-R1, QwQ, …) have a
    # chat template that silently opens ``<think>`` at the prompt, so the
    # tag never reaches us in ``result.text``. Without this signal the
    # blocking path leaks the entire chain-of-thought into ``content`` when
    # the model exhausts ``max_tokens`` before closing the block. Streaming
    # path computes the same caps below; keep them in lockstep.
    caps = infer_model_capabilities(
        model_name, backend=adapter.backend_name, fmt=getattr(adapter, "format", "")
    )
    try:
        with span(
            "chat.generate",
            **{
                "gen_ai.system": adapter.backend_name,
                "gen_ai.request.model": model_name,
                "gen_ai.request.max_tokens": params.max_tokens,
                "gen_ai.request.temperature": params.temperature,
                "n_messages": len(messages),
                **_request_key_attrs(adapter),
                **_identity_attrs(identity),
                **(intent_attrs or {}),
                **_prefix_cache_attrs(adapter),
                **_auto_eval_attrs(auto_eval, policy),
                **_fallback.span_attrs(fallback_info),
                **_model_routing.model_routing_span_attrs(
                    routing_decision,
                    candidate_model=model_name,
                    candidate_index=candidate_index,
                ),
                **scheduler_span_attrs(lease),
            },
        ) as s:
            # Audit inbound tool messages BEFORE generation — these record what
            # tools the agent has already executed and is now feeding back.
            n_tool_results = _tool_audit.emit_tool_results(s, list(messages))
            try:
                result = await adapter.generate(messages, params)
            except ContextLengthExceededError:
                # Prompt + forced generation overran the model's window. Answer with
                # a deterministic, typed 400 so clients branch on the error type
                # instead of pattern-matching an opaque 500 after a big tool result.
                s.bind(**{"error.type": "context_length_exceeded"})
                raise
            except GenerationTimeoutError as exc:
                s.bind(**_timeout_span_attrs(exc))
                raise
            # Single normalization seam: convert any leaked vendor markup
            # (Nemotron <tool_call>, DeepSeek-R1 </think>, etc.) into structured
            # OpenAI fields. Backends that already returned ``tool_calls`` keep
            # their ids — _normalize_blocking_result short-circuits on those.
            result = _normalize_blocking_result(
                result, params, expects_reasoning_prelude=bool(caps.get("reasoning"))
            )
            # Audit outbound tool calls AFTER generation — what the model
            # decided to invoke this turn.
            n_tool_calls = _tool_audit.emit_tool_calls(s, result.tool_calls)
            if n_tool_results or n_tool_calls:
                s.bind(
                    **{
                        "tool_audit.tool_results_in": n_tool_results,
                        "tool_audit.tool_calls_out": n_tool_calls,
                    }
                )
            cache_size_after = getattr(adapter, "prefix_cache_size_bytes", 0)
            post_attrs = _prefix_cache_post_call_attrs(adapter)
            # llama.cpp emits a delta because its raw cache_size grows with each
            # call; MLX exposes overlap directly so a delta is meaningless there.
            if "prefix_cache.size_bytes" in post_attrs:
                post_attrs["prefix_cache.size_delta_bytes"] = cache_size_after - cache_size_before
            s.bind(
                **{
                    "gen_ai.usage.input_tokens": result.prompt_tokens,
                    "gen_ai.usage.output_tokens": result.completion_tokens,
                    "gen_ai.response.finish_reason": result.finish_reason,
                    **post_attrs,
                }
            )
    finally:
        await app_state.scheduler.release(lease)

    return result


def _raise_generation_http_error(exc: Exception) -> None:
    if isinstance(exc, HTTPException):
        raise exc
    if isinstance(exc, ContextLengthExceededError):
        raise HTTPException(status_code=400, detail=exc.error_detail()) from exc
    if isinstance(exc, GenerationTimeoutError):
        raise HTTPException(status_code=504, detail=exc.error_detail()) from exc
    if isinstance(exc, UpstreamGenerationError):
        raise HTTPException(status_code=502, detail=exc.error_detail()) from exc
    raise exc


async def _blocking_response(
    adapter: InferenceAdapter,
    model_name: str,
    messages: list[ChatMessage],
    params: GenerationParams,
    identity: Identity,
    auto_eval: AutoEvalSpec | None = None,
    policy: PolicyEntry | None = None,
    intent_attrs: dict | None = None,
    routing_decision: _model_routing.ModelRoutingDecision | None = None,
    candidate_index: int | None = None,
    initial_fallback_info: _fallback.FallbackInfo | None = None,
) -> ChatCompletionResponse:
    active = _model_routing.ResolvedRoutingCandidate(
        adapter=adapter,
        model_name=model_name,
        candidate_index=candidate_index,
        fallback_info=initial_fallback_info,
    )
    while True:
        try:
            result = await _generate_blocking_once(
                active.adapter,
                active.model_name,
                messages,
                params,
                identity,
                auto_eval,
                policy,
                intent_attrs,
                active.fallback_info,
                routing_decision,
                active.candidate_index,
            )
            break
        except ContextLengthExceededError as exc:
            _raise_generation_http_error(exc)
        except HTTPException:
            raise
        except Exception as exc:
            if routing_decision is None and active.fallback_info is not None:
                _raise_generation_http_error(exc)
            fallback = await _model_routing.resolve_next_fallback(
                decision=routing_decision,
                current_candidate_index=active.candidate_index,
                adapter=active.adapter,
                model_name=active.model_name,
                exc=exc,
                identity=identity,
                extra_span_attrs=intent_attrs,
            )
            if fallback is None:
                _raise_generation_http_error(exc)
            active = fallback

    completion_id = f"chatcmpl-{uuid.uuid4().hex}"

    eval_results = None
    if auto_eval is not None:
        prompt_for_eval = _last_user_prompt(messages)
        if auto_eval.mode == "blocking":
            eval_results = await _auto_eval.run_blocking(
                app_state.eval_runner,
                app_state.rubric_registry,
                auto_eval,
                default_judge_model=settings.default_judge_model,
                prompt=prompt_for_eval,
                response=result.text,
                candidate_model=active.model_name,
                candidate_completion_id=completion_id,
                identity=identity,
            )
        else:
            # Background — fire-and-forget; surfaces only via spans/logs.
            _auto_eval.run_background(
                app_state.eval_runner,
                app_state.rubric_registry,
                auto_eval,
                default_judge_model=settings.default_judge_model,
                prompt=prompt_for_eval,
                response=result.text,
                candidate_model=active.model_name,
                candidate_completion_id=completion_id,
                identity=identity,
            )

    response_tool_calls: list[ToolCall] | None = None
    if result.tool_calls:
        response_tool_calls = [
            ToolCall(
                id=tc.get("id") or "",
                type=tc.get("type") or "function",
                function=ToolCallFunction(
                    name=(tc.get("function") or {}).get("name") or "",
                    arguments=(tc.get("function") or {}).get("arguments") or "",
                ),
            )
            for tc in result.tool_calls
        ]

    return ChatCompletionResponse(
        id=completion_id,
        created=int(time.time()),
        model=active.model_name,
        request_key_source=_request_key_source(active.adapter),
        **_fallback.response_fields(active.fallback_info),
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatMessage(
                    role="assistant",
                    content=result.text or None,
                    reasoning_content=result.reasoning_content,
                    tool_calls=response_tool_calls,
                ),
                finish_reason=result.finish_reason
                if result.finish_reason in ("stop", "length", "tool_calls")
                else "stop",
            )
        ],
        usage=Usage(
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            total_tokens=result.prompt_tokens + result.completion_tokens,
        ),
        evals=eval_results,
    )


async def _stream_response(
    adapter: InferenceAdapter,
    model_name: str,
    messages: list[ChatMessage],
    params: GenerationParams,
    identity: Identity,
    request: Request,
    auto_eval: AutoEvalSpec | None = None,
    policy: PolicyEntry | None = None,
    intent_attrs: dict | None = None,
    scheduler_lease=None,
    fallback_info: _fallback.FallbackInfo | None = None,
    routing_decision: _model_routing.ModelRoutingDecision | None = None,
    candidate_index: int | None = None,
) -> AsyncIterator[dict]:
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    def _chunk(delta: ChatCompletionDelta, finish: str | None = None) -> dict:
        chunk = ChatCompletionChunk(
            id=completion_id,
            created=created,
            model=model_name,
            request_key_source=_request_key_source(adapter),
            **_fallback.response_fields(fallback_info),
            choices=[ChatCompletionChunkChoice(index=0, delta=delta, finish_reason=finish)],
        )
        return {"data": chunk.model_dump_json()}

    try:
        finish_reason: str | None = None
        chunks_emitted = 0
        role_emitted = False
        cancelled = False
        accumulated: list[str] = []
        # Audit inbound tool messages first (same shape the blocking path uses).
        n_tool_results = 0
        n_tool_calls_streamed = 0

        # Streaming normalizer — converts leaked Nemotron/DeepSeek-R1 vendor XML
        # in the text stream into OpenAI ``reasoning_content`` + ``tool_calls``
        # deltas before they reach the client. Backends that emit structured
        # ``tool_call_deltas`` (vLLM tool parser, llama.cpp grammar) bypass this:
        # their tool_calls flow through untouched on the dedicated channel.
        caps = infer_model_capabilities(
            model_name, backend=adapter.backend_name, fmt=getattr(adapter, "format", "")
        )
        normalizer = StreamNormalizer(
            tools_requested=bool(params.tools),
            expects_reasoning_prelude=bool(caps.get("reasoning")),
        )
        next_tool_index = 0

        def _ingest(deltas: list[StreamDelta]) -> list[dict]:
            """Convert StreamNormalizer output into SSE-shaped chunks."""
            nonlocal next_tool_index, chunks_emitted
            out: list[dict] = []
            for d in deltas:
                if d.content:
                    chunks_emitted += 1
                    accumulated.append(d.content)
                    out.append(_chunk(ChatCompletionDelta(content=d.content)))
                if d.reasoning_content:
                    chunks_emitted += 1
                    out.append(_chunk(ChatCompletionDelta(reasoning_content=d.reasoning_content)))
                if d.tool_call:
                    tc = d.tool_call
                    fn = tc.get("function") or {}
                    idx = next_tool_index
                    next_tool_index += 1
                    # Emit the full call as a single delta — the inner XML can't
                    # be streamed token-by-token in OpenAI shape anyway, so we
                    # deliver it as one ``{id, type, function: {name, args}}``
                    # chunk and let the reassembler treat it like any other.
                    reassembler.feed(
                        [
                            {
                                "index": idx,
                                "id": tc.get("id"),
                                "type": tc.get("type", "function"),
                                "function": {
                                    "name": fn.get("name", ""),
                                    "arguments": fn.get("arguments", ""),
                                },
                            }
                        ]
                    )
                    out.append(
                        _chunk(
                            ChatCompletionDelta(
                                tool_calls=[
                                    ToolCallDelta(
                                        index=idx,
                                        id=tc.get("id"),
                                        type=tc.get("type", "function"),
                                        function=ToolCallFunctionDelta(
                                            name=fn.get("name", ""),
                                            arguments=fn.get("arguments", ""),
                                        ),
                                    )
                                ]
                            )
                        )
                    )
            return out

        async with watch_disconnect(request) as cancel:
            reassembler = _tool_audit.ToolCallReassembler()

            def _piece_events(piece) -> list[dict]:
                """Convert one backend stream piece into zero or more SSE chunks."""
                nonlocal finish_reason, next_tool_index, role_emitted

                events: list[dict] = []
                if not role_emitted:
                    events.append(_chunk(ChatCompletionDelta(role="assistant")))
                    role_emitted = True

                if piece.text:
                    events.extend(_ingest(normalizer.feed(piece.text)))
                if piece.tool_call_deltas:
                    # Backend already produced structured deltas — flush any
                    # text the normalizer was holding back (it cannot interleave
                    # with structured calls cleanly) and pass the deltas through.
                    events.extend(_ingest(normalizer.flush()))
                    reassembler.feed(piece.tool_call_deltas)
                    # Re-base our own tool index past whatever the backend emitted
                    # so post-flush normalized calls do not collide.
                    next_tool_index = max(
                        next_tool_index,
                        max(int(d.get("index", 0)) for d in piece.tool_call_deltas) + 1,
                    )
                    events.append(
                        _chunk(
                            ChatCompletionDelta(
                                tool_calls=[
                                    ToolCallDelta(
                                        index=int(d.get("index", 0)),
                                        id=d.get("id"),
                                        type=d.get("type"),
                                        function=(
                                            ToolCallFunctionDelta(**(d.get("function") or {}))
                                            if d.get("function") is not None
                                            else None
                                        ),
                                    )
                                    for d in piece.tool_call_deltas
                                ]
                            )
                        )
                    )
                if piece.finish_reason:
                    finish_reason = piece.finish_reason
                return events

            with span(
                "chat.stream",
                **{
                    "gen_ai.system": adapter.backend_name,
                    "gen_ai.request.model": model_name,
                    "n_messages": len(messages),
                    **_request_key_attrs(adapter),
                    **_identity_attrs(identity),
                    **(intent_attrs or {}),
                    **_auto_eval_attrs(auto_eval, policy),
                    **_fallback.span_attrs(fallback_info),
                    **_model_routing.model_routing_span_attrs(
                        routing_decision,
                        candidate_model=model_name,
                        candidate_index=candidate_index,
                    ),
                    **scheduler_span_attrs(scheduler_lease),
                },
            ) as s:
                n_tool_results = _tool_audit.emit_tool_results(s, list(messages))
                try:
                    pieces = adapter.stream(messages, params, cancel=cancel)
                    try:
                        first_piece = await anext(pieces)
                    except StopAsyncIteration:
                        first_piece = None

                    if first_piece is not None:
                        for chunk in _piece_events(first_piece):
                            yield chunk
                    async for piece in pieces:
                        for chunk in _piece_events(piece):
                            yield chunk
                    if not role_emitted:
                        yield _chunk(ChatCompletionDelta(role="assistant"))
                        role_emitted = True
                    # Drain any held-back text now that the adapter is done.
                    for chunk in _ingest(normalizer.flush()):
                        yield chunk
                    # If the normalizer parsed tool calls out of the text stream,
                    # the canonical finish_reason is ``tool_calls`` regardless of
                    # what the adapter signalled.
                    if normalizer.has_tool_calls():
                        finish_reason = "tool_calls"
                except ContextLengthExceededError as exc:
                    # The SSE response line may already be open, so we cannot
                    # reliably downgrade to a 400 here. Emit a typed terminal
                    # error event with the same payload shape as the blocking
                    # 400 body.
                    finish_reason = "error"
                    s.bind(**{"error.type": "context_length_exceeded"})
                    yield {"event": "error", "data": json.dumps({"error": exc.error_detail()})}
                    return
                except GenerationTimeoutError as exc:
                    if (
                        not role_emitted
                        and chunks_emitted == 0
                        and not (routing_decision is None and fallback_info is not None)
                    ):
                        fallback = await _model_routing.resolve_next_fallback(
                            decision=routing_decision,
                            current_candidate_index=candidate_index,
                            adapter=adapter,
                            model_name=model_name,
                            exc=exc,
                            identity=identity,
                            extra_span_attrs=intent_attrs,
                        )
                        if fallback is not None:
                            finish_reason = "fallback"
                            s.bind(
                                **{
                                    "llm.fallback.dispatched": True,
                                    "llm.fallback.to_model": fallback.model_name,
                                    "llm.fallback.to_backend": fallback.adapter.backend_name,
                                }
                            )
                            if scheduler_lease is not None:
                                await app_state.scheduler.release(scheduler_lease)
                                scheduler_lease = None
                            fallback_lease = await acquire_slot(
                                identity=identity,
                                adapter=fallback.adapter,
                                model_name=fallback.model_name,
                                workload="chat.stream",
                                priority=30.0,
                                estimated_tokens=_estimated_chat_tokens(messages, params),
                            )
                            async for chunk in _stream_response(
                                fallback.adapter,
                                fallback.model_name,
                                messages,
                                params,
                                identity,
                                request,
                                auto_eval,
                                policy,
                                intent_attrs,
                                fallback_lease,
                                fallback.fallback_info,
                                routing_decision,
                                fallback.candidate_index,
                            ):
                                yield chunk
                            return
                    finish_reason = "timeout"
                    s.bind(**_timeout_span_attrs(exc))
                    yield {"event": "error", "data": json.dumps({"error": exc.error_detail()})}
                    return
                except Exception as exc:
                    if (
                        not role_emitted
                        and chunks_emitted == 0
                        and not (routing_decision is None and fallback_info is not None)
                    ):
                        fallback = await _model_routing.resolve_next_fallback(
                            decision=routing_decision,
                            current_candidate_index=candidate_index,
                            adapter=adapter,
                            model_name=model_name,
                            exc=exc,
                            identity=identity,
                            extra_span_attrs=intent_attrs,
                        )
                        if fallback is not None:
                            finish_reason = "fallback"
                            s.bind(
                                **{
                                    "llm.fallback.dispatched": True,
                                    "llm.fallback.to_model": fallback.model_name,
                                    "llm.fallback.to_backend": fallback.adapter.backend_name,
                                }
                            )
                            if scheduler_lease is not None:
                                await app_state.scheduler.release(scheduler_lease)
                                scheduler_lease = None
                            fallback_lease = await acquire_slot(
                                identity=identity,
                                adapter=fallback.adapter,
                                model_name=fallback.model_name,
                                workload="chat.stream",
                                priority=30.0,
                                estimated_tokens=_estimated_chat_tokens(messages, params),
                            )
                            async for chunk in _stream_response(
                                fallback.adapter,
                                fallback.model_name,
                                messages,
                                params,
                                identity,
                                request,
                                auto_eval,
                                policy,
                                intent_attrs,
                                fallback_lease,
                                fallback.fallback_info,
                                routing_decision,
                                fallback.candidate_index,
                            ):
                                yield chunk
                            return
                    finish_reason = "error"
                    if isinstance(exc, UpstreamGenerationError):
                        error_detail = exc.error_detail()
                    else:
                        error_detail = {
                            "message": str(exc),
                            "type": "backend_error",
                            "code": "backend_error",
                        }
                    s.bind(
                        **{
                            "error.type": error_detail["type"],
                            "gen_ai.response.finish_reason": "error",
                        }
                    )
                    yield {
                        "event": "error",
                        "data": json.dumps({"error": error_detail}),
                    }
                    return
                finally:
                    cancelled = bool(cancel)
                    if not cancelled:
                        n_tool_calls_streamed = _tool_audit.emit_tool_calls(
                            s, reassembler.assembled() if reassembler.has_calls() else None
                        )
                    s.bind(
                        **{
                            "gen_ai.response.finish_reason": (
                                "cancelled" if cancelled else (finish_reason or "stop")
                            ),
                            "stream.chunks_emitted": chunks_emitted,
                            "stream.cancelled": cancelled,
                            "stream.cancel_reason": cancel.reason or "",
                            "tool_audit.tool_results_in": n_tool_results,
                            "tool_audit.tool_calls_out": n_tool_calls_streamed,
                            **_prefix_cache_post_call_attrs(adapter),
                        }
                    )

        if cancelled:
            # Client is gone — no point sending the trailing frames or running
            # evals on a partial response. SSE closes naturally on generator exit.
            return

        yield _chunk(ChatCompletionDelta(), finish=finish_reason or "stop")
        yield {"data": "[DONE]"}

        # Background auto-eval kicks off after the stream is delivered. Blocking
        # mode is rejected upfront in chat_completions(); only background reaches
        # here.
        if auto_eval is not None and auto_eval.mode == "background":
            _auto_eval.run_background(
                app_state.eval_runner,
                app_state.rubric_registry,
                auto_eval,
                default_judge_model=settings.default_judge_model,
                prompt=_last_user_prompt(messages),
                response="".join(accumulated),
                candidate_model=model_name,
                candidate_completion_id=completion_id,
                identity=identity,
            )
    finally:
        if scheduler_lease is not None:
            await app_state.scheduler.release(scheduler_lease)
