from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, Request
from sse_starlette.sse import EventSourceResponse

from ..adapters import GenerationParams, InferenceAdapter
from ..auth import Identity, require_identity
from ..cancellation import watch_disconnect
from ..config import settings
from ..evals import PolicyEntry
from ..manager import ModelNotFoundError
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
)
from . import _auto_eval, _tool_audit
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
    tools = (
        [t.model_dump() for t in req.tools]
        if req.tools
        else None
    )

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
    )


def _identity_attrs(identity: Identity) -> dict:
    """Span attributes that flag the calling tenant on every inference span."""
    return {
        "prometa.tenant": identity.tenant,
        "prometa.key_id": identity.key_id,
    }


async def _resolve(model_id: str, identity: Identity) -> tuple[InferenceAdapter, str]:
    """Resolve `model_id` against the manager. Returns the adapter and qualified name."""
    try:
        with span("model.acquire", model=model_id, **_identity_attrs(identity)):
            adapter, desc = await app_state.manager.get(model_id)
    except ModelNotFoundError:
        raise HTTPException(status_code=404, detail=f"model not found: {model_id!r}") from None
    return adapter, desc.qualified_name


@router.post("/v1/chat/completions")
async def chat_completions(
    req: ChatCompletionRequest,
    request: Request,
    identity: Identity = Depends(require_identity),
):
    adapter, model_name = await _resolve(req.model, identity)

    # Resolve the effective auto-eval spec from server policy + request.
    auto_eval, policy = _resolve_auto_eval(
        req.auto_eval, tenant=identity.tenant, model_name=model_name
    )

    if req.stream and auto_eval and auto_eval.mode == "blocking":
        # Blocking auto-eval needs the full response in hand — incompatible
        # with streaming by design. Reject before we start the stream.
        raise HTTPException(
            status_code=400,
            detail="auto_eval.mode='blocking' is incompatible with stream=true",
        )

    params = _params_from_request(req)

    if req.stream:
        return EventSourceResponse(
            _stream_response(
                adapter, model_name, req.messages, params, identity, request,
                auto_eval, policy,
            )
        )

    return await _blocking_response(
        adapter, model_name, req.messages, params, identity, auto_eval, policy
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


def _last_user_prompt(messages: list[ChatMessage]) -> str:
    """Pick the most recent user-turn content to use as the eval 'prompt'.

    Auto-eval treats the chat span as a single Q→A unit; the system prompt and
    earlier turns are assumed to be context, not the question being judged.
    """
    for m in reversed(messages):
        if m.role == "user":
            return m.content
    return ""


async def _blocking_response(
    adapter: InferenceAdapter,
    model_name: str,
    messages: list[ChatMessage],
    params: GenerationParams,
    identity: Identity,
    auto_eval: AutoEvalSpec | None = None,
    policy: PolicyEntry | None = None,
) -> ChatCompletionResponse:
    cache_size_before = getattr(adapter, "prefix_cache_size_bytes", 0)
    with span(
        "chat.generate",
        **{
            "gen_ai.system": adapter.backend_name,
            "gen_ai.request.model": model_name,
            "gen_ai.request.max_tokens": params.max_tokens,
            "gen_ai.request.temperature": params.temperature,
            "n_messages": len(messages),
            **_identity_attrs(identity),
            **_prefix_cache_attrs(adapter),
            **_auto_eval_attrs(auto_eval, policy),
        },
    ) as s:
        # Audit inbound tool messages BEFORE generation — these record what
        # tools the agent has already executed and is now feeding back.
        n_tool_results = _tool_audit.emit_tool_results(s, list(messages))
        result = await adapter.generate(messages, params)
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
                candidate_model=model_name,
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
                candidate_model=model_name,
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
        model=model_name,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatMessage(
                    role="assistant",
                    content=result.text or None,
                    tool_calls=response_tool_calls,
                ),
                finish_reason=result.finish_reason if result.finish_reason in ("stop", "length", "tool_calls") else "stop",
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
) -> AsyncIterator[dict]:
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    def _chunk(delta: ChatCompletionDelta, finish: str | None = None) -> dict:
        chunk = ChatCompletionChunk(
            id=completion_id,
            created=created,
            model=model_name,
            choices=[ChatCompletionChunkChoice(index=0, delta=delta, finish_reason=finish)],
        )
        return {"data": chunk.model_dump_json()}

    yield _chunk(ChatCompletionDelta(role="assistant"))

    finish_reason: str | None = None
    chunks_emitted = 0
    cancelled = False
    accumulated: list[str] = []
    # Audit inbound tool messages first (same shape the blocking path uses).
    n_tool_results = 0
    n_tool_calls_streamed = 0

    async with watch_disconnect(request) as cancel:
        reassembler = _tool_audit.ToolCallReassembler()
        with span(
            "chat.stream",
            **{
                "gen_ai.system": adapter.backend_name,
                "gen_ai.request.model": model_name,
                "n_messages": len(messages),
                **_identity_attrs(identity),
                **_auto_eval_attrs(auto_eval, policy),
            },
        ) as s:
            n_tool_results = _tool_audit.emit_tool_results(s, list(messages))
            try:
                async for piece in adapter.stream(messages, params, cancel=cancel):
                    if piece.text:
                        chunks_emitted += 1
                        accumulated.append(piece.text)
                        yield _chunk(ChatCompletionDelta(content=piece.text))
                    if piece.tool_call_deltas:
                        # Accumulate for end-of-stream audit emission.
                        reassembler.feed(piece.tool_call_deltas)
                        # Pass deltas through to the SSE client in OpenAI's
                        # standard wire format — clients reassemble themselves.
                        yield _chunk(
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
                    if piece.finish_reason:
                        finish_reason = piece.finish_reason
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
