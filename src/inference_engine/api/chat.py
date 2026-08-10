from __future__ import annotations

import json
import time
import uuid
from dataclasses import replace
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import ValidationError
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
import asyncio

from ..config import settings
from ..evals import PolicyEntry
from ..genai_metrics import genai_metrics
from ..observability import get_logger, span
from ..registry.breaker import breaker_span_attrs
from ..structured_outputs import (
    ResponseNotJson,
    SchemaViolation,
    repair_instruction,
    validate_json_document,
)
from .. import structured_output_capability as capability
from ..schemas import (
    AutoEvalSpec,
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionDelta,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionTokenLogprob,
    ChatMessage,
    ChoiceLogprobs,
    PromptTokensDetails,
    ToolCall,
    ToolCallDelta,
    ToolCallFunction,
    ToolCallFunctionDelta,
    Usage,
    chat_content_text,
)
from .. import usage_ledger
from . import _auto_eval, _fallback, _guardrail, _model_routing, _tool_audit, _usage
from ._scheduling import acquire_slot, scheduler_span_attrs
from .state import app_state

router = APIRouter()
log = get_logger("api.chat")

# Marks a delta that has not cleared the streaming output guard yet. The keys
# never reach the wire: ``_guarded`` replaces every marker with a real chunk
# built from the text the guard released. Answer text and reasoning text are
# separate channels with separate windows, so a pattern is never assembled out
# of two channels that the client renders apart.
_GUARDED_TEXT = "__guarded_text__"
_GUARDED_REASONING = "__guarded_reasoning__"

# SSE answers 200 even when the stream ends in an ``{"event": "error"}`` frame,
# so the ledger outcome has to come from the resolved finish reason rather than
# the HTTP status. Anything unmapped bills as "ok" — a cancelled stream still
# consumed tokens, and ``finish_reason`` keeps that distinction readable.
_STREAM_OUTCOMES = {"timeout": "timeout", "error": "error"}


def _params_from_request(req: ChatCompletionRequest) -> GenerationParams:
    stop: list[str] | None
    if req.stop is None:
        stop = None
    elif isinstance(req.stop, str):
        stop = [req.stop]
    else:
        stop = list(req.stop)

    rf = req.response_format
    # ``json_mode`` stays true for both JSON mode and Structured Outputs so the
    # existing JSON-repair path keeps applying; ``json_schema`` is what tells
    # an adapter it can constrain the sampler instead of merely asking nicely.
    json_mode = bool(rf and rf.wants_json)
    schema = rf.schema_payload if rf else None

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
        json_schema=schema,
        json_schema_name=(rf.json_schema.name if rf and rf.json_schema else "response"),
        json_schema_strict=bool(rf and rf.json_schema and rf.json_schema.strict),
        tools=tools,
        tool_choice=req.tool_choice,
        parallel_tool_calls=req.parallel_tool_calls,
        chat_template_kwargs=req.chat_template_kwargs,
        frequency_penalty=req.frequency_penalty,
        presence_penalty=req.presence_penalty,
        repetition_penalty=req.repetition_penalty,
        logit_bias=req.logit_bias,
        logprobs=req.logprobs,
        top_logprobs=req.top_logprobs,
    )


def _usage_from(prompt_tokens: int, completion_tokens: int, cached_tokens: int) -> Usage:
    """Build the wire ``usage`` object, including the cached-token breakdown.

    ``prompt_tokens_details`` stays ``null`` when nothing was served from cache
    rather than reporting a zero — same convention the ``fallback_*`` fields
    use, and it keeps "this backend can't measure it" distinguishable from
    "measured, and it was zero".

    ``cached_tokens`` is clamped to ``prompt_tokens``. llama.cpp's cache entries
    are keyed on the full context that produced them — prompt *plus* whatever
    that call generated — so a matched entry's token count can legitimately
    exceed the current request's prompt. Reporting ``cached > prompt`` would be
    incoherent to every cost dashboard downstream, so the sub-claim is the one
    we make.
    """
    cached = max(0, min(cached_tokens, prompt_tokens))
    return Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        prompt_tokens_details=(
            PromptTokensDetails(cached_tokens=cached) if cached > 0 else None
        ),
    )


def _choice_logprobs(entries: list[dict] | None) -> ChoiceLogprobs | None:
    """Validate adapter-supplied logprob dicts into the response model."""
    if not entries:
        return None
    try:
        return ChoiceLogprobs(
            content=[ChatCompletionTokenLogprob.model_validate(e) for e in entries]
        )
    except ValidationError:
        # A backend that reports logprobs in a shape we don't recognise should
        # not fail the whole completion — drop the field and keep the answer.
        log.warning("logprobs.unrecognized_shape")
        return None


def _genai_request_attrs(
    adapter: InferenceAdapter,
    model_name: str,
    params: GenerationParams,
    *,
    stream: bool,
) -> dict:
    """OTel GenAI semantic-convention request attributes.

    ``gen_ai.system`` is kept alongside the newer ``gen_ai.provider.name``:
    recent semconv versions renamed it, but the orchestra-python-sdk's own
    instrumentation (``prometa/integrations/openai.py``) still writes
    ``gen_ai.system``, and dropping it would split existing dashboards across
    two attribute names mid-flight. Emitting both costs one attribute and keeps
    old and new queries working.
    """
    return {
        "gen_ai.system": adapter.backend_name,
        "gen_ai.provider.name": adapter.backend_name,
        "gen_ai.operation.name": "chat",
        "gen_ai.request.model": model_name,
        "gen_ai.request.max_tokens": params.max_tokens,
        "gen_ai.request.temperature": params.temperature,
        "gen_ai.request.top_p": params.top_p,
        "gen_ai.request.stream": stream,
    }


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


def _end_user_attrs(user: str | None) -> dict:
    """Stamp OpenAI's ``user`` field onto the span as ``user.id``.

    OpenAI defines this as a stable end-user identifier for abuse tracing. It
    slots naturally beside the engine's existing tenant/key attribution, and
    stamping it is the only thing that makes accepting the field meaningful.
    Truncated because the value is caller-controlled and ends up in every
    exported span.
    """
    if not user:
        return {}
    return {"user.id": user[:128]}


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
    # Caller-supplied span attributes, threaded to model.acquire and the
    # generate/stream spans alike. ``user`` rides along with intent because
    # both describe who asked, not what the engine did.
    intent_attrs = {**_intent_attrs(req), **_end_user_attrs(req.user)}
    params = _params_from_request(req)
    input_token_upper_bound = _model_routing.chat_input_token_upper_bound(req)
    output_token_budget = int(params.max_tokens or 0)
    # Before enforcement, so a denied request still records the model that was
    # asked for and the bounds it was judged against.
    _usage.bind_request(
        identity=identity,
        requested_model=req.model,
        operation="chat",
        stream=bool(req.stream),
        input_token_upper_bound=input_token_upper_bound,
        output_token_budget=output_token_budget,
    )
    decision = await _model_routing.enforce_generation_request(
        identity=identity,
        requested_model=req.model,
        input_token_upper_bound=input_token_upper_bound,
        output_token_budget=output_token_budget,
    )
    # A stream hands its reservation to the SSE generator, which settles it
    # once real token counts exist; every other exit settles here, where the
    # route returns.
    settled_by_stream = False
    try:
        # Ahead of candidate resolution: a denied prompt must not acquire a
        # model, and the caller must not pay the load latency of one.
        request_id = _guardrail.new_request_id(request)
        # Tool output first, over the same inbound ``role="tool"`` messages
        # ``_tool_audit.emit_tool_results`` walks below, so a transform lands
        # before the prompt those messages are part of is evaluated — and so a
        # deny costs no model load.
        messages = await _guardrail.guard_tool_results(
            identity=identity,
            request_id=request_id,
            messages=req.messages,
        )
        messages = await _guardrail.guard_chat_messages(
            identity=identity,
            request_id=request_id,
            messages=messages,
        )
        active = await _model_routing.resolve_initial_candidate(
            requested_model=req.model,
            decision=decision,
            identity=identity,
            extra_span_attrs=intent_attrs,
        )

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
                estimated_tokens=_estimated_chat_tokens(messages, params),
            )
            stream = EventSourceResponse(
                _stream_response(
                    active.adapter,
                    active.model_name,
                    messages,
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
                    bool(req.stream_options and req.stream_options.include_usage),
                    request_id=request_id,
                )
            )
            settled_by_stream = True
            return stream

        return await _blocking_response(
            active.adapter,
            active.model_name,
            messages,
            params,
            identity,
            auto_eval,
            policy,
            intent_attrs,
            decision,
            active.candidate_index,
            active.fallback_info,
            request_id=request_id,
        )
    finally:
        if not settled_by_stream:
            await _model_routing.settle_reservation(decision)


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


def _generation_deadline_seconds() -> float | None:
    """Total elapsed budget for one generation, or None when disabled."""
    seconds = settings.chat_completion_timeout_seconds
    return seconds if seconds > 0 else None


async def _generate_within_deadline(adapter, messages, params, model_name: str):
    """Run one generation under a TOTAL-ELAPSED cap, where that is honest.

    WHY THE ADAPTER'S OWN TIMEOUT WAS NOT ENOUGH. Every HTTP adapter passes
    ``chat_completion_timeout_seconds`` to ``httpx.Timeout(...)``, whose read
    component is PER READ OPERATION rather than total elapsed. A response that
    streams steadily — which is exactly what a slow model looks like — resets
    that clock on every chunk and never trips it, so the setting described as a
    "server-side timeout for HTTP-backed /v1/chat/completions calls" did not
    bound the call at all.

    That was not merely a long request. The scheduler lease is held for the
    duration, and a client that gives up first does not stop the generation, so
    every abandoned request leaked a slot. Measured on this engine
    (CHAT_COMPLETION_TIMEOUT_SECONDS=300): one benchmark case ran 1010s across
    two legs — past a 420s client deadline — and the three cases after it were
    refused with ``tenant_queue_timeout`` on an otherwise idle process, CPU
    under 1% and memory 73% free. Only a restart cleared it, three times in one
    afternoon, each time taking the dependent platform down.

    WHY IT IS CONDITIONAL. ``asyncio.wait_for`` cancels the await. For an
    adapter waiting on a socket that closes the upstream request and the work
    really stops. For one running blocking native code in a worker thread it
    abandons the RESULT while the thread computes on — the resource stays busy,
    and a timeout raised on its behalf would claim something that did not
    happen. ``GenerationTimeoutError``'s own docstring draws that line, so the
    deadline applies only to adapters that declare
    ``generation_is_cancellable``. Non-cancellable backends keep exactly their
    previous behaviour, including the unbounded duration: fixing those needs
    interruptible native calls, not a lie at this layer.
    """
    deadline = _generation_deadline_seconds()
    if deadline is None or not getattr(adapter, "generation_is_cancellable", False):
        return await adapter.generate(messages, params)
    try:
        return await asyncio.wait_for(adapter.generate(messages, params), deadline)
    except asyncio.TimeoutError as exc:
        log.warning(
            "generation.deadline_exceeded",
            backend=adapter.backend_name,
            model=model_name,
            deadline_seconds=deadline,
        )
        # The typed error the route already maps to a 504, raised only where
        # cancellation genuinely ended the work.
        raise GenerationTimeoutError(
            timeout_seconds=deadline,
            backend=adapter.backend_name,
            model=model_name,
        ) from exc


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
    started = time.perf_counter()
    try:
        with span(
            "chat.generate",
            **{
                **_genai_request_attrs(adapter, model_name, params, stream=False),
                "n_messages": len(messages),
                **_request_key_attrs(adapter),
                **_identity_attrs(identity),
                **(intent_attrs or {}),
                **_prefix_cache_attrs(adapter),
                **breaker_span_attrs(adapter),
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
            _usage.bind_serving(
                adapter=adapter,
                model_name=model_name,
                fallback_info=fallback_info,
            )
            # Audit inbound tool messages BEFORE generation — these record what
            # tools the agent has already executed and is now feeding back.
            n_tool_results = _tool_audit.emit_tool_results(s, list(messages))
            try:
                result = await _generate_within_deadline(adapter, messages, params, model_name)
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
                    # semconv names the plural form for the array-valued
                    # attribute; the SDK's own instrumentation reads that one.
                    "gen_ai.response.finish_reasons": result.finish_reason,
                    "gen_ai.usage.cached_input_tokens": result.cached_tokens,
                    **post_attrs,
                }
            )
            genai_metrics.record_operation(
                operation="chat",
                provider=adapter.backend_name,
                model=model_name,
                duration_seconds=time.perf_counter() - started,
                input_tokens=result.prompt_tokens,
                output_tokens=result.completion_tokens,
            )
    finally:
        await app_state.scheduler.release(lease)

    return result


def _raise_generation_http_error(exc: Exception) -> None:
    _usage.bind_error(exc)
    if isinstance(exc, HTTPException):
        raise exc
    if isinstance(exc, ContextLengthExceededError):
        raise HTTPException(status_code=400, detail=exc.error_detail()) from exc
    if isinstance(exc, GenerationTimeoutError):
        raise HTTPException(status_code=504, detail=exc.error_detail()) from exc
    if isinstance(exc, UpstreamGenerationError):
        raise HTTPException(status_code=502, detail=exc.error_detail()) from exc
    raise exc


async def _enforce_structured_output(
    result,
    adapter: InferenceAdapter,
    model_name: str,
    messages: list[ChatMessage],
    params: GenerationParams,
    identity: Identity,
    intent_attrs: dict | None = None,
    routing_decision: _model_routing.ModelRoutingDecision | None = None,
):
    """Validate the document, and decide who to blame when it is wrong.

    A no-op unless the caller asked for a JSON Schema. Beyond that, this used
    to skip entirely for backends that constrain decoding, resting on a
    class-level claim. That claim is now a PRIOR (see
    structured_output_capability): every response is checked, and the belief
    about the backend decides only what to DO with a failure.

    * Trusted backend, keyword violation — AMBIGUOUS, because this codebase
      checks a subset of JSON Schema and the fault may be the checker's. Log
      and pass through, which is the outcome those backends already had and
      preserves the false-rejection safety the old skip provided.
    * Trusted backend, response is not JSON — PROOF the belief is wrong, since
      no grammar-constrained sampler emits prose. Demote the deployment and
      repair this request.
    * Untrusted backend — repair, as before.

    One retry, with the validation error fed back to the model. If the second
    attempt is still invalid we surface a typed 502 rather than handing back a
    document that silently violates the contract the caller asked for — a
    strict-mode client has no way to tell the difference otherwise.
    """
    if not params.json_schema:
        return result
    if result.tool_calls:
        # The model chose to call a tool instead of answering; there is no
        # document to validate against the schema.
        return result

    trusted = capability.constrains_decoding(adapter, model_name)

    try:
        validate_json_document(result.text, params.json_schema)
        return result
    except ResponseNotJson as exc:
        # Subclass first: `except SchemaViolation` below would swallow it.
        # Bind the message before the block ends — Python deletes the ``as``
        # name on exit, so it is not readable further down.
        first_error = str(exc)
        if trusted:
            if not (result.text or "").strip():
                # Truncation or a stop token, not an unconstrained sampler.
                # Demoting here would punish a correctly configured backend
                # that ran out of max_tokens, so the belief is left alone and
                # the repair path handles the empty document.
                log.warning(
                    "structured_output.empty_from_trusted_backend",
                    model=model_name,
                    backend=adapter.backend_name,
                )
            else:
                capability.record_unenforced(
                    adapter, model_name, evidence="response_not_json"
                )
    except SchemaViolation as exc:
        first_error = str(exc)
        if trusted:
            # AMBIGUOUS: this module checks a subset of JSON Schema, so the
            # fault may be the checker's. Passing the document through is the
            # outcome trusted backends already had, and rejecting a valid
            # document over our own gap is the worse failure.
            log.warning(
                "structured_output.trusted_backend_violation",
                model=model_name,
                backend=adapter.backend_name,
                error=first_error,
            )
            return result

    log.warning(
        "structured_output.invalid",
        model=model_name,
        backend=adapter.backend_name,
        error=first_error,
    )

    retry_messages = [
        *messages,
        ChatMessage(role="assistant", content=result.text or ""),
        ChatMessage(
            role="user",
            content=repair_instruction(
                params.json_schema, first_error, params.json_schema_name
            ),
        ),
    ]
    retried = await _generate_blocking_once(
        adapter,
        model_name,
        retry_messages,
        params,
        identity,
        intent_attrs=intent_attrs,
    )
    retried = _normalize_blocking_result(retried, params)

    try:
        validate_json_document(retried.text, params.json_schema)
    except SchemaViolation as exc:
        # This 502 does not go through _raise_generation_http_error, and
        # _blocking_response's usage bind is downstream of it, so the ledger
        # is completed here. Both generations really ran: rejecting the
        # document does not un-spend the tokens they consumed.
        _usage.bind_error_type("structured_output_invalid")
        _model_routing.observe_usage(
            routing_decision,
            model=model_name,
            input_tokens=result.prompt_tokens + retried.prompt_tokens,
            output_tokens=result.completion_tokens + retried.completion_tokens,
        )
        _usage.bind_usage(
            input_tokens=result.prompt_tokens + retried.prompt_tokens,
            output_tokens=result.completion_tokens + retried.completion_tokens,
            cached_tokens=result.cached_tokens + retried.cached_tokens,
        )
        raise HTTPException(
            status_code=502,
            detail={
                "message": (
                    f"The model could not produce a response satisfying the "
                    f"'{params.json_schema_name}' schema after one retry: {exc}"
                ),
                "type": "structured_output_invalid",
                "code": "structured_output_invalid",
                "param": "response_format",
                "backend": adapter.backend_name,
                "model": model_name,
            },
        ) from exc

    # Bill the caller for both attempts — they paid for both.
    return replace(
        retried,
        prompt_tokens=result.prompt_tokens + retried.prompt_tokens,
        completion_tokens=result.completion_tokens + retried.completion_tokens,
        cached_tokens=result.cached_tokens + retried.cached_tokens,
    )


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
    request_id: str | None = None,
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

    result = await _enforce_structured_output(
        result,
        active.adapter,
        active.model_name,
        messages,
        params,
        identity,
        intent_attrs,
        routing_decision,
    )
    # Bound here rather than inside _generate_blocking_once: the structured
    # output retry sums both attempts, and binding per-attempt would let the
    # retry's own smaller counts overwrite the total the caller is billed for.
    _model_routing.observe_usage(
        routing_decision,
        model=active.model_name,
        input_tokens=result.prompt_tokens,
        output_tokens=result.completion_tokens,
    )
    _usage.bind_usage(
        input_tokens=result.prompt_tokens,
        output_tokens=result.completion_tokens,
        cached_tokens=result.cached_tokens,
        finish_reason=result.finish_reason,
    )

    # After the usage bind on purpose: a blocked completion still consumed the
    # tokens that produced it, and the caller is billed for them either way.
    guard_request_id = request_id or _guardrail.new_request_id()
    if result.text:
        guarded_text = await _guardrail.guard_completion_text(
            identity=identity,
            request_id=guard_request_id,
            text=result.text,
        )
        if guarded_text != result.text:
            result = replace(result, text=guarded_text)
    # Reasoning content is model output the caller receives on its own channel,
    # so it is guarded on its own rather than riding on the answer's verdict.
    if result.reasoning_content:
        guarded_reasoning = await _guardrail.guard_completion_text(
            identity=identity,
            request_id=guard_request_id,
            text=result.reasoning_content,
        )
        if guarded_reasoning != result.reasoning_content:
            result = replace(result, reasoning_content=guarded_reasoning)

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
                logprobs=_choice_logprobs(result.logprobs),
            )
        ],
        usage=_usage_from(
            result.prompt_tokens, result.completion_tokens, result.cached_tokens
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
    include_usage: bool = False,
    request_id: str | None = None,
) -> AsyncIterator[dict]:
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    guard_request_id = request_id or _guardrail.new_request_id(request)
    stream_guard = _guardrail.stream_output_guard(
        identity=identity,
        request_id=guard_request_id,
    )
    reasoning_guard = _guardrail.stream_output_guard(
        identity=identity,
        request_id=guard_request_id,
    )

    def _chunk(
        delta: ChatCompletionDelta,
        finish: str | None = None,
        logprobs: ChoiceLogprobs | None = None,
    ) -> dict:
        chunk = ChatCompletionChunk(
            id=completion_id,
            created=created,
            model=model_name,
            request_key_source=_request_key_source(adapter),
            **_fallback.response_fields(fallback_info),
            choices=[
                ChatCompletionChunkChoice(
                    index=0, delta=delta, finish_reason=finish, logprobs=logprobs
                )
            ],
        )
        return {"data": chunk.model_dump_json()}

    def _usage_chunk(usage: Usage) -> dict:
        """The ``stream_options.include_usage`` trailer.

        OpenAI's shape for this frame is an empty ``choices`` array with the
        token counts attached — clients keyed on ``choices[0]`` skip it
        naturally and only usage-aware clients read it.
        """
        chunk = ChatCompletionChunk(
            id=completion_id,
            created=created,
            model=model_name,
            request_key_source=_request_key_source(adapter),
            **_fallback.response_fields(fallback_info),
            choices=[],
            usage=usage,
        )
        return {"data": chunk.model_dump_json()}

    try:
        finish_reason: str | None = None
        chunks_emitted = 0
        role_emitted = False
        cancelled = False
        # False while the adapter loop could still be producing; a teardown
        # reached with it False is a caller that walked away mid-generation.
        adapter_finished = False
        accumulated: list[str] = []
        # Token accounting for the usage trailer. Adapters report these on a
        # terminal, text-free StreamChunk — either from the upstream's own
        # include_usage frame (vLLM, Ollama, OpenRouter) or counted locally
        # (llama.cpp, MLX).
        usage_prompt_tokens = 0
        usage_completion_tokens = 0
        # TTFT / TPOT timing. ``stream_started`` is stamped immediately before
        # the adapter call so prefill is inside the measurement; scheduler wait
        # is not, since it's already reported separately as scheduler.wait_ms.
        stream_started = 0.0
        first_token_at: float | None = None
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
                    if stream_guard is None:
                        chunks_emitted += 1
                        accumulated.append(d.content)
                        out.append(_chunk(ChatCompletionDelta(content=d.content)))
                    else:
                        out.append({_GUARDED_TEXT: d.content})
                if d.reasoning_content:
                    if reasoning_guard is None:
                        chunks_emitted += 1
                        out.append(
                            _chunk(ChatCompletionDelta(reasoning_content=d.reasoning_content))
                        )
                    else:
                        out.append({_GUARDED_REASONING: d.reasoning_content})
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

        def _released_chunk(step, *, reasoning: bool) -> list[dict]:
            nonlocal chunks_emitted
            if not step.released:
                return []
            chunks_emitted += 1
            if reasoning:
                return [_chunk(ChatCompletionDelta(reasoning_content=step.released))]
            accumulated.append(step.released)
            return [_chunk(ChatCompletionDelta(content=step.released))]

        async def _guarded(events: list[dict]) -> tuple[list[dict], dict | None]:
            """Resolve held-back markers into wire chunks.

            Pass-through when no guard is configured, so the unguarded stream
            emits exactly the chunk sequence it always did. A frame that is not
            a marker — role, logprobs, a tool-call delta — goes out as it
            arrives and never forces a guarded channel to release early.
            Releasing it to preserve the interleaving would put text on the wire
            that the guard has not yet seen the following
            ``STREAM_HOLDBACK_CHARS`` of, and would cost a round trip per frame
            on a channel a logprobs-requesting caller interrupts every token.
            Content and tool calls reassemble into separate fields, so the
            client's message is the same either way.
            """
            if stream_guard is None or reasoning_guard is None:
                return events, None
            out: list[dict] = []
            for event in events:
                channels = (
                    (stream_guard, False, event.get(_GUARDED_TEXT)),
                    (reasoning_guard, True, event.get(_GUARDED_REASONING)),
                )
                if all(feed is None for _, _, feed in channels):
                    out.append(event)
                    continue
                for guard, is_reasoning, feed in channels:
                    if feed is None:
                        continue
                    step = await guard.feed(feed)
                    if step.blocked is not None:
                        return out, step.blocked
                    out.extend(_released_chunk(step, reasoning=is_reasoning))
            return out, None

        async with watch_disconnect(request) as cancel:
            reassembler = _tool_audit.ToolCallReassembler()

            def _piece_events(piece) -> list[dict]:
                """Convert one backend stream piece into zero or more SSE chunks."""
                nonlocal finish_reason, next_tool_index, role_emitted
                nonlocal usage_prompt_tokens, usage_completion_tokens, first_token_at

                if first_token_at is None and (piece.text or piece.tool_call_deltas):
                    first_token_at = time.perf_counter()

                # Token counts ride on their own terminal piece and must be
                # picked up before the empty-text early-outs below.
                if piece.prompt_tokens is not None:
                    usage_prompt_tokens = int(piece.prompt_tokens)
                if piece.completion_tokens is not None:
                    usage_completion_tokens = int(piece.completion_tokens)

                events: list[dict] = []
                if not role_emitted:
                    events.append(_chunk(ChatCompletionDelta(role="assistant")))
                    role_emitted = True

                if piece.logprobs:
                    resolved = _choice_logprobs(piece.logprobs)
                    if resolved is not None:
                        events.append(_chunk(ChatCompletionDelta(), logprobs=resolved))

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
                    **_genai_request_attrs(adapter, model_name, params, stream=True),
                    "n_messages": len(messages),
                    **_request_key_attrs(adapter),
                    **_identity_attrs(identity),
                    **(intent_attrs or {}),
                    **breaker_span_attrs(adapter),
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
                _usage.bind_serving(
                    adapter=adapter,
                    model_name=model_name,
                    fallback_info=fallback_info,
                )
                n_tool_results = _tool_audit.emit_tool_results(s, list(messages))
                try:
                    stream_started = time.perf_counter()
                    pieces = adapter.stream(messages, params, cancel=cancel)
                    try:
                        first_piece = await anext(pieces)
                    except StopAsyncIteration:
                        first_piece = None

                    blocked: dict | None = None
                    if first_piece is not None:
                        chunks, blocked = await _guarded(_piece_events(first_piece))
                        for chunk in chunks:
                            yield chunk
                    if blocked is None:
                        async for piece in pieces:
                            chunks, blocked = await _guarded(_piece_events(piece))
                            for chunk in chunks:
                                yield chunk
                            if blocked is not None:
                                break
                    if blocked is None:
                        if not role_emitted:
                            yield _chunk(ChatCompletionDelta(role="assistant"))
                            role_emitted = True
                        # Drain any held-back text now that the adapter is done.
                        chunks, blocked = await _guarded(_ingest(normalizer.flush()))
                        for chunk in chunks:
                            yield chunk
                    if blocked is None and stream_guard is not None and reasoning_guard is not None:
                        # The final window is always guarded, holdback included,
                        # on both channels.
                        for guard, is_reasoning in (
                            (reasoning_guard, True),
                            (stream_guard, False),
                        ):
                            step = await guard.flush()
                            blocked = step.blocked
                            if blocked is not None:
                                break
                            for chunk in _released_chunk(step, reasoning=is_reasoning):
                                yield chunk
                    if blocked is not None:
                        adapter_finished = True
                        finish_reason = "error"
                        s.bind(**{"error.type": blocked["type"]})
                        yield {"event": "error", "data": json.dumps({"error": blocked})}
                        return
                    # If the normalizer parsed tool calls out of the text stream,
                    # the canonical finish_reason is ``tool_calls`` regardless of
                    # what the adapter signalled.
                    if normalizer.has_tool_calls():
                        finish_reason = "tool_calls"
                    adapter_finished = True
                except ContextLengthExceededError as exc:
                    # The SSE response line may already be open, so we cannot
                    # reliably downgrade to a 400 here. Emit a typed terminal
                    # error event with the same payload shape as the blocking
                    # 400 body.
                    adapter_finished = True
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
                                include_usage,
                            ):
                                yield chunk
                            adapter_finished = True
                            return
                    adapter_finished = True
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
                                include_usage,
                            ):
                                yield chunk
                            adapter_finished = True
                            return
                    adapter_finished = True
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
                    resolved_finish = (
                        "cancelled" if cancelled else (finish_reason or "stop")
                    )
                    stream_attrs: dict = {
                        "gen_ai.response.finish_reason": resolved_finish,
                        "gen_ai.response.finish_reasons": resolved_finish,
                        "stream.chunks_emitted": chunks_emitted,
                        "stream.cancelled": cancelled,
                        "stream.cancel_reason": cancel.reason or "",
                        "tool_audit.tool_results_in": n_tool_results,
                        "tool_audit.tool_calls_out": n_tool_calls_streamed,
                        **_prefix_cache_post_call_attrs(adapter),
                    }
                    if usage_prompt_tokens or usage_completion_tokens:
                        stream_attrs["gen_ai.usage.input_tokens"] = usage_prompt_tokens
                        stream_attrs["gen_ai.usage.output_tokens"] = usage_completion_tokens

                    ended = time.perf_counter()
                    # ``stream_started`` is still 0.0 if we failed before the
                    # adapter call (e.g. tool-result auditing raised). Recording
                    # then would push a perf_counter epoch into the histogram.
                    measured = stream_started > 0.0
                    ttft_ms: float | None = None
                    if measured and first_token_at is not None:
                        ttft = first_token_at - stream_started
                        ttft_ms = round(ttft * 1000, 3)
                        stream_attrs["gen_ai.server.time_to_first_token"] = ttft
                        # Only a successful stream describes real serving
                        # latency — a cancelled or errored one would skew the
                        # histogram toward whatever it managed before failing.
                        if not cancelled and resolved_finish in ("stop", "length", "tool_calls"):
                            genai_metrics.record_time_to_first_token(
                                operation="chat",
                                provider=adapter.backend_name,
                                model=model_name,
                                seconds=ttft,
                            )
                            # TPOT excludes the first token by definition: it is
                            # the decode rate after prefill.
                            decode_tokens = usage_completion_tokens - 1
                            if decode_tokens > 0:
                                genai_metrics.record_time_per_output_token(
                                    operation="chat",
                                    provider=adapter.backend_name,
                                    model=model_name,
                                    seconds=(ended - first_token_at) / decode_tokens,
                                )
                    if measured and not cancelled:
                        genai_metrics.record_operation(
                            operation="chat",
                            provider=adapter.backend_name,
                            model=model_name,
                            duration_seconds=ended - stream_started,
                            input_tokens=usage_prompt_tokens or None,
                            output_tokens=usage_completion_tokens or None,
                        )
                    # Adapters report token counts on a terminal frame, so a
                    # stream torn down before its adapter finished carries at
                    # most a partial count, never the request total.
                    if cancelled or not adapter_finished:
                        _model_routing.retain_reservation(routing_decision)
                    else:
                        _model_routing.observe_usage(
                            routing_decision,
                            model=model_name,
                            input_tokens=usage_prompt_tokens,
                            output_tokens=usage_completion_tokens,
                        )
                    _usage.bind_usage(
                        input_tokens=usage_prompt_tokens,
                        output_tokens=usage_completion_tokens,
                        cached_tokens=adapter.last_cached_prompt_tokens,
                        ttft_ms=ttft_ms,
                        finish_reason=resolved_finish,
                        outcome=_STREAM_OUTCOMES.get(resolved_finish, "ok"),
                    )
                    s.bind(**stream_attrs)

        if cancelled:
            # Client is gone — no point sending the trailing frames or running
            # evals on a partial response. SSE closes naturally on generator exit.
            return

        yield _chunk(ChatCompletionDelta(), finish=finish_reason or "stop")
        if include_usage:
            yield _usage_chunk(
                _usage_from(
                    usage_prompt_tokens,
                    usage_completion_tokens,
                    adapter.last_cached_prompt_tokens,
                )
            )
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
        # Runs on a clean close, an error, and a client disconnect alike,
        # which is every way a stream that started its body can end.
        await _model_routing.settle_reservation(routing_decision)
        # On the fallback path the inner generator has already flushed the
        # model that really served; flush() is idempotent, so this second call
        # cannot produce a duplicate invoice line.
        usage_ledger.flush(usage_ledger.current())
        if scheduler_lease is not None:
            await app_state.scheduler.release(scheduler_lease)
