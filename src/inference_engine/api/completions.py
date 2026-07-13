"""``/v1/completions`` — OpenAI legacy completions, bypasses chat templating.

Closes the original document's Phase 2 "prompt-template overrides" item. Same
auth + tenant + span surface as the chat route; the only difference is that
the prompt is passed to the model verbatim instead of being run through
``apply_chat_template``.

Multiple prompts (``prompt: list[str]``) are processed serially per-prompt
inside the adapter — same shape as the embedding serial-fallback. Choices come
back ordered by request index.

Streaming completions are not implemented; chat streaming covers the dominant
case and the legacy completions endpoint sees diminishing client traffic.
"""

from __future__ import annotations

import time
import uuid

from fastapi import APIRouter, Depends, HTTPException

from ..adapters import (
    ContextLengthExceededError,
    GenerationParams,
    GenerationTimeoutError,
    InferenceAdapter,
    UpstreamGenerationError,
)
from ..auth import Identity, require_identity
from ..observability import span
from ..schemas import (
    CompletionChoice,
    CompletionRequest,
    CompletionResponse,
    Usage,
)
from . import _fallback, _model_routing
from ._scheduling import acquire_slot, scheduler_span_attrs
from .state import app_state

router = APIRouter()


def _identity_attrs(identity: Identity) -> dict:
    attrs = {"prometa.tenant": identity.tenant, "prometa.key_id": identity.key_id}
    if identity.org_id is not None:
        attrs["prometa.org_id"] = identity.org_id
    return attrs


def _request_key_source(adapter: InferenceAdapter) -> str:
    return getattr(adapter, "request_key_source", "local-inference")


def _request_key_attrs(adapter: InferenceAdapter) -> dict:
    return {"llm.request.key_source": _request_key_source(adapter)}


def _params(req: CompletionRequest) -> GenerationParams:
    stop: list[str] | None
    if req.stop is None:
        stop = None
    elif isinstance(req.stop, str):
        stop = [req.stop]
    else:
        stop = list(req.stop)

    return GenerationParams(
        temperature=req.temperature if req.temperature is not None else 0.7,
        top_p=req.top_p if req.top_p is not None else 0.95,
        top_k=req.top_k if req.top_k is not None else 40,
        max_tokens=req.max_tokens if req.max_tokens is not None else 128,
        stop=stop,
        seed=req.seed,
    )


def _estimated_completion_tokens(prompts: list[str], params: GenerationParams) -> int:
    chars = sum(len(prompt) for prompt in prompts)
    return max(1, (chars // 4) + len(prompts) * int(params.max_tokens or 0))


@router.post("/v1/completions", response_model=CompletionResponse)
async def create_completion(
    req: CompletionRequest,
    identity: Identity = Depends(require_identity),
) -> CompletionResponse:
    prompts = [req.prompt] if isinstance(req.prompt, str) else list(req.prompt)
    if not prompts:
        raise HTTPException(status_code=400, detail="prompt must contain at least one string")

    params = _params(req)
    decision = _model_routing.enforce_generation_request(
        identity=identity,
        requested_model=req.model,
        input_token_upper_bound=_model_routing.completion_input_token_upper_bound(req),
        output_token_budget=int(params.max_tokens or 0) * len(prompts),
    )
    active = await _model_routing.resolve_initial_candidate(
        requested_model=req.model,
        decision=decision,
        identity=identity,
    )

    while True:
        try:
            choices, total_prompt_tokens, total_completion_tokens = await _complete_once(
                active.adapter,
                active.model_name,
                prompts,
                params,
                identity,
                active.fallback_info,
                decision,
                active.candidate_index,
            )
            break
        except ContextLengthExceededError as exc:
            _raise_generation_http_error(exc)
        except HTTPException:
            raise
        except Exception as exc:
            if decision is None and active.fallback_info is not None:
                _raise_generation_http_error(exc)
            fallback = await _model_routing.resolve_next_fallback(
                decision=decision,
                current_candidate_index=active.candidate_index,
                adapter=active.adapter,
                model_name=active.model_name,
                exc=exc,
                identity=identity,
            )
            if fallback is None:
                _raise_generation_http_error(exc)
            active = fallback

    return CompletionResponse(
        id=f"cmpl-{uuid.uuid4().hex}",
        created=int(time.time()),
        model=active.model_name,
        request_key_source=_request_key_source(active.adapter),
        **_fallback.response_fields(active.fallback_info),
        choices=choices,
        usage=Usage(
            prompt_tokens=total_prompt_tokens,
            completion_tokens=total_completion_tokens,
            total_tokens=total_prompt_tokens + total_completion_tokens,
        ),
    )


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


async def _complete_once(
    adapter: InferenceAdapter,
    model_name: str,
    prompts: list[str],
    params: GenerationParams,
    identity: Identity,
    fallback_info: _fallback.FallbackInfo | None = None,
    routing_decision: _model_routing.ModelRoutingDecision | None = None,
    candidate_index: int | None = None,
) -> tuple[list[CompletionChoice], int, int]:
    lease = await acquire_slot(
        identity=identity,
        adapter=adapter,
        model_name=model_name,
        workload="completions.run",
        priority=5.0 if len(prompts) == 1 else -5.0,
        estimated_tokens=_estimated_completion_tokens(prompts, params),
    )

    choices: list[CompletionChoice] = []
    total_prompt_tokens = 0
    total_completion_tokens = 0

    try:
        with span(
            "completions.run",
            **{
                "gen_ai.system": adapter.backend_name,
                "gen_ai.request.model": model_name,
                "gen_ai.request.max_tokens": params.max_tokens,
                "gen_ai.request.temperature": params.temperature,
                "completion.batch_size": len(prompts),
                **_request_key_attrs(adapter),
                **_identity_attrs(identity),
                **_fallback.span_attrs(fallback_info),
                **_model_routing.model_routing_span_attrs(
                    routing_decision,
                    candidate_model=model_name,
                    candidate_index=candidate_index,
                ),
                **scheduler_span_attrs(lease),
            },
        ) as s:
            for index, prompt in enumerate(prompts):
                try:
                    result = await adapter.complete(prompt, params)
                except ContextLengthExceededError:
                    s.bind(**{"error.type": "context_length_exceeded"})
                    raise
                except GenerationTimeoutError as exc:
                    s.bind(
                        **{
                            "error.type": "generation_timeout",
                            "gen_ai.response.finish_reason": "timeout",
                        }
                    )
                    if exc.timeout_seconds is not None:
                        s.bind(**{"generation.timeout_seconds": exc.timeout_seconds})
                    raise
                total_prompt_tokens += result.prompt_tokens
                total_completion_tokens += result.completion_tokens
                choices.append(
                    CompletionChoice(
                        text=result.text or "",
                        index=index,
                        finish_reason=(
                            result.finish_reason
                            if result.finish_reason in ("stop", "length")
                            else "stop"
                        ),
                    )
                )
            s.bind(
                **{
                    "gen_ai.usage.input_tokens": total_prompt_tokens,
                    "gen_ai.usage.output_tokens": total_completion_tokens,
                }
            )
    finally:
        await app_state.scheduler.release(lease)

    return choices, total_prompt_tokens, total_completion_tokens
