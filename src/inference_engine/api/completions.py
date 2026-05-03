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

from ..adapters import GenerationParams, InferenceAdapter
from ..auth import Identity, require_identity
from ..manager import ModelNotFoundError
from ..observability import span
from ..schemas import (
    CompletionChoice,
    CompletionRequest,
    CompletionResponse,
    Usage,
)
from .state import app_state

router = APIRouter()


def _identity_attrs(identity: Identity) -> dict:
    return {"prometa.tenant": identity.tenant, "prometa.key_id": identity.key_id}


async def _resolve(model_id: str) -> tuple[InferenceAdapter, str]:
    try:
        adapter, desc = await app_state.manager.get(model_id)
    except ModelNotFoundError:
        raise HTTPException(status_code=404, detail=f"model not found: {model_id!r}") from None
    return adapter, desc.qualified_name


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


@router.post("/v1/completions", response_model=CompletionResponse)
async def create_completion(
    req: CompletionRequest,
    identity: Identity = Depends(require_identity),
) -> CompletionResponse:
    prompts = [req.prompt] if isinstance(req.prompt, str) else list(req.prompt)
    if not prompts:
        raise HTTPException(status_code=400, detail="prompt must contain at least one string")

    adapter, model_name = await _resolve(req.model)
    params = _params(req)

    completion_id = f"cmpl-{uuid.uuid4().hex}"
    choices: list[CompletionChoice] = []
    total_prompt_tokens = 0
    total_completion_tokens = 0

    with span(
        "completions.run",
        **{
            "gen_ai.system": adapter.backend_name,
            "gen_ai.request.model": model_name,
            "gen_ai.request.max_tokens": params.max_tokens,
            "gen_ai.request.temperature": params.temperature,
            "completion.batch_size": len(prompts),
            **_identity_attrs(identity),
        },
    ) as s:
        for index, prompt in enumerate(prompts):
            result = await adapter.complete(prompt, params)
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

    return CompletionResponse(
        id=completion_id,
        created=int(time.time()),
        model=model_name,
        choices=choices,
        usage=Usage(
            prompt_tokens=total_prompt_tokens,
            completion_tokens=total_completion_tokens,
            total_tokens=total_prompt_tokens + total_completion_tokens,
        ),
    )
