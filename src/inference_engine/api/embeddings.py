"""``/v1/embeddings`` — OpenAI-compatible embeddings for RAG retrievers.

The route resolves the model through the same ``ModelManager`` chat uses (so
embeddings benefit from the same load dedup, LRU, and warm-cache behaviour),
then delegates to ``adapter.embed()``. Adapters that don't implement embeddings
raise ``EmbeddingsNotSupportedError`` and we map that to HTTP 501 with the
backend name in the body — that's the signal a deployment needs to load a
proper embedding model alongside its chat model.

Spans follow the same ``gen_ai.*`` semconv shape as chat:

    span = embeddings.run
      gen_ai.system           = llama_cpp
      gen_ai.request.model    = bge-small-en:gguf
      gen_ai.usage.input_tokens = 12
      embedding.dimensions    = 384
      embedding.batch_size    = 1
      prometa.tenant          = ...
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from ..adapters import EmbeddingsNotSupportedError, InferenceAdapter
from ..auth import Identity, require_identity
from ..manager import ModelNotFoundError
from ..observability import span
from ..schemas import (
    EmbeddingObject,
    EmbeddingRequest,
    EmbeddingResponse,
    Usage,
)
from .state import app_state

router = APIRouter()


def _identity_attrs(identity: Identity) -> dict:
    # Same shape chat uses; duplicated to keep this module self-contained.
    return {"prometa.tenant": identity.tenant, "prometa.key_id": identity.key_id}


async def _resolve(model_id: str) -> tuple[InferenceAdapter, str]:
    try:
        adapter, desc = await app_state.manager.get(model_id)
    except ModelNotFoundError:
        raise HTTPException(status_code=404, detail=f"model not found: {model_id!r}") from None
    return adapter, desc.qualified_name


@router.post("/v1/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(
    req: EmbeddingRequest,
    identity: Identity = Depends(require_identity),
) -> EmbeddingResponse:
    inputs = [req.input] if isinstance(req.input, str) else list(req.input)
    if not inputs:
        raise HTTPException(status_code=400, detail="input must contain at least one string")

    adapter, model_name = await _resolve(req.model)

    with span(
        "embeddings.run",
        **{
            "gen_ai.system": adapter.backend_name,
            "gen_ai.request.model": model_name,
            "embedding.batch_size": len(inputs),
            **_identity_attrs(identity),
        },
    ) as s:
        try:
            outcome = await app_state.embed_coalescer.submit(adapter, inputs)
        except EmbeddingsNotSupportedError as exc:
            raise HTTPException(
                status_code=501,
                detail=f"embeddings not supported by {exc} backend",
            ) from exc

        dims = len(outcome.embeddings[0]) if outcome.embeddings else 0
        s.bind(
            **{
                "gen_ai.usage.input_tokens": outcome.prompt_tokens,
                "embedding.dimensions": dims,
                # Dynamic-batching observability — every embedding span
                # carries enough to reconstruct who batched with whom.
                "batch.id": outcome.batch_id,
                "batch.coalesced_with": outcome.coalesced_with,
                "batch.total_inputs": outcome.total_inputs,
                "batch.wait_ms": round(outcome.wait_ms, 2),
                "batch.adapter_action": outcome.adapter_action,
            }
        )

    return EmbeddingResponse(
        data=[
            EmbeddingObject(index=i, embedding=vec)
            for i, vec in enumerate(outcome.embeddings)
        ],
        model=model_name,
        usage=Usage(
            prompt_tokens=outcome.prompt_tokens,
            completion_tokens=0,
            total_tokens=outcome.prompt_tokens,
        ),
    )
