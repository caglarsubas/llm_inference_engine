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

from ..adapters import EmbeddingResult, EmbeddingsNotSupportedError, InferenceAdapter
from ..auth import Identity, require_identity
from ..observability import span
from ..schemas import (
    EmbeddingObject,
    EmbeddingRequest,
    EmbeddingResponse,
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


async def _embed_once(
    *,
    active: _model_routing.ResolvedRoutingCandidate,
    inputs: list[str],
    identity: Identity,
    decision: _model_routing.ModelRoutingDecision | None,
) -> EmbeddingResult:
    estimated_tokens = max(1, sum(len(item) for item in inputs) // 4)
    lease = await acquire_slot(
        identity=identity,
        adapter=active.adapter,
        model_name=active.model_name,
        workload="embeddings.run",
        priority=-10.0,
        estimated_tokens=estimated_tokens,
    )

    try:
        with span(
            "embeddings.run",
            **{
                "gen_ai.system": active.adapter.backend_name,
                "gen_ai.request.model": active.model_name,
                "embedding.batch_size": len(inputs),
                **_request_key_attrs(active.adapter),
                **_identity_attrs(identity),
                **_fallback.span_attrs(active.fallback_info),
                **_model_routing.model_routing_span_attrs(
                    decision,
                    candidate_model=active.model_name,
                    candidate_index=active.candidate_index,
                ),
                **scheduler_span_attrs(lease),
            },
        ) as embedding_span:
            outcome = await app_state.embed_coalescer.submit(active.adapter, inputs)
            dims = len(outcome.embeddings[0]) if outcome.embeddings else 0
            embedding_span.bind(
                **{
                    "gen_ai.usage.input_tokens": outcome.prompt_tokens,
                    "embedding.dimensions": dims,
                    "batch.id": outcome.batch_id,
                    "batch.coalesced_with": outcome.coalesced_with,
                    "batch.total_inputs": outcome.total_inputs,
                    "batch.wait_ms": round(outcome.wait_ms, 2),
                    "batch.adapter_action": outcome.adapter_action,
                }
            )
            return outcome
    finally:
        await app_state.scheduler.release(lease)


@router.post("/v1/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(
    req: EmbeddingRequest,
    identity: Identity = Depends(require_identity),
) -> EmbeddingResponse:
    inputs = [req.input] if isinstance(req.input, str) else list(req.input)
    if not inputs:
        raise HTTPException(status_code=400, detail="input must contain at least one string")

    decision = await _model_routing.enforce_generation_request(
        identity=identity,
        requested_model=req.model,
        input_token_upper_bound=_model_routing.embedding_input_token_upper_bound(inputs),
        output_token_budget=0,
    )
    active = await _model_routing.resolve_initial_candidate(
        requested_model=req.model,
        decision=decision,
        identity=identity,
    )

    while True:
        try:
            outcome = await _embed_once(
                active=active,
                inputs=inputs,
                identity=identity,
                decision=decision,
            )
            break
        except HTTPException:
            raise
        except Exception as exc:
            if decision is None:
                if isinstance(exc, EmbeddingsNotSupportedError):
                    raise HTTPException(
                        status_code=501,
                        detail=f"embeddings not supported by {exc} backend",
                    ) from exc
                raise
            fallback = await _model_routing.resolve_next_fallback(
                decision=decision,
                current_candidate_index=active.candidate_index,
                adapter=active.adapter,
                model_name=active.model_name,
                exc=exc,
                identity=identity,
            )
            if fallback is None:
                if isinstance(exc, EmbeddingsNotSupportedError):
                    raise HTTPException(
                        status_code=501,
                        detail=f"embeddings not supported by {exc} backend",
                    ) from exc
                raise
            active = fallback

    return EmbeddingResponse(
        data=[EmbeddingObject(index=i, embedding=vec) for i, vec in enumerate(outcome.embeddings)],
        model=active.model_name,
        request_key_source=_request_key_source(active.adapter),
        **_fallback.response_fields(active.fallback_info),
        usage=Usage(
            prompt_tokens=outcome.prompt_tokens,
            completion_tokens=0,
            total_tokens=outcome.prompt_tokens,
        ),
    )
