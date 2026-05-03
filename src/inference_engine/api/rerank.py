"""``/v1/rerank`` — Cohere/Jina-style relevance scoring.

Pragmatic implementation: embed the query and each document via the same
``adapter.embed()`` path the embeddings endpoint uses, then rank documents
by cosine similarity to the query. Goes through the dynamic-batching
coalescer (round 16) so concurrent rerank workloads share underlying batched
embed calls when the backend supports it.

Quality scales with the loaded embedding model. With a chat-model GGUF the
output is structurally correct but the relevance signal is weak (chat models
don't produce well-discriminated retrieval embeddings). For production RAG
drop a purpose-built embedding GGUF — `bge-small-en-v1.5`, `nomic-embed-text`,
`e5-small-v2` — into the Ollama model store and the registry picks it up.

Out of scope for this round: dedicated cross-encoder rerankers
(`bge-reranker`, `jina-reranker`) which take (query, doc) pairs through a
classification head. Those require a new adapter capability and ``embedding``
returning a scalar score per pair instead of a vector per text — distinct
enough to warrant its own round.
"""

from __future__ import annotations

import math
import time
import uuid

from fastapi import APIRouter, Depends, HTTPException

from ..adapters import EmbeddingsNotSupportedError, InferenceAdapter
from ..auth import Identity, require_identity
from ..manager import ModelNotFoundError
from ..observability import span
from ..schemas import RerankRequest, RerankResponse, RerankResult, Usage
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


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity. Returns 0.0 when either vector has zero norm — that
    happens when an embedding-mode allocation didn't produce a real vector
    (e.g. a chat model misconfigured). Better than NaN propagating into spans."""
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b, strict=True):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


@router.post("/v1/rerank", response_model=RerankResponse)
async def rerank(
    req: RerankRequest,
    identity: Identity = Depends(require_identity),
) -> RerankResponse:
    adapter, model_name = await _resolve(req.model)

    with span(
        "rerank.run",
        **{
            "gen_ai.system": adapter.backend_name,
            "gen_ai.request.model": model_name,
            "rerank.documents_count": len(req.documents),
            "rerank.top_n": req.top_n if req.top_n is not None else len(req.documents),
            **_identity_attrs(identity),
        },
    ) as s:
        # Embed query + documents in a single coalescer submit so they share
        # one underlying batched call when the model supports it.
        try:
            outcome = await app_state.embed_coalescer.submit(
                adapter, [req.query, *req.documents]
            )
        except EmbeddingsNotSupportedError as exc:
            raise HTTPException(
                status_code=501,
                detail=f"rerank not supported by {exc} backend (no embeddings)",
            ) from exc

        if len(outcome.embeddings) != 1 + len(req.documents):
            # Defensive — shouldn't happen, but if the backend silently dropped
            # an input we surface it loudly rather than mis-ranking.
            raise HTTPException(
                status_code=500,
                detail=(
                    f"embedding batch size mismatch: expected {1 + len(req.documents)}, "
                    f"got {len(outcome.embeddings)}"
                ),
            )

        query_vec = outcome.embeddings[0]
        doc_vecs = outcome.embeddings[1:]

        # Score every doc, then sort descending by relevance. Stable sort
        # keeps original index order when scores tie (important for caller
        # determinism on degenerate inputs).
        scored = [(i, _cosine(query_vec, dv)) for i, dv in enumerate(doc_vecs)]
        scored.sort(key=lambda t: t[1], reverse=True)
        if req.top_n is not None:
            scored = scored[: req.top_n]

        s.bind(
            **{
                "gen_ai.usage.input_tokens": outcome.prompt_tokens,
                "embedding.dimensions": len(query_vec),
                "rerank.results_returned": len(scored),
                "batch.adapter_action": outcome.adapter_action,
            }
        )

    results = [
        RerankResult(
            index=idx,
            relevance_score=float(score),
            document=req.documents[idx] if req.return_documents else None,
        )
        for idx, score in scored
    ]

    return RerankResponse(
        id=f"rerank-{uuid.uuid4().hex}",
        created=int(time.time()),
        model=model_name,
        results=results,
        usage=Usage(
            prompt_tokens=outcome.prompt_tokens,
            completion_tokens=0,
            total_tokens=outcome.prompt_tokens,
        ),
    )
