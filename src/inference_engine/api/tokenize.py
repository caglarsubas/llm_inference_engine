"""``/tokenize`` and ``/detokenize`` — vLLM/TGI-shaped tokenizer probes.

Lets a client size a prompt against the model's real context window *before*
paying for a generation that would be rejected with
``context_length_exceeded``. Without this the only way to learn a request is
too big is to send it.

Only the in-process backends can answer: llama.cpp and MLX hold the vocabulary
themselves, so the counts are exact. The HTTP-proxy adapters (vLLM, Ollama,
OpenRouter) don't have the tokenizer locally and return 501 — the same shape
the embeddings route uses for unsupported backends. For those, point the client
at the upstream's own ``/tokenize``.

Tokenization is CPU-bound and fast, but it is not free on a long prompt, so it
runs off the event loop thread.
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, HTTPException

from ..adapters import InferenceAdapter, TokenizationNotSupportedError
from ..auth import Identity, require_identity
from ..manager import ModelNotFoundError
from ..observability import span
from ..schemas import (
    DetokenizeRequest,
    DetokenizeResponse,
    TokenizeRequest,
    TokenizeResponse,
)
from . import _model_routing
from .state import app_state

router = APIRouter()

# Guard against a caller asking us to materialise an enormous string from a
# token array. Well above any real context window, low enough to bound memory.
_MAX_DETOKENIZE_TOKENS = 1_000_000


def _identity_attrs(identity: Identity) -> dict:
    return {"prometa.tenant": identity.tenant, "prometa.key_id": identity.key_id}


async def _resolve(model_id: str) -> tuple[InferenceAdapter, str]:
    try:
        adapter, desc = await app_state.manager.get(model_id)
    except ModelNotFoundError:
        raise HTTPException(
            status_code=404,
            detail={
                "message": f"model not found: {model_id!r}",
                "type": "model_not_found",
                "code": "model_not_found",
                "param": "model",
            },
        ) from None
    return adapter, desc.qualified_name


def _unsupported(adapter: InferenceAdapter) -> HTTPException:
    return HTTPException(
        status_code=501,
        detail={
            "message": (
                f"backend {adapter.backend_name!r} does not expose a local tokenizer; "
                "query the upstream server's own /tokenize endpoint instead"
            ),
            "type": "tokenization_not_supported",
            "code": "tokenization_not_supported",
            "param": "model",
            "backend": adapter.backend_name,
        },
    )


@router.post("/tokenize", response_model=TokenizeResponse)
async def tokenize(
    req: TokenizeRequest,
    identity: Identity = Depends(require_identity),
) -> TokenizeResponse:
    _model_routing.reject_unsupported_governed_workload(
        identity=identity,
        workload="tokenize.run",
    )
    adapter, model_name = await _resolve(req.model)

    with span(
        "tokenize.run",
        **{
            "gen_ai.system": adapter.backend_name,
            "gen_ai.operation.name": "tokenize",
            "gen_ai.request.model": model_name,
            **_identity_attrs(identity),
        },
    ) as s:
        if req.messages is not None:
            # Route chat input through the model's own chat template so the
            # count reflects what generation would actually see — template
            # overhead is exactly the part callers underestimate.
            try:
                text = await asyncio.to_thread(adapter.format_chat_prompt, req.messages)
            except TokenizationNotSupportedError as exc:
                raise _unsupported(adapter) from exc
        else:
            text = req.prompt or ""

        try:
            tokens = await asyncio.to_thread(
                adapter.tokenize, text, add_special_tokens=req.add_special_tokens
            )
        except TokenizationNotSupportedError as exc:
            raise _unsupported(adapter) from exc

        max_model_len = adapter.max_model_len
        s.bind(
            **{
                "tokenize.count": len(tokens),
                "tokenize.max_model_len": max_model_len or 0,
            }
        )
        return TokenizeResponse(
            count=len(tokens),
            max_model_len=max_model_len,
            tokens=tokens,
            model=model_name,
        )


@router.post("/detokenize", response_model=DetokenizeResponse)
async def detokenize(
    req: DetokenizeRequest,
    identity: Identity = Depends(require_identity),
) -> DetokenizeResponse:
    _model_routing.reject_unsupported_governed_workload(
        identity=identity,
        workload="detokenize.run",
    )
    if len(req.tokens) > _MAX_DETOKENIZE_TOKENS:
        raise HTTPException(
            status_code=400,
            detail={
                "message": (
                    f"tokens exceeds the {_MAX_DETOKENIZE_TOKENS} limit "
                    f"({len(req.tokens)} supplied)"
                ),
                "type": "invalid_request_error",
                "code": "tokens_too_many",
                "param": "tokens",
            },
        )
    adapter, model_name = await _resolve(req.model)

    with span(
        "detokenize.run",
        **{
            "gen_ai.system": adapter.backend_name,
            "gen_ai.operation.name": "detokenize",
            "gen_ai.request.model": model_name,
            "detokenize.count": len(req.tokens),
            **_identity_attrs(identity),
        },
    ):
        try:
            text = await asyncio.to_thread(adapter.detokenize, req.tokens)
        except TokenizationNotSupportedError as exc:
            raise _unsupported(adapter) from exc
        except (OverflowError, ValueError) as exc:
            # An out-of-vocabulary or negative id is caller error, not ours.
            raise HTTPException(
                status_code=400,
                detail={
                    "message": f"invalid token id in request: {exc}",
                    "type": "invalid_request_error",
                    "code": "invalid_token_id",
                    "param": "tokens",
                },
            ) from exc
        return DetokenizeResponse(prompt=text, model=model_name)
