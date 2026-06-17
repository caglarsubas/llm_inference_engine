"""Shared OpenRouter fallback helpers for generation endpoints."""

from __future__ import annotations

from dataclasses import dataclass

from ..adapters import GenerationTimeoutError, InferenceAdapter
from ..auth import Identity
from ..config import settings
from ..manager import ModelNotFoundError
from ..observability import get_logger, span
from .state import app_state

log = get_logger("api.fallback")

OPENROUTER_KEY_SOURCE = "openrouter-api-key"


@dataclass(frozen=True)
class FallbackInfo:
    from_model: str
    from_backend: str
    reason: str
    error_type: str


def request_key_source(adapter: InferenceAdapter) -> str:
    return getattr(adapter, "request_key_source", "local-inference")


def response_fields(info: FallbackInfo | None) -> dict:
    if info is None:
        return {}
    return {
        "fallback_from_model": info.from_model,
        "fallback_from_backend": info.from_backend,
        "fallback_reason": info.reason,
        "fallback_error_type": info.error_type,
    }


def span_attrs(info: FallbackInfo | None) -> dict:
    if info is None:
        return {}
    return {
        "llm.fallback.active": True,
        "llm.fallback.from_model": info.from_model,
        "llm.fallback.from_backend": info.from_backend,
        "llm.fallback.reason": info.reason,
        "llm.fallback.error_type": info.error_type,
    }


def classify_error(exc: Exception) -> tuple[str, str]:
    if isinstance(exc, GenerationTimeoutError):
        return "generation_timeout", exc.__class__.__name__
    return "backend_error", exc.__class__.__name__


def _eligible_backends() -> set[str]:
    return {
        item.strip()
        for item in settings.openrouter_fallback_backends.split(",")
        if item.strip()
    }


def is_eligible_local_backend(adapter: InferenceAdapter) -> bool:
    if not settings.openrouter_fallback_enabled:
        return False
    if request_key_source(adapter) == OPENROUTER_KEY_SOURCE:
        return False
    return adapter.backend_name in _eligible_backends()


def _same_name_openrouter_candidate(model_name: str) -> str:
    name = model_name.split(":", 1)[0]
    return f"{name}:openrouter"


def fallback_candidates(model_name: str) -> list[str]:
    raw_candidates: list[str] = []
    configured = settings.openrouter_fallback_model.strip()
    if configured:
        raw_candidates.append(configured)
    raw_candidates.append(_same_name_openrouter_candidate(model_name))

    candidates: list[str] = []
    for candidate in raw_candidates:
        if candidate and candidate != model_name and candidate not in candidates:
            candidates.append(candidate)
    return candidates


async def resolve_openrouter_fallback(
    *,
    adapter: InferenceAdapter,
    model_name: str,
    exc: Exception,
    identity: Identity,
    intent_attrs: dict | None = None,
) -> tuple[InferenceAdapter, str, FallbackInfo] | None:
    if not is_eligible_local_backend(adapter):
        return None

    reason, error_type = classify_error(exc)
    info = FallbackInfo(
        from_model=model_name,
        from_backend=adapter.backend_name,
        reason=reason,
        error_type=error_type,
    )
    for candidate in fallback_candidates(model_name):
        try:
            with span(
                "model.fallback.acquire",
                model=candidate,
                **span_attrs(info),
                **(intent_attrs or {}),
                **{
                    "prometa.tenant": identity.tenant,
                    "prometa.key_id": identity.key_id,
                },
            ) as s:
                fallback_adapter, desc = await app_state.manager.get(candidate)
                s.bind(
                    **{
                        "llm.request.key_source": request_key_source(fallback_adapter),
                        "llm.fallback.to_model": desc.qualified_name,
                        "llm.fallback.to_backend": fallback_adapter.backend_name,
                    }
                )
        except ModelNotFoundError:
            continue

        if request_key_source(fallback_adapter) != OPENROUTER_KEY_SOURCE:
            log.warning(
                "openrouter_fallback.rejected_non_openrouter",
                requested_model=model_name,
                candidate_model=desc.qualified_name,
                candidate_backend=fallback_adapter.backend_name,
                candidate_key_source=request_key_source(fallback_adapter),
            )
            continue

        log.warning(
            "openrouter_fallback.selected",
            requested_model=model_name,
            requested_backend=adapter.backend_name,
            fallback_model=desc.qualified_name,
            reason=reason,
            error_type=error_type,
        )
        return fallback_adapter, desc.qualified_name, info
    return None
