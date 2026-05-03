from fastapi import APIRouter, Depends, HTTPException

from ..auth import require_identity
from ..registry import OllamaRegistry, get_probe
from ..schemas import ModelInfo, ModelList, UnavailableModel
from .state import app_state

router = APIRouter()


_BACKEND_FOR_FORMAT = {
    "gguf": "llama_cpp",
    "mlx": "mlx",
    "vllm": "vllm",
    "ollama_http": "ollama_http",
}


def _to_info(desc) -> ModelInfo:
    return ModelInfo(
        id=desc.qualified_name,
        size_bytes=desc.size_bytes,
        backend=_BACKEND_FOR_FORMAT.get(desc.format, "unknown"),
        format=desc.format,
        model_path=str(desc.model_path),
    )


def _collect_registry_skips() -> list[UnavailableModel]:
    """Walk every composite source and harvest ``list_skipped()`` outputs.

    Only ``OllamaRegistry`` exposes that today; the helper is forward-looking
    (MLX/vLLM registries can grow the same method later without changing the
    API layer).
    """
    out: list[UnavailableModel] = []
    seen: set[str] = set()
    for source in getattr(app_state.registry, "_sources", ()):
        if not isinstance(source, OllamaRegistry):
            continue
        for skip in source.list_skipped():
            if skip.qualified_name in seen:
                continue
            seen.add(skip.qualified_name)
            out.append(
                UnavailableModel(
                    id=skip.qualified_name,
                    reason=skip.reason,
                    detail=skip.detail,
                )
            )
    return out


@router.get("/v1/models", response_model=ModelList)
async def list_models(_=Depends(require_identity)) -> ModelList:
    """Return models that are actually reachable end-to-end.

    Uses the composite registry's probe-aware ``list_loadable()`` so a
    GGUF that llama-cpp-python can't open silently falls through to the
    Ollama-HTTP source and lands in ``data`` (not ``unavailable``).
    Anything every source rejects shows up in ``unavailable`` with the
    first-source reason — typically the llama.cpp probe failure.
    """
    probe = get_probe()

    def _accept(desc) -> bool:
        if desc.format == "gguf":
            return probe.probe(desc).loadable
        return True

    loadable, rejected = app_state.registry.list_loadable(_accept)

    available: list[ModelInfo] = [_to_info(d) for d in loadable]
    unavailable: list[UnavailableModel] = []
    for desc in rejected:
        # GGUF rejections carry a structured probe reason; non-GGUF
        # rejections shouldn't really happen (we accept them
        # unconditionally) but we surface them honestly if they do.
        if desc.format == "gguf":
            result = probe.probe(desc)
            unavailable.append(
                UnavailableModel(
                    id=desc.qualified_name,
                    reason=result.reason or "load_failed",
                    detail=result.detail,
                    backend=_BACKEND_FOR_FORMAT.get(desc.format, "unknown"),
                    format=desc.format,
                )
            )
        else:
            unavailable.append(
                UnavailableModel(
                    id=desc.qualified_name,
                    reason="rejected_by_accept",
                    detail="",
                    backend=_BACKEND_FOR_FORMAT.get(desc.format, "unknown"),
                    format=desc.format,
                )
            )

    # Tack on registry-level skips (cloud-only manifests, missing blobs, …)
    # which never reach the probe stage — but skip any id that another
    # source (e.g. ollama_http) successfully covered, so a cloud manifest
    # served by Ollama doesn't appear in both ``data`` and ``unavailable``.
    available_ids = {m.id for m in available}
    for skip in _collect_registry_skips():
        if skip.id not in available_ids:
            unavailable.append(skip)
    unavailable.sort(key=lambda m: m.id)

    return ModelList(data=available, unavailable=unavailable)


@router.get("/v1/models/{model_id:path}", response_model=ModelInfo)
async def get_model(model_id: str, _=Depends(require_identity)) -> ModelInfo:
    """Resolve a single model id, probe-aware.

    Falls through across sources the same way chat completions do, so the
    response describes the descriptor that would actually serve the
    request — not just whichever one happened to be enumerated first.
    """
    probe = get_probe()

    def _accept(desc) -> bool:
        if desc.format == "gguf":
            return probe.probe(desc).loadable
        return True

    desc = app_state.registry.resolve(model_id, _accept)
    if desc is None:
        raise HTTPException(status_code=404, detail=f"model not found: {model_id!r}")
    return _to_info(desc)
