from fastapi import APIRouter, Depends, HTTPException

from ..auth import require_identity
from ..registry import OllamaRegistry, get_openrouter_probe, get_probe, get_vllm_probe
from ..response_normalize import infer_model_capabilities
from ..schemas import ModelCatalog, ModelCatalogEntry, ModelInfo, ModelList, UnavailableModel
from .state import app_state

router = APIRouter()


_BACKEND_FOR_FORMAT = {
    "gguf": "llama_cpp",
    "mlx": "mlx",
    "vllm": "vllm",
    "openrouter": "openrouter",
    "ollama_http": "ollama_http",
}


def _request_key_source_for_format(fmt: str) -> str:
    if fmt == "openrouter":
        return "openrouter-api-key"
    return "local-inference"


def _to_info(desc) -> ModelInfo:
    backend = _BACKEND_FOR_FORMAT.get(desc.format, "unknown")
    caps = infer_model_capabilities(desc.qualified_name, backend=backend, fmt=desc.format)
    return ModelInfo(
        id=desc.qualified_name,
        size_bytes=desc.size_bytes,
        backend=backend,
        format=desc.format,
        model_path=str(desc.model_path),
        request_key_source=_request_key_source_for_format(desc.format),
        **caps,
    )


def _accept_descriptor(desc) -> bool:
    if desc.format == "gguf":
        return get_probe().probe(desc).loadable
    if desc.format == "vllm":
        return get_vllm_probe().probe(desc).loadable
    if desc.format == "openrouter":
        return get_openrouter_probe().probe(desc).loadable
    return True


def _partition_models():
    return app_state.registry.list_loadable(_accept_descriptor)


def _unavailable_from_rejected(rejected) -> list[UnavailableModel]:
    unavailable: list[UnavailableModel] = []
    for desc in rejected:
        # GGUF rejections carry a structured probe reason; non-GGUF
        # rejections shouldn't really happen (we accept them
        # unconditionally) but we surface them honestly if they do.
        if desc.format == "gguf":
            result = get_probe().probe(desc)
            unavailable.append(
                UnavailableModel(
                    id=desc.qualified_name,
                    reason=result.reason or "load_failed",
                    detail=result.detail,
                    backend=_BACKEND_FOR_FORMAT.get(desc.format, "unknown"),
                    format=desc.format,
                )
            )
        elif desc.format == "vllm":
            result = get_vllm_probe().probe(desc)
            unavailable.append(
                UnavailableModel(
                    id=desc.qualified_name,
                    reason=result.reason or "vllm_unavailable",
                    detail=result.detail,
                    backend=_BACKEND_FOR_FORMAT.get(desc.format, "unknown"),
                    format=desc.format,
                )
            )
        elif desc.format == "openrouter":
            result = get_openrouter_probe().probe(desc)
            unavailable.append(
                UnavailableModel(
                    id=desc.qualified_name,
                    reason=result.reason or "openrouter_unavailable",
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
    return unavailable


def _append_registry_skips(
    unavailable: list[UnavailableModel],
    *,
    available_ids: set[str],
) -> None:
    # Tack on registry-level skips (cloud-only manifests, missing blobs, ...)
    # which never reach the probe stage — but skip any id that another
    # source (e.g. ollama_http) successfully covered, so a cloud manifest
    # served by Ollama doesn't appear in both ``data`` and ``unavailable``.
    for skip in _collect_registry_skips():
        if skip.id not in available_ids:
            unavailable.append(skip)
    unavailable.sort(key=lambda m: m.id)


def _supports_images(modality: str | None) -> bool | None:
    if modality is None:
        return None
    return "image" in modality.lower()


def _catalog_entry(desc) -> ModelCatalogEntry:
    backend = _BACKEND_FOR_FORMAT.get(desc.format, "unknown")
    params = desc.params or {}
    modality = params.get("modality")
    supports_json_mode = params.get("supports_json_mode")
    if supports_json_mode is None and desc.format in {"openrouter", "vllm"}:
        supports_json_mode = True
    provider = params.get("provider") or (
        "openrouter" if desc.format == "openrouter" else backend
    )
    return ModelCatalogEntry(
        id=desc.qualified_name,
        provider=str(provider),
        backend=backend,
        format=desc.format,
        registry=desc.registry,
        namespace=desc.namespace,
        model_path=str(desc.model_path) if desc.model_path else None,
        upstream_model_id=(
            str(params["model_id"]) if params.get("model_id") is not None else None
        ),
        request_key_source=_request_key_source_for_format(desc.format),
        size_bytes=desc.size_bytes,
        modality=str(modality) if modality is not None else None,
        supports_images=_supports_images(str(modality)) if modality is not None else None,
        context_length=(
            int(params["context_length"]) if params.get("context_length") is not None else None
        ),
        max_image_size=(
            str(params["max_image_size"]) if params.get("max_image_size") is not None else None
        ),
        max_image_side_px=(
            int(params["max_image_side_px"])
            if params.get("max_image_side_px") is not None
            else None
        ),
        max_image_pixels=(
            int(params["max_image_pixels"]) if params.get("max_image_pixels") is not None else None
        ),
        supports_json_mode=(
            bool(supports_json_mode) if supports_json_mode is not None else None
        ),
        family=str(params["family"]) if params.get("family") is not None else None,
        profile=str(params["profile"]) if params.get("profile") is not None else None,
        parameter_count_b=(
            float(params["parameter_count_b"])
            if params.get("parameter_count_b") is not None
            else None
        ),
        open_weight=bool(params["open_weight"]) if params.get("open_weight") is not None else None,
        proprietary=bool(params["proprietary"]) if params.get("proprietary") is not None else None,
        commercial_use=params.get("commercial_use"),
        benchmark_only=(
            bool(params["benchmark_only"]) if params.get("benchmark_only") is not None else None
        ),
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
    loadable, rejected = _partition_models()

    available: list[ModelInfo] = [_to_info(d) for d in loadable]
    unavailable = _unavailable_from_rejected(rejected)
    _append_registry_skips(unavailable, available_ids={m.id for m in available})

    return ModelList(data=available, unavailable=unavailable)


@router.get("/v1/models.data", response_model=ModelCatalog)
async def list_model_catalog(_=Depends(require_identity)) -> ModelCatalog:
    """Return the live model catalog with provider and capability metadata.

    This is the machine-readable operator surface for benchmark harnesses and
    internal UIs. It uses the same probe-aware availability check as
    ``/v1/models`` but includes non-standard metadata such as upstream provider,
    modality, context limits, image sizing hints, and JSON-mode support.
    """
    loadable, rejected = _partition_models()
    data = [_catalog_entry(d) for d in loadable]
    unavailable = _unavailable_from_rejected(rejected)
    _append_registry_skips(unavailable, available_ids={m.id for m in data})
    return ModelCatalog(data=data, unavailable=unavailable)


@router.get("/v1/models/{model_id:path}", response_model=ModelInfo)
async def get_model(model_id: str, _=Depends(require_identity)) -> ModelInfo:
    """Resolve a single model id, probe-aware.

    Falls through across sources the same way chat completions do, so the
    response describes the descriptor that would actually serve the
    request — not just whichever one happened to be enumerated first.
    """
    desc = app_state.registry.resolve(model_id, _accept_descriptor)
    if desc is None:
        raise HTTPException(status_code=404, detail=f"model not found: {model_id!r}")
    return _to_info(desc)
