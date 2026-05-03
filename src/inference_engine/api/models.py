from fastapi import APIRouter, Depends, HTTPException

from ..auth import require_identity
from ..schemas import ModelInfo, ModelList
from .state import app_state

router = APIRouter()


_BACKEND_FOR_FORMAT = {"gguf": "llama_cpp", "mlx": "mlx", "vllm": "vllm"}


def _to_info(desc) -> ModelInfo:
    return ModelInfo(
        id=desc.qualified_name,
        size_bytes=desc.size_bytes,
        backend=_BACKEND_FOR_FORMAT.get(desc.format, "unknown"),
        format=desc.format,
        model_path=str(desc.model_path),
    )


@router.get("/v1/models", response_model=ModelList)
async def list_models(_=Depends(require_identity)) -> ModelList:
    descriptors = app_state.registry.list_models()
    return ModelList(data=[_to_info(d) for d in descriptors])


@router.get("/v1/models/{model_id:path}", response_model=ModelInfo)
async def get_model(model_id: str, _=Depends(require_identity)) -> ModelInfo:
    desc = app_state.registry.get(model_id)
    if desc is None:
        raise HTTPException(status_code=404, detail=f"model not found: {model_id!r}")
    return _to_info(desc)
