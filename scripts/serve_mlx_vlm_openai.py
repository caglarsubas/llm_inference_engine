"""Serve an MLX-VLM model through a minimal OpenAI-compatible HTTP surface.

This is an optional Apple Silicon worker for local VLMs supported by
``mlx-vlm``. Point a vLLM/OpenAI-compatible descriptor at this worker; the
main engine still probes `/v1/models` and routes through the existing HTTP
adapter, so catalog honesty and strict-smoke gates stay in one place.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


DEFAULT_MODEL_PATH = "mlx-community/Molmo-7B-D-0924-4bit"
DEFAULT_SERVED_MODEL_NAME = "allenai/Molmo-7B-D-0924"


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[dict[str, Any]]
    max_tokens: int = Field(default=512, ge=1)
    temperature: float | None = None
    stream: bool = False


@dataclass(frozen=True)
class VLMInput:
    prompt: str
    image_url: str


@dataclass(frozen=True)
class ImageResource:
    locator: str
    remove_after_use: bool = False


def extract_vlm_input(messages: list[dict[str, Any]], *, worker_label: str = "MLX-VLM") -> VLMInput:
    prompt_parts: list[str] = []
    image_url: str | None = None
    for message in messages:
        if message.get("role") not in {"user", "system"}:
            continue
        content = message.get("content")
        if isinstance(content, str):
            prompt_parts.append(content)
            continue
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "text" and part.get("text"):
                prompt_parts.append(str(part["text"]))
            elif part.get("type") == "image_url" and image_url is None:
                value = part.get("image_url")
                if isinstance(value, dict):
                    image_url = str(value.get("url") or "")
                elif value:
                    image_url = str(value)

    if image_url is None or not image_url:
        raise HTTPException(
            status_code=400,
            detail=f"{worker_label} requires one image_url content part",
        )
    prompt = "\n".join(p for p in prompt_parts if p).strip() or "Describe this image."
    return VLMInput(prompt=prompt, image_url=image_url)


def materialize_image(image_url: str, *, worker_label: str = "MLX-VLM") -> ImageResource:
    if image_url.startswith("data:"):
        header, _, encoded = image_url.partition(",")
        if not encoded or ";base64" not in header:
            raise HTTPException(status_code=400, detail="image_url data URI must be base64")
        suffix = ".png"
        media_type = header[5:].split(";", 1)[0].lower()
        if "jpeg" in media_type or "jpg" in media_type:
            suffix = ".jpg"
        elif "webp" in media_type:
            suffix = ".webp"
        try:
            raw = base64.b64decode(encoded, validate=True)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="invalid base64 image_url") from exc
        handle = tempfile.NamedTemporaryFile(
            prefix="mlx-vlm-openai-",
            suffix=suffix,
            delete=False,
        )
        with handle:
            handle.write(raw)
        return ImageResource(locator=handle.name, remove_after_use=True)

    if image_url.startswith(("http://", "https://")):
        return ImageResource(locator=image_url)
    if image_url.startswith("file://"):
        return ImageResource(locator=image_url[7:])
    path = Path(image_url)
    if path.exists():
        return ImageResource(locator=str(path))
    raise HTTPException(
        status_code=400,
        detail=(
            f"{worker_label} supports data: image URLs, HTTP(S) image URLs, "
            "or local file paths"
        ),
    )


class MLXVLMRuntime:
    def __init__(
        self,
        *,
        model_path: str | Path,
        served_model_name: str = DEFAULT_SERVED_MODEL_NAME,
        max_kv_size: int | None = None,
        worker_label: str = "MLX-VLM worker",
    ) -> None:
        self.model_path = str(model_path)
        self.served_model_name = served_model_name
        self.max_kv_size = max_kv_size
        self.worker_label = worker_label
        self._loaded = False
        self._model: Any = None
        self._processor: Any = None
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        if self._loaded:
            return
        from mlx_vlm.utils import load as mlx_vlm_load  # noqa: PLC0415

        self._model, self._processor = mlx_vlm_load(self.model_path)
        self._loaded = True

    def generate(
        self,
        prompt: str,
        image_locator: str,
        *,
        max_tokens: int,
        temperature: float | None,
    ) -> Any:
        if not self._loaded:
            raise RuntimeError(f"{self.worker_label} runtime is not loaded")
        from mlx_vlm.generate import generate as mlx_vlm_generate  # noqa: PLC0415

        kwargs: dict[str, Any] = {
            "max_tokens": max_tokens,
            "temperature": 0.0 if temperature is None else temperature,
            "verbose": False,
            "skip_special_tokens": True,
        }
        if self.max_kv_size is not None:
            kwargs["max_kv_size"] = self.max_kv_size

        with self._lock:
            return mlx_vlm_generate(
                self._model,
                self._processor,
                prompt,
                image=image_locator,
                **kwargs,
            )


def create_app(runtime: MLXVLMRuntime) -> FastAPI:
    @contextlib.asynccontextmanager
    async def _lifespan(_app: FastAPI) -> Any:
        runtime.load()
        yield

    app = FastAPI(title=runtime.worker_label, lifespan=_lifespan)

    @app.get("/v1/models")
    def _models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": runtime.served_model_name,
                    "object": "model",
                    "created": 0,
                    "owned_by": "mlx-vlm",
                }
            ],
        }

    @app.get("/v1/health")
    def _health() -> dict[str, Any]:
        return {
            "status": "ok" if runtime.loaded else "starting",
            "model": runtime.served_model_name,
            "loaded": runtime.loaded,
        }

    @app.post("/v1/chat/completions")
    async def _chat(req: ChatCompletionRequest) -> dict[str, Any]:
        if req.stream:
            raise HTTPException(
                status_code=400,
                detail=f"{runtime.worker_label} does not support stream=true",
            )
        if req.model != runtime.served_model_name:
            raise HTTPException(status_code=404, detail=f"model not found: {req.model!r}")
        vlm_input = extract_vlm_input(req.messages, worker_label=runtime.worker_label)
        image = materialize_image(vlm_input.image_url, worker_label=runtime.worker_label)
        try:
            result = runtime.generate(
                vlm_input.prompt,
                image.locator,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
            )
        except Exception as exc:  # noqa: BLE001 - mapped to OpenAI-style upstream 500
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        finally:
            if image.remove_after_use:
                with contextlib.suppress(OSError):
                    Path(image.locator).unlink()

        content = str(getattr(result, "text", result)).strip()
        prompt_tokens = int(getattr(result, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(result, "generation_tokens", 0) or 0)
        if prompt_tokens <= 0:
            prompt_tokens = max(1, len(vlm_input.prompt.split()))
        if completion_tokens <= 0:
            completion_tokens = max(1, len(content.split()))
        return {
            "id": f"chatcmpl-mlx-vlm-{uuid.uuid4().hex[:16]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": runtime.served_model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": getattr(result, "finish_reason", None) or "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        default=DEFAULT_MODEL_PATH,
        help="Local MLX/HF model directory or Hugging Face repo id.",
    )
    parser.add_argument("--served-model-name", default=DEFAULT_SERVED_MODEL_NAME)
    parser.add_argument("--worker-label", default="MLX-VLM OpenAI-compatible worker")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--max-kv-size", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runtime = MLXVLMRuntime(
        model_path=args.model_path,
        served_model_name=args.served_model_name,
        max_kv_size=args.max_kv_size,
        worker_label=args.worker_label,
    )
    app = create_app(runtime)
    import uvicorn  # noqa: PLC0415

    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
