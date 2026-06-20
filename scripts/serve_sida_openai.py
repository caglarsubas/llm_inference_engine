"""Serve SIDA-13B through a minimal OpenAI-compatible HTTP surface.

This is an optional CUDA-worker bridge for the SIDA reference implementation.
Run it in a separate Python environment that has the upstream SIDA requirements
installed, then point this engine's ``sida-13b:vllm`` descriptor at it.

The worker intentionally loads SIDA during FastAPI startup. If the custom SIDA
stack, CUDA, or weights are unavailable, the server fails fast and the engine's
vLLM/OpenAI-compatible probe will not move the model into ``data[]``.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


DEFAULT_MODEL_DIR = (
    Path.home() / ".cache" / "inference_engine" / "hf-vlm" / "saberzl--SIDA-13B"
)
DEFAULT_PROMPT = (
    "Please answer begin with [CLS] for classification, if the image is tampered, "
    "output mask the tampered region."
)


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[dict[str, Any]]
    max_tokens: int = Field(default=512, ge=1)
    temperature: float | None = None
    stream: bool = False


@dataclass(frozen=True)
class SidaInput:
    prompt: str
    image_url: str


def _extract_sida_input(messages: list[dict[str, Any]]) -> SidaInput:
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
        raise HTTPException(status_code=400, detail="SIDA requires one image_url content part")
    prompt = "\n".join(p for p in prompt_parts if p).strip() or DEFAULT_PROMPT
    return SidaInput(prompt=prompt, image_url=image_url)


def _materialize_image(image_url: str) -> Path:
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
            prefix="sida-openai-",
            suffix=suffix,
            delete=False,
        )
        with handle:
            handle.write(raw)
        return Path(handle.name)

    if image_url.startswith("file://"):
        return Path(image_url[7:])
    path = Path(image_url)
    if path.exists():
        return path
    raise HTTPException(
        status_code=400,
        detail="SIDA worker supports data: image URLs or local file paths",
    )


class SidaRuntime:
    def __init__(
        self,
        *,
        model_dir: Path,
        sida_src_dir: Path,
        served_model_name: str,
        precision: str = "bf16",
        image_size: int = 1024,
        model_max_length: int = 512,
        conv_type: str = "llava_v1",
    ) -> None:
        self.model_dir = model_dir
        self.sida_src_dir = sida_src_dir
        self.served_model_name = served_model_name
        self.precision = precision
        self.image_size = image_size
        self.model_max_length = model_max_length
        self.conv_type = conv_type

        self._loaded = False
        self._torch: Any = None
        self._cv2: Any = None
        self._F: Any = None
        self._tokenizer: Any = None
        self._model: Any = None
        self._clip_image_processor: Any = None
        self._transform: Any = None
        self._conversation_lib: Any = None
        self._tokenizer_image_token: Any = None
        self._image_token_index: int | None = None
        self._default_image_token: str = ""
        self._default_im_start_token: str = ""
        self._default_im_end_token: str = ""

    @property
    def loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        if self._loaded:
            return
        if not self.model_dir.is_dir():
            raise RuntimeError(f"SIDA model directory not found: {self.model_dir}")
        if not self.sida_src_dir.is_dir():
            raise RuntimeError(f"SIDA source directory not found: {self.sida_src_dir}")
        sys.path.insert(0, str(self.sida_src_dir))

        import cv2  # noqa: PLC0415
        import torch  # noqa: PLC0415
        import torch.nn.functional as F  # noqa: PLC0415
        from model.SIDA import SIDAForCausalLM  # noqa: PLC0415
        from model.llava import conversation as conversation_lib  # noqa: PLC0415
        from model.llava.mm_utils import tokenizer_image_token  # noqa: PLC0415
        from model.segment_anything.utils.transforms import ResizeLongestSide  # noqa: PLC0415
        from transformers import AutoTokenizer, BitsAndBytesConfig, CLIPImageProcessor  # noqa: PLC0415
        from utils.utils import (  # noqa: PLC0415
            DEFAULT_IMAGE_TOKEN,
            DEFAULT_IM_END_TOKEN,
            DEFAULT_IM_START_TOKEN,
            IMAGE_TOKEN_INDEX,
        )

        if not torch.cuda.is_available():
            raise RuntimeError("SIDA reference inference requires CUDA")

        tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_dir),
            cache_dir=None,
            model_max_length=self.model_max_length,
            padding_side="right",
            use_fast=False,
        )
        tokenizer.pad_token = tokenizer.unk_token
        seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
        cls_token_idx = tokenizer("[CLS]", add_special_tokens=False).input_ids[0]

        torch_dtype = torch.float32
        if self.precision == "bf16":
            torch_dtype = torch.bfloat16
        elif self.precision == "fp16":
            torch_dtype = torch.half
        kwargs: dict[str, Any] = {"torch_dtype": torch_dtype}
        if self.precision == "int4":
            kwargs.update(
                {
                    "torch_dtype": torch.half,
                    "load_in_4bit": True,
                    "quantization_config": BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                        llm_int8_skip_modules=["visual_model"],
                    ),
                }
            )
        elif self.precision == "int8":
            kwargs.update(
                {
                    "torch_dtype": torch.half,
                    "quantization_config": BitsAndBytesConfig(
                        llm_int8_skip_modules=["visual_model"],
                        load_in_8bit=True,
                    ),
                }
            )

        model = SIDAForCausalLM.from_pretrained(
            str(self.model_dir),
            low_cpu_mem_usage=True,
            vision_tower="openai/clip-vit-large-patch14",
            seg_token_idx=seg_token_idx,
            cls_token_idx=cls_token_idx,
            **kwargs,
        )
        model.config.eos_token_id = tokenizer.eos_token_id
        model.config.bos_token_id = tokenizer.bos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
        model = model.cuda()
        model.get_model().initialize_vision_modules(model.get_model().config)
        vision_tower = model.get_model().get_vision_tower()
        vision_tower.to(dtype=torch_dtype)
        if self.precision == "bf16":
            model = model.bfloat16().cuda()
        elif self.precision in {"fp16", "int4", "int8"}:
            model = model.half().cuda()
        else:
            model = model.float().cuda()
        model.eval()

        self._torch = torch
        self._cv2 = cv2
        self._F = F
        self._tokenizer = tokenizer
        self._model = model
        self._clip_image_processor = CLIPImageProcessor.from_pretrained(
            model.config.vision_tower
        )
        self._transform = ResizeLongestSide(self.image_size)
        self._conversation_lib = conversation_lib
        self._tokenizer_image_token = tokenizer_image_token
        self._image_token_index = IMAGE_TOKEN_INDEX
        self._default_image_token = DEFAULT_IMAGE_TOKEN
        self._default_im_start_token = DEFAULT_IM_START_TOKEN
        self._default_im_end_token = DEFAULT_IM_END_TOKEN
        self._loaded = True

    def generate(self, prompt: str, image_path: Path, *, max_tokens: int) -> str:
        if not self._loaded:
            raise RuntimeError("SIDA runtime is not loaded")
        assert self._torch is not None
        assert self._cv2 is not None
        assert self._F is not None
        assert self._tokenizer is not None
        assert self._model is not None
        assert self._clip_image_processor is not None
        assert self._transform is not None
        assert self._conversation_lib is not None
        assert self._tokenizer_image_token is not None
        assert self._image_token_index is not None

        torch = self._torch
        cv2 = self._cv2
        tokenizer = self._tokenizer

        conv = self._conversation_lib.conv_templates[self.conv_type].copy()
        conv.messages = []
        sida_prompt = f"{self._default_image_token}\n{prompt}"
        replace_token = (
            self._default_im_start_token
            + self._default_image_token
            + self._default_im_end_token
        )
        sida_prompt = sida_prompt.replace(self._default_image_token, replace_token)
        conv.append_message(conv.roles[0], sida_prompt)
        conv.append_message(conv.roles[1], "")
        rendered_prompt = conv.get_prompt()

        image_np = cv2.imread(str(image_path))
        if image_np is None:
            raise ValueError(f"failed to read image: {image_path}")
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        original_size_list = [image_np.shape[:2]]

        image_clip = (
            self._clip_image_processor.preprocess(image_np, return_tensors="pt")[
                "pixel_values"
            ][0]
            .unsqueeze(0)
            .cuda()
        )
        image_clip = self._cast_float_tensor(image_clip)

        image = self._transform.apply_image(image_np)
        resize_list = [image.shape[:2]]
        image = (
            self._preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())
            .unsqueeze(0)
            .cuda()
        )
        image = self._cast_float_tensor(image)
        input_ids = self._tokenizer_image_token(rendered_prompt, tokenizer, return_tensors="pt")
        input_ids = input_ids.unsqueeze(0).cuda()

        output_ids, _pred_masks = self._model.evaluate(
            image_clip,
            image,
            input_ids,
            resize_list,
            original_size_list,
            max_new_tokens=max_tokens,
            tokenizer=tokenizer,
        )
        output_ids = output_ids[0][output_ids[0] != self._image_token_index]
        text_output = tokenizer.decode(output_ids, skip_special_tokens=False)
        return text_output.replace("\n", "").replace("  ", " ").strip()

    def _cast_float_tensor(self, tensor: Any) -> Any:
        if self.precision == "bf16":
            return tensor.bfloat16()
        if self.precision in {"fp16", "int4", "int8"}:
            return tensor.half()
        return tensor.float()

    def _preprocess(self, x: Any) -> Any:
        torch = self._torch
        assert torch is not None
        pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
        pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
        x = (x - pixel_mean) / pixel_std
        h, w = x.shape[-2:]
        return self._F.pad(x, (0, self.image_size - w, 0, self.image_size - h))


def create_app(runtime: SidaRuntime) -> FastAPI:
    @contextlib.asynccontextmanager
    async def _lifespan(_app: FastAPI) -> Any:
        runtime.load()
        yield

    app = FastAPI(title="SIDA OpenAI-compatible worker", lifespan=_lifespan)

    @app.get("/v1/models")
    def _models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": runtime.served_model_name,
                    "object": "model",
                    "created": 0,
                    "owned_by": "sida",
                }
            ],
        }

    @app.post("/v1/chat/completions")
    def _chat(req: ChatCompletionRequest) -> dict[str, Any]:
        if req.stream:
            raise HTTPException(status_code=400, detail="SIDA worker does not support stream=true")
        if req.model != runtime.served_model_name:
            raise HTTPException(status_code=404, detail=f"model not found: {req.model!r}")
        sida_input = _extract_sida_input(req.messages)
        image_path = _materialize_image(sida_input.image_url)
        remove_temp = image_path.name.startswith("sida-openai-")
        try:
            content = runtime.generate(sida_input.prompt, image_path, max_tokens=req.max_tokens)
        except Exception as exc:  # noqa: BLE001 - mapped to OpenAI-style upstream 500
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        finally:
            if remove_temp:
                with contextlib.suppress(OSError):
                    image_path.unlink()

        prompt_tokens = max(1, len(sida_input.prompt.split()))
        completion_tokens = max(1, len(content.split()))
        return {
            "id": f"chatcmpl-sida-{uuid.uuid4().hex[:16]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": runtime.served_model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
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
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument(
        "--sida-src-dir",
        type=Path,
        required=True,
        help="Path to a checkout of https://github.com/hzlsaber/SIDA",
    )
    parser.add_argument("--served-model-name", default="saberzl/SIDA-13B")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--precision",
        choices=["fp32", "bf16", "fp16", "int8", "int4"],
        default="bf16",
    )
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--model-max-length", type=int, default=512)
    parser.add_argument("--conv-type", default="llava_v1")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runtime = SidaRuntime(
        model_dir=args.model_dir.expanduser().resolve(),
        sida_src_dir=args.sida_src_dir.expanduser().resolve(),
        served_model_name=args.served_model_name,
        precision=args.precision,
        image_size=args.image_size,
        model_max_length=args.model_max_length,
        conv_type=args.conv_type,
    )
    app = create_app(runtime)
    import uvicorn  # noqa: PLC0415

    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
