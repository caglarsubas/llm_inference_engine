"""Discover vLLM-served models from a JSON config file.

Unlike the GGUF and MLX registries (which scan the filesystem), vLLM models
live behind an HTTP endpoint on a separate process. The mapping from "engine
model id" to "vLLM endpoint + model name" is a deployment concern, so the
registry just reads it from a config file written by the operator.

File format (a JSON array, one entry per vLLM-served model):

    [
      {
        "name": "llama-3.2-1b-instruct",
        "tag": "vllm",
        "endpoint": "http://vllm:8000",
        "model_id": "meta-llama/Llama-3.2-1B-Instruct",
        "chat_template_kwargs": {"enable_thinking": false},
        "size_bytes": 2400000000
      }
    ]

* ``name`` + ``tag`` form the qualified name clients use as ``model`` in
  ``/v1/chat/completions`` (e.g. ``llama-3.2-1b-instruct:vllm``). Tag is
  conventionally ``vllm`` so the format is visible from the id, but any
  string is accepted.
* ``endpoint`` is the base URL of the vLLM server (e.g.
  ``http://vllm:8000`` inside a Compose network).
* ``model_id`` is the name vLLM was started with (passed as ``--model``).
  Required; vLLM rejects requests whose ``model`` field doesn't match.
* ``size_bytes`` is informational — surfaced on ``/v1/models`` but not
  enforced by the manager's memory budget (vLLM does its own GPU memory
  management on a remote host).
* ``chat_template_kwargs`` is optional. When present, the adapter forwards it
  to OpenAI-compatible upstreams that need model-specific chat-template
  switches (for example Docker Model Runner llama.cpp models that need
  ``enable_thinking=false``).

Missing config file → empty registry, no vLLM models served. Malformed file
fails startup loudly via ``load_models()``. `/v1/models` and chat resolution
add one more honesty gate: the configured upstream must respond to `/v1/models`
with this exact ``model_id`` before the descriptor is treated as loadable.
"""

from __future__ import annotations

import json
from pathlib import Path

from .ollama import ModelDescriptor


class VLLMRegistry:
    def __init__(self, config_path: Path | str) -> None:
        self.config_path = Path(config_path)
        self._cache: dict[str, ModelDescriptor] = {}

    def list_models(self) -> list[ModelDescriptor]:
        if not self.config_path.is_file():
            return []

        raw = json.loads(self.config_path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise ValueError(
                f"vLLM config file must be a JSON array, got {type(raw).__name__}"
            )

        descriptors: list[ModelDescriptor] = []
        for i, entry in enumerate(raw):
            if not isinstance(entry, dict):
                raise ValueError(f"vLLM entry {i} is not an object: {entry!r}")
            try:
                name = str(entry["name"])
                tag = str(entry.get("tag", "vllm"))
                endpoint = str(entry["endpoint"]).rstrip("/")
                model_id = str(entry["model_id"])
            except KeyError as exc:
                raise ValueError(f"vLLM entry {i} missing required field: {exc.args[0]}") from exc

            params = {"model_id": model_id, "provider": "vllm", "supports_json_mode": True}
            chat_template_kwargs = entry.get("chat_template_kwargs")
            if chat_template_kwargs is not None:
                if not isinstance(chat_template_kwargs, dict):
                    raise ValueError(
                        f"vLLM entry {i} chat_template_kwargs must be an object"
                )
                params["chat_template_kwargs"] = chat_template_kwargs
            for field in (
                "family",
                "profile",
                "modality",
                "context_length",
                "max_image_size",
                "max_image_side_px",
                "max_image_pixels",
                "supports_json_mode",
            ):
                if field in entry:
                    params[field] = entry[field]

            desc = ModelDescriptor(
                name=name,
                tag=tag,
                namespace="vllm",
                registry="local",
                # Synthetic path so the rest of the codebase (which reads
                # ``model_path`` for display) doesn't trip on None.
                model_path=Path(f"vllm://{endpoint}/{model_id}"),
                format="vllm",
                params=params,
                size_bytes=int(entry.get("size_bytes", 0)),
                endpoint=endpoint,
            )
            descriptors.append(desc)
            self._cache[desc.qualified_name] = desc
            self._cache[desc.fully_qualified_name] = desc

        return sorted(descriptors, key=lambda d: d.qualified_name)

    def get(self, name_with_tag: str) -> ModelDescriptor | None:
        if not self._cache:
            self.list_models()
        if name_with_tag in self._cache:
            return self._cache[name_with_tag]
        if ":" not in name_with_tag:
            for desc in self._cache.values():
                if desc.name == name_with_tag:
                    return desc
        return None
