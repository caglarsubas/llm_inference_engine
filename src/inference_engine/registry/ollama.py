"""Read Ollama's on-disk model store directly — no Ollama daemon required.

Ollama stores models in a Docker-distribution-v2 layout:

    <root>/manifests/<registry>/<namespace>/<model>/<tag>   ← JSON manifest
    <root>/blobs/sha256-<digest>                            ← layer blobs

Each manifest's `layers` array contains entries with these media types:

    application/vnd.ollama.image.model      → the GGUF weights
    application/vnd.ollama.image.template   → chat template (Go text/template)
    application/vnd.ollama.image.params     → JSON inference params
    application/vnd.ollama.image.system     → system prompt text
    application/vnd.ollama.image.license    → license text

This module turns that on-disk layout into a list of `ModelDescriptor` records
that downstream adapters can load.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

OLLAMA_MEDIA_PREFIX = "application/vnd.ollama.image."

ModelFormat = Literal["gguf", "mlx", "vllm"]


@dataclass(frozen=True)
class ModelDescriptor:
    """A single resolved model — backend-agnostic.

    `model_path` points to whatever local artifact the format expects:
      * gguf → a single .gguf file (or content-addressed Ollama blob)
      * mlx  → a directory containing config.json + .safetensors + tokenizer files
      * vllm → a synthetic placeholder (e.g. ``Path("vllm://<endpoint>")``)
               since the real model lives behind an HTTP endpoint, not on disk.

    For HTTP-served backends (vllm), ``endpoint`` carries the base URL of the
    upstream server and ``params['model_id']`` carries the model name vLLM was
    configured with (the value passed as ``--model``).
    """

    name: str
    tag: str
    namespace: str
    registry: str
    model_path: Path
    format: ModelFormat = "gguf"
    template: str | None = None
    system: str | None = None
    params: dict = field(default_factory=dict)
    size_bytes: int = 0
    # HTTP backends only: base URL of the upstream server. None for local
    # backends (gguf, mlx) where ``model_path`` is the locator.
    endpoint: str | None = None

    @property
    def qualified_name(self) -> str:
        return f"{self.name}:{self.tag}"

    @property
    def fully_qualified_name(self) -> str:
        return f"{self.registry}/{self.namespace}/{self.name}:{self.tag}"


class OllamaRegistry:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.manifests_dir = self.root / "manifests"
        self.blobs_dir = self.root / "blobs"
        if not self.manifests_dir.is_dir():
            raise FileNotFoundError(f"manifests dir not found: {self.manifests_dir}")
        if not self.blobs_dir.is_dir():
            raise FileNotFoundError(f"blobs dir not found: {self.blobs_dir}")

        self._cache: dict[str, ModelDescriptor] = {}

    def _blob_path(self, digest: str) -> Path:
        # digest looks like "sha256:abcdef..."; on disk it's "sha256-abcdef..."
        normalized = digest.replace(":", "-", 1)
        return self.blobs_dir / normalized

    def _read_blob_text(self, digest: str) -> str | None:
        path = self._blob_path(digest)
        if not path.exists():
            return None
        try:
            return path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None

    def _read_blob_json(self, digest: str) -> dict | None:
        text = self._read_blob_text(digest)
        if not text:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None

    def _parse_manifest(self, manifest_path: Path) -> ModelDescriptor | None:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

        # Path: <root>/manifests/<registry>/<namespace>/<model>/<tag>
        try:
            tag = manifest_path.name
            model = manifest_path.parent.name
            namespace = manifest_path.parent.parent.name
            registry = manifest_path.parent.parent.parent.name
        except IndexError:
            return None

        gguf_path: Path | None = None
        size_bytes = 0
        template: str | None = None
        system: str | None = None
        params: dict = {}

        for layer in manifest.get("layers") or []:
            media = layer.get("mediaType", "")
            digest = layer.get("digest", "")
            if not media.startswith(OLLAMA_MEDIA_PREFIX) or not digest:
                continue

            kind = media[len(OLLAMA_MEDIA_PREFIX) :]
            if kind == "model":
                gguf_path = self._blob_path(digest)
                size_bytes = int(layer.get("size", 0))
            elif kind == "template":
                template = self._read_blob_text(digest)
            elif kind == "system":
                system = self._read_blob_text(digest)
            elif kind == "params":
                params = self._read_blob_json(digest) or {}

        if gguf_path is None or not gguf_path.exists():
            return None

        return ModelDescriptor(
            name=model,
            tag=tag,
            namespace=namespace,
            registry=registry,
            model_path=gguf_path,
            format="gguf",
            template=template,
            system=system,
            params=params,
            size_bytes=size_bytes,
        )

    def list_models(self) -> list[ModelDescriptor]:
        descriptors: list[ModelDescriptor] = []
        for manifest_path in self.manifests_dir.rglob("*"):
            if not manifest_path.is_file() or manifest_path.name.startswith("."):
                continue
            desc = self._parse_manifest(manifest_path)
            if desc is None:
                continue
            descriptors.append(desc)
            self._cache[desc.qualified_name] = desc
            self._cache[desc.fully_qualified_name] = desc
        return sorted(descriptors, key=lambda d: d.qualified_name)

    def get(self, name_with_tag: str) -> ModelDescriptor | None:
        if not self._cache:
            self.list_models()
        # accept "model:tag" or "model" (resolves to first matching tag)
        if name_with_tag in self._cache:
            return self._cache[name_with_tag]
        if ":" not in name_with_tag:
            for desc in self._cache.values():
                if desc.name == name_with_tag:
                    return desc
        return None
