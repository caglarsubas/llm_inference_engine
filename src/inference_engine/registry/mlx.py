"""Discover MLX-format models on disk.

MLX-LM expects a HuggingFace-style model directory containing at minimum:
  * ``config.json``
  * one or more ``*.safetensors`` weight files
  * tokenizer files (``tokenizer.json`` and/or ``tokenizer_config.json``)

A directory that contains all of these is treated as an MLX model. Each model
is named after its directory; the qualified name uses ``mlx`` as the tag so it
never collides with Ollama's ``model:tag`` namespace.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from .artifact import REPO_PATTERN, ArtifactIdentity
from .ollama import ModelDescriptor

REQUIRED_FILES = ("config.json",)
WEIGHT_GLOBS = ("*.safetensors", "*.npz")
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json")


def _is_mlx_model_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    if not all((path / req).exists() for req in REQUIRED_FILES):
        return False
    if not any(any(path.glob(pat)) for pat in WEIGHT_GLOBS):
        return False
    if not any((path / tok).exists() for tok in TOKENIZER_FILES):
        return False
    return True


def _dir_size_bytes(path: Path) -> int:
    total = 0
    for entry in path.rglob("*"):
        if entry.is_file():
            try:
                total += entry.stat().st_size
            except OSError:
                pass
    return total


def _mlx_quantization(path: Path) -> str | None:
    try:
        config = json.loads((path / "config.json").read_bytes())
    except (OSError, ValueError):
        return None
    quant = config.get("quantization") if isinstance(config, dict) else None
    if not isinstance(quant, dict):
        return None
    widths = [quant.get("bits")]
    widths.extend(val.get("bits") for val in quant.values() if isinstance(val, dict))
    if any(type(bits) is not int or not 0 < bits <= 32 for bits in widths):
        return None
    distinct = sorted(set(widths))
    if len(distinct) == 1:
        return f"{distinct[0]}-bit"
    mixed = "mixed-" + "-".join(str(bits) for bits in distinct) + "-bit"
    return mixed if len(mixed) <= 64 else "mixed"


def _artifact_identity(snapshot: Path) -> ArtifactIdentity | None:
    quant = _mlx_quantization(snapshot)
    # A service-facing alias can point directly at a pinned Hub snapshot.
    # A flat download directory or a model name containing "4bit" gives no revision.
    repo_dir = snapshot.parent.parent.name
    if snapshot.parent.name == "snapshots" and repo_dir.startswith("models--"):
        repo = repo_dir.removeprefix("models--").replace("--", "/")
        if re.fullmatch(r"[0-9a-f]{40}", snapshot.name) and re.fullmatch(REPO_PATTERN, repo) and len(repo) <= 97:
            return ArtifactIdentity(
                source="hf_snapshot", repo_id=repo, revision=snapshot.name, quantization=quant
            )
    if quant:
        return ArtifactIdentity(source="mlx_config", quantization=quant)
    return None


class MLXRegistry:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self._cache: dict[str, ModelDescriptor] = {}

    def _scan(self, base: Path, max_depth: int = 4) -> list[Path]:
        """Recursive scan up to `max_depth` for MLX model dirs.

        We descend through arbitrary parent directories (e.g. ``mlx-community``,
        ``models--<repo>``) but stop the moment we identify a model directory
        so we never recurse *into* it.
        """
        results: list[Path] = []

        def walk(p: Path, depth: int) -> None:
            if depth > max_depth or not p.is_dir():
                return
            if _is_mlx_model_dir(p):
                results.append(p)
                return
            for child in p.iterdir():
                if child.is_dir() and not child.name.startswith("."):
                    walk(child, depth + 1)

        if base.is_dir():
            walk(base, 0)
        return results

    def list_models(self) -> list[ModelDescriptor]:
        if not self.root.is_dir():
            self._cache = {}
            return []

        descriptors: list[ModelDescriptor] = []
        cache: dict[str, ModelDescriptor] = {}
        for path in self._scan(self.root):
            name = path.name
            # Strip HuggingFace cache prefix `models--<owner>--<repo>` → just `<repo>`.
            if name.startswith("models--") and "--" in name[len("models--") :]:
                name = name.split("--")[-1]
            qualified = f"{name}:mlx"
            resolved = path.resolve()
            identity = _artifact_identity(resolved)
            desc = ModelDescriptor(
                name=name,
                tag="mlx",
                namespace="mlx-community",
                registry="huggingface.co",
                model_path=resolved if identity and identity.source == "hf_snapshot" else path,
                format="mlx",
                size_bytes=_dir_size_bytes(path),
                artifact_identity=identity,
            )
            descriptors.append(desc)
            cache[qualified] = desc

        self._cache = cache
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
