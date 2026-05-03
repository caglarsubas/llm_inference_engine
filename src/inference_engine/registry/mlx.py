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

from pathlib import Path

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
            return []

        descriptors: list[ModelDescriptor] = []
        for path in self._scan(self.root):
            name = path.name
            # Strip HuggingFace cache prefix `models--<owner>--<repo>` → just `<repo>`.
            if name.startswith("models--") and "--" in name[len("models--") :]:
                name = name.split("--")[-1]
            qualified = f"{name}:mlx"
            desc = ModelDescriptor(
                name=name,
                tag="mlx",
                namespace="mlx-community",
                registry="huggingface.co",
                model_path=path,
                format="mlx",
                size_bytes=_dir_size_bytes(path),
            )
            descriptors.append(desc)
            self._cache[qualified] = desc

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
