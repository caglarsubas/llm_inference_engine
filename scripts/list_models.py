"""List models discoverable in the configured stores (Ollama GGUF + MLX).

Usage:
    uv run python scripts/list_models.py
"""

from __future__ import annotations

import sys

from inference_engine.config import settings
from inference_engine.registry import CompositeRegistry, MLXRegistry, OllamaRegistry


def human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def main() -> int:
    ollama = OllamaRegistry(settings.ollama_models_dir)
    mlx = MLXRegistry(settings.mlx_models_dir)
    sources = (mlx, ollama) if settings.prefer_mlx_over_gguf else (ollama, mlx)
    registry = CompositeRegistry(sources)

    descriptors = registry.list_models()
    if not descriptors:
        print(
            f"no models found in {settings.ollama_models_dir} or {settings.mlx_models_dir}",
            file=sys.stderr,
        )
        return 1

    width = max(len(d.qualified_name) for d in descriptors)
    print(f"{'model':<{width}}  {'fmt':<5}  {'size':>10}  artifact")
    print("-" * (width + 7 + 14 + 60))
    for d in descriptors:
        artifact = d.model_path.name
        if len(artifact) > 60:
            artifact = artifact[:57] + "..."
        print(
            f"{d.qualified_name:<{width}}  {d.format:<5}  "
            f"{human_size(d.size_bytes):>10}  {artifact}"
        )
    print(
        f"\n{len(descriptors)} models  "
        f"(ollama: {settings.ollama_models_dir}, mlx: {settings.mlx_models_dir})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
