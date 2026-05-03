"""Download an MLX-converted model from HuggingFace into MLX_MODELS_DIR.

Usage:
    uv run python scripts/download_mlx_model.py [REPO]

REPO defaults to ``mlx-community/Llama-3.2-1B-Instruct-4bit`` (~700 MB) which
is small enough for a quick smoke test on Apple Silicon. Override with any
other ``mlx-community/*`` repo to grab a larger model.
"""

from __future__ import annotations

import sys

from huggingface_hub import snapshot_download

from inference_engine.config import settings


DEFAULT_REPO = "mlx-community/Llama-3.2-1B-Instruct-4bit"


def main() -> int:
    repo = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_REPO
    target_root = settings.mlx_models_dir
    target_root.mkdir(parents=True, exist_ok=True)

    # snapshot_download materializes the repo into a single dir, which our
    # MLXRegistry happily treats as one model.
    local_dir = target_root / repo.split("/")[-1]
    print(f"downloading {repo} → {local_dir}")
    path = snapshot_download(
        repo_id=repo,
        local_dir=str(local_dir),
        # Skip artifacts that aren't needed at inference time.
        ignore_patterns=["*.gguf", "*.bin", "*.pt", "*.onnx", "*.md"],
    )
    print(f"done: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
