"""Download local FraudGuard VLM candidates from Hugging Face.

The engine only serves a vLLM/DMR model after a real upstream is configured and
passes its own `/v1/models` probe. This script is the earlier acquisition step:
materialize model snapshots under a local cache so operators can stand up and
smoke-test those upstreams without using OpenRouter.

Usage:
    uv run python scripts/download_vlm_models.py
    uv run python scripts/download_vlm_models.py --core-only
    HF_HUB_ENABLE_HF_TRANSFER=1 uv run --with hf-transfer python scripts/download_vlm_models.py --quiet
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download
from huggingface_hub.utils import disable_progress_bars


DEFAULT_ROOT = Path(
    os.environ.get(
        "HF_VLM_MODELS_DIR",
        str(Path.home() / ".cache" / "inference_engine" / "hf-vlm"),
    )
).expanduser()

FRAUDGUARD_LOCAL_MODELS: list[dict[str, Any]] = [
    {
        "repo_id": "Qwen/Qwen3-VL-8B-Instruct",
        "engine_id": "qwen3-vl-8b-instruct:vllm",
        "tier": "core",
        "parameter_count_b": 8,
        "size_gib": 16.3,
    },
    {
        "repo_id": "Qwen/Qwen3-VL-32B-Instruct",
        "engine_id": "qwen3-vl-32b-instruct:vllm",
        "tier": "core",
        "parameter_count_b": 32,
        "size_gib": 62.1,
    },
    {
        "repo_id": "Qwen/Qwen3-VL-30B-A3B-Instruct",
        "engine_id": "qwen3-vl-30b-a3b-instruct:vllm",
        "tier": "core",
        "parameter_count_b": 30,
        "size_gib": 57.9,
    },
    {
        "repo_id": "zai-org/GLM-4.6V-Flash",
        "engine_id": "glm-4.6v-flash:vllm",
        "tier": "core",
        "parameter_count_b": 9,
        "size_gib": 19.2,
    },
    {
        "repo_id": "zai-org/GLM-4.1V-9B-Thinking",
        "engine_id": "glm-4.1v-9b-thinking:vllm",
        "tier": "core",
        "parameter_count_b": 9,
        "size_gib": 19.2,
    },
    {
        "repo_id": "openbmb/MiniCPM-V-4_5",
        "engine_id": "minicpm-v-4.5:vllm",
        "tier": "core",
        "parameter_count_b": None,
        "size_gib": 16.2,
    },
    {
        "repo_id": "openbmb/MiniCPM-V-4.6",
        "engine_id": "minicpm-v-4.6:vllm",
        "tier": "core",
        "parameter_count_b": None,
        "size_gib": 2.4,
    },
    {
        "repo_id": "OpenGVLab/InternVL3_5-8B",
        "engine_id": "internvl3.5-8b:vllm",
        "tier": "core",
        "parameter_count_b": 8,
        "size_gib": 15.9,
    },
    {
        "repo_id": "OpenGVLab/InternVL3_5-14B",
        "engine_id": "internvl3.5-14b:vllm",
        "tier": "core",
        "parameter_count_b": 14,
        "size_gib": 28.2,
    },
    {
        "repo_id": "OpenGVLab/InternVL3_5-20B-A4B-Preview",
        "engine_id": "internvl3.5-20b-a4b-preview:vllm",
        "tier": "core",
        "parameter_count_b": 20,
        "size_gib": 39.6,
    },
    {
        "repo_id": "moonshotai/Kimi-VL-A3B-Thinking",
        "engine_id": "kimi-vl-a3b-thinking:vllm",
        "tier": "core",
        "size_gib": 30.6,
    },
    {
        "repo_id": "moonshotai/Kimi-VL-A3B-Instruct",
        "engine_id": "kimi-vl-a3b-instruct:vllm",
        "tier": "core",
        "size_gib": 30.6,
    },
    {
        "repo_id": "AIDC-AI/Ovis2.5-9B",
        "engine_id": "ovis2.5-9b:vllm",
        "tier": "core",
        "parameter_count_b": 9,
        "size_gib": 17.1,
    },
    {
        "repo_id": "deepseek-ai/deepseek-vl2-tiny",
        "engine_id": "deepseek-vl2-tiny:vllm",
        "tier": "core",
        "parameter_count_b": None,
        "size_gib": 6.3,
    },
    {
        "repo_id": "deepseek-ai/deepseek-vl2-small",
        "engine_id": "deepseek-vl2-small:vllm",
        "tier": "core",
        "parameter_count_b": None,
        "size_gib": 30.1,
    },
    {
        "repo_id": "allenai/Molmo-7B-D-0924",
        "engine_id": "molmo-7b-d:vllm",
        "tier": "core",
        "parameter_count_b": 7,
        "size_gib": 29.9,
    },
    {
        "repo_id": "allenai/Molmo2-4B",
        "engine_id": "molmo2-4b:vllm",
        "tier": "core",
        "parameter_count_b": 4,
        "size_gib": 18.1,
    },
    {
        "repo_id": "allenai/Molmo2-8B",
        "engine_id": "molmo2-8b:vllm",
        "tier": "core",
        "parameter_count_b": 8,
        "size_gib": 32.3,
    },
    {
        "repo_id": "allenai/Molmo2-O-7B",
        "engine_id": "molmo2-o-7b:vllm",
        "tier": "core",
        "parameter_count_b": 7,
        "size_gib": 28.9,
    },
    {
        "repo_id": "zhipeixu/fakeshield-v1-22b",
        "engine_id": "fakeshield-22b:vllm",
        "tier": "specialized",
        "parameter_count_b": 22,
        "size_gib": 41.1,
    },
    {
        "repo_id": "saberzl/SIDA-7B",
        "engine_id": "sida-7b:vllm",
        "tier": "specialized",
        "parameter_count_b": 7,
        "size_gib": 15.0,
    },
    {
        "repo_id": "saberzl/SIDA-13B",
        "engine_id": "sida-13b:vllm",
        "tier": "specialized",
        "parameter_count_b": 13,
        "size_gib": 26.8,
    },
]

IGNORE_PATTERNS = [
    ".git*",
    "*.h5",
    "*.msgpack",
    "*.onnx",
    "optimizer.pt",
    "scheduler.pt",
    "training_args.bin",
]


def _local_dir(root: Path, repo_id: str) -> Path:
    return root / repo_id.replace("/", "--")


def _dir_size_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def _select_models(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.repo:
        requested = set(args.repo)
        selected = [m for m in FRAUDGUARD_LOCAL_MODELS if m["repo_id"] in requested]
        unknown = sorted(requested - {m["repo_id"] for m in selected})
        if unknown:
            selected.extend(
                {
                    "repo_id": repo,
                    "engine_id": "",
                    "tier": "custom",
                    "parameter_count_b": None,
                    "size_gib": None,
                }
                for repo in unknown
            )
        return selected

    if args.core_only:
        return [m for m in FRAUDGUARD_LOCAL_MODELS if m["tier"] == "core"]
    return FRAUDGUARD_LOCAL_MODELS


def _write_manifest(root: Path, records: list[dict[str, Any]]) -> None:
    manifest_path = root / "local_vlm_manifest.json"
    manifest_path.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")


def _append_status(root: Path, record: dict[str, Any]) -> None:
    status_path = root / "download_status.jsonl"
    with status_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help=f"Snapshot target root. Default: {DEFAULT_ROOT}",
    )
    parser.add_argument(
        "--repo",
        action="append",
        help="Specific Hugging Face repo id to download. Repeat for multiple repos.",
    )
    parser.add_argument(
        "--core-only",
        action="store_true",
        help="Skip specialized FakeShield/SIDA follow-ups.",
    )
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable Hugging Face progress bars for log files and detached runs.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.quiet:
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        disable_progress_bars()

    root = args.root.expanduser()
    root.mkdir(parents=True, exist_ok=True)

    selected = _select_models(args)
    print(f"Selected {len(selected)} model snapshot(s); target root={root}", flush=True)
    manifest_records: list[dict[str, Any]] = []
    for model in selected:
        repo_id = model["repo_id"]
        target = _local_dir(root, repo_id)
        record = {
            **model,
            "local_dir": str(target),
            "downloaded_at": datetime.now(UTC).isoformat(),
        }
        print(f"BEGIN {repo_id} -> {target}", flush=True)
        if args.dry_run:
            record["status"] = "dry_run"
            print(json.dumps(record, sort_keys=True), flush=True)
            manifest_records.append(record)
            continue

        started = time.time()
        _append_status(root, {**record, "status": "started"})
        try:
            path = Path(
                snapshot_download(
                    repo_id=repo_id,
                    local_dir=str(target),
                    ignore_patterns=IGNORE_PATTERNS,
                    max_workers=args.max_workers,
                )
            )
            record.update(
                {
                    "status": "downloaded",
                    "resolved_dir": str(path),
                    "actual_size_gib": round(_dir_size_bytes(path) / (1024**3), 2),
                    "elapsed_s": round(time.time() - started, 1),
                }
            )
        except Exception as exc:  # noqa: BLE001 - operator-facing batch runner
            record.update(
                {
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
        print(json.dumps(record, sort_keys=True), flush=True)
        _append_status(root, record)
        manifest_records.append(record)

    _write_manifest(root, manifest_records)
    return 0 if all(r["status"] in {"downloaded", "dry_run"} for r in manifest_records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
