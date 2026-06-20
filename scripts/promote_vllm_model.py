"""Promote one demanded vLLM catalog entry into the ignored live manifest.

The repo keeps `.vllm_models.json` ignored because it is deployment-specific.
This helper copies one exact entry from `.vllm_models.demanded.example.json`
into that live file, optionally overriding the upstream endpoint/model id.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen


def _read_json_array(path: Path, *, missing_ok: bool = False) -> list[dict[str, Any]]:
    if missing_ok and not path.is_file():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"{path} must contain a JSON array")
    for i, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise ValueError(f"{path} entry {i} must be an object")
    return raw


def _qualified_id(entry: dict[str, Any]) -> str:
    return f"{entry['name']}:{entry.get('tag', 'vllm')}"


def _find_entry(entries: list[dict[str, Any]], model: str) -> dict[str, Any]:
    for entry in entries:
        if _qualified_id(entry) == model or entry.get("name") == model:
            return dict(entry)
    raise ValueError(f"{model!r} not found in demanded manifest")


def _verify_upstream(endpoint: str, model_id: str, *, timeout_seconds: float) -> None:
    url = f"{endpoint.rstrip('/')}/v1/models"
    try:
        with urlopen(url, timeout=timeout_seconds) as response:  # noqa: S310
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, URLError, ValueError) as exc:
        raise ValueError(f"GET {url} failed: {exc}") from exc

    data = payload.get("data") if isinstance(payload, dict) else None
    ids = [str(item.get("id")) for item in data or [] if isinstance(item, dict)]
    if model_id not in ids:
        listed = ", ".join(ids[:8]) if ids else "none"
        raise ValueError(f"upstream did not list {model_id!r}; listed: {listed}")


def promote_entry(
    *,
    model: str,
    live_path: Path,
    demanded_path: Path,
    endpoint: str | None,
    model_id: str | None,
    strict_status: str | None,
    strict_detail: str | None,
    strict_checked_at: str | None,
    require_upstream: bool,
    upstream_timeout_seconds: float,
) -> str:
    demanded = _read_json_array(demanded_path)
    live = _read_json_array(live_path, missing_ok=True)
    entry = _find_entry(demanded, model)

    if endpoint is not None:
        entry["endpoint"] = endpoint.rstrip("/")
    if model_id is not None:
        entry["model_id"] = model_id
    if strict_status is not None:
        entry["strict_image_json_status"] = strict_status
    if strict_detail is not None:
        entry["strict_image_json_detail"] = strict_detail
    if strict_checked_at is not None:
        entry["strict_image_json_checked_at"] = strict_checked_at
    entry.setdefault("benchmark_only", True)

    if require_upstream:
        _verify_upstream(
            str(entry["endpoint"]),
            str(entry["model_id"]),
            timeout_seconds=upstream_timeout_seconds,
        )

    qualified = _qualified_id(entry)
    replaced = False
    for i, existing in enumerate(live):
        if _qualified_id(existing) == qualified:
            live[i] = entry
            replaced = True
            break
    if not replaced:
        live.append(entry)

    live_path.write_text(json.dumps(live, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return qualified


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="Engine id such as fakeshield-22b:vllm")
    parser.add_argument("--live-file", type=Path, default=Path(".vllm_models.json"))
    parser.add_argument(
        "--demanded-file",
        type=Path,
        default=Path(".vllm_models.demanded.example.json"),
    )
    parser.add_argument("--endpoint", help="Override the OpenAI-compatible upstream base URL")
    parser.add_argument("--model-id", help="Override the upstream /v1/models id")
    parser.add_argument("--strict-image-json-status", default="pending_smoke")
    parser.add_argument(
        "--strict-image-json-detail",
        default=(
            "Promoted from demanded vLLM catalog. Keep supports_strict_image_json=false "
            "until repeated FraudGuard vehicle-image JSON smoke passes."
        ),
    )
    parser.add_argument(
        "--strict-image-json-checked-at",
        default=datetime.now(UTC).date().isoformat(),
    )
    parser.add_argument(
        "--require-upstream",
        action="store_true",
        help="Fail unless endpoint/v1/models already lists the selected upstream model_id.",
    )
    parser.add_argument("--upstream-timeout-seconds", type=float, default=5.0)
    args = parser.parse_args()

    try:
        qualified = promote_entry(
            model=args.model,
            live_path=args.live_file,
            demanded_path=args.demanded_file,
            endpoint=args.endpoint,
            model_id=args.model_id,
            strict_status=args.strict_image_json_status,
            strict_detail=args.strict_image_json_detail,
            strict_checked_at=args.strict_image_json_checked_at,
            require_upstream=args.require_upstream,
            upstream_timeout_seconds=args.upstream_timeout_seconds,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"promoted {qualified} into {args.live_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
