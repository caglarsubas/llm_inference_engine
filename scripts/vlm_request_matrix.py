"""Request-test VLM candidate inference through the engine.

This probe is intentionally broader and lighter than ``vlm_strict_json_smoke``:
it walks the configured candidate catalog, checks each model's catalog state,
then sends a minimal image chat request so operators can see which candidates
are actually callable today.

Usage:
    ENGINE_API_KEY=... uv run python scripts/vlm_request_matrix.py
    uv run python scripts/vlm_request_matrix.py --model minicpm-v-4.5-gguf-q4-k-m:dmr
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import httpx


DEFAULT_BASE_URL = "http://127.0.0.1:8080"
DEFAULT_CANDIDATE_FILE = Path(".vllm_models.demanded.example.json")

# 1x1 PNG. This is a reachability/request-shape probe, not a quality benchmark.
DEFAULT_IMAGE_URL = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
)

VLM_ID_MARKERS = (
    "aya",
    "deepseek-vl",
    "fakeshield",
    "fakevlm",
    "fastvlm",
    "gemma-3",
    "glm",
    "idefics",
    "internlm-xcomposer",
    "internvl",
    "kimi-vl",
    "llama-3.2",
    "minicpm",
    "molmo",
    "moondream",
    "ovis",
    "phi",
    "pixtral",
    "qwen2.5-vl",
    "qwen3-vl",
    "sida",
    "smolvlm",
)


def _headers(api_key: str | None) -> dict[str, str]:
    if not api_key:
        return {}
    return {"Authorization": f"Bearer {api_key}"}


def _qualified_id(entry: dict[str, Any]) -> str | None:
    name = entry.get("name")
    if not name:
        return None
    tag = entry.get("tag", "vllm")
    return f"{name}:{tag}"


def _looks_like_vlm(entry: dict[str, Any]) -> bool:
    if entry.get("supports_images") is True:
        return True
    modality = str(entry.get("modality") or "").lower()
    if "image" in modality:
        return True
    if entry.get("backend") == "mlx" or entry.get("format") == "mlx":
        return False
    candidate_id = str(entry.get("id") or _qualified_id(entry) or "").lower()
    family = str(entry.get("family") or "").lower()
    return any(marker in candidate_id or marker in family for marker in VLM_ID_MARKERS)


def _load_candidate_ids(path: Path) -> list[str]:
    if not path.is_file():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"{path} must contain a JSON array")
    ids: list[str] = []
    for entry in raw:
        if isinstance(entry, dict) and _looks_like_vlm(entry):
            candidate_id = _qualified_id(entry)
            if candidate_id:
                ids.append(candidate_id)
    return ids


def _catalog_lookup(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for section, available in (("data", True), ("unavailable", False)):
        for entry in payload.get(section) or []:
            if not isinstance(entry, dict) or not entry.get("id"):
                continue
            out[str(entry["id"])] = {
                "available": available,
                "section": section,
                "entry": entry,
            }
    return out


def _candidate_ids(
    *,
    explicit_models: list[str],
    candidate_files: list[Path],
    catalog_payload: dict[str, Any],
    include_catalog_candidates: bool,
) -> list[str]:
    seen: set[str] = set()
    ids: list[str] = []

    def add(model_id: str) -> None:
        if model_id not in seen:
            seen.add(model_id)
            ids.append(model_id)

    for model in explicit_models:
        add(model)

    if not explicit_models:
        for path in candidate_files:
            for model in _load_candidate_ids(path):
                add(model)

        if include_catalog_candidates:
            for section in ("data", "unavailable"):
                for entry in catalog_payload.get(section) or []:
                    if isinstance(entry, dict) and entry.get("id") and _looks_like_vlm(entry):
                        add(str(entry["id"]))

    return ids


def _chat_payload(model: str, image_url: str, *, max_tokens: int) -> dict[str, Any]:
    return {
        "model": model,
        "temperature": 0,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "system",
                "content": (
                    "Return one compact JSON object only with keys: "
                    "vehicle_visible, damage_visible, confidence, summary."
                ),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "This is a request-shape probe. Inspect the image and "
                            "return whether a vehicle and visible damage are present."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url, "detail": "low"},
                    },
                ],
            },
        ],
    }


def _catalog_state(model: str, lookup: dict[str, dict[str, Any]]) -> dict[str, Any]:
    hit = lookup.get(model)
    if hit is None:
        return {
            "catalog_state": "missing",
            "available": False,
            "availability_status": "missing_from_catalog",
            "availability_detail": "model id is absent from /v1/models.data",
        }

    entry = hit["entry"]
    status = entry.get("availability_status") or (
        "available" if hit["available"] else entry.get("reason") or "unavailable"
    )
    return {
        "catalog_state": "available" if hit["available"] else "unavailable",
        "available": bool(hit["available"]),
        "availability_status": status,
        "availability_detail": entry.get("availability_detail") or entry.get("detail"),
        "backend": entry.get("backend"),
        "provider": entry.get("provider"),
        "endpoint": entry.get("endpoint"),
        "upstream_model_id": entry.get("upstream_model_id"),
        "supports_images": entry.get("supports_images"),
        "supports_strict_image_json": entry.get("supports_strict_image_json"),
        "download_status": entry.get("download_status"),
        "local_snapshot_path": entry.get("local_snapshot_path"),
    }


def _request_one(
    client: httpx.Client,
    *,
    model: str,
    image_url: str,
    max_tokens: int,
) -> dict[str, Any]:
    payload = _chat_payload(model, image_url, max_tokens=max_tokens)
    started = time.perf_counter()
    try:
        response = client.post("/v1/chat/completions", json=payload)
        latency_ms = (time.perf_counter() - started) * 1000
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        return {
            "request_ok": False,
            "latency_ms": round(latency_ms, 2),
            "error_type": "http_error",
            "http_status": exc.response.status_code,
            "detail": exc.response.text[:1000],
        }
    except httpx.RequestError as exc:
        latency_ms = (time.perf_counter() - started) * 1000
        return {
            "request_ok": False,
            "latency_ms": round(latency_ms, 2),
            "error_type": "request_error",
            "detail": str(exc).splitlines()[0][:1000] if str(exc) else type(exc).__name__,
        }

    try:
        body = response.json()
    except ValueError as exc:
        return {
            "request_ok": False,
            "latency_ms": round(latency_ms, 2),
            "error_type": "bad_response_json",
            "detail": str(exc),
            "content": response.text[:1000],
        }

    choice = (body.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    content = message.get("content")
    return {
        "request_ok": bool(content),
        "latency_ms": round(latency_ms, 2),
        "finish_reason": choice.get("finish_reason"),
        "usage": body.get("usage") or {},
        "content_preview": str(content or "")[:500],
        "reasoning_content_present": bool(message.get("reasoning_content")),
        "error_type": None if content else "empty_content",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=os.environ.get("ENGINE_URL", DEFAULT_BASE_URL))
    parser.add_argument(
        "--api-key",
        default=os.environ.get("ENGINE_API_KEY") or os.environ.get("ENGINE_TOKEN"),
        help="Bearer token for AUTH_ENABLED deployments.",
    )
    parser.add_argument(
        "--candidate-file",
        type=Path,
        action="append",
        default=[DEFAULT_CANDIDATE_FILE],
        help="JSON candidate file to scan. Defaults to .vllm_models.demanded.example.json.",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="Specific model id to probe. Repeat to test multiple models.",
    )
    parser.add_argument(
        "--no-catalog-candidates",
        action="store_true",
        help="Do not add VLM-looking ids discovered in live /v1/models.data.",
    )
    parser.add_argument(
        "--available-only",
        action="store_true",
        help="Only send chat requests for models currently available in /v1/models.data.",
    )
    parser.add_argument(
        "--image-url",
        default=os.environ.get("VLM_PROBE_IMAGE_URL", DEFAULT_IMAGE_URL),
        help="Image URL/data URI to include in every request.",
    )
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON output file for the full matrix result.",
    )
    parser.add_argument(
        "--allow-failures",
        action="store_true",
        help="Always exit 0 after writing the report.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    headers = _headers(args.api_key)
    timeout = httpx.Timeout(args.timeout_seconds)
    with httpx.Client(base_url=args.base_url.rstrip("/"), headers=headers, timeout=timeout) as client:
        catalog_response = client.get("/v1/models.data")
        catalog_response.raise_for_status()
        catalog_payload = catalog_response.json()
        lookup = _catalog_lookup(catalog_payload)
        models = _candidate_ids(
            explicit_models=args.model,
            candidate_files=args.candidate_file,
            catalog_payload=catalog_payload,
            include_catalog_candidates=not args.no_catalog_candidates,
        )

        results: list[dict[str, Any]] = []
        for model in models:
            state = _catalog_state(model, lookup)
            should_request = not args.available_only or state["available"]
            request = (
                _request_one(
                    client,
                    model=model,
                    image_url=args.image_url,
                    max_tokens=args.max_tokens,
                )
                if should_request
                else {
                    "request_ok": False,
                    "error_type": "skipped_unavailable",
                    "detail": "request skipped because --available-only was set",
                }
            )
            results.append({"model": model, **state, **request})

    summary = {
        "total": len(results),
        "catalog_available": sum(1 for item in results if item["catalog_state"] == "available"),
        "catalog_unavailable": sum(
            1 for item in results if item["catalog_state"] == "unavailable"
        ),
        "catalog_missing": sum(1 for item in results if item["catalog_state"] == "missing"),
        "request_ok": sum(1 for item in results if item["request_ok"]),
        "request_failed": sum(1 for item in results if not item["request_ok"]),
    }
    report = {
        "base_url": args.base_url.rstrip("/"),
        "max_tokens": args.max_tokens,
        "summary": summary,
        "results": results,
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if args.allow_failures or summary["request_failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
