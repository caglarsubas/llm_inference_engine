"""Validate one vision model through the engine with image input and JSON mode.

Usage:
    uv run python scripts/vlm_strict_json_smoke.py \
      --model qwen3-vl-8b-instruct:vllm \
      --image /path/to/vehicle.jpg
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import sys
import time
from pathlib import Path
from typing import Any

import httpx


DEFAULT_BASE_URL = "http://127.0.0.1:8080"
REQUIRED_KEYS = {"vehicle_visible", "damage_visible", "anomaly_score", "confidence"}


def _data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _headers(api_key: str | None) -> dict[str, str]:
    if not api_key:
        return {}
    return {"Authorization": f"Bearer {api_key}"}


def _available_status(payload: dict[str, Any], model: str) -> tuple[bool, str]:
    data = payload.get("data") or []
    if any(isinstance(entry, dict) and entry.get("id") == model for entry in data):
        return True, ""

    unavailable = payload.get("unavailable") or []
    for entry in unavailable:
        if isinstance(entry, dict) and entry.get("id") == model:
            reason = entry.get("reason") or "unavailable"
            detail = entry.get("detail") or ""
            return False, f"{model} is unavailable: {reason} {detail}".strip()
    return False, f"{model} is not present in /v1/models.data"


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
                    "Return one JSON object only. Required keys: "
                    "vehicle_visible, damage_visible, anomaly_score, confidence, reasons."
                ),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Inspect this vehicle photo for fraud-relevant anomalies. "
                            "Use boolean vehicle_visible and damage_visible, numeric "
                            "anomaly_score and confidence from 0 to 1, and a reasons array."
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


def _parse_json_content(content: str) -> dict[str, Any]:
    parsed = json.loads(content)
    if not isinstance(parsed, dict):
        raise ValueError("model response JSON must be an object")
    missing = sorted(REQUIRED_KEYS.difference(parsed))
    if missing:
        raise ValueError(f"model response missing required keys: {', '.join(missing)}")
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=os.environ.get("ENGINE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--model", default=os.environ.get("ENGINE_MODEL"))
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=int(os.environ.get("ENGINE_MAX_TOKENS", "768")),
        help=(
            "Completion token budget for strict JSON. Defaults to 768 because "
            "256 truncates some local VLM/Ollama candidates."
        ),
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("ENGINE_API_KEY") or os.environ.get("ENGINE_TOKEN"),
        help="Bearer token for AUTH_ENABLED deployments.",
    )
    args = parser.parse_args()

    if not args.model:
        parser.error("--model is required, or set ENGINE_MODEL")
    if not args.image.is_file():
        parser.error(f"--image does not exist or is not a file: {args.image}")

    headers = _headers(args.api_key)
    with httpx.Client(base_url=args.base_url.rstrip("/"), headers=headers, timeout=600.0) as client:
        models_response = client.get("/v1/models")
        models_response.raise_for_status()
        ok, reason = _available_status(models_response.json(), args.model)
        if not ok:
            print(reason, file=sys.stderr)
            return 2

        payload = _chat_payload(args.model, _data_url(args.image), max_tokens=args.max_tokens)
        started = time.perf_counter()
        chat_response = client.post("/v1/chat/completions", json=payload)
        latency_ms = (time.perf_counter() - started) * 1000
        chat_response.raise_for_status()
        body = chat_response.json()

    choice = (body.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    content = message.get("content") or ""
    try:
        parsed = _parse_json_content(content)
    except (json.JSONDecodeError, ValueError) as exc:
        print(f"strict JSON smoke failed: {exc}", file=sys.stderr)
        print(content[:1000], file=sys.stderr)
        return 3

    result = {
        "model": args.model,
        "latency_ms": round(latency_ms, 2),
        "finish_reason": choice.get("finish_reason"),
        "max_tokens": args.max_tokens,
        "usage": body.get("usage") or {},
        "parsed": parsed,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
