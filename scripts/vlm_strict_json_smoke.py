"""Validate one vision model through the engine with image input and JSON mode.

Usage:
    uv run python scripts/vlm_strict_json_smoke.py \
      --model qwen3-vl-8b-instruct:vllm \
      --image /path/to/vehicle-a.jpg \
      --image /path/to/vehicle-b.jpg
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


def _available_status(payload: dict[str, Any], model: str) -> tuple[bool, str, dict[str, Any] | None]:
    data = payload.get("data") or []
    for entry in data:
        if isinstance(entry, dict) and entry.get("id") == model:
            return True, "", entry

    unavailable = payload.get("unavailable") or []
    for entry in unavailable:
        if isinstance(entry, dict) and entry.get("id") == model:
            reason = entry.get("reason") or "unavailable"
            detail = entry.get("detail") or ""
            return False, f"{model} is unavailable: {reason} {detail}".strip(), entry
    return False, f"{model} is not present in /v1/models.data", None


def _contract_skip_reason(entry: dict[str, Any] | None) -> str:
    if not isinstance(entry, dict):
        return ""
    if entry.get("supports_strict_image_json") is not False:
        return ""

    status = entry.get("strict_image_json_status") or "unsupported"
    checked_at = entry.get("strict_image_json_checked_at")
    detail = entry.get("strict_image_json_detail") or ""
    parts = ["supports_strict_image_json=false", f"status={status}"]
    if checked_at:
        parts.append(f"checked_at={checked_at}")
    if detail:
        parts.append(str(detail))
    return "; ".join(parts)


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


def _expectation_errors(
    parsed: dict[str, Any],
    *,
    expect_vehicle_visible: bool | None = None,
    expect_damage_visible: bool | None = None,
    require_reasons: bool = False,
    min_anomaly_score: float | None = None,
) -> list[str]:
    errors: list[str] = []
    if expect_vehicle_visible is not None and bool(parsed.get("vehicle_visible")) is not expect_vehicle_visible:
        errors.append(
            f"vehicle_visible expected {expect_vehicle_visible}, got {parsed.get('vehicle_visible')!r}"
        )
    if expect_damage_visible is not None and bool(parsed.get("damage_visible")) is not expect_damage_visible:
        errors.append(
            f"damage_visible expected {expect_damage_visible}, got {parsed.get('damage_visible')!r}"
        )
    if require_reasons:
        reasons = parsed.get("reasons")
        if not isinstance(reasons, list) or not reasons:
            errors.append("reasons expected a non-empty list")
    if min_anomaly_score is not None:
        try:
            anomaly_score = float(parsed.get("anomaly_score"))
        except (TypeError, ValueError):
            errors.append(f"anomaly_score expected numeric, got {parsed.get('anomaly_score')!r}")
        else:
            if anomaly_score < min_anomaly_score:
                errors.append(f"anomaly_score expected >= {min_anomaly_score}, got {anomaly_score}")
    return errors


def _run_one_image(
    client: httpx.Client,
    *,
    model: str,
    image: Path,
    max_tokens: int,
    expect_vehicle_visible: bool | None,
    expect_damage_visible: bool | None,
    require_reasons: bool,
    min_anomaly_score: float | None,
) -> dict[str, Any]:
    payload = _chat_payload(model, _data_url(image), max_tokens=max_tokens)
    started = time.perf_counter()
    try:
        chat_response = client.post("/v1/chat/completions", json=payload)
        latency_ms = (time.perf_counter() - started) * 1000
        chat_response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        return {
            "image": str(image),
            "ok": False,
            "latency_ms": round(latency_ms, 2),
            "error_type": "http_error",
            "http_status": exc.response.status_code,
            "detail": exc.response.text[:1000],
        }
    except httpx.RequestError as exc:
        latency_ms = (time.perf_counter() - started) * 1000
        return {
            "image": str(image),
            "ok": False,
            "latency_ms": round(latency_ms, 2),
            "error_type": "request_error",
            "detail": str(exc).splitlines()[0][:1000] if str(exc) else exc.__class__.__name__,
        }

    try:
        body = chat_response.json()
    except ValueError as exc:
        return {
            "image": str(image),
            "ok": False,
            "latency_ms": round(latency_ms, 2),
            "error_type": "bad_response_json",
            "detail": str(exc),
            "content": chat_response.text[:1000],
        }

    choice = (body.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    content = message.get("content") or ""
    reasoning_content = message.get("reasoning_content")
    try:
        parsed = _parse_json_content(content)
    except (json.JSONDecodeError, ValueError) as exc:
        return {
            "image": str(image),
            "ok": False,
            "latency_ms": round(latency_ms, 2),
            "finish_reason": choice.get("finish_reason"),
            "error_type": "json_parse_error",
            "detail": str(exc),
            "content": content[:1000],
            "reasoning_content_present": bool(reasoning_content),
        }

    errors = _expectation_errors(
        parsed,
        expect_vehicle_visible=expect_vehicle_visible,
        expect_damage_visible=expect_damage_visible,
        require_reasons=require_reasons,
        min_anomaly_score=min_anomaly_score,
    )
    if errors:
        return {
            "image": str(image),
            "ok": False,
            "latency_ms": round(latency_ms, 2),
            "finish_reason": choice.get("finish_reason"),
            "error_type": "expectation_error",
            "errors": errors,
            "parsed": parsed,
        }

    return {
        "image": str(image),
        "ok": True,
        "latency_ms": round(latency_ms, 2),
        "finish_reason": choice.get("finish_reason"),
        "usage": body.get("usage") or {},
        "parsed": parsed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=os.environ.get("ENGINE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--model", default=os.environ.get("ENGINE_MODEL"))
    parser.add_argument(
        "--image",
        type=Path,
        action="append",
        dest="images",
        required=True,
        help="Image to score. Repeat --image for a batch smoke.",
    )
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
    parser.add_argument(
        "--expect-vehicle-visible",
        action="store_true",
        help="Fail if the parsed JSON does not set vehicle_visible to true.",
    )
    parser.add_argument(
        "--expect-damage-visible",
        action="store_true",
        help="Fail if the parsed JSON does not set damage_visible to true.",
    )
    parser.add_argument(
        "--require-reasons",
        action="store_true",
        help="Fail if the parsed JSON does not include a non-empty reasons array.",
    )
    parser.add_argument(
        "--min-anomaly-score",
        type=float,
        default=None,
        help="Fail if anomaly_score is below this value.",
    )
    parser.add_argument(
        "--ignore-contract-metadata",
        action="store_true",
        help=(
            "Run even when /v1/models.data marks supports_strict_image_json=false. "
            "Use this for revalidation; benchmark harnesses should normally skip."
        ),
    )
    args = parser.parse_args()

    if not args.model:
        parser.error("--model is required, or set ENGINE_MODEL")
    missing = [path for path in args.images if not path.is_file()]
    if missing:
        parser.error(f"--image does not exist or is not a file: {missing[0]}")

    headers = _headers(args.api_key)
    with httpx.Client(base_url=args.base_url.rstrip("/"), headers=headers, timeout=600.0) as client:
        models_response = client.get("/v1/models.data")
        models_response.raise_for_status()
        ok, reason, catalog_entry = _available_status(models_response.json(), args.model)
        if not ok:
            print(reason, file=sys.stderr)
            return 2

        contract_skip_reason = _contract_skip_reason(catalog_entry)
        if contract_skip_reason and not args.ignore_contract_metadata:
            print(
                f"{args.model} is marked unavailable for strict image+JSON: "
                f"{contract_skip_reason}",
                file=sys.stderr,
            )
            return 2

        results = [
            _run_one_image(
                client,
                model=args.model,
                image=image,
                max_tokens=args.max_tokens,
                expect_vehicle_visible=True if args.expect_vehicle_visible else None,
                expect_damage_visible=True if args.expect_damage_visible else None,
                require_reasons=args.require_reasons,
                min_anomaly_score=args.min_anomaly_score,
            )
            for image in args.images
        ]

    passed = sum(1 for result in results if result["ok"])
    failed = len(results) - passed
    result = {
        "model": args.model,
        "max_tokens": args.max_tokens,
        "images": len(results),
        "passed": passed,
        "failed": failed,
        "results": results,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if not failed:
        return 0
    if any(item.get("error_type") != "expectation_error" for item in results if not item["ok"]):
        return 3
    return 4


if __name__ == "__main__":
    raise SystemExit(main())
