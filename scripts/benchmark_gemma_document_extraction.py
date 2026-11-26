"""Score two vision models on the checked-in invoice pages through an isolated gateway.

Use a dedicated Ollama instance and matching gateway. The driver releases each
tested model via POST /api/generate before proceeding to the next model.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

if __package__:
    from . import benchmark_ollama_keep_alive as monitor
else:
    import benchmark_ollama_keep_alive as monitor

FIXTURE = (
    Path(__file__).resolve().parents[1] / "docs/benchmarks/gemma-document-extraction/cases.json"
)
MODELS = ("gemma3:4b", "gemma3:12b")
PROMPT = (
    "Extract the five requested fields from the attached invoice. Use the invoice number "
    "for invoice_id, and copy the issuer's name exactly as printed, including case. "
    "Convert the issue date to YYYY-MM-DD. For currency use the printed three-letter "
    "currency code. For amount_due use the unpaid balance, as a string with a decimal "
    "point and two digits, no symbols or thousands separators. Use null for an absent "
    "field. Do not infer the invoice date from the payment deadline. Return only JSON."
)
INTERVAL_S = 0.25


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def verify_listener_pid(url: str, pid: int) -> None:
    address = httpx.URL(url)
    if address.scheme not in {"http", "https"} or address.host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("Process memory measurement requires a dedicated local Ollama daemon")
    port = address.port or (443 if address.scheme == "https" else 80)
    # Require a unique owner across IPv4 and IPv6. A different daemon on the
    # same localhost port would make the measured process tree ambiguous.
    found = subprocess.run(
        ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"],
        capture_output=True, text=True, timeout=10, check=False,
    )
    if found.returncode != 0 or set(found.stdout.split()) != {str(pid)}:
        raise ValueError(f"Cannot verify PID {pid} as the sole listener on Ollama port {port}")


def manifest_digest(value: str) -> str:
    digest = value.removeprefix("sha256:")
    if not re.fullmatch(r"[a-f0-9]{64}", digest):
        raise ValueError("Expected a full Ollama manifest SHA-256")
    return digest


def unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key {key!r}")
        result[key] = value
    return result


def reject_non_json_number(value):
    raise ValueError(f"{value} is not a JSON number")


def consume(client, model, payload):
    row = {"completed": False, "content": "", "ttft_s": None, "finish_reason": None}
    started = time.monotonic()
    try:
        with client.stream("POST", "/v1/chat/completions", json=payload) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line.startswith("data:"):
                    continue
                data = line.removeprefix("data:").strip()
                if data == "[DONE]":
                    row["completed"] = row["finish_reason"] == "stop" and bool(row["content"])
                    break
                frame = json.loads(data)
                if frame.get("error") or frame.get("model") != model:
                    raise ValueError("Unexpected stream error or served model")
                for choice in frame["choices"]:
                    content = choice["delta"].get("content") or ""
                    if row["ttft_s"] is None and content.strip():
                        row["ttft_s"] = time.monotonic() - started
                    row["content"] += content
                    if choice.get("finish_reason"):
                        row["finish_reason"] = choice["finish_reason"]
    except (httpx.HTTPError, ValueError, TypeError, KeyError) as exc:
        row["error_type"] = type(exc).__name__
        if isinstance(exc, httpx.HTTPStatusError):
            row["http_status"] = exc.response.status_code
    row["gateway_latency_s"] = time.monotonic() - started
    return row


def score(row, expected):
    row.update(valid_json=False, schema_valid=False, exact_page=False, exact_fields=0)
    if not row["completed"]:
        return
    try:
        parsed = json.loads(
            row["content"], object_pairs_hook=unique_keys, parse_constant=reject_non_json_number
        )
    except ValueError:
        return
    row["valid_json"] = True
    if not isinstance(parsed, dict) or parsed.keys() != expected.keys():
        return
    if not all(v is None or isinstance(v, str) for v in parsed.values()):
        return
    row["schema_valid"] = True
    row["exact_fields"] = sum(parsed[key] == value for key, value in expected.items())
    row["exact_page"] = parsed == expected


def peak(samples, key):
    values = [s[key] for s in samples if s.get(key) is not None]
    return max(values, default=None)


def run_model(gateway, daemon, pid, tag, fixture, images, schema):
    model = tag["name"]
    if daemon.get("/api/ps").raise_for_status().json()["models"]:
        raise ValueError("Start each arm with an empty dedicated Ollama daemon")
    rows = []
    stop = threading.Event()
    failures = []
    started_at = utc()
    with tempfile.TemporaryDirectory() as tmp:
        trace = Path(tmp) / "samples.jsonl"
        arm = monitor.Arm(model, daemon, gateway, pid, Path(tmp), "30m")
        sampler = threading.Thread(
            target=monitor.sample_memory, args=([arm], trace, INTERVAL_S, stop, failures)
        )
        sampler.start()
        try:
            for case, image in zip(fixture["cases"], images, strict=True):
                payload = {
                    "model": model,
                    "stream": True,
                    "temperature": 0,
                    "seed": 21,
                    "top_p": 1.0,
                    "max_tokens": 192,
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {"name": "invoice", "strict": True, "schema": schema},
                    },
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": PROMPT},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": "data:image/png;base64,"
                                        + base64.b64encode(image).decode()
                                    },
                                },
                            ],
                        }
                    ],
                }
                before = monitor.resident(arm)
                t0 = time.monotonic()
                row = consume(gateway, model, payload)
                t1 = time.monotonic()
                ps = daemon.get("/api/ps").raise_for_status().json()["models"]
                if row["completed"] and (
                    len(ps) != 1
                    or ps[0]["name"] != model
                    or manifest_digest(ps[0]["digest"]) != tag["digest"]
                ):
                    raise ValueError(
                        f"Observed runtime did not confirm the requested build of {model}"
                    )
                row.update(
                    case=case["id"],
                    cold_start=model not in before,
                    sample_start=t0,
                    sample_end=t1,
                    context_length=ps[0].get("context_length") if ps else None,
                    ollama_bytes_after=ps[0]["size"] if ps else None,
                )
                score(row, case["expected"])
                rows.append(row)
                print(
                    model,
                    case["id"],
                    "complete" if row["completed"] else row.get("error_type", "incomplete"),
                    row["exact_fields"],
                    round(row["gateway_latency_s"], 2),
                    flush=True,
                )
        finally:
            stop.set()
            sampler.join()
            daemon.post("/api/generate", json={"model": model, "keep_alive": 0}).raise_for_status()
        samples = [json.loads(line) for line in trace.read_text().splitlines()]
    samples = [s for s in samples if s.get("arm") == model]
    if not samples:
        raise RuntimeError("No memory measurements available")
    for row in rows:
        t0, t1 = row.pop("sample_start"), row.pop("sample_end")
        window = [s for s in samples if t0 <= s["monotonic_s"] <= t1]
        row["peak_footprint_bytes"] = peak(window, "footprint_bytes")
        row["peak_ollama_bytes"] = peak(window, "ollama_bytes")
    summary = {
        "cases": len(rows),
        "valid_json": sum(r["valid_json"] for r in rows),
        "schema_valid": sum(r["schema_valid"] for r in rows),
        "exact_pages": sum(r["exact_page"] for r in rows),
        "exact_fields": sum(r["exact_fields"] for r in rows),
        "total_fields": len(rows) * len(fixture["fields"]),
        "memory_samples": len(samples),
        "memory_sampling_errors": len(failures),
        "footprint_samples": sum(s["footprint_bytes"] is not None for s in samples),
        "peak_footprint_bytes": peak(samples, "footprint_bytes"),
        "peak_ollama_bytes": peak(samples, "ollama_bytes"),
        "max_memory_interval_s": max(
            (b["monotonic_s"] - a["monotonic_s"] for a, b in zip(samples, samples[1:])),
            default=None,
        ),
    }
    for name in ("gateway_latency_s", "ttft_s"):
        vals = sorted(r[name] for r in rows if r["completed"] and r[name] is not None)
        summary[f"median_{name}"] = statistics.median(vals) if vals else None
        summary[f"p95_{name}"] = vals[math.ceil(0.95 * len(vals)) - 1] if vals else None
    thresholds = fixture["thresholds"]
    passed = (
        summary["schema_valid"] == thresholds["schema_valid_pages"]
        and summary["exact_pages"] >= thresholds["exact_pages"]
        and summary["exact_fields"] >= thresholds["exact_fields"]
    )
    return {
        "model": model,
        "backend": "ollama_http",
        "digest": "sha256:" + tag["digest"],
        "quantization": tag["details"]["quantization_level"],
        "started_at": started_at,
        "completed_at": utc(),
        "status": "passed" if passed else "failed",
        "summary": summary,
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gateway-url", required=True)
    parser.add_argument("--dedicated-ollama-url", required=True)
    parser.add_argument("--ollama-pid", type=int, required=True)
    parser.add_argument("--gateway-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output path to preserve existing evidence")
    if args.ollama_pid <= 0 or not re.fullmatch(r"[a-f0-9]{40}", args.gateway_revision):
        parser.error("Supply a positive daemon PID and an exact Git commit for the gateway")
    try:
        verify_listener_pid(args.dedicated_ollama_url, args.ollama_pid)
    except (OSError, subprocess.TimeoutExpired, ValueError) as exc:
        parser.error(str(exc))
    fixture_data = FIXTURE.read_bytes()
    fixture = json.loads(fixture_data)
    images = []
    for case in fixture["cases"]:
        image = (FIXTURE.parent / case["image"]).read_bytes()
        if sha256(image) != case["sha256"]:
            raise ValueError(f"Fixture image digest mismatch for {case['id']}")
        images.append(image)
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": fixture["fields"],
        "properties": {key: {"type": ["string", "null"]} for key in fixture["fields"]},
    }
    headers = {"Connection": "close", "x-engine-model-substitution": "off"}
    key = os.getenv("INFERENCE_ENGINE_API_KEY")
    if key:
        headers["Authorization"] = f"Bearer {key}"
    with (
        httpx.Client(
            base_url=args.gateway_url, trust_env=False, timeout=120, headers=headers
        ) as gateway,
        httpx.Client(base_url=args.dedicated_ollama_url, trust_env=False, timeout=120) as daemon,
    ):
        routes = {
            r["id"]: r for r in gateway.get("/v1/models.data").raise_for_status().json()["data"]
        }
        tags = {r["name"]: r for r in daemon.get("/api/tags").raise_for_status().json()["models"]}
        result = {
            "workload": fixture["workload"],
            "fixture_sha256": sha256(fixture_data),
            "started_at": utc(),
            "gateway_revision": args.gateway_revision,
            "ollama_version": daemon.get("/api/version").raise_for_status().json()["version"],
            "seed": 21,
            "temperature": 0,
            "top_p": 1,
            "max_tokens": 192,
            "prompt": PROMPT,
            "response_schema": schema,
            "memory_interval_s": INTERVAL_S,
            "models": [],
        }
        for model in MODELS:
            if (
                routes[model]["backend"] != "ollama_http"
                or routes[model]["endpoint"].rstrip("/") != str(daemon.base_url).rstrip("/")
            ):
                raise ValueError("The catalog route must point to the dedicated Ollama daemon")
            tags[model]["digest"] = manifest_digest(tags[model]["digest"])
            result["models"].append(
                run_model(gateway, daemon, args.ollama_pid, tags[model], fixture, images, schema)
            )
            for _ in range(100):
                if not daemon.get("/api/ps").raise_for_status().json()["models"]:
                    break
                time.sleep(0.1)
            else:
                raise RuntimeError(f"Ollama still has a resident model after releasing {model}")
        result["completed_at"] = utc()
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
