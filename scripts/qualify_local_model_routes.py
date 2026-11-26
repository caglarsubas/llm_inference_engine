"""Run narrow, labeled route checks on a dedicated local gateway and Ollama daemon.

The Ollama daemon must contain only locally installed targets and have one runner slot.
The loopback gateway must point there with substitutions, external fallbacks and auth
disabled. Give exact API model IDs with --model. Local Docker Model Runner routes
are supported through the gateway's vLLM adapter.
"""

import argparse
import json
import math
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.parse import urlsplit

import httpx


CHAT_CASES = (
    (
        "bay",
        "Dispatch note: previous loading bay B-12; current loading bay D-04. "
        "Return only the current bay code.",
        "D-04",
    ),
    (
        "lot",
        "Inspection note: lot XK-31 passed; lot QP-82 was rejected. "
        "Return only the ID of the rejected lot.",
        "QP-82",
    ),
    (
        "count",
        "Packing note: 14 cartons were planned. The final count was 11. "
        "Return only the final count as an integer.",
        "11",
    ),
)
PASSAGES = (
    "To restore access after forgetting your password, follow the account recovery link.",
    "Postal parcels can be traced using the tracking number on the dispatch receipt.",
    "To reschedule a dental visit, contact the clinic's appointment desk.",
)
QUERIES = (
    ("How can I move my dentist appointment to another day?", 2),
    ("I cannot remember my login credentials. How do I reset them?", 0),
    ("Where is the package I mailed last week?", 1),
)
CHAT_SETTINGS = {"temperature": 0, "top_p": 1, "seed": 17, "max_tokens": 128, "stream": False}
RULES = {
    "short_text_extraction": "All three trimmed, visible assistant answers exactly match the labels.",
    "short_text_retrieval": "All six finite nonzero vectors have the declared dimension and each "
    "query's labeled passage has a strictly higher cosine similarity than both alternatives.",
}


def utc():
    return datetime.now(UTC).isoformat(timespec="seconds")


class CheckError(ValueError):
    """A per-route verification failure safe to record in the published report."""


def local_url(url):
    parsed = urlsplit(url)
    if parsed.scheme != "http" or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise CheckError("expected_loopback_endpoint")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise CheckError("invalid_endpoint")
    return url.rstrip("/")


def request(client, path, body=None):
    response = client.get(path) if body is None else client.post(path, json=body)
    response.raise_for_status()
    return response.json()


def catalog(gateway):
    return {row["id"]: row for row in request(gateway, "/v1/models.data")["data"]}


def chat_check(gateway, model, row):
    cases = row["rows"]
    for case_id, prompt, label in CHAT_CASES:
        body = {"model": model, "messages": [{"role": "user", "content": prompt}], **CHAT_SETTINGS}
        start = time.monotonic()
        answer = request(gateway, "/v1/chat/completions", body)
        if answer["model"] != model:
            raise CheckError("model_substituted")
        choice = answer["choices"][0]
        content = choice["message"]["content"] or ""
        cases.append(
            {
                "case": case_id,
                "answer": content,
                "correct": content.strip() == label,
                "finish_reason": choice["finish_reason"],
                "completion_tokens": answer["usage"]["completion_tokens"],
                "elapsed_seconds": round(time.monotonic() - start, 3),
            }
        )
    hits = sum(case["correct"] for case in cases)
    row["summary"] = {
        "correct": hits,
        "examples": len(CHAT_CASES),
        "accuracy": hits / len(CHAT_CASES),
    }
    return hits == len(CHAT_CASES)


def embedding_check(gateway, model, dimensions, row):
    inputs = [*PASSAGES, *(query for query, _ in QUERIES)]
    start = time.monotonic()
    result = request(gateway, "/v1/embeddings", {"model": model, "input": inputs})
    if result["model"] != model:
        raise CheckError("model_substituted")
    rows = sorted(result["data"], key=lambda item: item["index"])
    if [item["index"] for item in rows] != list(range(len(inputs))):
        raise CheckError("missing_or_duplicate_embedding")
    vectors = [item["embedding"] for item in rows]
    norms = [math.hypot(*v) for v in vectors]
    valid = sum(len(v) == dimensions and math.isfinite(n) and n > 0 for v, n in zip(vectors, norms))
    row["dimensions"] = [len(v) for v in vectors]
    row["expected_dimension"] = dimensions
    row["elapsed_seconds"] = round(time.monotonic() - start, 3)
    cases = row["rows"]
    if valid == len(inputs):
        for i, (_, label) in enumerate(QUERIES, start=len(PASSAGES)):
            scores = [
                math.fsum((x / norms[i]) * (y / norms[j]) for x, y in zip(vectors[i], vectors[j]))
                for j in range(len(PASSAGES))
            ]
            cases.append(
                {
                    "query_index": i - len(PASSAGES),
                    "cosines": [round(s, 7) for s in scores],
                    "correct": all(scores[label] > s for j, s in enumerate(scores) if j != label),
                }
            )
    hits = sum(case["correct"] for case in cases)
    row["summary"] = {
        "correct": hits,
        "queries": len(QUERIES),
        "valid_vectors": valid,
        "vectors": len(inputs),
        "dimension": dimensions,
        "accuracy": hits / len(QUERIES),
    }
    return valid == len(inputs) and hits == len(QUERIES)


def sha256(value):
    if isinstance(value, str):
        bare = value.removeprefix("sha256:").lower()
        if re.fullmatch(r"[0-9a-f]{64}", bare):
            return bare
    raise CheckError("invalid_sha256")


def installed_tag(client, url, model):
    tags = [tag for tag in request(client, url)["models"] if tag["name"] == model]
    if len(tags) != 1:
        raise CheckError("missing_or_ambiguous_tag")
    return tags[0]


def docker_digest(client, entry):
    # The native inventory lives alongside /engines, including under a proxy prefix.
    endpoint = httpx.URL(local_url(entry["endpoint"]))
    parts = endpoint.path.rstrip("/").split("/")
    prefix = (
        "/".join(parts[: parts.index("engines")])
        if "engines" in parts
        else endpoint.path.rstrip("/")
    )
    tag = installed_tag(
        client, endpoint.copy_with(path=prefix + "/api/tags"), entry["upstream_model_id"]
    )
    expected_quant = entry["artifact_identity"].get("quantization")
    actual_quant = tag.get("details", {}).get("quantization_level")
    if actual_quant and expected_quant and actual_quant != expected_quant:
        raise CheckError("docker_quantization_mismatch")
    return sha256(tag["digest"])


def check_route(gateway, ollama, model, row):
    entry = catalog(gateway).get(model)
    if entry is None:
        raise CheckError("route_not_advertised")
    identity = entry.get("artifact_identity") or {}
    backend = row["backend"] = entry["backend"]
    row["quantization"] = identity.get("quantization")
    if backend == "ollama_http":
        if entry["endpoint"].rstrip("/") != str(ollama.base_url).rstrip("/"):
            raise CheckError("different_ollama_endpoint")
        tag = installed_tag(ollama, "/api/tags", model)
        if tag.get("remote_model") or tag.get("remote_host") or model.endswith(":cloud"):
            raise CheckError("cloud_proxy_excluded")
        digest = sha256(tag["digest"])
        if (
            identity.get("source") != "ollama_tags"
            or sha256(identity.get("manifest_sha256")) != digest
        ):
            raise CheckError("ollama_manifest_mismatch")
        row["artifact"] = {"manifest_sha256": digest}
        info = request(ollama, "/api/show", {"model": model})
        caps = info["capabilities"]
        if "completion" in caps:
            workload = "short_text_extraction"
        elif "embedding" in caps:
            workload = "short_text_retrieval"
        else:
            raise CheckError("no_supported_workload")
    elif backend == "vllm" and identity.get("source") == "docker_model_runner":
        digest = docker_digest(ollama, entry)
        if sha256(identity.get("model_sha256")) != digest:
            raise CheckError("docker_model_id_mismatch")
        row["artifact"] = {"model_sha256": digest}
        workload = "short_text_extraction"
    else:
        raise CheckError("no_verified_local_build")

    row["workload"] = workload
    row["digest"] = "sha256:" + digest
    if workload == "short_text_extraction":
        passed = chat_check(gateway, model, row)
    else:
        metadata = info["model_info"]
        arch = metadata["general.architecture"]
        passed = embedding_check(gateway, model, metadata[f"{arch}.embedding_length"], row)
    if catalog(gateway)[model]["artifact_identity"] != identity:
        raise CheckError("route_identity_changed")
    if backend == "ollama_http":
        residents = request(ollama, "/api/ps")["models"]
        if (
            len(residents) != 1
            or residents[0]["name"] != model
            or sha256(residents[0]["digest"]) != digest
        ):
            raise CheckError("resident_digest_unconfirmed")
    elif docker_digest(ollama, entry) != digest:
        raise CheckError("docker_model_id_changed")
    row["status"] = "passed" if passed else "failed"


def qualify(gateway, ollama, model):
    row = {
        "model": model,
        "backend": None,
        "artifact": None,
        "digest": None,
        "workload": None,
        "quantization": None,
        "rows": [],
        "summary": {},
        "status": "incomplete",
    }
    try:
        check_route(gateway, ollama, model, row)
    except CheckError as exc:
        row["error"] = str(exc)
    except httpx.HTTPStatusError as exc:
        row["error"] = f"HTTP {exc.response.status_code}"
    except (
        httpx.HTTPError,
        ValueError,
        KeyError,
        IndexError,
        TypeError,
        AttributeError,
        ArithmeticError,
    ) as exc:
        row["error"] = type(exc).__name__
    row["completed_at"] = utc()
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gateway-url", required=True, type=local_url)
    parser.add_argument("--ollama-url", required=True, type=local_url)
    parser.add_argument("--model", required=True, action="append")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists() or len(set(args.model)) != len(args.model):
        parser.error("Use a new output file and unique model IDs")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "started_at": utc(),
        "rules": RULES,
        "chat_settings": CHAT_SETTINGS,
        "chat_cases": [{"id": i, "prompt": p, "label": v} for i, p, v in CHAT_CASES],
        "passages": PASSAGES,
        "queries": [{"text": q, "positive_index": i} for q, i in QUERIES],
        "requested_routes": args.model,
        "results": [],
    }
    with (
        httpx.Client(
            base_url=args.gateway_url,
            timeout=95,
            trust_env=False,
            headers={"x-engine-model-substitution": "off"},
        ) as gateway,
        httpx.Client(base_url=args.ollama_url, timeout=5, trust_env=False) as ollama,
    ):
        try:
            report["ollama_version"] = request(ollama, "/api/version")["version"]
        except (httpx.HTTPError, ValueError, KeyError, TypeError):
            report["ollama_version"] = None
        with args.output.open("x") as stream:
            json.dump(report, stream, indent=2)
            stream.write("\n")
        for model in args.model:
            row = qualify(gateway, ollama, model)
            report["results"].append(row)
            print(model, row["status"], row.get("summary", {}), row.get("error", ""), flush=True)
            with NamedTemporaryFile(mode="w", dir=args.output.parent, delete=False) as tmp:
                json.dump(report, tmp, indent=2, allow_nan=False)
                tmp.write("\n")
            Path(tmp.name).replace(args.output)
    if any(row["status"] == "incomplete" for row in report["results"]):
        raise SystemExit("Some routes were incomplete. See the per-route results.")


if __name__ == "__main__":
    main()
