"""End-to-end smoke test against a running server.

Start the server first:
    make run

Then in another shell:
    uv run python scripts/smoke_test.py
"""

from __future__ import annotations

import json
import os
import sys

import httpx

BASE_URL = os.environ.get("ENGINE_URL", "http://127.0.0.1:8080")
MODEL = os.environ.get("ENGINE_MODEL", "llama3.2:1b")


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def main() -> int:
    with httpx.Client(base_url=BASE_URL, timeout=600.0) as client:
        section("GET /v1/health")
        r = client.get("/v1/health")
        r.raise_for_status()
        print(json.dumps(r.json(), indent=2))

        section("GET /v1/models")
        r = client.get("/v1/models")
        r.raise_for_status()
        ids = [m["id"] for m in r.json()["data"]]
        print(f"{len(ids)} models: {', '.join(ids)}")
        if MODEL not in ids:
            print(f"WARNING: target model {MODEL!r} not in registry; falling back to {ids[0]!r}")

        target = MODEL if MODEL in ids else ids[0]

        section(f"POST /v1/chat/completions (blocking) — model={target}")
        payload = {
            "model": target,
            "messages": [
                {"role": "system", "content": "You are a terse assistant. Answer in one sentence."},
                {"role": "user", "content": "Why is the sky blue?"},
            ],
            "max_tokens": 80,
            "temperature": 0.4,
        }
        r = client.post("/v1/chat/completions", json=payload)
        r.raise_for_status()
        data = r.json()
        print(f"id: {data['id']}")
        print(f"finish_reason: {data['choices'][0]['finish_reason']}")
        print(f"usage: {data['usage']}")
        print(f"response: {data['choices'][0]['message']['content']}")

        section(f"POST /v1/chat/completions (streaming) — model={target}")
        payload["stream"] = True
        payload["messages"] = [{"role": "user", "content": "Count 1 to 5."}]
        with client.stream("POST", "/v1/chat/completions", json=payload) as resp:
            resp.raise_for_status()
            buf: list[str] = []
            for line in resp.iter_lines():
                if not line.startswith("data:"):
                    continue
                body = line[5:].strip()
                if body == "[DONE]":
                    break
                try:
                    chunk = json.loads(body)
                except json.JSONDecodeError:
                    continue
                delta = chunk["choices"][0]["delta"].get("content")
                if delta:
                    buf.append(delta)
                    sys.stdout.write(delta)
                    sys.stdout.flush()
            print()
            print(f"streamed {len(''.join(buf))} chars in {sum(1 for _ in buf)} chunks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
