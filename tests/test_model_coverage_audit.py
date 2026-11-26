from datetime import date

import httpx

from scripts.audit_model_coverage import audit, popular_repos


def test_audit_counts_publishers_and_explicit_direct_quantizations_once():
    def row(repo, tags=(), created="2026-05-25"):
        return {"id": repo, "tags": tags, "createdAt": created, "downloads": 100_000}

    direct = row("unsloth/qwen-gguf", ["base_model:quantized:Qwen/Qwen3"])
    rows = [
        row("Qwen/Qwen3"), direct, direct,
        row("mistralai/Voxtral", ["base_model:quantized:Qwen/Qwen3"]),
        row("org/distill", ["base_model:finetune:Qwen/Qwen3"]),
        row("org/indirect", ["base_model:quantized:unsloth/qwen-gguf"]),
        row("google/unrelated"), row("meta-llama/Llama-3", created="2025-01-01"),
    ]
    results, distillations = audit(rows, {"UNSLOTH/QWEN-GGUF"}, date(2026, 5, 25),
                                   date(2026, 11, 25))
    assert [(r["id"], r["kind"], r["covered"]) for r in results] == [
        ("Qwen/Qwen3", "publisher", False), ("mistralai/Voxtral", "publisher", False),
        ("unsloth/qwen-gguf", "quantization", True),
    ]
    assert results[-1]["parents"] == ["Qwen/Qwen3"]
    assert [r["id"] for r in distillations] == ["org/distill"]


def test_distillation_keeps_intermediate_parent_and_overlap_separate():
    row = {"id": "gguf/Student-Distill", "downloads": 100_000, "createdAt": "2026-10-01",
           "tags": ["base_model:studio/Student", "base_model:quantized:studio/Student"]}
    overlap = {**row, "id": "Qwen/Qwen-Distill", "tags": []}
    results, candidates = audit([row, row, overlap], {"gguf/student-distill"},
                                date(2026, 5, 25), date(2026, 11, 25))
    assert [r["id"] for r in results] == ["Qwen/Qwen-Distill"]
    assert [(r["id"], r["parents"], r["covered"]) for r in candidates] == [
        ("Qwen/Qwen-Distill", [], False), ("gguf/Student-Distill", ["studio/Student"], True),
    ]


def test_popularity_scan_follows_cursor_and_stops_at_cutoff():
    pages = []

    def respond(request):
        pages.append(request.url)
        if len(pages) == 1:
            return httpx.Response(200, json=[{"downloads": 101}],
                                  headers={"link": '<https://huggingface.co/next?a=1>; rel="next"'})
        assert str(request.url) == "https://huggingface.co/next?a=1"
        return httpx.Response(200, json=[{"downloads": 100}, {"downloads": 99}],
                              headers={"link": '<https://huggingface.co/unneeded>; rel="next"'})

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        assert list(popular_repos(client, 100)) == [{"downloads": 101}, {"downloads": 100}]
    assert len(pages) == 2
