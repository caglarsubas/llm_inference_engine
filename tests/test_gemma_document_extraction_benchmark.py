import hashlib
import json
import subprocess

import pytest

from scripts import benchmark_gemma_document_extraction as benchmark


@pytest.mark.parametrize(
    ("listeners", "correct"), [("811\n", True), ("722\n", False), ("811\n722\n", False)],
)
def test_memory_measurement_requires_the_dedicated_listener_pid(monkeypatch, listeners, correct):
    def lsof(command, **kwargs):
        assert command == ["lsof", "-nP", "-iTCP:12434", "-sTCP:LISTEN", "-t"]
        return subprocess.CompletedProcess(command, 0, stdout=listeners)

    monkeypatch.setattr(benchmark.subprocess, "run", lsof)
    if correct:
        benchmark.verify_listener_pid("http://127.0.0.1:12434", 811)
    else:
        with pytest.raises(ValueError, match="sole listener"):
            benchmark.verify_listener_pid("http://127.0.0.1:12434", 811)


@pytest.mark.parametrize(
    ("content", "completed", "valid_json", "schema_valid", "correct"),
    [
        ('{"field":null}', True, True, True, 1),
        ('{"field":"unknown"}', True, True, True, 0),
        ('{"field":null}', False, False, False, 0),
        ('{"field":null,"extra":null}', True, True, False, 0),
        ('{"field":2}', True, True, False, 0),
        ('{"field":null,"field":0}', True, False, False, 0),
        ('{"field":NaN}', True, False, False, 0),
    ],
)
def test_strict_scoring(content, completed, valid_json, schema_valid, correct):
    row = {"content": content, "completed": completed}
    benchmark.score(row, {"field": None})
    assert (row["valid_json"], row["schema_valid"], row["exact_fields"]) == (
        valid_json,
        schema_valid,
        correct,
    )
    assert row["exact_page"] is bool(correct)


def test_published_workload_scores_match_images_labels_and_raw_responses():
    fixture_file = benchmark.FIXTURE
    fixture = json.loads(fixture_file.read_text())
    results = json.loads((fixture_file.parent / "results.json").read_text())
    assert hashlib.sha256(fixture_file.read_bytes()).hexdigest() == results["fixture_sha256"]
    for case in fixture["cases"]:
        assert (
            case["sha256"]
            == hashlib.sha256((fixture_file.parent / case["image"]).read_bytes()).hexdigest()
        )

    for arm in results["models"]:
        for case, row in zip(fixture["cases"], arm["rows"], strict=True):
            assert case["id"] == row["case"]
            rescored = row.copy()
            benchmark.score(rescored, case["expected"])
            assert rescored == row
        s = arm["summary"]
        for scored, aggregate in (
            ("schema_valid", "schema_valid"),
            ("exact_fields", "exact_fields"),
            ("exact_page", "exact_pages"),
        ):
            assert sum(row[scored] for row in arm["rows"]) == s[aggregate]
        t = fixture["thresholds"]
        passed = (
            s["schema_valid"] == t["schema_valid_pages"]
            and s["exact_fields"] >= t["exact_fields"]
            and s["exact_pages"] >= t["exact_pages"]
        )
        assert arm["status"] == ("passed" if passed else "failed")
