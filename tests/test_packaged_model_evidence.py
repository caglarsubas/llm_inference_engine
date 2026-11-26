import json
from datetime import datetime
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI

from inference_engine import auth
from inference_engine.api import models
from inference_engine.config import settings
from inference_engine.model_evidence import load_evaluations
from inference_engine.registry import CompositeRegistry, OllamaHttpRegistry


ROOT = Path(__file__).resolve().parents[1]


def test_packaged_evaluations_match_dated_benchmark_outputs():
    reports = {}
    records = load_evaluations().evaluations
    assert {
        "docs/benchmarks/gemma-document-extraction/results.json",
        "docs/benchmarks/local-route-checks/results.json",
    }.issubset({record.evidence for record in records})
    expected = set()
    for record in records:
        if record.evidence not in reports:
            report = json.loads((ROOT / record.evidence).read_text())
            rows = report["models"] if "models" in report else report["results"]
            index = {
                (
                    row["model"],
                    row.get("workload", report.get("workload")),
                    datetime.fromisoformat(row["completed_at"]),
                ): row
                for row in rows
                if row["status"] in {"passed", "failed"}
            }
            reports[record.evidence] = index
            expected.update((record.evidence, *key) for key in index)
        key = (record.route_id, record.workload, record.checked_at)
        row = reports[record.evidence][key]
        assert row["backend"] == record.backend
        assert row["status"] == record.status
        digest_type = "model_sha256" if record.backend == "vllm" else "manifest_sha256"
        assert record.artifact.model_dump() == {digest_type: row["digest"].removeprefix("sha256:")}
        for metric, value in record.metrics.items():
            assert row["summary"][metric] == value
    assert {(r.evidence, r.route_id, r.workload, r.checked_at) for r in records} == expected


@pytest.mark.asyncio
async def test_authenticated_catalog_only_qualifies_the_measured_build(monkeypatch):
    record = next(
        r
        for r in load_evaluations().evaluations
        if r.route_id == "gemma3:12b" and r.workload == "document_extraction"
    )
    tag = {"name": record.route_id, "digest": record.artifact.manifest_sha256, "size": 10}
    registry = OllamaHttpRegistry("http://test-ollama")
    monkeypatch.setattr(registry, "_fetch_tags", lambda: [tag])
    monkeypatch.setattr(registry, "_fetch_metadata", lambda _: {"modality": "text+image->text"})
    monkeypatch.setattr(models.app_state, "registry", CompositeRegistry([registry]))
    app = FastAPI()
    app.include_router(models.router)
    monkeypatch.setattr(settings, "auth_enabled", True)
    monkeypatch.setattr(auth, "_keys_by_value", {})
    monkeypatch.setattr(auth, "_keys_loaded", False)
    auth._set_keys_for_tests([("catalog-test", "test")])
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as c:
        assert (await c.get("/v1/models.data")).status_code == 401
        c.headers["Authorization"] = "Bearer catalog-test"
        (entry,) = (await c.get("/v1/models.data")).json()["data"]
        assert entry["id"] == record.route_id and entry["supports_images"]
        (evidence,) = entry["workload_evaluations"]
        assert evidence["workload"] == record.workload
        assert evidence["last_passing"] == evidence["latest"]
        assert evidence["latest"]["status"] == "passed"
        assert evidence["latest"]["evidence"] == record.evidence
        assert entry["supports_strict_image_json"] is None
        (standard,) = (await c.get("/v1/models")).json()["data"]
        assert standard["id"] == entry["id"]
        assert "artifact_identity" not in standard and "workload_evaluations" not in standard

        tag["digest"] = "0" * 64
        registry.invalidate()
        (changed,) = (await c.get("/v1/models.data")).json()["data"]
        assert changed["artifact_identity"]["manifest_sha256"] == tag["digest"]
        assert changed["workload_evaluations"] == []
