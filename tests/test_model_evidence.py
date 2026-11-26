import json

import pytest
from pydantic import ValidationError

from inference_engine import model_evidence
from inference_engine.registry import ArtifactIdentity

DIGEST = "a" * 64
REVISION = "b" * 40
ROUTE = "example:4b"


def record(**changes):
    return {
        "route_id": ROUTE,
        "backend": "ollama_http",
        "artifact": {"manifest_sha256": DIGEST},
        "workload": "document_extraction",
        "checked_at": "2026-11-05T00:00:00Z",
        "status": "passed",
        "metrics": {"field_accuracy": 0.97, "pages": 10},
        "evidence": "docs/benchmarks/extraction.json",
        **changes,
    }


@pytest.fixture
def manifest(monkeypatch, tmp_path):
    monkeypatch.setattr(model_evidence.resources, "files", lambda _: tmp_path)
    model_evidence.load_evaluations.cache_clear()

    def write(*records):
        (tmp_path / "model_evaluations.json").write_text(json.dumps({"evaluations": records}))

    yield write
    model_evidence.load_evaluations.cache_clear()


def select(identity, route_id=ROUTE, backend="ollama_http"):
    return model_evidence.evaluations_for(
        route_id=route_id, backend=backend, artifact_identity=identity
    )


def test_latest_failure_preserves_older_passing_run_for_same_workload(manifest):
    manifest(
        record(status="failed", checked_at="2026-11-06T03:00:00+02:00"),
        record(status="failed", workload="retrieval"),
        record(),
    )
    extraction, retrieval = select(ArtifactIdentity(source="ollama_tags", manifest_sha256=DIGEST))
    assert extraction.workload == "document_extraction"
    assert extraction.latest.status == "failed"
    assert extraction.latest.checked_at.isoformat() == "2026-11-06T01:00:00+00:00"
    assert extraction.last_passing.status == "passed"
    assert extraction.last_passing.checked_at.isoformat() == "2026-11-05T00:00:00+00:00"
    assert extraction.last_passing.metrics == {"field_accuracy": 0.97, "pages": 10}
    assert retrieval.latest.evidence == "docs/benchmarks/extraction.json"
    assert retrieval.last_passing is None


def test_different_build_route_backend_or_unverified_identity_has_no_results(manifest):
    manifest(record())
    correct = ArtifactIdentity(source="ollama_tags", manifest_sha256=DIGEST)
    assert select(correct, route_id="other:4b") == []
    assert select(correct, backend="llama_cpp") == []
    assert select(None) == []
    for identity in (
        ArtifactIdentity(source="ollama_tags", manifest_sha256="b" * 64),
        ArtifactIdentity(source="ollama_tags", quantization="Q4_K_M"),
        ArtifactIdentity(source="ollama_manifest", weight_sha256=DIGEST),
        ArtifactIdentity(source="operator_config", manifest_sha256=DIGEST),
    ):
        assert select(identity) == []


def test_hub_snapshot_is_repository_and_revision_specific(manifest):
    pin = {"repo_id": "publisher/model", "revision": REVISION}
    manifest(record(backend="mlx", artifact=pin))
    assert select(ArtifactIdentity(source="hf_snapshot", **pin), backend="mlx")
    for identity in (
        ArtifactIdentity(source="hf_snapshot", repo_id="fork/model", revision=REVISION),
        ArtifactIdentity(source="hf_snapshot", repo_id="publisher/model", revision="f" * 40),
        ArtifactIdentity(source="operator_config", **pin),
        ArtifactIdentity(source="mlx_config", quantization="4bit"),
    ):
        assert select(identity, backend="mlx") == []


def test_docker_model_id_is_an_independent_digest_namespace(manifest):
    manifest(record(backend="vllm", artifact={"model_sha256": DIGEST}))
    assert select(
        ArtifactIdentity(source="docker_model_runner", model_sha256=DIGEST), backend="vllm"
    )
    for identity in (
        ArtifactIdentity(source="docker_model_runner", model_sha256="f" * 64),
        ArtifactIdentity(source="operator_config", model_sha256=DIGEST),
        ArtifactIdentity(source="ollama_tags", manifest_sha256=DIGEST),
    ):
        assert select(identity, backend="vllm") == []


@pytest.mark.parametrize(
    "changes",
    [
        {"artifact": {"repo_id": "publisher/model", "revision": "main"}},
        {"artifact": {"manifest_sha256": DIGEST, "model_sha256": DIGEST}},
        {"artifact": {"weight_sha256": DIGEST}},
        {"checked_at": "2026-11-05T01:00:00"},
        {"metrics": {"field_accuracy": float("nan")}},
        {"evidence": "../private.log"},
    ],
)
def test_invalid_or_ambiguous_evidence_is_rejected(manifest, changes):
    manifest(record(**changes))
    with pytest.raises(ValidationError):
        model_evidence.load_evaluations()


def test_same_build_and_workload_cannot_have_conflicting_results_at_same_time(manifest):
    manifest(record(), record(status="failed", checked_at="2026-11-05T02:00:00+02:00"))
    with pytest.raises(ValidationError, match="duplicate evaluation"):
        model_evidence.load_evaluations()
