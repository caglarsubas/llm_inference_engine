from pathlib import PurePosixPath

from fastapi.testclient import TestClient

from inference_engine import auth
from inference_engine.config import settings
from inference_engine.main import app
from inference_engine.portfolio import load_portfolio


def test_portfolio_entries_identify_distinct_pinned_artifacts():
    portfolio = load_portfolio()
    assert len({entry.id for entry in portfolio.entries}) == len(portfolio.entries)
    for entry in portfolio.entries:
        assert entry.release <= portfolio.reviewed_on.strftime("%Y-%m")
        assert entry.id.count("/") == 1
        assert bool(entry.ollama_tag) == (entry.integration == "ollama")
        assert entry.model_card.endswith(f"/blob/{entry.revision}/README.md")


def test_curated_releases_have_dated_adoption_and_scoped_quality_evidence():
    portfolio = load_portfolio()
    assert portfolio.popularity_observed_on is not None
    assert portfolio.popularity_observed_on <= portfolio.reviewed_on
    for entry in portfolio.entries:
        assert entry.downloads_30d is not None, entry.id
        assert entry.quality_evidence is not None, entry.id
        for build in entry.variants:
            assert build.downloads_30d is not None, build.repo_id


def test_distribution_artifacts_are_unique_and_gemma_vision_includes_projectors():
    seen = set()
    gemma_multimodal = set()
    for entry in load_portfolio().entries:
        for variant in entry.variants:
            key = (variant.repo_id, variant.quantization)
            assert key not in seen
            seen.add(key)
            paths = [artifact.path for artifact in variant.files]
            assert len(paths) == len(set(paths))
            assert all(not PurePosixPath(p).is_absolute() and ".." not in PurePosixPath(p).parts
                       for p in paths)
            if entry.family == "Gemma 3" and variant.format == "gguf":
                assert {artifact.role for artifact in variant.files} == {"model", "projector"}
                gemma_multimodal.add(entry.id)
    assert gemma_multimodal == {"google/gemma-3-4b-it", "google/gemma-3-12b-it"}


def test_portfolio_route_obeys_auth_and_filters_without_loading_models(monkeypatch, request):
    client = TestClient(app)
    request.addfinalizer(auth._reset_for_tests)
    auth._set_keys_for_tests([("portfolio-test-key", "catalog-reader")])
    monkeypatch.setattr(settings, "auth_enabled", True)
    assert client.get("/v1/model-portfolio").status_code == 401
    client.headers["Authorization"] = "Bearer portfolio-test-key"
    response = client.get("/v1/model-portfolio", params={"task": "decision"})
    assert response.status_code == 200
    catalog = response.json()
    assert catalog["popularity_observed_on"] == load_portfolio().popularity_observed_on.isoformat()
    entries = catalog["entries"]
    assert "autogluon/mitra-classifier-2" in {entry["id"] for entry in entries}
    assert all("decision" in entry["tasks"] for entry in entries)
    assert all(entry["quality_evidence"]["source"].startswith("https://") for entry in entries)
    assert all(entry["downloads_30d"] >= 0 for entry in entries)
    assert client.get("/v1/model-portfolio?task=no-such-task").json()["entries"] == []
