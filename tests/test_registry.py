"""Registry round-trip against a synthetic Ollama-format store."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from inference_engine.registry import OllamaRegistry


def _write_blob(blobs_dir: Path, content: bytes) -> tuple[str, Path]:
    digest = hashlib.sha256(content).hexdigest()
    path = blobs_dir / f"sha256-{digest}"
    path.write_bytes(content)
    return f"sha256:{digest}", path


def _make_store(root: Path, *, model_bytes: bytes = b"FAKE GGUF BYTES") -> None:
    blobs = root / "blobs"
    manifests = root / "manifests" / "registry.ollama.ai" / "library" / "demo"
    blobs.mkdir(parents=True)
    manifests.mkdir(parents=True)

    model_digest, _ = _write_blob(blobs, model_bytes)
    template_digest, _ = _write_blob(blobs, b"{{ .Prompt }}")
    params_digest, _ = _write_blob(blobs, json.dumps({"temperature": 0.5}).encode())

    manifest = {
        "schemaVersion": 2,
        "mediaType": "application/vnd.docker.distribution.manifest.v2+json",
        "config": {"mediaType": "application/vnd.docker.container.image.v1+json", "digest": "sha256:0", "size": 0},
        "layers": [
            {"mediaType": "application/vnd.ollama.image.model", "digest": model_digest, "size": len(model_bytes)},
            {"mediaType": "application/vnd.ollama.image.template", "digest": template_digest, "size": 12},
            {"mediaType": "application/vnd.ollama.image.params", "digest": params_digest, "size": 24},
        ],
    }
    (manifests / "1b").write_text(json.dumps(manifest))


def test_list_models_discovers_manifest(tmp_path: Path) -> None:
    _make_store(tmp_path)
    registry = OllamaRegistry(tmp_path)
    models = registry.list_models()

    assert len(models) == 1
    desc = models[0]
    assert desc.name == "demo"
    assert desc.tag == "1b"
    assert desc.qualified_name == "demo:1b"
    assert desc.format == "gguf"
    assert desc.model_path.exists()
    assert desc.template == "{{ .Prompt }}"
    assert desc.params == {"temperature": 0.5}


def test_get_resolves_short_name(tmp_path: Path) -> None:
    _make_store(tmp_path)
    registry = OllamaRegistry(tmp_path)
    assert registry.get("demo:1b") is not None
    assert registry.get("demo") is not None  # short form
    assert registry.get("nope") is None


def test_missing_model_blob_skips_manifest(tmp_path: Path) -> None:
    blobs = tmp_path / "blobs"
    manifests = tmp_path / "manifests" / "registry.ollama.ai" / "library" / "ghost"
    blobs.mkdir(parents=True)
    manifests.mkdir(parents=True)
    manifest = {
        "schemaVersion": 2,
        "layers": [
            {"mediaType": "application/vnd.ollama.image.model", "digest": "sha256:deadbeef", "size": 0},
        ],
    }
    (manifests / "1b").write_text(json.dumps(manifest))
    registry = OllamaRegistry(tmp_path)
    assert registry.list_models() == []


def test_missing_root_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        OllamaRegistry(tmp_path / "does-not-exist")


def test_n_ctx_train_reads_arch_prefixed_metadata_key() -> None:
    """GGUF stores it as ``<arch>.context_length``; match on the suffix so a
    new architecture doesn't silently report an unknown window."""
    from inference_engine.registry.probe import _n_ctx_train_from_metadata

    assert _n_ctx_train_from_metadata({"llama.context_length": "131072"}) == 131072
    assert _n_ctx_train_from_metadata({"gemma3.context_length": 8192}) == 8192


def test_n_ctx_train_returns_zero_when_unknown() -> None:
    from inference_engine.registry.probe import _n_ctx_train_from_metadata

    assert _n_ctx_train_from_metadata(None) == 0
    assert _n_ctx_train_from_metadata({}) == 0
    assert _n_ctx_train_from_metadata({"llama.context_length": "not-a-number"}) == 0
    assert _n_ctx_train_from_metadata({"llama.context_length": "0"}) == 0


def _make_multimodal_store(root: Path) -> Path:
    """A manifest carrying a projector layer, the shape ``qwen3.8:27b`` ships."""
    blobs = root / "blobs"
    manifests = root / "manifests" / "registry.ollama.ai" / "library" / "seer"
    blobs.mkdir(parents=True)
    manifests.mkdir(parents=True)

    model_digest, _ = _write_blob(blobs, b"FAKE GGUF BYTES")
    projector_digest, projector_path = _write_blob(blobs, b"FAKE PROJECTOR BYTES")

    manifest = {
        "schemaVersion": 2,
        "layers": [
            {"mediaType": "application/vnd.ollama.image.model", "digest": model_digest, "size": 15},
            {
                "mediaType": "application/vnd.ollama.image.projector",
                "digest": projector_digest,
                "size": 20,
            },
        ],
    }
    (manifests / "27b").write_text(json.dumps(manifest))
    return projector_path


def test_projector_layer_is_surfaced_not_dropped(tmp_path: Path) -> None:
    """A multimodal manifest must not look identical to a text-only one.

    The in-process llama.cpp adapter cannot use an Ollama projector, so a
    descriptor routed down the gguf path serves such a model text-only with
    no error. Recording the projector keeps that loss visible instead of
    letting vision disappear silently.
    """
    projector_path = _make_multimodal_store(tmp_path)
    registry = OllamaRegistry(tmp_path)

    desc = registry.get("seer:27b")
    assert desc is not None
    assert desc.params.get("projector_path") == str(projector_path)


def test_text_only_manifest_records_no_projector(tmp_path: Path) -> None:
    _make_store(tmp_path)
    desc = OllamaRegistry(tmp_path).get("demo:1b")
    assert desc is not None
    assert "projector_path" not in desc.params
    # The params blob must still survive alongside the new key.
    assert desc.params.get("temperature") == 0.5
