"""MLXRegistry — discover safetensors-based model directories."""

from __future__ import annotations

import json
from pathlib import Path

from inference_engine.registry import MLXRegistry


def _make_mlx_model(root: Path, name: str, *, weight_bytes: int = 1024) -> Path:
    model_dir = root / name
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text(json.dumps({"model_type": "llama"}))
    (model_dir / "tokenizer.json").write_text("{}")
    (model_dir / "tokenizer_config.json").write_text("{}")
    (model_dir / "model.safetensors").write_bytes(b"x" * weight_bytes)
    return model_dir


def test_lists_valid_mlx_directory(tmp_path: Path) -> None:
    _make_mlx_model(tmp_path, "DemoModel")
    reg = MLXRegistry(tmp_path)
    models = reg.list_models()
    assert len(models) == 1
    assert models[0].name == "DemoModel"
    assert models[0].tag == "mlx"
    assert models[0].format == "mlx"
    assert models[0].qualified_name == "DemoModel:mlx"
    assert models[0].size_bytes >= 1024


def test_strips_huggingface_models_prefix(tmp_path: Path) -> None:
    _make_mlx_model(tmp_path, "models--mlx-community--Llama-3.2-1B-Instruct-4bit")
    reg = MLXRegistry(tmp_path)
    models = reg.list_models()
    assert len(models) == 1
    assert models[0].name == "Llama-3.2-1B-Instruct-4bit"
    assert models[0].qualified_name == "Llama-3.2-1B-Instruct-4bit:mlx"


def test_skips_directories_missing_required_files(tmp_path: Path) -> None:
    incomplete = tmp_path / "Incomplete"
    incomplete.mkdir()
    (incomplete / "config.json").write_text("{}")
    # missing safetensors + tokenizer
    reg = MLXRegistry(tmp_path)
    assert reg.list_models() == []


def test_descends_through_parent_directories(tmp_path: Path) -> None:
    _make_mlx_model(tmp_path / "hub" / "mlx-community", "Phi-3-mini-4k-instruct-4bit")
    reg = MLXRegistry(tmp_path)
    models = reg.list_models()
    assert [m.name for m in models] == ["Phi-3-mini-4k-instruct-4bit"]


def test_get_resolves_short_and_qualified(tmp_path: Path) -> None:
    _make_mlx_model(tmp_path, "Foo")
    reg = MLXRegistry(tmp_path)
    assert reg.get("Foo:mlx") is not None
    assert reg.get("Foo") is not None
    assert reg.get("Bar") is None


def test_missing_root_returns_empty(tmp_path: Path) -> None:
    reg = MLXRegistry(tmp_path / "does-not-exist")
    assert reg.list_models() == []


def test_aliased_hub_snapshot_has_pinned_repository_identity(tmp_path: Path) -> None:
    commit = "a" * 40
    snapshot = _make_mlx_model(tmp_path / "hub/models--org--real-build/snapshots", commit)
    (snapshot / "config.json").write_text('{"quantization":{"bits":4,"group_size":64}}')
    aliases = tmp_path / "served"
    aliases.mkdir()
    (aliases / "deployment-alias").symlink_to(snapshot, target_is_directory=True)
    registry = MLXRegistry(aliases)
    descriptor = registry.get("deployment-alias:mlx")
    assert descriptor.model_path == snapshot.resolve()
    identity = descriptor.artifact_identity
    assert identity.source == "hf_snapshot"
    assert (identity.repo_id, identity.revision, identity.quantization) == (
        "org/real-build", commit, "4-bit",
    )
    (aliases / "deployment-alias").unlink()
    assert registry.list_models() == []
    assert registry.get("deployment-alias:mlx") is None


def test_flat_mlx_model_reports_only_configured_quantization(tmp_path: Path) -> None:
    model = _make_mlx_model(tmp_path, "community-best-2bit")
    registry = MLXRegistry(tmp_path)
    assert registry.list_models()[0].artifact_identity is None
    (model / "config.json").write_text('{"quantization":{"bits":4,"head":{"bits":8}}}')
    identity = registry.list_models()[0].artifact_identity
    assert identity.source == "mlx_config"
    assert identity.quantization == "mixed-4-8-bit"
    assert identity.repo_id is None and identity.revision is None
