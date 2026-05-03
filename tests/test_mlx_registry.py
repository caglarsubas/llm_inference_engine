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
