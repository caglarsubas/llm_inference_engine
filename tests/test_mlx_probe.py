"""MLXRuntimeProbe — keep unservable ``:mlx`` directories out of the catalog.

The in-process MLX adapter is text-only and built on the optional ``mlx-lm``
runtime. ``MLXRegistry`` happily reports any HF-shaped directory, so these
tests pin the two ways such a directory can still be unservable: the runtime
isn't installed, and the checkpoint is a VLM that belongs to the sidecar
workers in ``scripts/``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from inference_engine.registry import ModelDescriptor
from inference_engine.registry.mlx_probe import MLXRuntimeProbe


def _descriptor(model_dir: Path, *, name: str = "DemoModel") -> ModelDescriptor:
    return ModelDescriptor(
        name=name,
        tag="mlx",
        namespace="mlx-community",
        registry="huggingface.co",
        model_path=model_dir,
        format="mlx",
    )


def _checkpoint(root: Path, name: str, config: dict) -> Path:
    model_dir = root / name
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text(json.dumps(config))
    (model_dir / "tokenizer.json").write_text("{}")
    (model_dir / "model.safetensors").write_bytes(b"x" * 16)
    return model_dir


@pytest.fixture
def probe(monkeypatch: pytest.MonkeyPatch) -> MLXRuntimeProbe:
    """Probe with the runtime present and every architecture supported.

    Individual tests narrow this. Defaulting to "installed" keeps the suite
    meaningful on a machine that has never installed the ``mlx`` extra.
    """
    monkeypatch.setattr(MLXRuntimeProbe, "_runtime_installed", staticmethod(lambda: True))
    monkeypatch.setattr(MLXRuntimeProbe, "_architecture_supported", staticmethod(lambda _mt: True))
    return MLXRuntimeProbe()


def test_text_checkpoint_is_loadable(tmp_path: Path, probe: MLXRuntimeProbe) -> None:
    model_dir = _checkpoint(tmp_path, "Llama-3.2-1B-Instruct-4bit", {"model_type": "llama"})
    result = probe.probe(_descriptor(model_dir))
    assert result.loadable is True
    assert result.reason == ""
    assert result.model_type == "llama"


def test_missing_runtime_is_reported_not_loadable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(MLXRuntimeProbe, "_runtime_installed", staticmethod(lambda: False))
    model_dir = _checkpoint(tmp_path, "Llama-3.2-1B-Instruct-4bit", {"model_type": "llama"})

    result = MLXRuntimeProbe().probe(_descriptor(model_dir))

    assert result.loadable is False
    assert result.reason == "mlx_runtime_missing"
    assert "mlx" in result.detail


@pytest.mark.parametrize(
    ("name", "config"),
    [
        (
            "Molmo-7B-D-0924-4bit",
            {"model_type": "molmo", "architectures": ["MolmoForCausalLM"], "vision_config": {}},
        ),
        (
            "GLM-4.1V-9B-Thinking-5bit",
            {
                "model_type": "glm4v",
                "architectures": ["Glm4vForConditionalGeneration"],
                "vision_config": {},
            },
        ),
        # Architecture suffix alone is enough — no vision_config key present.
        (
            "SomeVLM",
            {"model_type": "somevlm", "architectures": ["SomeVLMForConditionalGeneration"]},
        ),
    ],
)
def test_vision_checkpoints_are_not_served_in_process(
    tmp_path: Path, probe: MLXRuntimeProbe, name: str, config: dict
) -> None:
    model_dir = _checkpoint(tmp_path, name, config)

    result = probe.probe(_descriptor(model_dir, name=name))

    assert result.loadable is False
    assert result.reason == "mlx_vision_unsupported"
    # The reason has to point somewhere useful, not just say "no".
    assert "serve_mlx_vlm_openai.py" in result.detail


def test_vision_verdict_does_not_depend_on_the_runtime_being_installed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Installing the ``mlx`` extra must not flip a VLM back to servable.

    Vision is checked before the runtime precisely so the operator is told to
    stand up the sidecar worker rather than to install a package that will
    never help.
    """
    monkeypatch.setattr(MLXRuntimeProbe, "_runtime_installed", staticmethod(lambda: False))
    model_dir = _checkpoint(
        tmp_path, "Molmo-7B-D-0924-4bit", {"model_type": "molmo", "vision_config": {}}
    )

    result = MLXRuntimeProbe().probe(_descriptor(model_dir))

    assert result.reason == "mlx_vision_unsupported"


def test_unsupported_architecture_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(MLXRuntimeProbe, "_runtime_installed", staticmethod(lambda: True))
    monkeypatch.setattr(MLXRuntimeProbe, "_architecture_supported", staticmethod(lambda _mt: False))
    model_dir = _checkpoint(tmp_path, "Exotic", {"model_type": "exotic_arch"})

    result = MLXRuntimeProbe().probe(_descriptor(model_dir))

    assert result.loadable is False
    assert result.reason == "mlx_architecture_unsupported"
    assert "exotic_arch" in result.detail


def test_missing_config_is_reported(tmp_path: Path, probe: MLXRuntimeProbe) -> None:
    model_dir = tmp_path / "Empty"
    model_dir.mkdir()

    result = probe.probe(_descriptor(model_dir))

    assert result.loadable is False
    assert result.reason == "mlx_config_unreadable"


def test_malformed_config_is_reported(tmp_path: Path, probe: MLXRuntimeProbe) -> None:
    model_dir = tmp_path / "Broken"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{not json")

    result = probe.probe(_descriptor(model_dir))

    assert result.loadable is False
    assert result.reason == "mlx_config_unreadable"


def test_non_mlx_descriptor_is_passed_through(tmp_path: Path, probe: MLXRuntimeProbe) -> None:
    """Other formats have their own probes; this one must not claim them."""
    desc = ModelDescriptor(
        name="qwen3.6",
        tag="27b",
        namespace="library",
        registry="registry.ollama.ai",
        model_path=tmp_path / "blob",
        format="gguf",
    )
    assert probe.probe(desc).loadable is True


def test_result_is_cached_per_config_stat(tmp_path: Path, probe: MLXRuntimeProbe) -> None:
    model_dir = _checkpoint(tmp_path, "Cached", {"model_type": "llama"})
    desc = _descriptor(model_dir)

    assert probe.probe(desc).loadable is True

    # Rewriting config.json to a VLM changes size/mtime, so the cache key
    # changes and the new verdict wins rather than the stale one sticking.
    (model_dir / "config.json").write_text(
        json.dumps({"model_type": "molmo", "vision_config": {"depth": 1}})
    )
    assert probe.probe(desc).reason == "mlx_vision_unsupported"
