"""Adapter factory dispatch — picks the right adapter per descriptor format."""

from __future__ import annotations

from pathlib import Path

import pytest

from inference_engine.api.state import _build_adapter_for
from inference_engine.registry import ModelDescriptor


def _desc(fmt: str) -> ModelDescriptor:
    return ModelDescriptor(
        name="x",
        tag="1",
        namespace="ns",
        registry="reg",
        model_path=Path("/tmp/x"),
        format=fmt,
    )


def test_gguf_descriptor_yields_llama_cpp_adapter() -> None:
    adapter = _build_adapter_for(_desc("gguf"))
    assert adapter.backend_name == "llama_cpp"


def test_mlx_descriptor_yields_mlx_adapter() -> None:
    adapter = _build_adapter_for(_desc("mlx"))
    assert adapter.backend_name == "mlx"


def test_unknown_format_raises() -> None:
    with pytest.raises(ValueError):
        _build_adapter_for(_desc("onnx"))
