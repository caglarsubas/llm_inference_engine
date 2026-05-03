"""CompositeRegistry — merge ordering + lookup."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from inference_engine.registry import CompositeRegistry, ModelDescriptor


def _desc(name: str, tag: str, fmt: str) -> ModelDescriptor:
    return ModelDescriptor(
        name=name,
        tag=tag,
        namespace="ns",
        registry="reg",
        model_path=Path(f"/tmp/{name}-{tag}"),
        format=fmt,
        size_bytes=1,
    )


@dataclass
class _StaticSource:
    items: list[ModelDescriptor]

    def list_models(self) -> list[ModelDescriptor]:
        return list(self.items)

    def get(self, name_with_tag: str) -> ModelDescriptor | None:
        for d in self.items:
            if d.qualified_name == name_with_tag:
                return d
        return None


def test_merges_distinct_models() -> None:
    a = _desc("alpha", "1", "gguf")
    b = _desc("beta", "1", "mlx")
    composite = CompositeRegistry([_StaticSource([a]), _StaticSource([b])])
    names = [m.qualified_name for m in composite.list_models()]
    assert names == ["alpha:1", "beta:1"]


def test_first_source_wins_on_collision() -> None:
    mlx_first = _desc("twin", "1", "mlx")
    gguf_second = _desc("twin", "1", "gguf")
    composite = CompositeRegistry([_StaticSource([mlx_first]), _StaticSource([gguf_second])])
    found = composite.get("twin:1")
    assert found is not None
    assert found.format == "mlx"


def test_get_falls_through_to_later_source() -> None:
    a = _desc("alpha", "1", "mlx")
    b = _desc("beta", "1", "gguf")
    composite = CompositeRegistry([_StaticSource([a]), _StaticSource([b])])
    assert composite.get("beta:1") is b
    assert composite.get("ghost") is None


def test_empty_sources_rejected() -> None:
    with pytest.raises(ValueError):
        CompositeRegistry([])
