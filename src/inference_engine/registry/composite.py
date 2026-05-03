"""Compose multiple model sources behind a single registry interface.

The order of `sources` matters: when two registries claim the same qualified
name, the **earlier** source wins. This is the seam for "prefer MLX over GGUF
on Apple Silicon" — pass MLX first.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from .ollama import ModelDescriptor


class _RegistrySource(Protocol):
    def list_models(self) -> list[ModelDescriptor]: ...
    def get(self, name_with_tag: str) -> ModelDescriptor | None: ...


class CompositeRegistry:
    def __init__(self, sources: Sequence[_RegistrySource]) -> None:
        if not sources:
            raise ValueError("CompositeRegistry requires at least one source")
        self._sources = list(sources)

    def list_models(self) -> list[ModelDescriptor]:
        seen: dict[str, ModelDescriptor] = {}
        for source in self._sources:
            for desc in source.list_models():
                # First source wins for a given qualified_name.
                seen.setdefault(desc.qualified_name, desc)
        return sorted(seen.values(), key=lambda d: d.qualified_name)

    def get(self, name_with_tag: str) -> ModelDescriptor | None:
        for source in self._sources:
            desc = source.get(name_with_tag)
            if desc is not None:
                return desc
        return None
