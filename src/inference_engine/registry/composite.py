"""Compose multiple model sources behind a single registry interface.

Source order matters
--------------------

When two registries claim the same qualified name, the **earlier** source
wins by default — that's the seam for "prefer MLX over GGUF on Apple
Silicon" (pass MLX first) and for "prefer in-process llama.cpp over the
ollama HTTP fallback" (pass the local Ollama registry before the HTTP
registry).

Probe-aware fallback
--------------------

The default ``get(qname)`` returns the first matching descriptor regardless
of whether the corresponding adapter could actually load it.  That's wrong
for the GGUF / Ollama-HTTP combo: a 2026 ``gemma4`` GGUF parses fine on
disk but llama-cpp-python rejects it at weight load, so we want to fall
through to the next source's descriptor for the same name.

``resolve(qname, accept)`` is the probe-aware lookup: it walks sources in
order, returns the first descriptor for which ``accept(desc)`` returns
``True``.  Callers pass ``accept = lambda d: probe(d).loadable`` for GGUFs
(non-GGUFs short-circuit to True inside the probe).  ``list_models()``
gains an analogous ``list_loadable(accept)`` that builds the merged view
the same way — so /v1/models and the ModelManager always agree on what's
reachable.

The original ``get()`` and ``list_models()`` keep their pre-probe
semantics so legacy callers (admin views, debug tooling) can still inspect
"every descriptor any source knows about" without filtering.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
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

    # ------------------------------------------------------------------
    # Pre-probe surface — returns whatever each source emits, first wins.
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Probe-aware surface — used by /v1/models and ModelManager so
    # unloadable descriptors fall through to the next source.
    # ------------------------------------------------------------------

    def resolve(
        self,
        name_with_tag: str,
        accept: Callable[[ModelDescriptor], bool],
    ) -> ModelDescriptor | None:
        """Return the first descriptor for ``name_with_tag`` accepted by ``accept``.

        ``accept`` is the per-descriptor predicate (typically the load
        probe).  We deliberately do not memoise here — the probe owns its
        own cache, and each registry source owns its own enumeration cache,
        so this loop is cheap.
        """
        for source in self._sources:
            desc = source.get(name_with_tag)
            if desc is None:
                continue
            if accept(desc):
                return desc
        return None

    def list_loadable(
        self,
        accept: Callable[[ModelDescriptor], bool],
    ) -> tuple[list[ModelDescriptor], list[ModelDescriptor]]:
        """Walk every source, partition into ``(loadable, rejected)``.

        For each ``qualified_name`` seen across any source, we keep the
        first descriptor accepted by ``accept`` in ``loadable``.  Names
        where every source's descriptor is rejected go to ``rejected``,
        carrying the *first* rejected descriptor (so callers can report a
        useful unavailability reason against a real source rather than
        "model not found").
        """
        # First pass: collect every (qname → ordered list of descriptors)
        # so we can apply accept() with full source-order context.
        per_name: dict[str, list[ModelDescriptor]] = {}
        for source in self._sources:
            for desc in source.list_models():
                per_name.setdefault(desc.qualified_name, []).append(desc)

        loadable: list[ModelDescriptor] = []
        rejected: list[ModelDescriptor] = []
        for qname, candidates in per_name.items():
            chosen: ModelDescriptor | None = None
            for desc in candidates:
                if accept(desc):
                    chosen = desc
                    break
            if chosen is not None:
                loadable.append(chosen)
            else:
                rejected.append(candidates[0])

        loadable.sort(key=lambda d: d.qualified_name)
        rejected.sort(key=lambda d: d.qualified_name)
        return loadable, rejected
