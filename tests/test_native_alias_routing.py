"""A catalog row must describe the artifact an unqualified route can serve."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

from inference_engine.registry import CompositeRegistry, MLXRegistry, ModelDescriptor, OllamaRegistry


def _check_alias(source: MLXRegistry | OllamaRegistry, route: str) -> list[ModelDescriptor]:
    entries = source.list_models()
    assert len(entries) == 2
    assert entries[0].artifact_identity != entries[1].artifact_identity

    registry = CompositeRegistry([source])
    selected = registry.get(route)
    assert selected is not None
    assert registry.list_models() == [selected]
    assert registry.list_loadable(lambda _: True) == ([selected], [])
    assert registry.resolve(route, lambda _: True) == selected

    # A loadable shadowed build cannot rescue an unavailable route.
    def accepts_shadowed(descriptor: ModelDescriptor) -> bool:
        return descriptor.model_path != selected.model_path

    assert registry.resolve(route, accepts_shadowed) is None
    assert registry.list_loadable(accepts_shadowed) == ([], [selected])

    fallback = replace(selected, format="ollama_http", model_path=Path("http"), artifact_identity=None)
    downstream = Mock()
    downstream.list_models.return_value = [fallback]
    downstream.get.return_value = fallback
    with_fallback = CompositeRegistry([source, downstream])
    assert with_fallback.resolve(route, accepts_shadowed) == fallback
    assert with_fallback.list_loadable(accepts_shadowed) == ([fallback], [])
    return entries


def test_ollama_namespaces_sharing_a_route(tmp_path: Path) -> None:
    blobs = tmp_path / "blobs"
    blobs.mkdir()
    for namespace in ("team-a", "team-b"):
        data = namespace.encode()
        digest = hashlib.sha256(data).hexdigest()
        (blobs / f"sha256-{digest}").write_bytes(data)
        path = tmp_path / "manifests" / "registry.ollama.ai" / namespace / "demo"
        path.mkdir(parents=True)
        layer = {
            "mediaType": "application/vnd.ollama.image.model",
            "digest": f"sha256:{digest}",
            "size": len(data),
        }
        (path / "latest").write_text(json.dumps({"layers": [layer]}))

    source = OllamaRegistry(tmp_path)
    entries = _check_alias(source, "demo:latest")
    # Both original namespaces retain their explicit lookup.
    for entry in entries:
        assert source.get(entry.fully_qualified_name) == entry


def test_mlx_snapshot_aliases_sharing_a_route(tmp_path: Path) -> None:
    root = tmp_path / "served"
    for revision in ("a" * 40, "b" * 40):
        snapshot = tmp_path / "hub/models--org--demo/snapshots" / revision
        snapshot.mkdir(parents=True)
        (snapshot / "config.json").write_text("{}")
        (snapshot / "tokenizer.json").write_text("{}")
        (snapshot / "model.safetensors").write_bytes(b"weights")
        alias = root / revision / "demo"
        alias.parent.mkdir(parents=True)
        alias.symlink_to(snapshot, target_is_directory=True)

    _check_alias(MLXRegistry(root), "demo:mlx")
