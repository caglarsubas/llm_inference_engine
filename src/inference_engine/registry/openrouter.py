"""Config-driven OpenRouter registry.

OpenRouter is deliberately opt-in: operators list only the large open-weight
models they want this engine to expose. The registry enforces the policy gate
from the config metadata before a descriptor can reach the manager:

* parameter count must be strictly greater than the configured minimum
* ``open_weight`` must be true
* ``proprietary`` must be explicitly false
* ``open_source`` may be omitted, but if present it cannot be false
"""

from __future__ import annotations

import json
from pathlib import Path

from .ollama import ModelDescriptor


class OpenRouterRegistry:
    def __init__(
        self,
        config_path: Path | str,
        *,
        default_endpoint: str,
        min_parameter_count_b: float,
    ) -> None:
        self.config_path = Path(config_path)
        self.default_endpoint = default_endpoint.rstrip("/")
        self.min_parameter_count_b = float(min_parameter_count_b)
        self._cache: dict[str, ModelDescriptor] = {}

    def list_models(self) -> list[ModelDescriptor]:
        if not self.config_path.is_file():
            return []

        raw = json.loads(self.config_path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise ValueError(
                f"OpenRouter config file must be a JSON array, got {type(raw).__name__}"
            )

        descriptors: list[ModelDescriptor] = []
        self._cache.clear()
        for i, entry in enumerate(raw):
            if not isinstance(entry, dict):
                raise ValueError(f"OpenRouter entry {i} is not an object: {entry!r}")
            desc = self._parse_entry(i, entry)
            descriptors.append(desc)
            self._cache[desc.qualified_name] = desc
            self._cache[desc.fully_qualified_name] = desc

        return sorted(descriptors, key=lambda d: d.qualified_name)

    def get(self, name_with_tag: str) -> ModelDescriptor | None:
        if not self._cache:
            self.list_models()
        if name_with_tag in self._cache:
            return self._cache[name_with_tag]
        if ":" not in name_with_tag:
            for desc in self._cache.values():
                if desc.name == name_with_tag:
                    return desc
        return None

    def _parse_entry(self, index: int, entry: dict) -> ModelDescriptor:
        try:
            name = str(entry["name"])
            model_id = str(entry["model_id"])
        except KeyError as exc:
            raise ValueError(
                f"OpenRouter entry {index} missing required field: {exc.args[0]}"
            ) from exc

        tag = str(entry.get("tag", "openrouter"))
        endpoint = str(entry.get("endpoint") or self.default_endpoint).rstrip("/")
        parameter_count_b = self._parameter_count(index, entry)
        self._enforce_policy(index, entry, parameter_count_b)

        params = {
            "provider": "openrouter",
            "model_id": model_id,
            "parameter_count_b": parameter_count_b,
            "open_weight": True,
            "open_source": bool(entry.get("open_source", True)),
            "proprietary": False,
            "request_key_source": "openrouter-api-key",
            "supports_json_mode": bool(entry.get("supports_json_mode", True)),
        }
        for field in (
            "family",
            "profile",
            "modality",
            "context_length",
            "max_image_size",
            "max_image_side_px",
            "max_image_pixels",
            "hugging_face_id",
            "openrouter_name",
            "commercial_use",
            "benchmark_only",
            "supports_strict_image_json",
            "strict_image_json_status",
            "strict_image_json_checked_at",
            "strict_image_json_detail",
        ):
            if field in entry:
                params[field] = entry[field]

        return ModelDescriptor(
            name=name,
            tag=tag,
            namespace="openrouter",
            registry="openrouter",
            model_path=Path(f"openrouter://{model_id}"),
            format="openrouter",
            params=params,
            size_bytes=int(entry.get("size_bytes", 0)),
            endpoint=endpoint,
        )

    @staticmethod
    def _parameter_count(index: int, entry: dict) -> float:
        raw = entry.get("parameter_count_b", entry.get("parameters_b"))
        if raw is None:
            raise ValueError(
                f"OpenRouter entry {index} missing required field: parameter_count_b"
            )
        try:
            return float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"OpenRouter entry {index} parameter_count_b must be numeric"
            ) from exc

    def _enforce_policy(self, index: int, entry: dict, parameter_count_b: float) -> None:
        if parameter_count_b <= self.min_parameter_count_b:
            raise ValueError(
                f"OpenRouter entry {index} parameter_count_b must be > "
                f"{self.min_parameter_count_b:g}"
            )
        if entry.get("open_weight") is not True:
            raise ValueError(f"OpenRouter entry {index} must set open_weight=true")
        if entry.get("open_source", True) is False:
            raise ValueError(f"OpenRouter entry {index} must not set open_source=false")
        if entry.get("proprietary") is not False:
            raise ValueError(f"OpenRouter entry {index} must set proprietary=false")


__all__ = ["OpenRouterRegistry"]
