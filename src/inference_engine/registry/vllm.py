"""Discover vLLM-served models from a JSON config file.

Unlike the GGUF and MLX registries (which scan the filesystem), vLLM models
live behind an HTTP endpoint on a separate process. The mapping from "engine
model id" to "vLLM endpoint + model name" is a deployment concern, so the
registry just reads it from a config file written by the operator.

File format (a JSON array, one entry per vLLM-served model):

    [
      {
        "name": "llama-3.2-1b-instruct",
        "tag": "vllm",
        "endpoint": "http://vllm:8000",
        "model_id": "meta-llama/Llama-3.2-1B-Instruct",
        "chat_template_kwargs": {"enable_thinking": false},
        "size_bytes": 2400000000
      }
    ]

* ``name`` + ``tag`` form the qualified name clients use as ``model`` in
  ``/v1/chat/completions`` (e.g. ``llama-3.2-1b-instruct:vllm``). Tag is
  conventionally ``vllm`` so the format is visible from the id, but any
  string is accepted.
* ``endpoint`` is the base URL of the vLLM server (e.g.
  ``http://vllm:8000`` inside a Compose network).
* ``model_id`` is the name vLLM was started with (passed as ``--model``).
  Required; vLLM rejects requests whose ``model`` field doesn't match.
* ``size_bytes`` is informational — surfaced on ``/v1/models`` but not
  enforced by the manager's memory budget (vLLM does its own GPU memory
  management on a remote host).
* capability metadata such as ``modality``, ``supports_json_mode``, and
  ``supports_strict_image_json`` is copied through to ``/v1/models.data`` so
  benchmark clients can make no-leakage readiness decisions from the catalog.
* ``chat_template_kwargs`` is optional. When present, the adapter forwards it
  to OpenAI-compatible upstreams that need model-specific chat-template
  switches (for example Docker Model Runner llama.cpp models that need
  ``enable_thinking=false``).

Missing live config file → empty live registry, no vLLM models served. A separate
optional demanded config file can still report catalog-only candidates as
``demanded_not_configured`` under `/v1/models*.unavailable`. Malformed live or
demanded files fail loudly when read. `/v1/models` and chat resolution add one
more honesty gate: the configured upstream must respond to `/v1/models` with
this exact ``model_id`` before the descriptor is treated as loadable.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from .ollama import ModelDescriptor, SkippedManifest


class VLLMRegistry:
    def __init__(
        self,
        config_path: Path | str,
        *,
        demanded_config_path: Path | str | None = None,
        local_snapshot_root: Path | str | None = None,
    ) -> None:
        self.config_path = Path(config_path)
        self.demanded_config_path = (
            Path(demanded_config_path) if demanded_config_path is not None else None
        )
        self.local_snapshot_root = (
            Path(local_snapshot_root).expanduser()
            if local_snapshot_root is not None
            else None
        )
        self._cache: dict[str, ModelDescriptor] = {}

    def list_models(self) -> list[ModelDescriptor]:
        if not self.config_path.is_file():
            return []

        raw = self._read_entries(self.config_path, "vLLM")
        descriptors: list[ModelDescriptor] = []
        for i, entry in enumerate(raw):
            desc = self._parse_entry(i, entry)
            descriptors.append(desc)
            self._cache[desc.qualified_name] = desc
            self._cache[desc.fully_qualified_name] = desc

        return sorted(descriptors, key=lambda d: d.qualified_name)

    @staticmethod
    def _read_entries(path: Path, label: str) -> list[dict]:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise ValueError(
                f"{label} config file must be a JSON array, got {type(raw).__name__}"
            )
        return raw

    @staticmethod
    def _parse_entry(index: int, entry: dict) -> ModelDescriptor:
        if not isinstance(entry, dict):
            raise ValueError(f"vLLM entry {index} is not an object: {entry!r}")
        try:
            name = str(entry["name"])
            tag = str(entry.get("tag", "vllm"))
            endpoint = str(entry["endpoint"]).rstrip("/")
            model_id = str(entry["model_id"])
        except KeyError as exc:
            raise ValueError(f"vLLM entry {index} missing required field: {exc.args[0]}") from exc

        params = {
            "model_id": model_id,
            "provider": str(entry.get("provider", "vllm")),
            "supports_json_mode": True,
        }
        chat_template_kwargs = entry.get("chat_template_kwargs")
        if chat_template_kwargs is not None:
            if not isinstance(chat_template_kwargs, dict):
                raise ValueError(
                    f"vLLM entry {index} chat_template_kwargs must be an object"
                )
            params["chat_template_kwargs"] = chat_template_kwargs
        for field in (
            "family",
            "profile",
            "modality",
            "context_length",
            "max_image_size",
            "max_image_side_px",
            "max_image_pixels",
            "supports_json_mode",
            "supports_strict_image_json",
            "strict_image_json_status",
            "strict_image_json_checked_at",
            "strict_image_json_detail",
            "commercial_use",
            "benchmark_only",
            "parameter_count_b",
            "open_weight",
            "proprietary",
            "download_status",
            "local_snapshot_path",
        ):
            if field in entry:
                params[field] = entry[field]

        return ModelDescriptor(
            name=name,
            tag=tag,
            namespace="vllm",
            registry="local",
            # Synthetic path so the rest of the codebase (which reads
            # ``model_path`` for display) doesn't trip on None.
            model_path=Path(f"vllm://{endpoint}/{model_id}"),
            format="vllm",
            params=params,
            size_bytes=int(entry.get("size_bytes", 0)),
            endpoint=endpoint,
        )

    def list_skipped(self) -> list[SkippedManifest]:
        if self.demanded_config_path is None or not self.demanded_config_path.is_file():
            return []

        live_ids = {desc.qualified_name for desc in self.list_models()}
        raw = self._read_entries(self.demanded_config_path, "vLLM demanded")
        download_records = self._latest_download_records()
        skipped: list[SkippedManifest] = []
        for i, entry in enumerate(raw):
            desc = self._parse_entry(i, entry)
            if desc.qualified_name in live_ids:
                continue
            model_id = str(desc.params.get("model_id") or "")
            status = str(desc.params.get("strict_image_json_status") or "pending_config")
            strict_safe = desc.params.get("supports_strict_image_json")
            reason = "demanded_not_configured"
            snapshot_path = self._downloaded_snapshot_path(desc, download_records)
            if snapshot_path is not None:
                reason = "downloaded_but_not_served"
                desc = replace(
                    desc,
                    params={
                        **desc.params,
                        "download_status": "downloaded",
                        "local_snapshot_path": str(snapshot_path),
                    },
                )
                detail = (
                    f"local snapshot downloaded at {snapshot_path}; start an "
                    f"OpenAI-compatible upstream that advertises {model_id} from "
                    f"/v1/models, then copy or promote the entry into {self.config_path}; "
                    f"strict_image_json_status={status}; supports_strict_image_json={strict_safe}"
                )
            else:
                detail = (
                    f"catalog candidate from {self.demanded_config_path}; copy into "
                    f"{self.config_path} with a reachable endpoint before serving; "
                    f"upstream_model_id={model_id}; strict_image_json_status={status}; "
                    f"supports_strict_image_json={strict_safe}"
                )
            skipped.append(
                SkippedManifest(
                    qualified_name=desc.qualified_name,
                    manifest_path=str(self.demanded_config_path),
                    reason=reason,
                    detail=detail,
                    descriptor=desc,
                )
            )
        return sorted(skipped, key=lambda s: s.qualified_name)

    def _latest_download_records(self) -> dict[str, dict[str, Any]]:
        if self.local_snapshot_root is None:
            return {}
        status_path = self.local_snapshot_root / "download_status.jsonl"
        if not status_path.is_file():
            return {}

        records: dict[str, dict[str, Any]] = {}
        for line in status_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict):
                continue
            for key in (record.get("engine_id"), record.get("repo_id")):
                if key:
                    records[str(key)] = record
        return records

    def _downloaded_snapshot_path(
        self,
        desc: ModelDescriptor,
        download_records: dict[str, dict[str, Any]],
    ) -> Path | None:
        if self.local_snapshot_root is None:
            return None

        repo_id = str(desc.params.get("model_id") or "")
        if not repo_id:
            return None
        engine_id = desc.qualified_name
        record = download_records.get(engine_id) or download_records.get(repo_id)
        candidate = self._snapshot_path_from_record(record, repo_id)
        if record is not None and record.get("status") == "downloaded":
            return candidate if candidate.is_dir() else None
        if self._looks_like_complete_snapshot(candidate):
            return candidate
        return None

    def _snapshot_path_from_record(
        self,
        record: dict[str, Any] | None,
        repo_id: str,
    ) -> Path:
        if record is not None:
            for key in ("resolved_dir", "local_dir"):
                value = record.get(key)
                if value:
                    return Path(str(value)).expanduser()
        assert self.local_snapshot_root is not None
        return self.local_snapshot_root / repo_id.replace("/", "--")

    @staticmethod
    def _looks_like_complete_snapshot(path: Path) -> bool:
        if not path.is_dir() or not (path / "config.json").is_file():
            return False
        if any(path.glob("*.incomplete")) or any(path.glob("*.lock")):
            return False

        for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
            index_path = path / index_name
            if not index_path.is_file():
                continue
            try:
                payload = json.loads(index_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return False
            weight_map = payload.get("weight_map") if isinstance(payload, dict) else None
            if not isinstance(weight_map, dict) or not weight_map:
                return False
            required = {str(filename) for filename in weight_map.values()}
            return all((path / filename).is_file() and (path / filename).stat().st_size > 0 for filename in required)

        weight_globs = ("*.safetensors", "pytorch_model*.bin", "*.gguf")
        return any(
            candidate.is_file() and candidate.stat().st_size > 0
            for pattern in weight_globs
            for candidate in path.glob(pattern)
        )

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
