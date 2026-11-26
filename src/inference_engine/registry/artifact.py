"""Immutable artifact references reported by the registry that resolved a model."""

from __future__ import annotations

import re
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

SHA256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
HF_REVISION = Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
REPO_PATTERN = r"^[A-Za-z0-9_][A-Za-z0-9._-]*(/[A-Za-z0-9_][A-Za-z0-9._-]*)?$"
QUANT_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._+ -]{0,63}$"


class ArtifactIdentity(BaseModel):
    """Discovery evidence. An operator declaration is distinct from upstream metadata."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    source: Literal[
        "ollama_tags", "ollama_manifest", "hf_snapshot", "mlx_config",
        "docker_model_runner", "operator_config",
    ]
    manifest_sha256: SHA256 | None = None
    weight_sha256: SHA256 | None = None
    model_sha256: SHA256 | None = None
    repo_id: Annotated[str, Field(pattern=REPO_PATTERN, max_length=97)] | None = None
    revision: HF_REVISION | None = None
    quantization: Annotated[str, Field(pattern=QUANT_PATTERN)] | None = None

    @model_validator(mode="after")
    def validate_evidence(self) -> Self:
        if (self.repo_id is None) != (self.revision is None):
            raise ValueError("A repository identity requires both repo_id and immutable revision")
        if not any((
            self.manifest_sha256, self.weight_sha256, self.model_sha256,
            self.revision, self.quantization,
        )):
            raise ValueError("Artifact identity requires a digest, pinned repository, or quantization")
        return self


def ollama_sha256(value: object) -> str | None:
    if isinstance(value, str):
        digest = value.removeprefix("sha256:").lower()
        if re.fullmatch(r"[0-9a-f]{64}", digest):
            return digest
    return None


def reported_quantization(value: object) -> str | None:
    if isinstance(value, str) and re.fullmatch(QUANT_PATTERN, value):
        if value.lower() not in {"unknown", "unspecified", "none"}:
            return value
    return None


def configured_artifact(entry: dict) -> ArtifactIdentity | None:
    raw = entry.get("artifact_identity")
    if raw is None:
        return None
    if not isinstance(raw, dict) or "source" in raw:
        raise ValueError("artifact_identity must be an object without a source override")
    return ArtifactIdentity.model_validate({**raw, "source": "operator_config"})
