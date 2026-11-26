"""Attach reviewed, workload-specific evaluations to the exact build a route serves."""

from __future__ import annotations

from datetime import UTC, datetime
from functools import cache
from importlib import resources
from pathlib import PurePosixPath
from typing import Annotated, Literal

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    StrictFloat,
    StrictInt,
    field_validator,
    model_validator,
)

from .registry.artifact import HF_REVISION, REPO_PATTERN, SHA256, ArtifactIdentity

_Name = Annotated[str, Field(min_length=1, pattern=r"^\S+$")]


class _EvidenceModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class _OllamaManifest(_EvidenceModel):
    # The complete manifest also identifies the template and any vision projector.
    manifest_sha256: SHA256


class _HubSnapshot(_EvidenceModel):
    repo_id: Annotated[str, Field(pattern=REPO_PATTERN, max_length=97)]
    revision: HF_REVISION


class _DockerModel(_EvidenceModel):
    model_sha256: SHA256


class EvaluationResult(_EvidenceModel):
    checked_at: AwareDatetime
    status: Literal["passed", "failed"]
    metrics: Annotated[dict[_Name, StrictInt | StrictFloat], Field(min_length=1)]
    evidence: str

    @field_validator("checked_at")
    @classmethod
    def _utc(cls, value: datetime) -> datetime:
        return value.astimezone(UTC)

    @field_validator("evidence")
    @classmethod
    def _repository_path(cls, value: str) -> str:
        path = PurePosixPath(value)
        if (
            not path.parts
            or path.is_absolute()
            or ".." in path.parts
            or "\\" in value
            or ":" in value
            or path.as_posix() != value
        ):
            raise ValueError("evidence must be a repository-relative file path")
        return value


class WorkloadEvaluation(_EvidenceModel):
    workload: str
    latest: EvaluationResult
    last_passing: EvaluationResult | None


class _Record(EvaluationResult):
    route_id: _Name
    backend: _Name
    artifact: _OllamaManifest | _HubSnapshot | _DockerModel
    workload: _Name

    def matches_build(self, identity: ArtifactIdentity) -> bool:
        if isinstance(self.artifact, _OllamaManifest):
            return (
                identity.source in {"ollama_tags", "ollama_manifest"}
                and identity.manifest_sha256 == self.artifact.manifest_sha256
            )
        if isinstance(self.artifact, _DockerModel):
            return (
                self.backend == "vllm"
                and identity.source == "docker_model_runner"
                and identity.model_sha256 == self.artifact.model_sha256
            )
        return (
            identity.source == "hf_snapshot"
            and identity.repo_id == self.artifact.repo_id
            and identity.revision == self.artifact.revision
        )

    def result(self) -> EvaluationResult:
        return EvaluationResult(
            checked_at=self.checked_at,
            status=self.status,
            metrics=self.metrics,
            evidence=self.evidence,
        )


class _EvidenceFile(_EvidenceModel):
    evaluations: tuple[_Record, ...]

    @model_validator(mode="after")
    def _unique_results(self) -> _EvidenceFile:
        seen = set()
        for row in self.evaluations:
            key = (row.route_id, row.backend, row.artifact, row.workload, row.checked_at)
            if key in seen:
                raise ValueError(
                    "duplicate evaluation for the same route, build, workload and time"
                )
            seen.add(key)
        return self


@cache
def load_evaluations() -> _EvidenceFile:
    """Validate the evidence shipped with this engine once per process."""
    source = resources.files(__package__).joinpath("model_evaluations.json")
    return _EvidenceFile.model_validate_json(source.read_bytes())


def evaluations_for(
    *, route_id: str, backend: str, artifact_identity: ArtifactIdentity | None
) -> list[WorkloadEvaluation]:
    """Return the latest result and last pass per workload for a verified build."""
    if artifact_identity is None:
        return []

    matches: dict[str, list[_Record]] = {}
    for row in load_evaluations().evaluations:
        if (
            row.route_id == route_id
            and row.backend == backend
            and row.matches_build(artifact_identity)
        ):
            matches.setdefault(row.workload, []).append(row)

    results = []
    for workload, records in sorted(matches.items()):
        records.sort(key=lambda row: row.checked_at, reverse=True)
        passed = next((row for row in records if row.status == "passed"), None)
        results.append(
            WorkloadEvaluation(
                workload=workload,
                latest=records[0].result(),
                last_passing=passed.result() if passed else None,
            )
        )
    return results
