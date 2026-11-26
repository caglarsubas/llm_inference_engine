"""Curated upstream releases, independent of a deployment's serving registry."""

from datetime import date
from functools import cache
from importlib.resources import files
from typing import Annotated, Literal

from pydantic import AnyUrl, BaseModel, ConfigDict, Field, UrlConstraints, computed_field


class QualityEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    use_case: str = Field(min_length=1)
    result: str = Field(min_length=1)
    limitation: str = Field(min_length=1)
    source: Annotated[AnyUrl, UrlConstraints(allowed_schemes=["https"], host_required=True)]
    provenance: Literal["publisher", "independent"]


class VariantFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1)
    size_bytes: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    role: str


class PortfolioVariant(BaseModel):
    """One downloadable build of a parent checkpoint, including required companions."""

    model_config = ConfigDict(extra="forbid")

    repo_id: str
    revision: str = Field(pattern=r"^[a-f0-9]{40}$")
    license: str
    format: Literal["gguf", "bitsandbytes", "mlx", "safetensors", "strata", "edge0", "onnx"]
    quantization: str
    runtime_note: str
    files: list[VariantFile] = Field(min_length=1)
    downloads_30d: int | None = Field(default=None, ge=0)

    @computed_field
    @property
    def size_bytes(self) -> int:
        return sum(artifact.size_bytes for artifact in self.files)

    @computed_field
    @property
    def source(self) -> str:
        return f"https://huggingface.co/{self.repo_id}/tree/{self.revision}"


class PortfolioEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    revision: str = Field(pattern=r"^[a-f0-9]{40}$")
    family: str
    release: str = Field(pattern=r"^\d{4}-(0[1-9]|1[0-2])$")
    tasks: list[str] = Field(min_length=1)
    inputs: list[str] = Field(min_length=1)
    outputs: list[str] = Field(min_length=1)
    license: str
    size: str
    architecture: str
    description: str
    integration: Literal["ollama", "chat-upstream", "worker-required", "standalone-asset"]
    ollama_tag: str | None = None
    artifact_files: list[str] | None = None
    sources: list[str] = Field(default_factory=list)
    downloads_30d: int | None = Field(default=None, ge=0)
    quality_evidence: QualityEvidence | None = None
    variants: list[PortfolioVariant] = Field(default_factory=list)

    @computed_field
    @property
    def model_card(self) -> str:
        return f"https://huggingface.co/{self.id}/blob/{self.revision}/README.md"


class ModelPortfolio(BaseModel):
    schema_version: Literal[1]
    reviewed_on: date
    popularity_observed_on: date | None = None
    entries: list[PortfolioEntry]


@cache
def load_portfolio() -> ModelPortfolio:
    resource = files("inference_engine").joinpath("model_portfolio.json")
    return ModelPortfolio.model_validate_json(resource.read_bytes())
