"""Eval API schemas — request, response, verdict envelope."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class EvalRequest(BaseModel):
    rubric: str = Field(..., description="Rubric name, e.g. 'helpfulness'.")
    prompt: str = Field(..., description="The original user prompt the candidate was responding to.")
    response: str = Field(..., description="The candidate response to evaluate.")
    response_b: str | None = Field(
        default=None,
        description="Second candidate response — required for pairwise rubrics.",
    )
    expected: str | None = Field(
        default=None, description="Reference answer — required for rubrics like 'correctness'."
    )

    # Optional override; when omitted the engine uses settings.default_judge_model.
    judge_model: str | None = None

    # Provenance fields — not interpreted by the runner, just stamped onto spans
    # and the response so Prometa can correlate evals back to candidate signals.
    candidate_model: str | None = None
    candidate_completion_id: str | None = None
    # Pairwise: identifies the second candidate's chat completion id for joining.
    candidate_b_completion_id: str | None = None

    # Determinism knob; passed straight to the judge model.
    seed: int | None = 0


class Verdict(BaseModel):
    """The judge's structured output, normalised."""

    score: float = Field(..., description="Primary numeric signal (rubric-defined extraction).")
    parsed: dict[str, Any] = Field(
        default_factory=dict,
        description="Validated structured fields the judge returned.",
    )
    raw: str = Field(..., description="The judge model's full text response.")
    parse_status: Literal["clean", "repaired", "failed"] = "clean"


class EvalResponse(BaseModel):
    id: str
    object: Literal["eval"] = "eval"
    created: int
    rubric: str
    judge_model: str
    candidate_model: str | None = None
    candidate_completion_id: str | None = None
    verdict: Verdict
    duration_ms: float


class RubricInfo(BaseModel):
    name: str
    description: str
    requires_expected: bool
    expected_keys: list[str]
    pairwise: bool = False


class RubricList(BaseModel):
    object: Literal["list"] = "list"
    data: list[RubricInfo]


class PolicyMatchInfo(BaseModel):
    tenant: str
    model: str


class PolicyEntryInfo(BaseModel):
    name: str
    match: PolicyMatchInfo
    rubrics: list[str]
    mode: str
    judge_model: str | None = None


class PolicyList(BaseModel):
    object: Literal["list"] = "list"
    data: list[PolicyEntryInfo]
