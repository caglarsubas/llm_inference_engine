"""Eval API schemas — request, response, verdict envelope."""

from __future__ import annotations

from string import Formatter
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# Tenant rubric names: lowercase, so a name reads the same in a URL, a span and
# a file. Verdict keys follow Python identifiers because judges are asked for
# JSON objects and the keys land in span attributes.
RUBRIC_NAME_PATTERN = r"^[a-z][a-z0-9_]{1,63}$"
VERDICT_KEY_PATTERN = r"^[A-Za-z_][A-Za-z0-9_]{0,63}$"

# The only markers a template may use; see ``RubricSpec.user_prompt_template``.
TEMPLATE_PLACEHOLDERS = frozenset({"prompt", "response", "expected", "response_b"})

RubricSource = Literal["builtin", "tenant"]


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
    # and the response so Planeon can correlate evals back to candidate signals.
    candidate_model: str | None = None
    candidate_completion_id: str | None = None
    # Pairwise: identifies the second candidate's chat completion id for joining.
    candidate_b_completion_id: str | None = None

    # Determinism knob; passed straight to the judge model. Repeat ``i`` of
    # ``n`` runs with ``seed + i``.
    seed: int | None = 0

    # Repeated judgments. Each repeat is a separate judge call; the response
    # carries every verdict and how far they agree. Repeats need a temperature
    # above 0, since greedy decoding would return the same verdict n times.
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    n: int = Field(default=1, ge=1, le=8)

    @model_validator(mode="after")
    def _repeats_need_temperature(self) -> "EvalRequest":
        if self.n > 1 and self.temperature == 0.0:
            raise ValueError(
                "n > 1 needs temperature > 0: at temperature 0 every repeat "
                "returns the same verdict"
            )
        return self


class Verdict(BaseModel):
    """The judge's structured output, normalised."""

    score: float = Field(..., description="Primary numeric signal (rubric-defined extraction).")
    parsed: dict[str, Any] = Field(
        default_factory=dict,
        description="Validated structured fields the judge returned.",
    )
    raw: str = Field(..., description="The judge model's full text response.")
    parse_status: Literal["clean", "repaired", "truncated", "failed"] = "clean"


class RepeatSummary(BaseModel):
    """How far ``n`` repeated verdicts agree. Verdicts that failed to parse
    count in ``n`` but not in the statistics, since their 0 score is a
    placeholder rather than a judgment."""

    n: int
    parsed: int
    mean: float | None = None
    stdev: float | None = Field(default=None, description="Population standard deviation.")
    min: float | None = None
    max: float | None = None
    agreement: float | None = Field(
        default=None,
        description="Share of parsed verdicts giving the most common score.",
    )


class EvalResponse(BaseModel):
    id: str
    object: Literal["eval"] = "eval"
    created: int
    rubric: str
    rubric_source: RubricSource = "builtin"
    # Set for tenant rubrics: identifies the exact definition that judged.
    rubric_digest: str | None = None
    judge_model: str  # the judge that actually ran
    # Set when a resident interchangeable judge ran instead of the one asked
    # for (``MODEL_SUBSTITUTION_GROUPS``); names the one asked for.
    substituted_from_model: str | None = None
    substitution_reason: str | None = None
    candidate_model: str | None = None
    candidate_completion_id: str | None = None
    temperature: float = 0.0
    # The first repeat's verdict; with n == 1 the only one.
    verdict: Verdict
    verdicts: list[Verdict] = Field(default_factory=list)
    repeats: RepeatSummary | None = None
    duration_ms: float


class RubricInfo(BaseModel):
    name: str
    description: str
    requires_expected: bool
    expected_keys: list[str]
    pairwise: bool = False
    source: RubricSource = "builtin"
    digest: str | None = None


class RubricList(BaseModel):
    object: Literal["list"] = "list"
    data: list[RubricInfo]


# ---------------------------------------------------------------------------
# Tenant rubrics — declarative, so nothing a caller registers is executed
# ---------------------------------------------------------------------------

_Finite = Annotated[float, Field(allow_inf_nan=False)]
_VerdictKey = Annotated[str, Field(pattern=VERDICT_KEY_PATTERN)]


class NumberScore(BaseModel):
    """Score is the number under ``key``. With a scale, it must fall inside
    ``[min, max]`` and is normalised to 0-1; outside it the verdict fails."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["number"]
    key: _VerdictKey
    min: _Finite | None = None
    max: _Finite | None = None

    @model_validator(mode="after")
    def _scale(self) -> "NumberScore":
        if (self.min is None) != (self.max is None):
            raise ValueError("a number score needs both min and max, or neither")
        if self.min is not None and self.max is not None and self.min >= self.max:
            raise ValueError("a number score needs min < max")
        return self


class BooleanScore(BaseModel):
    """Score is 1.0 when ``key`` is true and 0.0 when it is false."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["boolean"]
    key: _VerdictKey


class ChoiceScore(BaseModel):
    """Score is the value mapped to the label under ``key``. An unlisted label
    fails the verdict rather than inheriting a default."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["choice"]
    key: _VerdictKey
    values: dict[Annotated[str, Field(min_length=1, max_length=64)], _Finite] = Field(
        min_length=1, max_length=32
    )


ScoreRule = Annotated[NumberScore | BooleanScore | ChoiceScore, Field(discriminator="kind")]


def template_fields(template: str) -> set[str]:
    """The placeholders ``template`` uses. Anything else — an unknown name, a
    positional ``{}``, attribute access, a format spec — is refused, and literal
    braces must be doubled."""
    try:
        parts = list(Formatter().parse(template))
    except ValueError as exc:
        raise ValueError(f"user_prompt_template: {exc}; double literal braces") from exc
    fields: set[str] = set()
    for _literal, field, spec, conversion in parts:
        if field is None:
            continue
        if field not in TEMPLATE_PLACEHOLDERS or spec or conversion:
            allowed = ", ".join(f"{{{name}}}" for name in sorted(TEMPLATE_PLACEHOLDERS))
            raise ValueError(
                f"user_prompt_template: unsupported placeholder {{{field}}}; "
                f"use only {allowed} and double literal braces"
            )
        fields.add(field)
    return fields


class RubricDefinition(BaseModel):
    """A rubric a tenant registers through ``POST /v1/evals/rubrics``."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., pattern=RUBRIC_NAME_PATTERN)
    description: str = Field(default="", max_length=500)
    # Sent to the judge as written: it is not a template.
    system_prompt: str = Field(..., min_length=1, max_length=8000)
    user_prompt_template: str = Field(..., min_length=1, max_length=8000)
    expected_keys: list[_VerdictKey] = Field(..., min_length=1, max_length=16)
    score: ScoreRule
    requires_expected: bool = False
    pairwise: bool = False

    @model_validator(mode="after")
    def _consistent(self) -> "RubricDefinition":
        if len(set(self.expected_keys)) != len(self.expected_keys):
            raise ValueError("expected_keys must be unique")
        if self.score.key not in self.expected_keys:
            raise ValueError(f"score.key {self.score.key!r} must be one of expected_keys")
        fields = template_fields(self.user_prompt_template)
        if "response" not in fields:
            raise ValueError("user_prompt_template must include {response}")
        if self.pairwise and "response_b" not in fields:
            raise ValueError("a pairwise rubric's template must include {response_b}")
        if self.requires_expected and "expected" not in fields:
            raise ValueError("a rubric that requires expected must include {expected}")
        return self


class RubricDetail(RubricInfo):
    """One rubric in full. ``score`` and the timestamps are set for tenant
    rubrics only; a built-in's score rule is code."""

    system_prompt: str
    user_prompt_template: str
    score: ScoreRule | None = None
    created_at: int | None = None
    updated_at: int | None = None


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
