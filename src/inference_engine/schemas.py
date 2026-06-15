"""OpenAI-compatible request/response schemas (chat completions subset).

Kept narrow on purpose — only the fields we actually serve. Add more as adapters grow.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import AliasChoices, BaseModel, Field, field_validator, model_validator


def _split_trace_value(value) -> list[str] | None:
    """Accept dotted trace fields as arrays or comma-separated strings."""
    if value is None:
        return None

    raw_items = value if isinstance(value, list | tuple | set) else [value]
    items: list[str] = []
    for raw in raw_items:
        for part in str(raw).split(","):
            item = part.strip()
            if item:
                items.append(item)
    return items or None


def _split_unique_trace_value(value) -> list[str] | None:
    items = _split_trace_value(value)
    if items is None:
        return None

    out: list[str] = []
    for item in items:
        if item not in out:
            out.append(item)
    return out


class ToolCallFunction(BaseModel):
    name: str
    # OpenAI sends arguments as a JSON-encoded string, not a parsed object.
    # Keep it as a string so the round-trip stays bit-exact for replay.
    arguments: str


class ToolCall(BaseModel):
    id: str
    type: Literal["function"] = "function"
    function: ToolCallFunction


class ChatTextContentPart(BaseModel):
    type: Literal["text"]
    text: str


class ChatImageUrl(BaseModel):
    url: str
    detail: Literal["auto", "low", "high"] | None = None


class ChatImageUrlContentPart(BaseModel):
    type: Literal["image_url"]
    image_url: ChatImageUrl


ChatContentPart = Annotated[
    ChatTextContentPart | ChatImageUrlContentPart,
    Field(discriminator="type"),
]
ChatContent = str | list[ChatContentPart]


def dump_chat_content(content: ChatContent | None):
    """Return OpenAI-compatible chat content for backend request bodies."""
    if isinstance(content, list):
        return [part.model_dump(exclude_none=True) for part in content]
    return content


def chat_content_text(content: ChatContent | None) -> str:
    """Extract only textual content from OpenAI chat content parts."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return "\n".join(part.text for part in content if isinstance(part, ChatTextContentPart))


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    # Content is optional — assistant messages with only tool_calls have null
    # content, and tool result messages obviously have content but in JSON or
    # plain text form. User messages can also carry OpenAI-style multimodal
    # content parts for VLM backends.
    content: ChatContent | None = None
    # OpenAI extension for reasoning models (o-series, DeepSeek-R1, Nemotron).
    reasoning_content: str | None = None
    # Set on assistant messages that include function/tool invocations.
    tool_calls: list[ToolCall] | None = None
    # Set on tool result messages — references the call this is replying to.
    tool_call_id: str | None = None
    # Optional friendly name used by some toolchains.
    name: str | None = None


class ToolDefinition(BaseModel):
    """OpenAI-style tool definition passed in by the client."""

    type: Literal["function"] = "function"
    function: dict  # {"name": str, "description": str, "parameters": <JSON schema>}


class AutoEvalSpec(BaseModel):
    """Opt-in auto-eval that runs rubrics against the assistant's response.

    * ``mode="blocking"`` (only valid when ``stream=false``): waits for evals to
      finish, returns verdicts inline on ``ChatCompletionResponse.evals``.
    * ``mode="background"``: returns the chat response immediately and runs
      evals on a fire-and-forget asyncio task; verdicts surface only as
      ``eval.run`` spans (joined back to the chat by candidate_completion_id).
    """

    rubrics: list[str] = Field(..., min_length=1, description="Rubric names, e.g. ['safety', 'helpfulness'].")
    judge_model: str | None = Field(default=None, description="Default judge model for any rubric not overridden in judge_models.")
    judge_models: dict[str, str] | None = Field(
        default=None,
        description=(
            "Per-rubric judge override. Use a fast/cheap model for high-volume rubrics "
            "(e.g. {'safety': 'llama3.2:1b'}) and a stronger one for accuracy-sensitive ones "
            "(e.g. {'correctness': 'llama3.2:3b'}) without forking the spec."
        ),
    )
    expected: str | None = Field(default=None, description="Reference answer — required for rubrics like 'correctness'.")
    mode: Literal["background", "blocking"] = "background"


class IntentMetadata(BaseModel):
    """Optional caller-supplied intent metadata."""

    labels: list[str] | None = None
    label_names: list[str] | None = None
    source: str | None = None
    preclassified: bool | None = None
    classifier_version: str | None = None

    @field_validator("labels", mode="before")
    @classmethod
    def _normalize_labels(cls, value):
        return _split_unique_trace_value(value)

    @field_validator("label_names", mode="before")
    @classmethod
    def _normalize_label_names(cls, value):
        return _split_trace_value(value)


class RequestMetadata(BaseModel):
    """Generic request metadata extensions."""

    intent: IntentMetadata | None = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: float | None = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float | None = Field(default=0.95, ge=0.0, le=1.0)
    top_k: int | None = Field(default=40, ge=0)
    max_tokens: int | None = Field(default=512, ge=1)
    stream: bool = False
    stop: list[str] | str | None = None
    seed: int | None = None
    response_format: dict | None = None  # {"type": "json_object"} accepted
    chat_template_kwargs: dict | None = Field(
        default=None,
        description=(
            "Optional OpenAI-compatible chat-template kwargs for upstream servers "
            "that expose model-specific template controls."
        ),
    )
    auto_eval: AutoEvalSpec | None = Field(
        default=None,
        description="Opt-in auto-judge. See AutoEvalSpec.",
    )
    # OpenAI-compatible tool calling. Passed straight through to the backend.
    tools: list[ToolDefinition] | None = None
    tool_choice: str | dict | None = None
    metadata: RequestMetadata | None = Field(
        default=None,
        description="Optional generic request metadata. `metadata.intent` is stamped on spans.",
    )
    # Top-level intent fields are kept for clients that cannot send nested
    # metadata. The preferred request shape is ``metadata.intent``.
    intent_labels: list[str] | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "intent_labels",
            "intent.labels",
        ),
    )
    intent_label_names: list[str] | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "intent_label_names",
            "intent.label_names",
        ),
    )
    intent_source: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "intent_source",
            "intent.source",
        ),
    )
    intent_preclassified: bool | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "intent_preclassified",
            "intent.preclassified",
        ),
    )
    intent_classifier_version: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "intent_classifier_version",
            "intent.classifier_version",
        ),
    )

    @field_validator("intent_labels", mode="before")
    @classmethod
    def _normalize_intent_labels(cls, value):
        return _split_unique_trace_value(value)

    @field_validator("intent_label_names", mode="before")
    @classmethod
    def _normalize_intent_label_names(cls, value):
        return _split_trace_value(value)

    @model_validator(mode="after")
    def _merge_metadata_intent(self):
        intent = self.metadata.intent if self.metadata else None
        if intent is None:
            return self

        if self.intent_labels is None:
            self.intent_labels = intent.labels
        if self.intent_label_names is None:
            self.intent_label_names = intent.label_names
        if self.intent_source is None:
            self.intent_source = intent.source
        if self.intent_preclassified is None:
            self.intent_preclassified = intent.preclassified
        if self.intent_classifier_version is None:
            self.intent_classifier_version = intent.classifier_version
        return self


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: Literal["stop", "length", "tool_calls"] | None = "stop"


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class AutoEvalResult(BaseModel):
    """One rubric's verdict, attached to a chat completion in blocking auto-eval mode."""

    rubric: str
    judge_model: str
    verdict: dict  # the Verdict pydantic model dumped to a plain dict
    duration_ms: float
    error: str | None = None  # populated if the judge call itself raised


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: Usage
    evals: list[AutoEvalResult] | None = None


class ToolCallFunctionDelta(BaseModel):
    """Per-chunk fragment of a streamed tool call's function metadata.

    OpenAI streams ``arguments`` as a sequence of partial strings — clients
    concatenate them per ``index``. ``name`` is typically present only in the
    first chunk for a given index.
    """

    name: str | None = None
    arguments: str | None = None


class ToolCallDelta(BaseModel):
    """One streamed tool-call fragment, keyed by ``index`` across chunks."""

    index: int
    id: str | None = None
    type: Literal["function"] | None = None
    function: ToolCallFunctionDelta | None = None


# streaming chunk
class ChatCompletionDelta(BaseModel):
    role: Literal["system", "user", "assistant", "tool"] | None = None
    content: str | None = None
    # OpenAI o-series / DeepSeek-R1 streaming reasoning channel. Mirrors the
    # ``content`` field but carries chain-of-thought text that clients should
    # render separately (or not at all) from the user-facing answer.
    reasoning_content: str | None = None
    tool_calls: list[ToolCallDelta] | None = None


class ChatCompletionChunkChoice(BaseModel):
    index: int = 0
    delta: ChatCompletionDelta
    finish_reason: Literal["stop", "length", "tool_calls"] | None = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChatCompletionChunkChoice]


# --- /v1/completions (legacy) ------------------------------------------------


class CompletionRequest(BaseModel):
    """OpenAI-compatible legacy completions request — bypasses chat templates.

    Use this when the chat-template wrapping ``/v1/chat/completions`` applies
    is wrong for your workload (custom prompt formats, base models, raw
    text-in-text-out generation).
    """

    model: str
    prompt: str | list[str]
    temperature: float | None = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float | None = Field(default=0.95, ge=0.0, le=1.0)
    top_k: int | None = Field(default=40, ge=0)
    max_tokens: int | None = Field(default=128, ge=1)
    stop: list[str] | str | None = None
    seed: int | None = None


class CompletionChoice(BaseModel):
    text: str
    index: int = 0
    finish_reason: Literal["stop", "length"] | None = "stop"


class CompletionResponse(BaseModel):
    id: str
    object: Literal["text_completion"] = "text_completion"
    created: int
    model: str
    choices: list[CompletionChoice]
    usage: Usage


# --- /v1/embeddings ---------------------------------------------------------


class EmbeddingRequest(BaseModel):
    model: str
    # OpenAI accepts a single string or a list of strings — we normalise to
    # list[str] in the route handler.
    input: str | list[str]
    encoding_format: Literal["float"] = "float"


class EmbeddingObject(BaseModel):
    object: Literal["embedding"] = "embedding"
    index: int
    embedding: list[float]


class EmbeddingResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[EmbeddingObject]
    model: str
    usage: Usage


# --- /v1/rerank --------------------------------------------------------------


class RerankRequest(BaseModel):
    """Cohere/Jina-shaped rerank — query + documents → relevance-ranked indices.

    Implemented via embedding cosine-similarity (no dedicated cross-encoder
    needed). Quality scales with the embedding model: drop a real embedding
    GGUF (bge / nomic / e5) into the model store for production retrieval.
    """

    model: str
    query: str
    documents: list[str] = Field(..., min_length=1)
    top_n: int | None = Field(default=None, ge=1, description="Truncate to top N results; None = return all.")
    return_documents: bool = Field(default=False, description="Echo document text alongside the score.")


class RerankResult(BaseModel):
    index: int
    relevance_score: float
    document: str | None = None


class RerankResponse(BaseModel):
    id: str
    object: Literal["rerank"] = "rerank"
    created: int
    model: str
    results: list[RerankResult]
    usage: Usage


class ModelInfo(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int = 0
    owned_by: str = "local"
    # extension fields (non-standard but useful)
    size_bytes: int = 0
    backend: str = "llama_cpp"
    format: str = "gguf"
    model_path: str | None = None
    # Capability hints for clients (model registry, UI badges).
    # ``reasoning``/``thinking`` mean the underlying model emits private chain
    # of thought; the engine strips it into a separate channel before clients
    # see it. ``tool_calling_mode`` reflects what the engine *delivers* —
    # always ``"native"`` for chat-capable adapters because vendor XML is
    # normalized server-side. Adapters with no tool plumbing at all surface
    # ``"unsupported"``.
    reasoning: bool | None = None
    thinking: bool | None = None
    thinking_level: Literal["low", "med", "high"] | None = None
    tool_calling_mode: Literal["native", "unsupported"] | None = None


class UnavailableModel(BaseModel):
    """A manifest the engine knows about but cannot serve.

    Two reason families:

    * **Registry skip** — the manifest was rejected at parse time
      (``no_local_model_layer`` for cloud-only entries, ``missing_blob``,
      ``unreadable_manifest``).  ``backend`` is ``"none"`` because no
      adapter was attempted.
    * **Load probe failure** — llama.cpp can't open the GGUF (typically
      because the architecture isn't supported by the bundled
      ``llama-cpp-python``).  ``reason`` carries the exception class name,
      ``detail`` carries the first line of its message.

    Surfaced in ``ModelList.unavailable`` so clients (and operators
    eyeballing ``/v1/models``) can see *why* a model is missing instead
    of getting a 500 on first chat call.
    """

    id: str
    reason: str
    detail: str = ""
    backend: str = "none"
    format: str = "gguf"


class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelInfo]
    # Empty for clients that pre-date this field — fully backwards-compatible
    # with the previous schema (additional pydantic field with a default).
    unavailable: list[UnavailableModel] = Field(default_factory=list)
