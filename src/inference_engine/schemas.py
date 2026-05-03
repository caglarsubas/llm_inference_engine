"""OpenAI-compatible request/response schemas (chat completions subset).

Kept narrow on purpose — only the fields we actually serve. Add more as adapters grow.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ToolCallFunction(BaseModel):
    name: str
    # OpenAI sends arguments as a JSON-encoded string, not a parsed object.
    # Keep it as a string so the round-trip stays bit-exact for replay.
    arguments: str


class ToolCall(BaseModel):
    id: str
    type: Literal["function"] = "function"
    function: ToolCallFunction


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    # Content is optional — assistant messages with only tool_calls have null
    # content, and tool result messages obviously have content but in JSON or
    # plain text form.
    content: str | None = None
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
    auto_eval: AutoEvalSpec | None = Field(
        default=None,
        description="Opt-in auto-judge. See AutoEvalSpec.",
    )
    # OpenAI-compatible tool calling. Passed straight through to the backend.
    tools: list[ToolDefinition] | None = None
    tool_choice: str | dict | None = None


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


class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelInfo]
