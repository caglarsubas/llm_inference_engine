"""Backend-agnostic adapter interface.

Every inference backend (llama.cpp, MLX-LM, vLLM, …) implements this same
contract. The API layer only knows about `InferenceAdapter` — never about a
specific backend. This is the seam that lets us swap engines without touching
the API or the registry.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass

from ..cancellation import Cancellation
from ..registry import ModelDescriptor
from ..schemas import ChatMessage


@dataclass
class GenerationParams:
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 40
    max_tokens: int = 512
    stop: list[str] | None = None
    seed: int | None = None
    json_mode: bool = False
    # OpenAI-compatible tool calling. Passed straight through to the backend
    # when supported. Adapters that don't support tools (mlx-lm) ignore these.
    tools: list[dict] | None = None
    tool_choice: str | dict | None = None


@dataclass
class StreamChunk:
    text: str
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    # OpenAI-style streamed tool-call deltas. Each entry has at minimum
    # ``index``; ``id``/``type``/``function.name`` typically only appear in the
    # first chunk for that index, while ``function.arguments`` arrives in
    # fragments that get concatenated per index. ``None`` when the chunk
    # carries only text.
    tool_call_deltas: list[dict] | None = None


@dataclass
class GenerationResult:
    text: str
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int
    # Set when the model returned OpenAI-style tool calls. None when not
    # supported by the backend or when the model didn't invoke any tools.
    tool_calls: list[dict] | None = None
    # Reasoning text stripped from ``text`` by the backend itself (e.g. when
    # llama.cpp grammars start parsing ``<think>`` natively). Most adapters
    # leave this ``None`` and let the chat-layer normalizer split it out.
    reasoning_content: str | None = None


@dataclass
class EmbeddingResult:
    """One embedding vector per input string, in request order."""

    embeddings: list[list[float]]
    prompt_tokens: int


class EmbeddingsNotSupportedError(NotImplementedError):
    """Raised by adapters that don't implement the embeddings API.

    The route catches this and returns HTTP 501 with the backend name so the
    caller can pick a backend that does (e.g. drop a llama.cpp embedding model
    next to the MLX chat model).
    """


class ContextLengthExceededError(Exception):
    """The request's prompt (plus any forced generation) does not fit the
    model's context window.

    Backends translate their native overflow error — e.g. llama.cpp's
    ``ValueError: Requested tokens (N) exceed context window of M`` — into this
    typed exception so the API layer can answer with a deterministic
    ``400 context_length_exceeded`` instead of an opaque ``500``. Clients
    (DeclarAI's tool loop, agentic dashboards) then branch on the error type
    rather than heuristically pattern-matching 500s after a big tool result.
    """

    def __init__(
        self,
        message: str = "",
        *,
        requested_tokens: int | None = None,
        context_window: int | None = None,
        backend: str = "",
    ) -> None:
        self.requested_tokens = requested_tokens
        self.context_window = context_window
        self.backend = backend
        if not message:
            if requested_tokens is not None and context_window is not None:
                message = (
                    f"This model's maximum context length is {context_window} tokens, "
                    f"but the request needs {requested_tokens}. Shorten the prompt or "
                    f"the prior tool results."
                )
            else:
                message = "The request exceeds the model's maximum context length."
        super().__init__(message)

    def error_detail(self) -> dict:
        """OpenAI-style error payload for the FastAPI ``detail`` field, so
        clients can read ``detail.type == 'context_length_exceeded'``."""
        detail: dict = {
            "message": str(self),
            "type": "context_length_exceeded",
            "code": "context_length_exceeded",
            "param": "messages",
        }
        if self.requested_tokens is not None:
            detail["requested_tokens"] = self.requested_tokens
        if self.context_window is not None:
            detail["context_window"] = self.context_window
        return detail


class InferenceAdapter(ABC):
    """Abstract base for all inference backends."""

    backend_name: str = "abstract"

    @abstractmethod
    async def load(self, descriptor: ModelDescriptor) -> None: ...

    @abstractmethod
    async def unload(self) -> None: ...

    @abstractmethod
    async def generate(
        self,
        messages: Iterable[ChatMessage],
        params: GenerationParams,
        cancel: Cancellation | None = None,
    ) -> GenerationResult: ...

    @abstractmethod
    async def stream(
        self,
        messages: Iterable[ChatMessage],
        params: GenerationParams,
        cancel: Cancellation | None = None,
    ) -> AsyncIterator[StreamChunk]: ...

    async def embed(self, inputs: list[str]) -> EmbeddingResult:
        """Compute embedding vectors for a batch of input strings.

        Adapters that don't implement embeddings raise ``EmbeddingsNotSupportedError``.
        The default implementation lives on the ABC (rather than being abstract)
        so subclasses don't need to override it just to declare unsupport.
        """
        raise EmbeddingsNotSupportedError(self.backend_name)

    async def complete(
        self,
        prompt: str,
        params: GenerationParams,
        cancel: Cancellation | None = None,
    ) -> GenerationResult:
        """Run text-in-text-out completion on a raw prompt.

        Bypasses the chat template ``generate()`` would apply — useful for
        non-chat workloads, prompt-template overrides, and the legacy
        ``/v1/completions`` endpoint. Default implementation routes through
        ``generate()`` with a single user-role message wrapping the prompt;
        adapters that have a true raw-completion API override this.
        """
        # Default fallback — most backends should override for a real raw path.
        return await self.generate(
            [ChatMessage(role="user", content=prompt)], params, cancel=cancel
        )

    @property
    @abstractmethod
    def is_loaded(self) -> bool: ...

    @property
    @abstractmethod
    def loaded_model(self) -> ModelDescriptor | None: ...
