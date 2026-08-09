"""The total-elapsed generation cap, and the line it must not cross.

The cap exists because `chat_completion_timeout_seconds` was applied as
`httpx.Timeout(...)`, whose read component is per-read rather than total
elapsed — so a steadily streaming response never tripped it and a generation
had no bound at all. That was not a slow-request problem: the scheduler lease
is held for the duration, a client that gives up does not stop the work, and
enough abandoned requests exhaust the tenant queue.

The line: cancelling an await only ends the work when the adapter is waiting on
something cancellable. These tests pin both sides of it.
"""

from __future__ import annotations

import asyncio

import pytest

from inference_engine.adapters.base import (
    GenerationParams,
    GenerationResult,
    GenerationTimeoutError,
    InferenceAdapter,
)
from inference_engine.api.chat import _generate_within_deadline
from inference_engine.config import settings


class _SlowAdapter(InferenceAdapter):
    backend_name = "slow"

    def __init__(self, *, cancellable: bool, seconds: float = 30.0) -> None:
        self.generation_is_cancellable = cancellable
        self._seconds = seconds
        self.completed = False

    @property
    def is_loaded(self) -> bool:
        return True

    @property
    def loaded_model(self):
        return None

    async def load(self, descriptor) -> None: ...
    async def unload(self) -> None: ...

    async def generate(self, messages, params, cancel=None) -> GenerationResult:
        await asyncio.sleep(self._seconds)
        self.completed = True
        return GenerationResult(
            text="done", finish_reason="stop", prompt_tokens=1, completion_tokens=1
        )

    async def stream(self, messages, params, cancel=None):  # pragma: no cover
        yield None


@pytest.fixture
def deadline(monkeypatch):
    def _set(seconds: float) -> None:
        monkeypatch.setattr(
            settings, "chat_completion_timeout_seconds", seconds, raising=False
        )

    return _set


@pytest.mark.asyncio
async def test_a_cancellable_generation_is_cut_off_at_the_deadline(deadline) -> None:
    deadline(0.05)
    adapter = _SlowAdapter(cancellable=True, seconds=5.0)

    with pytest.raises(GenerationTimeoutError) as caught:
        await _generate_within_deadline(adapter, [], GenerationParams(), "m")

    # The typed error the route already maps to a 504 — NOT the builtin
    # TimeoutError. An earlier version of this wrapper raised the builtin and
    # claimed in its docstring that "the route's existing 504 mapping applies
    # unchanged"; the route catches GenerationTimeoutError, so the claim was
    # false and the mapping would have been missed.
    assert caught.value.timeout_seconds == 0.05
    assert caught.value.backend == "slow"
    assert caught.value.model == "m"
    # And the work really stopped, which is what makes the lease release honest.
    assert adapter.completed is False


@pytest.mark.asyncio
async def test_a_non_cancellable_generation_is_left_alone(deadline) -> None:
    """THE LINE. Cancelling an await does not stop blocking native code.

    llama.cpp and MLX run generation in a worker thread: `asyncio.wait_for`
    would abandon the RESULT while the thread computed on, so the resource
    stays busy and the timeout would assert something that did not happen.
    `GenerationTimeoutError`'s docstring draws exactly this line — the route
    "only maps this typed error when an adapter can raise it honestly".

    So a non-cancellable backend keeps its previous behaviour, unbounded
    duration included. Fixing those needs interruptible native calls, not a
    lie at this layer.
    """
    deadline(0.05)
    adapter = _SlowAdapter(cancellable=False, seconds=0.2)

    result = await _generate_within_deadline(adapter, [], GenerationParams(), "m")

    assert result.text == "done"
    assert adapter.completed is True


@pytest.mark.asyncio
async def test_zero_disables_the_deadline(deadline) -> None:
    """0 already meant "no timeout" for the adapters; it must mean the same here."""
    deadline(0)
    adapter = _SlowAdapter(cancellable=True, seconds=0.05)

    result = await _generate_within_deadline(adapter, [], GenerationParams(), "m")

    assert result.text == "done"
    assert adapter.completed is True


@pytest.mark.asyncio
async def test_a_fast_generation_is_untouched(deadline) -> None:
    deadline(30.0)
    adapter = _SlowAdapter(cancellable=True, seconds=0.0)

    result = await _generate_within_deadline(adapter, [], GenerationParams(), "m")

    assert result.text == "done"


def test_only_socket_backed_adapters_claim_cancellability() -> None:
    """The flag is a promise about the implementation, so it is pinned here.

    Conservative by default: an adapter opts in only when cancelling its await
    genuinely closes the upstream request.
    """
    from inference_engine.adapters.llama_cpp import LlamaCppAdapter
    from inference_engine.adapters.ollama_http import OllamaHttpAdapter
    from inference_engine.adapters.vllm_adapter import VLLMAdapter

    assert InferenceAdapter.generation_is_cancellable is False
    assert OllamaHttpAdapter.generation_is_cancellable is True
    assert VLLMAdapter.generation_is_cancellable is True
    # Blocking native generation in a worker thread — cancellation abandons the
    # result, it does not stop the compute.
    assert LlamaCppAdapter.generation_is_cancellable is False
