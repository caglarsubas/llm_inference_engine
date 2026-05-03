"""MLXAdapter prompt-cache state machine.

Stubs out mlx_lm.{load,generate,models.cache,...} so we can verify the cache
resolution logic deterministically without paying the cost of loading a real
MLX model. These tests exercise the four states ``_resolve_cache`` produces:

* ``miss``     — first call, or cache empty
* ``full``     — incoming prompt is identical to cache_tokens
* ``trimmed``  — incoming prompt shares a prefix with cache_tokens
* ``disabled`` — settings.mlx_prefix_cache_enabled is False
"""

from __future__ import annotations

import asyncio
import sys
import types
from pathlib import Path

import pytest

from inference_engine.config import settings
from inference_engine.registry import ModelDescriptor


# ---------------------------------------------------------------------------
# Stub modules — installed via monkeypatching sys.modules so the lazy imports
# inside MLXAdapter pick them up. Each test gets a fresh adapter instance.
# ---------------------------------------------------------------------------


class _StubTokenizer:
    """Encode "abc..." → [0, 1, 2, ...] (positional id per char)."""

    def encode(self, text: str) -> list[int]:
        return [ord(c) for c in text]

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        # Concatenate role+content into a deterministic flat string.
        return "".join(f"{m['role'][0]}{m['content']}" for m in messages)


def _install_mlx_stubs(monkeypatch):
    """Replace mlx-lm's modules with deterministic stubs.

    Returns a dict of recorded calls so tests can assert the sequence the
    adapter took (which cache primitive it invoked, with what args)."""
    record: dict = {
        "make_calls": 0,
        "trim_calls": [],
        "can_trim": True,
        "generate_text": "ok",
        "stream_chunks": [("ok", 1000)],  # (text, token_id)
    }

    def _make_prompt_cache(model):
        record["make_calls"] += 1
        return {"id": record["make_calls"]}

    def _trim_prompt_cache(cache, n):
        record["trim_calls"].append((id(cache), n))

    def _can_trim_prompt_cache(cache):
        return record["can_trim"]

    fake_cache_mod = types.SimpleNamespace(
        make_prompt_cache=_make_prompt_cache,
        trim_prompt_cache=_trim_prompt_cache,
        can_trim_prompt_cache=_can_trim_prompt_cache,
    )
    monkeypatch.setitem(sys.modules, "mlx_lm.models.cache", fake_cache_mod)

    def _mlx_load(path):
        # Returns (model, tokenizer)
        return object(), _StubTokenizer()

    def _mlx_generate(**kwargs):
        return record["generate_text"]

    def _mlx_stream_generate(**kwargs):
        for text, token_id in record["stream_chunks"]:
            yield types.SimpleNamespace(text=text, token=token_id, finish_reason=None)

    fake_mlx_lm = types.SimpleNamespace(
        load=_mlx_load,
        generate=_mlx_generate,
        stream_generate=_mlx_stream_generate,
    )
    monkeypatch.setitem(sys.modules, "mlx_lm", fake_mlx_lm)

    # The adapter also tries `from mlx_lm.sample_utils import make_sampler`. Make
    # that import succeed cleanly with a sampler stand-in.
    fake_sample_utils = types.SimpleNamespace(make_sampler=lambda **kw: object())
    monkeypatch.setitem(sys.modules, "mlx_lm.sample_utils", fake_sample_utils)

    return record


def _desc() -> ModelDescriptor:
    return ModelDescriptor(
        name="m", tag="mlx", namespace="ns", registry="reg",
        model_path=Path("/tmp/m"), format="mlx", size_bytes=1024,
    )


@pytest.fixture
def adapter_kit(monkeypatch):
    """Install stubs + import a fresh adapter against them."""
    record = _install_mlx_stubs(monkeypatch)
    monkeypatch.setattr(settings, "mlx_prefix_cache_enabled", True)
    # Defer the import so it picks up the stubbed modules.
    from inference_engine.adapters.mlx_lm import MLXAdapter  # noqa: PLC0415

    adapter = MLXAdapter()
    return adapter, record


# ---------------------------------------------------------------------------
# State transitions
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_first_generate_is_a_miss(adapter_kit) -> None:
    adapter, rec = adapter_kit
    await adapter.load(_desc())
    # `load()` resets internal state; make_prompt_cache fires on first generate.
    assert rec["make_calls"] == 0

    from inference_engine.adapters.base import GenerationParams  # noqa: PLC0415
    from inference_engine.schemas import ChatMessage  # noqa: PLC0415

    await adapter.generate(
        [ChatMessage(role="user", content="hello")], GenerationParams()
    )

    assert rec["make_calls"] == 1
    assert adapter.prefix_cache_last_action == "miss"
    assert adapter.prefix_cache_last_overlap_tokens == 0
    # cache_tokens = prompt_tokens + generated_tokens
    assert adapter.prefix_cache_tokens > 0


@pytest.mark.asyncio
async def test_repeat_with_identical_prompt_is_full_reuse(adapter_kit) -> None:
    adapter, rec = adapter_kit
    await adapter.load(_desc())

    from inference_engine.adapters.base import GenerationParams  # noqa: PLC0415
    from inference_engine.schemas import ChatMessage  # noqa: PLC0415

    msgs = [ChatMessage(role="user", content="same prompt")]
    rec["generate_text"] = ""  # so the post-call slot.tokens equals the prompt exactly

    await adapter.generate(msgs, GenerationParams())
    cache_tokens_after_first = adapter.prefix_cache_tokens

    await adapter.generate(msgs, GenerationParams())

    # No additional make_prompt_cache call — we kept the same slot.
    assert rec["make_calls"] == 1
    assert rec["trim_calls"] == []
    assert adapter.prefix_cache_last_action == "full"
    assert adapter.prefix_cache_last_overlap_tokens == cache_tokens_after_first


@pytest.mark.asyncio
async def test_partial_overlap_at_capacity_triggers_trim(adapter_kit, monkeypatch) -> None:
    """At capacity, partial overlap evicts via trim — preserves the cache savings."""
    adapter, rec = adapter_kit
    monkeypatch.setattr(settings, "mlx_prefix_cache_max_slots", 1)
    await adapter.load(_desc())

    from inference_engine.adapters.base import GenerationParams  # noqa: PLC0415
    from inference_engine.schemas import ChatMessage  # noqa: PLC0415

    rec["generate_text"] = ""

    await adapter.generate([ChatMessage(role="user", content="hello world")], GenerationParams())
    cache_len_first = adapter.prefix_cache_tokens
    assert cache_len_first > 0

    # New prompt sharing only the "uhello " prefix (7 tokens). At capacity=1
    # so the policy is to trim, not to allocate a new slot.
    await adapter.generate([ChatMessage(role="user", content="hello there")], GenerationParams())

    assert rec["make_calls"] == 1  # same slot, just trimmed
    assert len(rec["trim_calls"]) == 1
    assert adapter.prefix_cache_last_action == "trimmed"
    assert 0 < adapter.prefix_cache_last_overlap_tokens < cache_len_first


@pytest.mark.asyncio
async def test_partial_overlap_under_capacity_creates_new_slot(adapter_kit, monkeypatch) -> None:
    """When capacity allows, partial overlap allocates a NEW slot rather than
    trimming — the original slot is preserved for future hits."""
    adapter, rec = adapter_kit
    monkeypatch.setattr(settings, "mlx_prefix_cache_max_slots", 4)
    await adapter.load(_desc())

    from inference_engine.adapters.base import GenerationParams  # noqa: PLC0415
    from inference_engine.schemas import ChatMessage  # noqa: PLC0415

    rec["generate_text"] = ""

    await adapter.generate(
        [ChatMessage(role="user", content="hello world")], GenerationParams()
    )
    await adapter.generate(
        [ChatMessage(role="user", content="hello there")], GenerationParams()
    )

    # Both slots coexist — original wasn't trimmed.
    assert rec["make_calls"] == 2
    assert rec["trim_calls"] == []
    assert adapter.prefix_cache_slots_used == 2
    # The new slot is "miss" from a slot's-eye view (its own KV cache is fresh).
    assert adapter.prefix_cache_last_action == "miss"

    # Re-asking for the original prompt is a full reuse — slot 1 was preserved.
    await adapter.generate(
        [ChatMessage(role="user", content="hello world")], GenerationParams()
    )
    assert adapter.prefix_cache_last_action == "full"


@pytest.mark.asyncio
async def test_disabled_via_config_skips_cache(adapter_kit, monkeypatch) -> None:
    adapter, rec = adapter_kit
    monkeypatch.setattr(settings, "mlx_prefix_cache_enabled", False)
    await adapter.load(_desc())

    from inference_engine.adapters.base import GenerationParams  # noqa: PLC0415
    from inference_engine.schemas import ChatMessage  # noqa: PLC0415

    await adapter.generate(
        [ChatMessage(role="user", content="hi")], GenerationParams()
    )
    await adapter.generate(
        [ChatMessage(role="user", content="hi")], GenerationParams()
    )

    assert rec["make_calls"] == 0
    assert adapter.prefix_cache_last_action == "disabled"
    assert adapter.prefix_cache_enabled is False  # introspection agrees


@pytest.mark.asyncio
async def test_disjoint_prompts_evict_to_fresh_slot(adapter_kit) -> None:
    """Two prompts with no shared prefix → second call gets a brand-new cache."""
    adapter, rec = adapter_kit
    await adapter.load(_desc())

    from inference_engine.adapters.base import GenerationParams  # noqa: PLC0415
    from inference_engine.schemas import ChatMessage  # noqa: PLC0415

    # apply_chat_template prepends role[0], so role="user" → 'u', role="system" → 's'.
    # Switching role changes the very first token, guaranteeing zero overlap.
    rec["generate_text"] = ""

    await adapter.generate(
        [ChatMessage(role="user", content="abc")], GenerationParams()
    )
    await adapter.generate(
        [ChatMessage(role="system", content="abc")], GenerationParams()
    )

    assert rec["make_calls"] == 2  # second call rebuilt the slot
    assert adapter.prefix_cache_last_action == "miss"
    assert adapter.prefix_cache_last_overlap_tokens == 0


@pytest.mark.asyncio
async def test_unload_resets_cache_state(adapter_kit) -> None:
    adapter, rec = adapter_kit
    await adapter.load(_desc())

    from inference_engine.adapters.base import GenerationParams  # noqa: PLC0415
    from inference_engine.schemas import ChatMessage  # noqa: PLC0415

    await adapter.generate(
        [ChatMessage(role="user", content="x")], GenerationParams()
    )
    assert adapter.prefix_cache_tokens > 0

    await adapter.unload()
    assert adapter.prefix_cache_enabled is False
    assert adapter.prefix_cache_tokens == 0
    assert adapter.prefix_cache_slots_used == 0
    assert adapter.prefix_cache_last_action == "none"


# ---------------------------------------------------------------------------
# Multi-slot LRU
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_two_distinct_prefixes_coexist_under_capacity(adapter_kit, monkeypatch) -> None:
    """With max_slots>=2, alternating prefix-disjoint calls should both hit warm
    on the second pass — the slots coexist instead of evicting each other."""
    adapter, rec = adapter_kit
    monkeypatch.setattr(settings, "mlx_prefix_cache_max_slots", 4)
    await adapter.load(_desc())

    from inference_engine.adapters.base import GenerationParams  # noqa: PLC0415
    from inference_engine.schemas import ChatMessage  # noqa: PLC0415

    rec["generate_text"] = ""

    # Two disjoint prefixes (different roles → different first tokens).
    msgs_a = [ChatMessage(role="user", content="alpha")]
    msgs_b = [ChatMessage(role="system", content="alpha")]

    await adapter.generate(msgs_a, GenerationParams())
    await adapter.generate(msgs_b, GenerationParams())
    assert adapter.prefix_cache_slots_used == 2

    # Re-hit both — neither was evicted, so both should be full reuse.
    await adapter.generate(msgs_a, GenerationParams())
    assert adapter.prefix_cache_last_action == "full"

    await adapter.generate(msgs_b, GenerationParams())
    assert adapter.prefix_cache_last_action == "full"

    # Two slots created in the cold pass; no further allocations on warm hits.
    assert rec["make_calls"] == 2


@pytest.mark.asyncio
async def test_max_slots_one_reproduces_single_slot_thrash(adapter_kit, monkeypatch) -> None:
    """Before this round, the cache held one slot only; capacity=1 is the same."""
    adapter, rec = adapter_kit
    monkeypatch.setattr(settings, "mlx_prefix_cache_max_slots", 1)
    await adapter.load(_desc())

    from inference_engine.adapters.base import GenerationParams  # noqa: PLC0415
    from inference_engine.schemas import ChatMessage  # noqa: PLC0415

    rec["generate_text"] = ""

    msgs_a = [ChatMessage(role="user", content="alpha")]
    msgs_b = [ChatMessage(role="system", content="alpha")]

    await adapter.generate(msgs_a, GenerationParams())
    await adapter.generate(msgs_b, GenerationParams())  # evicts a
    await adapter.generate(msgs_a, GenerationParams())  # miss again — thrash

    assert rec["make_calls"] == 3
    assert adapter.prefix_cache_last_action == "miss"
    assert adapter.prefix_cache_slots_used == 1


@pytest.mark.asyncio
async def test_lru_evicts_oldest_when_capacity_exceeded(adapter_kit, monkeypatch) -> None:
    """Three disjoint prefixes with max_slots=2 → first slot must be evicted."""
    adapter, rec = adapter_kit
    monkeypatch.setattr(settings, "mlx_prefix_cache_max_slots", 2)
    await adapter.load(_desc())

    from inference_engine.adapters.base import GenerationParams  # noqa: PLC0415
    from inference_engine.schemas import ChatMessage  # noqa: PLC0415

    rec["generate_text"] = ""

    # apply_chat_template prepends role[0], so all three first tokens differ.
    msgs_user = [ChatMessage(role="user", content="x")]
    msgs_sys = [ChatMessage(role="system", content="x")]
    msgs_asst = [ChatMessage(role="assistant", content="x")]

    await adapter.generate(msgs_user, GenerationParams())
    # Force last_used skew so the LRU pick is unambiguous.
    await asyncio.sleep(0.001)
    await adapter.generate(msgs_sys, GenerationParams())

    assert adapter.prefix_cache_slots_used == 2

    await asyncio.sleep(0.001)
    await adapter.generate(msgs_asst, GenerationParams())

    # Still 2 slots — user (oldest) was evicted to make room for assistant.
    assert adapter.prefix_cache_slots_used == 2

    # Re-asking for the user prefix is now a miss.
    await adapter.generate(msgs_user, GenerationParams())
    assert adapter.prefix_cache_last_action == "miss"


@pytest.mark.asyncio
async def test_lru_touches_keep_active_slot_alive(adapter_kit, monkeypatch) -> None:
    """A slot we keep hitting must NOT be the eviction victim."""
    adapter, rec = adapter_kit
    monkeypatch.setattr(settings, "mlx_prefix_cache_max_slots", 2)
    await adapter.load(_desc())

    from inference_engine.adapters.base import GenerationParams  # noqa: PLC0415
    from inference_engine.schemas import ChatMessage  # noqa: PLC0415

    rec["generate_text"] = ""

    msgs_user = [ChatMessage(role="user", content="x")]
    msgs_sys = [ChatMessage(role="system", content="x")]
    msgs_asst = [ChatMessage(role="assistant", content="x")]

    await adapter.generate(msgs_user, GenerationParams())
    await asyncio.sleep(0.001)
    await adapter.generate(msgs_sys, GenerationParams())

    # Touch the user slot so SYSTEM becomes oldest.
    await asyncio.sleep(0.001)
    await adapter.generate(msgs_user, GenerationParams())
    assert adapter.prefix_cache_last_action == "full"

    # New disjoint prefix — should evict SYSTEM (now oldest), keep USER.
    await asyncio.sleep(0.001)
    await adapter.generate(msgs_asst, GenerationParams())

    # USER should still be a hit (not evicted).
    await adapter.generate(msgs_user, GenerationParams())
    assert adapter.prefix_cache_last_action == "full"

    # SYSTEM should be a miss (evicted).
    await adapter.generate(msgs_sys, GenerationParams())
    assert adapter.prefix_cache_last_action == "miss"


@pytest.mark.asyncio
async def test_disjoint_prefix_in_third_role_creates_third_slot(
    adapter_kit, monkeypatch
) -> None:
    """Three roles with the same content but different prefix-tokens (role[0])
    fan out to three independent slots when capacity allows it."""
    adapter, rec = adapter_kit
    monkeypatch.setattr(settings, "mlx_prefix_cache_max_slots", 4)
    await adapter.load(_desc())

    from inference_engine.adapters.base import GenerationParams  # noqa: PLC0415
    from inference_engine.schemas import ChatMessage  # noqa: PLC0415

    rec["generate_text"] = ""

    for role in ("user", "system", "assistant"):
        await adapter.generate(
            [ChatMessage(role=role, content="hello")], GenerationParams()  # type: ignore[arg-type]
        )

    assert adapter.prefix_cache_slots_used == 3
    assert rec["make_calls"] == 3

    # Touching any of them is a full-reuse hit.
    await adapter.generate(
        [ChatMessage(role="system", content="hello")], GenerationParams()
    )
    assert adapter.prefix_cache_last_action == "full"
