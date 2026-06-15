from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_smoke_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "vlm_strict_json_smoke.py"
    spec = importlib.util.spec_from_file_location("vlm_strict_json_smoke", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_vlm_smoke_payload_uses_configured_token_budget() -> None:
    smoke = _load_smoke_module()

    payload = smoke._chat_payload(  # noqa: SLF001
        "gemma4:31b",
        "data:image/jpeg;base64,abc",
        max_tokens=768,
    )

    assert payload["max_tokens"] == 768
    assert payload["response_format"] == {"type": "json_object"}
