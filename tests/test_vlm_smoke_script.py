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


def test_expectation_errors_accepts_vehicle_damage_hit() -> None:
    smoke = _load_smoke_module()

    parsed = {
        "vehicle_visible": True,
        "damage_visible": True,
        "anomaly_score": 0.75,
        "confidence": 0.9,
        "reasons": ["rear damage is visible"],
    }

    assert smoke._expectation_errors(  # noqa: SLF001
        parsed,
        expect_vehicle_visible=True,
        expect_damage_visible=True,
        require_reasons=True,
        min_anomaly_score=0.1,
    ) == []


def test_expectation_errors_rejects_degenerate_vehicle_damage_response() -> None:
    smoke = _load_smoke_module()

    parsed = {
        "vehicle_visible": False,
        "damage_visible": False,
        "anomaly_score": 0.0,
        "confidence": 0.0,
        "reasons": [],
    }

    errors = smoke._expectation_errors(  # noqa: SLF001
        parsed,
        expect_vehicle_visible=True,
        expect_damage_visible=True,
        require_reasons=True,
        min_anomaly_score=0.1,
    )

    assert errors == [
        "vehicle_visible expected True, got False",
        "damage_visible expected True, got False",
        "reasons expected a non-empty list",
        "anomaly_score expected >= 0.1, got 0.0",
    ]
