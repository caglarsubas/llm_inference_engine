"""Server-side auto-eval policy — file loading, match resolution, request precedence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from inference_engine.evals import (
    PolicyEntry,
    PolicyMatch,
    PolicyRegistry,
    load_policy,
)
from inference_engine.schemas import AutoEvalSpec


# ---------------------------------------------------------------------------
# PolicyMatch — wildcard semantics
# ---------------------------------------------------------------------------


def test_match_wildcard_covers_anything() -> None:
    m = PolicyMatch(tenant="*", model="*")
    assert m.matches(tenant="agent-runtime", model="llama3.2:1b")
    assert m.matches(tenant="anonymous", model="anything:foo")


def test_match_specific_tenant_rejects_other() -> None:
    m = PolicyMatch(tenant="evals", model="*")
    assert m.matches(tenant="evals", model="x")
    assert not m.matches(tenant="dev", model="x")


def test_match_specific_model_rejects_other() -> None:
    m = PolicyMatch(tenant="*", model="llama3.2:1b")
    assert m.matches(tenant="any", model="llama3.2:1b")
    assert not m.matches(tenant="any", model="llama3.2:3b")


def test_match_both_specific() -> None:
    m = PolicyMatch(tenant="evals", model="llama3.2:3b")
    assert m.matches(tenant="evals", model="llama3.2:3b")
    assert not m.matches(tenant="evals", model="llama3.2:1b")
    assert not m.matches(tenant="dev", model="llama3.2:3b")


# ---------------------------------------------------------------------------
# PolicyRegistry.resolve — first-match-wins ordering
# ---------------------------------------------------------------------------


def _entry(name: str, *, tenant: str = "*", model: str = "*", rubrics: list[str] | None = None) -> PolicyEntry:
    return PolicyEntry(
        name=name,
        match=PolicyMatch(tenant=tenant, model=model),
        spec=AutoEvalSpec(rubrics=rubrics or ["safety"]),
    )


def test_empty_registry_resolves_to_none() -> None:
    reg = PolicyRegistry([])
    assert reg.resolve(tenant="x", model="y") is None
    assert len(reg) == 0


def test_first_matching_entry_wins() -> None:
    reg = PolicyRegistry([
        _entry("specific", tenant="evals", model="llama3.2:3b", rubrics=["correctness"]),
        _entry("baseline", tenant="*", model="*", rubrics=["safety"]),
    ])
    found = reg.resolve(tenant="evals", model="llama3.2:3b")
    assert found is not None
    assert found.name == "specific"
    assert found.spec.rubrics == ["correctness"]


def test_falls_through_to_wildcard_when_specific_misses() -> None:
    reg = PolicyRegistry([
        _entry("specific", tenant="evals", rubrics=["correctness"]),
        _entry("baseline", tenant="*", rubrics=["safety"]),
    ])
    found = reg.resolve(tenant="dev", model="any")
    assert found is not None
    assert found.name == "baseline"


def test_no_match_returns_none() -> None:
    reg = PolicyRegistry([_entry("specific", tenant="evals", model="llama3.2:3b")])
    assert reg.resolve(tenant="dev", model="other") is None


# ---------------------------------------------------------------------------
# load_policy — file parsing
# ---------------------------------------------------------------------------


def test_missing_file_returns_empty_registry(tmp_path: Path) -> None:
    reg = load_policy(tmp_path / "nope.json")
    assert len(reg) == 0


def test_parses_well_formed_file(tmp_path: Path) -> None:
    path = tmp_path / "policies.json"
    path.write_text(json.dumps([
        {
            "name": "p1",
            "match": {"tenant": "evals", "model": "*"},
            "auto_eval": {
                "rubrics": ["safety", "correctness"],
                "mode": "background",
                "judge_model": "llama3.2:3b",
                "expected": "ref",
            },
        },
        {
            "name": "p2",
            "match": {"tenant": "*"},
            "auto_eval": {"rubrics": ["safety"]},
        },
    ]))

    reg = load_policy(path)
    assert len(reg) == 2

    e1, e2 = reg.all()
    assert e1.name == "p1"
    assert e1.match.tenant == "evals"
    assert e1.match.model == "*"
    assert e1.spec.rubrics == ["safety", "correctness"]
    assert e1.spec.judge_model == "llama3.2:3b"
    assert e1.spec.expected == "ref"

    assert e2.name == "p2"
    # Default-mode tightening from AutoEvalSpec applies.
    assert e2.spec.mode == "background"


def test_unnamed_entry_gets_synthetic_name(tmp_path: Path) -> None:
    path = tmp_path / "policies.json"
    path.write_text(json.dumps([
        {"match": {"tenant": "x"}, "auto_eval": {"rubrics": ["safety"]}},
    ]))
    reg = load_policy(path)
    assert reg.all()[0].name == "unnamed-0"


def test_top_level_must_be_array(tmp_path: Path) -> None:
    path = tmp_path / "policies.json"
    path.write_text(json.dumps({"policies": []}))
    with pytest.raises(ValueError, match="JSON array"):
        load_policy(path)


def test_missing_auto_eval_field_raises(tmp_path: Path) -> None:
    path = tmp_path / "policies.json"
    path.write_text(json.dumps([{"name": "broken", "match": {}}]))
    with pytest.raises(ValueError, match="auto_eval"):
        load_policy(path)


def test_invalid_match_type_raises(tmp_path: Path) -> None:
    path = tmp_path / "policies.json"
    path.write_text(json.dumps([
        {"name": "broken", "match": "not-an-object", "auto_eval": {"rubrics": ["safety"]}},
    ]))
    with pytest.raises(ValueError, match="must be an object"):
        load_policy(path)


# ---------------------------------------------------------------------------
# chat.py spec resolver — policy wins over request when both present
# ---------------------------------------------------------------------------


def test_resolver_prefers_policy_when_match(monkeypatch) -> None:
    """When server policy matches, the request's auto_eval is ignored — the
    policy plane is authoritative."""
    from inference_engine.api import chat as chat_mod  # noqa: PLC0415
    from inference_engine.api.state import app_state  # noqa: PLC0415

    policy_reg = PolicyRegistry([
        _entry("p", tenant="evals", model="llama3.2:1b", rubrics=["safety"]),
    ])
    monkeypatch.setattr(app_state, "policy_registry", policy_reg)

    request_spec = AutoEvalSpec(rubrics=["helpfulness"], mode="blocking")
    spec, policy = chat_mod._resolve_auto_eval(
        request_spec, tenant="evals", model_name="llama3.2:1b"
    )
    assert policy is not None
    assert policy.name == "p"
    assert spec.rubrics == ["safety"]  # request's "helpfulness" was overruled
    assert spec.mode == "background"


def test_resolver_falls_back_to_request_when_no_match(monkeypatch) -> None:
    from inference_engine.api import chat as chat_mod  # noqa: PLC0415
    from inference_engine.api.state import app_state  # noqa: PLC0415

    monkeypatch.setattr(app_state, "policy_registry", PolicyRegistry([]))

    request_spec = AutoEvalSpec(rubrics=["helpfulness"])
    spec, policy = chat_mod._resolve_auto_eval(
        request_spec, tenant="dev", model_name="x"
    )
    assert policy is None
    assert spec is request_spec  # passed through unchanged


def test_resolver_returns_none_when_neither_policy_nor_request(monkeypatch) -> None:
    from inference_engine.api import chat as chat_mod  # noqa: PLC0415
    from inference_engine.api.state import app_state  # noqa: PLC0415

    monkeypatch.setattr(app_state, "policy_registry", PolicyRegistry([]))

    spec, policy = chat_mod._resolve_auto_eval(None, tenant="x", model_name="y")
    assert spec is None
    assert policy is None
