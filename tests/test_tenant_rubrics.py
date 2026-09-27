"""Tenant rubrics — declarative definitions, score rules, per-tenant files, routes."""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator, Iterable
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from inference_engine.adapters import GenerationParams, InferenceAdapter, StreamChunk
from inference_engine.adapters.base import GenerationResult
from inference_engine.auth import Identity, require_identity
from inference_engine.cancellation import Cancellation
from inference_engine.evals import (
    EvalRunner,
    RubricDefinition,
    RubricLimitError,
    RubricRegistry,
    RubricStoreError,
    TenantRubricStore,
)
from inference_engine.evals.tenant_rubrics import digest, tenant_file_name, to_spec
from inference_engine.main import app
from inference_engine.manager import ModelManager
from inference_engine.registry import ModelDescriptor


def _definition(**overrides) -> dict:
    body = {
        "name": "policy_adherence",
        "description": "Did the agent follow the refund policy?",
        "system_prompt": 'Judge policy adherence. Reply {"score": 1-5, "reason": str}.',
        "user_prompt_template": "CUSTOMER:\n{prompt}\n\nAGENT:\n{response}\n\nVerdict:",
        "expected_keys": ["score", "reason"],
        "score": {"kind": "number", "key": "score", "min": 1, "max": 5},
    }
    body.update(overrides)
    return body


# ---------------------------------------------------------------------------
# Definitions
# ---------------------------------------------------------------------------


def test_definition_accepts_a_complete_rubric() -> None:
    definition = RubricDefinition.model_validate(_definition())
    assert definition.score.kind == "number"
    assert digest(definition).startswith("sha256:")


def test_digest_follows_content() -> None:
    a = RubricDefinition.model_validate(_definition())
    b = RubricDefinition.model_validate(_definition())
    c = RubricDefinition.model_validate(_definition(description="changed"))
    assert digest(a) == digest(b)
    assert digest(a) != digest(c)


@pytest.mark.parametrize(
    "template",
    [
        "{response} and {secret}",  # unknown name
        "{response.__class__}",  # attribute access
        "{response!r}",  # conversion
        "{response:>40}",  # format spec
        "{} {response}",  # positional
        "{response} {",  # unbalanced brace
        '{response} reply {"score": 1}',  # undoubled literal braces
    ],
)
def test_template_refuses_anything_but_the_four_markers(template: str) -> None:
    with pytest.raises(ValidationError, match="user_prompt_template"):
        RubricDefinition.model_validate(_definition(user_prompt_template=template))


def test_template_allows_doubled_braces() -> None:
    definition = RubricDefinition.model_validate(
        _definition(user_prompt_template='{response}\nReply {{"score": 3}}')
    )
    assert "{{" in definition.user_prompt_template


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"user_prompt_template": "only {prompt}"}, "{response}"),
        ({"pairwise": True}, "{response_b}"),
        ({"requires_expected": True}, "{expected}"),
        ({"score": {"kind": "boolean", "key": "pass"}}, "expected_keys"),
        ({"expected_keys": ["score", "score"]}, "unique"),
        ({"score": {"kind": "number", "key": "score", "min": 1}}, "both min and max"),
        ({"score": {"kind": "number", "key": "score", "min": 5, "max": 1}}, "min < max"),
        ({"name": "Policy"}, "pattern"),
        ({"name": "../escape"}, "pattern"),
        ({"expected_keys": ["score", "a b"]}, "pattern"),
        ({"score": {"kind": "choice", "key": "score", "values": {}}}, "at least 1"),
        ({"unknown_field": 1}, "Extra inputs"),
    ],
)
def test_definition_refuses_inconsistent_rubrics(overrides: dict, message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        RubricDefinition.model_validate(_definition(**overrides))


def test_definition_caps_prompt_size() -> None:
    with pytest.raises(ValidationError, match="8000"):
        RubricDefinition.model_validate(_definition(system_prompt="x" * 8001))


# ---------------------------------------------------------------------------
# Score rules
# ---------------------------------------------------------------------------


def _spec(**overrides):
    return to_spec(RubricDefinition.model_validate(_definition(**overrides)))


def test_number_score_normalises_to_its_scale() -> None:
    extract = _spec().score_extractor
    assert extract({"score": 1}) == 0.0
    assert extract({"score": 4}) == 0.75
    assert extract({"score": 5.0}) == 1.0


@pytest.mark.parametrize("value", [0, 6, True, "4", None])
def test_number_score_refuses_values_off_its_scale(value) -> None:
    with pytest.raises((ValueError, TypeError)):
        _spec().score_extractor({"score": value})


def test_unscaled_number_score_is_the_raw_value() -> None:
    extract = _spec(score={"kind": "number", "key": "score"}).score_extractor
    assert extract({"score": 17.5}) == 17.5


def test_boolean_score() -> None:
    extract = _spec(
        expected_keys=["pass", "reason"], score={"kind": "boolean", "key": "pass"}
    ).score_extractor
    assert extract({"pass": True}) == 1.0
    assert extract({"pass": False}) == 0.0
    assert extract({"pass": "False"}) == 0.0
    with pytest.raises(ValueError):
        extract({"pass": "yes"})
    with pytest.raises(ValueError):
        extract({"pass": 1})


def test_choice_score() -> None:
    extract = _spec(
        expected_keys=["verdict", "reason"],
        score={
            "kind": "choice",
            "key": "verdict",
            "values": {"follows": 1.0, "partial": 0.5, "violates": 0.0},
        },
    ).score_extractor
    assert extract({"verdict": " partial "}) == 0.5
    with pytest.raises(ValueError):
        extract({"verdict": "unsure"})


# ---------------------------------------------------------------------------
# Runner with a tenant rubric — an off-scale verdict fails instead of scoring
# ---------------------------------------------------------------------------


class _ScriptedJudge(InferenceAdapter):
    backend_name = "scripted-judge"

    def __init__(self) -> None:
        self.next_text = ""
        self.last_messages: list = []
        self._descriptor: ModelDescriptor | None = None

    @property
    def is_loaded(self) -> bool:
        return self._descriptor is not None

    @property
    def loaded_model(self) -> ModelDescriptor | None:
        return self._descriptor

    async def load(self, descriptor: ModelDescriptor) -> None:
        self._descriptor = descriptor

    async def unload(self) -> None:
        self._descriptor = None

    async def generate(
        self, messages: Iterable, params: GenerationParams, cancel: Cancellation | None = None
    ) -> GenerationResult:
        self.last_messages = list(messages)
        return GenerationResult(
            text=self.next_text, finish_reason="stop", prompt_tokens=12, completion_tokens=8
        )

    async def stream(
        self, messages: Iterable, params: GenerationParams, cancel: Cancellation | None = None
    ) -> AsyncIterator[StreamChunk]:
        yield StreamChunk(text="", finish_reason="stop")


@pytest.fixture
def judge_kit():
    desc = ModelDescriptor(
        name="judge", tag="1", namespace="ns", registry="reg",
        model_path=Path("/tmp/judge"), format="gguf", size_bytes=1,
    )

    class _Reg:
        def get(self, name: str) -> ModelDescriptor | None:
            return desc if name == "judge:1" else None

        def list_models(self) -> list[ModelDescriptor]:
            return [desc]

    judge = _ScriptedJudge()
    mgr = ModelManager(_Reg(), adapter_factory=lambda d: judge, memory_budget_bytes=100)
    return EvalRunner(mgr), judge


@pytest.mark.asyncio
async def test_runner_scores_a_tenant_rubric_and_fails_off_scale(judge_kit) -> None:
    runner, judge = judge_kit
    spec = _spec()

    judge.next_text = '{"score": 4, "reason": "refund within policy"}'
    verdict, _ = await runner.run(
        spec, prompt="refund?", response="Refunded.", expected=None, judge_model="judge:1"
    )
    assert (verdict.score, verdict.parse_status) == (0.75, "clean")
    user = next(m for m in judge.last_messages if m.role == "user").content
    assert user == "CUSTOMER:\nrefund?\n\nAGENT:\nRefunded.\n\nVerdict:"

    judge.next_text = '{"score": 9, "reason": "off the scale"}'
    verdict, _ = await runner.run(
        spec, prompt="refund?", response="Refunded.", expected=None, judge_model="judge:1"
    )
    assert (verdict.score, verdict.parse_status) == (0.0, "failed")


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


def _def(**overrides) -> RubricDefinition:
    return RubricDefinition.model_validate(_definition(**overrides))


def test_store_registers_replaces_and_deletes() -> None:
    store = TenantRubricStore()
    first, created = store.put("acme", _def(), now=100)
    assert created and first.created_at == first.updated_at == 100

    second, created = store.put("acme", _def(description="v2"), now=200)
    assert not created
    assert (second.created_at, second.updated_at) == (100, 200)
    assert second.digest != first.digest
    assert store.get("acme", "policy_adherence").definition.description == "v2"
    assert [r.definition.name for r in store.list("acme")] == ["policy_adherence"]

    assert store.delete("acme", "policy_adherence") is True
    assert store.delete("acme", "policy_adherence") is False
    assert store.get("acme", "policy_adherence") is None


def test_store_keeps_tenants_apart() -> None:
    store = TenantRubricStore()
    store.put("acme", _def())
    assert store.get("globex", "policy_adherence") is None
    assert store.list("globex") == []


def test_store_limits_rubrics_per_tenant_but_allows_replacement() -> None:
    store = TenantRubricStore(max_per_tenant=2)
    store.put("acme", _def(name="one"))
    store.put("acme", _def(name="two"))
    with pytest.raises(RubricLimitError):
        store.put("acme", _def(name="three"))
    store.put("acme", _def(name="two", description="replaced"))
    store.put("globex", _def(name="three"))


def test_store_persists_per_tenant_and_reloads(tmp_path: Path) -> None:
    store = TenantRubricStore(tmp_path)
    stored, _ = store.put("acme/../../etc", _def(), now=100)
    store.put("globex", _def(name="other"), now=100)

    files = sorted(p.name for p in tmp_path.iterdir())
    assert files == sorted([tenant_file_name("acme/../../etc"), tenant_file_name("globex")])

    reloaded = TenantRubricStore.load(tmp_path)
    again = reloaded.get("acme/../../etc", "policy_adherence")
    assert again is not None
    assert again.digest == stored.digest
    assert (again.created_at, again.updated_at) == (100, 100)
    assert reloaded.get("globex", "policy_adherence") is None

    reloaded.delete("globex", "other")
    assert not (tmp_path / tenant_file_name("globex")).exists()


def test_store_load_of_a_missing_directory_is_empty(tmp_path: Path) -> None:
    store = TenantRubricStore.load(tmp_path / "absent")
    assert store.list("acme") == []
    assert not (tmp_path / "absent").exists()


def test_store_refuses_a_corrupt_file(tmp_path: Path) -> None:
    (tmp_path / tenant_file_name("acme")).write_text("{not json")
    with pytest.raises(RubricStoreError, match="cannot load"):
        TenantRubricStore.load(tmp_path)


def test_store_refuses_a_file_named_for_another_tenant(tmp_path: Path) -> None:
    TenantRubricStore(tmp_path).put("acme", _def())
    (tmp_path / tenant_file_name("acme")).rename(tmp_path / tenant_file_name("globex"))
    with pytest.raises(RubricStoreError, match="does not match"):
        TenantRubricStore.load(tmp_path)


def test_failed_write_leaves_memory_and_disk_unchanged(tmp_path: Path, monkeypatch) -> None:
    store = TenantRubricStore(tmp_path)
    store.put("acme", _def())
    before = (tmp_path / tenant_file_name("acme")).read_text()

    def _boom(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", _boom)
    with pytest.raises(OSError, match="disk full"):
        store.put("acme", _def(description="never lands"))

    assert store.get("acme", "policy_adherence").definition.description != "never lands"
    assert (tmp_path / tenant_file_name("acme")).read_text() == before
    assert [p.suffix for p in tmp_path.iterdir()] == [".json"]


def test_replicas_sharing_a_directory_see_each_others_writes(tmp_path: Path) -> None:
    a = TenantRubricStore.load(tmp_path)
    b = TenantRubricStore.load(tmp_path)

    a.put("acme", _def(), now=100)
    assert b.get("acme", "policy_adherence") is not None

    b.put("acme", _def(name="second"), now=200)
    assert [r.definition.name for r in a.list("acme")] == ["policy_adherence", "second"]

    a.delete("acme", "policy_adherence")
    a.delete("acme", "second")
    assert b.list("acme") == []


def test_an_unreadable_replacement_keeps_the_last_good_copy(tmp_path: Path) -> None:
    store = TenantRubricStore(tmp_path)
    store.put("acme", _def())
    (tmp_path / tenant_file_name("acme")).write_text("{torn")
    assert store.get("acme", "policy_adherence") is not None


def test_written_file_is_readable_json(tmp_path: Path) -> None:
    TenantRubricStore(tmp_path).put("acme", _def())
    document = json.loads((tmp_path / tenant_file_name("acme")).read_text())
    assert document["tenant"] == "acme"
    assert document["rubrics"][0]["definition"]["name"] == "policy_adherence"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@pytest.fixture
def routes(monkeypatch, tmp_path, judge_kit):
    runner, judge = judge_kit
    from inference_engine.api.state import app_state  # noqa: PLC0415

    store = TenantRubricStore(tmp_path, max_per_tenant=3)
    monkeypatch.setattr(app_state, "eval_runner", runner)
    monkeypatch.setattr(app_state, "rubric_registry", RubricRegistry.with_builtins())
    monkeypatch.setattr(app_state, "tenant_rubrics", store)
    return TestClient(app), judge, store


def _run(client: TestClient, rubric: str = "policy_adherence"):
    return client.post(
        "/v1/evals/run",
        json={
            "rubric": rubric,
            "prompt": "Can I get a refund?",
            "response": "Yes, within 30 days.",
            "judge_model": "judge:1",
        },
    )


def test_route_registers_lists_and_runs_a_tenant_rubric(routes) -> None:
    client, judge, _ = routes

    created = client.post("/v1/evals/rubrics", json=_definition())
    assert created.status_code == 201, created.text
    body = created.json()
    assert body["source"] == "tenant"
    assert body["score"] == {"kind": "number", "key": "score", "min": 1.0, "max": 5.0}
    first_digest = body["digest"]

    listed = {r["name"]: r for r in client.get("/v1/evals/rubrics").json()["data"]}
    assert listed["policy_adherence"]["source"] == "tenant"
    assert listed["policy_adherence"]["digest"] == first_digest
    assert listed["helpfulness"]["source"] == "builtin"

    judge.next_text = '{"score": 5, "reason": "follows policy"}'
    run = _run(client)
    assert run.status_code == 200, run.text
    assert run.json()["rubric_source"] == "tenant"
    assert run.json()["rubric_digest"] == first_digest
    assert run.json()["verdict"]["score"] == 1.0
    system = next(m for m in judge.last_messages if m.role == "system").content
    assert system.startswith("Judge policy adherence.")

    replaced = client.post("/v1/evals/rubrics", json=_definition(description="v2"))
    assert replaced.status_code == 200
    assert replaced.json()["digest"] != first_digest
    assert _run(client).json()["rubric_digest"] == replaced.json()["digest"]

    detail = client.get("/v1/evals/rubrics/policy_adherence")
    assert detail.status_code == 200
    assert detail.json()["description"] == "v2"
    assert detail.json()["created_at"] <= detail.json()["updated_at"]

    assert client.delete("/v1/evals/rubrics/policy_adherence").status_code == 204
    assert _run(client).status_code == 404


def test_route_shows_a_builtin_in_full(routes) -> None:
    client, _, _ = routes
    detail = client.get("/v1/evals/rubrics/safety").json()
    assert detail["source"] == "builtin"
    assert detail["score"] is None
    assert "{prompt}" in detail["user_prompt_template"]


def test_route_refuses_builtin_names(routes) -> None:
    client, _, _ = routes
    r = client.post("/v1/evals/rubrics", json=_definition(name="safety"))
    assert r.status_code == 409
    assert r.json()["detail"]["type"] == "rubric_name_reserved"
    assert r.json()["error"]["code"] == "rubric_name_reserved"
    assert client.delete("/v1/evals/rubrics/safety").status_code == 409
    assert client.delete("/v1/evals/rubrics/ghost").status_code == 404
    assert client.get("/v1/evals/rubrics/ghost").status_code == 404


def test_route_refuses_an_invalid_definition(routes) -> None:
    client, _, store = routes
    r = client.post(
        "/v1/evals/rubrics", json=_definition(user_prompt_template="{response.__class__}")
    )
    assert r.status_code == 422
    assert store.list("anonymous") == []


def test_route_reports_the_tenant_limit(routes) -> None:
    client, _, _ = routes
    for name in ("one", "two", "three"):
        assert client.post("/v1/evals/rubrics", json=_definition(name=name)).status_code == 201
    r = client.post("/v1/evals/rubrics", json=_definition(name="four"))
    assert r.status_code == 409
    assert r.json()["detail"]["type"] == "rubric_limit_reached"
    assert r.json()["detail"]["limit"] == 3


def test_route_keeps_tenants_apart(routes) -> None:
    client, judge, _ = routes
    try:
        app.dependency_overrides[require_identity] = lambda: Identity(tenant="acme", key_id="a")
        assert client.post("/v1/evals/rubrics", json=_definition()).status_code == 201

        app.dependency_overrides[require_identity] = lambda: Identity(tenant="globex", key_id="g")
        names = {r["name"] for r in client.get("/v1/evals/rubrics").json()["data"]}
        assert "policy_adherence" not in names
        assert client.get("/v1/evals/rubrics/policy_adherence").status_code == 404
        assert client.delete("/v1/evals/rubrics/policy_adherence").status_code == 404
        judge.next_text = '{"score": 5, "reason": "x"}'
        assert _run(client).status_code == 404
    finally:
        app.dependency_overrides.pop(require_identity, None)
