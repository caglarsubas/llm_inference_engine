"""Schema validation used by the non-grammar Structured Outputs path."""

from __future__ import annotations

import pytest

from inference_engine.structured_outputs import (
    SchemaViolation,
    repair_instruction,
    validate_json_document,
)

_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "score": {"type": "integer"},
        "tags": {"type": "array", "items": {"type": "string"}},
        "mood": {"enum": ["good", "bad"]},
    },
    "required": ["answer", "score"],
    "additionalProperties": False,
}


def test_accepts_a_conforming_document() -> None:
    doc = validate_json_document(
        '{"answer": "yes", "score": 3, "tags": ["a"], "mood": "good"}', _SCHEMA
    )
    assert doc["score"] == 3


def test_rejects_non_json() -> None:
    with pytest.raises(SchemaViolation, match="not valid JSON"):
        validate_json_document("I'm afraid I can't do that", _SCHEMA)


def test_rejects_missing_required_property() -> None:
    with pytest.raises(SchemaViolation, match="missing required property 'score'"):
        validate_json_document('{"answer": "yes"}', _SCHEMA)


def test_rejects_wrong_scalar_type() -> None:
    with pytest.raises(SchemaViolation, match="expected integer"):
        validate_json_document('{"answer": "yes", "score": "three"}', _SCHEMA)


def test_rejects_additional_property_when_closed() -> None:
    with pytest.raises(SchemaViolation, match="unexpected property 'extra'"):
        validate_json_document('{"answer": "y", "score": 1, "extra": 1}', _SCHEMA)


def test_rejects_value_outside_enum() -> None:
    with pytest.raises(SchemaViolation, match="not one of"):
        validate_json_document('{"answer": "y", "score": 1, "mood": "meh"}', _SCHEMA)


def test_validates_inside_arrays() -> None:
    with pytest.raises(SchemaViolation, match=r"tags\[1\]"):
        validate_json_document('{"answer": "y", "score": 1, "tags": ["a", 2]}', _SCHEMA)


def test_booleans_are_not_integers() -> None:
    """JSON Schema separates them even though Python's bool subclasses int."""
    with pytest.raises(SchemaViolation, match="expected integer"):
        validate_json_document('{"answer": "y", "score": true}', _SCHEMA)


def test_integers_satisfy_number() -> None:
    validate_json_document("3", {"type": "number"})


def test_nullable_union_type_accepts_both_members() -> None:
    schema = {"type": "object", "properties": {"note": {"type": ["string", "null"]}}}
    validate_json_document('{"note": null}', schema)
    validate_json_document('{"note": "hi"}', schema)
    with pytest.raises(SchemaViolation):
        validate_json_document('{"note": 5}', schema)


def test_any_of_accepts_a_matching_branch() -> None:
    schema = {"anyOf": [{"type": "string"}, {"type": "integer"}]}
    validate_json_document('"hi"', schema)
    validate_json_document("5", schema)
    with pytest.raises(SchemaViolation, match="matched no anyOf branch"):
        validate_json_document("[]", schema)


def test_resolves_local_refs_into_defs() -> None:
    schema = {
        "type": "object",
        "properties": {"item": {"$ref": "#/$defs/Item"}},
        "required": ["item"],
        "$defs": {
            "Item": {
                "type": "object",
                "properties": {"id": {"type": "integer"}},
                "required": ["id"],
            }
        },
    }
    validate_json_document('{"item": {"id": 1}}', schema)
    with pytest.raises(SchemaViolation, match="missing required property 'id'"):
        validate_json_document('{"item": {}}', schema)


def test_unresolvable_ref_is_permissive_rather_than_a_false_failure() -> None:
    """A gap in our subset must never reject a document we can't reason about."""
    schema = {"$ref": "https://example.com/remote.json"}
    validate_json_document('{"anything": true}', schema)


def test_unchecked_keywords_pass_silently() -> None:
    """minimum/pattern are outside the documented subset — must not raise."""
    validate_json_document("1", {"type": "integer", "minimum": 100})


def test_empty_schema_accepts_anything() -> None:
    validate_json_document('{"a": 1}', {})


def test_repair_instruction_carries_schema_and_error() -> None:
    text = repair_instruction(_SCHEMA, "missing required property 'score'", "my_output")
    assert "my_output" in text
    assert "missing required property 'score'" in text
    assert '"additionalProperties": false' in text.replace("False", "false")
