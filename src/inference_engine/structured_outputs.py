"""Schema validation for Structured Outputs on backends without grammar support.

Most backends enforce ``response_format={"type":"json_schema"}`` in the sampler
and never reach this module: llama.cpp compiles the schema to GBNF, vLLM routes
it through guided decoding, and the OpenRouter/Ollama upstreams do their own.
Those adapters set ``supports_structured_outputs = True`` and the chat route
trusts them.

MLX-LM has no constrained-decoding hook at all, so a schema request there would
otherwise be a suggestion in the prompt and nothing more. For that path the
route generates, validates here, and retries once with the validation error fed
back to the model.

Scope of the validator
----------------------

This deliberately implements the *subset of JSON Schema that OpenAI's strict
mode itself permits* — ``type``, ``properties``, ``required``,
``additionalProperties``, ``items``, ``enum``, ``anyOf``, and ``$ref`` into
``$defs``. It is not a general JSON Schema engine: ``allOf``, ``oneOf``,
``not``, ``patternProperties``, and the numeric/string assertion keywords
(``minimum``, ``pattern``, ``minLength``, …) are **not** checked and pass
silently. That matches what strict mode allows callers to send, and keeps the
engine free of a JSON Schema dependency.

A permissive gap here means an occasional invalid document reaches the client
on MLX — the same as today's behaviour. It never rejects a valid document,
which is the property that matters for not breaking working requests.
"""

from __future__ import annotations

import json
from typing import Any

_TYPE_CHECKS: dict[str, Any] = {
    "object": dict,
    "array": list,
    "string": str,
    "boolean": bool,
    "null": type(None),
}


class SchemaViolation(Exception):
    """The generated document does not satisfy the requested schema."""


class ResponseNotJson(SchemaViolation):
    """The text was not JSON at all.

    A subclass, so every existing ``except SchemaViolation`` still catches it.
    It is separated because the two carry very different WEIGHTS of evidence
    about the backend, and the capability registry turns on that difference:

    * A keyword violation is ambiguous. The document is JSON, and this module
      checks a subset of JSON Schema, so the fault may be a gap in the checker
      rather than in the backend.
    * Non-JSON is decisive. A sampler constrained to a JSON grammar cannot emit
      prose, so a backend that returns "Mars has two moons" to a schema request
      demonstrably did not constrain anything.
    """


def _resolve_ref(schema: dict, root: dict) -> dict:
    """Follow a local ``$ref`` into ``$defs``/``definitions``; else pass through."""
    ref = schema.get("$ref")
    if not isinstance(ref, str) or not ref.startswith("#/"):
        return schema
    node: Any = root
    for part in ref[2:].split("/"):
        if not isinstance(node, dict) or part not in node:
            return schema  # unresolvable — treat as unconstrained
        node = node[part]
    return node if isinstance(node, dict) else schema


def _check_type(value: Any, expected: str, path: str) -> None:
    if expected == "integer":
        # bool is a subclass of int in Python; JSON Schema treats them apart.
        if isinstance(value, bool) or not isinstance(value, int):
            raise SchemaViolation(f"{path or 'value'}: expected integer")
        return
    if expected == "number":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise SchemaViolation(f"{path or 'value'}: expected number")
        return
    if expected == "boolean":
        if not isinstance(value, bool):
            raise SchemaViolation(f"{path or 'value'}: expected boolean")
        return
    py_type = _TYPE_CHECKS.get(expected)
    if py_type is None:
        return  # unknown type keyword — do not invent a failure
    if not isinstance(value, py_type):
        raise SchemaViolation(f"{path or 'value'}: expected {expected}")


def _validate(value: Any, schema: dict, root: dict, path: str) -> None:
    if not isinstance(schema, dict) or not schema:
        return
    schema = _resolve_ref(schema, root)

    if "anyOf" in schema and isinstance(schema["anyOf"], list):
        errors = []
        for option in schema["anyOf"]:
            try:
                _validate(value, option, root, path)
                return
            except SchemaViolation as exc:
                errors.append(str(exc))
        raise SchemaViolation(f"{path or 'value'}: matched no anyOf branch ({'; '.join(errors)})")

    enum = schema.get("enum")
    if isinstance(enum, list) and value not in enum:
        raise SchemaViolation(f"{path or 'value'}: not one of {enum!r}")

    declared = schema.get("type")
    # ``type`` may be a list ("nullable string" is ["string","null"]).
    if isinstance(declared, str):
        _check_type(value, declared, path)
    elif isinstance(declared, list) and declared:
        for candidate in declared:
            try:
                _check_type(value, candidate, path)
                break
            except SchemaViolation:
                continue
        else:
            raise SchemaViolation(f"{path or 'value'}: expected one of {declared!r}")

    if isinstance(value, dict):
        properties = schema.get("properties")
        properties = properties if isinstance(properties, dict) else {}
        for key in schema.get("required") or []:
            if key not in value:
                raise SchemaViolation(f"{path or 'value'}: missing required property {key!r}")
        if schema.get("additionalProperties") is False and properties:
            for key in value:
                if key not in properties:
                    raise SchemaViolation(f"{path or 'value'}: unexpected property {key!r}")
        for key, sub_schema in properties.items():
            if key in value:
                _validate(value[key], sub_schema, root, f"{path}.{key}" if path else key)

    elif isinstance(value, list):
        items = schema.get("items")
        if isinstance(items, dict):
            for index, item in enumerate(value):
                _validate(item, items, root, f"{path}[{index}]")


def validate_json_document(text: str, schema: dict) -> Any:
    """Parse ``text`` and validate it against ``schema``.

    Returns the decoded document. Raises ``SchemaViolation`` when the text
    isn't JSON at all or violates a keyword this module checks.
    """
    try:
        document = json.loads(text)
    except (TypeError, ValueError) as exc:
        raise ResponseNotJson(f"response is not valid JSON: {exc}") from exc
    _validate(document, schema, schema, "")
    return document


def repair_instruction(schema: dict, error: str, name: str = "response") -> str:
    """The retry turn sent back to a model that produced an invalid document."""
    return (
        f"Your previous reply did not satisfy the required '{name}' JSON schema.\n"
        f"Validation error: {error}\n\n"
        f"Schema:\n{json.dumps(schema, ensure_ascii=False)}\n\n"
        "Reply with a single JSON document that satisfies the schema. "
        "Output only the JSON — no prose, no code fences."
    )


__all__ = [
    "ResponseNotJson",
    "SchemaViolation",
    "repair_instruction",
    "validate_json_document",
]
