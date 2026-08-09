from __future__ import annotations

import re

import pytest
from starlette.datastructures import Headers

from inference_engine.request_identity import (
    MAX_EXTERNAL_IDENTITY_LENGTH,
    InvalidInvocationIdentity,
    new_engine_request_id,
    new_usage_record_id,
    read_upstream_invocation_identity,
)


def _headers(*pairs: tuple[bytes, bytes]) -> Headers:
    return Headers(raw=list(pairs))


def test_server_ids_are_typed_unique_lowercase_uuid_values() -> None:
    engine_ids = {new_engine_request_id() for _ in range(4)}
    usage_ids = {new_usage_record_id() for _ in range(4)}

    assert len(engine_ids) == 4
    assert len(usage_ids) == 4
    assert all(re.fullmatch(r"req_[0-9a-f]{32}", value) for value in engine_ids)
    assert all(re.fullmatch(r"usage_[0-9a-f]{32}", value) for value in usage_ids)
    assert engine_ids.isdisjoint(usage_ids)


def test_all_upstream_identities_are_optional() -> None:
    identity = read_upstream_invocation_identity(Headers())

    assert identity.runtime_request_id is None
    assert identity.model_invocation_id is None
    assert identity.model_attempt_id is None


def test_upstream_identities_are_independent_and_preserved_exactly() -> None:
    identity = read_upstream_invocation_identity(
        _headers(
            (b"x-orchestra-runtime-request-id", b"runtime/A:1@tenant"),
            (b"x-orchestra-model-invocation-id", b"same-prefix.invocation"),
            (b"x-orchestra-model-attempt-id", b"same-prefix.attempt"),
        )
    )

    assert identity.runtime_request_id == "runtime/A:1@tenant"
    assert identity.model_invocation_id == "same-prefix.invocation"
    assert identity.model_attempt_id == "same-prefix.attempt"


def test_external_identity_accepts_the_exact_maximum_without_truncating() -> None:
    value = "z" * MAX_EXTERNAL_IDENTITY_LENGTH

    identity = read_upstream_invocation_identity(
        _headers((b"x-orchestra-runtime-request-id", value.encode("ascii")))
    )

    assert identity.runtime_request_id == value
    assert len(identity.runtime_request_id) == MAX_EXTERNAL_IDENTITY_LENGTH


def test_runtime_and_invocation_is_a_valid_ordered_prefix() -> None:
    identity = read_upstream_invocation_identity(
        _headers(
            (b"x-orchestra-runtime-request-id", b"runtime-1"),
            (b"x-orchestra-model-invocation-id", b"invocation-1"),
        )
    )

    assert identity.runtime_request_id == "runtime-1"
    assert identity.model_invocation_id == "invocation-1"
    assert identity.model_attempt_id is None


@pytest.mark.parametrize(
    ("raw", "reason"),
    [
        (b"", "1-256"),
        (b"z" * (MAX_EXTERNAL_IDENTITY_LENGTH + 1), "1-256"),
        (b"has space", "visible ASCII"),
        (b"tab\tvalue", "visible ASCII"),
        (b"non-ascii-\x80", "visible ASCII"),
    ],
)
def test_invalid_external_identity_is_rejected_not_normalized(raw: bytes, reason: str) -> None:
    with pytest.raises(InvalidInvocationIdentity, match=reason) as caught:
        read_upstream_invocation_identity(
            _headers((b"x-orchestra-model-invocation-id", raw))
        )

    assert caught.value.header == "x-orchestra-model-invocation-id"


@pytest.mark.parametrize(
    "value",
    [
        "null",
        "NULL",
        "NuLl",
        "none",
        "NONE",
        "NoNe",
        "nil",
        "NIL",
        "NiL",
        "undefined",
        "UNDEFINED",
        "UnDeFiNeD",
    ],
)
def test_flattened_null_sentinels_are_reserved_case_insensitively(value: str) -> None:
    with pytest.raises(InvalidInvocationIdentity, match="reserved") as caught:
        read_upstream_invocation_identity(
            _headers((b"x-orchestra-runtime-request-id", value.encode("ascii")))
        )

    assert caught.value.header == "x-orchestra-runtime-request-id"


def test_sentinel_prefixes_and_suffixes_remain_exact_valid_identities() -> None:
    identity = read_upstream_invocation_identity(
        _headers((b"x-orchestra-runtime-request-id", b"null-runtime"))
    )

    assert identity.runtime_request_id == "null-runtime"


def test_duplicate_identity_header_lines_are_rejected_as_ambiguous() -> None:
    with pytest.raises(InvalidInvocationIdentity, match="at most once"):
        read_upstream_invocation_identity(
            _headers(
                (b"x-orchestra-model-attempt-id", b"attempt-1"),
                (b"x-orchestra-model-attempt-id", b"attempt-2"),
            )
        )


@pytest.mark.parametrize(
    ("pairs", "invalid_header"),
    [
        (
            ((b"x-orchestra-model-invocation-id", b"invocation-1"),),
            "x-orchestra-model-invocation-id",
        ),
        (
            ((b"x-orchestra-model-attempt-id", b"attempt-1"),),
            "x-orchestra-model-attempt-id",
        ),
        (
            (
                (b"x-orchestra-runtime-request-id", b"runtime-1"),
                (b"x-orchestra-model-attempt-id", b"attempt-1"),
            ),
            "x-orchestra-model-attempt-id",
        ),
        (
            (
                (b"x-orchestra-model-invocation-id", b"invocation-1"),
                (b"x-orchestra-model-attempt-id", b"attempt-1"),
            ),
            "x-orchestra-model-invocation-id",
        ),
    ],
)
def test_upstream_identities_must_form_an_ordered_prefix(
    pairs: tuple[tuple[bytes, bytes], ...],
    invalid_header: str,
) -> None:
    with pytest.raises(InvalidInvocationIdentity) as caught:
        read_upstream_invocation_identity(_headers(*pairs))

    assert caught.value.header == invalid_header
    assert "requires" in caught.value.reason
