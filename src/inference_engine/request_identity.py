"""Server and tenant-runtime identities for one model-plane HTTP request.

The model plane owns two identifiers: ``engine_request_id`` for the HTTP hop
and ``usage_record_id`` for a best-effort buffered ledger record. A tenant runtime
may add three correlation identifiers, but none of them is allowed to replace
or alias either server-owned value.

External identifiers are intentionally validated rather than shortened.  A
prefix slice turns two distinct upstream values into the same billing
correlation key, which is worse than rejecting the request.  The accepted wire
alphabet and length match ``contracts/prometa-model-usage-v2.schema.json``.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from starlette.datastructures import Headers

MAX_EXTERNAL_IDENTITY_LENGTH = 256

# OTLP attributes have no null value. Collectors that flatten a JSON ledger
# record commonly render an actual null as one of these strings, so admitting
# the same bytes as a caller-owned identity would make absence and identity
# indistinguishable at the control-plane intake. The shared wire schema reserves
# them case-insensitively for exactly that reason. We still return every other
# accepted value byte-for-byte; this check is validation, not normalization.
_RESERVED_EXTERNAL_IDENTITY_VALUES = frozenset(
    {"null", "none", "nil", "undefined"}
)

RUNTIME_REQUEST_ID_HEADER = "x-orchestra-runtime-request-id"
MODEL_INVOCATION_ID_HEADER = "x-orchestra-model-invocation-id"
MODEL_ATTEMPT_ID_HEADER = "x-orchestra-model-attempt-id"
USAGE_RECORD_ID_HEADER = "x-orchestra-usage-record-id"
ENGINE_REQUEST_ID_HEADER = "x-request-id"

_UPSTREAM_HEADERS = (
    RUNTIME_REQUEST_ID_HEADER,
    MODEL_INVOCATION_ID_HEADER,
    MODEL_ATTEMPT_ID_HEADER,
)


class InvalidInvocationIdentity(ValueError):
    """A tenant-runtime identity header is ambiguous or outside the contract."""

    def __init__(self, header: str, reason: str) -> None:
        self.header = header
        self.reason = reason
        super().__init__(f"{header}: {reason}")


@dataclass(frozen=True, slots=True)
class UpstreamInvocationIdentity:
    """Optional tenant-runtime correlation carried into a ledger record."""

    runtime_request_id: str | None
    model_invocation_id: str | None
    model_attempt_id: str | None


def new_engine_request_id() -> str:
    """Mint the response correlation for exactly one engine HTTP request."""
    return f"req_{uuid.uuid4().hex}"


def new_usage_record_id() -> str:
    """Mint the sole dedupe identity for exactly one ledger record."""
    return f"usage_{uuid.uuid4().hex}"


def _validated_header(headers: Headers, name: str) -> str | None:
    values = headers.getlist(name)
    if not values:
        return None
    if len(values) != 1:
        raise InvalidInvocationIdentity(name, "header must occur at most once")
    value = values[0]
    if not 1 <= len(value) <= MAX_EXTERNAL_IDENTITY_LENGTH:
        raise InvalidInvocationIdentity(
            name,
            f"value must contain 1-{MAX_EXTERNAL_IDENTITY_LENGTH} characters",
        )
    if any(not "!" <= character <= "~" for character in value):
        raise InvalidInvocationIdentity(name, "value must contain visible ASCII only")
    if value.lower() in _RESERVED_EXTERNAL_IDENTITY_VALUES:
        raise InvalidInvocationIdentity(
            name,
            "value is reserved for a flattened absent/null identity",
        )
    return value


def read_upstream_invocation_identity(headers: Headers) -> UpstreamInvocationIdentity:
    """Read three independent upstream identities without normalization.

    Duplicate header lines are rejected rather than choosing one.  Values are
    returned byte-for-byte at the visible-ASCII layer: no whitespace trim,
    case fold, prefix slice, or other collision-producing normalization.
    """
    values = {name: _validated_header(headers, name) for name in _UPSTREAM_HEADERS}
    runtime_request_id = values[RUNTIME_REQUEST_ID_HEADER]
    model_invocation_id = values[MODEL_INVOCATION_ID_HEADER]
    model_attempt_id = values[MODEL_ATTEMPT_ID_HEADER]
    if model_invocation_id is not None and runtime_request_id is None:
        raise InvalidInvocationIdentity(
            MODEL_INVOCATION_ID_HEADER,
            f"header requires {RUNTIME_REQUEST_ID_HEADER}",
        )
    if model_attempt_id is not None and (
        runtime_request_id is None or model_invocation_id is None
    ):
        raise InvalidInvocationIdentity(
            MODEL_ATTEMPT_ID_HEADER,
            "header requires "
            f"{RUNTIME_REQUEST_ID_HEADER} and {MODEL_INVOCATION_ID_HEADER}",
        )
    return UpstreamInvocationIdentity(
        runtime_request_id=runtime_request_id,
        model_invocation_id=model_invocation_id,
        model_attempt_id=model_attempt_id,
    )


__all__ = [
    "ENGINE_REQUEST_ID_HEADER",
    "MAX_EXTERNAL_IDENTITY_LENGTH",
    "MODEL_ATTEMPT_ID_HEADER",
    "MODEL_INVOCATION_ID_HEADER",
    "RUNTIME_REQUEST_ID_HEADER",
    "USAGE_RECORD_ID_HEADER",
    "InvalidInvocationIdentity",
    "UpstreamInvocationIdentity",
    "new_engine_request_id",
    "new_usage_record_id",
    "read_upstream_invocation_identity",
]
