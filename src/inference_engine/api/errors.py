"""OpenAI-shaped error envelope.

Every OpenAI SDK — and everything built on one (LangChain, LlamaIndex, the
Vercel AI SDK, the orchestra-python-sdk's model gateway) — reads failures out
of a top-level ``error`` object::

    {"error": {"message": ..., "type": ..., "param": ..., "code": ...}}

FastAPI instead serialises ``HTTPException(detail=...)`` as ``{"detail": ...}``.
The engine's typed errors already carry exactly the right *payload*
(``ContextLengthExceededError.error_detail()`` and friends) — it was only ever
nested under the wrong key, so ``APIStatusError.body`` came back empty and
``BadRequestError``/``RateLimitError`` lost their ``.code``.

These handlers emit **both**: ``error`` for OpenAI-compatible clients, and the
original ``detail`` unchanged so existing Prometa consumers and the admin
tooling keep working. Nothing is taken away.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

# Status → OpenAI error ``type``, used when a raiser passed a bare string and
# there is no richer typed payload to read a code off.
_STATUS_ERROR_TYPES: dict[int, str] = {
    400: "invalid_request_error",
    401: "authentication_error",
    403: "permission_error",
    404: "not_found_error",
    409: "conflict_error",
    413: "invalid_request_error",
    422: "invalid_request_error",
    429: "rate_limit_error",
    500: "api_error",
    501: "not_supported_error",
    502: "upstream_error",
    503: "service_unavailable_error",
    504: "timeout_error",
}


def error_type_for_status(status_code: int) -> str:
    if status_code in _STATUS_ERROR_TYPES:
        return _STATUS_ERROR_TYPES[status_code]
    if status_code >= 500:
        return "api_error"
    return "invalid_request_error"


def build_error_object(detail: Any, status_code: int) -> dict:
    """Normalise a FastAPI ``detail`` into an OpenAI ``error`` object.

    Dict details are already the right shape in this codebase — they carry
    ``message``/``type``/``code``/``param`` plus typed extras like
    ``context_window`` or ``retry_after_seconds``. Those extras are preserved:
    a client that knows to look for them still finds them, and one that doesn't
    ignores them.
    """
    fallback_type = error_type_for_status(status_code)

    if isinstance(detail, dict):
        error = dict(detail)
        error.setdefault("message", "")
        error.setdefault("type", fallback_type)
        error.setdefault("code", error["type"])
        error.setdefault("param", None)
        return error

    if isinstance(detail, str):
        message = detail
    elif detail is None:
        message = ""
    else:
        # Validation errors arrive as a list of per-field dicts. Keep the
        # structure under ``errors`` rather than stringifying it away.
        return {
            "message": "Request validation failed.",
            "type": fallback_type,
            "code": fallback_type,
            "param": None,
            "errors": detail,
        }

    return {
        "message": message,
        "type": fallback_type,
        "code": fallback_type,
        "param": None,
    }


def error_response(
    detail: Any,
    status_code: int,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"detail": detail, "error": build_error_object(detail, status_code)},
        headers=headers,
    )


async def http_exception_handler(
    request: Request,  # noqa: ARG001 — signature fixed by Starlette
    exc: StarletteHTTPException,
) -> JSONResponse:
    return error_response(exc.detail, exc.status_code, getattr(exc, "headers", None))


async def validation_exception_handler(
    request: Request,  # noqa: ARG001
    exc: RequestValidationError,
) -> JSONResponse:
    # Keep FastAPI's 422 — clients and tests already branch on it — but hand
    # back an ``error`` object so an OpenAI SDK surfaces a usable message
    # instead of an opaque UnprocessableEntity with an empty body.
    return error_response(_jsonable_errors(exc.errors()), 422)


def _jsonable_errors(errors: list) -> list:
    """Strip non-serialisable payloads out of pydantic validation errors.

    ``ValidationError.errors()`` embeds the offending input under ``ctx`` /
    ``input``, which can be an arbitrary object (a raw exception, bytes) that
    ``JSONResponse`` cannot encode. Dropping ``ctx`` keeps the useful parts —
    ``loc``, ``msg``, ``type`` — always encodable.
    """
    cleaned = []
    for err in errors:
        if not isinstance(err, dict):
            cleaned.append({"msg": str(err)})
            continue
        entry = {k: v for k, v in err.items() if k != "ctx"}
        if "input" in entry:
            try:
                JSONResponse(content=entry["input"])
            except Exception:  # noqa: BLE001 — unencodable input, summarise it
                entry["input"] = repr(entry["input"])[:200]
        cleaned.append(entry)
    return cleaned


def install_error_handlers(app: FastAPI) -> None:
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)


__all__ = [
    "build_error_object",
    "error_response",
    "error_type_for_status",
    "install_error_handlers",
]
