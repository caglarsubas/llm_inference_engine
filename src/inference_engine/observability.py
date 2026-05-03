"""Structured logging + span emission.

The ``span()`` context manager is the single seam every route uses. Each call
emits **two** records when OTel is enabled:

  1. A structlog ``span.start`` / ``span.end`` pair with all attributes (always).
  2. A real OpenTelemetry span carrying the same attributes (when OTEL is on).

The yielded ``Span`` lets callers add attributes mid-flight via ``.bind(...)``.
That call updates *both* sinks in place — fixing a bug in the previous version
where the inner structlog ``BoundLogger`` returned by ``bind()`` was silently
discarded so usage attrs (input/output tokens) never reached ``span.end``.
"""

from __future__ import annotations

import logging
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import structlog

from . import otel as _otel


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper(), logging.INFO),
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(colors=True),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> structlog.BoundLogger:
    return structlog.get_logger(name)


class Span:
    """Bridge object yielded by ``span()``.

    Wraps a structlog logger and an OpenTelemetry span. Calls to ``bind()``
    mutate both: the local structlog reference is rebound (so subsequent log
    records inherit the new attrs) and OTel attributes are set on the live
    span. Non-primitive values are coerced to strings for OTel.
    """

    __slots__ = ("_attrs", "_log", "_otel")

    def __init__(self, name: str, attrs: dict[str, Any], otel_span: Any) -> None:
        self._attrs: dict[str, Any] = dict(attrs)
        self._log = get_logger("span").bind(span=name, **attrs)
        self._otel = otel_span
        for k, v in attrs.items():
            self._otel.set_attribute(k, _otel._coerce_attribute(v))

    def bind(self, **attrs: Any) -> "Span":
        self._attrs.update(attrs)
        self._log = self._log.bind(**attrs)
        for k, v in attrs.items():
            self._otel.set_attribute(k, _otel._coerce_attribute(v))
        return self

    def event(self, name: str, **attrs: Any) -> None:
        """Emit a named event scoped to this span.

        Goes to BOTH sinks: a structlog record (so plain-text log readers see
        it) and an OTel span event (so distributed traces carry it as a
        timestamped child of the span). Attribute values are coerced to
        OTel-compatible primitives the same way ``bind()`` does.
        """
        self._log.info(name, **attrs)
        # OTel's add_event API expects the name + a dict of attributes. NoOp
        # spans accept the call and drop it — checking is_real here would be
        # redundant. We only need add_event when the underlying span has it.
        if hasattr(self._otel, "add_event"):
            coerced = {k: _otel._coerce_attribute(v) for k, v in attrs.items()}
            self._otel.add_event(name, attributes=coerced)

    @property
    def log(self) -> structlog.BoundLogger:
        """Direct logger handle — for ad-hoc events inside the span body."""
        return self._log


@contextmanager
def span(name: str, **fields: Any) -> Iterator[Span]:
    with _otel.start_span(name) as otel_span:
        s = Span(name, fields, otel_span)
        start = time.perf_counter()
        s.log.info("span.start")
        try:
            yield s
        except Exception as exc:
            s.log.error("span.error", error=str(exc), error_type=type(exc).__name__)
            _otel.mark_span_error(otel_span, exc)
            raise
        finally:
            duration_ms = (time.perf_counter() - start) * 1000
            otel_span.set_attribute("duration_ms", round(duration_ms, 2))
            s.log.info("span.end", duration_ms=round(duration_ms, 2))
