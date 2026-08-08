"""OpenTelemetry integration — optional, lazy, no-op when disabled.

This module is the only place that imports the OTel SDK. The rest of the codebase
treats spans as opaque objects with three duck-typed methods (``set_attribute``,
``record_exception``, ``set_status``). When ``OTEL_ENABLED=false`` the spans are
the ``_NoOpSpan`` shim defined here, so calling code never branches on whether
tracing is on.

Wired in two places:

* ``main.py`` lifespan — calls ``configure_tracing()`` once on startup and
  ``shutdown_tracing()`` on the way down so the BatchSpanProcessor flushes.
* ``observability.span()`` — wraps every emitted structlog span with a real
  OTel span when configured, propagating attribute updates from ``Span.bind()``.

Trace destination: an OTLP/gRPC collector or an OTLP/HTTP traces endpoint.
The default remains ``http://localhost:4317`` with protocol ``grpc``, matching
the Jaeger all-in-one container in ``docker-compose.otel.yml``.
"""

from __future__ import annotations

import re
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Iterator
from urllib.parse import unquote, urlsplit, urlunsplit

from .config import settings

if TYPE_CHECKING:
    from opentelemetry.trace import Tracer


class _NoOpSpan:
    """Drop-in for an OTel span when tracing is off.

    Implements the methods our code calls; everything else returns silently
    so callers can stay branch-free.
    """

    def set_attribute(self, key: str, value: Any) -> None:  # noqa: ARG002
        return None

    def record_exception(self, exception: BaseException) -> None:  # noqa: ARG002
        return None

    def set_status(self, status: Any) -> None:  # noqa: ARG002
        return None

    def add_event(self, name: str, attributes: Any = None) -> None:  # noqa: ARG002
        return None


_tracer: Tracer | None = None
_initialized = False
_HEADER_NAME = re.compile(r"^[!#$%&'*+\-.^_`|~0-9A-Za-z]+$")


def is_enabled() -> bool:
    return _tracer is not None


def _coerce_attribute(value: Any) -> Any:
    """OTel only accepts primitives or sequences of primitives — coerce the rest."""
    if isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, list | tuple):
        return [_coerce_attribute(v) for v in value]
    return str(value)


def _validated_endpoint(protocol: str, endpoint: str) -> tuple[str, bool]:
    value = endpoint.strip()
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("otel_exporter_endpoint_invalid")
    if parsed.fragment or parsed.query or parsed.username or parsed.password:
        raise ValueError("otel_exporter_endpoint_invalid")
    if protocol == "grpc":
        if parsed.path not in {"", "/"}:
            raise ValueError("otel_grpc_endpoint_path_forbidden")
        return value, parsed.scheme == "http"
    if protocol != "http/protobuf":
        raise ValueError("otel_exporter_protocol_invalid")
    if parsed.path in {"", "/"}:
        parsed = parsed._replace(path="/v1/traces")
        value = urlunsplit(parsed)
    return value, False


def _parse_headers(value: str) -> dict[str, str] | None:
    if not value:
        return None
    headers: dict[str, str] = {}
    for encoded_item in value.split(","):
        if "=" not in encoded_item:
            raise ValueError("otel_exporter_headers_invalid")
        encoded_name, encoded_value = encoded_item.split("=", 1)
        name = unquote(encoded_name).strip()
        header_value = unquote(encoded_value).strip()
        if (
            not name
            or not header_value
            or not _HEADER_NAME.fullmatch(name)
            or any(ord(character) < 32 or ord(character) == 127 for character in header_value)
            or name.lower() in headers
        ):
            raise ValueError("otel_exporter_headers_invalid")
        headers[name.lower()] = header_value
    return headers


def _build_otlp_exporter(config: Any) -> Any:
    protocol = config.otel_exporter_otlp_protocol
    endpoint, insecure = _validated_endpoint(
        protocol,
        config.otel_exporter_otlp_endpoint,
    )
    headers_value = config.otel_exporter_otlp_headers.strip()
    headers = _parse_headers(headers_value)

    if protocol == "http/protobuf":
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (  # noqa: PLC0415
            OTLPSpanExporter,
        )

        return OTLPSpanExporter(endpoint=endpoint, headers=headers)

    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (  # noqa: PLC0415
        OTLPSpanExporter,
    )

    return OTLPSpanExporter(
        endpoint=endpoint,
        insecure=insecure,
        headers=headers,
    )


def configure_tracing() -> None:
    """Initialize the OTel SDK if enabled. Idempotent.

    Safe to call multiple times — only the first invocation sets the global
    TracerProvider. When ``OTEL_ENABLED=false`` (default) this is a no-op and
    every subsequent ``start_span`` call yields a ``_NoOpSpan``.
    """
    global _tracer, _initialized
    if _initialized:
        return
    _initialized = True

    # Defer log import so observability.py can import this module without cycles.
    from .observability import get_logger  # noqa: PLC0415

    log = get_logger("otel")

    if not settings.otel_enabled:
        log.info("otel.disabled")
        return

    try:
        from opentelemetry import trace  # noqa: PLC0415
        from opentelemetry.sdk.resources import Resource  # noqa: PLC0415
        from opentelemetry.sdk.trace import TracerProvider  # noqa: PLC0415
        from opentelemetry.sdk.trace.export import BatchSpanProcessor  # noqa: PLC0415
    except ImportError as exc:
        log.error(
            "otel.import_failed",
            error=str(exc),
            hint="install OpenTelemetry deps (native: `make install-otel`, then restart the engine)",
        )
        return

    from . import __version__  # noqa: PLC0415

    resource = Resource.create(
        {
            "service.name": settings.otel_service_name,
            "service.version": __version__,
        }
    )

    provider = TracerProvider(resource=resource)
    try:
        exporter = _build_otlp_exporter(settings)
    except ImportError as exc:
        log.error(
            "otel.import_failed",
            error=str(exc),
            hint="install the OpenTelemetry exporter extra and restart the engine",
        )
        return
    except ValueError as exc:
        log.error("otel.configuration_invalid", error=str(exc))
        return
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    _tracer = trace.get_tracer("inference_engine")
    log.info(
        "otel.configured",
        endpoint=settings.otel_exporter_otlp_endpoint,
        protocol=settings.otel_exporter_otlp_protocol,
        service_name=settings.otel_service_name,
    )


@contextmanager
def start_span(name: str) -> Iterator[Any]:
    """Yield an OTel span if configured, else a no-op shim.

    The yielded object always exposes ``set_attribute`` / ``record_exception``
    / ``set_status`` — callers don't branch on tracing state.
    """
    if _tracer is None:
        yield _NoOpSpan()
        return
    with _tracer.start_as_current_span(name) as otel_span:
        yield otel_span


def mark_span_error(span: Any, exc: BaseException) -> None:
    """Record an exception and flip the span status to ERROR (if real)."""
    if isinstance(span, _NoOpSpan):
        return
    try:
        from opentelemetry.trace import Status, StatusCode  # noqa: PLC0415
    except ImportError:
        return
    span.record_exception(exc)
    span.set_status(Status(StatusCode.ERROR, str(exc)))


def current_trace_context() -> tuple[str, str] | None:
    """Hex ``(trace_id, span_id)`` of the span in flight, else ``None``.

    Uses only the stable ``opentelemetry.trace`` API so records emitted outside
    the span pipeline (the usage ledger) can still be joined back to a trace
    without any caller branching on whether tracing is configured.
    """
    if _tracer is None:
        return None
    try:
        from opentelemetry import trace  # noqa: PLC0415
    except ImportError:
        return None
    context = trace.get_current_span().get_span_context()
    if not context.is_valid:
        return None
    return format(context.trace_id, "032x"), format(context.span_id, "016x")


def instrument_fastapi(app: Any) -> None:
    """Auto-create one parent span per HTTP request (POST /v1/chat/completions, etc.)."""
    if _tracer is None:
        return
    try:
        from opentelemetry.instrumentation.fastapi import (  # noqa: PLC0415
            FastAPIInstrumentor,
        )
    except ImportError:
        return
    FastAPIInstrumentor.instrument_app(app)


def shutdown_tracing() -> None:
    """Flush the BatchSpanProcessor on shutdown so we don't lose tail spans."""
    if _tracer is None:
        return
    try:
        from opentelemetry import trace  # noqa: PLC0415
    except ImportError:
        return
    provider = trace.get_tracer_provider()
    if hasattr(provider, "shutdown"):
        provider.shutdown()


# ---------------------------------------------------------------------------
# Test helper — install an in-memory exporter for assertions.
# ---------------------------------------------------------------------------


def _install_in_memory_exporter() -> Any:
    """Reset + install an InMemorySpanExporter. Returns the exporter for assertions.

    Used only by the test suite — kept here so the import path doesn't leak
    OTel imports into observability.py.
    """
    global _tracer, _initialized

    from opentelemetry import trace  # noqa: PLC0415
    from opentelemetry.sdk.resources import Resource  # noqa: PLC0415
    from opentelemetry.sdk.trace import TracerProvider  # noqa: PLC0415
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # noqa: PLC0415
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (  # noqa: PLC0415
        InMemorySpanExporter,
    )

    exporter = InMemorySpanExporter()
    provider = TracerProvider(resource=Resource.create({"service.name": "test"}))
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    # Note: trace.set_tracer_provider() is one-shot — once set in a process it
    # warns on re-set. We call it anyway so tests get fresh state; the warning
    # is harmless during tests.
    trace.set_tracer_provider(provider)
    _tracer = trace.get_tracer("inference_engine.test")
    _initialized = True
    return exporter


def _reset_for_tests() -> None:
    global _tracer, _initialized
    _tracer = None
    _initialized = False
