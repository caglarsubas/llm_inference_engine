from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from inference_engine import otel
from inference_engine.config import Settings


def config(
    endpoint: str,
    protocol: str,
    headers: str = "",
) -> SimpleNamespace:
    return SimpleNamespace(
        otel_exporter_otlp_endpoint=endpoint,
        otel_exporter_otlp_protocol=protocol,
        otel_exporter_otlp_headers=headers,
    )


def test_settings_accept_only_supported_otlp_protocols() -> None:
    loaded = Settings(
        _env_file=None,
        otel_exporter_otlp_protocol="http/protobuf",
        otel_exporter_otlp_headers="x-api-key=secret",
    )
    assert loaded.otel_exporter_otlp_protocol == "http/protobuf"
    assert loaded.otel_exporter_otlp_headers == "x-api-key=secret"
    assert "otel_exporter_otlp_headers" not in loaded.model_dump()

    with pytest.raises(ValidationError):
        Settings(_env_file=None, otel_exporter_otlp_protocol="http/json")


def test_http_exporter_appends_standard_trace_path_and_parses_headers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class Exporter:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    from opentelemetry.exporter.otlp.proto.http import trace_exporter

    monkeypatch.setattr(trace_exporter, "OTLPSpanExporter", Exporter)
    exporter = otel._build_otlp_exporter(
        config(
            "https://orchestra.example.test/api/v2/otlp",
            "http/protobuf",
            "x-api-key=prm_test,tenant=alpha",
        )
    )

    assert isinstance(exporter, Exporter)
    assert captured == {
        "endpoint": "https://orchestra.example.test/api/v2/otlp",
        "headers": {"x-api-key": "prm_test", "tenant": "alpha"},
    }

    otel._build_otlp_exporter(config("https://collector.example.test", "http/protobuf"))
    assert captured["endpoint"] == "https://collector.example.test/v1/traces"


@pytest.mark.parametrize(
    ("endpoint", "expected_insecure"),
    [
        ("http://otel-collector.example.test:4317", True),
        ("https://otel-collector.example.test:4317", False),
    ],
)
def test_grpc_exporter_derives_transport_security_from_scheme(
    monkeypatch: pytest.MonkeyPatch,
    endpoint: str,
    expected_insecure: bool,
) -> None:
    captured: dict[str, object] = {}

    class Exporter:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    from opentelemetry.exporter.otlp.proto.grpc import trace_exporter

    monkeypatch.setattr(trace_exporter, "OTLPSpanExporter", Exporter)
    exporter = otel._build_otlp_exporter(config(endpoint, "grpc", "authorization=Bearer%20token"))

    assert isinstance(exporter, Exporter)
    assert captured == {
        "endpoint": endpoint,
        "insecure": expected_insecure,
        "headers": {"authorization": "Bearer token"},
    }


@pytest.mark.parametrize(
    ("protocol", "endpoint", "error"),
    [
        ("grpc", "https://collector.example.test/v1/traces", "path_forbidden"),
        ("grpc", "file:///tmp/collector", "endpoint_invalid"),
        ("http/json", "https://collector.example.test", "protocol_invalid"),
        ("http/protobuf", "https://user:pass@collector.example.test", "endpoint_invalid"),
        ("http/protobuf", "https://collector.example.test?token=secret", "endpoint_invalid"),
    ],
)
def test_exporter_rejects_ambiguous_or_secret_bearing_endpoints(
    protocol: str,
    endpoint: str,
    error: str,
) -> None:
    with pytest.raises(ValueError, match=error):
        otel._build_otlp_exporter(config(endpoint, protocol))


@pytest.mark.parametrize(
    "headers",
    [
        "missing-equals",
        "=missing-name",
        "x-api-key=",
        "bad%20name=value",
        "x-api-key=line%0Abreak",
        "x-api-key=one,X-API-Key=two",
    ],
)
def test_exporter_rejects_malformed_or_duplicate_headers(headers: str) -> None:
    with pytest.raises(ValueError, match="otel_exporter_headers_invalid"):
        otel._build_otlp_exporter(
            config(
                "https://orchestra.example.test/api/v2/otlp/v1/traces",
                "http/protobuf",
                headers,
            )
        )


def test_settings_default_span_export_delay_beats_the_otel_default() -> None:
    """OTel's own default is 5000ms. Ours must be materially lower."""
    loaded = Settings(_env_file=None)
    assert loaded.otel_bsp_schedule_delay_millis == 1000

    with pytest.raises(ValidationError):
        Settings(_env_file=None, otel_bsp_schedule_delay_millis=0)


def test_span_export_delay_reaches_the_batch_processor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The configured delay must actually be passed to BatchSpanProcessor.

    Constructing it as ``BatchSpanProcessor(exporter)`` silently reinstates
    OTel's 5s default -- the first of several buffers between a request
    finishing and it reaching a dashboard, and the only one inside this
    process. The setting existing is not evidence that it is wired.
    """
    captured: dict[str, object] = {}

    class RecordingProcessor:
        def __init__(self, exporter: object, **kwargs: object) -> None:
            captured.update(kwargs)

        def on_start(self, *args: object, **kwargs: object) -> None: ...

        def on_end(self, *args: object, **kwargs: object) -> None: ...

        def shutdown(self) -> None: ...

        def force_flush(self, *args: object, **kwargs: object) -> bool:
            return True

    from opentelemetry.sdk.trace import export as sdk_export

    monkeypatch.setattr(sdk_export, "BatchSpanProcessor", RecordingProcessor)
    monkeypatch.setattr(otel.settings, "otel_enabled", True)
    monkeypatch.setattr(otel.settings, "otel_bsp_schedule_delay_millis", 1234)
    # configure_tracing() is idempotent by design; reset the latch so this test
    # exercises a real first initialisation.
    monkeypatch.setattr(otel, "_initialized", False)
    monkeypatch.setattr(otel, "_tracer", None)

    otel.configure_tracing()

    assert captured.get("schedule_delay_millis") == 1234
