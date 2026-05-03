"""observability.span() — structlog + OTel bridge."""

from __future__ import annotations

import pytest

from inference_engine import otel as otel_mod
from inference_engine.observability import span


# Session-scoped fixture lives in conftest.py — it has to be shared across
# every test module, since the OTel TracerProvider can only be set once per
# process and a second installation silently dangles.
@pytest.fixture
def exporter(_session_exporter):
    """Per-test view of the session-scoped exporter; cleared on entry."""
    _session_exporter.clear()
    return _session_exporter


def _by_name(spans):
    return {s.name: s for s in spans}


def test_span_emits_otel_span_with_initial_attributes(exporter) -> None:
    with span("chat.generate", **{"gen_ai.system": "llama_cpp", "gen_ai.request.model": "llama3.2:1b"}):
        pass

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    s = spans[0]
    assert s.name == "chat.generate"
    assert s.attributes["gen_ai.system"] == "llama_cpp"
    assert s.attributes["gen_ai.request.model"] == "llama3.2:1b"
    # duration_ms is set on the way out.
    assert "duration_ms" in s.attributes
    assert s.attributes["duration_ms"] >= 0


def test_bind_mutates_in_place_and_propagates_to_otel(exporter) -> None:
    with span("chat.generate", **{"gen_ai.request.model": "llama3.2:1b"}) as s:
        s.bind(
            **{
                "gen_ai.usage.input_tokens": 44,
                "gen_ai.usage.output_tokens": 7,
                "gen_ai.response.finish_reason": "stop",
            }
        )

    s = _by_name(exporter.get_finished_spans())["chat.generate"]
    # Both initial AND post-bind attrs land on the same span.
    assert s.attributes["gen_ai.request.model"] == "llama3.2:1b"
    assert s.attributes["gen_ai.usage.input_tokens"] == 44
    assert s.attributes["gen_ai.usage.output_tokens"] == 7
    assert s.attributes["gen_ai.response.finish_reason"] == "stop"


def test_exception_inside_span_is_recorded(exporter) -> None:
    with pytest.raises(RuntimeError):
        with span("chat.generate"):
            raise RuntimeError("boom")

    s = exporter.get_finished_spans()[0]
    # OTel records exceptions as events; check we got one.
    assert s.events, "expected an exception event"
    exc_event = s.events[0]
    assert exc_event.name == "exception"
    assert exc_event.attributes["exception.type"] == "RuntimeError"
    assert exc_event.attributes["exception.message"] == "boom"
    # And status flipped to ERROR.
    assert s.status.status_code.name == "ERROR"


def test_non_primitive_attribute_is_coerced(exporter) -> None:
    class Thing:
        def __str__(self) -> str:
            return "thing-as-str"

    with span("opaque", obj=Thing(), nums=[1, 2, 3]):
        pass

    s = exporter.get_finished_spans()[0]
    assert s.attributes["obj"] == "thing-as-str"
    # OTel preserves homogeneous primitive sequences.
    assert tuple(s.attributes["nums"]) == (1, 2, 3)


def test_span_works_when_otel_disabled() -> None:
    """When the tracer is None, start_span yields a NoOpSpan; span() still runs.

    We monkeypatch the module-level _tracer to None for this single test so
    we don't disturb the session-scoped TracerProvider.
    """
    saved = otel_mod._tracer
    otel_mod._tracer = None
    try:
        with span("standalone", x=1) as s:
            s.bind(y=2)
        assert s.log is not None  # structlog handle still works
    finally:
        otel_mod._tracer = saved
