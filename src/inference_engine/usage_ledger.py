"""Per-request billing ledger — one ``prometa.model-usage.v2`` record per call.

Chargeback needs a record that exists for *every* priced request, including the
ones that never reach a model: a policy denial opens no ``chat.generate`` span,
and a sampled trace is structurally unfit for billing because sampling drops
exactly the records an invoice needs. So the ledger is its own channel rather
than a span attribute or a span event.

Transport is a dedicated structlog JSON logger on stdout, drained by a
background task and shipped by whatever log pipeline the deployment already
runs. It is deliberately *not* an OTLP log record: every ``opentelemetry-*``
package sits behind the optional ``otel`` extra (``make install`` is a plain
``uv sync``), so an OTel-only ledger would simply not exist in a default
deployment — unacceptable for a billing artefact — and the logs SDK lives in
the private ``opentelemetry.sdk._logs`` namespace whose record type has already
been renamed inside the ``>=1.27`` range this project pins. ``_sink`` is the
single seam where an OTLP-logs or file sink can be added later without touching
a single bind site. Correlation back to traces goes through the stable
``opentelemetry.trace`` *API* instead (see ``otel.current_trace_context``).

The record carries **metadata only**. ``SCHEMA_FIELDS`` is the reviewed key set
and ``_render`` iterates exactly that list rather than ``dataclasses.asdict``,
so a field added to the accumulator cannot leak into an emitted record without
a deliberate schema change. The two are coupled at import by
``_check_schema_coverage``: every accumulator attribute must be either a schema
field or declared control state, and every schema field must be either computed
by ``_render`` or backed by an attribute. Neither list can drift from the other
without failing at import. No prompt, completion, tool argument, or embedding
input is reachable from any of these fields.

Everything here is off unless ``USAGE_LEDGER_ENABLED=true``: ``begin()`` returns
``None``, ``bind()`` becomes a no-op, and no buffer, task, or logger is ever
touched.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections import deque
from contextvars import ContextVar
from dataclasses import dataclass, field, fields
from datetime import UTC, datetime
from typing import Any

import structlog

from .config import settings
from .observability import get_logger
from .otel import current_trace_context
from .request_identity import new_usage_record_id

log = get_logger("usage_ledger")

SCHEMA = "prometa.model-usage.v2"
EVENT = "prometa.model-usage"

# The exact, reviewed key set of an emitted record, in emission order. Anything
# not listed here is not billing metadata and does not leave the process on this
# channel. ``_render`` walks this tuple, so the list and the payload are one
# thing rather than two that agree by hand.
_SCHEMA_FIELD_ORDER: tuple[str, ...] = (
    "schema",
    "event",
    "timestamp",
    "usage_record_id",
    "engine_request_id",
    "runtime_request_id",
    "model_invocation_id",
    "model_attempt_id",
    "route",
    "operation",
    "stream",
    "tenant",
    "org_id",
    "key_id",
    "outcome",
    "finish_reason",
    "http_status",
    "error_type",
    "denial_code",
    "requested_model",
    "resolved_model",
    "backend",
    "request_key_source",
    "policy_id",
    "policy_digest",
    "route_id",
    "pricing_digest",
    "input_token_upper_bound",
    "output_token_budget",
    "input_tokens",
    "output_tokens",
    "cached_tokens",
    "cost_micros",
    "fallback",
    "fallback_from_model",
    "fallback_from_backend",
    "fallback_reason",
    "duration_ms",
    "ttft_ms",
    "trace_id",
    "span_id",
)

SCHEMA_FIELDS = frozenset(_SCHEMA_FIELD_ORDER)


def _computed() -> dict:
    """The schema keys ``_render`` computes rather than reads off the record."""
    return {
        "schema": SCHEMA,
        "event": EVENT,
        "timestamp": datetime.now(UTC).isoformat(),
    }


_COMPUTED_FIELDS = frozenset(_computed())

# Accumulator attributes that drive behaviour instead of describing the call.
# They are the only ones allowed to exist without a schema key.
_CONTROL_FIELDS = frozenset({"started_at", "flushed", "billable"})


@dataclass(slots=True)
class UsageRecord:
    """Request-scoped accumulator, filled in by the seams a request passes.

    ``started_at``, ``flushed``, and ``billable`` are control state and never
    emitted; every other attribute maps 1:1 onto a ``SCHEMA_FIELDS`` key.

    **Attribution can be null on a billable record.** ``_usage.bind_request``
    declares the request billable *before* it maps the caller onto ``tenant``,
    ``org_id``, and ``key_id``, so a fault in that mapping yields a record with
    those three (and ``operation``, ``requested_model``, the two token bounds)
    left at ``None``. That ordering is deliberate: sharing one failure boundary
    would let a mapping bug delete the invoice line entirely, and a record that
    says *some* request happened is worth more than silence. Only the fields
    that failed are missing — everything bound later, including
    ``resolved_model``, the token counts, and ``cost_micros``, is intact, and
    ``usage_record_id`` and ``engine_request_id`` are always set.

    Reconciliation must therefore treat ``tenant is None`` as *unattributed*,
    never as a tenant named ``null`` and never as absent traffic: the call was
    served and cost real capacity. Route those records to an exceptions queue
    keyed by ``usage_record_id``. ``engine_request_id`` correlates the line
    back to the request logs that do carry the caller. They should be rare
    enough to alarm on; a non-zero rate means a bug in the attribution binder,
    not a billing edge case, and the ``usage_ledger.bind_failed`` warning
    naming ``_bind_attribution`` is emitted at the same moment.
    """

    engine_request_id: str
    route: str
    started_at: float
    usage_record_id: str = field(default_factory=new_usage_record_id)
    runtime_request_id: str | None = None
    model_invocation_id: str | None = None
    model_attempt_id: str | None = None
    flushed: bool = False
    billable: bool = False
    operation: str | None = None
    stream: bool = False
    tenant: str | None = None
    org_id: str | None = None
    key_id: str | None = None
    outcome: str | None = None
    finish_reason: str | None = None
    http_status: int | None = None
    error_type: str | None = None
    denial_code: str | None = None
    requested_model: str | None = None
    resolved_model: str | None = None
    backend: str | None = None
    request_key_source: str | None = None
    policy_id: str | None = None
    policy_digest: str | None = None
    route_id: str | None = None
    pricing_digest: str | None = None
    input_token_upper_bound: int | None = None
    output_token_budget: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cached_tokens: int | None = None
    cost_micros: int | None = None
    fallback: bool = False
    fallback_from_model: str | None = None
    fallback_from_backend: str | None = None
    fallback_reason: str | None = None
    duration_ms: float | None = None
    ttft_ms: float | None = None
    trace_id: str | None = None
    span_id: str | None = None


def _check_schema_coverage(
    *,
    schema_fields: frozenset[str],
    record_fields: frozenset[str],
) -> None:
    """Fail at import if the key set and the accumulator have drifted apart.

    Import time, not request time: a schema key with no attribute behind it
    would otherwise surface as a swallowed ``AttributeError`` inside ``flush``,
    which loses the invoice line and reports nothing but a warning.
    """
    unknown = record_fields - _CONTROL_FIELDS - schema_fields
    if unknown:
        raise RuntimeError(
            "usage ledger accumulator fields are neither emitted nor declared "
            f"control state: {sorted(unknown)}"
        )
    missing = schema_fields - _COMPUTED_FIELDS - record_fields
    if missing:
        raise RuntimeError(
            f"usage ledger schema fields have no accumulator attribute: {sorted(missing)}"
        )
    control = record_fields & _CONTROL_FIELDS
    if control != _CONTROL_FIELDS:
        raise RuntimeError(
            "usage ledger control fields are not all present on the accumulator: "
            f"{sorted(_CONTROL_FIELDS - control)}"
        )
    uncarried = _COMPUTED_FIELDS - schema_fields
    if uncarried:
        raise RuntimeError(f"usage ledger computed fields are not emitted: {sorted(uncarried)}")


_check_schema_coverage(
    schema_fields=SCHEMA_FIELDS,
    record_fields=frozenset(field.name for field in fields(UsageRecord)),
)


_current: ContextVar[UsageRecord | None] = ContextVar("usage_record", default=None)


def _report(operation: str, exc: BaseException) -> None:
    log.warning(
        "usage_ledger.failed",
        operation=operation,
        error_type=type(exc).__name__,
    )


def begin(
    *,
    engine_request_id: str,
    route: str,
    runtime_request_id: str | None = None,
    model_invocation_id: str | None = None,
    model_attempt_id: str | None = None,
) -> UsageRecord | None:
    """Open a record for this request, or ``None`` when the ledger is off."""
    if not settings.usage_ledger_enabled:
        # Clear rather than leave whatever the surrounding context held: a
        # disabled ledger must not let a stale record collect binds.
        _current.set(None)
        return None
    try:
        record = UsageRecord(
            engine_request_id=engine_request_id,
            route=route,
            started_at=time.perf_counter(),
            runtime_request_id=runtime_request_id,
            model_invocation_id=model_invocation_id,
            model_attempt_id=model_attempt_id,
        )
    except Exception as exc:  # noqa: BLE001 - the ledger must not fail a request
        _report("begin", exc)
        return None
    _current.set(record)
    return record


def current() -> UsageRecord | None:
    return _current.get()


def bind(**fields: Any) -> None:
    """Set accumulator fields for the in-flight request.

    A no-op outside a request scope, which is what lets the route coroutines be
    called directly (as most of the test suite does) with no ledger wiring.
    """
    record = _current.get()
    if record is None:
        return
    try:
        for key, value in fields.items():
            setattr(record, key, value)
    except Exception as exc:  # noqa: BLE001 - the ledger must not fail a request
        _report("bind", exc)


def mark_billable() -> None:
    """Declare the in-flight request a billable event.

    Call this only from a seam that has both a resolved caller identity and a
    request the schema accepted — that pair is what makes a record an invoice
    line. Anything rejected ahead of it (no credentials, a body that fails
    validation) is attributable to nobody and never reached the priced path, so
    ``flush`` refuses to submit it: the buffer is bounded, and an unattributed
    record in it costs a real one. Without this gate an unauthenticated caller
    chooses which invoice lines survive.
    """
    record = _current.get()
    if record is None:
        return
    record.billable = True


def capture_trace_context() -> None:
    """Stamp the ids of the span in flight so the record joins its trace.

    Called from the serving seam rather than from ``flush``: by flush time the
    generate/stream span has closed, and the ids of whatever span happens to be
    current then describe the HTTP hop, not the model call being billed.
    """
    record = _current.get()
    if record is None:
        return
    try:
        context = current_trace_context()
    except Exception as exc:  # noqa: BLE001 - the ledger must not fail a request
        _report("capture_trace_context", exc)
        return
    if context is None:
        return
    record.trace_id, record.span_id = context


def outcome_for_status(status: int | None) -> str:
    if status is None:
        return "error"
    if 200 <= status < 300:
        return "ok"
    if status == 504:
        return "timeout"
    if status in {401, 403, 429}:
        return "denied"
    return "error"


def _render(record: UsageRecord) -> dict:
    computed = _computed()
    return {
        name: computed[name] if name in computed else getattr(record, name)
        for name in _SCHEMA_FIELD_ORDER
    }


def flush(record: UsageRecord | None) -> None:
    """Snapshot the record into the buffer. Idempotent, and never raises.

    A record nobody called ``mark_billable`` on is discarded here rather than
    submitted, so it never occupies a slot in the bounded buffer.

    Snapshotting matters on the streaming-fallback path: the inner recursive
    ``_stream_response`` flushes the model that actually served, and the outer
    one then binds its own stale ``finish_reason`` on the way out. Because the
    emitted payload was already copied, those late writes are inert.
    """
    if record is None or record.flushed:
        return
    # Set before doing any work, so a failure below cannot leave the record
    # eligible for a second, half-built emission.
    record.flushed = True
    if not record.billable:
        return
    try:
        if record.outcome is None:
            # A policy denial is a denial whatever status it wore — the
            # bounds checks answer 400 and the rate limiter 429.
            record.outcome = (
                "denied" if record.denial_code else outcome_for_status(record.http_status)
            )
        record.duration_ms = round((time.perf_counter() - record.started_at) * 1000, 3)
        usage_ledger.submit(_render(record))
    except Exception as exc:  # noqa: BLE001 - the ledger must not fail a request
        _report("flush", exc)


class UsageLedger:
    """Bounded in-memory hand-off between the request path and the drain task.

    ``submit`` does a dict append under a short lock and no I/O at all — the
    constraint ranks request latency above ledger completeness, so a full
    buffer drops rather than applying backpressure to inference. It is the
    *arriving* record that is dropped: one already buffered is an invoice line
    the drain task still owes, and a burst must not be able to destroy it.

    Loss is never silent. Every drop increments ``dropped_total`` (scraped off
    ``/v1/metrics``) and each overflow episode logs once — a ledger that
    quietly discards invoice lines is indistinguishable from one that discards
    none, which is the failure mode an invoice cannot tolerate.
    """

    def __init__(self, max_buffer: int) -> None:
        self._lock = threading.Lock()
        self._records: deque[dict] = deque(maxlen=max_buffer)
        self._emitted_total = 0
        self._dropped_total = 0
        self._sink_failures_total = 0
        self._overflowing = False

    def submit(self, record: dict) -> None:
        with self._lock:
            if len(self._records) != self._records.maxlen:
                self._records.append(record)
                self._emitted_total += 1
                self._overflowing = False
                return
            self._dropped_total += 1
            announce = not self._overflowing
            self._overflowing = True
            dropped_total = self._dropped_total
            max_buffer = self._records.maxlen
        # Once per episode, not once per drop: sustained overflow must not turn
        # a lost invoice line into a lost log pipeline.
        if announce:
            log.warning(
                "usage_ledger.buffer_full",
                max_buffer=max_buffer,
                dropped_total=dropped_total,
            )

    def peek(self) -> list[dict]:
        """Copy the buffered records without giving up the slots they hold.

        The drain task ships this batch and only then calls ``commit``, so a
        cancellation mid-sink leaves the records where the next drain will find
        them instead of dropping them on the floor.
        """
        with self._lock:
            return list(self._records)

    def commit(self, count: int) -> None:
        """Retire the ``count`` oldest records, which the sink has accepted.

        Correct only because the drain task is the sole remover and ``submit``
        only ever appends: the first ``count`` records are exactly the ones
        ``peek`` returned.
        """
        with self._lock:
            for _ in range(min(count, len(self._records))):
                self._records.popleft()

    def note_sink_failure(self, count: int) -> None:
        with self._lock:
            self._sink_failures_total += count

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "emitted_total": self._emitted_total,
                "dropped_total": self._dropped_total,
                "sink_failures_total": self._sink_failures_total,
                "buffered": len(self._records),
            }

    def _reset(self, max_buffer: int) -> None:
        with self._lock:
            self._records = deque(maxlen=max_buffer)
            self._emitted_total = 0
            self._dropped_total = 0
            self._sink_failures_total = 0
            self._overflowing = False


usage_ledger = UsageLedger(settings.usage_ledger_max_buffer)

_sink_logger: Any = None


def _ledger_logger() -> Any:
    """Build the JSON emitter on first use, independent of global log config.

    ``configure_logging()`` installs a ConsoleRenderer and only runs inside the
    ASGI lifespan, so a logger built at import time would render the ledger as
    coloured prose (or not at all). Filtering is pinned below INFO because this
    is a billing artefact, not diagnostics — ``LOG_LEVEL=WARNING`` must not
    silently discard invoice lines.
    """
    global _sink_logger
    if _sink_logger is None:
        _sink_logger = structlog.wrap_logger(
            structlog.PrintLogger(),
            processors=[structlog.processors.JSONRenderer()],
            wrapper_class=structlog.make_filtering_bound_logger(logging.DEBUG),
        )
    return _sink_logger


def _sink(batch: list[dict]) -> None:
    """Write one drained batch out. Blocking — always call it off the loop."""
    logger = _ledger_logger()
    for record in batch:
        fields = {key: value for key, value in record.items() if key != "event"}
        logger.info(record["event"], **fields)


async def drain_once(ledger: UsageLedger) -> None:
    """Ship one batch, retiring it from the buffer only once the sink took it.

    Shutdown cancels this task, and a cancellation landing between the read and
    the sink used to lose the batch with nothing counting it — the one loss path
    ``dropped_total`` could not see. Holding the records until the sink returns
    turns that window into a *possible duplicate* window instead: the worker
    thread the cancellation abandons may still complete its writes, and the
    post-cancel drain in the lifespan then ships the same records again.
    ``usage_record_id`` is the sole dedupe key for that uncertain-acceptance
    window. It is generated when the accumulator is constructed and the same
    rendered dict remains buffered across the retry, so a duplicate retains
    the same value.

    This is not a durable outbox or a general at-least-once channel. Buffer
    overflow refuses the arriving record, and a sink *failure* discards the
    failed batch: retrying a batch a broken sink refuses would wedge the buffer
    and starve every record behind it. Both loss paths are counted.
    """
    batch = ledger.peek()
    if not batch:
        return
    try:
        await asyncio.to_thread(_sink, batch)
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 - a sink failure must not stop the loop
        ledger.commit(len(batch))
        ledger.note_sink_failure(len(batch))
        log.warning(
            "usage_ledger.sink_failed",
            error_type=type(exc).__name__,
            records=len(batch),
        )
        return
    ledger.commit(len(batch))


async def run_usage_ledger_drain(ledger: UsageLedger, *, interval_seconds: float) -> None:
    """Ship buffered records on a fixed cadence until cancelled."""
    while True:
        await drain_once(ledger)
        await asyncio.sleep(interval_seconds)


def _reset_for_tests() -> None:
    global _sink_logger
    _current.set(None)
    _sink_logger = None
    usage_ledger._reset(settings.usage_ledger_max_buffer)


__all__ = [
    "EVENT",
    "SCHEMA",
    "SCHEMA_FIELDS",
    "UsageLedger",
    "UsageRecord",
    "begin",
    "bind",
    "capture_trace_context",
    "current",
    "drain_once",
    "flush",
    "mark_billable",
    "outcome_for_status",
    "run_usage_ledger_drain",
    "usage_ledger",
]
