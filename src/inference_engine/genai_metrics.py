"""OpenTelemetry GenAI semantic-convention metrics.

The engine already emits rich *spans*, but every metric it published was named
``inference_engine_*`` — which means no off-the-shelf GenAI dashboard (Grafana,
Datadog, Honeycomb) lights up against it, and the two numbers that matter most
for serving, **time to first token** and **time per output token**, weren't
measured anywhere at all.

This module records the four standard instruments:

* ``gen_ai.client.token.usage`` — tokens in/out, split by ``gen_ai.token.type``
* ``gen_ai.client.operation.duration`` — end-to-end seconds per operation
* ``gen_ai.server.time_to_first_token`` — prefill latency (streaming only)
* ``gen_ai.server.time_per_output_token`` — inter-token latency (streaming only)

Rendering
---------

Output is the Prometheus text form of those instruments — dots become
underscores, exactly the mapping an OTel Collector applies on the way to a
Prometheus remote-write. A dashboard built against a Collector-scraped vLLM
deployment therefore reads these series unchanged. What is *not* wired here is
OTLP metrics export: ``otel.py`` installs a TracerProvider only, so metrics
reach you by scrape, not by push.

Cardinality
-----------

Label sets are ``(operation, provider, model)``, all engine-controlled except
the model name, which a caller can influence via an unknown-model request that
still reaches a span. ``_MAX_SERIES`` bounds the dictionary; past the cap new
label sets are folded into ``model="__other__"`` rather than growing without
limit or being dropped silently.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

# Boundaries recommended by the GenAI semantic conventions. Latency buckets are
# the doubling sequence from 10ms to ~82s; token buckets are powers of four
# from 1 to ~67M so a long-context prefill still lands in a real bucket.
_DURATION_BUCKETS: tuple[float, ...] = (
    0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28,
    2.56, 5.12, 10.24, 20.48, 40.96, 81.92,
)
_TOKEN_BUCKETS: tuple[float, ...] = (
    1, 4, 16, 64, 256, 1024, 4096, 16384,
    65536, 262144, 1048576, 4194304, 16777216, 67108864,
)
_TPOT_BUCKETS: tuple[float, ...] = (
    0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 2.5,
)

_MAX_SERIES = 2048
_OVERFLOW_MODEL = "__other__"


@dataclass
class _Histogram:
    """A minimal cumulative histogram in Prometheus semantics."""

    buckets: tuple[float, ...]
    counts: list[int] = field(default_factory=list)
    total: float = 0.0
    count: int = 0

    def __post_init__(self) -> None:
        if not self.counts:
            self.counts = [0] * len(self.buckets)

    def observe(self, value: float) -> None:
        self.total += value
        self.count += 1
        for index, boundary in enumerate(self.buckets):
            if value <= boundary:
                self.counts[index] += 1
        # Values above the last boundary are still in +Inf, which is ``count``.


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


class GenAIMetrics:
    """Process-wide GenAI metric registry.

    Every method is safe to call from a worker thread; observations take a
    short lock rather than being made lock-free, because the write rate is
    bounded by request throughput and contention here is irrelevant next to
    the cost of a token.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # key -> histogram, where key is the ordered label tuple.
        self._tokens: dict[tuple[str, str, str, str], _Histogram] = {}
        self._duration: dict[tuple[str, str, str], _Histogram] = {}
        self._ttft: dict[tuple[str, str, str], _Histogram] = {}
        self._tpot: dict[tuple[str, str, str], _Histogram] = {}

    def reset(self) -> None:
        """Drop every series. Used by tests; not called in serving."""
        with self._lock:
            self._tokens.clear()
            self._duration.clear()
            self._ttft.clear()
            self._tpot.clear()

    def _bounded_model(self, store: dict, key_prefix: tuple, model: str) -> str:
        """Fold a new model label into ``__other__`` once the cap is reached."""
        if (*key_prefix, model) in store or len(store) < _MAX_SERIES:
            return model
        return _OVERFLOW_MODEL

    def record_operation(
        self,
        *,
        operation: str,
        provider: str,
        model: str,
        duration_seconds: float,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
    ) -> None:
        with self._lock:
            model_label = self._bounded_model(self._duration, (operation, provider), model)
            key = (operation, provider, model_label)
            self._duration.setdefault(key, _Histogram(_DURATION_BUCKETS)).observe(
                max(0.0, duration_seconds)
            )
            for token_type, value in (("input", input_tokens), ("output", output_tokens)):
                if value is None:
                    continue
                token_key = (operation, provider, model_label, token_type)
                self._tokens.setdefault(token_key, _Histogram(_TOKEN_BUCKETS)).observe(
                    max(0, int(value))
                )

    def record_time_to_first_token(
        self, *, operation: str, provider: str, model: str, seconds: float
    ) -> None:
        with self._lock:
            model_label = self._bounded_model(self._ttft, (operation, provider), model)
            key = (operation, provider, model_label)
            self._ttft.setdefault(key, _Histogram(_DURATION_BUCKETS)).observe(max(0.0, seconds))

    def record_time_per_output_token(
        self, *, operation: str, provider: str, model: str, seconds: float
    ) -> None:
        with self._lock:
            model_label = self._bounded_model(self._tpot, (operation, provider), model)
            key = (operation, provider, model_label)
            self._tpot.setdefault(key, _Histogram(_TPOT_BUCKETS)).observe(max(0.0, seconds))

    # ------------------------------------------------------------------
    # Prometheus rendering
    # ------------------------------------------------------------------

    def render(self) -> list[str]:
        lines: list[str] = []
        with self._lock:
            lines.extend(
                _render_histograms(
                    "gen_ai_client_token_usage",
                    "Number of input and output tokens used.",
                    self._tokens,
                    ("gen_ai_operation_name", "gen_ai_provider_name", "gen_ai_request_model",
                     "gen_ai_token_type"),
                )
            )
            lines.extend(
                _render_histograms(
                    "gen_ai_client_operation_duration_seconds",
                    "GenAI operation duration.",
                    self._duration,
                    ("gen_ai_operation_name", "gen_ai_provider_name", "gen_ai_request_model"),
                )
            )
            lines.extend(
                _render_histograms(
                    "gen_ai_server_time_to_first_token_seconds",
                    "Time to generate the first token for successful streaming responses.",
                    self._ttft,
                    ("gen_ai_operation_name", "gen_ai_provider_name", "gen_ai_request_model"),
                )
            )
            lines.extend(
                _render_histograms(
                    "gen_ai_server_time_per_output_token_seconds",
                    "Time per output token generated after the first token.",
                    self._tpot,
                    ("gen_ai_operation_name", "gen_ai_provider_name", "gen_ai_request_model"),
                )
            )
        return lines


def _render_histograms(
    name: str,
    help_text: str,
    store: dict,
    label_names: tuple[str, ...],
) -> list[str]:
    if not store:
        return []
    lines = [f"# HELP {name} {help_text}", f"# TYPE {name} histogram"]
    for key, hist in sorted(store.items()):
        labels = ",".join(
            f'{label}="{_escape(str(value))}"'
            for label, value in zip(label_names, key, strict=True)
        )
        for boundary, count in zip(hist.buckets, hist.counts, strict=True):
            lines.append(f'{name}_bucket{{{labels},le="{boundary:g}"}} {count}')
        lines.append(f'{name}_bucket{{{labels},le="+Inf"}} {hist.count}')
        lines.append(f"{name}_sum{{{labels}}} {hist.total:.6f}")
        lines.append(f"{name}_count{{{labels}}} {hist.count}")
    return lines


# Process-wide singleton — the metrics endpoint and the routes share it.
genai_metrics = GenAIMetrics()

__all__ = ["GenAIMetrics", "genai_metrics"]
