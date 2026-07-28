"""OTel GenAI semantic-convention metrics and their Prometheus rendering."""

from __future__ import annotations

import pytest

from inference_engine.genai_metrics import GenAIMetrics


@pytest.fixture
def metrics() -> GenAIMetrics:
    return GenAIMetrics()


def _series(lines: list[str], prefix: str) -> list[str]:
    return [line for line in lines if line.startswith(prefix)]


def test_renders_nothing_before_any_observation(metrics: GenAIMetrics) -> None:
    assert metrics.render() == []


def test_operation_duration_uses_semconv_metric_name(metrics: GenAIMetrics) -> None:
    metrics.record_operation(
        operation="chat", provider="llama_cpp", model="m:1", duration_seconds=0.5
    )
    lines = metrics.render()
    assert any(
        line.startswith("# TYPE gen_ai_client_operation_duration_seconds histogram")
        for line in lines
    )
    counts = _series(lines, "gen_ai_client_operation_duration_seconds_count")
    assert len(counts) == 1
    assert counts[0].endswith(" 1")


def test_token_usage_splits_input_and_output(metrics: GenAIMetrics) -> None:
    metrics.record_operation(
        operation="chat",
        provider="vllm",
        model="m:1",
        duration_seconds=1.0,
        input_tokens=100,
        output_tokens=20,
    )
    lines = metrics.render()
    sums = _series(lines, "gen_ai_client_token_usage_sum")
    assert len(sums) == 2
    assert any('gen_ai_token_type="input"' in line and line.endswith(" 100.000000") for line in sums)
    assert any('gen_ai_token_type="output"' in line and line.endswith(" 20.000000") for line in sums)


def test_labels_carry_semconv_attribute_names(metrics: GenAIMetrics) -> None:
    metrics.record_operation(
        operation="chat", provider="mlx", model="qwen3:1b", duration_seconds=0.1
    )
    line = _series(metrics.render(), "gen_ai_client_operation_duration_seconds_count")[0]
    assert 'gen_ai_operation_name="chat"' in line
    assert 'gen_ai_provider_name="mlx"' in line
    assert 'gen_ai_request_model="qwen3:1b"' in line


def test_ttft_and_tpot_are_separate_instruments(metrics: GenAIMetrics) -> None:
    metrics.record_time_to_first_token(
        operation="chat", provider="llama_cpp", model="m:1", seconds=0.25
    )
    metrics.record_time_per_output_token(
        operation="chat", provider="llama_cpp", model="m:1", seconds=0.02
    )
    lines = metrics.render()
    assert _series(lines, "gen_ai_server_time_to_first_token_seconds_count")
    assert _series(lines, "gen_ai_server_time_per_output_token_seconds_count")


def test_histogram_buckets_are_cumulative(metrics: GenAIMetrics) -> None:
    for value in (0.005, 0.05, 5.0):
        metrics.record_operation(
            operation="chat", provider="p", model="m", duration_seconds=value
        )
    buckets = _series(metrics.render(), "gen_ai_client_operation_duration_seconds_bucket")
    by_le = {line.split('le="')[1].split('"')[0]: int(line.rsplit(" ", 1)[1]) for line in buckets}
    # 0.005 lands in every bucket; 0.05 in 0.08 and up; 5.0 only from 5.12.
    assert by_le["0.01"] == 1
    assert by_le["0.08"] == 2
    assert by_le["5.12"] == 3
    assert by_le["+Inf"] == 3


def test_values_above_the_last_boundary_still_count_in_inf(metrics: GenAIMetrics) -> None:
    metrics.record_operation(
        operation="chat", provider="p", model="m", duration_seconds=1000.0
    )
    buckets = _series(metrics.render(), "gen_ai_client_operation_duration_seconds_bucket")
    by_le = {line.split('le="')[1].split('"')[0]: int(line.rsplit(" ", 1)[1]) for line in buckets}
    assert by_le["81.92"] == 0
    assert by_le["+Inf"] == 1


def test_negative_durations_are_clamped_not_recorded_as_negative(
    metrics: GenAIMetrics,
) -> None:
    metrics.record_operation(
        operation="chat", provider="p", model="m", duration_seconds=-1.0
    )
    total = _series(metrics.render(), "gen_ai_client_operation_duration_seconds_sum")[0]
    assert total.endswith(" 0.000000")


def test_model_label_cardinality_is_bounded(metrics: GenAIMetrics) -> None:
    from inference_engine import genai_metrics as mod

    for index in range(mod._MAX_SERIES + 50):
        metrics.record_operation(
            operation="chat", provider="p", model=f"m{index}", duration_seconds=0.01
        )
    counts = _series(metrics.render(), "gen_ai_client_operation_duration_seconds_count")
    assert len(counts) <= mod._MAX_SERIES + 1
    # Overflow is folded into a visible bucket rather than silently dropped.
    assert any('gen_ai_request_model="__other__"' in line for line in counts)


def test_label_values_are_escaped(metrics: GenAIMetrics) -> None:
    metrics.record_operation(
        operation="chat", provider="p", model='we"ird', duration_seconds=0.01
    )
    line = _series(metrics.render(), "gen_ai_client_operation_duration_seconds_count")[0]
    assert 'gen_ai_request_model="we\\"ird"' in line


def test_reset_clears_every_instrument(metrics: GenAIMetrics) -> None:
    metrics.record_operation(operation="chat", provider="p", model="m", duration_seconds=1.0)
    metrics.record_time_to_first_token(operation="chat", provider="p", model="m", seconds=0.1)
    metrics.reset()
    assert metrics.render() == []
