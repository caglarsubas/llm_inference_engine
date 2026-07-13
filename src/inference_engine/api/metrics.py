"""Lightweight Prometheus-compatible /v1/metrics endpoint.

Hand-rolled rather than pulling in `prometheus-client` so the dependency
surface stays small. Outputs the OpenMetrics text format Prometheus scrapes.
"""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import PlainTextResponse

from .. import __version__
from .state import app_state

router = APIRouter()


def _label_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


@router.get("/v1/metrics", response_class=PlainTextResponse)
async def metrics() -> str:
    lines: list[str] = []

    lines.append("# HELP inference_engine_info Engine build info.")
    lines.append("# TYPE inference_engine_info gauge")
    lines.append(
        f'inference_engine_info{{version="{__version__}",backend="{app_state.backend_name}"}} 1'
    )

    observer = app_state.model_plane_observer
    observer_metrics = observer.metrics_snapshot if observer is not None else {}
    lines.append(
        "# HELP inference_engine_model_plane_observer_enabled "
        "Asynchronous model-plane observer enabled flag."
    )
    lines.append("# TYPE inference_engine_model_plane_observer_enabled gauge")
    lines.append(
        f"inference_engine_model_plane_observer_enabled {1 if observer is not None else 0}"
    )
    for metric, metric_type, help_text in (
        ("running", "gauge", "Model-plane observer background loop running flag."),
        ("attempts_total", "counter", "Model-plane observation cycles attempted."),
        ("successes_total", "counter", "Model-plane observations accepted."),
        ("failures_total", "counter", "Model-plane observation cycles failed."),
        ("consecutive_failures", "gauge", "Consecutive model-plane observation failures."),
        ("pending", "gauge", "Observation retained for idempotent retry."),
        (
            "last_success_unixtime",
            "gauge",
            "Unix timestamp of the last accepted model-plane observation.",
        ),
    ):
        name = f"inference_engine_model_plane_observer_{metric}"
        lines.append(f"# HELP {name} {help_text}")
        lines.append(f"# TYPE {name} {metric_type}")
        value = observer_metrics.get(metric)
        lines.append(f"{name} {value if value is not None else 0}")

    lines.append("# HELP inference_engine_models_loaded Number of models currently loaded.")
    lines.append("# TYPE inference_engine_models_loaded gauge")
    lines.append(f"inference_engine_models_loaded {len(app_state.manager.loaded_models())}")

    lines.append("# HELP inference_engine_loaded_bytes Total bytes of loaded model weights on disk.")
    lines.append("# TYPE inference_engine_loaded_bytes gauge")
    lines.append(f"inference_engine_loaded_bytes {app_state.manager.loaded_bytes}")

    lines.append("# HELP inference_engine_memory_budget_bytes Configured hot-keep memory budget.")
    lines.append("# TYPE inference_engine_memory_budget_bytes gauge")
    lines.append(f"inference_engine_memory_budget_bytes {app_state.manager.memory_budget_bytes}")

    lines.append("# HELP inference_engine_models_available Total models discoverable in the registry.")
    lines.append("# TYPE inference_engine_models_available gauge")
    lines.append(f"inference_engine_models_available {len(app_state.registry.list_models())}")

    # Per-loaded-adapter prefix cache state (one series per model).
    lines.append("# HELP inference_engine_prefix_cache_capacity_bytes Configured prefix-cache capacity (per loaded model).")
    lines.append("# TYPE inference_engine_prefix_cache_capacity_bytes gauge")
    lines.append("# HELP inference_engine_prefix_cache_size_bytes Bytes of cached prefix state currently held.")
    lines.append("# TYPE inference_engine_prefix_cache_size_bytes gauge")
    for name, adapter, _desc in app_state.manager.iter_loaded():
        cap = getattr(adapter, "prefix_cache_capacity_bytes", 0)
        size = getattr(adapter, "prefix_cache_size_bytes", 0)
        labels = f'{{model="{name}",backend="{adapter.backend_name}"}}'
        lines.append(f"inference_engine_prefix_cache_capacity_bytes{labels} {cap}")
        lines.append(f"inference_engine_prefix_cache_size_bytes{labels} {size}")

    sched = app_state.scheduler.snapshot()
    lines.append("# HELP inference_engine_scheduler_enabled Tenant scheduler enabled flag.")
    lines.append("# TYPE inference_engine_scheduler_enabled gauge")
    lines.append(f"inference_engine_scheduler_enabled {1 if sched.enabled else 0}")
    lines.append("# HELP inference_engine_scheduler_queued Requests waiting in tenant queues.")
    lines.append("# TYPE inference_engine_scheduler_queued gauge")
    lines.append(f"inference_engine_scheduler_queued {sched.total_queued}")
    lines.append("# HELP inference_engine_scheduler_in_flight Requests currently holding scheduler slots.")
    lines.append("# TYPE inference_engine_scheduler_in_flight gauge")
    lines.append(f"inference_engine_scheduler_in_flight {sched.total_in_flight}")
    lines.append("# HELP inference_engine_scheduler_accepted_total Requests accepted into scheduler queues.")
    lines.append("# TYPE inference_engine_scheduler_accepted_total counter")
    lines.append(f"inference_engine_scheduler_accepted_total {sched.accepted_total}")
    lines.append("# HELP inference_engine_scheduler_rejected_total Requests rejected by tenant admission.")
    lines.append("# TYPE inference_engine_scheduler_rejected_total counter")
    lines.append(f"inference_engine_scheduler_rejected_total {sched.rejected_total}")
    lines.append("# HELP inference_engine_scheduler_timed_out_total Requests timed out waiting for scheduler capacity.")
    lines.append("# TYPE inference_engine_scheduler_timed_out_total counter")
    lines.append(f"inference_engine_scheduler_timed_out_total {sched.timed_out_total}")
    lines.append("# HELP inference_engine_scheduler_completed_total Requests released from scheduler slots.")
    lines.append("# TYPE inference_engine_scheduler_completed_total counter")
    lines.append(f"inference_engine_scheduler_completed_total {sched.completed_total}")
    lines.append("# HELP inference_engine_scheduler_wait_seconds_total Total scheduler wait time.")
    lines.append("# TYPE inference_engine_scheduler_wait_seconds_total counter")
    lines.append(f"inference_engine_scheduler_wait_seconds_total {sched.wait_seconds_total:.6f}")
    lines.append("# HELP inference_engine_scheduler_wait_observations_total Scheduler wait observations.")
    lines.append("# TYPE inference_engine_scheduler_wait_observations_total counter")
    lines.append(f"inference_engine_scheduler_wait_observations_total {sched.wait_observations_total}")
    lines.append("# HELP inference_engine_scheduler_max_queue_wait_seconds Longest observed scheduler wait.")
    lines.append("# TYPE inference_engine_scheduler_max_queue_wait_seconds gauge")
    lines.append(f"inference_engine_scheduler_max_queue_wait_seconds {sched.max_queue_wait_seconds:.6f}")

    lines.append("# HELP inference_engine_scheduler_queued_by_tenant Requests waiting by tenant.")
    lines.append("# TYPE inference_engine_scheduler_queued_by_tenant gauge")
    for tenant, count in sched.queued_by_tenant.items():
        labels = f'{{tenant="{_label_value(tenant)}"}}'
        lines.append(f"inference_engine_scheduler_queued_by_tenant{labels} {count}")

    lines.append("# HELP inference_engine_scheduler_in_flight_by_tenant Active scheduler slots by tenant.")
    lines.append("# TYPE inference_engine_scheduler_in_flight_by_tenant gauge")
    for tenant, count in sched.in_flight_by_tenant.items():
        labels = f'{{tenant="{_label_value(tenant)}"}}'
        lines.append(f"inference_engine_scheduler_in_flight_by_tenant{labels} {count}")

    lines.append("# HELP inference_engine_scheduler_in_flight_by_resource Active scheduler slots by backend/model resource.")
    lines.append("# TYPE inference_engine_scheduler_in_flight_by_resource gauge")
    for resource, count in sched.in_flight_by_resource.items():
        labels = f'{{resource="{_label_value(resource)}"}}'
        lines.append(f"inference_engine_scheduler_in_flight_by_resource{labels} {count}")

    return "\n".join(lines) + "\n"
