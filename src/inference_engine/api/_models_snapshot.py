"""Background-refreshed snapshot of the reachable model catalog.

Why this exists
---------------

``/v1/models`` (and its ``/v1/models.data`` catalog sibling) must return a
metadata payload in well under a second regardless of inference load. But
building the *honest* reachable-model view is expensive: the GGUF load probe
opens every model's header + architecture graph (``registry/probe.py``),
which costs tens of ms for small models and low *seconds* for a 30 GB model on
a cold cache — and that work contends with active inference for memory and IO.

Doing that probe pass inline on every request means a busy box stalls model
discovery past client timeouts. That is exactly the intermittent
``/v1/models`` stall reported in issue #69: fast TCP accept, fast 401, but a
multi-second hang on the authenticated model list whenever the engine is under
load or a model's probe cache entry was invalidated (a re-pull, memory
pressure, a model warming up).

The fix (issue #69, suggested fix #1): serve the endpoint from a **cached
capability snapshot refreshed in the background**. The heavy probe pass runs on
a worker thread on a fixed cadence; the request path just hands back the last
good snapshot instantly. A metadata call never queues behind generation and
never probe-loads on the hot path.

When no snapshot exists yet — e.g. a ``TestClient`` that never ran the ASGI
lifespan, or the sliver between "engine ready" and the refresher's first build
— the request path falls back to computing a fresh view *off the event loop*
(``asyncio.to_thread``) so it stays correct without ever blocking the loop.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from threading import Lock

from ..observability import get_logger
from ..schemas import ModelCatalog, ModelList

log = get_logger("models.snapshot")

# A callable that runs the (expensive) probe-aware partition pass once and
# returns both response views. Injected so this module never imports the route
# layer back — keeps the dependency edge one-directional.
BuildViews = Callable[[], "tuple[ModelList, ModelCatalog]"]


@dataclass(frozen=True)
class ModelsSnapshot:
    """An immutable point-in-time view of the reachable model surface."""

    model_list: ModelList
    catalog: ModelCatalog
    generated_at: float  # wall-clock epoch seconds the snapshot was built
    duration_ms: float  # how long the build took (probe pass cost)


class ModelsSnapshotCache:
    """Thread-safe holder for the latest :class:`ModelsSnapshot`.

    The background refresher (event loop) is the sole writer in production; the
    request path (also event loop) and any diagnostic caller read it. The lock
    guards the single reference swap so a reader never observes a torn write —
    snapshots themselves are frozen, so a returned snapshot is safe to use
    without holding the lock.
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._snapshot: ModelsSnapshot | None = None

    def get(self) -> ModelsSnapshot | None:
        with self._lock:
            return self._snapshot

    def set(self, snapshot: ModelsSnapshot) -> None:
        with self._lock:
            self._snapshot = snapshot

    def clear(self) -> None:
        with self._lock:
            self._snapshot = None


def build_snapshot(build_views: BuildViews) -> ModelsSnapshot:
    """Run one probe pass and wrap both views in a timestamped snapshot.

    Synchronous and potentially slow (the probe pass) — always call it from a
    worker thread (``asyncio.to_thread``) so it never blocks the event loop.
    """
    t0 = time.perf_counter()
    model_list, catalog = build_views()
    return ModelsSnapshot(
        model_list=model_list,
        catalog=catalog,
        generated_at=time.time(),
        duration_ms=(time.perf_counter() - t0) * 1000,
    )


async def run_models_snapshot_refresher(
    cache: ModelsSnapshotCache,
    build_views: BuildViews,
    *,
    interval_seconds: float,
    wait_for: asyncio.Future | None = None,
) -> None:
    """Rebuild the model snapshot on a fixed cadence, off the event loop.

    A rebuild that raises keeps the *previous* snapshot in place rather than
    surfacing the failure to callers — a transient probe error should degrade
    freshness, never availability of the metadata endpoint. The task runs until
    cancelled (lifespan shutdown), which propagates out of the ``to_thread``
    build and the inter-cycle sleep.
    """
    if wait_for is not None:
        # Let the startup probe pass warm the shared probe cache first, so our
        # first build is a cache hit rather than a duplicate cold probe storm
        # racing startup. ``suppress(Exception)`` deliberately does not catch
        # ``CancelledError`` (a BaseException), so shutdown during the wait
        # still tears the task down promptly.
        with suppress(Exception):
            await asyncio.shield(wait_for)

    while True:
        try:
            snapshot = await asyncio.to_thread(build_snapshot, build_views)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - refresh must not take the loop down
            log.warning(
                "models_snapshot_refresh_failed",
                error_type=type(exc).__name__,
                error=str(exc),
            )
        else:
            cache.set(snapshot)
            log.debug(
                "models_snapshot_refreshed",
                n_available=len(snapshot.model_list.data),
                n_unavailable=len(snapshot.model_list.unavailable),
                duration_ms=round(snapshot.duration_ms, 2),
            )
        await asyncio.sleep(interval_seconds)


__all__ = [
    "ModelsSnapshot",
    "ModelsSnapshotCache",
    "build_snapshot",
    "run_models_snapshot_refresher",
]
