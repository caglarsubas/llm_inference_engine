"""Background-refreshed /v1/models snapshot (issue #69).

The endpoint must return a metadata payload without probe-loading on the
request path. These tests pin the three behaviours that guarantee that:

* the routes serve the cached snapshot verbatim when one exists (no probe pass
  on the hot path, even while the underlying registry is changing);
* they fall back to an off-thread fresh compute when the cache is empty (a
  lifespan-less client, or the window before the first refresh);
* the background refresher populates and updates the cache, and a failing
  rebuild keeps the last good snapshot instead of surfacing a stall.
"""

from __future__ import annotations

import asyncio

import pytest

from inference_engine.api import models as models_api
from inference_engine.api._models_snapshot import (
    ModelsSnapshot,
    ModelsSnapshotCache,
    build_snapshot,
    run_models_snapshot_refresher,
)
from inference_engine.schemas import ModelCatalog, ModelInfo, ModelList


def _model_list(*ids: str) -> ModelList:
    return ModelList(
        data=[
            ModelInfo(id=i, size_bytes=0, backend="llama_cpp", format="gguf", model_path="/x")
            for i in ids
        ]
    )


def _snapshot(*ids: str) -> ModelsSnapshot:
    return ModelsSnapshot(
        model_list=_model_list(*ids),
        catalog=ModelCatalog(data=[]),
        generated_at=1.0,
        duration_ms=0.0,
    )


# ---------------------------------------------------------------------------
# Cache holder
# ---------------------------------------------------------------------------


def test_cache_starts_empty_and_round_trips() -> None:
    cache = ModelsSnapshotCache()
    assert cache.get() is None

    snap = _snapshot("a:gguf")
    cache.set(snap)
    assert cache.get() is snap

    cache.clear()
    assert cache.get() is None


def test_build_snapshot_records_both_views_and_duration() -> None:
    views = (_model_list("a:gguf"), ModelCatalog(data=[]))
    snap = build_snapshot(lambda: views)

    assert snap.model_list is views[0]
    assert snap.catalog is views[1]
    assert snap.duration_ms >= 0.0
    assert snap.generated_at > 0.0


# ---------------------------------------------------------------------------
# Routes serve the cache and never probe on the hot path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_models_serves_cached_snapshot_without_probing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Autouse fixture clears the cache after the test; set it directly here.
    models_api.snapshot_cache.set(_snapshot("cached:gguf"))

    def _boom():  # pragma: no cover - must never run on the cache-hit path
        raise AssertionError("request path probe-loaded instead of serving cache")

    monkeypatch.setattr(models_api, "collect_model_list", _boom)

    result = await models_api.list_models(_=object())

    assert [m.id for m in result.data] == ["cached:gguf"]


@pytest.mark.asyncio
async def test_list_model_catalog_serves_cached_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snap = ModelsSnapshot(
        model_list=_model_list(),
        catalog=ModelCatalog(data=[], unavailable=[]),
        generated_at=1.0,
        duration_ms=0.0,
    )
    models_api.snapshot_cache.set(snap)

    def _boom():  # pragma: no cover - must never run on the cache-hit path
        raise AssertionError("catalog request path recomputed instead of serving cache")

    monkeypatch.setattr(models_api, "collect_model_catalog", _boom)

    result = await models_api.list_model_catalog(_=object())

    assert result is snap.catalog


@pytest.mark.asyncio
async def test_list_models_falls_back_to_fresh_compute_when_cache_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert models_api.snapshot_cache.get() is None  # autouse fixture cleared it
    calls: list[int] = []

    def _fresh() -> ModelList:
        calls.append(1)
        return _model_list("fresh:gguf")

    monkeypatch.setattr(models_api, "collect_model_list", _fresh)

    result = await models_api.list_models(_=object())

    assert [m.id for m in result.data] == ["fresh:gguf"]
    assert calls == [1]  # computed on demand, exactly once


# ---------------------------------------------------------------------------
# Background refresher
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_refresher_populates_and_updates_cache() -> None:
    cache = ModelsSnapshotCache()
    versions = iter(
        [
            (_model_list("v1:gguf"), ModelCatalog(data=[])),
            (_model_list("v2:gguf"), ModelCatalog(data=[])),
        ]
    )

    def _build():
        return next(versions)

    task = asyncio.create_task(
        run_models_snapshot_refresher(cache, _build, interval_seconds=0.01)
    )
    try:
        await _wait_until(lambda: _ids(cache) == ["v1:gguf"])
        await _wait_until(lambda: _ids(cache) == ["v2:gguf"])
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
async def test_refresher_keeps_last_good_snapshot_when_build_fails() -> None:
    cache = ModelsSnapshotCache()
    state = {"fail": False}

    def _build():
        if state["fail"]:
            raise RuntimeError("probe storm")
        return (_model_list("good:gguf"), ModelCatalog(data=[]))

    task = asyncio.create_task(
        run_models_snapshot_refresher(cache, _build, interval_seconds=0.01)
    )
    try:
        await _wait_until(lambda: _ids(cache) == ["good:gguf"])
        # Flip to failing builds; the last good snapshot must survive.
        state["fail"] = True
        await asyncio.sleep(0.05)
        assert _ids(cache) == ["good:gguf"]
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
async def test_refresher_waits_for_startup_before_first_build() -> None:
    cache = ModelsSnapshotCache()
    gate: asyncio.Future = asyncio.get_running_loop().create_future()

    def _build():
        return (_model_list("late:gguf"), ModelCatalog(data=[]))

    task = asyncio.create_task(
        run_models_snapshot_refresher(
            cache, _build, interval_seconds=0.01, wait_for=gate
        )
    )
    try:
        # Refresher is blocked on the startup gate: no build yet.
        await asyncio.sleep(0.03)
        assert cache.get() is None

        gate.set_result(None)
        await _wait_until(lambda: _ids(cache) == ["late:gguf"])
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


def _ids(cache: ModelsSnapshotCache) -> list[str] | None:
    snap = cache.get()
    return None if snap is None else [m.id for m in snap.model_list.data]


async def _wait_until(pred, *, timeout: float = 2.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if pred():
            return
        await asyncio.sleep(0.005)
    raise AssertionError("condition not met within timeout")
