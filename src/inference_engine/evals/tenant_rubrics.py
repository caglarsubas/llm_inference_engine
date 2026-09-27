"""Tenant rubrics — judges a tenant registers over the API, kept per tenant.

A built-in rubric is code (``rubrics.py``). A tenant rubric is data: a prompt
pair, the keys a verdict must carry, and a score rule picked from a closed set
(``NumberScore``, ``BooleanScore``, ``ChoiceScore``), so nothing a caller sends
is ever executed.

Each tenant's rubrics live in one JSON file under ``EVAL_RUBRICS_DIR``, named
by a hash of the tenant so a tenant id never becomes a path. The lifespan loads
the directory at startup; a file that will not parse fails startup loudly, as a
malformed auto-eval policy does, because the engine is the only writer and
every write goes through a temp file and ``os.replace``.

Replicas that share the directory (the compose stack mounts one volume into
all of them) see each other's writes: every lookup compares the tenant file's
stamp with the one it loaded and re-reads it when another replica replaced it.
Two replicas writing the same tenant at the same moment is last-write-wins.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import threading
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..observability import get_logger
from .rubrics import RubricSpec
from .schemas import (
    BooleanScore,
    ChoiceScore,
    NumberScore,
    RubricDefinition,
)

log = get_logger("evals.tenant_rubrics")

_FILE_VERSION = 1


class RubricStoreError(RuntimeError):
    """A rubric file on disk could not be loaded."""


class RubricLimitError(ValueError):
    """The tenant already holds the maximum number of rubrics."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        super().__init__(f"a tenant may register at most {limit} rubrics")


# ---------------------------------------------------------------------------
# Score rules
# ---------------------------------------------------------------------------


def score_extractor(rule: NumberScore | BooleanScore | ChoiceScore) -> Callable[[dict], float]:
    """Turn a declarative score rule into the extractor ``RubricSpec`` expects.

    Each extractor raises ``ValueError`` on a value the rule does not cover, and
    the runner turns that into ``parse_status="failed"``: a judge that answered
    off the rubric's scale has not given a score.
    """
    key = rule.key
    if isinstance(rule, NumberScore):
        low, high = rule.min, rule.max

        def number(verdict: dict[str, Any]) -> float:
            value = verdict[key]
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise ValueError(f"{key!r} is not a number")
            value = float(value)
            if not math.isfinite(value):
                raise ValueError(f"{key!r} is not finite")
            if low is None or high is None:
                return value
            if not low <= value <= high:
                raise ValueError(f"{key!r}={value} is outside [{low}, {high}]")
            return (value - low) / (high - low)

        return number
    if isinstance(rule, BooleanScore):

        def boolean(verdict: dict[str, Any]) -> float:
            value = verdict[key]
            if isinstance(value, str) and value.strip().lower() in ("true", "false"):
                value = value.strip().lower() == "true"
            if not isinstance(value, bool):
                raise ValueError(f"{key!r} is not a boolean")
            return 1.0 if value else 0.0

        return boolean
    values = dict(rule.values)

    def choice(verdict: dict[str, Any]) -> float:
        label = str(verdict[key]).strip()
        if label not in values:
            raise ValueError(f"{key!r}={label!r} is not one of {sorted(values)}")
        return values[label]

    return choice


def to_spec(definition: RubricDefinition) -> RubricSpec:
    return RubricSpec(
        name=definition.name,
        description=definition.description,
        system_prompt=definition.system_prompt,
        user_prompt_template=definition.user_prompt_template,
        expected_keys=tuple(definition.expected_keys),
        score_extractor=score_extractor(definition.score),
        requires_expected=definition.requires_expected,
        pairwise=definition.pairwise,
    )


def digest(definition: RubricDefinition) -> str:
    """Content hash of a definition, so a verdict names the exact judge that
    produced it even after the rubric is replaced."""
    canonical = json.dumps(
        definition.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
    )
    return "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StoredRubric:
    definition: RubricDefinition
    spec: RubricSpec
    digest: str
    created_at: int
    updated_at: int

    @classmethod
    def build(cls, definition: RubricDefinition, *, created_at: int, updated_at: int):
        return cls(
            definition=definition,
            spec=to_spec(definition),
            digest=digest(definition),
            created_at=created_at,
            updated_at=updated_at,
        )

    def as_record(self) -> dict[str, Any]:
        return {
            "definition": self.definition.model_dump(mode="json"),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


def tenant_file_name(tenant: str) -> str:
    return hashlib.sha256(tenant.encode()).hexdigest()[:32] + ".json"


class TenantRubricStore:
    """Tenant → name → rubric. ``directory=None`` keeps rubrics in memory only,
    which is the state before the lifespan loads the configured directory."""

    def __init__(self, directory: Path | None = None, *, max_per_tenant: int = 64) -> None:
        self._dir = directory
        self._max = max_per_tenant
        self._by_tenant: dict[str, dict[str, StoredRubric]] = {}
        # (inode, mtime_ns, size) of each tenant file as last read or written;
        # None when the tenant has no file. Every write replaces the file with
        # a new one, so the inode changes even where mtime is coarse.
        self._stamps: dict[str, tuple[int, int, int] | None] = {}
        self._lock = threading.Lock()

    @classmethod
    def load(cls, directory: Path, *, max_per_tenant: int = 64) -> "TenantRubricStore":
        store = cls(directory, max_per_tenant=max_per_tenant)
        if not directory.exists():
            return store
        for path in sorted(directory.glob("*.json")):
            stamp = _stamp(path)
            tenant, rubrics = _read(path)
            store._by_tenant[tenant] = rubrics
            store._stamps[tenant] = stamp
        log.info(
            "eval_rubrics.loaded",
            directory=str(directory),
            tenants=len(store._by_tenant),
            rubrics=sum(len(r) for r in store._by_tenant.values()),
        )
        return store

    def get(self, tenant: str, name: str) -> StoredRubric | None:
        with self._lock:
            self._refresh(tenant)
            return self._by_tenant.get(tenant, {}).get(name)

    def list(self, tenant: str) -> list[StoredRubric]:
        with self._lock:
            self._refresh(tenant)
            rubrics = list(self._by_tenant.get(tenant, {}).values())
        return sorted(rubrics, key=lambda r: r.definition.name)

    def put(
        self, tenant: str, definition: RubricDefinition, *, now: int | None = None
    ) -> tuple[StoredRubric, bool]:
        """Register or replace. Returns the stored rubric and whether it is new."""
        stamp = int(time.time()) if now is None else now
        with self._lock:
            self._refresh(tenant)
            current = dict(self._by_tenant.get(tenant, {}))
            previous = current.get(definition.name)
            if previous is None and len(current) >= self._max:
                raise RubricLimitError(self._max)
            stored = StoredRubric.build(
                definition,
                created_at=previous.created_at if previous else stamp,
                updated_at=stamp,
            )
            current[definition.name] = stored
            # Disk first: a failed write must not leave memory ahead of the file.
            self._write(tenant, current)
            self._by_tenant[tenant] = current
        return stored, previous is None

    def delete(self, tenant: str, name: str) -> bool:
        with self._lock:
            self._refresh(tenant)
            current = dict(self._by_tenant.get(tenant, {}))
            if current.pop(name, None) is None:
                return False
            self._write(tenant, current)
            if current:
                self._by_tenant[tenant] = current
            else:
                self._by_tenant.pop(tenant, None)
        return True

    def _refresh(self, tenant: str) -> None:
        """Re-read the tenant's file if another writer replaced it. Caller
        holds the lock."""
        if self._dir is None:
            return
        path = self._dir / tenant_file_name(tenant)
        stamp = _stamp(path)
        if tenant in self._stamps and self._stamps[tenant] == stamp:
            return
        if stamp is None:
            self._by_tenant.pop(tenant, None)
        else:
            try:
                _tenant, rubrics = _read(path)
            except RubricStoreError as exc:
                # Keep serving what was last read rather than failing requests.
                log.warning("eval_rubrics.refresh_failed", error=str(exc))
                return
            self._by_tenant[tenant] = rubrics
        self._stamps[tenant] = stamp

    def _write(self, tenant: str, rubrics: dict[str, StoredRubric]) -> None:
        if self._dir is None:
            return
        self._dir.mkdir(parents=True, exist_ok=True)
        path = self._dir / tenant_file_name(tenant)
        if not rubrics:
            path.unlink(missing_ok=True)
            self._stamps[tenant] = None
            return
        document = {
            "version": _FILE_VERSION,
            "tenant": tenant,
            "rubrics": [rubrics[name].as_record() for name in sorted(rubrics)],
        }
        # ``.tmp`` rather than ``.json`` so a leftover never loads as a tenant.
        fd, tmp = tempfile.mkstemp(dir=self._dir, prefix=".rubrics-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(document, handle, indent=2, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, path)
        except BaseException:
            with suppress(FileNotFoundError):
                os.unlink(tmp)
            raise
        self._stamps[tenant] = _stamp(path)


def _stamp(path: Path) -> tuple[int, int, int] | None:
    try:
        st = path.stat()
    except FileNotFoundError:
        return None
    return st.st_ino, st.st_mtime_ns, st.st_size


def _read(path: Path) -> tuple[str, dict[str, StoredRubric]]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        if document.get("version") != _FILE_VERSION:
            raise ValueError(f"unsupported version {document.get('version')!r}")
        tenant = document["tenant"]
        if not isinstance(tenant, str) or tenant_file_name(tenant) != path.name:
            raise ValueError("file name does not match its tenant")
        rubrics: dict[str, StoredRubric] = {}
        for record in document["rubrics"]:
            definition = RubricDefinition.model_validate(record["definition"])
            rubrics[definition.name] = StoredRubric.build(
                definition,
                created_at=int(record["created_at"]),
                updated_at=int(record["updated_at"]),
            )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        # pydantic's ValidationError is a ValueError.
        raise RubricStoreError(f"cannot load tenant rubrics from {path}: {exc}") from exc
    return tenant, rubrics
