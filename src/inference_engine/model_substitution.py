"""Serve a resident, interchangeable model instead of loading the requested one.

Why this exists
---------------

Ollama decides what stays in memory, not the engine. On a host whose *system*
RAM cannot hold two large models at once, Ollama evicts one to load the other
— even with tens of GiB of GPU memory free — and each reload costs 40-100 s.
Two tenants alternating between ``gemma4:26b`` and ``qwen3.8:27b`` turned
every request into an evict-and-reload, and a 512-token judge call spent its
whole 240 s deadline waiting for a model to come back (2026-09-26).

The rule
--------

The operator declares groups of models it considers interchangeable
(``MODEL_SUBSTITUTION_GROUPS``). When a request names a group member that
Ollama does not currently hold, and another member of the same group *is*
resident on the same Ollama, the resident one serves the request. Nothing is
substituted when the requested model is resident, when no other member is,
when residency cannot be read, for governed (signed-route) requests, for
embeddings, or when the caller sends ``x-engine-model-substitution: off``.

The caller is always told. The body's ``model`` names what served, and
``substituted_from_model`` / ``substitution_reason`` name what was asked for
and why it was not used. The same facts ride on response headers, which on a
stream arrive before the first delta.

Residency comes from Ollama's ``GET /api/ps``, cached for a couple of seconds
per endpoint so the check costs one local round trip per burst, not per
request. A model the engine has just sent to Ollama cold is remembered as
"loading" for a while, because ``/api/ps`` does not list a model until its load
finishes — without that, a request for the other member arriving mid-load
would start a second load and the two would evict each other all over again.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Iterable
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Protocol

import httpx

from .observability import get_logger

log = get_logger("model_substitution")

REASON_NOT_RESIDENT = "requested_model_not_resident"

# Request header a caller sends to insist on the exact model it named.
OPT_OUT_HEADER = "x-engine-model-substitution"
_OPT_OUT_VALUES = frozenset({"off", "false", "0", "no", "disabled"})

# Response headers, present only on a response that was substituted.
SUBSTITUTED_FROM_HEADER = "x-engine-model-substituted-from"
SERVED_MODEL_HEADER = "x-engine-served-model"
REASON_HEADER = "x-engine-model-substitution-reason"

RESIDENCY_TTL_SECONDS = 2.0
PROBE_TIMEOUT_SECONDS = 1.0
# Longer than the slowest cold load seen on this host (~97 s), so a load in
# flight keeps counting as resident until ``/api/ps`` can confirm it.
PENDING_LOAD_SECONDS = 150.0


@dataclass(frozen=True)
class ModelSubstitution:
    requested_model: str
    served_model: str
    reason: str = REASON_NOT_RESIDENT


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------


def _normalize(name: str) -> str:
    """Ollama's own spelling: an untagged name means ``:latest``."""
    name = name.strip()
    return name if ":" in name.rsplit("/", 1)[-1] else f"{name}:latest"


def parse_groups(raw: str) -> tuple[tuple[str, ...], ...]:
    """``'a:1,b:2;c:3,d:4'`` -> ``(('a:1', 'b:2'), ('c:3', 'd:4'))``.

    A group needs two members to mean anything; a single-member group is
    dropped rather than treated as an error.
    """
    groups: list[tuple[str, ...]] = []
    for chunk in raw.split(";"):
        members: list[str] = []
        for item in chunk.split(","):
            if item.strip():
                name = _normalize(item)
                if name not in members:
                    members.append(name)
        if len(members) >= 2:
            groups.append(tuple(members))
    return tuple(groups)


# ---------------------------------------------------------------------------
# residency
# ---------------------------------------------------------------------------


async def ollama_resident_models(endpoint: str) -> frozenset[str]:
    """Names Ollama at ``endpoint`` holds in memory right now. Raises on failure."""
    async with httpx.AsyncClient(base_url=endpoint, timeout=PROBE_TIMEOUT_SECONDS) as client:
        response = await client.get("/api/ps")
        response.raise_for_status()
        payload = response.json()
    names: set[str] = set()
    for entry in payload.get("models") or []:
        if not isinstance(entry, dict):
            continue
        for key in ("name", "model"):
            value = entry.get(key)
            if isinstance(value, str) and value:
                names.add(_normalize(value))
    return frozenset(names)


class ResidencyCache:
    """Short-lived, per-endpoint view of what Ollama holds, plus loads in flight."""

    def __init__(
        self,
        probe: Callable[[str], Awaitable[frozenset[str]]] = ollama_resident_models,
        *,
        ttl_seconds: float = RESIDENCY_TTL_SECONDS,
        pending_seconds: float = PENDING_LOAD_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._probe = probe
        self._ttl = ttl_seconds
        self._pending_seconds = pending_seconds
        self._clock = clock
        self._entries: dict[str, tuple[float, frozenset[str] | None]] = {}
        self._pending: dict[str, dict[str, float]] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    async def resident(self, endpoint: str) -> frozenset[str] | None:
        """Resident names, or None when residency could not be read."""
        observed = await self._observed(endpoint)
        if observed is None:
            return None
        now = self._clock()
        pending = self._pending.get(endpoint, {})
        for name in [n for n, expires in pending.items() if expires <= now or n in observed]:
            del pending[name]
        return observed | frozenset(pending)

    def note_loading(self, endpoint: str, model_id: str) -> None:
        """Count ``model_id`` as resident while Ollama loads it."""
        self._pending.setdefault(endpoint, {})[_normalize(model_id)] = (
            self._clock() + self._pending_seconds
        )

    async def _observed(self, endpoint: str) -> frozenset[str] | None:
        cached = self._entries.get(endpoint)
        if cached is not None and self._clock() - cached[0] < self._ttl:
            return cached[1]
        lock = self._locks.setdefault(endpoint, asyncio.Lock())
        async with lock:
            cached = self._entries.get(endpoint)
            if cached is not None and self._clock() - cached[0] < self._ttl:
                return cached[1]
            try:
                value: frozenset[str] | None = await self._probe(endpoint)
            except Exception as exc:  # noqa: BLE001 — degrade to "no substitution"
                log.warning(
                    "model_substitution.residency_unavailable",
                    endpoint=endpoint,
                    error=str(exc),
                    error_type=type(exc).__name__,
                )
                value = None
            # A failure is cached too, so an unreachable Ollama is asked once
            # per TTL rather than once per request.
            self._entries[endpoint] = (self._clock(), value)
            return value


# ---------------------------------------------------------------------------
# per-request scope
# ---------------------------------------------------------------------------


@dataclass
class SubstitutionScope:
    """Per-request holder, opened by middleware and mutated by the route.

    Mutated rather than rebound for the same reason as
    ``_scheduling.AdmissionTelemetry``: a ContextVar the route rebinds does not
    travel back up through ``BaseHTTPMiddleware``'s child tasks.
    """

    allowed: bool = True
    applied: ModelSubstitution | None = None


_current_scope: ContextVar[SubstitutionScope | None] = ContextVar(
    "model_substitution_scope", default=None
)


def begin_request(headers: Iterable[tuple[bytes, bytes]]) -> SubstitutionScope:
    """Open this request's scope from its raw ASGI headers."""
    allowed = True
    for name, value in headers:
        if name.lower() == OPT_OUT_HEADER.encode("latin-1"):
            if value.decode("latin-1").strip().lower() in _OPT_OUT_VALUES:
                allowed = False
    scope = SubstitutionScope(allowed=allowed)
    _current_scope.set(scope)
    return scope


def substitution_allowed() -> bool:
    scope = _current_scope.get()
    return scope is None or scope.allowed


def bind(substitution: ModelSubstitution) -> None:
    """Record the substitution that decided which model serves this response.

    First one wins. Only the request's primary model binds here; a judge
    substituted inside chat-attached auto-eval reports on its own result
    instead, so it cannot relabel the completion it is grading.
    """
    scope = _current_scope.get()
    if scope is None or scope.applied is not None:
        return
    scope.applied = substitution


def current() -> ModelSubstitution | None:
    scope = _current_scope.get()
    return scope.applied if scope is not None else None


def fields(substitution: ModelSubstitution | None) -> dict:
    """Body fields for one substitution. Empty when there was none."""
    if substitution is None:
        return {}
    return {
        "substituted_from_model": substitution.requested_model,
        "substitution_reason": substitution.reason,
    }


def response_fields() -> dict:
    """Body fields for this request's primary substitution."""
    return fields(current())


def span_attrs(substitution: ModelSubstitution | None) -> dict:
    if substitution is None:
        return {}
    return {
        "llm.model_substitution.active": True,
        "llm.model_substitution.requested_model": substitution.requested_model,
        "llm.model_substitution.served_model": substitution.served_model,
        "llm.model_substitution.reason": substitution.reason,
    }


def _header_safe(value: str) -> bool:
    return bool(value) and value.isascii() and value.isprintable()


def response_headers(scope: SubstitutionScope | None) -> dict[str, str]:
    substitution = scope.applied if scope is not None else None
    if substitution is None:
        return {}
    headers = {
        SUBSTITUTED_FROM_HEADER: substitution.requested_model,
        SERVED_MODEL_HEADER: substitution.served_model,
        REASON_HEADER: substitution.reason,
    }
    # A model id is whatever the upstream registry reported; one that will not
    # encode must drop the header, never fail a response the caller paid for.
    return {name: value for name, value in headers.items() if _header_safe(value)}


# ---------------------------------------------------------------------------
# the decision
# ---------------------------------------------------------------------------


class _Resolver(Protocol):
    def resolve(self, model_id: str): ...


def _ollama_id(descriptor) -> str:
    return _normalize(str((descriptor.params or {}).get("model_id") or descriptor.qualified_name))


class ModelSubstituter:
    def __init__(
        self,
        groups: tuple[tuple[str, ...], ...],
        residency: ResidencyCache | None = None,
    ) -> None:
        self._groups = groups
        self._residency = residency or ResidencyCache()

    @property
    def enabled(self) -> bool:
        return bool(self._groups)

    def _group_of(self, name: str) -> tuple[str, ...] | None:
        normalized = _normalize(name)
        for group in self._groups:
            if normalized in group:
                return group
        return None

    async def choose(self, requested_model: str, manager: _Resolver) -> ModelSubstitution | None:
        """The substitution to apply to ``requested_model``, or None to serve it as asked."""
        if not self._groups or not substitution_allowed():
            return None
        requested = manager.resolve(requested_model)
        if requested is None or requested.format != "ollama_http" or not requested.endpoint:
            return None
        group = self._group_of(requested.qualified_name)
        if group is None:
            return None

        resident = await self._residency.resident(requested.endpoint)
        if resident is None:
            return None
        if _ollama_id(requested) in resident:
            return None

        for member in group:
            if member == _normalize(requested.qualified_name):
                continue
            candidate = manager.resolve(member)
            if (
                candidate is None
                or candidate.format != "ollama_http"
                or candidate.endpoint != requested.endpoint
            ):
                continue
            if _ollama_id(candidate) in resident:
                substitution = ModelSubstitution(
                    requested_model=requested.qualified_name,
                    served_model=candidate.qualified_name,
                )
                log.info(
                    "model_substitution.applied",
                    requested_model=substitution.requested_model,
                    served_model=substitution.served_model,
                    reason=substitution.reason,
                    endpoint=requested.endpoint,
                )
                return substitution

        # Nothing interchangeable is resident, so this request will load the
        # requested model. Say so before ``/api/ps`` can, so a request for
        # another member arriving mid-load joins this one instead of evicting it.
        self._residency.note_loading(requested.endpoint, _ollama_id(requested))
        return None


__all__ = [
    "OPT_OUT_HEADER",
    "REASON_HEADER",
    "REASON_NOT_RESIDENT",
    "SERVED_MODEL_HEADER",
    "SUBSTITUTED_FROM_HEADER",
    "ModelSubstituter",
    "ModelSubstitution",
    "ResidencyCache",
    "SubstitutionScope",
    "begin_request",
    "bind",
    "current",
    "fields",
    "ollama_resident_models",
    "parse_groups",
    "response_fields",
    "response_headers",
    "span_attrs",
    "substitution_allowed",
]
