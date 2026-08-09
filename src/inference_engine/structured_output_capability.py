"""Which DEPLOYMENTS actually constrain decoding — learned, not asserted.

``InferenceAdapter.supports_structured_outputs`` is a claim about a backend
CLASS: "llama.cpp has GBNF", "Ollama implements structured outputs". Those
claims are true of the software and false of particular deployments. Ollama
gained JSON-Schema support in its OpenAI shim at a specific release; point the
same adapter at an older one and the key is accepted, ignored, and answered
with prose. The class constant cannot see that, because the difference is not
in the class.

Measured on a live deployment (Ollama, gemma4:26b): a strict ``json_schema``
request for an object with one integer key returned HTTP 200 and the body
``Mars has **two** moons: Phobos and Deimos.`` The adapter declared
enforcement; this deployment had none.

WHY OBSERVATION AND NOT A PROBE. The obvious alternative is a capability probe
at load: send a tiny constrained request and see. It costs a generation per
deployment, it can time out, and — the part that matters — a CONFORMING answer
would not prove anything. An unconstrained model asked for something simple can
produce a conforming document by luck, so a probe can only ever prove the
negative. This registry waits for that same negative to arrive on its own, for
free, from a request the caller was making anyway.

WHAT COUNTS AS EVIDENCE. Only ``ResponseNotJson``. A constrained sampler cannot
emit non-JSON, so prose is proof. A keyword violation is NOT proof: this
codebase validates a subset of JSON Schema, so a violation may be a gap in the
checker, which is exactly the false-rejection risk that made trusting backends
attractive in the first place. Nothing here ever promotes a deployment —
evidence only ever flows toward less trust.

Blank output is excluded too. An empty completion means truncation or a stop
token, not an unconstrained sampler, and demoting on it would punish a
correctly-configured backend that hit ``max_tokens``.

LIFETIME. Process-local and unbounded in the only direction that matters: a
demotion holds until restart. The set is keyed per (deployment, model) and
gains at most one entry per deployment, so it cannot grow with traffic.
"""

from __future__ import annotations

import structlog

from .adapters.base import InferenceAdapter

log = structlog.get_logger(__name__)

# (deployment_id, model) pairs proven NOT to constrain decoding.
_unenforced: set[tuple[str, str]] = set()


def deployment_key(adapter: InferenceAdapter, model_name: str) -> tuple[str, str]:
    """Identify the deployment, not the backend class.

    Two Ollama endpoints on different hosts are different deployments and may
    be different releases, so the endpoint belongs in the key. Adapters that
    serve a single in-process model fall back to the backend name.
    """
    return (adapter.deployment_id(), model_name)


def constrains_decoding(adapter: InferenceAdapter, model_name: str) -> bool:
    """Should this deployment's output be taken on trust?

    The class-level declaration is the PRIOR; an observed non-JSON response
    overrides it permanently. A deployment that never misbehaves is never
    demoted, so backends with real grammar hooks keep their existing treatment.
    """
    if not adapter.supports_structured_outputs:
        return False
    return deployment_key(adapter, model_name) not in _unenforced


def record_unenforced(adapter: InferenceAdapter, model_name: str, *, evidence: str) -> bool:
    """Record proof that this deployment ignores the schema.

    Returns True the first time, so the caller can log the transition once
    rather than on every subsequent request.
    """
    key = deployment_key(adapter, model_name)
    if key in _unenforced:
        return False
    _unenforced.add(key)
    log.warning(
        "structured_output.capability_demoted",
        deployment=key[0],
        model=key[1],
        backend=adapter.backend_name,
        declared=adapter.supports_structured_outputs,
        evidence=evidence,
    )
    return True


def observed_unenforced() -> frozenset[tuple[str, str]]:
    """Read-only view, for diagnostics and tests."""
    return frozenset(_unenforced)


def reset_for_tests() -> None:
    """Clear observations. Test-only; production never calls this."""
    _unenforced.clear()


__all__ = [
    "constrains_decoding",
    "deployment_key",
    "observed_unenforced",
    "record_unenforced",
    "reset_for_tests",
]
