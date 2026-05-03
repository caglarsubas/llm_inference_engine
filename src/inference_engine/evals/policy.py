"""Server-side auto-eval policy.

Lets the engine attach rubrics by ``(tenant, model)`` instead of relying on
clients to send ``auto_eval`` per request. The policy plane (Prometa) writes
this file; the engine reads it at startup and matches every chat completion
against it.

Design choice: **policy wins entirely when matched.** When a policy entry
covers a request, the request's own ``auto_eval`` field is ignored. This
keeps the policy plane authoritative — compliance / safety rubrics can't
silently be opted out of by a client. Clients who want fully request-driven
evals are simply not covered by a policy.

File format (JSON array, first-match-wins, list most-specific first):

    [
      {
        "name": "agent-runtime-quality",
        "match": {"tenant": "agent-runtime", "model": "llama3.2:1b"},
        "auto_eval": {
          "rubrics": ["safety", "helpfulness"],
          "mode": "background",
          "judge_model": "llama3.2:3b"
        }
      },
      {
        "name": "compliance-baseline",
        "match": {"tenant": "*", "model": "*"},
        "auto_eval": {
          "rubrics": ["safety"],
          "mode": "background",
          "judge_model": "llama3.2:3b"
        }
      }
    ]

``"*"`` matches any value. Match dimensions left unspecified are treated as
``"*"``. The first entry whose match clause covers the call wins; ordering is
the operator's responsibility.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ..observability import get_logger
from ..schemas import AutoEvalSpec

log = get_logger("evals.policy")


@dataclass(frozen=True)
class PolicyMatch:
    """Wildcard-aware match clause."""

    tenant: str = "*"
    model: str = "*"

    def matches(self, *, tenant: str, model: str) -> bool:
        return self._covers(self.tenant, tenant) and self._covers(self.model, model)

    @staticmethod
    def _covers(pattern: str, value: str) -> bool:
        return pattern == "*" or pattern == value


@dataclass(frozen=True)
class PolicyEntry:
    """One ``(match → auto_eval)`` rule. Names are optional but useful in spans."""

    name: str
    match: PolicyMatch
    spec: AutoEvalSpec


class PolicyRegistry:
    """First-match-wins resolver. Empty registry = policy disabled by default."""

    def __init__(self, entries: list[PolicyEntry]) -> None:
        self._entries = list(entries)

    def __len__(self) -> int:
        return len(self._entries)

    def all(self) -> list[PolicyEntry]:
        return list(self._entries)

    def resolve(self, *, tenant: str, model: str) -> PolicyEntry | None:
        for entry in self._entries:
            if entry.match.matches(tenant=tenant, model=model):
                return entry
        return None


def load_policy(path: Path | str) -> PolicyRegistry:
    """Read the policy file and return a registry.

    Missing file → empty registry (policy disabled). Malformed file raises so
    the operator gets a loud failure at startup rather than a silent miss.
    """
    p = Path(path)
    if not p.exists():
        log.info("auto_eval.policy_missing", path=str(p))
        return PolicyRegistry([])

    raw = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"auto_eval policy file must be a JSON array, got {type(raw).__name__}")

    entries: list[PolicyEntry] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"policy entry {i} is not an object: {item!r}")
        name = str(item.get("name") or f"unnamed-{i}")
        match_raw = item.get("match", {}) or {}
        if not isinstance(match_raw, dict):
            raise ValueError(f"policy entry {name!r}: 'match' must be an object")
        match = PolicyMatch(
            tenant=str(match_raw.get("tenant", "*")),
            model=str(match_raw.get("model", "*")),
        )
        spec_raw = item.get("auto_eval")
        if not isinstance(spec_raw, dict):
            raise ValueError(f"policy entry {name!r}: 'auto_eval' must be an object")
        # Reuse the schema's own validation — same constraints as a request.
        spec = AutoEvalSpec(**spec_raw)
        entries.append(PolicyEntry(name=name, match=match, spec=spec))

    log.info("auto_eval.policy_loaded", path=str(p), count=len(entries))
    return PolicyRegistry(entries)
