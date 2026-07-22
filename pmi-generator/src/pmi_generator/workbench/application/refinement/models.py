from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class RefinementError(ValueError):
    pass


@dataclass(slots=True)
class RefinementArguments:
    outcome: str
    updates: list[dict[str, Any]]
    gaps: list[dict[str, Any]]
    reason: str


@dataclass(frozen=True, slots=True)
class RefinementResult:
    card_id: str
    revision: int
    outcome: str
    changed: bool
    gap_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RefinementProposalResult:
    card_id: str
    revision: int
    outcome: str
    proposal_id: str | None
