from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class SelectionReviewError(ValueError):
    pass


@dataclass(slots=True)
class SelectionReviewArguments:
    outcome: str
    issues: list[dict[str, Any]]


@dataclass(frozen=True, slots=True)
class SelectionReviewResult:
    selection_id: str
    outcome: str
    issue_ids: tuple[str, ...]
    card_revisions: dict[str, int]

