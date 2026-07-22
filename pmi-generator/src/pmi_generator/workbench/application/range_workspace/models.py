from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class WorkspaceItem:
    skeleton_id: str
    title: str
    status: str
    style: str
    card_id: str | None = None
    revision: int | None = None
    session_id: str | None = None


@dataclass(frozen=True, slots=True)
class RangeWorkspaceState:
    selection_id: str
    items: tuple[WorkspaceItem, ...]
    can_review: bool
    review_current: bool
    review_stale: bool
    included: int
    included_incomplete: int
    excluded: int
    terminal_status: str | None = None
    terminal_explanation: str = ""
