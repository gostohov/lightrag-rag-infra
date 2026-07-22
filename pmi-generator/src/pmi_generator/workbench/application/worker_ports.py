from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Protocol

from .source import SavedSelection


@dataclass(frozen=True, slots=True)
class WorkerOperation:
    awaitable: Awaitable[Any]
    cancel: Callable[[], object]
    progress: Callable[[], object | None] = lambda: None


class PromptWorkers(Protocol):
    def decompose(self, selection: SavedSelection, attempt_id: str) -> WorkerOperation: ...

    def populate(
        self,
        selection: SavedSelection,
        skeleton_id: str,
        session_id: str,
        card_id: str,
        attempt_id: str,
    ) -> WorkerOperation: ...

    def investigate_gap(
        self,
        selection: SavedSelection,
        session_id: str,
        card_id: str,
        gap_id: str,
        attempt_id: str,
        research_question: str | None = None,
        research_message_id: str | None = None,
    ) -> WorkerOperation: ...

    def plan_refinement(
        self,
        session_id: str,
        card_id: str,
        message_id: str,
        attempt_id: str,
        expected_revision: int,
    ) -> WorkerOperation: ...

    def review_selection(
        self,
        selection: SavedSelection,
        attempt_id: str,
    ) -> WorkerOperation: ...


__all__ = ["PromptWorkers", "WorkerOperation"]
