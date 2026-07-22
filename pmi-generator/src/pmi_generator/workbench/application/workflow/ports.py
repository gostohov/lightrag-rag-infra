from __future__ import annotations

from typing import Protocol

from .commands import WorkflowCommand
from .models import WorkflowState


class WorkflowRuntime(Protocol):
    def execute(self, thread_id: str, command: WorkflowCommand) -> WorkflowState: ...

    def current_state(self, thread_id: str) -> WorkflowState: ...

    def waiting_for_input(self, thread_id: str) -> bool: ...


class DecompositionWorker(Protocol):
    def decompose(self, selection_id: str) -> list[str]: ...


class PopulationWorker(Protocol):
    def populate(self, card_id: str) -> list[str]: ...


class RetrievalWorker(Protocol):
    def resolve(self, card_id: str, gap_id: str) -> bool: ...


class RangeReviewWorker(Protocol):
    def review(self, selection_id: str, card_ids: list[str]) -> list[str]: ...
