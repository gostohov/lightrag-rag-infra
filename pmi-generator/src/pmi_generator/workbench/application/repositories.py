from __future__ import annotations

from typing import Protocol

from ..domain import TestCard
from .state import AttemptRecord, EventRecord, SessionRecord, StoredRecord


class CardRepository(Protocol):
    def get(self, card_id: str) -> TestCard | None: ...

    def list_all(self) -> list[TestCard]: ...

    def save(self, card: TestCard) -> None: ...


class SessionRepository(Protocol):
    def get(self, session_id: str) -> SessionRecord | None: ...

    def list_all(self) -> list[SessionRecord]: ...

    def save(self, session: SessionRecord) -> None: ...


class AttemptRepository(Protocol):
    def get(self, attempt_id: str) -> AttemptRecord | None: ...

    def active_for_session(self, session_id: str) -> AttemptRecord | None: ...

    def list_for_session(self, session_id: str) -> list[AttemptRecord]: ...

    def list_all(self) -> list[AttemptRecord]: ...

    def save(self, attempt: AttemptRecord) -> None: ...


class RecordRepository(Protocol):
    def get(self, kind: str, record_id: str) -> StoredRecord | None: ...

    def list_kind(self, kind: str) -> list[StoredRecord]: ...

    def save(self, record: StoredRecord) -> None: ...


class EventRepository(Protocol):
    def append(self, aggregate_id: str, event_type: str, payload: dict[str, object]) -> int: ...

    def list_for(self, aggregate_id: str) -> list[EventRecord]: ...


class UnitOfWork(Protocol):
    cards: CardRepository
    sessions: SessionRepository
    attempts: AttemptRepository
    records: RecordRepository
    events: EventRepository

    def __enter__(self) -> UnitOfWork: ...

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None: ...
