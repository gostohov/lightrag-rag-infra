from __future__ import annotations

from dataclasses import dataclass, field

from ...application.state import AttemptRecord, AttemptStatus, EventRecord, SessionRecord, StoredRecord
from ...domain import TestCard
from .card_codec import decode_card, encode_card
from .errors import StorageConflictError


@dataclass(slots=True)
class InMemoryDatabase:
    cards: dict[str, str] = field(default_factory=dict)
    sessions: dict[str, SessionRecord] = field(default_factory=dict)
    attempts: dict[str, AttemptRecord] = field(default_factory=dict)
    records: dict[tuple[str, str], StoredRecord] = field(default_factory=dict)
    events: list[EventRecord] = field(default_factory=list)


class InMemoryCardRepository:
    def __init__(self, records: dict[str, str]) -> None:
        self._records = records

    def get(self, card_id: str) -> TestCard | None:
        raw = self._records.get(card_id)
        return decode_card(raw) if raw is not None else None

    def list_all(self) -> list[TestCard]:
        return [decode_card(self._records[key]) for key in sorted(self._records)]

    def save(self, card: TestCard) -> None:
        payload = encode_card(card)
        current = self._records.get(card.card_id)
        if current is not None:
            saved = decode_card(current)
            if saved.revision > card.revision:
                raise StorageConflictError(
                    f"Карточка {card.card_id} уже сохранена в ревизии {saved.revision}"
                )
            if saved.revision == card.revision and current == payload:
                return
        self._records[card.card_id] = payload


class InMemorySessionRepository:
    def __init__(self, records: dict[str, SessionRecord]) -> None:
        self._records = records

    def get(self, session_id: str) -> SessionRecord | None:
        return self._records.get(session_id)

    def list_all(self) -> list[SessionRecord]:
        return [self._records[key] for key in sorted(self._records)]

    def save(self, session: SessionRecord) -> None:
        self._records[session.session_id] = session


class InMemoryAttemptRepository:
    def __init__(self, records: dict[str, AttemptRecord]) -> None:
        self._records = records

    def get(self, attempt_id: str) -> AttemptRecord | None:
        return self._records.get(attempt_id)

    def active_for_session(self, session_id: str) -> AttemptRecord | None:
        matches = [
            item
            for item in self._records.values()
            if item.session_id == session_id and item.status is AttemptStatus.ACTIVE
        ]
        return max(matches, key=lambda item: item.updated_at, default=None)

    def list_for_session(self, session_id: str) -> list[AttemptRecord]:
        return sorted(
            (item for item in self._records.values() if item.session_id == session_id),
            key=lambda item: (item.updated_at, item.attempt_id),
        )

    def list_all(self) -> list[AttemptRecord]:
        return sorted(self._records.values(), key=lambda item: (item.updated_at, item.attempt_id))

    def save(self, attempt: AttemptRecord) -> None:
        self._records[attempt.attempt_id] = attempt


class InMemoryRecordRepository:
    def __init__(self, records: dict[tuple[str, str], StoredRecord]) -> None:
        self._records = records

    def get(self, kind: str, record_id: str) -> StoredRecord | None:
        return self._records.get((kind, record_id))

    def list_kind(self, kind: str) -> list[StoredRecord]:
        return sorted(
            (record for (record_kind, _), record in self._records.items() if record_kind == kind),
            key=lambda record: record.record_id,
        )

    def save(self, record: StoredRecord) -> None:
        self._records[(record.kind, record.record_id)] = record


class InMemoryEventRepository:
    def __init__(self, events: list[EventRecord]) -> None:
        self._events = events

    def append(self, aggregate_id: str, event_type: str, payload: dict[str, object]) -> int:
        sequence = len(self._events) + 1
        self._events.append(
            EventRecord(
                sequence=sequence,
                aggregate_id=aggregate_id,
                event_type=event_type,
                payload=dict(payload),
            )
        )
        return sequence

    def list_for(self, aggregate_id: str) -> list[EventRecord]:
        return [event for event in self._events if event.aggregate_id == aggregate_id]


class InMemoryUnitOfWork:
    def __init__(self, database: InMemoryDatabase) -> None:
        self._database = database

    def __enter__(self) -> InMemoryUnitOfWork:
        self._cards = dict(self._database.cards)
        self._sessions = dict(self._database.sessions)
        self._attempts = dict(self._database.attempts)
        self._records = dict(self._database.records)
        self._events = list(self._database.events)
        self.cards = InMemoryCardRepository(self._cards)
        self.sessions = InMemorySessionRepository(self._sessions)
        self.attempts = InMemoryAttemptRepository(self._attempts)
        self.records = InMemoryRecordRepository(self._records)
        self.events = InMemoryEventRepository(self._events)
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if exc_type is not None:
            return
        self._database.cards = self._cards
        self._database.sessions = self._sessions
        self._database.attempts = self._attempts
        self._database.records = self._records
        self._database.events = self._events
