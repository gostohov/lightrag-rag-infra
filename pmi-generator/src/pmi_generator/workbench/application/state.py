from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from enum import StrEnum
from typing import Any


class AttemptStatus(StrEnum):
    CREATED = "создана"
    ACTIVE = "выполняется"
    RESULT_READY = "результат готов к применению"
    APPLYING = "результат применяется"
    COMPLETED = "завершена"
    FAILED = "ошибка"
    CANCELLED = "отменена"
    DISCARDED = "поздний результат отброшен"


@dataclass(frozen=True, slots=True)
class SessionRecord:
    session_id: str
    selection_id: str
    card_id: str | None
    current_stage: str
    payload: dict[str, Any]
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class AttemptRecord:
    attempt_id: str
    session_id: str
    stage: str
    status: AttemptStatus
    payload: dict[str, Any]
    updated_at: datetime

    def with_status(self, status: AttemptStatus, at: datetime) -> AttemptRecord:
        return replace(self, status=status, updated_at=at)


@dataclass(frozen=True, slots=True)
class StoredRecord:
    kind: str
    record_id: str
    payload: dict[str, Any]


@dataclass(frozen=True, slots=True)
class EventRecord:
    sequence: int
    aggregate_id: str
    event_type: str
    payload: dict[str, Any]
