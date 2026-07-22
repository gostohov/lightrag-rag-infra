from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from typing import Any, Callable

from ..repositories import UnitOfWork
from ..state import AttemptRecord, AttemptStatus, SessionRecord, StoredRecord
from .models import SessionEvent, SessionEventKind


class SessionService:
    def __init__(
        self,
        *,
        uow_factory: Callable[[], UnitOfWork],
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.uow_factory = uow_factory
        self.clock = clock or (lambda: datetime.now(UTC))

    def open(self, session_id: str, selection_id: str, card_id: str) -> None:
        with self.uow_factory() as uow:
            existing = uow.sessions.get(session_id)
            if existing is not None:
                if existing.card_id != card_id or existing.selection_id != selection_id:
                    raise ValueError("Session ID уже относится к другой карточке")
                return
            uow.sessions.save(
                SessionRecord(
                    session_id=session_id,
                    selection_id=selection_id,
                    card_id=card_id,
                    current_stage="ожидает первоначального заполнения",
                    payload={
                        "active_intent": None,
                        "continuation": "population",
                    },
                    updated_at=self.clock(),
                )
            )

    def append(
        self,
        session_id: str,
        kind: SessionEventKind,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        with self.uow_factory() as uow:
            if uow.sessions.get(session_id) is None:
                raise ValueError(f"Session {session_id} не найдена")
            return uow.events.append(
                session_id,
                "session_event",
                {
                    "kind": kind.value,
                    "text": text,
                    "created_at": self.clock().isoformat(),
                    "metadata": metadata or {},
                },
            )

    def history(self, session_id: str) -> list[SessionEvent]:
        with self.uow_factory() as uow:
            events = uow.events.list_for(session_id)
        return [
            SessionEvent(
                sequence=event.sequence,
                kind=SessionEventKind(event.payload["kind"]),
                text=str(event.payload["text"]),
                created_at=datetime.fromisoformat(str(event.payload["created_at"])),
                metadata=dict(event.payload.get("metadata", {})),
            )
            for event in events
            if event.event_type == "session_event"
        ]

    def set_stage(
        self,
        session_id: str,
        stage: str,
        *,
        active_intent: dict[str, Any] | None = None,
        continuation: str | None = None,
    ) -> None:
        with self.uow_factory() as uow:
            session = uow.sessions.get(session_id)
            if session is None:
                raise ValueError(f"Session {session_id} не найдена")
            payload = dict(session.payload)
            payload["active_intent"] = active_intent
            if continuation is not None:
                payload["continuation"] = continuation
            uow.sessions.save(
                replace(
                    session,
                    current_stage=stage,
                    payload=payload,
                    updated_at=self.clock(),
                )
            )

    def resume_route(self, session_id: str) -> str:
        with self.uow_factory() as uow:
            session = uow.sessions.get(session_id)
        if session is None:
            raise ValueError(f"Session {session_id} не найдена")
        return str(session.payload.get("continuation") or "population")

    def start_operation(
        self,
        session_id: str,
        attempt_id: str,
        *,
        operation: str,
        attempt_number: int,
    ) -> None:
        now = self.clock()
        with self.uow_factory() as uow:
            if uow.attempts.active_for_session(session_id):
                raise ValueError("В session уже выполняется операция")
            uow.attempts.save(
                AttemptRecord(
                    attempt_id=attempt_id,
                    session_id=session_id,
                    stage=operation,
                    status=AttemptStatus.ACTIVE,
                    payload={"attempt_number": attempt_number},
                    updated_at=now,
                )
            )
        self.append(
            session_id,
            SessionEventKind.OPERATION,
            f"{operation}\nСтатус: выполняется\nПопытка: {attempt_number}",
            {"attempt_id": attempt_id},
        )

    def cancel_operation(self, session_id: str, attempt_id: str) -> None:
        with self.uow_factory() as uow:
            attempt = uow.attempts.get(attempt_id)
            if (
                attempt is None
                or attempt.session_id != session_id
                or attempt.status not in {
                    AttemptStatus.ACTIVE,
                    AttemptStatus.RESULT_READY,
                }
            ):
                raise ValueError("Операция уже не является активной")
            uow.attempts.save(attempt.with_status(AttemptStatus.CANCELLED, self.clock()))
        self.append(
            session_id,
            SessionEventKind.WORKBENCH,
            "Подготовка прервана аналитиком.",
            {"attempt_id": attempt_id},
        )

    def complete_operation(
        self,
        session_id: str,
        attempt_id: str,
        *,
        summary: str,
        result_event: tuple[
            SessionEventKind,
            str,
            dict[str, Any],
        ]
        | None = None,
    ) -> bool:
        accepted = False
        with self.uow_factory() as uow:
            attempt = uow.attempts.get(attempt_id)
            if attempt is None or attempt.session_id != session_id:
                raise ValueError("Операция не найдена")
            if attempt.status is AttemptStatus.ACTIVE:
                uow.attempts.save(attempt.with_status(AttemptStatus.COMPLETED, self.clock()))
                uow.events.append(
                    session_id,
                    "session_event",
                    {
                        "kind": SessionEventKind.OPERATION.value,
                        "text": f"{summary}\nСтатус: завершено",
                        "created_at": self.clock().isoformat(),
                        "metadata": {"attempt_id": attempt_id},
                    },
                )
                if result_event is not None:
                    kind, text, metadata = result_event
                    uow.events.append(
                        session_id,
                        "session_event",
                        {
                            "kind": kind.value,
                            "text": text,
                            "created_at": self.clock().isoformat(),
                            "metadata": metadata,
                        },
                    )
                accepted = True
            else:
                uow.attempts.save(attempt.with_status(AttemptStatus.DISCARDED, self.clock()))
                uow.records.save(
                    StoredRecord(
                        "session_diagnostic",
                        attempt_id,
                        {
                            "status": AttemptStatus.DISCARDED.value,
                            "summary": summary,
                            "previous_status": attempt.status.value,
                        },
                    )
                )
        return accepted

    def fail_operation(
        self,
        session_id: str,
        attempt_id: str,
        *,
        error: str,
        technical: dict[str, Any] | None = None,
    ) -> None:
        with self.uow_factory() as uow:
            attempt = uow.attempts.get(attempt_id)
            if attempt is None or attempt.status is not AttemptStatus.ACTIVE:
                raise ValueError("Операция уже не является активной")
            uow.attempts.save(attempt.with_status(AttemptStatus.FAILED, self.clock()))
            uow.records.save(
                StoredRecord(
                    "session_diagnostic",
                    attempt_id,
                    {
                        "status": AttemptStatus.FAILED.value,
                        "error": error,
                        "technical": _sanitize(technical or {}),
                    },
                )
            )
        self.append(
            session_id,
            SessionEventKind.ERROR,
            f"{error}\nСтатус: ошибка",
            {"attempt_id": attempt_id},
        )

    def active_attempt(self, session_id: str) -> AttemptRecord | None:
        with self.uow_factory() as uow:
            return uow.attempts.active_for_session(session_id)


def _sanitize(value: Any) -> Any:
    forbidden = ("key", "token", "authorization", "secret", "password")
    if isinstance(value, dict):
        return {
            key: _sanitize(item)
            for key, item in value.items()
            if not any(part in key.casefold() for part in forbidden)
        }
    if isinstance(value, list):
        return [_sanitize(item) for item in value]
    return value
