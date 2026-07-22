from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from typing import Callable

from .repositories import UnitOfWork
from .state import AttemptStatus, StoredRecord


class RecoveryService:
    def __init__(
        self,
        uow_factory: Callable[[], UnitOfWork],
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.uow_factory = uow_factory
        self.clock = clock or (lambda: datetime.now(UTC))

    def recover(self) -> tuple[str, ...]:
        recovered: list[str] = []
        with self.uow_factory() as uow:
            for attempt in uow.attempts.list_all():
                if attempt.status not in {
                    AttemptStatus.ACTIVE,
                    AttemptStatus.RESULT_READY,
                    AttemptStatus.APPLYING,
                }:
                    continue
                interrupted_apply = attempt.status is AttemptStatus.APPLYING
                recovered_status = (
                    AttemptStatus.FAILED
                    if interrupted_apply
                    else AttemptStatus.CANCELLED
                )
                reason = (
                    "Workbench перезапущен во время применения результата"
                    if interrupted_apply
                    else "Workbench перезапущен во время активной операции"
                )
                uow.attempts.save(
                    attempt.with_status(recovered_status, self.clock())
                )
                uow.records.save(
                    StoredRecord(
                        "recovery_diagnostic",
                        attempt.attempt_id,
                        {
                            "previous_status": attempt.status.value,
                            "status": recovered_status.value,
                            "reason": reason,
                        },
                    )
                )
                uow.events.append(
                    attempt.attempt_id,
                    "операция прервана при восстановлении",
                    {
                        "stage": attempt.stage,
                        "previous_status": attempt.status.value,
                        "status": recovered_status.value,
                    },
                )
                recovered.append(attempt.attempt_id)
            recovered_set = set(recovered)
            for session in uow.sessions.list_all():
                interrupted = any(
                    attempt.attempt_id in recovered_set
                    and (
                        attempt.session_id == session.session_id
                        or attempt.session_id.startswith(f"{session.session_id}:")
                    )
                    for attempt in uow.attempts.list_all()
                )
                if not interrupted:
                    continue
                card = uow.cards.get(session.card_id) if session.card_id else None
                if card is None or card.revision == 0:
                    continuation = "population"
                elif any(gap.status.value == "открыт" for gap in card.gaps.values()):
                    continuation = "gap_investigation"
                else:
                    continuation = "card_decision"
                payload = dict(session.payload)
                payload.update(
                    {
                        "active_intent": None,
                        "continuation": continuation,
                        "recovered_attempt_ids": sorted(recovered_set),
                    }
                )
                uow.sessions.save(
                    replace(
                        session,
                        current_stage="операция прервана после перезапуска",
                        payload=payload,
                        updated_at=self.clock(),
                    )
                )
        return tuple(sorted(recovered))
