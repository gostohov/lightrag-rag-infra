from __future__ import annotations

from datetime import datetime
from typing import Callable

from ..repositories import UnitOfWork
from ..state import StoredRecord


class AnalystProposalService:
    KIND = "analyst_answer_proposal"

    def __init__(
        self,
        *,
        uow_factory: Callable[[], UnitOfWork],
        next_id: Callable[[str], str],
        clock: Callable[[], datetime],
    ) -> None:
        self.uow_factory = uow_factory
        self.next_id = next_id
        self.clock = clock

    def pending(
        self,
        *,
        session_id: str,
        card_id: str,
        card_revision: int,
        open_gap_id: str | None,
    ) -> StoredRecord | None:
        with self.uow_factory() as uow:
            active: list[StoredRecord] = []
            for record in uow.records.list_kind(self.KIND):
                if (
                    record.payload.get("session_id") != session_id
                    or record.payload.get("card_id") != card_id
                    or record.payload.get("status") != "pending"
                ):
                    continue
                proposal_kind = str(
                    record.payload.get("proposal_kind", "gap_answer")
                )
                if (
                    proposal_kind not in {"gap_answer", "refinement"}
                    or (
                        proposal_kind == "gap_answer"
                        and (
                            open_gap_id is None
                            or record.payload.get("gap_id") != open_gap_id
                        )
                    )
                    or (
                        proposal_kind == "refinement"
                        and open_gap_id is not None
                    )
                    or record.payload.get("expected_revision") != card_revision
                ):
                    self.transition(
                        record.record_id,
                        "invalidated",
                        message_id=None,
                        reason="card revision or open gap changed",
                        uow=uow,
                    )
                    continue
                active.append(record)
        return max(active, key=lambda item: item.record_id, default=None)

    def create(
        self,
        *,
        session_id: str,
        card_id: str,
        gap_id: str,
        source_message_id: str,
        expected_revision: int,
        values: list[dict[str, object]],
        closure_evaluation: dict[str, object],
    ) -> StoredRecord:
        proposal_id = self.next_id("PROPOSAL")
        created_at = self.clock().isoformat()
        payload: dict[str, object] = {
            "proposal_kind": "gap_answer",
            "proposal_id": proposal_id,
            "session_id": session_id,
            "card_id": card_id,
            "gap_id": gap_id,
            "source_message_id": source_message_id,
            "expected_revision": expected_revision,
            "values": values,
            "closure_evaluation": closure_evaluation,
            "status": "pending",
            "created_at": created_at,
            "status_history": [
                {
                    "status": "pending",
                    "message_id": source_message_id,
                    "at": created_at,
                }
            ],
        }
        return self._save(payload)

    def create_refinement(
        self,
        *,
        session_id: str,
        card_id: str,
        source_message_id: str,
        expected_revision: int,
        arguments: dict[str, object],
        values: list[dict[str, object]],
        uow: UnitOfWork,
    ) -> StoredRecord:
        proposal_id = self.next_id("PROPOSAL")
        created_at = self.clock().isoformat()
        payload: dict[str, object] = {
            "proposal_kind": "refinement",
            "proposal_id": proposal_id,
            "session_id": session_id,
            "card_id": card_id,
            "gap_id": None,
            "source_message_id": source_message_id,
            "expected_revision": expected_revision,
            "values": values,
            "refinement_arguments": arguments,
            "status": "pending",
            "created_at": created_at,
            "status_history": [
                {
                    "status": "pending",
                    "message_id": source_message_id,
                    "at": created_at,
                }
            ],
        }
        return self._save(payload, uow=uow)

    def _save(
        self,
        payload: dict[str, object],
        *,
        uow: UnitOfWork | None = None,
    ) -> StoredRecord:
        if uow is None:
            with self.uow_factory() as active_uow:
                return self._save(payload, uow=active_uow)
        proposal_id = str(payload["proposal_id"])
        session_id = str(payload["session_id"])
        card_id = str(payload["card_id"])
        source_message_id = str(payload["source_message_id"])
        for existing in uow.records.list_kind(self.KIND):
            if (
                existing.payload.get("session_id") == session_id
                and existing.payload.get("card_id") == card_id
                and existing.payload.get("status") == "pending"
            ):
                self.transition(
                    existing.record_id,
                    "replaced",
                    message_id=source_message_id,
                    reason=f"replaced by {proposal_id}",
                    uow=uow,
                )
        proposal = StoredRecord(self.KIND, proposal_id, payload)
        uow.records.save(proposal)
        uow.events.append(
            card_id,
            "предложена интерпретация ответа аналитика",
            {
                "proposal_id": proposal_id,
                "proposal_kind": payload.get("proposal_kind"),
                "gap_id": payload.get("gap_id"),
                "source_message_id": source_message_id,
                "expected_revision": payload.get("expected_revision"),
                "values": payload.get("values"),
                "closure_evaluation": payload.get("closure_evaluation"),
            },
        )
        return proposal

    def require_pending(
        self,
        *,
        proposal_id: str,
        session_id: str,
        card_id: str,
        expected_revision: int,
    ) -> StoredRecord:
        with self.uow_factory() as uow:
            proposal = uow.records.get(self.KIND, proposal_id)
        if (
            proposal is None
            or proposal.payload.get("status") != "pending"
            or proposal.payload.get("session_id") != session_id
            or proposal.payload.get("card_id") != card_id
            or proposal.payload.get("expected_revision") != expected_revision
        ):
            raise ValueError(
                "Предложенная интерпретация отсутствует или более не актуальна"
            )
        return proposal

    def transition(
        self,
        proposal_id: str,
        status: str,
        *,
        message_id: str | None,
        reason: str | None = None,
        applied_revision: int | None = None,
        uow: UnitOfWork | None = None,
    ) -> None:
        if uow is None:
            with self.uow_factory() as active_uow:
                self.transition(
                    proposal_id,
                    status,
                    message_id=message_id,
                    reason=reason,
                    applied_revision=applied_revision,
                    uow=active_uow,
                )
            return
        proposal = uow.records.get(self.KIND, proposal_id)
        if proposal is None or proposal.payload.get("status") != "pending":
            raise ValueError(
                "Предложенная интерпретация отсутствует или уже обработана"
            )
        changed_at = self.clock().isoformat()
        payload = dict(proposal.payload)
        payload["status"] = status
        if status == "confirmed":
            payload["confirmation_message_id"] = message_id
        elif status == "rejected":
            payload["rejection_message_id"] = message_id
        if reason is not None:
            payload["transition_reason"] = reason
        if applied_revision is not None:
            payload["applied_revision"] = applied_revision
        payload["status_history"] = [
            *list(payload.get("status_history", [])),
            {
                "status": status,
                "message_id": message_id,
                "reason": reason,
                "applied_revision": applied_revision,
                "at": changed_at,
            },
        ]
        uow.records.save(
            StoredRecord(proposal.kind, proposal.record_id, payload)
        )
        uow.events.append(
            str(proposal.payload["card_id"]),
            f"интерпретация ответа аналитика: {status}",
            {
                "proposal_id": proposal_id,
                "message_id": message_id,
                "reason": reason,
                "applied_revision": applied_revision,
            },
        )
