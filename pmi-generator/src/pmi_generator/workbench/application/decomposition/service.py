from __future__ import annotations

import hashlib
import json
from typing import Callable

from ...domain import TestCard
from ...domain.source import SourceDocument
from ..card_history import save_card_revision
from ..repositories import UnitOfWork
from ..source import SavedSelection
from ..state import StoredRecord
from .models import DecompositionArguments, DecompositionError, DecompositionResult
from .validation import DecompositionValidator


class DecompositionService:
    def __init__(
        self,
        *,
        document: SourceDocument,
        uow_factory: Callable[[], UnitOfWork],
        next_id: Callable[[str], str],
    ) -> None:
        self.document = document
        self.uow_factory = uow_factory
        self.next_id = next_id
        self.validator = DecompositionValidator(document)

    def apply(
        self,
        selection: SavedSelection,
        arguments: DecompositionArguments,
        *,
        uow: UnitOfWork | None = None,
    ) -> DecompositionResult:
        if uow is not None:
            return self._apply(selection, arguments, uow)
        with self.uow_factory() as owned_uow:
            return self._apply(selection, arguments, owned_uow)

    def _apply(
        self,
        selection: SavedSelection,
        arguments: DecompositionArguments,
        uow: UnitOfWork,
    ) -> DecompositionResult:
        fingerprint = self._fingerprint(arguments)
        existing = uow.records.get("decomposition", selection.selection_id)
        if existing is not None:
            if existing.payload["fingerprint"] != fingerprint:
                raise DecompositionError(
                    "Для диапазона уже сохранён другой результат декомпозиции"
                )
            return self._result_from_record(existing)

        validated = self.validator.validate(selection.selection, arguments)
        section_label = next(
            (
                section.label
                for section in self.document.sections
                if section.section_id == selection.section_id
            ),
            "",
        )
        skeleton_ids = tuple(
            self.next_id("SKELETON") for _item in validated
        )
        for skeleton_id, payload in zip(
            skeleton_ids,
            validated,
            strict=True,
        ):
            uow.records.save(
                StoredRecord(
                    kind="card_skeleton",
                    record_id=skeleton_id,
                    payload={
                        **payload,
                        "selection_id": selection.selection_id,
                        "section_id": selection.section_id,
                        "section_number": section_label,
                        "decision": None,
                        "decision_author": None,
                        "card_id": None,
                    },
                )
            )
        decomposition_payload = {
            "selection_id": selection.selection_id,
            "outcome": arguments.outcome,
            "explanation": arguments.explanation,
            "skeleton_ids": list(skeleton_ids),
            "line_assessments": arguments.line_assessments,
            "fingerprint": fingerprint,
        }
        uow.records.save(
            StoredRecord(
                "decomposition",
                selection.selection_id,
                decomposition_payload,
            )
        )
        uow.events.append(
            selection.selection_id,
            "декомпозиция сохранена",
            {
                "outcome": arguments.outcome,
                "skeleton_count": len(skeleton_ids),
            },
        )
        return DecompositionResult(
            selection_id=selection.selection_id,
            outcome=arguments.outcome,
            explanation=arguments.explanation,
            skeleton_ids=skeleton_ids,
        )

    def take(self, skeleton_id: str, *, author: str) -> str:
        with self.uow_factory() as uow:
            record = self._skeleton(uow, skeleton_id)
            if record.payload["decision"] == "selected":
                return str(record.payload["card_id"])
            if record.payload["decision"] is not None:
                raise DecompositionError(
                    "По каркасу уже принято другое решение"
                )
            card_id = self.next_id("CARD")
            card = TestCard.create(
                card_id=card_id,
                selection_id=str(record.payload["selection_id"]),
                title=str(record.payload["title"]),
                section_number=str(
                    record.payload.get("section_number", "")
                ),
                changed_factors=(str(record.payload["changed_factor"]),),
                consequences=tuple(
                    str(item["text"])
                    for item in record.payload["consequences"]
                ),
            )
            payload = dict(record.payload)
            payload.update(
                {
                    "decision": "selected",
                    "decision_author": author,
                    "card_id": card_id,
                }
            )
            uow.cards.save(card)
            save_card_revision(
                uow,
                card,
                reason="карточка создана из каркаса",
            )
            uow.records.save(
                StoredRecord(record.kind, record.record_id, payload)
            )
            uow.events.append(
                skeleton_id,
                "каркас взят в работу",
                {"card_id": card_id},
            )
            return card_id

    def exclude(
        self,
        skeleton_id: str,
        *,
        author: str,
        reason: str,
    ) -> None:
        if not reason.strip():
            raise DecompositionError(
                "Исключение каркаса требует основания"
            )
        with self.uow_factory() as uow:
            record = self._skeleton(uow, skeleton_id)
            if record.payload["decision"] is not None:
                raise DecompositionError(
                    "По каркасу уже принято решение"
                )
            payload = dict(record.payload)
            payload.update(
                {
                    "decision": "excluded",
                    "decision_author": author,
                    "decision_reason": reason,
                }
            )
            uow.records.save(
                StoredRecord(record.kind, record.record_id, payload)
            )
            uow.events.append(
                skeleton_id,
                "каркас исключён",
                {"reason": reason},
            )

    @staticmethod
    def _fingerprint(arguments: DecompositionArguments) -> str:
        raw = json.dumps(
            {
                "outcome": arguments.outcome,
                "explanation": arguments.explanation,
                "skeletons": arguments.skeletons,
                "line_assessments": arguments.line_assessments,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    @staticmethod
    def _result_from_record(record: StoredRecord) -> DecompositionResult:
        return DecompositionResult(
            selection_id=str(record.payload["selection_id"]),
            outcome=str(record.payload["outcome"]),
            explanation=str(record.payload["explanation"]),
            skeleton_ids=tuple(record.payload["skeleton_ids"]),
        )

    @staticmethod
    def _skeleton(uow: UnitOfWork, skeleton_id: str) -> StoredRecord:
        record = uow.records.get("card_skeleton", skeleton_id)
        if record is None:
            raise DecompositionError(
                f"Каркас {skeleton_id} не найден"
            )
        return record
