from __future__ import annotations

from contextlib import nullcontext
from datetime import UTC, datetime
from typing import Callable

from ...domain import SourcePosition
from ...domain.schema import CARD_FIELD_PATHS
from ..range_workspace import RangeWorkspaceService
from ..repositories import UnitOfWork
from ..review_status import selection_review_is_current
from ..source import SavedSelection
from ..state import StoredRecord
from .models import SelectionReviewArguments, SelectionReviewError, SelectionReviewResult


ISSUE_KINDS = {
    "пропуск",
    "неподтверждённое добавление",
    "неверное объединение",
    "неверное разделение",
    "дефект карточки",
}


class SelectionReviewService:
    def __init__(
        self,
        *,
        uow_factory: Callable[[], UnitOfWork],
        workspace: RangeWorkspaceService,
        next_id: Callable[[str], str],
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.uow_factory = uow_factory
        self.workspace = workspace
        self.next_id = next_id
        self.clock = clock or (lambda: datetime.now(UTC))

    def apply(
        self,
        selection: SavedSelection,
        arguments: SelectionReviewArguments,
        *,
        uow: UnitOfWork | None = None,
    ) -> SelectionReviewResult:
        if not self.workspace.load(selection.selection_id, uow=uow).can_review:
            raise SelectionReviewError("Проверка недоступна до решений по всем каркасам и карточкам")
        if arguments.outcome not in {"approved", "issues_found"}:
            raise SelectionReviewError("Неизвестный исход проверки диапазона")
        if arguments.outcome == "approved" and arguments.issues:
            raise SelectionReviewError("approved требует пустой список замечаний")
        if arguments.outcome == "issues_found" and not arguments.issues:
            raise SelectionReviewError("issues_found требует замечания")

        context = nullcontext(uow) if uow is not None else self.uow_factory()
        with context as active_uow:
            cards = tuple(
                card for card in active_uow.cards.list_all()
                if card.selection_id == selection.selection_id
            )
            card_ids = {card.card_id for card in cards}
            issues = tuple(
                self._validate_issue(selection, item, card_ids)
                for item in arguments.issues
            )
            revisions = {card.card_id: card.revision for card in cards}
            payload = {
                "selection_id": selection.selection_id,
                "outcome": arguments.outcome,
                "issues": list(issues),
                "card_revisions": revisions,
                "analyst_decision": None,
                "created_at": self.clock().isoformat(),
            }
            active_uow.records.save(StoredRecord("selection_review", selection.selection_id, payload))
            for card in cards:
                card.mark_selection_review_current()
                active_uow.cards.save(card)
            active_uow.events.append(
                selection.selection_id,
                "диапазон проверен",
                {"outcome": arguments.outcome, "issues": len(issues), "card_revisions": revisions},
            )
        return SelectionReviewResult(
            selection.selection_id,
            arguments.outcome,
            tuple(str(item["issue_id"]) for item in issues),
            revisions,
        )

    def proceed(self, selection_id: str, *, author: str, reason: str) -> None:
        if not author.strip() or not reason.strip():
            raise SelectionReviewError("Продолжение с замечаниями требует автора и основание")
        with self.uow_factory() as uow:
            record = uow.records.get("selection_review", selection_id)
            if record is None or record.payload.get("outcome") != "issues_found":
                raise SelectionReviewError("Нет актуальных замечаний для явного продолжения")
            if not self._record_current(uow, record):
                raise SelectionReviewError("Проверка диапазона устарела")
            payload = dict(record.payload)
            payload["analyst_decision"] = {
                "action": "продолжить с замечаниями",
                "author": author,
                "reason": reason,
                "created_at": self.clock().isoformat(),
            }
            uow.records.save(StoredRecord(record.kind, record.record_id, payload))
            uow.events.append(selection_id, "аналитик принял замечания", {"reason": reason})

    def is_current(self, selection_id: str) -> bool:
        with self.uow_factory() as uow:
            record = uow.records.get("selection_review", selection_id)
            return bool(record and self._record_current(uow, record))

    def can_export(self, selection_id: str) -> bool:
        with self.uow_factory() as uow:
            record = uow.records.get("selection_review", selection_id)
            if record is None or not self._record_current(uow, record):
                return False
            return bool(
                record.payload.get("outcome") == "approved"
                or (record.payload.get("analyst_decision") or {}).get("action")
                == "продолжить с замечаниями"
            )

    def _validate_issue(
        self,
        selection: SavedSelection,
        item: object,
        card_ids: set[str],
    ) -> dict[str, object]:
        expected = {"kind", "card_ids", "field_paths", "source_ranges", "explanation"}
        if not isinstance(item, dict) or set(item) != expected:
            raise SelectionReviewError("Замечание имеет неверную структуру")
        if str(item["kind"]) not in ISSUE_KINDS:
            raise SelectionReviewError("Неизвестный тип замечания")
        referenced_cards = {str(value) for value in item["card_ids"]}
        if referenced_cards - card_ids:
            raise SelectionReviewError("Замечание ссылается на неизвестную карточку")
        paths = {str(value) for value in item["field_paths"]}
        if paths - CARD_FIELD_PATHS:
            raise SelectionReviewError("Замечание ссылается на неизвестное поле")
        if not str(item["explanation"]).strip():
            raise SelectionReviewError("Замечание требует объяснение")
        ranges = item["source_ranges"]
        if not isinstance(ranges, list) or not ranges:
            raise SelectionReviewError("Замечание требует координаты источника")
        allowed = set(selection.selection.positions)
        normalized_ranges: list[dict[str, int]] = []
        for raw in ranges:
            if not isinstance(raw, dict) or set(raw) != {"page", "line_start", "line_end"}:
                raise SelectionReviewError("Координаты замечания имеют неверную структуру")
            page = int(raw["page"])
            start = int(raw["line_start"])
            end = int(raw["line_end"])
            positions = {SourcePosition(page, line) for line in range(start, end + 1)}
            if not positions or not positions <= allowed:
                raise SelectionReviewError("Координаты замечания выходят за выбранный диапазон")
            normalized_ranges.append({"page": page, "line_start": start, "line_end": end})
        return {
            "issue_id": self.next_id("ISSUE"),
            "kind": str(item["kind"]),
            "card_ids": sorted(referenced_cards),
            "field_paths": sorted(paths),
            "source_ranges": normalized_ranges,
            "explanation": str(item["explanation"]),
        }

    @staticmethod
    def _record_current(uow: UnitOfWork, record: StoredRecord) -> bool:
        selection_id = str(record.payload["selection_id"])
        return selection_review_is_current(
            record,
            uow.cards.list_all(),
            selection_id,
        )
