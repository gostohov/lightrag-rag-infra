from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from ...domain.source import SourceDocument, SourcePosition, TextSelection
from ..range_workspace.derivation import derive_workspace
from ..range_workspace.models import RangeWorkspaceState
from ..repositories import UnitOfWork
from ..state import StoredRecord


@dataclass(frozen=True, slots=True)
class SavedSelection:
    selection_id: str
    section_id: str
    selection: TextSelection
    document_version: str = ""
    anchor_outline_node_id: str | None = None

    def __post_init__(self) -> None:
        if self.anchor_outline_node_id is None:
            object.__setattr__(self, "anchor_outline_node_id", self.section_id)


class SelectionRangeStatus(StrEnum):
    ACTIVE = "active"
    COMPLETED = "completed"
    TERMINAL = "terminal"


class SelectionConflictError(ValueError):
    pass


class StaleSelectionError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class SelectionRangeSummary:
    selection_id: str
    section_id: str
    selection: TextSelection
    status: SelectionRangeStatus
    status_text: str
    document_version: str = ""
    anchor_outline_node_id: str | None = None

    def __post_init__(self) -> None:
        if self.anchor_outline_node_id is None:
            object.__setattr__(self, "anchor_outline_node_id", self.section_id)


class SelectionService:
    def __init__(
        self,
        uow: UnitOfWork,
        *,
        document: SourceDocument | None = None,
    ) -> None:
        self._uow = uow
        self._document = document

    def save(
        self,
        selection_id: str,
        section_id: str,
        selection: TextSelection,
        *,
        supersede_selection_ids: tuple[str, ...] = (),
    ) -> None:
        document_version = ""
        if self._document is not None:
            self._outline_node(section_id)
            self._validate_selection(selection)
            document_version = self._document.metadata.document_version
        overlaps = self.overlaps(selection)
        overlapping_ids = {item.selection_id for item in overlaps}
        superseded_ids = set(supersede_selection_ids)
        if superseded_ids != overlapping_ids:
            conflict = ", ".join(sorted(overlapping_ids or superseded_ids))
            raise SelectionConflictError(
                f"Новый диапазон пересекается с сохранёнными диапазонами: {conflict}"
            )
        if len(overlaps) > 1:
            raise SelectionConflictError(
                "Можно заменить ровно один полностью поглощённый диапазон: "
                + ", ".join(sorted(overlapping_ids))
            )
        if overlaps:
            previous = overlaps[0]
            if not set(previous.selection.positions) < set(selection.positions):
                raise SelectionConflictError(
                    "Можно заменить только один полностью поглощённый диапазон: "
                    + previous.selection_id
                )
            if self._uow.attempts.active_for_session(previous.selection_id) is not None:
                raise SelectionConflictError(
                    "Перед заменой диапазона остановите его активную операцию: "
                    + previous.selection_id
                )
        self._uow.records.save(
            StoredRecord(
                kind="source_selection",
                record_id=selection_id,
                payload={
                    "section_id": section_id,
                    "anchor_outline_node_id": section_id,
                    "document_version": document_version,
                    "start": self._encode_position(selection.start),
                    "end": self._encode_position(selection.end),
                    "positions": [self._encode_position(item) for item in selection.positions],
                    "text": selection.text,
                },
            )
        )
        self._uow.events.append(selection_id, "диапазон сохранен", {"section_id": section_id})
        for superseded_id in supersede_selection_ids:
            self._uow.records.save(
                StoredRecord(
                    kind="selection_supersession",
                    record_id=superseded_id,
                    payload={
                        "selection_id": superseded_id,
                        "superseded_by": selection_id,
                    },
                )
            )
            self._uow.events.append(
                superseded_id,
                "диапазон замещён",
                {"superseded_by": selection_id},
            )

    def get(self, selection_id: str) -> SavedSelection | None:
        record = self._uow.records.get("source_selection", selection_id)
        if record is None:
            return None
        payload = record.payload
        saved = SavedSelection(
            selection_id=selection_id,
            section_id=str(payload["section_id"]),
            selection=TextSelection(
                start=self._decode_position(payload["start"]),
                end=self._decode_position(payload["end"]),
                positions=tuple(
                    self._decode_position(item) for item in payload.get("positions", [])
                ),
                text=str(payload["text"]),
            ),
            document_version=str(payload.get("document_version") or ""),
            anchor_outline_node_id=str(
                payload.get("anchor_outline_node_id") or payload["section_id"]
            ),
        )
        if self._document is None:
            return saved
        if (
            saved.document_version
            and saved.document_version != self._document.metadata.document_version
        ):
            raise StaleSelectionError(
                f"Диапазон {selection_id} относится к другой версии source document"
            )
        self._outline_node(saved.anchor_outline_node_id or saved.section_id)
        self._validate_selection(saved.selection)
        if saved.document_version:
            return saved
        return SavedSelection(
            saved.selection_id,
            saved.section_id,
            saved.selection,
            self._document.metadata.document_version,
            saved.anchor_outline_node_id,
        )

    def ranges(self) -> tuple[SelectionRangeSummary, ...]:
        superseded = {
            record.record_id
            for record in self._uow.records.list_kind("selection_supersession")
        }
        skeletons = tuple(self._uow.records.list_kind("card_skeleton"))
        cards = tuple(self._uow.cards.list_all())
        sessions = tuple(self._uow.sessions.list_all())
        result: list[SelectionRangeSummary] = []
        for record in self._uow.records.list_kind("source_selection"):
            if record.record_id in superseded:
                continue
            saved = self.get(record.record_id)
            if saved is None:
                continue
            state = derive_workspace(
                saved.selection_id,
                skeletons,
                cards,
                self._uow.records.get("selection_review", saved.selection_id),
                {
                    item.card_id: item.session_id
                    for item in sessions
                    if item.selection_id == saved.selection_id
                },
                self._uow.records.get("decomposition", saved.selection_id),
            )
            status, status_text = self._range_status(state)
            result.append(
                SelectionRangeSummary(
                    saved.selection_id,
                    saved.section_id,
                    saved.selection,
                    status,
                    status_text,
                    saved.document_version,
                    saved.anchor_outline_node_id,
                )
            )
        return tuple(result)

    def overlaps(
        self,
        selection: TextSelection,
    ) -> tuple[SelectionRangeSummary, ...]:
        positions = set(selection.positions)
        return tuple(
            item
            for item in self.ranges()
            if bool(positions.intersection(item.selection.positions))
        )

    @staticmethod
    def _range_status(
        state: RangeWorkspaceState,
    ) -> tuple[SelectionRangeStatus, str]:
        terminal_status = state.terminal_status
        if terminal_status:
            return SelectionRangeStatus.TERMINAL, str(terminal_status)
        if state.review_current:
            included = state.included + state.included_incomplete
            suffix = f"; включено карточек — {included}" if included else ""
            return SelectionRangeStatus.COMPLETED, f"диапазон проверен{suffix}"
        if state.review_stale:
            return SelectionRangeStatus.ACTIVE, "проверка диапазона устарела"
        items = state.items
        if not items:
            return SelectionRangeStatus.ACTIVE, "диапазон сохранён — продолжить"
        if state.can_review:
            return SelectionRangeStatus.ACTIVE, "диапазон готов к проверке"
        cards = sum(item.card_id is not None for item in items)
        if cards:
            return SelectionRangeStatus.ACTIVE, f"карточки в работе — {cards}"
        return (
            SelectionRangeStatus.ACTIVE,
            f"каркасы требуют решения — {len(items)}",
        )

    @staticmethod
    def _encode_position(position: SourcePosition) -> dict[str, int]:
        return {"page_index": position.page_index, "line_number": position.line_number}

    @staticmethod
    def _decode_position(value: object) -> SourcePosition:
        payload = dict(value)  # type: ignore[arg-type]
        return SourcePosition(
            page_index=int(payload["page_index"]),
            line_number=int(payload["line_number"]),
        )

    def _outline_node(self, section_id: str) -> None:
        assert self._document is not None
        if not self._document.sections:
            return
        if not any(
            section.section_id == section_id
            for section in self._document.sections
        ):
            raise StaleSelectionError(
                f"Навигационный anchor {section_id} отсутствует в source document"
            )

    def _validate_selection(self, selection: TextSelection) -> None:
        assert self._document is not None
        try:
            current = self._document.select(selection.start, selection.end)
        except ValueError as error:
            raise StaleSelectionError(
                f"Координаты сохранённого диапазона недоступны: {error}"
            ) from error
        if current.positions != selection.positions:
            raise StaleSelectionError(
                "Координаты сохранённого диапазона не совпадают с source document"
            )
        if current.text != selection.text:
            raise StaleSelectionError(
                "Сохранённый текст диапазона не совпадает с source document"
            )
