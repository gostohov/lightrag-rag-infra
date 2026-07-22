from __future__ import annotations

from pathlib import Path
from typing import Callable

from ...domain import TestCard
from ...domain.enums import CardDecisionKind
from ..repositories import UnitOfWork
from ..selection_review import SelectionReviewService
from .renderer import MarkdownCardRenderer


class ExportBlockedError(RuntimeError):
    pass


class FullPmiExportService:
    def __init__(
        self,
        *,
        run_dir: Path,
        uow_factory: Callable[[], UnitOfWork],
        reviews: SelectionReviewService,
        renderer: MarkdownCardRenderer,
    ) -> None:
        self.run_dir = run_dir
        self.uow_factory = uow_factory
        self.reviews = reviews
        self.renderer = renderer

    def export_card(self, card_id: str) -> Path:
        with self.uow_factory() as uow:
            card = uow.cards.get(card_id)
        if card is None:
            raise ExportBlockedError(f"Карточка {card_id} не найдена")
        if not self.reviews.can_export(card.selection_id):
            raise ExportBlockedError("Диапазон карточки не имеет актуальной проверки")
        path = self.run_dir / "review" / "exports" / f"{card_id}.md"
        self._write(path, self.renderer.render(card))
        return path

    def export_selection(self, selection_id: str) -> Path:
        if not self.reviews.can_export(selection_id):
            raise ExportBlockedError("Диапазон не имеет актуальной принятой проверки")
        with self.uow_factory() as uow:
            cards = tuple(
                card
                for card in uow.cards.list_all()
                if card.selection_id == selection_id
            )
        included = self._included(cards)
        included.sort(key=lambda card: card.card_id)
        lines = [f"# Технический ПМИ диапазона {selection_id}", ""]
        if not included:
            lines.append("Включённые карточки отсутствуют.")
        for index, card in enumerate(included):
            if index:
                lines.extend(["", "---", ""])
            lines.append(self.renderer.render(card).rstrip())
        slug = selection_id.lower().replace("_", "-")
        path = self.run_dir / "review" / "exports" / f"pmi-{slug}.md"
        self._write(path, "\n".join(lines).rstrip() + "\n")
        return path

    def export_full(self) -> Path:
        with self.uow_factory() as uow:
            superseded = {
                record.record_id
                for record in uow.records.list_kind("selection_supersession")
            }
            cards = tuple(
                card
                for card in uow.cards.list_all()
                if card.selection_id not in superseded
            )
            decomposition = tuple(
                record
                for record in uow.records.list_kind("decomposition")
                if str(record.payload.get("selection_id") or record.record_id)
                not in superseded
            )
            selections = {
                record.record_id: record
                for record in uow.records.list_kind("source_selection")
                if record.record_id not in superseded
            }
        touched = {
            str(record.payload.get("selection_id") or record.record_id)
            for record in decomposition
            if record.payload.get("outcome") == "skeletons_created"
        }
        touched.update(card.selection_id for card in cards)
        blocked = sorted(selection_id for selection_id in touched if not self.reviews.can_export(selection_id))
        if blocked:
            raise ExportBlockedError(
                "Полный экспорт заблокирован диапазонами без актуальной проверки: "
                + ", ".join(blocked)
            )
        included = self._included(cards)
        included.sort(key=lambda card: (*self._selection_order(selections.get(card.selection_id)), card.card_id))
        lines = ["# Технический ПМИ", ""]
        if not included:
            lines.append("Включённые карточки отсутствуют.")
        for index, card in enumerate(included):
            if index:
                lines.extend(["", "---", ""])
            lines.append(self.renderer.render(card).rstrip())
        path = self.run_dir / "review" / "exports" / "pmi-full.md"
        self._write(path, "\n".join(lines).rstrip() + "\n")
        return path

    @staticmethod
    def _included(cards: tuple[TestCard, ...]) -> list[TestCard]:
        return [
            card
            for card in cards
            if card.decision
            and card.decision.revision == card.revision
            and card.decision.kind
            in {CardDecisionKind.INCLUDE, CardDecisionKind.INCLUDE_INCOMPLETE}
        ]

    @staticmethod
    def _selection_order(record: object | None) -> tuple[int, int, str]:
        if record is None:
            return (10**9, 10**9, "")
        start = record.payload.get("start", {})
        return (
            int(start.get("page_index", 10**9)),
            int(start.get("line_number", 10**9)),
            record.record_id,
        )

    @staticmethod
    def _write(path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(path)
