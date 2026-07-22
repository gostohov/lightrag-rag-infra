from __future__ import annotations

import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path

from pmi_generator.workbench.application.exporting import (
    ExportBlockedError,
    FullPmiExportService,
    MarkdownCardRenderer,
)
from pmi_generator.workbench.application.range_workspace import RangeWorkspaceService
from pmi_generator.workbench.application.selection_review import SelectionReviewService
from pmi_generator.workbench.application.session.diagnostics import (
    export_selection_diagnostics,
)
from pmi_generator.workbench.application.state import SessionRecord, StoredRecord
from pmi_generator.workbench.domain import (
    AnalystResolution,
    CardMutation,
    ContentField,
    Evidence,
    RelatedGap,
    SourceAddress,
    TestCard,
)
from pmi_generator.workbench.infrastructure.storage import InMemoryDatabase, InMemoryUnitOfWork
from pmi_generator.workbench.presentation.session import export_session_diagnostics


NOW = datetime(2026, 7, 16, 12, 0, tzinfo=UTC)


def build_card(card_id: str, selection_id: str, *, incomplete: bool = False) -> TestCard:
    card = TestCard.create(
        card_id=card_id, selection_id=selection_id,
        title=f"Проверка {card_id}", section_number="4.16.5",
        changed_factors=("первый байт",), consequences=("SW 6987",),
    )
    source = Evidence.source_fragment(
        evidence_id=f"SRC-{card_id}", card_id=card_id, selection_id=selection_id,
        quote="Если первый байт неверен, карта возвращает 6987.",
        address=SourceAddress("spec_2.3.pdf", "2.3", 283, 20, 24, "section-0270"),
        collected_at=NOW,
    )
    human = Evidence.human_knowledge(
        evidence_id=f"HUM-{card_id}", card_id=card_id, selection_id=selection_id,
        quote="Использовать значение 80.", author="Аналитик", message_id=f"MSG-{card_id}", collected_at=NOW,
    )
    fields = {
        "requirement.condition": ContentField.confirmed("первый байт не равен 81", (source.evidence_id,)),
        "requirement.behavior": ContentField.confirmed("прекратить PUT DATA", (source.evidence_id,)),
        "test.action": ContentField.confirmed("отправить PUT DATA", (source.evidence_id,)),
        "test.changed_factor": ContentField.confirmed("первый байт", (source.evidence_id,)),
        "test.control_values": ContentField.analyst_confirmed(
            ["80"],
            (human.evidence_id,),
        ),
        "test.expected.status_word": ContentField.confirmed("6987", (source.evidence_id,)),
        "test.observation.method": ContentField.confirmed("проверить SW1SW2", (source.evidence_id,)),
    }
    gaps = ()
    if incomplete:
        gaps = (
            RelatedGap(
                gap_id=f"GAP-{card_id}", card_id=card_id,
                question="Как проверить запрет следующих команд?",
                blocking_reason="Нет способа наблюдения",
                allowed_paths=("test.observation.causal_link",),
                dependencies=("test.observation.method",),
                closure_criterion="найден способ",
            ),
        )
    resolution = AnalystResolution(
        resolution_id=f"RES-{card_id}",
        card_id=card_id,
        author="Аналитик",
        created_at=NOW,
        reason="Подтверждённое контрольное значение",
        target_paths=("test.control_values",),
        evidence_ids=(human.evidence_id,),
        source_message_id=human.message_id,
        confirmation_message_id=human.message_id,
        values=(
            {"path": "test.control_values", "value": ["80"]},
        ),
    )
    card.apply(
        CardMutation(
            evidence=(source, human),
            fields=fields,
            gaps=gaps,
            resolutions=(resolution,),
        )
    )
    if incomplete:
        card.include_incomplete(author="Аналитик", reason="Допустимо для технического PoC", at=NOW)
    else:
        card.include(author="Аналитик", at=NOW)
    return card


class CardRendererTests(unittest.TestCase):
    def test_working_snapshot_does_not_require_analyst_decision(self) -> None:
        card = build_card("CARD_1", "SEL_1")
        card._decision = None

        rendered = MarkdownCardRenderer().render_working(card)

        self.assertIn("# Проверка CARD_1", rendered)
        self.assertIn("- Статус: **готова**", rendered)
        self.assertIn("SW1SW2", rendered)
        self.assertIn("подтверждено аналитиком", rendered)

    def test_complete_card_snapshot_uses_only_stored_fields(self) -> None:
        rendered = MarkdownCardRenderer().render(build_card("CARD_1", "SEL_1"))
        self.assertIn("# Проверка CARD_1", rendered)
        self.assertIn("первый байт не равен 81", rendered)
        self.assertIn("отправить PUT DATA", rendered)
        self.assertIn("SW1SW2", rendered)
        self.assertNotIn("None", rendered)
        self.assertNotIn("неизвестно", rendered)

    def test_incomplete_card_shows_gap_and_decision_reason(self) -> None:
        rendered = MarkdownCardRenderer().render(build_card("CARD_1", "SEL_1", incomplete=True))
        self.assertIn("Статус: **включена неполной**", rendered)
        self.assertIn("Допустимо для технического PoC", rendered)
        self.assertIn("Как проверить запрет следующих команд?", rendered)
        self.assertIn("Тип разрешения: `source_fact`", rendered)

    def test_human_value_does_not_render_conflicting_source_quote(self) -> None:
        card = build_card("CARD_1", "SEL_1")
        rendered = MarkdownCardRenderer().render(card)
        self.assertIn("Использовать значение 80.", rendered)
        self.assertNotIn("Конфликт", rendered)


class FullExportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.run_dir = Path(self.temp.name)
        self.database = InMemoryDatabase()
        with InMemoryUnitOfWork(self.database) as uow:
            for current in (
                build_card("CARD_2", "SEL_2"),
                build_card("CARD_1", "SEL_1", incomplete=True),
            ):
                current.mark_selection_review_current()
                uow.cards.save(current)
            for index, selection_id in enumerate(("SEL_1", "SEL_2"), start=1):
                uow.records.save(StoredRecord("source_selection", selection_id, {
                    "section_id": f"section-{index}",
                    "start": {"page_index": 100 + index, "line_number": 2},
                    "end": {"page_index": 100 + index, "line_number": 4},
                    "positions": [], "text": "source",
                }))
                uow.records.save(StoredRecord("decomposition", selection_id, {"selection_id": selection_id}))
                card_id = f"CARD_{index}"
                uow.records.save(StoredRecord("selection_review", selection_id, {
                    "selection_id": selection_id,
                    "outcome": "approved",
                    "issues": [],
                    "card_revisions": {card_id: uow.cards.get(card_id).revision},
                    "analyst_decision": None,
                }))
        workspace = RangeWorkspaceService(uow_factory=lambda: InMemoryUnitOfWork(self.database))
        reviews = SelectionReviewService(
            uow_factory=lambda: InMemoryUnitOfWork(self.database), workspace=workspace,
            next_id=lambda prefix: f"{prefix}_1", clock=lambda: NOW,
        )
        self.service = FullPmiExportService(
            run_dir=self.run_dir,
            uow_factory=lambda: InMemoryUnitOfWork(self.database),
            reviews=reviews,
            renderer=MarkdownCardRenderer(),
        )

    def test_one_stale_selection_blocks_whole_export_without_file(self) -> None:
        with InMemoryUnitOfWork(self.database) as uow:
            review = uow.records.get("selection_review", "SEL_2")
            payload = dict(review.payload)
            payload["card_revisions"] = {"CARD_2": -1}
            uow.records.save(StoredRecord(review.kind, review.record_id, payload))
        with self.assertRaises(ExportBlockedError):
            self.service.export_full()
        self.assertFalse((self.run_dir / "review" / "exports" / "pmi-full.md").exists())

    def test_order_is_source_then_card_id_and_repeat_is_byte_identical(self) -> None:
        first = self.service.export_full()
        first_bytes = first.read_bytes()
        second = self.service.export_full()
        self.assertEqual(first_bytes, second.read_bytes())
        content = first.read_text(encoding="utf-8")
        self.assertLess(content.index("CARD_1"), content.index("CARD_2"))

    def test_old_revision_decision_is_not_exported(self) -> None:
        with InMemoryUnitOfWork(self.database) as uow:
            card = uow.cards.get("CARD_1")
            card.revision += 1
            card._decision = None
            uow.cards.save(card)
        with self.assertRaises(ExportBlockedError):
            self.service.export_full()

    def test_export_has_no_llm_or_retrieval_dependency(self) -> None:
        self.assertFalse(hasattr(self.service, "llm"))
        self.assertFalse(hasattr(self.service, "retrieval"))

    def test_empty_decomposition_outcomes_do_not_block_full_export(self) -> None:
        with InMemoryUnitOfWork(self.database) as uow:
            for selection_id, outcome in (
                ("SEL_EMPTY", "no_testable_behavior"),
                ("SEL_SHORT", "insufficient_selection"),
            ):
                uow.records.save(
                    StoredRecord(
                        "decomposition",
                        selection_id,
                        {"selection_id": selection_id, "outcome": outcome},
                    )
                )

        path = self.service.export_full()

        self.assertTrue(path.exists())

    def test_selection_export_contains_only_current_included_cards(self) -> None:
        path = self.service.export_selection("SEL_1")

        self.assertEqual(path.name, "pmi-sel-1.md")
        content = path.read_text(encoding="utf-8")
        self.assertIn("CARD_1", content)
        self.assertNotIn("CARD_2", content)

    def test_full_export_excludes_superseded_selection_but_keeps_its_history(self) -> None:
        with InMemoryUnitOfWork(self.database) as uow:
            uow.records.save(
                StoredRecord(
                    "selection_supersession",
                    "SEL_1",
                    {
                        "selection_id": "SEL_1",
                        "superseded_by": "SEL_NEW",
                    },
                )
            )

        path = self.service.export_full()
        content = path.read_text(encoding="utf-8")

        self.assertNotIn("CARD_1", content)
        self.assertIn("CARD_2", content)
        with InMemoryUnitOfWork(self.database) as uow:
            self.assertIsNotNone(uow.records.get("source_selection", "SEL_1"))
            self.assertIsNotNone(uow.records.get("selection_review", "SEL_1"))


class DiagnosticTests(unittest.TestCase):
    def test_conflict_is_present_only_in_session_diagnostic(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            database = InMemoryDatabase()
            card = build_card("CARD_1", "SEL_1")
            from pmi_generator.workbench.application.state import SessionRecord
            with InMemoryUnitOfWork(database) as uow:
                uow.cards.save(card)
                uow.sessions.save(SessionRecord("SESSION_1", "SEL_1", "CARD_1", "готова", {}, NOW))
                uow.records.save(StoredRecord("refinement_conflict", "CONFLICT_1", {
                    "card_id": "CARD_1", "path": "test.control_values",
                    "previous_value": ["00"], "expert_value": ["80"],
                }))
            path = export_session_diagnostics(
                Path(temp), "SESSION_1", "CARD_1",
                lambda: InMemoryUnitOfWork(database),
            )
            self.assertIn("CONFLICT_1", path.read_text(encoding="utf-8"))

    def test_selection_diagnostic_aggregates_card_sessions_and_review(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            database = InMemoryDatabase()
            card = build_card("CARD_1", "SELECTION_0001")
            with InMemoryUnitOfWork(database) as uow:
                uow.cards.save(card)
                uow.sessions.save(
                    SessionRecord(
                        "SESSION_1",
                        "SELECTION_0001",
                        "CARD_1",
                        "готова",
                        {},
                        NOW,
                    )
                )
                uow.records.save(
                    StoredRecord(
                        "source_selection",
                        "SELECTION_0001",
                        {
                            "section_id": "section-1",
                            "start": {"page_index": 283, "line_number": 1},
                            "end": {"page_index": 283, "line_number": 3},
                            "text": "source",
                        },
                    )
                )
                uow.records.save(
                    StoredRecord(
                        "selection_review",
                        "SELECTION_0001",
                        {
                            "selection_id": "SELECTION_0001",
                            "outcome": "approved",
                            "issues": [],
                            "card_revisions": {"CARD_1": card.revision},
                        },
                    )
                )

            path = export_selection_diagnostics(
                Path(temp),
                "SELECTION_0001",
                lambda: InMemoryUnitOfWork(database),
            )
            content = path.read_text(encoding="utf-8")

            self.assertEqual(path.name, "pmi-selection-0001-session.md")
            self.assertIn("Сессия карточки CARD_1", content)
            self.assertIn("SESSION_1", content)
            self.assertIn('"outcome": "approved"', content)
