from __future__ import annotations

import unittest
from datetime import UTC, datetime

from prompt_toolkit.formatted_text.utils import fragment_list_to_text
from prompt_toolkit.input.defaults import create_pipe_input
from prompt_toolkit.output import DummyOutput

from pmi_generator.workbench.application.range_workspace import (
    RangeWorkspaceController,
    RangeWorkspaceService,
    derive_workspace,
)
from pmi_generator.workbench.application.state import SessionRecord, StoredRecord
from pmi_generator.workbench.application.source import SavedSelection
from pmi_generator.workbench.domain import CardMutation, ContentField, Evidence, SourceAddress, TestCard
from pmi_generator.workbench.domain.source import SourcePosition, TextSelection
from pmi_generator.workbench.infrastructure.storage import InMemoryDatabase, InMemoryUnitOfWork
from pmi_generator.workbench.presentation.range_workspace import (
    RangeWorkspaceScreen,
    render_range_workspace,
)


NOW = datetime(2026, 7, 16, 12, 0, tzinfo=UTC)


def skeleton(skeleton_id: str, decision: str | None, card_id: str | None = None) -> StoredRecord:
    return StoredRecord(
        "card_skeleton",
        skeleton_id,
        {
            "selection_id": "SELECTION_0001",
            "title": f"Каркас {skeleton_id}",
            "decision": decision,
            "card_id": card_id,
            "decision_reason": "не нужен" if decision == "excluded" else None,
        },
    )


def card(card_id: str, *, decision: str | None = None) -> TestCard:
    result = TestCard.create(
        card_id=card_id,
        selection_id="SELECTION_0001",
        title=f"Очень длинное название карточки {card_id}, которое должно переноситься",
        section_number="4.16.5",
        changed_factors=("первый байт",),
        consequences=("SW 6987",),
    )
    evidence = Evidence.source_fragment(
        evidence_id=f"E-{card_id}", card_id=card_id, selection_id=result.selection_id,
        quote="Источник", address=SourceAddress("spec.pdf", "2.3", 1, 1, 1), collected_at=NOW,
    )
    result.apply(CardMutation(evidence=(evidence,), fields={"test.action": ContentField.confirmed("APDU", (evidence.evidence_id,))}))
    if decision == "include":
        result.include_incomplete(author="Аналитик", reason="PoC", at=NOW)
    elif decision == "exclude":
        result.exclude(author="Аналитик", reason="Не нужен", at=NOW)
    return result


class PureWorkspaceTests(unittest.TestCase):
    def test_unresolved_skeleton_blocks_review(self) -> None:
        state = derive_workspace("SELECTION_0001", (skeleton("S1", None),), (), None)
        self.assertFalse(state.can_review)
        self.assertEqual(state.items[0].status, "требует решения")

    def test_excluded_skeleton_is_explicit_decision(self) -> None:
        state = derive_workspace("SELECTION_0001", (skeleton("S1", "excluded"),), (), None)
        self.assertTrue(state.can_review)

    def test_empty_decomposition_has_terminal_user_status(self) -> None:
        state = derive_workspace(
            "SELECTION_0001",
            (),
            (),
            None,
            decomposition=StoredRecord(
                "decomposition",
                "SELECTION_0001",
                {
                    "selection_id": "SELECTION_0001",
                    "outcome": "no_testable_behavior",
                },
            ),
        )

        self.assertFalse(state.can_review)
        self.assertEqual(state.terminal_status, "нет тестируемого поведения")
        self.assertEqual(state.terminal_explanation, "")

    def test_selection_without_decomposition_offers_prompt_1_retry(self) -> None:
        state = derive_workspace("SELECTION_0001", (), (), None)

        self.assertEqual(state.items, ())
        self.assertIsNone(state.terminal_status)

    def test_incomplete_card_without_decision_blocks_review(self) -> None:
        state = derive_workspace(
            "SELECTION_0001", (skeleton("S1", "selected", "C1"),), (card("C1"),), None
        )
        self.assertFalse(state.can_review)
        self.assertIn("требуется решение", state.items[0].status)

    def test_current_include_and_exclude_finish_range(self) -> None:
        state = derive_workspace(
            "SELECTION_0001",
            (skeleton("S1", "selected", "C1"), skeleton("S2", "selected", "C2")),
            (card("C1", decision="include"), card("C2", decision="exclude")),
            None,
        )
        self.assertTrue(state.can_review)
        self.assertEqual(state.included_incomplete, 1)
        self.assertEqual(state.excluded, 1)

    def test_review_is_stale_when_revisions_do_not_match(self) -> None:
        test_card = card("C1", decision="include")
        review = StoredRecord(
            "selection_review", "SELECTION_0001",
            {"card_revisions": {"C1": test_card.revision - 1}, "outcome": "approved"},
        )
        state = derive_workspace(
            "SELECTION_0001", (skeleton("S1", "selected", "C1"),), (test_card,), review
        )
        self.assertFalse(state.review_current)
        self.assertTrue(state.review_stale)

    def test_review_is_stale_when_card_decision_changes(self) -> None:
        test_card = card("C1", decision="include")
        test_card.mark_selection_review_current()
        review = StoredRecord(
            "selection_review",
            "SELECTION_0001",
            {
                "selection_id": "SELECTION_0001",
                "card_revisions": {"C1": test_card.revision},
                "outcome": "approved",
            },
        )
        test_card.exclude(author="Аналитик", reason="Не включать", at=NOW)

        state = derive_workspace(
            "SELECTION_0001",
            (skeleton("S1", "selected", "C1"),),
            (test_card,),
            review,
        )

        self.assertFalse(state.review_current)
        self.assertTrue(state.review_stale)


class WorkspaceServiceAndUiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.database = InMemoryDatabase()
        with InMemoryUnitOfWork(self.database) as uow:
            uow.records.save(skeleton("S1", "selected", "C1"))
            uow.cards.save(card("C1", decision="include"))
            uow.sessions.save(SessionRecord("SESSION_0001", "SELECTION_0001", "C1", "готова", {}, NOW))
        self.service = RangeWorkspaceService(
            uow_factory=lambda: InMemoryUnitOfWork(self.database)
        )

    def test_enter_reuses_existing_session(self) -> None:
        before = len(self.database.sessions)
        session_id = self.service.session_for_card("SELECTION_0001", "C1")
        self.assertEqual(session_id, "SESSION_0001")
        self.assertEqual(len(self.database.sessions), before)

    def test_controller_routes_card_to_existing_session(self) -> None:
        controller = RangeWorkspaceController(self.service, "SELECTION_0001")

        self.assertEqual(controller.activate(), ("session", "SESSION_0001"))

    def test_controller_exposes_prompt_1_retry_without_parallel_dialog_logic(self) -> None:
        empty = InMemoryDatabase()
        controller = RangeWorkspaceController(
            RangeWorkspaceService(
                uow_factory=lambda: InMemoryUnitOfWork(empty)
            ),
            "SELECTION_EMPTY",
        )

        self.assertEqual(controller.activate(), ("decompose", "SELECTION_EMPTY"))

    def test_unresolved_skeleton_is_available_after_reopening_range(self) -> None:
        with InMemoryUnitOfWork(self.database) as uow:
            uow.records.save(skeleton("S2", None))
        controller = RangeWorkspaceController(self.service, "SELECTION_0001")
        controller.cursor = 1

        self.assertEqual(controller.activate(), ("skeleton", "S2"))

    def test_controller_restores_cursor_and_scroll_after_session(self) -> None:
        controller = RangeWorkspaceController(self.service, "SELECTION_0001", viewport_height=3)
        controller.cursor = 2
        controller.offset = 1
        controller.leave_for_session()
        controller.cursor = 0
        controller.offset = 0
        controller.return_from_session()
        self.assertEqual((controller.cursor, controller.offset), (2, 1))

    def test_render_wraps_text_and_uses_full_height_scrollbar(self) -> None:
        controller = RangeWorkspaceController(self.service, "SELECTION_0001", viewport_height=5)
        output = render_range_workspace(controller, width=44, height=5)
        body = output.split("\n")
        scrollbar_rows = [line for line in body if line.endswith("│") or line.endswith("█")]
        self.assertEqual(len(scrollbar_rows), 5)
        self.assertNotIn("Очень длинное название карточки C1, которое должно переноситься [", output)

    def test_full_screen_workspace_uses_actual_viewport_and_explicit_counts(self) -> None:
        controller = RangeWorkspaceController(self.service, "SELECTION_0001")
        screen = RangeWorkspaceScreen(
            controller,
            self._selection(),
            section_number="4.16.5",
        )

        text = fragment_list_to_text(screen.render(width=46, height=18))

        self.assertIn("PMI Workbench / 4.16.5", text)
        self.assertIn("Каркасов предложено: 1", text)
        self.assertIn("Включено неполными: 1", text)
        self.assertIn("[включена неполной]", text)
        self.assertEqual(len(text.splitlines()), 18)
        self.assertTrue(any(line.endswith(("│", "█")) for line in text.splitlines()[8:-2]))

    def test_workspace_footer_stays_on_one_line_when_width_is_sufficient(self) -> None:
        screen = RangeWorkspaceScreen(
            RangeWorkspaceController(self.service, "SELECTION_0001"),
            self._selection(),
            section_number="4.16.5",
        )

        text = fragment_list_to_text(screen.render(width=120, height=18))
        footer_lines = [
            line
            for line in text.splitlines()
            if "[↑/↓]" in line or "[Esc] к структуре" in line
        ]

        self.assertEqual(len(footer_lines), 1)
        self.assertIn("[PgUp/PgDn]", footer_lines[0])
        self.assertIn("[Esc] к структуре", footer_lines[0])

    def test_full_screen_enter_routes_to_saved_session(self) -> None:
        controller = RangeWorkspaceController(self.service, "SELECTION_0001")
        with create_pipe_input() as pipe_input:
            pipe_input.send_text("\r")
            result = RangeWorkspaceScreen(
                controller,
                self._selection(),
                section_number="4.16.5",
                input=pipe_input,
                output=DummyOutput(),
            ).run()

        self.assertEqual(result, ("session", "SESSION_0001"))

    @staticmethod
    def _selection() -> SavedSelection:
        return SavedSelection(
            "SELECTION_0001",
            "section-0270",
            TextSelection(
                start=SourcePosition(1, 1),
                end=SourcePosition(1, 3),
                positions=(
                    SourcePosition(1, 1),
                    SourcePosition(1, 2),
                    SourcePosition(1, 3),
                ),
                text="Первая\nВторая\nТретья",
            ),
        )
