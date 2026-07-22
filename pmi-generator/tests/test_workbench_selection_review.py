from __future__ import annotations

import unittest
from datetime import UTC, datetime
from pathlib import Path

from prompt_toolkit.formatted_text.utils import fragment_list_to_text

from pmi_generator.workbench.application.llm import LlmToolRuntime, RawCompletion, TypedToolRegistry
from pmi_generator.workbench.application.prompting import default_policy
from pmi_generator.workbench.application.range_workspace import RangeWorkspaceService
from pmi_generator.workbench.application.selection_review import (
    SelectionReviewArguments,
    SelectionReviewError,
    SelectionReviewFlow,
    SelectionReviewService,
    selection_review_tool,
)
from pmi_generator.workbench.application.source import SavedSelection
from pmi_generator.workbench.application.state import StoredRecord
from pmi_generator.workbench.domain import SourcePosition, TestCard, TextSelection
from pmi_generator.workbench.infrastructure.llm import ScriptedLlmTransport
from pmi_generator.workbench.infrastructure.storage import InMemoryDatabase, InMemoryUnitOfWork
from pmi_generator.workbench.presentation.selection_review import SelectionReviewScreen


NOW = datetime(2026, 7, 16, 12, 0, tzinfo=UTC)


def selection() -> SavedSelection:
    positions = tuple(SourcePosition(283, line) for line in range(19, 25))
    return SavedSelection(
        "SELECTION_0001", "section-0270",
        TextSelection(positions[0], positions[-1], positions, "\n".join(f"строка {p.line_number}" for p in positions)),
    )


def card() -> TestCard:
    result = TestCard.create(
        card_id="CARD_0001", selection_id="SELECTION_0001",
        title="Проверка первого байта", section_number="4.16.5",
        changed_factors=("первый байт",), consequences=("SW 6987",),
    )
    result.include_incomplete(author="Аналитик", reason="PoC", at=NOW)
    return result


def skeleton(decision: str = "selected") -> StoredRecord:
    return StoredRecord(
        "card_skeleton", "SKELETON_0001",
        {
            "selection_id": "SELECTION_0001", "title": "Проверка первого байта",
            "decision": decision, "card_id": "CARD_0001" if decision == "selected" else None,
            "decision_author": "Аналитик", "decision_reason": "не нужен" if decision == "excluded" else None,
        },
    )


def issue() -> dict[str, object]:
    return {
        "kind": "дефект карточки",
        "card_ids": ["CARD_0001"],
        "field_paths": ["test.action"],
        "source_ranges": [{"page": 283, "line_start": 20, "line_end": 21}],
        "explanation": "Не описано конкретное действие терминала.",
    }


class SelectionReviewServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.database = InMemoryDatabase()
        with InMemoryUnitOfWork(self.database) as uow:
            uow.cards.save(card())
            uow.records.save(skeleton())
        self.counter = 0
        self.workspace = RangeWorkspaceService(uow_factory=lambda: InMemoryUnitOfWork(self.database))
        self.service = SelectionReviewService(
            uow_factory=lambda: InMemoryUnitOfWork(self.database),
            workspace=self.workspace,
            next_id=self.next_id,
            clock=lambda: NOW,
        )

    def next_id(self, prefix: str) -> str:
        self.counter += 1
        return f"{prefix}_{self.counter:04d}"

    def test_review_cannot_start_before_all_skeleton_decisions(self) -> None:
        with InMemoryUnitOfWork(self.database) as uow:
            uow.records.save(StoredRecord("card_skeleton", "SKELETON_0002", {"selection_id": "SELECTION_0001", "title": "Без решения", "decision": None, "card_id": None}))
        with self.assertRaisesRegex(SelectionReviewError, "решений"):
            self.service.apply(selection(), SelectionReviewArguments("approved", []))

    def test_approved_requires_empty_issues(self) -> None:
        with self.assertRaises(SelectionReviewError):
            self.service.apply(selection(), SelectionReviewArguments("approved", [issue()]))

    def test_issues_found_requires_nonempty_valid_list_and_assigns_ids(self) -> None:
        result = self.service.apply(selection(), SelectionReviewArguments("issues_found", [issue()]))
        self.assertEqual(result.issue_ids, ("ISSUE_0001",))
        self.assertEqual(result.card_revisions, {"CARD_0001": 0})

    def test_unknown_card_field_and_coordinates_are_rejected(self) -> None:
        mutations = (
            ("card_ids", ["UNKNOWN"]),
            ("field_paths", ["unknown.path"]),
            ("source_ranges", [{"page": 283, "line_start": 10, "line_end": 11}]),
        )
        for key, value in mutations:
            with self.subTest(key=key):
                payload = issue()
                payload[key] = value
                with self.assertRaises(SelectionReviewError):
                    self.service.apply(selection(), SelectionReviewArguments("issues_found", [payload]))

    def test_rejected_skeleton_finishes_range_without_card(self) -> None:
        database = InMemoryDatabase()
        with InMemoryUnitOfWork(database) as uow:
            uow.records.save(skeleton("excluded"))
        workspace = RangeWorkspaceService(uow_factory=lambda: InMemoryUnitOfWork(database))
        service = SelectionReviewService(
            uow_factory=lambda: InMemoryUnitOfWork(database), workspace=workspace,
            next_id=lambda prefix: f"{prefix}_1", clock=lambda: NOW,
        )
        result = service.apply(selection(), SelectionReviewArguments("approved", []))
        self.assertEqual(result.outcome, "approved")

    def test_review_becomes_stale_after_card_revision_change(self) -> None:
        self.service.apply(selection(), SelectionReviewArguments("approved", []))
        with InMemoryUnitOfWork(self.database) as uow:
            current = uow.cards.get("CARD_0001")
            current.revision += 1
            current._decision = None
            uow.cards.save(current)
        self.assertFalse(self.service.is_current("SELECTION_0001"))

    def test_review_marks_cards_current_and_decision_change_makes_it_stale(
        self,
    ) -> None:
        self.service.apply(selection(), SelectionReviewArguments("approved", []))
        with InMemoryUnitOfWork(self.database) as uow:
            current = uow.cards.get("CARD_0001")
            self.assertTrue(current.selection_review_current)
            current.exclude(author="Аналитик", reason="Не включать", at=NOW)
            uow.cards.save(current)

        self.assertFalse(self.service.is_current("SELECTION_0001"))

    def test_continue_with_issues_unlocks_export_for_current_revision_only(self) -> None:
        self.service.apply(selection(), SelectionReviewArguments("issues_found", [issue()]))
        self.assertFalse(self.service.can_export("SELECTION_0001"))
        self.service.proceed("SELECTION_0001", author="Аналитик", reason="Замечание принято")
        self.assertTrue(self.service.can_export("SELECTION_0001"))


class SelectionReviewFlowAndScreenTests(unittest.IsolatedAsyncioTestCase):
    async def test_prompt_context_has_rejected_skeletons_and_no_retrieval_tool(self) -> None:
        database = InMemoryDatabase()
        with InMemoryUnitOfWork(database) as uow:
            uow.cards.save(card())
            uow.records.save(skeleton())
            uow.records.save(StoredRecord("card_skeleton", "S2", {"selection_id": "SELECTION_0001", "title": "Исключённый", "decision": "excluded", "card_id": None, "decision_author": "Аналитик", "decision_reason": "дубликат"}))
        workspace = RangeWorkspaceService(uow_factory=lambda: InMemoryUnitOfWork(database))
        service = SelectionReviewService(
            uow_factory=lambda: InMemoryUnitOfWork(database), workspace=workspace,
            next_id=lambda prefix: f"{prefix}_1", clock=lambda: NOW,
        )
        registry = TypedToolRegistry()
        registry.register(selection_review_tool())
        transport = ScriptedLlmTransport([
            RawCompletion("tool_calls", ({"id": "call", "name": "submit_selection_review", "arguments": {"outcome": "approved", "issues": []}},), {}, "fake")
        ])
        runtime = LlmToolRuntime(
            transport=transport, tools=registry,
            uow_factory=lambda: InMemoryUnitOfWork(database), max_retries=0,
        )
        flow = SelectionReviewFlow(policy=default_policy(), runtime=runtime, service=service)
        await flow.run("ATTEMPT_1", "SESSION_1", selection())
        call = transport.calls[0]["call"]
        self.assertEqual(call.allowed_tools, ("submit_selection_review",))
        self.assertEqual(len(call.context["skeleton_decisions"]), 2)
        self.assertNotIn("retrieval", call.context)

    async def test_screen_wraps_issues_and_full_height_scrollbar(self) -> None:
        text = fragment_list_to_text(
            SelectionReviewScreen(
                [
                    {
                        **issue(),
                        "issue_id": "ISSUE_1",
                        "explanation": "Очень длинное замечание " * 8,
                    }
                ]
            ).render(width=46, height=12)
        )
        rows = [line for line in text.splitlines() if line.endswith("│") or line.endswith("█")]
        self.assertEqual(len(rows), 6)

    async def test_screen_stays_open_with_both_export_paths(self) -> None:
        screen = SelectionReviewScreen([])
        screen.export_paths = (
            Path("review/exports/pmi-selection-0001.md"),
            Path("review/diagnostics/pmi-selection-0001-session.md"),
        )
        screen.follow_cursor = True

        text = fragment_list_to_text(screen.render(width=60, height=18))

        self.assertIn("Файлы сформированы:", text)
        self.assertIn("pmi-selection-0001.md", text)
        self.assertIn("pmi-selection-0001-session.md", text)
