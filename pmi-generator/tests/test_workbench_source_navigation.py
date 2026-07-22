from __future__ import annotations

import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path

from prompt_toolkit.formatted_text.utils import fragment_list_to_text
from prompt_toolkit.input.defaults import create_pipe_input
from prompt_toolkit.output import DummyOutput

from pmi_generator.workbench.application.source import (
    SelectionConflictError,
    SelectionRangeStatus,
    SelectionRangeSummary,
    SelectionService,
    StaleSelectionError,
)
from pmi_generator.workbench.application.decomposition import (
    DecompositionBudget,
    DecompositionRoute,
    WindowingDecision,
)
from pmi_generator.workbench.application.state import (
    AttemptRecord,
    AttemptStatus,
    StoredRecord,
)
from pmi_generator.workbench.domain.source import (
    SourceDocument,
    SourcePage,
    SourcePosition,
    SourceSection,
    TextSelection,
)
from pmi_generator.workbench.infrastructure.source import load_source_document
from pmi_generator.workbench.infrastructure.storage import (
    InMemoryDatabase,
    InMemoryUnitOfWork,
    SqliteUnitOfWork,
)
from pmi_generator.workbench.presentation.source import (
    ConfirmationScreen,
    CreateSelection,
    OpenRange,
    SelectionScreen,
    SourceNavigationState,
    StructureScreen,
    select_text_range,
)
from tests.source_fixture import write_source_snapshot


def write_source(run_dir: Path) -> None:
    write_source_snapshot(
        run_dir,
        pages=(
            SourcePage(
                1,
                283,
                (
                    "4.16.5 Обработка команды",
                    "Первая длинная строка требования",
                    "Вторая строка",
                ),
            ),
            SourcePage(2, 284, ("Продолжение требования", "Последняя строка")),
        ),
        sections=(
            SourceSection(
                "section-0270",
                "4.16.5",
                "Обработка команды",
                ("4", "4.16", "4.16.5"),
                (1, 2),
            ),
        ),
        original_name="EMV spec 2.3.pdf",
    )


class SourceSelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.run_dir = Path(self.temp_dir.name)
        write_source(self.run_dir)
        self.document = load_source_document(self.run_dir)

    def test_nonexistent_page_and_line_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "страницы 99"):
            self.document.line(SourcePosition(99, 1))
        with self.assertRaisesRegex(ValueError, "строки 9"):
            self.document.line(SourcePosition(1, 9))

    def test_selection_is_contiguous_and_normalized_in_both_directions(self) -> None:
        forward = self.document.select(SourcePosition(1, 2), SourcePosition(2, 1))
        backward = self.document.select(SourcePosition(2, 1), SourcePosition(1, 2))

        self.assertEqual(forward, backward)
        self.assertEqual(
            forward.positions,
            (SourcePosition(1, 2), SourcePosition(1, 3), SourcePosition(2, 1)),
        )

    def test_selection_text_is_copied_verbatim_from_source_snapshot(self) -> None:
        selection = self.document.select(SourcePosition(1, 2), SourcePosition(2, 1))

        self.assertEqual(
            selection.text,
            "Первая длинная строка требования\nВторая строка\nПродолжение требования",
        )

    def test_selection_survives_sqlite_restart(self) -> None:
        database = self.run_dir / "review" / "workbench.sqlite3"
        selection = self.document.select(SourcePosition(1, 2), SourcePosition(2, 1))
        with SqliteUnitOfWork(database) as uow:
            SelectionService(uow, document=self.document).save(
                "SELECTION_0001",
                "section-0270",
                selection,
            )
        with SqliteUnitOfWork(database) as uow:
            restored = SelectionService(uow, document=self.document).get(
                "SELECTION_0001"
            )

        self.assertEqual(restored.selection, selection)
        self.assertEqual(restored.section_id, "section-0270")
        self.assertEqual(
            restored.document_version,
            self.document.metadata.document_version,
        )
        self.assertEqual(restored.anchor_outline_node_id, "section-0270")

    def test_reopen_rejects_stale_document_version_and_changed_source_text(self) -> None:
        database = InMemoryDatabase()
        selection = self.document.select(SourcePosition(1, 2), SourcePosition(2, 1))
        with InMemoryUnitOfWork(database) as uow:
            SelectionService(uow, document=self.document).save(
                "SELECTION_0001",
                "section-0270",
                selection,
            )
            record = uow.records.get("source_selection", "SELECTION_0001")
            assert record is not None
            uow.records.save(
                StoredRecord(
                    record.kind,
                    record.record_id,
                    {**record.payload, "document_version": "sha256:" + "f" * 64},
                )
            )

        with self.assertRaisesRegex(StaleSelectionError, "верси"):
            with InMemoryUnitOfWork(database) as uow:
                SelectionService(uow, document=self.document).get("SELECTION_0001")

        with InMemoryUnitOfWork(database) as uow:
            record = uow.records.get("source_selection", "SELECTION_0001")
            assert record is not None
            uow.records.save(
                StoredRecord(
                    record.kind,
                    record.record_id,
                    {
                        **record.payload,
                        "document_version": self.document.metadata.document_version,
                        "text": "изменённый текст",
                    },
                )
            )
        with self.assertRaisesRegex(StaleSelectionError, "текст"):
            with InMemoryUnitOfWork(database) as uow:
                SelectionService(uow, document=self.document).get("SELECTION_0001")


class SourceNavigationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.run_dir = Path(self.temp_dir.name)
        write_source(self.run_dir)
        self.document = load_source_document(self.run_dir)

    def test_saved_selection_is_exposed_as_complete_range(self) -> None:
        selection = TextSelection(
            start=SourcePosition(1, 2),
            end=SourcePosition(1, 3),
            positions=(SourcePosition(1, 2), SourcePosition(1, 3)),
            text="Первая длинная строка требования\nВторая строка",
        )
        database = InMemoryDatabase()
        with InMemoryUnitOfWork(database) as uow:
            service = SelectionService(uow)
            service.save("SELECTION_0001", "section-0270", selection)
        with InMemoryUnitOfWork(database) as uow:
            ranges = SelectionService(uow).ranges()

        self.assertEqual(len(ranges), 1)
        self.assertEqual(ranges[0].selection, selection)
        self.assertIs(ranges[0].status, SelectionRangeStatus.ACTIVE)
        self.assertEqual(ranges[0].status_text, "диапазон сохранён — продолжить")

    def test_overlapping_active_selection_is_rejected(self) -> None:
        existing = self.document.select(SourcePosition(1, 1), SourcePosition(1, 2))
        overlapping = self.document.select(SourcePosition(1, 2), SourcePosition(1, 3))
        database = InMemoryDatabase()
        with InMemoryUnitOfWork(database) as uow:
            service = SelectionService(uow)
            service.save("SELECTION_ACTIVE", "section-0270", existing)

        with self.assertRaisesRegex(SelectionConflictError, "SELECTION_ACTIVE"):
            with InMemoryUnitOfWork(database) as uow:
                SelectionService(uow).save(
                    "SELECTION_NEW",
                    "section-0270",
                    overlapping,
                )

    def test_overlap_is_rejected_across_navigation_sections(self) -> None:
        existing = self.document.select(SourcePosition(1, 1), SourcePosition(1, 2))
        overlapping = self.document.select(SourcePosition(1, 2), SourcePosition(1, 3))
        database = InMemoryDatabase()
        with InMemoryUnitOfWork(database) as uow:
            SelectionService(uow).save(
                "SELECTION_PARENT",
                "section-parent",
                existing,
            )

        with self.assertRaisesRegex(SelectionConflictError, "SELECTION_PARENT"):
            with InMemoryUnitOfWork(database) as uow:
                SelectionService(uow).save(
                    "SELECTION_CHILD",
                    "section-child",
                    overlapping,
                )

    def test_larger_selection_supersedes_contained_terminal_range(self) -> None:
        terminal = self.document.select(SourcePosition(1, 1), SourcePosition(1, 2))
        expanded = self.document.select(SourcePosition(1, 1), SourcePosition(1, 3))
        database = InMemoryDatabase()
        with InMemoryUnitOfWork(database) as uow:
            service = SelectionService(uow)
            service.save("SELECTION_TERMINAL", "section-0270", terminal)
            uow.records.save(
                StoredRecord(
                    "decomposition",
                    "SELECTION_TERMINAL",
                    {
                        "selection_id": "SELECTION_TERMINAL",
                        "outcome": "no_testable_behavior",
                        "explanation": "Нет проверяемого поведения",
                        "skeleton_ids": [],
                        "fingerprint": "terminal",
                    },
                )
            )
        with InMemoryUnitOfWork(database) as uow:
            SelectionService(uow).save(
                "SELECTION_EXPANDED",
                "section-0270",
                expanded,
                supersede_selection_ids=("SELECTION_TERMINAL",),
            )
        with InMemoryUnitOfWork(database) as uow:
            service = SelectionService(uow)
            visible = service.ranges()
            historical = service.get("SELECTION_TERMINAL")

        self.assertEqual(
            tuple(item.selection_id for item in visible),
            ("SELECTION_EXPANDED",),
        )
        self.assertIsNotNone(historical)

    def test_larger_selection_can_supersede_exactly_one_contained_active_range(self) -> None:
        contained = self.document.select(SourcePosition(1, 1), SourcePosition(1, 2))
        expanded = self.document.select(SourcePosition(1, 1), SourcePosition(1, 3))
        database = InMemoryDatabase()
        with InMemoryUnitOfWork(database) as uow:
            SelectionService(uow).save("SELECTION_OLD", "section-0270", contained)
        with InMemoryUnitOfWork(database) as uow:
            SelectionService(uow).save(
                "SELECTION_NEW",
                "section-0270",
                expanded,
                supersede_selection_ids=("SELECTION_OLD",),
            )
        with InMemoryUnitOfWork(database) as uow:
            self.assertEqual(
                tuple(item.selection_id for item in SelectionService(uow).ranges()),
                ("SELECTION_NEW",),
            )

    def test_multiple_contained_ranges_cannot_be_superseded_together(self) -> None:
        first = self.document.select(SourcePosition(1, 1), SourcePosition(1, 1))
        second = self.document.select(SourcePosition(1, 3), SourcePosition(1, 3))
        expanded = self.document.select(SourcePosition(1, 1), SourcePosition(2, 1))
        database = InMemoryDatabase()
        with InMemoryUnitOfWork(database) as uow:
            service = SelectionService(uow)
            service.save("SELECTION_FIRST", "section-0270", first)
            service.save("SELECTION_SECOND", "section-0270", second)

        with self.assertRaisesRegex(SelectionConflictError, "ровно один"):
            with InMemoryUnitOfWork(database) as uow:
                SelectionService(uow).save(
                    "SELECTION_NEW",
                    "section-0270",
                    expanded,
                    supersede_selection_ids=(
                        "SELECTION_FIRST",
                        "SELECTION_SECOND",
                    ),
                )

    def test_active_operation_must_be_cancelled_before_supersede(self) -> None:
        contained = self.document.select(SourcePosition(1, 1), SourcePosition(1, 2))
        expanded = self.document.select(SourcePosition(1, 1), SourcePosition(1, 3))
        database = InMemoryDatabase()
        with InMemoryUnitOfWork(database) as uow:
            SelectionService(uow).save("SELECTION_OLD", "section-0270", contained)
            uow.attempts.save(
                AttemptRecord(
                    "ATTEMPT_1",
                    "SELECTION_OLD",
                    "prompt_1",
                    AttemptStatus.ACTIVE,
                    {},
                    datetime(2026, 7, 19, tzinfo=UTC),
                )
            )

        with self.assertRaisesRegex(SelectionConflictError, "остановите"):
            with InMemoryUnitOfWork(database) as uow:
                SelectionService(uow).save(
                    "SELECTION_NEW",
                    "section-0270",
                    expanded,
                    supersede_selection_ids=("SELECTION_OLD",),
                )

    def test_saved_range_status_comes_from_current_workspace_state(self) -> None:
        terminal = self.document.select(SourcePosition(1, 1), SourcePosition(1, 1))
        completed = self.document.select(SourcePosition(1, 2), SourcePosition(1, 2))
        database = InMemoryDatabase()
        with InMemoryUnitOfWork(database) as uow:
            service = SelectionService(uow)
            service.save("SELECTION_TERMINAL", "section-0270", terminal)
            service.save("SELECTION_COMPLETED", "section-0270", completed)
            uow.records.save(
                StoredRecord(
                    "decomposition",
                    "SELECTION_TERMINAL",
                    {
                        "selection_id": "SELECTION_TERMINAL",
                        "outcome": "no_testable_behavior",
                        "explanation": "Нет проверяемого поведения",
                        "skeleton_ids": [],
                        "fingerprint": "terminal",
                    },
                )
            )
            uow.records.save(
                StoredRecord(
                    "selection_review",
                    "SELECTION_COMPLETED",
                    {
                        "selection_id": "SELECTION_COMPLETED",
                        "card_revisions": {},
                    },
                )
            )
        with InMemoryUnitOfWork(database) as uow:
            ranges = {
                item.selection_id: item for item in SelectionService(uow).ranges()
            }

        self.assertIs(
            ranges["SELECTION_TERMINAL"].status,
            SelectionRangeStatus.TERMINAL,
        )
        self.assertEqual(
            ranges["SELECTION_TERMINAL"].status_text,
            "нет тестируемого поведения",
        )
        self.assertIs(
            ranges["SELECTION_COMPLETED"].status,
            SelectionRangeStatus.COMPLETED,
        )
        self.assertEqual(
            ranges["SELECTION_COMPLETED"].status_text,
            "диапазон проверен",
        )


class StructureScreenTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.run_dir = Path(self.temp_dir.name)
        write_source(self.run_dir)
        self.document = load_source_document(self.run_dir)

    def _range(
        self,
        status: SelectionRangeStatus,
        *,
        selection_id: str = "SELECTION_0001",
    ) -> SelectionRangeSummary:
        positions = self.document.positions_for_pages((1,))
        return SelectionRangeSummary(
            selection_id=selection_id,
            section_id="section-0270",
            selection=self.document.select(positions[0], positions[1]),
            status=status,
            status_text=status.value,
        )

    def test_structure_colors_section_by_aggregate_range_status(self) -> None:
        for status, expected_style in (
            (SelectionRangeStatus.ACTIVE, "class:range.active.cursor"),
            (SelectionRangeStatus.COMPLETED, "class:range.completed.cursor"),
            (SelectionRangeStatus.TERMINAL, "class:range.terminal.cursor"),
        ):
            with self.subTest(status=status):
                rendered = StructureScreen(
                    self.document,
                    (self._range(status),),
                ).render(width=100, height=12)

                work_styles = {
                    style
                    for style, text in rendered.fragments
                    if "[1 диапазон:" in text
                }

                self.assertEqual(work_styles, {expected_style})

    def test_active_range_has_priority_in_mixed_section(self) -> None:
        rendered = StructureScreen(
            self.document,
            (
                self._range(
                    SelectionRangeStatus.COMPLETED,
                    selection_id="SELECTION_COMPLETED",
                ),
                self._range(
                    SelectionRangeStatus.ACTIVE,
                    selection_id="SELECTION_ACTIVE",
                ),
            ),
        ).render(width=100, height=12)

        work_styles = {
            style
            for style, text in rendered.fragments
            if "[2 диапазона:" in text
        }

        self.assertEqual(work_styles, {"class:range.active.cursor"})

    def test_structure_status_follows_coordinate_intersection_not_saved_anchor(self) -> None:
        document = SourceDocument(
            pages=(
                SourcePage(1, "1", ("First", "Context")),
                SourcePage(2, "2", ("Second", "Tail")),
            ),
            sections=(
                SourceSection("first", "1", "First", ("1",), (1,)),
                SourceSection("second", "2", "Second", ("2",), (2,)),
            ),
        )
        cross_node = SelectionRangeSummary(
            selection_id="SELECTION_CROSS",
            section_id="first",
            selection=document.select(
                SourcePosition(1, 2),
                SourcePosition(2, 1),
            ),
            status=SelectionRangeStatus.ACTIVE,
            status_text="в работе",
        )

        entries = StructureScreen(document, (cross_node,)).entries()

        self.assertEqual(
            [(entry.key, entry.status) for entry in entries],
            [
                ("section:first", "1 диапазон: в работе"),
                ("section:second", "1 диапазон: в работе"),
            ],
        )
        self.assertTrue(all(entry.work_status is SelectionRangeStatus.ACTIVE for entry in entries))

    def test_same_page_only_anchors_share_status_without_invented_boundaries(self) -> None:
        document = SourceDocument(
            pages=(SourcePage(1, "1", ("Shared anchor", "Body")),),
            sections=(
                SourceSection("parent", "1", "Parent", ("1",), (1,)),
                SourceSection(
                    "child",
                    "1.1",
                    "Child",
                    ("1", "1.1"),
                    (1,),
                    parent_section_id="parent",
                ),
            ),
        )
        saved = SelectionRangeSummary(
            selection_id="SELECTION_SHARED",
            section_id="parent",
            selection=document.select(SourcePosition(1, 1), SourcePosition(1, 1)),
            status=SelectionRangeStatus.COMPLETED,
            status_text="готов",
        )

        entries = StructureScreen(document, (saved,)).entries()

        self.assertEqual(
            [entry.status for entry in entries],
            ["1 диапазон: готов", "1 диапазон: готов"],
        )

    def test_structure_uses_full_viewport_wraps_rows_and_keeps_scrollbar(self) -> None:
        screen = StructureScreen(
            self.document,
            (self._range(SelectionRangeStatus.ACTIVE),),
        )

        rendered = screen.render(width=32, height=12)
        text = fragment_list_to_text(rendered.fragments)

        self.assertIn("PMI Workbench / Структура", text)
        self.assertIn("[1 диапазон: в работе]", text)
        self.assertNotIn("Диапазон стр.", text)
        self.assertEqual(len(text.splitlines()), 12)
        self.assertTrue(any(line.endswith(("│", "█")) for line in text.splitlines()[5:-2]))
        self.assertEqual(len(screen.entries()), 1)
        self.assertNotIn("Сформировать весь ПМИ", text)
        self.assertIn("[E] экспорт", text)

    def test_structure_footer_stays_on_one_line_when_width_is_sufficient(self) -> None:
        screen = StructureScreen(self.document, ())

        text = fragment_list_to_text(screen.render(width=120, height=12).fragments)
        footer_lines = [
            line
            for line in text.splitlines()
            if "[↑/↓]" in line or "[E] экспорт" in line
        ]

        self.assertEqual(len(footer_lines), 1)
        self.assertIn("[↑/↓]", footer_lines[0])
        self.assertIn("[E] экспорт", footer_lines[0])
        self.assertIn("[Esc] выход", footer_lines[0])

    def test_inline_search_filters_sections_without_modal_dialog(self) -> None:
        screen = StructureScreen(self.document, ())
        screen.query = "не существует"

        entries = screen.entries()

        self.assertEqual(entries, ())
        rendered = fragment_list_to_text(screen.render(width=32, height=12).fragments)
        self.assertIn("Разделы не найдены", rendered)

    def test_enter_returns_current_full_screen_entry(self) -> None:
        with create_pipe_input() as pipe_input:
            pipe_input.send_text("\r")
            choice = StructureScreen(
                self.document,
                (),
                input=pipe_input,
                output=DummyOutput(),
            ).run()

        self.assertEqual(choice, "section:section-0270")

    def test_export_is_a_keyboard_shortcut_not_a_list_entry(self) -> None:
        with create_pipe_input() as pipe_input:
            pipe_input.send_text("e")
            choice = StructureScreen(
                self.document,
                (),
                input=pipe_input,
                output=DummyOutput(),
            ).run()

        self.assertEqual(choice, "export")
        self.assertNotIn(
            "export",
            [
                entry.key
                for entry in StructureScreen(self.document, ()).entries()
            ],
        )

    def test_structure_cursor_query_and_scroll_restore_from_shared_state(self) -> None:
        state = SourceNavigationState()
        first = StructureScreen(self.document, (), state=state)
        first.query = "4.16"
        first._cursor = 0
        first._offset = 1
        first.render(width=80, height=12)

        restored = StructureScreen(self.document, (), state=state)

        self.assertEqual(restored.query, "4.16")
        self.assertEqual(restored._cursor, 0)
        self.assertEqual(restored._offset, 0)


class FullScreenSelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.run_dir = Path(self.temp_dir.name)
        write_source(self.run_dir)
        self.document = load_source_document(self.run_dir)
        self.section = self.document.sections[0]

    def test_saved_range_colors_every_source_row_and_renders_status(self) -> None:
        positions = self.document.positions_for_pages(self.section.page_indexes)
        saved = SelectionRangeSummary(
            selection_id="SELECTION_0001",
            section_id=self.section.section_id,
            selection=self.document.select(positions[1], positions[2]),
            status=SelectionRangeStatus.ACTIVE,
            status_text="карточки в работе — 1",
        )
        screen = SelectionScreen(
            self.document,
            self.section,
            ranges=(saved,),
        )

        rendered = screen.render(width=36, height=14)
        text = fragment_list_to_text(rendered.fragments)
        rows, _ = screen._source_rows(width=36)
        line_2_style = next(style for style, line in rows if "002 |" in line)
        line_3_style = next(style for style, line in rows if "003 |" in line)

        self.assertIn("PMI Workbench / Структура /", text)
        self.assertIn("Страница 283", text)
        self.assertIn("[карточки в работе — 1]", text)
        self.assertEqual(line_2_style, "class:range.active")
        self.assertEqual(line_3_style, "class:range.active")
        self.assertEqual(len(text.splitlines()), 14)
        self.assertTrue(any(line.endswith(("│", "█")) for line in text.splitlines()[4:-2]))

    def test_enter_on_saved_range_returns_open_range_action(self) -> None:
        positions = self.document.positions_for_pages(self.section.page_indexes)
        saved = SelectionRangeSummary(
            selection_id="SELECTION_0001",
            section_id=self.section.section_id,
            selection=self.document.select(positions[1], positions[2]),
            status=SelectionRangeStatus.ACTIVE,
            status_text="карточки в работе — 1",
        )

        with create_pipe_input() as pipe_input:
            pipe_input.send_text("\x1b[B\r")
            action = select_text_range(
                self.document,
                self.section,
                ranges=(saved,),
                input=pipe_input,
                output=DummyOutput(),
            )

        self.assertEqual(action, OpenRange("SELECTION_0001"))

    def test_legacy_overlap_prefers_active_range_regardless_of_input_order(self) -> None:
        positions = self.document.positions_for_pages(self.section.page_indexes)
        terminal = SelectionRangeSummary(
            selection_id="SELECTION_TERMINAL",
            section_id=self.section.section_id,
            selection=self.document.select(positions[0], positions[1]),
            status=SelectionRangeStatus.TERMINAL,
            status_text="нет тестируемого поведения",
        )
        active = SelectionRangeSummary(
            selection_id="SELECTION_ACTIVE",
            section_id=self.section.section_id,
            selection=self.document.select(positions[0], positions[2]),
            status=SelectionRangeStatus.ACTIVE,
            status_text="карточки в работе — 1",
        )

        for ranges in ((terminal, active), (active, terminal)):
            with self.subTest(order=tuple(item.selection_id for item in ranges)):
                screen = SelectionScreen(self.document, self.section, ranges=ranges)
                rows, _ = screen._source_rows(width=80)

                self.assertEqual(
                    next(style for style, line in rows if "001 |" in line),
                    "class:range.active.cursor",
                )
                self.assertIn(
                    ("class:range.active", "    [карточки в работе — 1]"),
                    rows,
                )
                self.assertNotIn(
                    ("class:range.terminal", "    [нет тестируемого поведения]"),
                    rows,
                )

    def test_expanded_selection_can_supersede_contained_terminal_range(self) -> None:
        positions = self.document.positions_for_pages(self.section.page_indexes)
        terminal = SelectionRangeSummary(
            selection_id="SELECTION_TERMINAL",
            section_id=self.section.section_id,
            selection=self.document.select(positions[0], positions[1]),
            status=SelectionRangeStatus.TERMINAL,
            status_text="нет тестируемого поведения",
        )

        with create_pipe_input() as pipe_input:
            pipe_input.send_text(" \x1b[B\x1b[B \r\r")
            action = select_text_range(
                self.document,
                self.section,
                ranges=(terminal,),
                input=pipe_input,
                output=DummyOutput(),
            )

        self.assertEqual(
            action,
            CreateSelection(
                self.document.select(positions[0], positions[2]),
                supersede_selection_ids=("SELECTION_TERMINAL",),
            ),
        )

    def test_expanded_selection_can_supersede_contained_active_range(self) -> None:
        positions = self.document.positions_for_pages(self.section.page_indexes)
        active = SelectionRangeSummary(
            selection_id="SELECTION_ACTIVE",
            section_id=self.section.section_id,
            selection=self.document.select(positions[0], positions[1]),
            status=SelectionRangeStatus.ACTIVE,
            status_text="карточки в работе — 1",
        )

        with create_pipe_input() as pipe_input:
            pipe_input.send_text(" \x1b[B\x1b[B \r\r")
            action = select_text_range(
                self.document,
                self.section,
                ranges=(active,),
                input=pipe_input,
                output=DummyOutput(),
            )

        self.assertEqual(
            action,
            CreateSelection(
                self.document.select(positions[0], positions[2]),
                supersede_selection_ids=("SELECTION_ACTIVE",),
            ),
        )

    def test_partial_overlap_is_blocked_in_confirmation(self) -> None:
        positions = self.document.positions_for_pages(self.section.page_indexes)
        active = SelectionRangeSummary(
            selection_id="SELECTION_ACTIVE",
            section_id=self.section.section_id,
            selection=self.document.select(positions[0], positions[1]),
            status=SelectionRangeStatus.ACTIVE,
            status_text="карточки в работе — 1",
        )
        partial = self.document.select(positions[1], positions[2])

        screen = ConfirmationScreen(
            self.section,
            partial,
            source_name="specification.pdf",
            overlapping_ranges=(active,),
        )

        self.assertEqual(
            screen.actions,
            (("change", "Изменить диапазон"), ("cancel", "Отмена")),
        )

    def test_saved_range_colors_reflect_all_workspace_statuses(self) -> None:
        positions = self.document.positions_for_pages(self.section.page_indexes)
        ranges = tuple(
            SelectionRangeSummary(
                selection_id=f"SELECTION_{status.value}",
                section_id=self.section.section_id,
                selection=self.document.select(position, position),
                status=status,
                status_text=status.value,
            )
            for position, status in zip(
                positions[:3],
                (
                    SelectionRangeStatus.ACTIVE,
                    SelectionRangeStatus.COMPLETED,
                    SelectionRangeStatus.TERMINAL,
                ),
                strict=True,
            )
        )
        screen = SelectionScreen(self.document, self.section, ranges=ranges)

        rows, _ = screen._source_rows(width=80)
        styles = {
            number: next(style for style, line in rows if f"{number} |" in line)
            for number in ("001", "002", "003")
        }

        self.assertEqual(styles["001"], "class:range.active.cursor")
        self.assertEqual(styles["002"], "class:range.completed")
        self.assertEqual(styles["003"], "class:range.terminal")

    def test_selection_footer_stays_on_one_line_when_width_is_sufficient(self) -> None:
        screen = SelectionScreen(
            self.document,
            self.section,
            ranges=(),
        )

        text = fragment_list_to_text(screen.render(width=120, height=14).fragments)
        footer_lines = [
            line
            for line in text.splitlines()
            if "[Space] начать выбор" in line or "[Esc] назад" in line
        ]

        self.assertEqual(len(footer_lines), 1)
        self.assertIn("[Space] начать выбор", footer_lines[0])
        self.assertIn("[Esc] к структуре", footer_lines[0])

    def test_selection_and_confirmation_run_as_one_keyboard_flow(self) -> None:
        with create_pipe_input() as pipe_input:
            pipe_input.send_text(" \x1b[B \r\r")
            selected = select_text_range(
                self.document,
                self.section,
                input=pipe_input,
                output=DummyOutput(),
            )

        self.assertIsInstance(selected, CreateSelection)
        self.assertEqual(
            selected.selection.positions,
            (SourcePosition(1, 1), SourcePosition(1, 2)),
        )

    def test_confirmation_opens_at_start_and_page_scroll_is_not_overridden(self) -> None:
        selection = self.document.select(SourcePosition(1, 1), SourcePosition(2, 2))
        screen = ConfirmationScreen(
            self.section,
            selection,
            source_name="spec.pdf",
        )

        first = fragment_list_to_text(screen.render(width=32, height=12))
        screen.offset = 3
        second = fragment_list_to_text(screen.render(width=32, height=12))

        self.assertIn("Источник: spec.pdf", first)
        self.assertEqual(screen.offset, 3)
        self.assertNotEqual(first, second)

    def test_confirmation_blocks_over_budget_selection_without_truncating_it(self) -> None:
        selection = self.document.select(SourcePosition(1, 1), SourcePosition(2, 2))
        budget = DecompositionBudget(
            line_count=5,
            estimated_tokens=12_345,
            max_lines=4,
            max_estimated_tokens=12_000,
        )

        screen = ConfirmationScreen(
            self.section,
            selection,
            source_name="spec.pdf",
            decomposition_budget=budget,
        )
        text = fragment_list_to_text(screen.render(width=80, height=24))

        self.assertFalse(budget.within_single_call)
        self.assertEqual(
            screen.actions,
            (("change", "Изменить диапазон"), ("cancel", "Отмена")),
        )
        self.assertIn("превышает технический бюджет Prompt 1", text)
        self.assertIn("5 строк / 12 345", text)
        self.assertIn("4 строк / 12 000", text)
        self.assertNotIn("Построить каркасы карточек", text)
        self.assertIn("Изменить диапазон", text)
        self.assertEqual(selection.text, screen.selection.text)

    def test_confirmation_allows_windowed_selection_with_neutral_notice(
        self,
    ) -> None:
        selection = self.document.select(
            SourcePosition(1, 1),
            SourcePosition(2, 2),
        )
        budget = DecompositionBudget(
            line_count=5,
            estimated_tokens=12_345,
            max_lines=4,
            max_estimated_tokens=12_000,
        )
        decision = WindowingDecision(
            route=DecompositionRoute.WINDOWED,
            budget=budget,
            hard_max_lines=64,
            hard_max_estimated_tokens=96_000,
        )

        screen = ConfirmationScreen(
            self.section,
            selection,
            source_name="spec.pdf",
            decomposition_budget=budget,
            windowing_decision=decision,
        )
        text = fragment_list_to_text(screen.render(width=80, height=24))

        self.assertEqual(screen.actions, ConfirmationScreen.DEFAULT_ACTIONS)
        self.assertIn(
            "Большой диапазон: обработка может занять больше времени.",
            text,
        )
        self.assertIn("Построить каркасы карточек", text)
        self.assertNotIn("превышает технический бюджет Prompt 1", text)

    def test_confirmation_blocks_only_absolute_hard_limit(self) -> None:
        selection = self.document.select(
            SourcePosition(1, 1),
            SourcePosition(2, 2),
        )
        decision = WindowingDecision(
            route=DecompositionRoute.HARD_LIMIT,
            budget=DecompositionBudget(
                line_count=5,
                estimated_tokens=120_000,
                max_lines=4,
                max_estimated_tokens=12_000,
            ),
            hard_max_lines=4,
            hard_max_estimated_tokens=96_000,
        )

        screen = ConfirmationScreen(
            self.section,
            selection,
            source_name="spec.pdf",
            windowing_decision=decision,
        )
        text = fragment_list_to_text(screen.render(width=80, height=24))

        self.assertEqual(
            screen.actions,
            (("change", "Изменить диапазон"), ("cancel", "Отмена")),
        )
        self.assertIn("превышает абсолютный лимит обработки", text)
        self.assertIn("Hard limit: 4 строк / 96 000", text)
        self.assertNotIn("Построить каркасы карточек", text)

    def test_confirmation_uses_bounded_first_and_last_line_preview(self) -> None:
        document = SourceDocument(
            pages=(
                SourcePage(
                    1,
                    "1",
                    tuple(f"Source line {line}" for line in range(1, 11)),
                ),
                SourcePage(
                    2,
                    "2",
                    tuple(f"Source line {line}" for line in range(11, 21)),
                ),
            ),
            sections=(
                SourceSection("all", "1", "All", ("1",), (1, 2)),
            ),
        )
        selection = document.select(SourcePosition(1, 1), SourcePosition(2, 10))
        screen = ConfirmationScreen(
            document.sections[0],
            selection,
            source_name="spec.pdf",
        )

        text = fragment_list_to_text(screen.render(width=80, height=24))

        self.assertIn("Строк: 20", text)
        self.assertIn("Страниц: 2", text)
        self.assertIn("Первые строки:", text)
        self.assertIn("стр. 1:001 | Source line 1", text)
        self.assertIn("стр. 1:002 | Source line 2", text)
        self.assertIn("Последние строки:", text)
        self.assertIn("стр. 2:009 | Source line 19", text)
        self.assertIn("стр. 2:010 | Source line 20", text)
        self.assertNotIn("Source line 10", text)
        self.assertIn("Построить каркасы карточек", text)

    def test_source_name_comes_from_snapshot_metadata(self) -> None:
        self.assertEqual(self.document.metadata.original_name, "EMV spec 2.3.pdf")

    def test_outline_opens_one_global_canvas_at_its_anchor(self) -> None:
        document = SourceDocument(
            pages=(
                SourcePage(1, "1", ("First section", "Before")),
                SourcePage(2, "2", ("Second section", "After")),
                SourcePage(3, "3", ("Outside selected outline node",)),
            ),
            sections=(
                SourceSection(
                    "first",
                    "1",
                    "First",
                    ("1",),
                    (1,),
                    anchor_page_index=1,
                ),
                SourceSection(
                    "second",
                    "2",
                    "Second",
                    ("2",),
                    (2,),
                    anchor_page_index=2,
                ),
                SourceSection(
                    "third",
                    "3",
                    "Third",
                    ("3",),
                    (3,),
                    anchor_page_index=3,
                ),
            ),
        )

        screen = SelectionScreen(document, document.sections[1])

        self.assertEqual(screen.positions, document.positions)
        self.assertEqual(
            screen.positions[screen.cursor],
            SourcePosition(2, 1),
        )
        screen._move_cursor(2)
        self.assertEqual(
            screen.positions[screen.cursor],
            SourcePosition(3, 1),
        )

    def test_selected_outline_breaks_ties_between_page_only_anchors(self) -> None:
        document = SourceDocument(
            pages=(
                SourcePage(1, "1", ("Shared page start", "Shared page body")),
                SourcePage(2, "2", ("Next distinct section",)),
            ),
            sections=(
                SourceSection("first", "1.1", "First", ("1", "1.1"), (1,)),
                SourceSection("second", "1.2", "Second", ("1", "1.2"), (1,)),
                SourceSection("third", "1.3", "Third", ("1", "1.3"), (1,)),
                SourceSection("next", "2", "Next", ("2",), (2,)),
            ),
        )

        screen = SelectionScreen(document, document.sections[0])

        self.assertEqual(screen.current_outline.section_id, "first")
        screen._move_cursor(1)
        self.assertEqual(screen.current_outline.section_id, "first")
        screen._move_cursor(1)
        self.assertEqual(screen.current_outline.section_id, "next")

    def test_canvas_render_reads_only_a_bounded_window_of_large_document(self) -> None:
        class CountingDocument(SourceDocument):
            def __init__(self) -> None:
                self.line_reads = 0
                pages = tuple(
                    SourcePage(index, str(index), (f"Строка {index}",))
                    for index in range(1, 451)
                )
                sections = tuple(
                    SourceSection(
                        f"page-{index}",
                        str(index),
                        f"Раздел {index}",
                        (str(index),),
                        (index,),
                        anchor_page_index=index,
                    )
                    for index in range(1, 451)
                )
                super().__init__(pages=pages, sections=sections)

            def line(self, position: SourcePosition) -> str:
                self.line_reads += 1
                return super().line(position)

        document = CountingDocument()
        screen = SelectionScreen(document, document.sections[399])
        document.line_reads = 0

        rendered = screen.render(width=80, height=14)
        text = fragment_list_to_text(rendered.fragments)

        self.assertIn("Строка 400", text)
        self.assertLessEqual(document.line_reads, 20)
        self.assertLess(rendered.content_height, len(document.positions))

    def test_canvas_breadcrumb_tracks_cursor_and_search_is_literal_and_cyclic(self) -> None:
        document = SourceDocument(
            pages=(
                SourcePage(1, "1", ("Alpha [x+y]",)),
                SourcePage(2, "2", ("Middle",)),
                SourcePage(3, "3", ("alpha [X+Y]",)),
            ),
            sections=(
                SourceSection("first", "1", "First", ("1",), (1,)),
                SourceSection("second", "2", "Second", ("2",), (2,)),
                SourceSection("third", "3", "Third", ("3",), (3,)),
            ),
        )
        screen = SelectionScreen(document, document.sections[0])

        self.assertEqual(
            screen.find("[x+y]", direction=1),
            SourcePosition(3, 1),
        )
        self.assertEqual(screen.current_outline.section_id, "third")
        self.assertEqual(
            screen.find("[x+y]", direction=1),
            SourcePosition(1, 1),
        )
        text = fragment_list_to_text(screen.render(width=100, height=12).fragments)

        self.assertIn("PMI Workbench / Структура / 1 First / Исходный текст", text)
        self.assertTrue(screen.search_wrapped)

    def test_canvas_search_accepts_unicode_keyboard_input(self) -> None:
        document = SourceDocument(
            pages=(
                SourcePage(1, "1", ("Начало",)),
                SourcePage(2, "2", ("Команда ЁЖ",)),
                SourcePage(3, "3", ("команда ёж",)),
            ),
            sections=(
                SourceSection("first", "1", "First", ("1",), (1,)),
                SourceSection("second", "2", "Second", ("2",), (2,)),
                SourceSection("third", "3", "Third", ("3",), (3,)),
            ),
        )
        screen = SelectionScreen(document, document.sections[0])

        with create_pipe_input() as pipe_input:
            screen.input = pipe_input
            screen.output = DummyOutput()
            pipe_input.send_text("/ЁЖ\r\x1b\x1b")
            result = screen.run()

        self.assertIsNone(result)
        self.assertEqual(screen.query, "ЁЖ")
        self.assertEqual(screen.positions[screen.cursor], SourcePosition(2, 1))

    def test_canvas_resize_keeps_cursor_and_selection_coordinates(self) -> None:
        screen = SelectionScreen(self.document, self.section)
        screen._move_cursor(1)
        screen.selection_start = screen.positions[screen.cursor]
        screen._move_cursor(2)
        screen.selection = self.document.select(
            screen.selection_start,
            screen.positions[screen.cursor],
        )
        before = (
            screen.positions[screen.cursor],
            screen.selection,
        )

        screen.render(width=24, height=8)
        screen.render(width=100, height=20)

        self.assertEqual(
            (screen.positions[screen.cursor], screen.selection),
            before,
        )

    def test_active_selection_tracks_cursor_in_both_directions_until_second_space(self) -> None:
        screen = SelectionScreen(self.document, self.section)
        screen._move_cursor(3)
        screen._toggle_selection()
        screen._move_cursor(-2)

        self.assertFalse(screen.selection_complete)
        self.assertEqual(
            screen.selection.positions,
            (
                SourcePosition(1, 2),
                SourcePosition(1, 3),
                SourcePosition(2, 1),
            ),
        )

        screen._toggle_selection()

        self.assertTrue(screen.selection_complete)

    def test_navigation_state_restores_cursor_scroll_and_draft_in_memory(self) -> None:
        state = SourceNavigationState()
        first = SelectionScreen(self.document, self.section, state=state)
        first._move_cursor(1)
        first._toggle_selection()
        first._move_cursor(2)
        first.render(width=40, height=10)
        expected = (
            first.positions[first.cursor],
            first.selection_start,
            first.selection,
            first.offset,
        )

        restored = SelectionScreen(
            self.document,
            self.section,
            state=state,
            navigate_to_anchor=False,
        )

        self.assertEqual(
            (
                restored.positions[restored.cursor],
                restored.selection_start,
                restored.selection,
                restored.offset,
            ),
            expected,
        )
        self.assertFalse(restored.selection_complete)

    def test_first_escape_cancels_draft_without_leaving_canvas(self) -> None:
        screen = SelectionScreen(self.document, self.section)
        screen._toggle_selection()
        screen._move_cursor(1)

        self.assertTrue(screen.cancel_draft())
        self.assertIsNone(screen.selection_start)
        self.assertIsNone(screen.selection)
        self.assertFalse(screen.cancel_draft())


if __name__ == "__main__":
    unittest.main()
