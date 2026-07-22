from __future__ import annotations

import asyncio
import threading
import time
import unittest
from unittest.mock import Mock, patch

from prompt_toolkit.input.defaults import create_pipe_input
from prompt_toolkit.layout.containers import Window
from prompt_toolkit.output import DummyOutput

from pmi_generator.workbench.presentation.operation import (
    OperationCancelledByUser,
    TerminalOperationRunner,
)
from pmi_generator.workbench.presentation.result import ResultScreen
from pmi_generator.workbench.application.conversation import (
    ConversationAction,
    ConversationToolCall,
)
from pmi_generator.workbench.application.session import SessionEventKind
from pmi_generator.workbench.domain import (
    CardMutation,
    GapResolutionMode,
    RelatedGap,
    TestCard,
)
from pmi_generator.workbench.presentation.terminal import TerminalWorkbench
from pmi_generator.workbench.application.decomposition import (
    DecompositionProgress,
)


class SourceRangeRoutingTests(unittest.TestCase):
    def test_global_canvas_receives_ranges_from_all_outline_nodes(self) -> None:
        workbench = TerminalWorkbench.__new__(TerminalWorkbench)
        overlapping = Mock(
            section_id="section-parent",
            selection=Mock(positions=("278:008", "278:009")),
        )
        unrelated = Mock(
            section_id="section-other",
            selection=Mock(positions=("300:001",)),
        )
        workbench.facade = Mock()
        workbench.facade.selection_ranges.return_value = (
            unrelated,
            overlapping,
        )

        ranges = workbench._canvas_ranges()

        self.assertEqual(ranges, (unrelated, overlapping))

    def test_large_decomposition_context_shows_only_aggregate_progress(
        self,
    ) -> None:
        workbench = TerminalWorkbench.__new__(TerminalWorkbench)
        workbench.document = Mock(
            sections=(Mock(section_id="root", label="1 Большой раздел"),)
        )
        workbench.source_name = "spec.pdf"
        workbench.mode_label = None
        saved = Mock(
            section_id="root",
            selection=Mock(
                start=Mock(page_index=1, line_number=1),
                end=Mock(page_index=2, line_number=150),
                text="Первая строка\nПоследняя строка",
            ),
        )

        text = workbench._decomposition_operation_context(
            saved,
            width=100,
            height=24,
            progress=DecompositionProgress(
                route="windowed",
                completed_windows=2,
                total_windows=3,
                completed_conflicts=0,
                total_conflicts=0,
                stage="анализ диапазона",
            ),
        )

        self.assertIn("Обработано фрагментов: 2 из 3", text)
        self.assertIn("Стадия: анализ диапазона", text)
        self.assertIn("Источник: spec.pdf", text)
        self.assertIn("Диапазон: стр. 1:001 — стр. 2:150", text)
        self.assertNotIn("Первая строка", text)
        self.assertNotIn("Последняя строка", text)
        self.assertNotIn("WINDOW:", text)
        self.assertNotIn("CANDIDATE:", text)


class ContinuationRoutingTests(unittest.TestCase):
    @patch(
        "pmi_generator.workbench.presentation.terminal.SkeletonDetailScreen"
    )
    def test_selected_skeleton_reopens_existing_session(
        self,
        detail_screen_type: Mock,
    ) -> None:
        workbench = TerminalWorkbench.__new__(TerminalWorkbench)
        workbench.facade = Mock()
        workbench.facade.skeleton.return_value = Mock(
            payload={"decision": "selected", "card_id": "CARD_1"}
        )
        workbench.facade.ensure_card_session.return_value = ("SESSION_1", False)
        workbench._open_card_session = Mock()
        detail_screen_type.return_value.run.return_value = Mock(
            action="open_session"
        )
        selection = Mock(selection_id="SELECTION_1")

        decided = workbench._decide_skeleton(selection, "S1")

        self.assertFalse(decided)
        workbench.facade.ensure_card_session.assert_called_once_with(
            "SELECTION_1",
            "CARD_1",
        )
        workbench.facade.open_card_session.assert_not_called()
        workbench.facade.append.assert_not_called()
        workbench._open_card_session.assert_called_once_with(
            selection,
            "S1",
            "SESSION_1",
            "CARD_1",
            start_preparation=False,
        )

    @patch(
        "pmi_generator.workbench.presentation.terminal.SkeletonDetailScreen"
    )
    def test_selected_skeleton_recovers_missing_session_without_starting_prompt(
        self,
        detail_screen_type: Mock,
    ) -> None:
        workbench = TerminalWorkbench.__new__(TerminalWorkbench)
        workbench.facade = Mock()
        workbench.facade.skeleton.return_value = Mock(
            payload={"decision": "selected", "card_id": "CARD_1"}
        )
        workbench.facade.ensure_card_session.return_value = ("SESSION_1", True)
        workbench._open_card_session = Mock()
        detail_screen_type.return_value.run.return_value = Mock(
            action="open_session"
        )
        selection = Mock(selection_id="SELECTION_1")

        workbench._decide_skeleton(selection, "S1")

        workbench.facade.append.assert_called_once_with(
            "SESSION_1",
            SessionEventKind.WORKBENCH,
            (
                "Первоначальное заполнение карточки не завершено.\n"
                "/continue — повторить заполнение"
            ),
        )
        workbench._open_card_session.assert_called_once_with(
            selection,
            "S1",
            "SESSION_1",
            "CARD_1",
            start_preparation=False,
        )

    def test_new_card_enters_session_before_starting_prompt_2(self) -> None:
        workbench = TerminalWorkbench.__new__(TerminalWorkbench)
        workbench.facade = Mock()
        workbench.facade.open_card_session.return_value = "SESSION_1"
        workbench._open_card_session = Mock()
        selection = Mock(selection_id="SELECTION_1")

        workbench._prepare_card(selection, "S1", "CARD_1")

        workbench.facade.open_card_session.assert_called_once_with(
            "SELECTION_1",
            "CARD_1",
        )
        workbench._open_card_session.assert_called_once_with(
            selection,
            "S1",
            "SESSION_1",
            "CARD_1",
            start_preparation=True,
        )

    def test_new_card_session_starts_with_creation_event(self) -> None:
        workbench, card = self._workbench_with_card()
        workbench.facade.open_card_session.return_value = "SESSION_1"
        workbench.facade.history.return_value = []
        workbench._open_card_session = Mock()
        selection = Mock(selection_id="SELECTION_1")

        workbench._prepare_card(selection, "S1", card.card_id)

        workbench.facade.append.assert_called_once_with(
            "SESSION_1",
            SessionEventKind.WORKBENCH,
            "Создана рабочая карточка «Карточка».",
        )

    @patch(
        "pmi_generator.workbench.presentation.terminal.TerminalSessionShell"
    )
    def test_take_skeleton_opens_session_and_starts_population(
        self,
        shell_type: Mock,
    ) -> None:
        workbench = TerminalWorkbench.__new__(TerminalWorkbench)
        workbench.facade = Mock()
        workbench.facade.open_card_session.return_value = "SESSION_1"
        workbench._continue_card = Mock()
        selection = Mock(selection_id="SELECTION_1")

        workbench._prepare_card(selection, "S1", "CARD_1")

        shell_type.return_value.run.assert_called_once_with()
        startup_handler = shell_type.call_args.kwargs["startup_handler"]
        self.assertIsNotNone(startup_handler)
        startup_handler()
        workbench._continue_card.assert_called_once_with(
            selection,
            "S1",
            "SESSION_1",
            "CARD_1",
        )

    def test_continue_uses_persisted_population_route(self) -> None:
        workbench, card = self._workbench_with_card()
        workbench.facade.continuation.return_value = "population"
        workbench._populate_card = Mock(return_value=True)
        workbench._investigate_open_gaps = Mock(return_value=True)
        workbench._append_card_snapshot = Mock()
        selection = Mock()

        workbench._continue_card(selection, "S1", "SESSION_1", card.card_id)

        workbench._populate_card.assert_called_once_with(
            selection,
            "S1",
            "SESSION_1",
            card.card_id,
        )
        workbench._investigate_open_gaps.assert_called_once_with(
            selection,
            "SESSION_1",
            card.card_id,
        )
        workbench._append_card_snapshot.assert_called_once_with(
            "SESSION_1",
            card.card_id,
        )

    def test_continue_uses_persisted_gap_route(self) -> None:
        workbench, card = self._workbench_with_card()
        workbench.facade.continuation.return_value = "gap_investigation"
        card.apply(
            CardMutation(
                gaps=(
                    RelatedGap(
                        gap_id="GAP_1",
                        card_id=card.card_id,
                        question="Как наблюдать результат?",
                        blocking_reason="Неизвестен способ наблюдения",
                        allowed_paths=("test.observation.method",),
                        dependencies=(),
                        closure_criterion="Найден способ наблюдения",
                    ),
                ),
            )
        )
        workbench._populate_card = Mock()
        workbench._investigate_open_gaps = Mock(return_value=True)
        workbench._append_card_snapshot = Mock()
        selection = Mock()

        workbench._continue_card(selection, "S1", "SESSION_1", card.card_id)

        workbench._populate_card.assert_not_called()
        workbench._investigate_open_gaps.assert_called_once_with(
            selection,
            "SESSION_1",
            card.card_id,
        )
        workbench._append_card_snapshot.assert_called_once_with(
            "SESSION_1",
            card.card_id,
        )

    def test_continue_at_decision_stage_does_not_restart_prompt(self) -> None:
        workbench, card = self._workbench_with_card()
        workbench.facade.continuation.return_value = "card_decision"
        workbench._populate_card = Mock()
        workbench._investigate_open_gaps = Mock()
        selection = Mock()

        workbench._continue_card(selection, "S1", "SESSION_1", card.card_id)

        workbench._populate_card.assert_not_called()
        workbench._investigate_open_gaps.assert_not_called()
        workbench.facade.append.assert_called_once()

    def test_continue_repairs_uncovered_fields_before_gap_investigation(self) -> None:
        workbench, card = self._workbench_with_card()
        workbench.facade.continuation.return_value = "coverage_repair"
        workbench.facade.repair_card_coverage.return_value = Mock(
            open_gap_ids=("GAP_1", "GAP_2")
        )
        workbench._investigate_open_gaps = Mock(return_value=True)
        workbench._append_card_snapshot = Mock()
        selection = Mock()

        workbench._continue_card(selection, "S1", "SESSION_1", card.card_id)

        workbench.facade.repair_card_coverage.assert_called_once_with(
            "SESSION_1",
            card.card_id,
        )
        workbench._investigate_open_gaps.assert_called_once_with(
            selection,
            "SESSION_1",
            card.card_id,
        )
        workbench._append_card_snapshot.assert_called_once_with(
            "SESSION_1",
            card.card_id,
        )

    def test_gap_queue_stops_when_current_gap_needs_analyst(self) -> None:
        workbench, card = self._workbench_with_card()
        workbench.facade.open_gap_ids.return_value = ("GAP_1", "GAP_2")
        operation = Mock(awaitable=Mock(), cancel=Mock())
        workbench.facade.investigate_gap.return_value = operation
        workbench._wait = Mock(return_value=Mock(outcome="not_found"))
        workbench._session_operation_context = Mock(return_value="")
        selection = Mock()

        completed = workbench._investigate_open_gaps(
            selection,
            "SESSION_1",
            card.card_id,
        )

        self.assertFalse(completed)
        workbench.facade.investigate_gap.assert_called_once_with(
            selection,
            "SESSION_1",
            card.card_id,
            "GAP_1",
        )

    def test_design_decision_gap_does_not_start_lightrag(self) -> None:
        workbench, card = self._workbench_with_card()
        card.apply(
            CardMutation(
                gaps=(
                    RelatedGap(
                        gap_id="GAP_DESIGN",
                        card_id=card.card_id,
                        question="Как наблюдать изменение?",
                        blocking_reason="Нужен проект теста",
                        allowed_paths=("test.observation.method",),
                        dependencies=(),
                        closure_criterion="Аналитик выбрал способ",
                        resolution_mode=GapResolutionMode.DESIGN_DECISION,
                    ),
                )
            )
        )
        workbench.facade.card.return_value = card
        workbench.facade.open_gap_ids.return_value = ("GAP_DESIGN",)

        completed = workbench._investigate_open_gaps(
            Mock(),
            "SESSION_1",
            card.card_id,
        )

        self.assertFalse(completed)
        workbench.facade.investigate_gap.assert_not_called()
        event = workbench.facade.append.call_args
        self.assertEqual(event.args[3]["resolution_mode"], "design_decision")
        self.assertIn("проектное решение", event.args[2])

    @patch(
        "pmi_generator.workbench.presentation.terminal.TerminalSessionShell"
    )
    def test_analyst_message_is_routed_through_conversation_agent(
        self,
        shell_type: Mock,
    ) -> None:
        workbench, card = self._workbench_with_card()
        workbench._append_card_snapshot = Mock()
        operation = Mock(awaitable=Mock(), cancel=Mock())
        workbench.facade.conversation_turn.return_value = operation
        workbench._wait = Mock()
        workbench._session_operation_context = Mock(return_value="")
        selection = Mock()

        workbench._open_card_session(
            selection,
            "S1",
            "SESSION_1",
            card.card_id,
        )
        message_handler = shell_type.call_args.kwargs["message_handler"]
        message_handler("MSG_0001")

        workbench.facade.conversation_turn.assert_called_once_with(
            selection,
            "S1",
            "SESSION_1",
            card.card_id,
            "MSG_0001",
        )
        workbench._wait.assert_called_once()
        workbench.facade.refine_card.assert_not_called()
        workbench._append_card_snapshot.assert_not_called()

    @patch(
        "pmi_generator.workbench.presentation.terminal.TerminalSessionShell"
    )
    def test_continue_shortcut_uses_same_application_tool_catalog(
        self,
        shell_type: Mock,
    ) -> None:
        workbench, card = self._workbench_with_card()
        operation = Mock(awaitable=Mock(), cancel=Mock())
        workbench.facade.dispatch_conversation_tool.return_value = Mock(
            action=ConversationAction.RESUME,
            effect=Mock(value="expensive"),
            text="Продолжаю сохранённую стадию.",
            awaitable=operation.awaitable,
            cancel=operation.cancel,
        )
        workbench._wait = Mock()
        workbench._session_operation_context = Mock(return_value="")
        selection = Mock()

        workbench._open_card_session(
            selection,
            "S1",
            "SESSION_1",
            card.card_id,
        )
        handlers = shell_type.call_args.kwargs["command_handlers"]
        handlers["/continue"]()

        workbench.facade.dispatch_conversation_tool.assert_called_once_with(
            selection,
            "S1",
            "SESSION_1",
            card.card_id,
            "",
            ConversationToolCall(
                ConversationAction.RESUME,
                {"expected_revision": card.revision},
            ),
        )
        workbench._wait.assert_called_once()

    @patch(
        "pmi_generator.workbench.presentation.terminal.TerminalSessionShell"
    )
    def test_mode_switching_argument_commands_are_not_exposed(
        self,
        shell_type: Mock,
    ) -> None:
        workbench, card = self._workbench_with_card()
        selection = Mock()

        workbench._open_card_session(
            selection,
            "S1",
            "SESSION_1",
            card.card_id,
        )
        kwargs = shell_type.call_args.kwargs
        self.assertNotIn("/ask", kwargs["command_handlers"])
        self.assertNotIn("/ask", kwargs["command_descriptions"])

    def test_card_snapshot_is_appended_once_for_current_revision(self) -> None:
        workbench, card = self._workbench_with_card()
        workbench.facade.history.return_value = []
        workbench.facade.working_card_snapshot.return_value = "# Карточка\n"

        workbench._append_card_snapshot("SESSION_1", card.card_id)

        kind = workbench.facade.append.call_args.args[1]
        text = workbench.facade.append.call_args.args[2]
        metadata = workbench.facade.append.call_args.args[3]
        self.assertIs(kind, SessionEventKind.ASSISTANT)
        self.assertIn("Подготовка рабочей карточки завершена.", text)
        self.assertIn("# Карточка", text)
        self.assertEqual(metadata["revision"], card.revision)
        self.assertTrue(metadata["card_snapshot"])

        workbench.facade.append.reset_mock()
        workbench.facade.history.return_value = [
            Mock(metadata={"revision": card.revision, "card_snapshot": True})
        ]
        workbench._append_card_snapshot("SESSION_1", card.card_id)
        workbench.facade.append.assert_not_called()

    @patch(
        "pmi_generator.workbench.presentation.terminal.TerminalSessionShell"
    )
    def test_card_session_receives_documented_breadcrumb(
        self,
        shell_type: Mock,
    ) -> None:
        workbench, card = self._workbench_with_card()
        selection = Mock(section_id="section-1")

        workbench._open_card_session(
            selection,
            "S1",
            "SESSION_1",
            card.card_id,
        )

        self.assertEqual(
            shell_type.call_args.kwargs["breadcrumb"],
            "PMI Workbench / 4.16.5 / Карточки / Карточка / Сессия",
        )
        self.assertEqual(
            shell_type.call_args.kwargs["command_descriptions"]["/include"],
            "оставить пробелы и включить карточку неполной",
        )

    @staticmethod
    def _workbench_with_card() -> tuple[TerminalWorkbench, TestCard]:
        card = TestCard.create(
            card_id="CARD_1",
            selection_id="SELECTION_1",
            title="Карточка",
            section_number="4.16.5",
            changed_factors=("первый байт",),
            consequences=("SW 6987",),
        )
        workbench = TerminalWorkbench.__new__(TerminalWorkbench)
        workbench.facade = Mock()
        workbench.facade.card.return_value = card
        return workbench, card


class TerminalOperationRunnerTests(unittest.TestCase):
    def test_session_wait_uses_inline_operation_context(self) -> None:
        workbench = TerminalWorkbench.__new__(TerminalWorkbench)
        workbench._operation_runner = Mock()
        workbench._operation_runner.run.return_value = "готово"
        workbench._active_session_shell = Mock()
        awaitable = Mock()
        cancellation = Mock()
        context = Mock()

        result = workbench._wait(
            "Исследование",
            awaitable,
            cancel=cancellation,
            context=context,
        )

        self.assertEqual(result, "готово")
        workbench._operation_runner.run.assert_called_once_with(
            "Исследование",
            awaitable,
            cancellation,
            context,
            full_screen=False,
        )

    def test_escape_returns_control_and_invokes_cancellation(self) -> None:
        cancellation = Mock()

        async def slow_operation() -> str:
            await asyncio.to_thread(time.sleep, 1.0)
            return "late result"

        with create_pipe_input() as pipe_input:
            runner = TerminalOperationRunner(
                input=pipe_input,
                output=DummyOutput(),
            )
            timer = threading.Timer(0.05, lambda: pipe_input.send_text("\x1b"))
            timer.start()
            started = time.monotonic()
            try:
                with self.assertRaises(OperationCancelledByUser):
                    runner.run("Длительная операция", slow_operation(), cancellation)
            finally:
                timer.cancel()

        self.assertLess(time.monotonic() - started, 0.4)
        cancellation.assert_called_once_with()

    def test_completed_operation_returns_result(self) -> None:
        async def operation() -> str:
            await asyncio.sleep(0)
            return "готово"

        with create_pipe_input() as pipe_input:
            result = TerminalOperationRunner(
                input=pipe_input,
                output=DummyOutput(),
            ).run("Операция", operation())

        self.assertEqual(result, "готово")

    def test_non_interruptible_sync_operation_ignores_escape(self) -> None:
        worker_thread: list[int] = []

        def operation() -> str:
            worker_thread.append(threading.get_ident())
            time.sleep(0.08)
            return "готово"

        with create_pipe_input() as pipe_input:
            runner = TerminalOperationRunner(
                input=pipe_input,
                output=DummyOutput(),
            )
            timer = threading.Timer(0.02, lambda: pipe_input.send_text("\x1b"))
            timer.start()
            started = time.monotonic()
            try:
                result = runner.run_sync(
                    "Подготовка источника",
                    operation,
                    interruptible=False,
                )
            finally:
                timer.cancel()

        self.assertEqual(result, "готово")
        self.assertGreaterEqual(time.monotonic() - started, 0.07)
        self.assertNotEqual(worker_thread, [threading.get_ident()])

    def test_operation_refreshes_context_while_worker_is_running(self) -> None:
        async def operation() -> str:
            await asyncio.sleep(0.16)
            return "готово"

        context = Mock(return_value="LightRAG: предметный вопрос")
        with create_pipe_input() as pipe_input:
            result = TerminalOperationRunner(
                input=pipe_input,
                output=DummyOutput(),
            ).run("Операция", operation(), context=context)

        self.assertEqual(result, "готово")
        self.assertGreaterEqual(context.call_count, 2)
        width, height = context.call_args.args
        self.assertGreaterEqual(width, 24)
        self.assertGreaterEqual(height, 1)

    def test_operation_viewport_wraps_long_context_lines(self) -> None:
        async def operation() -> str:
            await asyncio.sleep(0)
            return "готово"

        with (
            create_pipe_input() as pipe_input,
            patch(
                "pmi_generator.workbench.presentation.operation.Window",
                wraps=Window,
            ) as window_type,
        ):
            TerminalOperationRunner(
                input=pipe_input,
                output=DummyOutput(),
            ).run(
                "Операция",
                operation(),
                context=lambda _width, _height: "Очень длинный вопрос LightRAG",
            )

        self.assertIs(window_type.call_args.kwargs["wrap_lines"], True)

    def test_full_screen_operation_receives_dynamic_context_viewport(self) -> None:
        async def operation() -> str:
            await asyncio.sleep(0)
            return "готово"

        context = Mock(return_value="PMI Workbench / Контекст")
        with create_pipe_input() as pipe_input:
            result = TerminalOperationRunner(
                input=pipe_input,
                output=DummyOutput(),
            ).run(
                "Проверка",
                operation(),
                context=context,
                full_screen=True,
            )

        self.assertEqual(result, "готово")
        context.assert_called()

    def test_rejected_cancellation_does_not_report_cancelled_operation(self) -> None:
        async def operation() -> str:
            await asyncio.sleep(0.08)
            return "применено"

        cancellation = Mock(side_effect=RuntimeError("apply already started"))
        with create_pipe_input() as pipe_input:
            runner = TerminalOperationRunner(
                input=pipe_input,
                output=DummyOutput(),
            )
            timer = threading.Timer(0.02, lambda: pipe_input.send_text("\x1b"))
            timer.start()
            try:
                result = runner.run("Применение результата", operation(), cancellation)
            finally:
                timer.cancel()

        self.assertEqual(result, "применено")
        self.assertGreaterEqual(cancellation.call_count, 1)


class ResultScreenTests(unittest.TestCase):
    def test_result_screen_uses_full_viewport_and_returns_selected_action(self) -> None:
        with create_pipe_input() as pipe_input:
            pipe_input.send_text("\x1b[B\r")
            result = ResultScreen(
                "PMI Workbench / Ошибка",
                "Причина: неверный результат",
                (("retry", "Повторить"), ("back", "Назад")),
                input=pipe_input,
                output=DummyOutput(),
            ).run()

        self.assertEqual(result, "back")


if __name__ == "__main__":
    unittest.main()
