from __future__ import annotations

import io
import tempfile
import threading
import time
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch

from prompt_toolkit.input.defaults import create_pipe_input
from prompt_toolkit.application.current import set_app
from prompt_toolkit.output import DummyOutput
from prompt_toolkit.shortcuts import CompleteStyle

from pmi_generator.workbench.application.session import (
    SessionEventKind,
    SessionService,
    SessionShellController,
)
from pmi_generator.workbench.application.state import (
    AttemptRecord,
    AttemptStatus,
    StoredRecord,
)
from pmi_generator.workbench.infrastructure.storage import InMemoryDatabase, InMemoryUnitOfWork
from pmi_generator.workbench.presentation.session import (
    AppendOnlySessionRenderer,
    SessionRenderState,
    SlashCommandCompleter,
    TerminalSessionShell,
    export_session_diagnostics,
    render_session_context,
    render_session_context_fragments,
)


NOW = datetime(2026, 7, 16, 12, 0, tzinfo=UTC)


class TerminalSessionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.database = InMemoryDatabase()
        self.clock_value = NOW
        self.service = SessionService(
            uow_factory=lambda: InMemoryUnitOfWork(self.database),
            clock=lambda: self.clock_value,
        )
        self.service.open("SESSION_0001", "SELECTION_0001", "CARD_0001")

    def test_events_render_in_original_chronological_order(self) -> None:
        self.service.append("SESSION_0001", SessionEventKind.WORKBENCH, "Начало")
        self.clock_value += timedelta(seconds=1)
        self.service.append("SESSION_0001", SessionEventKind.ANALYST, "Уточнение")
        self.clock_value += timedelta(seconds=1)
        self.service.append("SESSION_0001", SessionEventKind.ASSISTANT, "Ответ")

        output = io.StringIO()
        AppendOnlySessionRenderer(output, width=50).render(
            self.service.history("SESSION_0001")
        )

        rendered = output.getvalue()
        self.assertLess(rendered.index("Начало"), rendered.index("Уточнение"))
        self.assertLess(rendered.index("Уточнение"), rendered.index("Ответ"))

    def test_analyst_message_has_role_time_padding_and_gray_background(self) -> None:
        self.service.append("SESSION_0001", SessionEventKind.ANALYST, "Точный текст")
        output = io.StringIO()

        AppendOnlySessionRenderer(output, width=50).render(
            self.service.history("SESSION_0001")
        )

        rendered = output.getvalue()
        self.assertIn(f"Аналитик · {NOW.astimezone().strftime('%H:%M')}", rendered)
        self.assertIn("  Точный текст", rendered)
        self.assertIn("\x1b[38;5;252;48;5;236m", rendered)

    def test_separator_tracks_width_and_operation_terminal_state_replaces_active(self) -> None:
        self.service.start_operation(
            "SESSION_0001",
            "ATTEMPT_0001",
            operation="Заполнение карточки",
            attempt_number=1,
        )
        self.service.complete_operation(
            "SESSION_0001",
            "ATTEMPT_0001",
            summary="Карточка заполнена",
        )
        output = io.StringIO()

        AppendOnlySessionRenderer(output, width=40).render(
            self.service.history("SESSION_0001")
        )

        rendered = output.getvalue()
        self.assertIn("─" * 39, rendered)
        self.assertNotIn("Статус: выполняется", rendered)
        self.assertIn("Статус: завершено", rendered)
        self.assertIn("\x1b[38;5;83m", rendered)

    def test_reopening_session_does_not_duplicate_rendered_history(self) -> None:
        self.service.append("SESSION_0001", SessionEventKind.WORKBENCH, "Одно событие")
        output = io.StringIO()
        state = SessionRenderState()
        renderer = AppendOnlySessionRenderer(output, width=40, state=state)

        renderer.render(self.service.history("SESSION_0001"))
        renderer.render(self.service.history("SESSION_0001"))

        self.assertEqual(output.getvalue().count("Одно событие"), 1)

    def test_card_snapshot_is_rendered_as_markdown_without_changing_history(
        self,
    ) -> None:
        markdown = (
            "Подготовка рабочей карточки завершена.\n\n"
            "# Проверка PUT DATA\n\n"
            "- Статус: **готова**\n"
            "- Команда: `80 DA`\n\n"
            "> Точная цитата."
        )
        self.service.append(
            "SESSION_0001",
            SessionEventKind.ASSISTANT,
            markdown,
            {"card_snapshot": True, "revision": 1},
        )
        output = io.StringIO()

        AppendOnlySessionRenderer(output, width=80).render(
            self.service.history("SESSION_0001")
        )

        rendered = output.getvalue()
        self.assertIn("\x1b[", rendered)
        self.assertIn("Проверка PUT DATA", rendered)
        self.assertIn("80 DA", rendered)
        self.assertNotIn("# Проверка PUT DATA", rendered)
        self.assertNotIn("**готова**", rendered)
        self.assertEqual(self.service.history("SESSION_0001")[0].text, markdown)

    def test_regular_assistant_message_is_not_interpreted_as_markdown(self) -> None:
        self.service.append(
            "SESSION_0001",
            SessionEventKind.ASSISTANT,
            "# Не заголовок",
        )
        output = io.StringIO()

        AppendOnlySessionRenderer(output, width=80).render(
            self.service.history("SESSION_0001")
        )

        self.assertIn("# Не заголовок", output.getvalue())

    def test_conversation_answer_is_rendered_as_markdown_without_changing_history(
        self,
    ) -> None:
        markdown = (
            "### 1. Анализ\n\n"
            "* **Открытый пробел:** `GAP_1`\n"
            "* **Вывод:** поиск не нашёл метод наблюдения."
        )
        self.service.append(
            "SESSION_0001",
            SessionEventKind.ASSISTANT,
            markdown,
            {"conversation_response": True, "turn_kind": "answer"},
        )
        output = io.StringIO()

        AppendOnlySessionRenderer(output, width=80).render(
            self.service.history("SESSION_0001")
        )

        rendered = output.getvalue()
        self.assertIn("\x1b[", rendered)
        self.assertIn("Открытый пробел", rendered)
        self.assertIn("GAP_1", rendered)
        self.assertNotIn("### 1. Анализ", rendered)
        self.assertNotIn("**Открытый пробел:**", rendered)
        self.assertEqual(self.service.history("SESSION_0001")[0].text, markdown)

    def test_card_markdown_wraps_to_current_terminal_width(self) -> None:
        from pmi_generator.workbench.presentation.session.markdown import (
            render_markdown_ansi,
        )

        markdown = "# Карточка\n\n" + "Очень длинное описание поведения карты. " * 8

        narrow = render_markdown_ansi(markdown, width=40)
        wide = render_markdown_ansi(markdown, width=100)

        self.assertGreater(narrow.count("\n"), wide.count("\n"))

    def test_lightrag_operation_keeps_plain_header_and_renders_markdown_body(
        self,
    ) -> None:
        markdown = (
            "### References\n\n"
            "- Найдено **требование** `4.16.5`.\n"
            "> Точная цитата."
        )
        raw = (
            "LightRAG: Как наблюдать PTH?\n"
            "Профиль: узкий поиск\n"
            "Статус: завершено\n"
            "Время: 12.3 с\n"
            "Точных фрагментов: 1\n"
            f"{markdown}"
        )
        self.service.append(
            "SESSION_0001",
            SessionEventKind.OPERATION,
            raw,
            {
                "lightrag_result": True,
                "markdown_body_start_line": 5,
            },
        )
        output = io.StringIO()

        AppendOnlySessionRenderer(output, width=80).render(
            self.service.history("SESSION_0001")
        )

        rendered = output.getvalue()
        self.assertIn("LightRAG: Как наблюдать PTH?", rendered)
        self.assertIn("Профиль: узкий поиск", rendered)
        self.assertIn("References", rendered)
        self.assertNotIn("### References", rendered)
        self.assertNotIn("**требование**", rendered)
        self.assertEqual(self.service.history("SESSION_0001")[0].text, raw)

    def test_lightrag_markdown_body_wraps_in_append_only_renderer(self) -> None:
        raw = (
            "LightRAG: вопрос\n"
            "Профиль: расширенный поиск\n"
            "Статус: завершено\n"
            "Время: 30.0 с\n"
            "Точных фрагментов: 0\n"
            "### Результат\n\n"
            + "Очень длинное описание найденного контекста. " * 8
        )
        self.service.append(
            "SESSION_0001",
            SessionEventKind.OPERATION,
            raw,
            {
                "lightrag_result": True,
                "markdown_body_start_line": 5,
            },
        )
        events = self.service.history("SESSION_0001")
        narrow = io.StringIO()
        wide = io.StringIO()

        AppendOnlySessionRenderer(narrow, width=40).render(events)
        AppendOnlySessionRenderer(wide, width=100).render(events)

        self.assertGreater(narrow.getvalue().count("\n"), wide.getvalue().count("\n"))
        self.assertNotIn("### Результат", narrow.getvalue())
        self.assertIn("Профиль: расширенный поиск", narrow.getvalue())

    @patch(
        "pmi_generator.workbench.presentation.session.renderer.render_markdown_ansi",
        side_effect=RuntimeError("renderer failed"),
    )
    def test_lightrag_markdown_falls_back_without_losing_body(
        self,
        render_markdown: Mock,
    ) -> None:
        raw = (
            "LightRAG: вопрос\n"
            "Профиль: узкий поиск\n"
            "Статус: завершено\n"
            "Время: 1.0 с\n"
            "Точных фрагментов: 0\n"
            "### Ответ\n\n- **Факт** не найден."
        )
        self.service.append(
            "SESSION_0001",
            SessionEventKind.OPERATION,
            raw,
            {
                "lightrag_result": True,
                "markdown_body_start_line": 5,
            },
        )
        output = io.StringIO()

        AppendOnlySessionRenderer(output, width=60).render(
            self.service.history("SESSION_0001")
        )

        render_markdown.assert_called_once()
        self.assertIn("### Ответ", output.getvalue())
        self.assertIn("**Факт**", output.getvalue())

    @patch(
        "pmi_generator.workbench.presentation.session.renderer.render_markdown_ansi",
        side_effect=RuntimeError("renderer failed"),
    )
    def test_card_markdown_falls_back_to_plain_text(
        self,
        render_markdown: Mock,
    ) -> None:
        self.service.append(
            "SESSION_0001",
            SessionEventKind.ASSISTANT,
            "# Карточка\n\n- **Статус:** готова",
            {"card_snapshot": True},
        )
        output = io.StringIO()

        AppendOnlySessionRenderer(output, width=50).render(
            self.service.history("SESSION_0001")
        )

        render_markdown.assert_called_once()
        self.assertIn("# Карточка", output.getvalue())
        self.assertIn("**Статус:**", output.getvalue())

    def test_session_renderer_tracks_current_terminal_width(self) -> None:
        self.service.append(
            "SESSION_0001",
            SessionEventKind.WORKBENCH,
            "Событие",
        )
        shell = TerminalSessionShell(
            self.service,
            "SESSION_0001",
            output=io.StringIO(),
            prompt_output=DummyOutput(),
        )

        with patch(
            "pmi_generator.workbench.presentation.session.shell.get_terminal_size",
            return_value=Mock(columns=112, lines=24),
        ):
            shell._render_history()

        self.assertEqual(shell.renderer.width, 112)
        self.assertIn("─" * 111, shell.output.getvalue())

    def test_live_context_keeps_completed_retrieval_and_current_operation(self) -> None:
        self.service.append(
            "SESSION_0001",
            SessionEventKind.OPERATION,
            (
                "LightRAG: Какое конкретное значение первого байта используется?\n"
                "Статус: завершено\n"
                "Ответ: конкретное значение в источнике не найдено."
            ),
            {"attempt_id": "ATTEMPT_1", "call_id": "CALL_1"},
        )
        current_question = (
            "Какое значение первого байта данных команды PUT DATA используется "
            "в примерах тестирования или спецификации для проверки обработки ошибки?"
        )
        self.service.append(
            "SESSION_0001",
            SessionEventKind.OPERATION,
            f"LightRAG: {current_question}\nСтатус: выполняется",
            {"attempt_id": "ATTEMPT_1", "call_id": "CALL_2"},
        )

        rendered = render_session_context(
            self.service.history("SESSION_0001"),
            width=52,
            height=18,
        )

        normalized = " ".join(line.strip() for line in rendered.splitlines())
        self.assertIn("Статус: завершено", rendered)
        self.assertIn("конкретное значение в источнике не найдено", normalized)
        self.assertIn("Статус: выполняется", rendered)
        self.assertIn(current_question, normalized)
        self.assertTrue(all(len(line) <= 52 for line in rendered.splitlines()))
        self.assertLessEqual(len(rendered.splitlines()), 18)

    def test_live_context_fragments_preserve_event_styles(self) -> None:
        self.service.append(
            "SESSION_0001",
            SessionEventKind.OPERATION,
            "LightRAG: вопрос\nСтатус: завершено\nТочных фрагментов: 0",
            {"attempt_id": "ATTEMPT_1", "call_id": "CALL_1"},
        )
        self.service.append(
            "SESSION_0001",
            SessionEventKind.WORKBENCH,
            "Диагностика сессии обновлена:\n/tmp/session.md",
        )
        self.service.append(
            "SESSION_0001",
            SessionEventKind.ANALYST,
            "Проверить альтернативную команду.",
        )
        self.service.append(
            "SESSION_0001",
            SessionEventKind.OPERATION,
            "Консультативный вопрос\nСтатус: выполняется",
            {"attempt_id": "ATTEMPT_2", "command": "/ask"},
        )

        fragments = render_session_context_fragments(
            self.service.history("SESSION_0001"),
            width=80,
            height=30,
        )

        self.assertIn(("class:success", "Операция\n"), fragments)
        self.assertIn(("class:breadcrumb", "Workbench\n"), fragments)
        self.assertIn(("class:user", "Аналитик\n"), fragments)
        self.assertIn(("class:warning", "Операция\n"), fragments)
        self.assertIn(("class:warning", "Консультативный вопрос\n"), fragments)

    def test_live_context_formats_card_and_lightrag_markdown(self) -> None:
        self.service.append(
            "SESSION_0001",
            SessionEventKind.ASSISTANT,
            "## Карточка\n\n- **Поле:** `значение`",
            {"card_snapshot": True},
        )
        self.service.append(
            "SESSION_0001",
            SessionEventKind.OPERATION,
            (
                "LightRAG: вопрос\n"
                "Профиль: узкий поиск\n"
                "Статус: завершено\n"
                "Время: 1.0 с\n"
                "Точных фрагментов: 1\n"
                "### Ответ\n\n- **Факт:** найден."
            ),
            {
                "lightrag_result": True,
                "markdown_body_start_line": 5,
            },
        )

        fragments = render_session_context_fragments(
            self.service.history("SESSION_0001"),
            width=80,
            height=30,
        )
        rendered = "".join(text for _, text in fragments)

        self.assertIn("Карточка", rendered)
        self.assertIn("Поле:", rendered)
        self.assertIn("Ответ", rendered)
        self.assertIn("Факт:", rendered)
        self.assertNotIn("## Карточка", rendered)
        self.assertNotIn("### Ответ", rendered)
        self.assertNotIn("**Поле:**", rendered)
        self.assertNotIn("**Факт:**", rendered)

    def test_shell_uses_inline_prompt_session_and_native_scrollback(self) -> None:
        shell = TerminalSessionShell(
            self.service,
            "SESSION_0001",
            output=io.StringIO(),
            prompt_output=DummyOutput(),
            breadcrumb="PMI Workbench / 4.16.5 / Карточка / Сессия",
        )

        prompt = shell._create_prompt_session()

        self.assertFalse(prompt.app.full_screen)
        with set_app(prompt.app):
            self.assertFalse(prompt.app.mouse_support())

    def test_prompt_session_owns_visible_slash_completion_menu(self) -> None:
        shell = TerminalSessionShell(
            self.service,
            "SESSION_0001",
            output=io.StringIO(),
            prompt_output=DummyOutput(),
        )

        prompt = shell._create_prompt_session()
        containers = {type(container).__name__ for container in prompt.app.layout.walk()}

        self.assertIs(prompt.completer, shell.completer)
        self.assertTrue(prompt.complete_while_typing)
        self.assertIs(prompt.complete_style, CompleteStyle.COLUMN)
        self.assertIn("CompletionsMenu", containers)
        self.assertEqual(
            prompt.reserve_space_for_menu,
            len(shell.completer.commands) + 2,
        )

    def test_shell_keeps_user_hint_in_prompt_session_toolbar(self) -> None:
        shell = TerminalSessionShell(
            self.service,
            "SESSION_0001",
            output=io.StringIO(),
            prompt_output=DummyOutput(),
        )
        prompt = shell._create_prompt_session()

        self.assertEqual(
            list(prompt.bottom_toolbar),
            [
                (
                    "class:footer",
                    "[Enter] отправить  [/] команды [Esc] к карточкам",
                )
            ],
        )

    def test_shell_deduplicates_builtin_and_handler_commands(self) -> None:
        shell = TerminalSessionShell(
            self.service,
            "SESSION_0001",
            output=io.StringIO(),
            command_handlers={"/continue": lambda: None},
        )

        self.assertEqual(shell.completer.commands.count("/continue"), 1)

    def test_real_prompt_toolkit_renders_input_and_handles_escape(self) -> None:
        with create_pipe_input() as pipe_input:
            pipe_input.send_bytes(b"\x1b")
            shell = TerminalSessionShell(
                self.service,
                "SESSION_0001",
                output=io.StringIO(),
                prompt_input=pipe_input,
                prompt_output=DummyOutput(),
            )

            shell.run()

        self.assertTrue(shell.controller.should_exit)

    def test_prompt_session_submits_unicode_and_preserves_chronology(self) -> None:
        with create_pipe_input() as pipe_input:
            def drive_shell() -> None:
                time.sleep(0.05)
                pipe_input.send_text("уточнение аналитика\r")
                time.sleep(0.05)
                pipe_input.send_bytes(b"\x1b")

            driver = threading.Thread(target=drive_shell)
            driver.start()
            shell = TerminalSessionShell(
                self.service,
                "SESSION_0001",
                output=io.StringIO(),
                prompt_input=pipe_input,
                prompt_output=DummyOutput(),
            )

            shell.run()
            driver.join()

        analyst_events = [
            event
            for event in self.service.history("SESSION_0001")
            if event.kind is SessionEventKind.ANALYST
        ]
        self.assertEqual(analyst_events[-1].text, "уточнение аналитика")

    def test_first_escape_cancels_active_operation_and_stays_in_session(self) -> None:
        self.service.start_operation(
            "SESSION_0001",
            "ATTEMPT_0001",
            operation="Заполнение карточки",
            attempt_number=1,
        )
        controller = SessionShellController(self.service, "SESSION_0001")

        action = controller.escape()

        self.assertEqual(action, "operation_cancelled")
        self.assertFalse(controller.should_exit)

    def test_late_result_is_hidden_from_history_and_saved_to_diagnostics(self) -> None:
        self.service.start_operation(
            "SESSION_0001", "ATTEMPT_0001", operation="Fake", attempt_number=1
        )
        self.service.cancel_operation("SESSION_0001", "ATTEMPT_0001")

        accepted = self.service.complete_operation(
            "SESSION_0001", "ATTEMPT_0001", summary="Поздний ответ"
        )

        self.assertFalse(accepted)
        self.assertNotIn("Поздний ответ", str(self.service.history("SESSION_0001")))
        diagnostic = self.database.records[("session_diagnostic", "ATTEMPT_0001")]
        self.assertEqual(diagnostic.payload["status"], "поздний результат отброшен")

    def test_escape_clears_input_before_exiting(self) -> None:
        controller = SessionShellController(self.service, "SESSION_0001")
        controller.input_text = "неотправленное сообщение"

        self.assertEqual(controller.escape(), "input_cleared")
        self.assertFalse(controller.should_exit)
        self.assertEqual(controller.escape(), "exit")
        self.assertTrue(controller.should_exit)

    def test_autocomplete_only_opens_for_slash_at_first_character(self) -> None:
        completer = SlashCommandCompleter(
            ["/continue", "/export-diagnostics", "/include"]
        )

        self.assertEqual(completer.suggestions("/ex"), ("/export-diagnostics",))
        self.assertEqual(completer.suggestions("текст /ex"), ())
        self.assertEqual(completer.suggestions(" /ex"), ())
        self.assertEqual(completer.suggestions("текст "), ())

    def test_autocomplete_explains_structural_decision(self) -> None:
        completer = SlashCommandCompleter(
            ["/include"],
            {"/include": "включить карточку в итоговый ПМИ"},
        )

        completion = next(
            iter(completer.get_completions(Mock(text_before_cursor="/i"), Mock()))
        )

        self.assertEqual(completion.display_meta_text, "включить карточку в итоговый ПМИ")

    def test_diagnostic_markdown_contains_history_attempts_and_errors_without_secrets(self) -> None:
        secret = "secret-api-key"
        self.service.append("SESSION_0001", SessionEventKind.ANALYST, "Проверить сценарий")
        self.service.start_operation(
            "SESSION_0001", "ATTEMPT_0001", operation="Fake", attempt_number=1
        )
        self.service.fail_operation(
            "SESSION_0001",
            "ATTEMPT_0001",
            error="HTTP 500",
            technical={"api_key": secret, "model": "fake"},
        )
        self.service.append(
            "SESSION_0001",
            SessionEventKind.ASSISTANT,
            "## Ответ\n\nТекст без преобразования.",
            {
                "conversation_response": True,
                "turn_kind": "answer",
                "message_id": "MSG_0002",
            },
        )
        self.service.append(
            "SESSION_0001",
            SessionEventKind.WORKBENCH,
            "Дорабатываю карточку.",
            {
                "conversation_action": "refine_card",
                "conversation_arguments": {"expected_revision": 3},
                "message_id": "MSG_0003",
            },
        )
        with InMemoryUnitOfWork(self.database) as uow:
            uow.attempts.save(
                AttemptRecord(
                    "ATTEMPT_CHILD",
                    "SESSION_0001:ATTEMPT_0001",
                    "prompt 3",
                    AttemptStatus.COMPLETED,
                    {},
                    NOW,
                )
            )
            uow.records.save(
                StoredRecord(
                    "llm_diagnostic",
                    "ATTEMPT_CHILD",
                    {"tool_calls": [{"name": "ask_lightrag"}]},
                )
            )
            uow.records.save(
                StoredRecord(
                    "retrieval_observation",
                    "ATTEMPT_0001:CALL_1",
                    {
                        "question": "Как наблюдать результат?",
                        "profile": "узкий поиск",
                        "answer": "Проверить SW1SW2.",
                        "evidence_ids": ["EVIDENCE_1"],
                    },
                )
            )
        with tempfile.TemporaryDirectory() as tmp:
            path = export_session_diagnostics(
                Path(tmp),
                "SESSION_0001",
                "CARD_0001",
                lambda: InMemoryUnitOfWork(self.database),
            )
            rendered = path.read_text(encoding="utf-8")

        self.assertIn("Проверить сценарий", rendered)
        self.assertIn("ATTEMPT_0001", rendered)
        self.assertIn("HTTP 500", rendered)
        self.assertIn("ATTEMPT_CHILD", rendered)
        self.assertIn("ask_lightrag", rendered)
        self.assertIn("Как наблюдать результат?", rendered)
        self.assertIn("Проверить SW1SW2.", rendered)
        self.assertIn("## Ответ\n\nТекст без преобразования.", rendered)
        self.assertIn('"conversation_response": true', rendered)
        self.assertIn('"turn_kind": "answer"', rendered)
        self.assertIn('"conversation_action": "refine_card"', rendered)
        self.assertIn('"expected_revision": 3', rendered)
        self.assertNotIn(secret, rendered)

    def test_conversation_action_ids_are_hidden_only_at_render_boundary(self) -> None:
        event = self.service.append(
            "SESSION_0001",
            SessionEventKind.ASSISTANT,
            "Доступны include_card, exclude_card и refine_card.",
            {
                "conversation_response": True,
                "turn_kind": "answer",
            },
        )
        history = self.service.history("SESSION_0001")
        output = io.StringIO()
        AppendOnlySessionRenderer(output, width=180).render(history)
        rendered = output.getvalue()

        self.assertNotIn("include_card", rendered)
        self.assertNotIn("exclude_card", rendered)
        self.assertNotIn("refine_card", rendered)
        self.assertIn("включить карточку в итоговый ПМИ", rendered)
        self.assertIn("исключить карточку из итогового ПМИ", rendered)
        self.assertIn("продолжить доработку карточки", rendered)
        self.assertEqual(
            history[-1].text,
            "Доступны include_card, exclude_card и refine_card.",
        )
        self.assertEqual(event, history[-1].sequence)


    def test_resize_and_common_escape_sequences_do_not_exit(self) -> None:
        controller = SessionShellController(self.service, "SESSION_0001")

        controller.resize(132, 40)
        controller.handle_sequence("\x1b[1;5D")
        controller.handle_sequence("\x1b[1;5C")
        controller.handle_sequence("ctrl-plus")

        self.assertEqual(controller.size, (132, 40))
        self.assertFalse(controller.should_exit)


if __name__ == "__main__":
    unittest.main()
