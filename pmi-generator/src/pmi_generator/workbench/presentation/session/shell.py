from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from shutil import get_terminal_size
from typing import TextIO

from prompt_toolkit import PromptSession
from prompt_toolkit.shortcuts import CompleteStyle
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.input.base import Input
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.output.base import Output
from prompt_toolkit.styles import Style

from ...application.session import SessionEventKind, SessionShellController
from ...application.session.ports import SessionGateway
from .completion import SlashCommandCompleter
from .renderer import AppendOnlySessionRenderer, SessionRenderState


SESSION_FOOTER = "[Enter] отправить  [/] команды [Esc] к карточкам"


class TerminalSessionShell:
    """Append-only terminal history with a PromptSession line editor."""

    def __init__(
        self,
        service: SessionGateway,
        session_id: str,
        *,
        output: TextIO | None = None,
        diagnostics_exporter: Callable[[], Path] | None = None,
        command_handlers: dict[str, Callable[[], object]] | None = None,
        command_descriptions: dict[str, str] | None = None,
        message_handler: Callable[[str], object] | None = None,
        startup_handler: Callable[[], object] | None = None,
        breadcrumb: str = "PMI Workbench / Сессия подготовки карточки",
        mode_label: str | None = None,
        prompt_input: Input | None = None,
        prompt_output: Output | None = None,
    ) -> None:
        self.service = service
        self.session_id = session_id
        self.output = output or sys.stdout
        self.diagnostics_exporter = diagnostics_exporter
        self.command_handlers = dict(command_handlers or {})
        descriptions = {
            "/continue": "продолжить текущую стадию",
            "/include": "включить карточку в итоговый ПМИ",
            "/exclude": "исключить карточку",
            "/export-diagnostics": "обновить диагностический Markdown",
        }
        descriptions.update(command_descriptions or {})
        self.message_handler = message_handler
        self.startup_handler = startup_handler
        self.breadcrumb = breadcrumb
        self.mode_label = mode_label
        self.prompt_input = prompt_input
        self.prompt_output = prompt_output
        self.controller = SessionShellController(service, session_id)
        self.render_state = SessionRenderState()
        self.renderer = AppendOnlySessionRenderer(
            self.output,
            width=get_terminal_size((80, 24)).columns,
            state=self.render_state,
        )
        commands = dict.fromkeys(
            [
                "/continue",
                "/export-diagnostics",
                *self.command_handlers,
            ]
        )
        self.completer = SlashCommandCompleter(list(commands), descriptions)

    def run(self) -> None:
        self._render_header()
        self._render_history()
        if self.startup_handler is not None:
            self.startup_handler()
            self._render_history()
        prompt = self._create_prompt_session()
        while not self.controller.should_exit:
            text = prompt.prompt(
                FormattedText([("class:prompt", "\n> ")]),
            )
            if text.startswith("/"):
                self.handle_command(text)
            elif text:
                message_id = self.controller.submit(text)
                if message_id and self.message_handler:
                    self.message_handler(message_id)
            self._render_history()

    def _create_prompt_session(self) -> PromptSession[str]:
        return PromptSession(
            completer=self.completer,
            complete_while_typing=True,
            complete_style=CompleteStyle.COLUMN,
            reserve_space_for_menu=min(
                8,
                len(self.completer.commands) + 2,
            ),
            key_bindings=self._bindings(),
            bottom_toolbar=FormattedText([("class:footer", SESSION_FOOTER)]),
            style=_session_style(),
            mouse_support=False,
            input=self.prompt_input,
            output=self.prompt_output,
        )

    def handle_command(self, text: str) -> bool:
        handler = self.command_handlers.get(text)
        if handler is not None:
            handler()
            return True
        if text == "/continue":
            self.service.append(
                self.session_id,
                SessionEventKind.WORKBENCH,
                "Исследование продолжено без дополнительного сообщения.",
            )
            return True
        if text == "/export-diagnostics" and self.diagnostics_exporter:
            path = self.diagnostics_exporter()
            self.service.append(
                self.session_id,
                SessionEventKind.WORKBENCH,
                f"Диагностика сессии обновлена:\n{path}",
            )
            return True
        return False

    def _bindings(self) -> KeyBindings:
        bindings = KeyBindings()

        @bindings.add("escape")
        def escape(event: object) -> None:
            buffer = event.app.current_buffer  # type: ignore[attr-defined]
            self.controller.input_text = buffer.text
            action = self.controller.escape()
            if action == "input_cleared":
                buffer.reset()
            elif action == "exit":
                event.app.exit(result="")  # type: ignore[attr-defined]

        return bindings

    def _render_history(self) -> None:
        self.renderer.resize(get_terminal_size((80, 24)).columns)
        self.renderer.render(self.service.history(self.session_id))

    def _render_header(self) -> None:
        self.output.write("\x1b[2J\x1b[H")
        self.output.write(f"\x1b[1;38;5;39m{self.breadcrumb}\x1b[0m\n")
        if self.mode_label:
            self.output.write(f"\x1b[1;33m{self.mode_label}\x1b[0m\n")


def _session_style() -> Style:
    return Style.from_dict(
        {
            "prompt": "fg:ansicyan bold",
            "bottom-toolbar": "noreverse bg:default",
            "footer": "noreverse bg:default",
            "completion-menu.completion": "fg:ansiwhite bg:ansibrightblack",
            "completion-menu.completion.current": "fg:ansiblack bg:ansicyan",
            "completion-menu.meta.completion": "fg:ansibrightblack",
        }
    )
