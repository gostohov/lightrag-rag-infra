from __future__ import annotations

import asyncio
import threading
from collections.abc import Awaitable, Callable
from time import monotonic
from typing import Any

from prompt_toolkit.application import Application
from prompt_toolkit.application.current import get_app
from prompt_toolkit.formatted_text import AnyFormattedText, FormattedText
from prompt_toolkit.input.base import Input
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.key_binding.key_bindings import (
    KeyBindingsBase,
    merge_key_bindings,
)
from prompt_toolkit.layout import Layout
from prompt_toolkit.layout.containers import AnyContainer, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.output.base import Output
from prompt_toolkit.styles import BaseStyle

from .navigation import NAVIGATION_STYLE


class OperationCancelledByUser(RuntimeError):
    """Пользователь вернул управление, не дожидаясь сетевого ответа."""


class TerminalOperationRunner:
    """Выполняет coroutine, продолжая читать клавиатуру терминала."""

    _frames = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

    def __init__(self, *, input: Input | None = None, output: Output | None = None) -> None:
        self.input = input
        self.output = output

    def run(
        self,
        label: str,
        awaitable: Awaitable[Any],
        cancel: Callable[[], object] | None = None,
        context: Callable[[int, int], str | AnyFormattedText] | None = None,
        *,
        full_screen: bool = False,
        interruptible: bool = True,
        view: Callable[[Callable[[], FormattedText]], AnyContainer] | None = None,
        navigation_bindings: KeyBindingsBase | None = None,
        mouse_support: bool = False,
        style: BaseStyle | None = None,
    ) -> Any:
        started = monotonic()
        result: dict[str, Any] = {}
        finished = threading.Event()
        cancelled = False

        def status_content() -> FormattedText:
            elapsed = monotonic() - started
            frame = self._frames[int(elapsed * 10) % len(self._frames)]
            values: list[tuple[str, str]] = []
            values.extend(
                [
                    ("", "\n"),
                    ("class:warning", f"{frame} {label}\n"),
                    ("", f"  Выполняется: {_format_elapsed(elapsed)}\n"),
                    ("", f"  Стадия: {label.casefold()}\n\n"),
                    (
                        "class:footer",
                        (
                            "Ввод временно заблокирован.  [Esc] прервать"
                            if interruptible
                            else "Ввод временно заблокирован. Подождите завершения."
                        ),
                    ),
                ]
            )
            return FormattedText(values)

        def content() -> FormattedText:
            size = get_app().output.get_size()
            width = max(24, size.columns)
            height = max(8, size.rows)
            values: list[tuple[str, str]] = []
            if context is not None:
                rendered = context(width, max(1, height - 6))
                if isinstance(rendered, str):
                    text = rendered.rstrip()
                    if text:
                        values.append(("", f"{text}\n"))
                else:
                    fragments = list(rendered)
                    if fragments:
                        values.extend(fragments)
                        if not fragments[-1][1].endswith("\n"):
                            values.append(("", "\n"))
            values.extend(status_content())
            return FormattedText(values)

        bindings = KeyBindings()

        if interruptible:

            @bindings.add("escape", eager=True)
            def escape(event: object) -> None:
                nonlocal cancelled
                if finished.is_set():
                    event.app.exit()  # type: ignore[attr-defined]
                    return
                if cancel is not None:
                    try:
                        cancel()
                    except Exception:
                        self._retry_cancel(
                            cancel,
                            finished,
                            lambda: self._mark_cancelled(application, result),
                        )
                        return
                cancelled = True
                event.app.exit()  # type: ignore[attr-defined]

        else:

            @bindings.add("escape", eager=True)
            @bindings.add("c-c", eager=True)
            def ignore_interrupt(event: object) -> None:
                return

        if view is None:
            control = FormattedTextControl(content, focusable=True)
            container: AnyContainer = Window(control, wrap_lines=True)
        else:
            container = view(status_content)
        application_bindings: KeyBindingsBase = bindings
        if navigation_bindings is not None:
            application_bindings = merge_key_bindings(
                [bindings, navigation_bindings]
            )
        application: Application[None] = Application(
            layout=Layout(container),
            key_bindings=application_bindings,
            style=style or NAVIGATION_STYLE,
            full_screen=full_screen,
            erase_when_done=not full_screen,
            refresh_interval=0.1,
            mouse_support=mouse_support,
            input=self.input,
            output=self.output,
        )
        application.ttimeoutlen = 0.05

        def execute() -> None:
            try:
                result["value"] = asyncio.run(awaitable)
            except BaseException as error:
                result["error"] = error
            finally:
                finished.set()
                loop = application.loop
                if not cancelled and loop is not None and not loop.is_closed():
                    loop.call_soon_threadsafe(self._finish_application, application)

        def start() -> None:
            threading.Thread(target=execute, daemon=True).start()

        application.run(pre_run=start)
        if result.pop("cancelled", False):
            cancelled = True
        if cancelled:
            raise OperationCancelledByUser(f"Операция «{label}» отменена")
        if "error" in result:
            raise result["error"]
        return result.get("value")

    def run_sync(
        self,
        label: str,
        operation: Callable[[], Any],
        context: Callable[[int, int], str] | None = None,
        *,
        full_screen: bool = False,
        interruptible: bool = False,
    ) -> Any:
        async def invoke() -> Any:
            return operation()

        return self.run(
            label,
            invoke(),
            context=context,
            full_screen=full_screen,
            interruptible=interruptible,
        )

    @staticmethod
    def _finish_application(application: Application[None]) -> None:
        if not application.is_done:
            application.exit()

    @staticmethod
    def _retry_cancel(
        cancel: Callable[[], object],
        finished: threading.Event,
        on_success: Callable[[], None],
    ) -> None:
        def retry() -> None:
            while not finished.wait(0.01):
                try:
                    cancel()
                except Exception:
                    continue
                on_success()
                return

        threading.Thread(target=retry, daemon=True).start()

    @staticmethod
    def _mark_cancelled(
        application: Application[None],
        result: dict[str, Any],
    ) -> None:
        loop = application.loop
        if loop is None or loop.is_closed():
            return

        def finish() -> None:
            result["cancelled"] = True
            TerminalOperationRunner._finish_application(application)

        loop.call_soon_threadsafe(finish)


def _format_elapsed(seconds: float) -> str:
    total = max(0, int(seconds))
    minutes, secs = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"
