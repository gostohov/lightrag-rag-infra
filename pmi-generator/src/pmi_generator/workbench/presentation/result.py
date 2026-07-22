from __future__ import annotations

import textwrap

from prompt_toolkit.application import Application, get_app
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.input.base import Input
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.containers import Window
from prompt_toolkit.output.base import Output

from .navigation import (
    NAVIGATION_STYLE,
    navigation_footer,
    scrollbar,
    wrap_styled_lines,
)


class ResultScreen:
    def __init__(
        self,
        breadcrumb: str,
        body: str,
        actions: tuple[tuple[str, str], ...],
        *,
        kind: str = "normal",
        mode_label: str | None = None,
        input: Input | None = None,
        output: Output | None = None,
    ) -> None:
        if not actions:
            raise ValueError("Экран результата требует хотя бы одно действие")
        self.breadcrumb = breadcrumb
        self.body = body
        self.actions = actions
        self.kind = kind
        self.mode_label = mode_label
        self.input = input
        self.output = output
        self.cursor = 0
        self.offset = 0
        self.follow_cursor = False

    def run(self) -> str:
        application: Application[str] = Application(
            layout=Layout(
                Window(
                    FormattedTextControl(text=self._formatted_text, focusable=True),
                    wrap_lines=False,
                )
            ),
            key_bindings=self._bindings(),
            style=NAVIGATION_STYLE,
            full_screen=True,
            input=self.input,
            output=self.output,
        )
        return application.run()

    def render(self, *, width: int, height: int) -> FormattedText:
        width = max(24, width)
        height = max(8, height)
        body_style = {
            "error": "class:error",
            "warning": "class:warning",
            "success": "class:success",
        }.get(self.kind, "")
        header_source = [("class:breadcrumb", self.breadcrumb)]
        if self.mode_label:
            header_source.append(("class:warning", self.mode_label))
        header_source.append(("", ""))
        header = wrap_styled_lines(
            header_source,
            width=width,
        )
        rows: list[tuple[str, str]] = []
        for source_line in self.body.splitlines() or [""]:
            if not source_line:
                rows.append(("", ""))
                continue
            indent = len(source_line) - len(source_line.lstrip(" "))
            prefix = source_line[:indent]
            wrapped = textwrap.wrap(
                source_line[indent:],
                width=max(1, width - 1 - indent),
                replace_whitespace=False,
                break_long_words=True,
                break_on_hyphens=False,
            ) or [""]
            rows.append((body_style, prefix + wrapped[0]))
            rows.extend((body_style, prefix + line) for line in wrapped[1:])
        rows.extend([("", ""), ("class:breadcrumb", "Действия:")])
        spans: list[tuple[int, int]] = []
        for index, (_, label) in enumerate(self.actions):
            start = len(rows)
            marker = ">" if index == self.cursor else " "
            wrapped = textwrap.wrap(label, width=max(1, width - 3)) or [""]
            style = "class:selected" if index == self.cursor else ""
            rows.append((style, f"{marker} {wrapped[0]}"))
            rows.extend((style, f"  {line}") for line in wrapped[1:])
            spans.append((start, len(rows) - 1))

        footer = navigation_footer(
            "[↑/↓] выбор  [PgUp/PgDn] прокрутка  [Enter] выполнить  [Esc] назад",
            width=width,
        )
        viewport_height = max(1, height - len(header) - len(footer))
        selected_start, selected_end = spans[self.cursor]
        max_offset = max(0, len(rows) - viewport_height)
        if self.follow_cursor:
            if selected_start < self.offset:
                self.offset = selected_start
            elif selected_end >= self.offset + viewport_height:
                self.offset = selected_end - viewport_height + 1
        self.offset = min(max(0, self.offset), max_offset)
        visible = rows[self.offset : self.offset + viewport_height]
        visible.extend([("", "")] * (viewport_height - len(visible)))
        thumb_start, thumb_size = scrollbar(
            total=len(rows),
            visible=viewport_height,
            offset=self.offset,
        )

        fragments: list[tuple[str, str]] = []
        for style, line in header:
            fragments.extend([(style, line[:width]), ("", "\n")])
        for index, (style, line) in enumerate(visible):
            active = thumb_start <= index < thumb_start + thumb_size
            fragments.extend(
                [
                    (style, line[: width - 1].ljust(width - 1)),
                    (
                        "class:scrollbar.thumb"
                        if active
                        else "class:scrollbar.track",
                        "█" if active else "│",
                    ),
                    ("", "\n"),
                ]
            )
        for index, (style, line) in enumerate(footer):
            fragments.append((style, line[:width]))
            if index < len(footer) - 1:
                fragments.append(("", "\n"))
        return FormattedText(fragments)

    def _formatted_text(self) -> FormattedText:
        size = get_app().output.get_size()
        return self.render(width=size.columns, height=size.rows)

    def _bindings(self) -> KeyBindings:
        bindings = KeyBindings()

        def move(delta: int) -> None:
            self.cursor = min(max(0, self.cursor + delta), len(self.actions) - 1)
            self.follow_cursor = True

        @bindings.add("up")
        @bindings.add("<scroll-up>")
        def up(event: object) -> None:
            del event
            move(-1)

        @bindings.add("down")
        @bindings.add("<scroll-down>")
        def down(event: object) -> None:
            del event
            move(1)

        @bindings.add("pageup")
        def page_up(event: object) -> None:
            del event
            self.follow_cursor = False
            self.offset = max(0, self.offset - 10)

        @bindings.add("pagedown")
        def page_down(event: object) -> None:
            del event
            self.follow_cursor = False
            self.offset += 10

        @bindings.add("enter")
        def enter(event: object) -> None:
            event.app.exit(result=self.actions[self.cursor][0])  # type: ignore[attr-defined]

        @bindings.add("escape")
        def escape(event: object) -> None:
            event.app.exit(result=self.actions[-1][0])  # type: ignore[attr-defined]

        return bindings
