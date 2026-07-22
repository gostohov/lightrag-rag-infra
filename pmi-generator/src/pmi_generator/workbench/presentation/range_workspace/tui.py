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

from ...application.range_workspace import RangeWorkspaceController
from ...application.source import SavedSelection
from ..navigation import (
    NAVIGATION_STYLE,
    navigation_footer,
    scrollbar,
    wrap_styled_lines,
)


class RangeWorkspaceScreen:
    def __init__(
        self,
        controller: RangeWorkspaceController,
        selection: SavedSelection,
        *,
        section_number: str,
        mode_label: str | None = None,
        input: Input | None = None,
        output: Output | None = None,
    ) -> None:
        self.controller = controller
        self.selection = selection
        self.section_number = section_number
        self.mode_label = mode_label
        self.input = input
        self.output = output
        self.row_offset = 0

    def run(self) -> tuple[str, str] | None:
        application: Application[tuple[str, str] | None] = Application(
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
        state = self.controller.state
        self.controller.cursor = min(
            self.controller.cursor,
            max(0, self.controller.rows_count - 1),
        )
        start = self.selection.selection.start
        end = self.selection.selection.end
        review = (
            "актуальна"
            if state.review_current
            else "устарела"
            if state.review_stale
            else "не выполнена"
        )
        header_source = [
                (
                    "class:breadcrumb",
                    (
                        f"PMI Workbench / {self.section_number} / "
                        f"стр. {start.page_index}:{start.line_number:03d}–"
                        f"{end.page_index}:{end.line_number:03d} / Карточки"
                    ),
                ),
        ]
        if self.mode_label:
            header_source.append(("class:warning", self.mode_label))
        header_source.extend(
            [
                ("", ""),
                ("", f"Каркасов предложено: {len(state.items)}"),
                ("class:success", f"Включено карточек: {state.included}"),
                (
                    "class:warning",
                    f"Включено неполными: {state.included_incomplete}",
                ),
                ("class:excluded", f"Исключено: {state.excluded}"),
                ("", f"Проверка диапазона: {review}"),
                ("", ""),
            ]
        )
        header = wrap_styled_lines(
            header_source,
            width=width,
        )
        footer = navigation_footer(
            (
                "[↑/↓] карточка или действие  [Enter] открыть  "
                "[PgUp/PgDn] страница  [Esc] к структуре"
            ),
            width=width,
            compact_text=(
                "[↑/↓ PgUp/PgDn] навигация  [Enter] открыть  [Esc] назад"
            ),
        )
        viewport_height = max(1, height - len(header) - len(footer))
        rows, spans = self._rows(width=width)
        selected_start, selected_end = spans[self.controller.cursor]
        max_offset = max(0, len(rows) - viewport_height)
        if selected_start < self.row_offset:
            self.row_offset = selected_start
        elif selected_end >= self.row_offset + viewport_height:
            self.row_offset = selected_end - viewport_height + 1
        self.row_offset = min(max(0, self.row_offset), max_offset)
        visible = rows[self.row_offset : self.row_offset + viewport_height]
        visible.extend([("", "")] * (viewport_height - len(visible)))
        thumb_start, thumb_size = scrollbar(
            total=len(rows),
            visible=viewport_height,
            offset=self.row_offset,
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

    def _rows(
        self,
        *,
        width: int,
    ) -> tuple[list[tuple[str, str]], list[tuple[int, int]]]:
        state = self.controller.state
        rows: list[tuple[str, str]] = []
        spans: list[tuple[int, int]] = []
        style_names = {
            "success": "class:success",
            "warning": "class:warning",
            "muted": "class:excluded",
            "error": "class:error",
            "ready": "class:warning",
        }
        for index, item in enumerate(state.items):
            start = len(rows)
            marker = ">" if index == self.controller.cursor else " "
            prefix = f"{marker} "
            label = f"{item.title}  [{item.status}]"
            wrapped = textwrap.wrap(
                label,
                width=max(1, width - 1 - len(prefix)),
                replace_whitespace=False,
                break_long_words=True,
                break_on_hyphens=False,
            ) or [""]
            style = (
                "class:selected"
                if index == self.controller.cursor
                else style_names.get(item.style, "")
            )
            rows.append((style, prefix + wrapped[0]))
            rows.extend((style, "  " + line) for line in wrapped[1:])
            spans.append((start, len(rows) - 1))

        action: str | None = None
        if state.can_review:
            action = "Проверить выбранный диапазон"
        elif not state.items and state.terminal_status is None:
            action = "Построить каркасы карточек"
        if action is not None:
            if rows:
                rows.append(("", ""))
            rows.append(("class:breadcrumb", "Действия:"))
            start = len(rows)
            index = len(state.items)
            marker = ">" if index == self.controller.cursor else " "
            wrapped = textwrap.wrap(
                action,
                width=max(1, width - 3),
                replace_whitespace=False,
                break_long_words=True,
                break_on_hyphens=False,
            ) or [""]
            style = "class:selected" if index == self.controller.cursor else ""
            rows.append((style, f"{marker} {wrapped[0]}"))
            rows.extend((style, f"  {line}") for line in wrapped[1:])
            spans.append((start, len(rows) - 1))

        if not rows:
            rows.append(("class:warning", "Для диапазона пока нет доступных действий."))
            spans.append((0, 0))
        return rows, spans

    def _formatted_text(self) -> FormattedText:
        size = get_app().output.get_size()
        return self.render(width=size.columns, height=size.rows)

    def _bindings(self) -> KeyBindings:
        bindings = KeyBindings()

        @bindings.add("up")
        @bindings.add("<scroll-up>")
        def up(event: object) -> None:
            del event
            self.controller.move(-1)

        @bindings.add("down")
        @bindings.add("<scroll-down>")
        def down(event: object) -> None:
            del event
            self.controller.move(1)

        @bindings.add("pageup")
        def page_up(event: object) -> None:
            size = event.app.output.get_size()  # type: ignore[attr-defined]
            self.controller.viewport_height = max(1, size.rows // 3)
            self.controller.page(-1)

        @bindings.add("pagedown")
        def page_down(event: object) -> None:
            size = event.app.output.get_size()  # type: ignore[attr-defined]
            self.controller.viewport_height = max(1, size.rows // 3)
            self.controller.page(1)

        @bindings.add("enter")
        def enter(event: object) -> None:
            try:
                result = self.controller.activate()
            except ValueError:
                return
            event.app.exit(result=result)  # type: ignore[attr-defined]

        @bindings.add("escape")
        def escape(event: object) -> None:
            event.app.exit(result=None)  # type: ignore[attr-defined]

        return bindings
