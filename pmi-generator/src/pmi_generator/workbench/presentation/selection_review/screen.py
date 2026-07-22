from __future__ import annotations

import textwrap
from pathlib import Path

from prompt_toolkit.application import Application, get_app
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.input.base import Input
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.containers import Window
from prompt_toolkit.output.base import Output

from ...application.source import SavedSelection
from ..navigation import (
    NAVIGATION_STYLE,
    navigation_footer,
    scrollbar,
    wrap_styled_lines,
)


class SelectionReviewScreen:
    ACTIONS = (
        ("back", "Вернуться к карточкам"),
        ("export", "Зафиксировать решение и сформировать Markdown"),
    )

    def __init__(
        self,
        issues: list[dict[str, object]],
        *,
        selection: SavedSelection | None = None,
        section_number: str = "",
        mode_label: str | None = None,
        input: Input | None = None,
        output: Output | None = None,
    ) -> None:
        self.issues = issues
        self.selection = selection
        self.section_number = section_number
        self.mode_label = mode_label
        self.input = input
        self.output = output
        self.cursor = 0
        self.offset = 0
        self.follow_cursor = False
        self.export_paths: tuple[Path, Path] | None = None

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
        breadcrumb = "PMI Workbench"
        if self.selection is not None:
            start = self.selection.selection.start
            end = self.selection.selection.end
            breadcrumb += (
                f" / {self.section_number} / "
                f"стр. {start.page_index}:{start.line_number:03d}–"
                f"{end.page_index}:{end.line_number:03d}"
            )
        breadcrumb += " / Результат проверки"
        header_source = [("class:breadcrumb", breadcrumb)]
        if self.mode_label:
            header_source.append(("class:warning", self.mode_label))
        header_source.extend(
            [
                ("", ""),
                ("", f"Замечания: {len(self.issues)}"),
                ("", ""),
            ]
        )
        header = wrap_styled_lines(
            header_source,
            width=width,
        )
        rows = self._issue_rows(width=width)
        if self.export_paths is not None:
            markdown, diagnostics = self.export_paths
            rows.extend(
                [
                    ("", ""),
                    ("class:success", "Файлы сформированы:"),
                    ("", ""),
                    ("class:success", "- Итоговый Markdown:"),
                    ("class:success", f"  {markdown}"),
                    ("", ""),
                    ("class:success", "- Диагностика:"),
                    ("class:success", f"  {diagnostics}"),
                ]
            )
        rows.extend([("", ""), ("class:breadcrumb", "Действия:")])
        spans: list[tuple[int, int]] = []
        for index, (_, label) in enumerate(self.ACTIONS):
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

    def _issue_rows(self, *, width: int) -> list[tuple[str, str]]:
        if not self.issues:
            return [("class:success", "Существенных замечаний не найдено.")]
        rows: list[tuple[str, str]] = []
        for item in self.issues:
            kind = str(item.get("kind") or "Замечание")
            issue_id = str(item.get("issue_id") or "")
            heading = f"{kind}  [{issue_id}]" if issue_id else kind
            rows.extend(
                ("class:warning", line)
                for line in (
                    textwrap.wrap(
                        heading,
                        width=max(1, width - 1),
                        replace_whitespace=False,
                        break_long_words=True,
                        break_on_hyphens=False,
                    )
                    or [""]
                )
            )
            explanation = str(item.get("explanation") or "")
            rows.extend(
                ("", f"  {line}")
                for line in (
                    textwrap.wrap(
                        explanation,
                        width=max(1, width - 3),
                        replace_whitespace=False,
                        break_long_words=True,
                        break_on_hyphens=False,
                    )
                    or [""]
                )
            )
            for source in item.get("source_ranges", []):
                if not isinstance(source, dict):
                    continue
                rows.append(
                    (
                        "class:muted",
                        (
                            f"  Источник: стр. {source.get('page')}:"
                            f"{source.get('line_start')}–{source.get('line_end')}"
                        ),
                    )
                )
            rows.append(("", ""))
        return rows

    def _formatted_text(self) -> FormattedText:
        size = get_app().output.get_size()
        return self.render(width=size.columns, height=size.rows)

    def _bindings(self) -> KeyBindings:
        bindings = KeyBindings()

        def move(delta: int) -> None:
            self.cursor = min(max(0, self.cursor + delta), len(self.ACTIONS) - 1)
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
            event.app.exit(result=self.ACTIONS[self.cursor][0])  # type: ignore[attr-defined]

        @bindings.add("escape")
        def escape(event: object) -> None:
            event.app.exit(result="back")  # type: ignore[attr-defined]

        return bindings


def render_selection_review(
    issues: list[dict[str, object]],
    *,
    width: int,
    height: int,
    offset: int,
) -> str:
    screen = SelectionReviewScreen(issues)
    screen.offset = offset
    return "".join(fragment for _, fragment in screen.render(width=width, height=height))


def show_selection_review(issues: list[dict[str, object]]) -> bool:
    return SelectionReviewScreen(issues).run() == "export"
