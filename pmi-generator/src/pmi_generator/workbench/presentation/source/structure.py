from __future__ import annotations

import textwrap
from dataclasses import dataclass

from prompt_toolkit.application import Application, get_app
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.filters import Condition
from prompt_toolkit.input.base import Input
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.containers import Window
from prompt_toolkit.output.base import Output

from ...application.source import SelectionRangeStatus, SelectionRangeSummary
from ...domain.source import SourceDocument
from ..navigation import (
    NAVIGATION_STYLE,
    navigation_footer,
    scrollbar,
    wrap_styled_lines,
)
from .state import SourceNavigationState


@dataclass(frozen=True, slots=True)
class StructureEntry:
    key: str
    label: str
    indent: int = 0
    status: str = ""
    work_status: SelectionRangeStatus | None = None


@dataclass(frozen=True, slots=True)
class RenderedStructure:
    fragments: FormattedText
    selected_row_start: int
    selected_row_end: int
    content_height: int


@dataclass(frozen=True, slots=True)
class _SectionWork:
    status: SelectionRangeStatus
    text: str


class StructureScreen:
    def __init__(
        self,
        document: SourceDocument,
        ranges: tuple[SelectionRangeSummary, ...],
        *,
        source_name: str = "specification.pdf",
        notice: tuple[str, str] | None = None,
        mode_label: str | None = None,
        state: SourceNavigationState | None = None,
        input: Input | None = None,
        output: Output | None = None,
    ) -> None:
        self.document = document
        self.section_statuses = self._section_statuses(document, ranges)
        self.source_name = source_name
        self.notice = notice
        self.mode_label = mode_label
        self.state = state or SourceNavigationState()
        self.input = input
        self.output = output
        self.query = self.state.outline_query
        self._search_draft = ""
        self._searching = False
        self._cursor = self.state.outline_cursor
        self._offset = self.state.outline_offset
        self._result: str | None = None

    def run(self) -> str | None:
        bindings = self._bindings()
        control = FormattedTextControl(text=self._formatted_text, focusable=True)
        application: Application[str | None] = Application(
            layout=Layout(Window(control, wrap_lines=False)),
            key_bindings=bindings,
            style=NAVIGATION_STYLE,
            full_screen=True,
            input=self.input,
            output=self.output,
        )
        return application.run()

    def entries(self) -> tuple[StructureEntry, ...]:
        normalized = self.query.casefold().strip()

        entries: list[StructureEntry] = []
        for section in self.document.sections:
            if normalized and normalized not in section.label.casefold():
                continue
            entries.append(
                StructureEntry(
                    key=f"section:{section.section_id}",
                    label=section.label,
                    indent=max(0, len(section.path) - 1),
                    status=(
                        self.section_statuses[section.section_id].text
                        if section.section_id in self.section_statuses
                        else ""
                    ),
                    work_status=(
                        self.section_statuses[section.section_id].status
                        if section.section_id in self.section_statuses
                        else None
                    ),
                )
            )
        return tuple(entries)

    @staticmethod
    def _section_statuses(
        document: SourceDocument,
        ranges: tuple[SelectionRangeSummary, ...],
    ) -> dict[str, _SectionWork]:
        priority = {
            SelectionRangeStatus.TERMINAL: 0,
            SelectionRangeStatus.COMPLETED: 1,
            SelectionRangeStatus.ACTIVE: 2,
        }
        anchor_indexes = {
            section.section_id: document.position_index(
                document.anchor_position(section)
            )
            for section in document.sections
        }
        ordered_anchors = sorted(set(anchor_indexes.values()))
        result: dict[str, _SectionWork] = {}
        for section in document.sections:
            start = anchor_indexes[section.section_id]
            end = next(
                (anchor - 1 for anchor in ordered_anchors if anchor > start),
                len(document.positions) - 1,
            )
            intersecting = tuple(
                item
                for item in ranges
                if (
                    document.position_index(item.selection.start) <= end
                    and document.position_index(item.selection.end) >= start
                )
            )
            if not intersecting:
                continue
            status = max(
                (item.status for item in intersecting),
                key=priority.__getitem__,
            )
            result[section.section_id] = _SectionWork(
                status,
                _work_summary(intersecting),
            )
        return result

    def render(self, *, width: int, height: int) -> RenderedStructure:
        width = max(24, width)
        height = max(8, height)
        entries = self.entries()
        self._cursor = min(self._cursor, max(0, len(entries) - 1))

        search_value = self._search_draft if self._searching else self.query
        search_suffix = " _" if self._searching else ""
        header_source = [
            ("class:breadcrumb", "PMI Workbench / Структура спецификации"),
        ]
        if self.mode_label:
            header_source.append(("class:warning", self.mode_label))
        header_source.extend(
            [
                ("", ""),
                ("", f"Спецификация: {self.source_name}"),
                (
                    "class:search",
                    f"Поиск: {search_value or 'не задан'}{search_suffix}",
                ),
            ]
        )
        if self.notice is not None:
            notice_style, notice_text = self.notice
            header_source.extend([("", ""), (notice_style, notice_text)])
        header_source.append(("", ""))
        header = wrap_styled_lines(header_source, width=width)
        footer = navigation_footer(
            (
                "[↑/↓] раздел  [PgUp/PgDn] страница  [Enter] перейти к разделу  "
                "[E] экспорт ПМИ  [/] поиск  [Esc] выход"
            ),
            width=width,
            compact_text=(
                "[↑/↓ PgUp/PgDn] навигация  [Enter] перейти  "
                "[E] экспорт  [/] поиск  [Esc] выход"
            ),
        )
        viewport_height = max(1, height - len(header) - len(footer))
        rows, spans = self._entry_rows(entries, width=width)
        if spans:
            selected_start, selected_end = spans[self._cursor]
        else:
            rows.append(("class:muted", "  Разделы не найдены."))
            selected_start = selected_end = 0
        max_offset = max(0, len(rows) - viewport_height)
        if selected_start < self._offset:
            self._offset = selected_start
        elif selected_end >= self._offset + viewport_height:
            self._offset = selected_end - viewport_height + 1
        self._offset = min(max(0, self._offset), max_offset)
        self._sync_state()

        visible = rows[self._offset : self._offset + viewport_height]
        visible.extend([("", "")] * (viewport_height - len(visible)))
        thumb_start, thumb_size = scrollbar(
            total=len(rows),
            visible=viewport_height,
            offset=self._offset,
        )

        fragments: list[tuple[str, str]] = []
        for style, line in header:
            fragments.extend([(style, line[:width]), ("", "\n")])
        for index, (style, line) in enumerate(visible):
            track_style = "class:scrollbar.thumb" if thumb_start <= index < thumb_start + thumb_size else "class:scrollbar.track"
            track = "█" if thumb_start <= index < thumb_start + thumb_size else "│"
            fragments.extend(
                [
                    (style, line[: width - 1].ljust(width - 1)),
                    (track_style, track),
                    ("", "\n"),
                ]
            )
        for index, (style, line) in enumerate(footer):
            fragments.append((style, line[:width]))
            if index < len(footer) - 1:
                fragments.append(("", "\n"))
        return RenderedStructure(
            fragments=FormattedText(fragments),
            selected_row_start=selected_start,
            selected_row_end=selected_end,
            content_height=len(rows),
        )

    def _sync_state(self) -> None:
        self.state.outline_cursor = self._cursor
        self.state.outline_offset = self._offset
        self.state.outline_query = self.query

    def _formatted_text(self) -> FormattedText:
        size = get_app().output.get_size()
        return self.render(width=size.columns, height=size.rows).fragments

    def _entry_rows(
        self,
        entries: tuple[StructureEntry, ...],
        *,
        width: int,
    ) -> tuple[list[tuple[str, str]], list[tuple[int, int]]]:
        rows: list[tuple[str, str]] = []
        spans: list[tuple[int, int]] = []
        for index, entry in enumerate(entries):
            start = len(rows)
            selected = index == self._cursor
            marker = ">" if selected else " "
            indentation = "  " * entry.indent
            suffix = f"  [{entry.status}]" if entry.status else ""
            prefix = f"{marker} {indentation}"
            available = max(1, width - 1 - len(prefix))
            wrapped = textwrap.wrap(
                entry.label + suffix,
                width=available,
                replace_whitespace=False,
                break_long_words=True,
                break_on_hyphens=False,
            ) or [""]
            if entry.work_status is None:
                style = "class:selected" if selected else "class:section"
            else:
                cursor = ".cursor" if selected else ""
                style = f"class:range.{entry.work_status.value}{cursor}"
            rows.append((style, prefix + wrapped[0]))
            continuation = " " * len(prefix)
            rows.extend((style, continuation + line) for line in wrapped[1:])
            spans.append((start, len(rows) - 1))
        return rows, spans

    def _bindings(self) -> KeyBindings:
        bindings = KeyBindings()
        navigating = Condition(lambda: not self._searching)
        searching = Condition(lambda: self._searching)

        def move(delta: int) -> None:
            entries = self.entries()
            if not entries:
                self._cursor = 0
                return
            self._cursor = min(max(0, self._cursor + delta), len(entries) - 1)
            self._sync_state()

        @bindings.add("up", filter=navigating)
        @bindings.add("<scroll-up>", filter=navigating)
        def up(event: object) -> None:
            del event
            move(-1)

        @bindings.add("down", filter=navigating)
        @bindings.add("<scroll-down>", filter=navigating)
        def down(event: object) -> None:
            del event
            move(1)

        @bindings.add("pageup", filter=navigating)
        def page_up(event: object) -> None:
            size = event.app.output.get_size()  # type: ignore[attr-defined]
            move(-max(1, size.rows - 7))

        @bindings.add("pagedown", filter=navigating)
        def page_down(event: object) -> None:
            size = event.app.output.get_size()  # type: ignore[attr-defined]
            move(max(1, size.rows - 7))

        @bindings.add("enter")
        def enter(event: object) -> None:
            if self._searching:
                self.query = self._search_draft.strip()
                self._searching = False
                self._cursor = 0
                self._offset = 0
                self._sync_state()
                return
            entries = self.entries()
            if not entries:
                return
            self._result = entries[self._cursor].key
            self._sync_state()
            event.app.exit(result=self._result)  # type: ignore[attr-defined]

        @bindings.add("e", filter=navigating)
        @bindings.add("E", filter=navigating)
        def export(event: object) -> None:
            self._result = "export"
            self._sync_state()
            event.app.exit(result=self._result)  # type: ignore[attr-defined]

        @bindings.add("/")
        def search(event: object) -> None:
            del event
            if self._searching:
                self._search_draft += "/"
                return
            self._search_draft = self.query
            self._searching = True

        @bindings.add("backspace", filter=searching)
        @bindings.add("c-h", filter=searching)
        def backspace(event: object) -> None:
            del event
            self._search_draft = self._search_draft[:-1]

        @bindings.add("escape")
        def escape(event: object) -> None:
            if self._searching:
                self._search_draft = self.query
                self._searching = False
                self._sync_state()
                return
            self._sync_state()
            event.app.exit(result=None)  # type: ignore[attr-defined]

        @bindings.add("<any>", filter=searching)
        def insert(event: object) -> None:
            data = event.data  # type: ignore[attr-defined]
            if data and data.isprintable():
                self._search_draft += data

        return bindings


def select_structure_action(
    document: SourceDocument,
    ranges: tuple[SelectionRangeSummary, ...],
    *,
    source_name: str = "specification.pdf",
    notice: tuple[str, str] | None = None,
    mode_label: str | None = None,
    state: SourceNavigationState | None = None,
    input: Input | None = None,
    output: Output | None = None,
) -> str | None:
    return StructureScreen(
        document,
        ranges,
        source_name=source_name,
        notice=notice,
        mode_label=mode_label,
        state=state,
        input=input,
        output=output,
    ).run()


def _work_summary(ranges: tuple[SelectionRangeSummary, ...]) -> str:
    counts = {
        status: sum(item.status is status for item in ranges)
        for status in SelectionRangeStatus
    }
    total = len(ranges)
    if total == 1:
        status = ranges[0].status
        label = {
            SelectionRangeStatus.ACTIVE: "в работе",
            SelectionRangeStatus.COMPLETED: "готов",
            SelectionRangeStatus.TERMINAL: "нет поведения",
        }[status]
        return f"1 диапазон: {label}"
    parts: list[str] = []
    if counts[SelectionRangeStatus.COMPLETED]:
        parts.append(f"{counts[SelectionRangeStatus.COMPLETED]} готовы")
    if counts[SelectionRangeStatus.ACTIVE]:
        parts.append(f"{counts[SelectionRangeStatus.ACTIVE]} в работе")
    if counts[SelectionRangeStatus.TERMINAL]:
        parts.append(f"{counts[SelectionRangeStatus.TERMINAL]} без поведения")
    suffix = "диапазона" if total < 5 else "диапазонов"
    return f"{total} {suffix}: " + ", ".join(parts)
