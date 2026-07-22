from __future__ import annotations

import textwrap
from collections.abc import Callable
from dataclasses import dataclass

from prompt_toolkit.application import Application, get_app
from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.input.base import Input
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.containers import Window
from prompt_toolkit.output.base import Output

from ...application.decomposition import (
    DecompositionBudget,
    DecompositionRoute,
    WindowingDecision,
)
from ...application.source import SelectionRangeSummary
from ...domain.source import (
    SourceDocument,
    SourcePosition,
    SourceSection,
    TextSelection,
)
from ..navigation import (
    NAVIGATION_STYLE,
    navigation_footer,
    scrollbar,
    wrap_styled_lines,
)
from .state import SourceNavigationState


@dataclass(frozen=True, slots=True)
class RenderedSelection:
    fragments: FormattedText
    cursor_row_start: int
    cursor_row_end: int
    content_height: int


@dataclass(frozen=True, slots=True)
class CreateSelection:
    selection: TextSelection
    supersede_selection_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class OpenRange:
    selection_id: str


SelectionScreenResult = CreateSelection | OpenRange


class SelectionScreen:
    def __init__(
        self,
        document: SourceDocument,
        section: SourceSection,
        *,
        ranges: tuple[SelectionRangeSummary, ...] = (),
        mode_label: str | None = None,
        state: SourceNavigationState | None = None,
        navigate_to_anchor: bool = True,
        input: Input | None = None,
        output: Output | None = None,
    ) -> None:
        self.document = document
        self.section = section
        self.ranges = ranges
        ranges_by_position: dict[SourcePosition, list[SelectionRangeSummary]] = {}
        for saved in ranges:
            for position in saved.selection.positions:
                ranges_by_position.setdefault(position, []).append(saved)
        self._ranges_by_position = {
            position: tuple(items)
            for position, items in ranges_by_position.items()
        }
        self._range_by_position = {
            position: _preferred_range(items)
            for position, items in self._ranges_by_position.items()
        }
        self.mode_label = mode_label
        self.state = state or SourceNavigationState()
        self.input = input
        self.output = output
        self.positions = document.positions
        if not self.positions:
            raise ValueError("В исходном документе нет строк")
        if not navigate_to_anchor and self.state.canvas_cursor is not None:
            self.cursor = document.position_index(self.state.canvas_cursor)
        else:
            self.cursor = document.position_index(document.anchor_position(section))
        if not navigate_to_anchor and self.state.canvas_scroll_anchor is not None:
            self.offset = document.position_index(self.state.canvas_scroll_anchor)
        else:
            self.offset = self.cursor
        self.selection_start = self.state.selection_anchor
        self.selection = self.state.draft_selection
        self.selection_complete = self.state.selection_complete
        self.query = self.state.canvas_query
        self._search_draft = ""
        self._searching = False
        self.search_wrapped = False
        self._last_width = 80
        self._last_height = 24
        self._rendered_positions: tuple[SourcePosition, ...] = ()
        self._outline_anchor_positions = frozenset(
            document.anchor_position(item) for item in document.sections
        )
        self.state.anchor_outline_node_id = section.section_id
        self._sync_state()

    @property
    def current_outline(self) -> SourceSection:
        return self.document.outline_at(
            self.positions[self.cursor],
            preferred_section_id=self.state.anchor_outline_node_id,
        )

    def _move_cursor(self, delta: int) -> None:
        self.cursor = min(max(0, self.cursor + delta), len(self.positions) - 1)
        self._refresh_active_selection()
        self._sync_state()

    def _toggle_selection(self) -> None:
        current = self.positions[self.cursor]
        if self.selection_start is None or self.selection_complete:
            self.selection_start = current
            self.selection = self.document.select(current, current)
            self.selection_complete = False
        else:
            self.selection = self.document.select(self.selection_start, current)
            self.selection_complete = True
        self._sync_state()

    def cancel_draft(self) -> bool:
        if self.selection_start is None and self.selection is None:
            return False
        self.selection_start = None
        self.selection = None
        self.selection_complete = False
        self._sync_state()
        return True

    def edit_selection(self) -> None:
        if self.selection is None:
            return
        self.selection_start = self.selection.start
        self.cursor = self.document.position_index(self.selection.end)
        self.selection_complete = False
        self._sync_state()

    def find(self, query: str, *, direction: int) -> SourcePosition | None:
        normalized = query.casefold()
        self.search_wrapped = False
        if not normalized:
            return None
        step = 1 if direction >= 0 else -1
        total = len(self.positions)
        for distance in range(1, total + 1):
            raw_index = self.cursor + step * distance
            index = raw_index % total
            if raw_index < 0 or raw_index >= total:
                self.search_wrapped = True
            position = self.positions[index]
            if normalized in self.document.line(position).casefold():
                self.cursor = index
                self._refresh_active_selection()
                self._sync_state()
                return position
        self._sync_state()
        return None

    def _refresh_active_selection(self) -> None:
        if self.selection_start is not None and not self.selection_complete:
            self.selection = self.document.select(
                self.selection_start,
                self.positions[self.cursor],
            )

    def _sync_state(self) -> None:
        self.state.canvas_cursor = self.positions[self.cursor]
        self.state.canvas_scroll_anchor = self.positions[self.offset]
        self.state.selection_anchor = self.selection_start
        self.state.draft_selection = self.selection
        self.state.selection_complete = self.selection_complete
        self.state.canvas_query = self.query

    def run(self) -> SelectionScreenResult | None:
        application: Application[SelectionScreenResult | None] = Application(
            layout=Layout(
                Window(
                    FormattedTextControl(
                        text=self._formatted_text,
                        focusable=True,
                    ),
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

    def render(self, *, width: int, height: int) -> RenderedSelection:
        width = max(24, width)
        height = max(8, height)
        self._last_width = width
        self._last_height = height
        label = _selection_label(self.selection_start, self.selection)
        outline = self.current_outline
        header_source = [
            (
                "class:breadcrumb",
                f"PMI Workbench / Структура / {outline.label} / Исходный текст",
            ),
        ]
        if self.mode_label:
            header_source.append(("class:warning", self.mode_label))
        header_source.extend(
            [
                ("", ""),
                ("class:selection", f"Выбор: {label}"),
                ("", ""),
            ]
        )
        if self._searching or self.query:
            query = self._search_draft if self._searching else self.query
            suffix = " — поиск продолжен с границы документа" if self.search_wrapped else ""
            header_source.extend(
                [
                    ("class:search", f"Поиск: {query or 'не задан'}{suffix}"),
                    ("", ""),
                ]
            )
        header = wrap_styled_lines(
            header_source,
            width=width,
        )
        if self._searching:
            footer = navigation_footer(
                "[Enter] следующее  [Shift+Enter] предыдущее  [Esc] закрыть поиск",
                width=width,
            )
        elif self.selection_complete:
            footer = navigation_footer(
                "[Enter] проверить диапазон  [Space] выбрать заново  [Esc] отменить выбор",
                width=width,
            )
        elif self.selection_start is not None:
            footer = navigation_footer(
                "[Space] завершить выбор  [/] поиск  [Esc] отменить выбор",
                width=width,
            )
        else:
            open_hint = (
                "  [Enter] открыть диапазон"
                if self.positions[self.cursor] in self._range_by_position
                else ""
            )
            footer = navigation_footer(
                (
                    "[↑/↓ PgUp/PgDn Home/End] навигация  [Space] начать выбор  "
                    f"[/] поиск{open_hint}  [Esc] к структуре"
                ),
                width=width,
                compact_text=(
                    "[↑/↓ PgUp/PgDn Home/End] навигация  [Space] выбор  "
                    f"[/] поиск{open_hint}  [Esc] назад"
                ),
            )
        viewport_height = max(1, height - len(header) - len(footer))
        self._keep_cursor_in_source_window(viewport_height)
        rows, spans = self._source_rows(
            width=width,
            max_rows=viewport_height + 4,
        )
        cursor_start = cursor_end = 0
        if self.positions[self.cursor] in self._rendered_positions:
            rendered_index = self._rendered_positions.index(self.positions[self.cursor])
            cursor_start, cursor_end = spans[rendered_index]
        while cursor_start >= viewport_height and self.offset < self.cursor:
            self.offset += 1
            rows, spans = self._source_rows(
                width=width,
                max_rows=viewport_height + 4,
            )
            rendered_index = self._rendered_positions.index(self.positions[self.cursor])
            cursor_start, cursor_end = spans[rendered_index]
        self._sync_state()

        visible = rows[:viewport_height]
        visible.extend([("", "")] * (viewport_height - len(visible)))
        thumb_start, thumb_size = scrollbar(
            total=len(self.positions),
            visible=max(1, len(self._rendered_positions)),
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
                        "class:scrollbar.thumb" if active else "class:scrollbar.track",
                        "█" if active else "│",
                    ),
                    ("", "\n"),
                ]
            )
        for index, (style, line) in enumerate(footer):
            fragments.append((style, line[:width]))
            if index < len(footer) - 1:
                fragments.append(("", "\n"))
        return RenderedSelection(
            fragments=FormattedText(fragments),
            cursor_row_start=cursor_start,
            cursor_row_end=cursor_end,
            content_height=len(rows),
        )

    def _source_rows(
        self,
        *,
        width: int,
        max_rows: int | None = None,
    ) -> tuple[list[tuple[str, str]], list[tuple[int, int]]]:
        rows: list[tuple[str, str]] = []
        spans: list[tuple[int, int]] = []
        selected = set(self.selection.positions) if self.selection else (
            {self.selection_start} if self.selection_start else set()
        )
        rendered_positions: list[SourcePosition] = []
        row_limit = max_rows or self._last_height + 4
        previous_page: int | None = None
        for position in self.positions[self.offset :]:
            if len(rows) >= row_limit and rendered_positions:
                break
            if position.page_index != previous_page:
                page = self.document.page(position.page_index)
                if previous_page is not None:
                    rows.append(("", ""))
                previous_page = position.page_index
                label = f" Страница {page.display_number} "
                rows.append(
                    (
                        "class:muted",
                        f"──{label}{'─' * max(0, width - len(label) - 3)}",
                    )
                )
            saved_range = self._range_by_position.get(position)
            if saved_range is not None and position == saved_range.selection.start:
                rows.append(("", ""))
                rows.append(
                    (
                        f"class:range.{saved_range.status.value}",
                        f"    [{saved_range.status_text}]",
                    )
                )
            start = len(rows)
            cursor = self.positions[self.cursor] == position
            prefix = f"{'>' if cursor else ' '} {position.line_number:03d} | "
            available = max(1, width - 1 - len(prefix))
            wrapped = textwrap.wrap(
                self.document.line(position),
                width=available,
                replace_whitespace=False,
                drop_whitespace=False,
                break_long_words=True,
                break_on_hyphens=False,
            ) or [""]
            heading = position in self._outline_anchor_positions
            if position in selected and cursor:
                style = "class:selection.cursor"
            elif position in selected:
                style = "class:selection"
            elif saved_range is not None and cursor:
                style = f"class:range.{saved_range.status.value}.cursor"
            elif saved_range is not None:
                style = f"class:range.{saved_range.status.value}"
            elif heading and cursor:
                style = "class:source.heading.cursor"
            elif heading:
                style = "class:source.heading"
            elif cursor:
                style = "class:cursor"
            else:
                style = ""
            rows.append((style, prefix + wrapped[0]))
            continuation = " " * len(prefix)
            rows.extend((style, continuation + line) for line in wrapped[1:])
            rendered_positions.append(position)
            spans.append((start, len(rows) - 1))
        self._rendered_positions = tuple(rendered_positions)
        return rows, spans

    def _keep_cursor_in_source_window(self, viewport_height: int) -> None:
        source_window = max(1, viewport_height - 2)
        if self.cursor < self.offset:
            self.offset = self.cursor
        elif self.cursor >= self.offset + source_window:
            self.offset = self.cursor - source_window + 1
        self.offset = min(max(0, self.offset), len(self.positions) - 1)

    def _formatted_text(self) -> FormattedText:
        size = get_app().output.get_size()
        return self.render(width=size.columns, height=size.rows).fragments

    def _bindings(self) -> KeyBindings:
        bindings = KeyBindings()
        navigating = Condition(lambda: not self._searching)
        searching = Condition(lambda: self._searching)

        @bindings.add("up", filter=navigating)
        @bindings.add("<scroll-up>", filter=navigating)
        def up(event: object) -> None:
            del event
            self._move_cursor(-1)

        @bindings.add("down", filter=navigating)
        @bindings.add("<scroll-down>", filter=navigating)
        def down(event: object) -> None:
            del event
            self._move_cursor(1)

        @bindings.add("pageup", filter=navigating)
        def page_up(event: object) -> None:
            del event
            self._move_page(-1)

        @bindings.add("pagedown", filter=navigating)
        def page_down(event: object) -> None:
            del event
            self._move_page(1)

        @bindings.add("home", filter=navigating)
        def home(event: object) -> None:
            del event
            self._move_cursor(-len(self.positions))

        @bindings.add("end", filter=navigating)
        def end(event: object) -> None:
            del event
            self._move_cursor(len(self.positions))

        @bindings.add(" ", filter=navigating)
        def space(event: object) -> None:
            del event
            self._toggle_selection()

        @bindings.add("enter", filter=navigating)
        def enter(event: object) -> None:
            if self.selection is not None and self.selection_complete:
                event.app.exit(result=CreateSelection(self.selection))  # type: ignore[attr-defined]
                return
            saved_range = self._range_by_position.get(self.positions[self.cursor])
            if saved_range is not None:
                event.app.exit(result=OpenRange(saved_range.selection_id))  # type: ignore[attr-defined]

        @bindings.add("/")
        def search(event: object) -> None:
            del event
            if self._searching:
                self._search_draft += "/"
                return
            self._search_draft = self.query
            self._searching = True
            self.search_wrapped = False

        @bindings.add("backspace", filter=searching)
        @bindings.add("c-h", filter=searching)
        def backspace(event: object) -> None:
            del event
            self._search_draft = self._search_draft[:-1]

        @bindings.add("enter", filter=searching)
        def search_next(event: object) -> None:
            del event
            self.query = self._search_draft
            self.find(self.query, direction=1)

        @bindings.add("c-j", filter=searching)
        @bindings.add("up", filter=searching)
        def search_previous(event: object) -> None:
            del event
            self.query = self._search_draft
            self.find(self.query, direction=-1)

        @bindings.add("<any>", filter=searching)
        def insert_search_text(event: object) -> None:
            data = event.data  # type: ignore[attr-defined]
            if data and data.isprintable():
                self._search_draft += data

        @bindings.add("escape")
        def escape(event: object) -> None:
            if self._searching:
                self.query = self._search_draft
                self._searching = False
                self._sync_state()
                return
            if self.cancel_draft():
                return
            event.app.exit(result=None)  # type: ignore[attr-defined]

        return bindings

    def _move_page(self, direction: int) -> None:
        viewport = max(1, self._last_height - 8)
        overlap = min(2, max(0, viewport - 1))
        self._move_cursor(direction * max(1, viewport - overlap))


class ConfirmationScreen:
    DEFAULT_ACTIONS = (
        ("build", "Построить каркасы карточек"),
        ("change", "Изменить диапазон"),
        ("cancel", "Отмена"),
    )

    def __init__(
        self,
        section: SourceSection,
        selection: TextSelection,
        *,
        source_name: str,
        overlapping_ranges: tuple[SelectionRangeSummary, ...] = (),
        replaceable: bool = False,
        exact_existing: bool = False,
        decomposition_budget: DecompositionBudget | None = None,
        windowing_decision: WindowingDecision | None = None,
        mode_label: str | None = None,
        input: Input | None = None,
        output: Output | None = None,
    ) -> None:
        self.section = section
        self.selection = selection
        self.source_name = source_name
        self.overlapping_ranges = overlapping_ranges
        self.replaceable = replaceable
        self.exact_existing = exact_existing
        self.decomposition_budget = decomposition_budget
        self.windowing_decision = windowing_decision
        hard_limit = (
            windowing_decision is not None
            and windowing_decision.route is DecompositionRoute.HARD_LIMIT
        )
        over_budget = (
            windowing_decision is None
            and decomposition_budget is not None
            and not decomposition_budget.within_single_call
        )
        if exact_existing:
            self.actions = (
                ("open", "Открыть существующий диапазон"),
                ("change", "Изменить диапазон"),
                ("cancel", "Отмена"),
            )
        elif hard_limit or over_budget:
            self.actions = (
                ("change", "Изменить диапазон"),
                ("cancel", "Отмена"),
            )
        elif replaceable:
            self.actions = (
                ("replace", "Заменить диапазон расширенным"),
                ("change", "Изменить диапазон"),
                ("cancel", "Отмена"),
            )
        elif overlapping_ranges:
            self.actions = (
                ("change", "Изменить диапазон"),
                ("cancel", "Отмена"),
            )
        else:
            self.actions = self.DEFAULT_ACTIONS
        self.mode_label = mode_label
        self.input = input
        self.output = output
        self.cursor = 0
        self.offset = 0
        self.follow_cursor = False

    def run(self) -> str | None:
        application: Application[str | None] = Application(
            layout=Layout(
                Window(
                    FormattedTextControl(
                        text=self._formatted_text,
                        focusable=True,
                    ),
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
        start = self.selection.start
        end = self.selection.end
        header_source = [
            (
                "class:breadcrumb",
                f"PMI Workbench / Структура / {self.section.label} / Новый диапазон",
            ),
        ]
        if self.mode_label:
            header_source.append(("class:warning", self.mode_label))
        header_source.extend(
            [
                ("", ""),
                ("", f"Источник: {self.source_name}"),
                (
                    "",
                    (
                        f"Диапазон: стр. {start.page_index}:{start.line_number:03d} — "
                        f"стр. {end.page_index}:{end.line_number:03d}"
                    ),
                ),
                ("", f"Строк: {len(self.selection.positions)}"),
                (
                    "",
                    f"Страниц: {len({item.page_index for item in self.selection.positions})}",
                ),
            ]
        )
        if self.overlapping_ranges:
            if self.replaceable:
                overlap_text = (
                    "Поглощённый диапазон будет сохранён в истории."
                )
            elif self.exact_existing:
                overlap_text = "Такой диапазон уже существует."
            else:
                overlap_text = (
                    "Новый диапазон пересекается с существующей работой и не будет создан."
                )
            header_source.extend(
                [
                    ("class:warning", overlap_text),
                    (
                        "class:muted",
                        "Диапазоны: "
                        + ", ".join(
                            item.selection_id for item in self.overlapping_ranges
                        ),
                    ),
                ]
            )
        if (
            self.windowing_decision is not None
            and self.windowing_decision.route is DecompositionRoute.WINDOWED
            and not self.exact_existing
        ):
            budget = self.windowing_decision.budget
            header_source.extend(
                [
                    (
                        "class:warning",
                        "Большой диапазон: обработка может занять больше времени.",
                    ),
                    (
                        "class:muted",
                        "Выбрано: "
                        f"{_format_number(budget.line_count)} строк / "
                        f"{_format_number(budget.estimated_tokens)} "
                        "оценочных токенов",
                    ),
                ]
            )
        if (
            self.windowing_decision is not None
            and self.windowing_decision.route is DecompositionRoute.HARD_LIMIT
            and not self.exact_existing
        ):
            decision = self.windowing_decision
            header_source.extend(
                [
                    (
                        "class:warning",
                        "Диапазон превышает абсолютный лимит обработки.",
                    ),
                    (
                        "class:muted",
                        "Выбрано: "
                        f"{_format_number(decision.budget.line_count)} строк / "
                        f"{_format_number(decision.budget.estimated_tokens)} "
                        "оценочных токенов",
                    ),
                    (
                        "class:muted",
                        "Hard limit: "
                        f"{_format_number(decision.hard_max_lines)} строк / "
                        f"{_format_number(decision.hard_max_estimated_tokens)} "
                        "оценочных токенов",
                    ),
                ]
            )
        if (
            self.decomposition_budget is not None
            and self.windowing_decision is None
            and not self.decomposition_budget.within_single_call
            and not self.exact_existing
        ):
            budget = self.decomposition_budget
            header_source.extend(
                [
                    (
                        "class:warning",
                        "Диапазон превышает технический бюджет Prompt 1.",
                    ),
                    (
                        "class:muted",
                        "Выбрано: "
                        f"{_format_number(budget.line_count)} строк / "
                        f"{_format_number(budget.estimated_tokens)} "
                        "оценочных токенов",
                    ),
                    (
                        "class:muted",
                        "Допустимо текущей policy: "
                        f"{_format_number(budget.max_lines)} строк / "
                        f"{_format_number(budget.max_estimated_tokens)} "
                        "оценочных токенов",
                    ),
                ]
            )
        header_source.append(("", ""))
        header = wrap_styled_lines(
            header_source,
            width=width,
        )
        rows: list[tuple[str, str]] = []
        over_budget = (
            self.windowing_decision is None
            and self.decomposition_budget is not None
            and not self.decomposition_budget.within_single_call
            and not self.exact_existing
        )
        hard_limit = (
            self.windowing_decision is not None
            and self.windowing_decision.route is DecompositionRoute.HARD_LIMIT
            and not self.exact_existing
        )
        if not over_budget and not hard_limit:
            rows.extend(
                _selection_preview_rows(
                    self.selection,
                    width=width - 1,
                )
            )
        rows.extend([("", ""), ("class:breadcrumb", "Действия:")])
        action_spans: list[tuple[int, int]] = []
        for index, (_, label) in enumerate(self.actions):
            start_row = len(rows)
            marker = ">" if index == self.cursor else " "
            wrapped = textwrap.wrap(label, width=max(1, width - 3)) or [""]
            style = "class:selected" if index == self.cursor else "class:action"
            rows.append((style, f"{marker} {wrapped[0]}"))
            rows.extend((style, f"  {line}") for line in wrapped[1:])
            action_spans.append((start_row, len(rows) - 1))

        footer = navigation_footer(
            "[↑/↓] выбор  [PgUp/PgDn] прокрутка  [Enter] выполнить  [Esc] назад",
            width=width,
        )
        viewport_height = max(1, height - len(header) - len(footer))
        selected_start, selected_end = action_spans[self.cursor]
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
                        "class:scrollbar.thumb" if active else "class:scrollbar.track",
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
            event.app.exit(result=None)  # type: ignore[attr-defined]

        return bindings


def select_text_range(
    document: SourceDocument,
    section: SourceSection,
    *,
    ranges: tuple[SelectionRangeSummary, ...] = (),
    source_name: str = "specification.pdf",
    mode_label: str | None = None,
    state: SourceNavigationState | None = None,
    navigate_to_anchor: bool = True,
    input: Input | None = None,
    output: Output | None = None,
    assess_decomposition: Callable[[TextSelection], DecompositionBudget] | None = None,
    assess_windowing: Callable[[TextSelection], WindowingDecision] | None = None,
) -> SelectionScreenResult | None:
    selection_screen = SelectionScreen(
        document,
        section,
        ranges=ranges,
        mode_label=mode_label,
        state=state,
        navigate_to_anchor=navigate_to_anchor,
        input=input,
        output=output,
    )
    while True:
        result = selection_screen.run()
        if result is None or isinstance(result, OpenRange):
            return result
        selection = result.selection
        if selection is None:
            return None
        overlaps = _overlapping_ranges(selection, ranges)
        replacement_ids = _replacement_ids(selection, overlaps)
        exact_existing = _exact_range(selection, overlaps)
        decomposition_budget = (
            assess_decomposition(selection)
            if assess_decomposition is not None
            else None
        )
        windowing_decision = (
            assess_windowing(selection)
            if assess_windowing is not None
            else None
        )
        action = ConfirmationScreen(
            section,
            selection,
            source_name=source_name,
            overlapping_ranges=overlaps,
            replaceable=bool(replacement_ids),
            exact_existing=exact_existing is not None,
            decomposition_budget=decomposition_budget,
            windowing_decision=windowing_decision,
            mode_label=mode_label,
            input=input,
            output=output,
        ).run()
        if action == "build":
            selection_screen.cancel_draft()
            return result
        if action == "replace":
            selection_screen.cancel_draft()
            return CreateSelection(
                selection,
                supersede_selection_ids=replacement_ids,
            )
        if action == "open":
            assert exact_existing is not None
            return OpenRange(exact_existing.selection_id)
        if action == "change":
            selection_screen.edit_selection()
            continue
        if action == "cancel":
            selection_screen.cancel_draft()
            return None


def _format_number(value: int) -> str:
    return f"{value:,}".replace(",", " ")


def _selection_preview_rows(
    selection: TextSelection,
    *,
    width: int,
) -> list[tuple[str, str]]:
    source_lines = selection.text.split("\n")
    addressed = tuple(zip(selection.positions, source_lines, strict=True))
    groups: tuple[tuple[str, tuple[tuple[SourcePosition, str], ...]], ...]
    if len(addressed) <= 6:
        groups = (("Выбранные строки:", addressed),)
    else:
        groups = (
            ("Первые строки:", addressed[:2]),
            ("Последние строки:", addressed[-2:]),
        )

    rows: list[tuple[str, str]] = []
    for group_index, (label, items) in enumerate(groups):
        if group_index:
            rows.append(("", ""))
        rows.append(("class:breadcrumb", label))
        for position, source_line in items:
            prefix = (
                f"  стр. {position.page_index}:{position.line_number:03d} | "
            )
            wrapped = textwrap.wrap(
                source_line,
                width=max(1, width - len(prefix)),
                replace_whitespace=False,
                drop_whitespace=False,
                break_long_words=True,
                break_on_hyphens=False,
            ) or [""]
            rows.append(("", prefix + wrapped[0]))
            rows.extend(("", " " * len(prefix) + line) for line in wrapped[1:])
    return rows


def _selection_label(
    start: SourcePosition | None,
    selection: TextSelection | None,
) -> str:
    if selection is not None:
        return (
            f"стр. {selection.start.page_index}:{selection.start.line_number:03d} — "
            f"стр. {selection.end.page_index}:{selection.end.line_number:03d}"
        )
    if start is not None:
        return f"начало стр. {start.page_index}:{start.line_number:03d}"
    return "начало не задано"


def _overlapping_ranges(
    selection: TextSelection,
    ranges: tuple[SelectionRangeSummary, ...],
) -> tuple[SelectionRangeSummary, ...]:
    positions = set(selection.positions)
    return tuple(
        item
        for item in ranges
        if bool(positions.intersection(item.selection.positions))
    )


def _replacement_ids(
    selection: TextSelection,
    overlaps: tuple[SelectionRangeSummary, ...],
) -> tuple[str, ...]:
    selected_positions = set(selection.positions)
    if len(overlaps) != 1:
        return ()
    if not set(overlaps[0].selection.positions) < selected_positions:
        return ()
    return (overlaps[0].selection_id,)


def _exact_range(
    selection: TextSelection,
    overlaps: tuple[SelectionRangeSummary, ...],
) -> SelectionRangeSummary | None:
    if len(overlaps) != 1:
        return None
    exact = tuple(item for item in overlaps if item.selection.positions == selection.positions)
    return _preferred_range(exact) if exact else None


def _preferred_range(
    ranges: tuple[SelectionRangeSummary, ...] | list[SelectionRangeSummary],
) -> SelectionRangeSummary:
    priorities = {"terminal": 0, "completed": 1, "active": 2}
    if not ranges:
        raise ValueError("Не указан существующий диапазон")
    return max(
        ranges,
        key=lambda item: (priorities[item.status.value], item.selection_id),
    )
