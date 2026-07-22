from __future__ import annotations

import textwrap
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

from ...application.source import SavedSelection
from ...application.state import StoredRecord
from ..navigation import (
    NAVIGATION_STYLE,
    navigation_footer,
    scrollbar,
    wrap_styled_lines,
)


@dataclass(frozen=True, slots=True)
class SkeletonDetailResult:
    action: str
    reason: str = ""


class SkeletonListScreen:
    def __init__(
        self,
        records: tuple[StoredRecord, ...],
        selection: SavedSelection,
        *,
        mode_label: str | None = None,
        input: Input | None = None,
        output: Output | None = None,
    ) -> None:
        self.records = records
        self.selection = selection
        self.mode_label = mode_label
        self.input = input
        self.output = output
        self.cursor = 0
        self.offset = 0

    def update(self, records: tuple[StoredRecord, ...]) -> None:
        current_id = self.records[self.cursor].record_id if self.records else None
        self.records = records
        if current_id:
            self.cursor = next(
                (
                    index
                    for index, record in enumerate(records)
                    if record.record_id == current_id
                ),
                min(self.cursor, max(0, len(records) - 1)),
            )

    def advance_after_decision(self, skeleton_id: str) -> None:
        unresolved = [
            index
            for index, record in enumerate(self.records)
            if record.payload.get("decision") is None
        ]
        if not unresolved:
            return
        current = next(
            (
                index
                for index, record in enumerate(self.records)
                if record.record_id == skeleton_id
            ),
            self.cursor,
        )
        self.cursor = next(
            (index for index in unresolved if index > current),
            unresolved[0],
        )

    def run(self) -> str | None:
        if not self.records:
            return None
        application: Application[str | None] = Application(
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
        start = self.selection.selection.start
        end = self.selection.selection.end
        decided = sum(
            record.payload.get("decision") is not None for record in self.records
        )
        section_number = str(self.records[0].payload.get("section_number", ""))
        header_source = [
                (
                    "class:breadcrumb",
                    (
                        f"PMI Workbench / {section_number} / "
                        f"стр. {start.page_index}:{start.line_number:03d}–"
                        f"{end.page_index}:{end.line_number:03d} / Каркасы"
                    ),
                ),
        ]
        if self.mode_label:
            header_source.append(("class:warning", self.mode_label))
        header_source.extend(
            [
                ("", ""),
                ("", f"Решения: {decided} из {len(self.records)}"),
                ("", ""),
            ]
        )
        header = wrap_styled_lines(
            header_source,
            width=width,
        )
        footer = navigation_footer(
            "[↑/↓] каркас  [Enter] открыть  [PgUp/PgDn] страница  [Esc] назад",
            width=width,
        )
        viewport_height = max(1, height - len(header) - len(footer))
        rows, spans = self._rows(width=width)
        selected_start, selected_end = spans[self.cursor]
        max_offset = max(0, len(rows) - viewport_height)
        if selected_start < self.offset:
            self.offset = selected_start
        elif selected_end >= self.offset + viewport_height:
            self.offset = selected_end - viewport_height + 1
        self.offset = min(max(0, self.offset), max_offset)
        visible = rows[self.offset : self.offset + viewport_height]
        visible.extend([("", "")] * (viewport_height - len(visible)))
        return _compose(
            header,
            visible,
            footer,
            width=width,
            total=len(rows),
            offset=self.offset,
        )

    def _rows(
        self,
        *,
        width: int,
    ) -> tuple[list[tuple[str, str]], list[tuple[int, int]]]:
        rows: list[tuple[str, str]] = []
        spans: list[tuple[int, int]] = []
        labels = {
            None: ("без решения", "class:warning"),
            "selected": ("в работу", "class:warning"),
            "excluded": ("исключён", "class:excluded"),
        }
        for index, record in enumerate(self.records):
            start = len(rows)
            decision, decision_style = labels.get(
                record.payload.get("decision"),
                ("неизвестно", "class:error"),
            )
            marker = ">" if index == self.cursor else " "
            title = str(record.payload.get("title") or record.record_id)
            prefix = f"{marker} [{decision}] "
            wrapped = textwrap.wrap(
                title,
                width=max(1, width - 1 - len(prefix)),
                replace_whitespace=False,
                break_long_words=True,
                break_on_hyphens=False,
            ) or [""]
            style = "class:selected" if index == self.cursor else decision_style
            rows.append((style, prefix + wrapped[0]))
            rows.extend(
                (style, " " * len(prefix) + line)
                for line in wrapped[1:]
            )
            condition = str(record.payload.get("condition") or "не указано")
            details = [
                f"  Условие: {condition}",
                f"  Обязательные последствия: {len(record.payload.get('consequences', []))}",
                f"  Пробелы каркаса: {len(record.payload.get('gaps', []))}",
            ]
            for value in details:
                rows.extend(
                    (
                        "class:muted",
                        line,
                    )
                    for line in (
                        textwrap.wrap(
                            value,
                            width=max(1, width - 1),
                            replace_whitespace=False,
                            break_long_words=True,
                            break_on_hyphens=False,
                        )
                        or [""]
                    )
                )
            spans.append((start, len(rows) - 1))
        return rows, spans

    def _formatted_text(self) -> FormattedText:
        size = get_app().output.get_size()
        return self.render(width=size.columns, height=size.rows)

    def _bindings(self) -> KeyBindings:
        bindings = KeyBindings()

        def move(delta: int) -> None:
            self.cursor = min(max(0, self.cursor + delta), len(self.records) - 1)

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
            size = event.app.output.get_size()  # type: ignore[attr-defined]
            move(-max(1, size.rows // 4))

        @bindings.add("pagedown")
        def page_down(event: object) -> None:
            size = event.app.output.get_size()  # type: ignore[attr-defined]
            move(max(1, size.rows // 4))

        @bindings.add("enter")
        def enter(event: object) -> None:
            event.app.exit(result=self.records[self.cursor].record_id)  # type: ignore[attr-defined]

        @bindings.add("escape")
        def escape(event: object) -> None:
            event.app.exit(result=None)  # type: ignore[attr-defined]

        return bindings


class SkeletonDetailScreen:
    def __init__(
        self,
        record: StoredRecord,
        *,
        mode_label: str | None = None,
        input: Input | None = None,
        output: Output | None = None,
    ) -> None:
        self.record = record
        self.mode_label = mode_label
        self.input = input
        self.output = output
        self.cursor = 0
        self.offset = 0
        self.follow_cursor = False
        self.reason_mode = False
        self.reason = ""

    @property
    def actions(self) -> tuple[tuple[str, str], ...]:
        if (
            self.record.payload.get("decision") == "selected"
            and self.record.payload.get("card_id")
        ):
            return (
                ("open_session", "Открыть сессию"),
                ("back", "Назад к каркасам"),
            )
        if self.record.payload.get("decision") is not None:
            return (("back", "Назад к каркасам"),)
        return (
            ("take", "Взять в работу"),
            ("exclude", "Исключить"),
            ("back", "Назад к каркасам"),
        )

    def run(self) -> SkeletonDetailResult:
        application: Application[SkeletonDetailResult] = Application(
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
        payload = self.record.payload
        title = str(payload.get("title") or self.record.record_id)
        section_number = str(payload.get("section_number") or "")
        header_source = [
                (
                    "class:breadcrumb",
                    f"PMI Workbench / {section_number} / Каркасы / {title}",
                ),
        ]
        if self.mode_label:
            header_source.append(("class:warning", self.mode_label))
        header_source.append(("", ""))
        header = wrap_styled_lines(
            header_source,
            width=width,
        )
        rows = _detail_rows(payload, width=width)
        rows.extend([("", ""), ("class:breadcrumb", "Действия:")])
        action_spans: list[tuple[int, int]] = []
        for index, (_, label) in enumerate(self.actions):
            start = len(rows)
            marker = ">" if index == self.cursor else " "
            style = "class:selected" if index == self.cursor else "class:action"
            wrapped = textwrap.wrap(label, width=max(1, width - 3)) or [""]
            rows.append((style, f"{marker} {wrapped[0]}"))
            rows.extend((style, f"  {line}") for line in wrapped[1:])
            action_spans.append((start, len(rows) - 1))
        if self.reason_mode:
            rows.extend(
                [
                    ("", ""),
                    ("class:warning", "Причина исключения:"),
                    ("class:selected", f"> {self.reason}_"),
                ]
            )

        footer_source = (
            "[Enter] сохранить  [Backspace] удалить  [Esc] отменить"
            if self.reason_mode
            else "[↑/↓] выбор  [PgUp/PgDn] прокрутка  [Enter] выполнить  [Esc] назад"
        )
        footer = navigation_footer(
            footer_source,
            width=width,
        )
        viewport_height = max(1, height - len(header) - len(footer))
        if self.reason_mode:
            selected_start = selected_end = len(rows) - 1
        else:
            selected_start, selected_end = action_spans[self.cursor]
        max_offset = max(0, len(rows) - viewport_height)
        if self.follow_cursor or self.reason_mode:
            if selected_start < self.offset:
                self.offset = selected_start
            elif selected_end >= self.offset + viewport_height:
                self.offset = selected_end - viewport_height + 1
        self.offset = min(max(0, self.offset), max_offset)
        visible = rows[self.offset : self.offset + viewport_height]
        visible.extend([("", "")] * (viewport_height - len(visible)))
        return _compose(
            header,
            visible,
            footer,
            width=width,
            total=len(rows),
            offset=self.offset,
        )

    def _formatted_text(self) -> FormattedText:
        size = get_app().output.get_size()
        return self.render(width=size.columns, height=size.rows)

    def _bindings(self) -> KeyBindings:
        bindings = KeyBindings()
        navigating = Condition(lambda: not self.reason_mode)
        entering_reason = Condition(lambda: self.reason_mode)

        def move(delta: int) -> None:
            self.cursor = min(max(0, self.cursor + delta), len(self.actions) - 1)
            self.follow_cursor = True

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
            del event
            self.follow_cursor = False
            self.offset = max(0, self.offset - 10)

        @bindings.add("pagedown", filter=navigating)
        def page_down(event: object) -> None:
            del event
            self.follow_cursor = False
            self.offset += 10

        @bindings.add("enter")
        def enter(event: object) -> None:
            if self.reason_mode:
                if self.reason.strip():
                    event.app.exit(  # type: ignore[attr-defined]
                        result=SkeletonDetailResult("exclude", self.reason.strip())
                    )
                return
            action = self.actions[self.cursor][0]
            if action == "exclude":
                self.reason_mode = True
                self.follow_cursor = True
                return
            event.app.exit(result=SkeletonDetailResult(action))  # type: ignore[attr-defined]

        @bindings.add("backspace", filter=entering_reason)
        @bindings.add("c-h", filter=entering_reason)
        def backspace(event: object) -> None:
            del event
            self.reason = self.reason[:-1]

        @bindings.add("<any>", filter=entering_reason)
        def insert(event: object) -> None:
            data = event.data  # type: ignore[attr-defined]
            if data and data.isprintable():
                self.reason += data

        @bindings.add("escape")
        def escape(event: object) -> None:
            if self.reason_mode:
                self.reason_mode = False
                self.reason = ""
                return
            event.app.exit(result=SkeletonDetailResult("back"))  # type: ignore[attr-defined]

        return bindings


def _detail_rows(
    payload: dict[str, object],
    *,
    width: int,
) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []

    def block(label: str, values: list[str]) -> None:
        if rows:
            rows.append(("", ""))
        rows.append(("class:breadcrumb", f"{label}:"))
        for value in values or ["Не определено"]:
            wrapped = textwrap.wrap(
                value,
                width=max(1, width - 3),
                replace_whitespace=False,
                break_long_words=True,
                break_on_hyphens=False,
            ) or [""]
            rows.append(("", f"  {wrapped[0]}"))
            rows.extend(("", f"  {line}") for line in wrapped[1:])

    block("Проверяемое условие", [str(payload.get("condition") or "Не определено")])
    block("Изменяемый фактор", [str(payload.get("changed_factor") or "Не определено")])
    input_value = payload.get("input_value")
    block(
        "Конкретное значение",
        [
            str(input_value)
            if input_value is not None
            else "Не определено — требуется решение"
        ],
    )
    action = payload.get("action")
    block(
        "Воздействие",
        [str(action) if action is not None else "Не определено — требуется решение"],
    )
    consequences = [
        f"{index}. {item.get('text', '')}"
        for index, item in enumerate(payload.get("consequences", []), start=1)
        if isinstance(item, dict)
    ]
    block("Обязательные последствия", consequences)
    sources = []
    for evidence in payload.get("condition_evidence", []):
        if not isinstance(evidence, dict):
            continue
        sources.append(
            (
                f"стр. {evidence.get('page')}:{int(evidence.get('line_start', 0)):03d}–"
                f"{int(evidence.get('line_end', 0)):03d}"
            )
        )
    block("Источник", sources)
    gaps = [
        str(item.get("question") or item.get("kind") or "")
        for item in payload.get("gaps", [])
        if isinstance(item, dict)
    ]
    if gaps:
        block("Пробелы каркаса", gaps)
    return rows


def _compose(
    header: list[tuple[str, str]],
    visible: list[tuple[str, str]],
    footer: list[tuple[str, str]],
    *,
    width: int,
    total: int,
    offset: int,
) -> FormattedText:
    thumb_start, thumb_size = scrollbar(
        total=total,
        visible=len(visible),
        offset=offset,
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
