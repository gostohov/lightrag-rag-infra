from __future__ import annotations

import textwrap
from dataclasses import dataclass, field
from typing import TextIO

from prompt_toolkit.formatted_text import ANSI, FormattedText, to_formatted_text
from prompt_toolkit.formatted_text.utils import split_lines

from ...application.conversation import user_facing_conversation_text
from ...application.session import SessionEvent, SessionEventKind
from .markdown import render_markdown_ansi


ANSI_RESET = "\x1b[0m"
ANSI_ASSISTANT = "\x1b[38;5;255m"
ANSI_ERROR = "\x1b[38;5;203m"
ANSI_MUTED = "\x1b[38;5;245m"
ANSI_OPERATION_ACTIVE = "\x1b[38;5;221m"
ANSI_OPERATION_DONE = "\x1b[38;5;83m"
ANSI_TITLE = "\x1b[1;38;5;39m"
ANSI_USER = "\x1b[38;5;252;48;5;236m"


@dataclass(slots=True)
class SessionRenderState:
    rendered_sequences: set[int] = field(default_factory=set)


class AppendOnlySessionRenderer:
    def __init__(
        self,
        output: TextIO,
        *,
        width: int,
        state: SessionRenderState | None = None,
    ) -> None:
        self.output = output
        self.width = max(20, width)
        self.state = state or SessionRenderState()

    def resize(self, width: int) -> None:
        self.width = max(20, width)

    def render(self, events: list[SessionEvent]) -> None:
        visible_sequences = self._visible_sequences(events)
        for event in events:
            if event.sequence in self.state.rendered_sequences:
                continue
            if event.sequence not in visible_sequences:
                self.state.rendered_sequences.add(event.sequence)
                continue
            self.output.write(self._block(event))
            self.state.rendered_sequences.add(event.sequence)
        self.output.flush()

    def _block(self, event: SessionEvent) -> str:
        separator = "─" * max(1, self.width - 1)
        display_text = _event_display_text(event)
        wrapped_lines = _wrap_text(display_text, width=max(10, self.width - 4))
        wrapped = "\n".join(wrapped_lines)
        if event.kind is SessionEventKind.ANALYST:
            timestamp = event.created_at.astimezone().strftime("%H:%M")
            user_lines = [f"Аналитик · {timestamp}", "", *wrapped_lines]
            padded = [
                f"  {line}".ljust(max(1, self.width - 1))
                for line in user_lines
            ]
            body = f"{ANSI_USER}{chr(10).join(padded)}{ANSI_RESET}"
        elif event.kind is SessionEventKind.ERROR:
            body = f"{ANSI_ERROR}{event.kind.value}\n{wrapped}{ANSI_RESET}"
        elif event.kind is SessionEventKind.OPERATION:
            color = (
                ANSI_OPERATION_DONE
                if _operation_completed(event.text)
                else ANSI_OPERATION_ACTIVE
            )
            markdown_parts = _operation_markdown_parts(event)
            if markdown_parts is None:
                body = f"{color}{event.kind.value}\n{wrapped}{ANSI_RESET}"
            else:
                header, markdown = markdown_parts
                wrapped_header = "\n".join(
                    _wrap_text(header, width=max(10, self.width - 4))
                )
                try:
                    rendered_body = render_markdown_ansi(
                        markdown,
                        width=max(20, self.width - 4),
                    )
                except Exception:
                    rendered_body = "\n".join(
                        _wrap_text(markdown, width=max(10, self.width - 4))
                    )
                body = (
                    f"{color}{event.kind.value}\n{wrapped_header}{ANSI_RESET}\n"
                    f"{rendered_body}"
                )
        elif event.kind is SessionEventKind.WORKBENCH:
            color = (
                ANSI_MUTED
                if "прерван" in event.text.casefold()
                else ANSI_TITLE
            )
            body = f"{color}{event.kind.value}\n{wrapped}{ANSI_RESET}"
        elif _render_as_markdown(event):
            try:
                markdown = render_markdown_ansi(
                    display_text,
                    width=max(20, self.width - 4),
                )
            except Exception:
                markdown = wrapped
            body = (
                f"{ANSI_ASSISTANT}{event.kind.value}{ANSI_RESET}\n"
                f"{markdown}"
            )
        else:
            body = f"{ANSI_ASSISTANT}{event.kind.value}\n{wrapped}{ANSI_RESET}"
        return f"\n{separator}\n\n{body}\n"

    @staticmethod
    def _visible_sequences(events: list[SessionEvent]) -> set[int]:
        latest_terminal: dict[tuple[str, str], int] = {}
        for event in events:
            attempt_id = str(event.metadata.get("attempt_id") or "")
            if not attempt_id:
                continue
            call_id = str(event.metadata.get("call_id") or "")
            key = attempt_id, call_id
            if event.kind in {
                SessionEventKind.ERROR,
                SessionEventKind.WORKBENCH,
            } or _operation_completed(event.text):
                latest_terminal[key] = event.sequence
        visible: set[int] = set()
        for event in events:
            attempt_id = str(event.metadata.get("attempt_id") or "")
            call_id = str(event.metadata.get("call_id") or "")
            key = attempt_id, call_id
            if (
                event.kind is SessionEventKind.OPERATION
                and "статус: выполняется" in event.text.casefold()
                and key in latest_terminal
                and latest_terminal[key] > event.sequence
            ):
                continue
            visible.add(event.sequence)
        return visible


def render_session_context(
    events: list[SessionEvent],
    *,
    width: int,
    height: int,
    breadcrumb: str = "PMI Workbench / Сессия подготовки карточки",
    mode_label: str | None = None,
) -> str:
    width = max(20, width)
    height = max(1, height)
    header = _wrap_text(breadcrumb, width=width)
    if mode_label:
        header.extend(_wrap_text(mode_label, width=width))
    visible_sequences = AppendOnlySessionRenderer._visible_sequences(events)
    body: list[str] = []
    separator = "─" * max(1, width - 1)
    for event in events:
        if event.sequence not in visible_sequences:
            continue
        body.extend(
            [
                separator,
                "",
                event.kind.value,
                *_wrap_text(
                    _event_display_text(event),
                    width=max(10, width - 4),
                ),
            ]
        )
    available = max(0, height - len(header))
    visible_body = body[-available:] if available else []
    return "\n".join([*header, *visible_body])


def render_session_context_fragments(
    events: list[SessionEvent],
    *,
    width: int,
    height: int,
    breadcrumb: str = "PMI Workbench / Сессия подготовки карточки",
    mode_label: str | None = None,
) -> FormattedText:
    width = max(20, width)
    height = max(1, height)
    header_rows: list[list[tuple[str, str]]] = []
    header_rows.extend(
        [("class:breadcrumb", f"{line}\n")]
        for line in _wrap_text(breadcrumb, width=width)
    )
    if mode_label:
        header_rows.extend(
            [("class:warning", f"{line}\n")]
            for line in _wrap_text(mode_label, width=width)
        )
    visible_sequences = AppendOnlySessionRenderer._visible_sequences(events)
    body_rows: list[list[tuple[str, str]]] = []
    separator = "─" * max(1, width - 1)
    for event in events:
        if event.sequence not in visible_sequences:
            continue
        style = _event_fragment_style(event)
        body_rows.append([("class:muted", f"{separator}\n")])
        body_rows.append([("", "\n")])
        body_rows.append([(style, f"{event.kind.value}\n")])
        operation_markdown = _operation_markdown_parts(event)
        if operation_markdown is not None:
            header, markdown = operation_markdown
            body_rows.extend(
                [(style, f"{line}\n")]
                for line in _wrap_text(header, width=max(10, width - 4))
            )
            body_rows.extend(
                _markdown_context_rows(
                    markdown,
                    width=max(20, width - 4),
                )
            )
            continue
        if _render_as_markdown(event):
            body_rows.extend(
                _markdown_context_rows(
                    _event_display_text(event),
                    width=max(20, width - 4),
                )
            )
            continue
        body_rows.extend(
            [(style, f"{line}\n")]
            for line in _wrap_text(
                _event_display_text(event),
                width=max(10, width - 4),
            )
        )
    available = max(0, height - len(header_rows))
    visible_body = body_rows[-available:] if available else []
    return FormattedText(
        [
            fragment
            for row in [*header_rows, *visible_body]
            for fragment in row
        ]
    )


def _markdown_context_rows(
    markdown: str,
    *,
    width: int,
) -> list[list[tuple[str, str]]]:
    try:
        rendered = render_markdown_ansi(markdown, width=width)
        formatted_lines = split_lines(to_formatted_text(ANSI(rendered)))
        rows: list[list[tuple[str, str]]] = []
        for fragments in formatted_lines:
            row = [(style, text) for style, text, *_ in fragments]
            row.append(("", "\n"))
            rows.append(row)
        return rows
    except Exception:
        return [
            [("", f"{line}\n")]
            for line in _wrap_text(markdown, width=width)
        ]


def _event_fragment_style(event: SessionEvent) -> str:
    if event.kind is SessionEventKind.ERROR:
        return "class:error"
    if event.kind is SessionEventKind.OPERATION:
        return (
            "class:success"
            if _operation_completed(event.text)
            else "class:warning"
        )
    if event.kind is SessionEventKind.WORKBENCH:
        return (
            "class:muted"
            if "прерван" in event.text.casefold()
            else "class:breadcrumb"
        )
    if event.kind is SessionEventKind.ANALYST:
        return "class:user"
    return ""


def _render_as_markdown(event: SessionEvent) -> bool:
    return bool(
        event.metadata.get("card_snapshot")
        or event.metadata.get("conversation_response")
    )


def _event_display_text(event: SessionEvent) -> str:
    if (
        event.metadata.get("conversation_response")
        or event.metadata.get("conversation_action")
    ):
        return user_facing_conversation_text(event.text)
    return event.text


def _operation_markdown_parts(
    event: SessionEvent,
) -> tuple[str, str] | None:
    if (
        event.kind is not SessionEventKind.OPERATION
        or not event.metadata.get("lightrag_result")
    ):
        return None
    start = event.metadata.get("markdown_body_start_line")
    if not isinstance(start, int) or start < 1:
        return None
    lines = event.text.splitlines()
    if start >= len(lines):
        return None
    return "\n".join(lines[:start]), "\n".join(lines[start:])


def _operation_completed(text: str) -> bool:
    normalized = text.casefold()
    return (
        "статус: заверш" in normalized
        or "исход: resolved" in normalized
    )


def _wrap_text(value: str, *, width: int) -> list[str]:
    result: list[str] = []
    for source_line in value.splitlines() or [""]:
        if not source_line:
            result.append("")
            continue
        indentation = len(source_line) - len(source_line.lstrip(" "))
        prefix = source_line[:indentation]
        wrapped = textwrap.wrap(
            source_line[indentation:],
            width=max(1, width - indentation),
            replace_whitespace=False,
            break_long_words=True,
            break_on_hyphens=False,
        ) or [""]
        result.extend(prefix + line for line in wrapped)
    return result
