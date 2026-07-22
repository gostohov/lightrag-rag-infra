from __future__ import annotations

import textwrap

from ...application.range_workspace import RangeWorkspaceController
from ..navigation import render_scrollbar


def render_range_workspace(
    controller: RangeWorkspaceController,
    *,
    width: int,
    height: int,
) -> str:
    state = controller.state
    width = max(24, width)
    rows: list[str] = []
    for index, item in enumerate(state.items):
        cursor = ">" if controller.cursor == index else " "
        prefix = f"{cursor} "
        value = f"{item.title}  [{item.status}]"
        wrapped = textwrap.wrap(value, width=max(8, width - len(prefix) - 1)) or [""]
        rows.append(prefix + wrapped[0])
        rows.extend("  " + part for part in wrapped[1:])
    if state.can_review:
        index = len(state.items)
        cursor = ">" if controller.cursor == index else " "
        rows.append(f"{cursor} Проверить выбранный диапазон")
    elif not state.items and state.terminal_status is None:
        cursor = ">" if controller.cursor == 0 else " "
        rows.append(f"{cursor} Построить каркасы карточек")
    body = render_scrollbar(rows, offset=controller.offset, width=width, height=height)
    review = "актуальна" if state.review_current else "устарела" if state.review_stale else "не выполнена"
    header = [
        "PMI Workbench / Карточки выбранного диапазона",
        f"Каркасов: {len(state.items)}  Включено: {state.included}  Неполных: {state.included_incomplete}  Исключено: {state.excluded}",
        f"Проверка диапазона: {review}",
        "",
    ]
    footer = ["", "[↑/↓] карточка или действие  [Enter] открыть  [PgUp/PgDn] страница  [Esc] назад"]
    return "\n".join([*header, *body, *footer])
