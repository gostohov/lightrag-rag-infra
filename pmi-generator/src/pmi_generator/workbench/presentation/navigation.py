from __future__ import annotations

import textwrap

from prompt_toolkit.styles import Style


NAVIGATION_STYLE = Style.from_dict(
    {
        "breadcrumb": "fg:ansicyan bold",
        "selected": "reverse",
        "cursor": "reverse",
        "section": "",
        "source.heading": "fg:ansicyan bold",
        "source.heading.cursor": "fg:ansicyan bold reverse",
        "selection": "fg:ansiblue",
        "selection.cursor": "fg:ansiblue reverse",
        "range.active": "fg:ansiyellow",
        "range.active.cursor": "fg:ansiyellow reverse",
        "range.completed": "fg:ansigreen",
        "range.completed.cursor": "fg:ansigreen reverse",
        "range.terminal": "fg:ansibrightblack",
        "range.terminal.cursor": "fg:ansibrightblack reverse",
        "action": "",
        "success": "fg:ansigreen",
        "warning": "fg:ansiyellow",
        "error": "fg:ansired",
        "user": "fg:ansiwhite bg:ansibrightblack",
        "excluded": "fg:ansibrightblack",
        "muted": "fg:ansibrightblack",
        "footer": "fg:ansiwhite",
        "search": "fg:ansiblue",
        "scrollbar.track": "fg:ansibrightblack",
        "scrollbar.thumb": "fg:ansiblue bold",
    }
)


def scrollbar(*, total: int, visible: int, offset: int) -> tuple[int, int]:
    visible = max(1, visible)
    if total <= visible:
        return 0, visible
    max_offset = max(1, total - visible)
    thumb_size = max(1, round(visible * visible / total))
    thumb_size = min(visible, thumb_size)
    thumb_start = round(offset * (visible - thumb_size) / max_offset)
    return thumb_start, thumb_size


def render_scrollbar(
    rows: list[str],
    *,
    offset: int,
    width: int,
    height: int,
) -> list[str]:
    width = max(2, width)
    height = max(1, height)
    max_offset = max(0, len(rows) - height)
    offset = min(max(0, offset), max_offset)
    visible = rows[offset : offset + height]
    visible.extend("" for _ in range(height - len(visible)))
    thumb_start, thumb_size = scrollbar(
        total=len(rows),
        visible=height,
        offset=offset,
    )
    return [
        row[: width - 1].ljust(width - 1)
        + ("█" if thumb_start <= index < thumb_start + thumb_size else "│")
        for index, row in enumerate(visible)
    ]


def wrap_styled_lines(
    lines: list[tuple[str, str]],
    *,
    width: int,
) -> list[tuple[str, str]]:
    result: list[tuple[str, str]] = []
    for style, value in lines:
        if not value:
            result.append((style, ""))
            continue
        wrapped = textwrap.wrap(
            value,
            width=max(1, width),
            replace_whitespace=False,
            break_long_words=True,
            break_on_hyphens=False,
        ) or [""]
        result.extend((style, line) for line in wrapped)
    return result


def navigation_footer(
    text: str,
    *,
    width: int,
    compact_text: str | None = None,
) -> list[tuple[str, str]]:
    value = (
        compact_text
        if compact_text is not None and len(text) > max(1, width)
        else text
    )
    return wrap_styled_lines(
        [("class:footer", value)],
        width=width,
    )
