from __future__ import annotations

import io

from rich.console import Console
from rich.markdown import Markdown


def render_markdown_ansi(markdown: str, *, width: int) -> str:
    output = io.StringIO()
    console = Console(
        file=output,
        force_terminal=True,
        color_system="truecolor",
        width=max(20, width),
        height=25,
        soft_wrap=False,
        highlight=False,
    )
    console.print(Markdown(markdown, code_theme="monokai"))
    return output.getvalue().rstrip("\n")


__all__ = ["render_markdown_ansi"]
