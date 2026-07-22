from __future__ import annotations

from dataclasses import dataclass

from ...domain.source import SourcePosition, TextSelection


@dataclass(slots=True)
class SourceNavigationState:
    outline_cursor: int = 0
    outline_offset: int = 0
    outline_query: str = ""
    anchor_outline_node_id: str | None = None
    canvas_cursor: SourcePosition | None = None
    canvas_scroll_anchor: SourcePosition | None = None
    selection_anchor: SourcePosition | None = None
    draft_selection: TextSelection | None = None
    selection_complete: bool = False
    canvas_query: str = ""

    def clear_draft(self) -> None:
        self.selection_anchor = None
        self.draft_selection = None
        self.selection_complete = False
