from __future__ import annotations

from .models import RangeWorkspaceState, WorkspaceItem
from .service import RangeWorkspaceService


class RangeWorkspaceController:
    def __init__(
        self,
        service: RangeWorkspaceService,
        selection_id: str,
        *,
        viewport_height: int = 20,
    ) -> None:
        self.service = service
        self.selection_id = selection_id
        self.viewport_height = max(1, viewport_height)
        self.cursor = 0
        self.offset = 0
        self._return_position: tuple[int, int] | None = None

    @property
    def state(self) -> RangeWorkspaceState:
        return self.service.load(self.selection_id)

    @property
    def rows_count(self) -> int:
        state = self.state
        can_decompose = not state.items and state.terminal_status is None
        return len(state.items) + int(state.can_review or can_decompose)

    @property
    def current_item(self) -> WorkspaceItem | None:
        state = self.state
        return state.items[self.cursor] if self.cursor < len(state.items) else None

    def move(self, delta: int) -> None:
        self.cursor = min(max(0, self.cursor + delta), max(0, self.rows_count - 1))
        if self.cursor < self.offset:
            self.offset = self.cursor
        if self.cursor >= self.offset + self.viewport_height:
            self.offset = self.cursor - self.viewport_height + 1

    def page(self, delta: int) -> None:
        self.move(delta * self.viewport_height)

    def leave_for_session(self) -> None:
        self._return_position = (self.cursor, self.offset)

    def return_from_session(self) -> None:
        if self._return_position is not None:
            self.cursor, self.offset = self._return_position

    def activate(self) -> tuple[str, str]:
        item = self.current_item
        if item is None:
            if not self.state.items and self.state.terminal_status is None:
                return "decompose", self.selection_id
            if self.state.can_review:
                return "review", self.selection_id
            raise ValueError("Проверка диапазона пока недоступна")
        if item.card_id is None:
            return "skeleton", item.skeleton_id
        self.leave_for_session()
        if item.session_id:
            return "session", item.session_id
        return "card", item.card_id
