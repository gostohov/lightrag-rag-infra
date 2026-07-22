from __future__ import annotations

from .models import SessionEventKind
from .ports import SessionGateway


class SessionShellController:
    def __init__(self, service: SessionGateway, session_id: str) -> None:
        self.service = service
        self.session_id = session_id
        self.input_text = ""
        self.should_exit = False
        self.size = (80, 24)

    def escape(self) -> str:
        active = self.service.active_attempt(self.session_id)
        if active is not None:
            self.service.cancel_operation(self.session_id, active.attempt_id)
            return "operation_cancelled"
        if self.input_text:
            self.input_text = ""
            return "input_cleared"
        self.should_exit = True
        return "exit"

    def submit(self, text: str) -> str | None:
        if not text.strip():
            return None
        sequence = self.service.append(
            self.session_id,
            SessionEventKind.ANALYST,
            text,
            {"author": "Аналитик"},
        )
        self.input_text = ""
        return f"MSG_{sequence:06d}"

    def resize(self, width: int, height: int) -> None:
        self.size = (max(20, width), max(5, height))

    def handle_sequence(self, sequence: str) -> None:
        # Редактор строки обрабатывает Ctrl-комбинации; controller не трактует их как выход.
        return None
