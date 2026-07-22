from __future__ import annotations

from typing import Any, Protocol

from ..state import AttemptRecord
from .models import SessionEvent, SessionEventKind


class SessionGateway(Protocol):
    def append(
        self,
        session_id: str,
        kind: SessionEventKind,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> int: ...

    def history(self, session_id: str) -> list[SessionEvent]: ...

    def active_attempt(self, session_id: str) -> AttemptRecord | None: ...

    def cancel_operation(self, session_id: str, attempt_id: str) -> None: ...


__all__ = ["SessionGateway"]
