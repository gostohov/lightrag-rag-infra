from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any


class SessionEventKind(StrEnum):
    WORKBENCH = "Workbench"
    ASSISTANT = "Ассистент"
    ANALYST = "Аналитик"
    OPERATION = "Операция"
    ERROR = "Ошибка"


@dataclass(frozen=True, slots=True)
class SessionEvent:
    sequence: int
    kind: SessionEventKind
    text: str
    created_at: datetime
    metadata: dict[str, Any]
