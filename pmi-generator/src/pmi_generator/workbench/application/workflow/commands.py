from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class CommandKind(StrEnum):
    CONFIRM_SELECTION = "подтвердить диапазон"
    APPLY_DECOMPOSITION = "применить декомпозицию"
    TAKE_SKELETON = "взять каркас в работу"
    EXCLUDE_SKELETON = "исключить каркас"
    BEGIN_ATTEMPT = "начать попытку"
    CANCEL_ATTEMPT = "отменить попытку"
    FAIL_ATTEMPT = "завершить попытку ошибкой"
    APPLY_ATTEMPT_RESULT = "применить результат попытки"
    REQUEST_ANALYST = "запросить решение аналитика"
    REFINE_CARD = "доработать карточку"
    DECIDE_CARD = "сохранить решение по карточке"
    SAVE_RANGE_REVIEW = "сохранить проверку диапазона"
    CONTINUE_WITH_ISSUES = "продолжить с замечаниями"
    REQUEST_EXPORT = "разрешить экспорт"


@dataclass(frozen=True, slots=True)
class WorkflowCommand:
    kind: CommandKind
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind.value, "payload": self.payload}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> WorkflowCommand:
        return cls(kind=CommandKind(value["kind"]), payload=dict(value.get("payload", {})))
