from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from typing import Any

from .enums import CardDecisionKind
from .errors import DomainValidationError


@dataclass(frozen=True, slots=True)
class AnalystResolution:
    resolution_id: str
    card_id: str
    author: str
    created_at: datetime
    reason: str
    target_paths: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    source_message_id: str | None = None
    confirmation_message_id: str | None = None
    proposal_id: str | None = None
    gap_id: str | None = None
    expected_revision: int | None = None
    values: tuple[dict[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if not self.resolution_id.strip() or not self.card_id.strip():
            raise DomainValidationError("Решение аналитика должно иметь ID и card_id")
        if not self.author.strip() or not self.reason.strip():
            raise DomainValidationError("Решение аналитика должно иметь автора и основание")
        if not self.target_paths or not self.evidence_ids:
            raise DomainValidationError("Решение аналитика должно иметь целевые поля и evidence")
        if self.expected_revision is not None and self.expected_revision < 0:
            raise DomainValidationError(
                "Решение аналитика содержит неверную ревизию"
            )
        if self.values:
            if any(
                not isinstance(item, dict)
                or set(item) != {"path", "value"}
                for item in self.values
            ):
                raise DomainValidationError(
                    "Решение аналитика содержит неверные значения"
                )
            value_paths = tuple(str(item["path"]) for item in self.values)
            if (
                len(value_paths) != len(set(value_paths))
                or set(value_paths) != set(self.target_paths)
            ):
                raise DomainValidationError(
                    "Значения решения не совпадают с целевыми полями"
                )


@dataclass(frozen=True, slots=True)
class CardDecision:
    kind: CardDecisionKind
    card_id: str
    revision: int
    author: str
    created_at: datetime
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.revision < 0 or not self.author.strip():
            raise DomainValidationError("Решение по карточке должно иметь ревизию и автора")
        requires_reason = self.kind in {
            CardDecisionKind.INCLUDE_INCOMPLETE,
            CardDecisionKind.EXCLUDE,
        }
        if requires_reason and not (self.reason or "").strip():
            raise DomainValidationError("Решение по неполной или исключённой карточке требует основания")
