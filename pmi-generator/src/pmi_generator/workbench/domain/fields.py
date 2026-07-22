from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .enums import EpistemicStatus
from .errors import DomainValidationError


@dataclass(frozen=True, slots=True)
class ContentField:
    status: EpistemicStatus
    value: Any = None
    evidence_ids: tuple[str, ...] = ()
    derivation_id: str | None = None
    reason: str | None = None

    @classmethod
    def unknown(cls) -> ContentField:
        return cls(status=EpistemicStatus.UNKNOWN)

    @classmethod
    def not_applicable(cls, reason: str) -> ContentField:
        if not reason.strip():
            raise DomainValidationError("Для неприменимого поля нужна причина")
        return cls(status=EpistemicStatus.NOT_APPLICABLE, reason=reason)

    @classmethod
    def confirmed(cls, value: Any, evidence_ids: tuple[str, ...]) -> ContentField:
        if not evidence_ids:
            raise DomainValidationError("Подтверждённое поле требует evidence")
        return cls(
            status=EpistemicStatus.SOURCE_CONFIRMED,
            value=value,
            evidence_ids=tuple(dict.fromkeys(evidence_ids)),
        )

    @classmethod
    def analyst_confirmed(
        cls,
        value: Any,
        evidence_ids: tuple[str, ...],
    ) -> ContentField:
        if not evidence_ids:
            raise DomainValidationError(
                "Подтверждённое аналитиком поле требует evidence"
            )
        return cls(
            status=EpistemicStatus.ANALYST_CONFIRMED,
            value=value,
            evidence_ids=tuple(dict.fromkeys(evidence_ids)),
        )

    @classmethod
    def derived(cls, value: Any, derivation_id: str) -> ContentField:
        if not derivation_id.strip():
            raise DomainValidationError("Выведенное поле требует ID вывода")
        return cls(
            status=EpistemicStatus.DERIVED,
            value=value,
            derivation_id=derivation_id,
        )

    @property
    def is_known(self) -> bool:
        return self.status is not EpistemicStatus.UNKNOWN


@dataclass(frozen=True, slots=True)
class Derivation:
    derivation_id: str
    card_id: str
    source_evidence_ids: tuple[str, ...]
    rule: str
    scope: str

    def __post_init__(self) -> None:
        if not self.derivation_id.strip() or not self.card_id.strip():
            raise DomainValidationError("Вывод должен иметь ID и card_id")
        if not self.source_evidence_ids:
            raise DomainValidationError("Вывод должен иметь подтверждённые основания")
        if not self.rule.strip() or not self.scope.strip():
            raise DomainValidationError("Вывод должен иметь правило и область применимости")
