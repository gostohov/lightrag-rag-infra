from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from ...domain import Evidence, SourceAddress


class GapInvestigationError(ValueError):
    pass


class TechnicalRetrievalError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class AnalystConfirmation:
    proposal_id: str
    source_message_id: str
    confirmation_message_id: str
    gap_id: str
    expected_revision: int
    values: tuple[dict[str, Any], ...]

    def __post_init__(self) -> None:
        identifiers = (
            self.proposal_id,
            self.source_message_id,
            self.confirmation_message_id,
            self.gap_id,
        )
        if any(not value.strip() for value in identifiers):
            raise GapInvestigationError(
                "Подтверждение ответа аналитика требует полные ID"
            )
        if self.expected_revision < 0 or not self.values:
            raise GapInvestigationError(
                "Подтверждение ответа аналитика имеет неверный контекст"
            )
        for item in self.values:
            if not isinstance(item, dict) or set(item) != {"path", "value"}:
                raise GapInvestigationError(
                    "Подтверждённые значения имеют неверную структуру"
                )


@dataclass(frozen=True, slots=True)
class RetrievalProfile:
    name: str
    kg_top_k: int
    chunk_top_k: int
    max_entity_tokens: int
    max_relation_tokens: int
    max_total_tokens: int


@dataclass(frozen=True, slots=True)
class RetrievalFragment:
    document_id: str | None
    document_version: str | None
    page: int | None
    line_start: int | None
    line_end: int | None
    chunk_id: str | None
    quote: str | None

    @property
    def is_exact(self) -> bool:
        return bool(
            self.document_id
            and self.document_version
            and self.page
            and self.line_start
            and self.line_end
            and self.quote
        )

    def to_evidence(
        self,
        evidence_id: str,
        card_id: str,
        selection_id: str,
        collected_at: datetime,
    ) -> Evidence:
        if not self.is_exact:
            raise GapInvestigationError("Фрагмент без точного адреса не является evidence")
        return Evidence.source_fragment(
            evidence_id=evidence_id,
            card_id=card_id,
            selection_id=selection_id,
            quote=str(self.quote),
            address=SourceAddress(
                document_id=str(self.document_id),
                document_version=str(self.document_version),
                page=int(self.page),
                line_start=int(self.line_start),
                line_end=int(self.line_end),
                chunk_id=self.chunk_id,
            ),
            collected_at=collected_at,
        )


@dataclass(frozen=True, slots=True)
class RetrievalResponse:
    answer: str
    fragments: tuple[RetrievalFragment, ...]


@dataclass(frozen=True, slots=True)
class RetrievalCall:
    call_id: str
    question: str
    profile: RetrievalProfile
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class RetrievalObservation:
    call_id: str
    question: str
    profile_name: str
    answer: str
    evidence_ids: tuple[str, ...]
    duration_seconds: float

    def as_context(self) -> dict[str, Any]:
        return {
            "call_id": self.call_id,
            "question": self.question,
            "profile": self.profile_name,
            "answer": self.answer,
            "evidence_ids": list(self.evidence_ids),
        }


@dataclass(frozen=True, slots=True)
class AskLightRagArguments:
    question: str

    def __post_init__(self) -> None:
        value = self.question.strip()
        if not value or len(value) > 500 or "\n" in value:
            raise GapInvestigationError("LightRAG-вопрос должен быть одним коротким вопросом")


@dataclass(frozen=True, slots=True)
class ExpandLightRagArguments:
    call_id: str
    reason: str

    def __post_init__(self) -> None:
        if not self.call_id.strip() or not self.reason.strip():
            raise GapInvestigationError("Расширение требует ID вызова и причину")


@dataclass(slots=True)
class GapArguments:
    outcome: str
    updates: list[dict[str, Any]]
    unknown_fields: list[str]
    missing_fact: str | dict[str, object] | None
    summary: str
    contradictions: list[dict[str, Any]]

    def __post_init__(self) -> None:
        if isinstance(self.missing_fact, str):
            self.missing_fact = self.missing_fact.strip()
        elif self.missing_fact is not None:
            if not isinstance(self.missing_fact, dict) or set(
                self.missing_fact
            ) != {"field", "description"}:
                raise GapInvestigationError(
                    "Недостающий факт имеет неверную структуру"
                )
            field = self.missing_fact["field"]
            description = self.missing_fact["description"]
            if not isinstance(field, str) or not field.strip():
                raise GapInvestigationError("Недостающий факт требует путь поля")
            if not isinstance(description, str) or not description.strip():
                raise GapInvestigationError("Недостающий факт требует описание")
            if field not in self.unknown_fields:
                raise GapInvestigationError(
                    "Путь недостающего факта отсутствует в неизвестных полях"
                )
            self.missing_fact = description.strip()

        if self.outcome == "resolved":
            if (
                not self.updates
                or self.unknown_fields
                or self.missing_fact is not None
                or self.contradictions
            ):
                raise GapInvestigationError(
                    "Исход resolved требует только подтверждённые обновления"
                )
        elif self.outcome == "partially_resolved":
            if (
                not self.updates
                or not self.unknown_fields
                or not self.missing_fact
                or self.contradictions
            ):
                raise GapInvestigationError(
                    "Исход partially_resolved требует обновления и "
                    "остаточный неизвестный факт"
                )
        elif self.outcome == "not_found":
            if (
                self.updates
                or not self.unknown_fields
                or not self.missing_fact
                or self.contradictions
            ):
                raise GapInvestigationError(
                    "Исход not_found требует только неизвестный факт и поля"
                )
        elif self.outcome == "contradiction":
            if self.updates or self.missing_fact is not None or len(
                self.contradictions
            ) < 2:
                raise GapInvestigationError(
                    "Исход contradiction требует минимум два противоречия"
                )
        else:
            raise GapInvestigationError("Неизвестный исход исследования")


@dataclass(frozen=True, slots=True)
class GapInvestigationResult:
    card_id: str
    gap_id: str
    outcome: str
    revision: int
    observations: int = 0
    remaining_questions: tuple[str, ...] = ()
