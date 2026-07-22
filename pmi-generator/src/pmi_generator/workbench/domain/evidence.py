from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from .enums import EvidenceKind, EvidenceScope
from .errors import DomainValidationError


@dataclass(frozen=True, slots=True)
class SourceAddress:
    document_id: str
    document_version: str
    page: int
    line_start: int
    line_end: int
    chunk_id: str | None = None

    def __post_init__(self) -> None:
        if not self.document_id.strip() or not self.document_version.strip():
            raise DomainValidationError("Источник должен иметь имя и версию")
        if self.page < 1 or self.line_start < 1 or self.line_end < self.line_start:
            raise DomainValidationError("Некорректные координаты фрагмента источника")


@dataclass(frozen=True, slots=True)
class Evidence:
    evidence_id: str
    kind: EvidenceKind
    scope: EvidenceScope
    selection_id: str
    quote: str
    collected_at: datetime
    card_id: str | None = None
    address: SourceAddress | None = None
    author: str | None = None
    message_id: str | None = None

    def __post_init__(self) -> None:
        if not self.evidence_id.strip() or not self.selection_id.strip() or not self.quote.strip():
            raise DomainValidationError("Evidence должно иметь ID, selection и точное утверждение")
        if self.scope is EvidenceScope.CARD and not self.card_id:
            raise DomainValidationError("Card-local evidence должно иметь card_id")
        if self.kind is EvidenceKind.SOURCE_FRAGMENT and self.address is None:
            raise DomainValidationError("Фрагмент источника должен иметь точный адрес")
        if self.kind is EvidenceKind.HUMAN_KNOWLEDGE:
            if not self.author or not self.message_id or self.scope is not EvidenceScope.CARD:
                raise DomainValidationError(
                    "Экспертное знание должно иметь автора, message_id и card-local область"
                )

    @classmethod
    def source_fragment(
        cls,
        *,
        evidence_id: str,
        card_id: str,
        selection_id: str,
        quote: str,
        address: SourceAddress,
        collected_at: datetime,
    ) -> Evidence:
        return cls(
            evidence_id=evidence_id,
            kind=EvidenceKind.SOURCE_FRAGMENT,
            scope=EvidenceScope.CARD,
            card_id=card_id,
            selection_id=selection_id,
            quote=quote,
            address=address,
            collected_at=collected_at,
        )

    @classmethod
    def selection_fragment(
        cls,
        *,
        evidence_id: str,
        selection_id: str,
        quote: str,
        address: SourceAddress,
        collected_at: datetime,
    ) -> Evidence:
        return cls(
            evidence_id=evidence_id,
            kind=EvidenceKind.SOURCE_FRAGMENT,
            scope=EvidenceScope.SELECTION,
            selection_id=selection_id,
            quote=quote,
            address=address,
            collected_at=collected_at,
        )

    @classmethod
    def human_knowledge(
        cls,
        *,
        evidence_id: str,
        card_id: str,
        selection_id: str,
        quote: str,
        author: str,
        message_id: str,
        collected_at: datetime,
    ) -> Evidence:
        return cls(
            evidence_id=evidence_id,
            kind=EvidenceKind.HUMAN_KNOWLEDGE,
            scope=EvidenceScope.CARD,
            card_id=card_id,
            selection_id=selection_id,
            quote=quote,
            author=author,
            message_id=message_id,
            collected_at=collected_at,
        )

    def supports(self, card_id: str, selection_id: str) -> bool:
        if self.selection_id != selection_id:
            return False
        if self.scope is EvidenceScope.SELECTION:
            return True
        return self.card_id == card_id
