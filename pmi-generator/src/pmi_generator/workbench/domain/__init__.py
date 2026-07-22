"""Предметная модель рабочего места аналитика ПМИ."""

from .card import CardMutation, CardState, TestCard
from .decisions import AnalystResolution, CardDecision
from .enums import (
    CardDecisionKind,
    EpistemicStatus,
    EvidenceKind,
    EvidenceScope,
    GapResolutionMode,
    GapStatus,
)
from .errors import (
    DomainError,
    DomainValidationError,
    EvidenceScopeError,
    PathNotAllowedError,
)
from .evidence import Evidence, SourceAddress
from .fields import ContentField, Derivation
from .gaps import (
    GapClosureContract,
    GapClosureEvaluation,
    GapClosureOutcome,
    GapPathClosure,
    GapValueForm,
    RelatedGap,
)
from .source import (
    ExecutionMode,
    SourceDocument,
    SourceMetadata,
    SourcePage,
    SourcePosition,
    SourceSection,
    TextSelection,
)

__all__ = [
    "ExecutionMode",
    "AnalystResolution",
    "CardDecision",
    "CardDecisionKind",
    "CardMutation",
    "CardState",
    "ContentField",
    "Derivation",
    "DomainError",
    "DomainValidationError",
    "EpistemicStatus",
    "Evidence",
    "EvidenceKind",
    "EvidenceScope",
    "EvidenceScopeError",
    "GapResolutionMode",
    "GapClosureContract",
    "GapClosureEvaluation",
    "GapClosureOutcome",
    "GapPathClosure",
    "GapStatus",
    "GapValueForm",
    "PathNotAllowedError",
    "RelatedGap",
    "SourceAddress",
    "SourceDocument",
    "SourceMetadata",
    "SourcePage",
    "SourcePosition",
    "SourceSection",
    "TestCard",
    "TextSelection",
]
