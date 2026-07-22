from __future__ import annotations

from dataclasses import dataclass

from ...domain.source import SourcePosition


@dataclass(frozen=True, slots=True)
class EvaluationRange:
    page: int
    line_start: int
    line_end: int


@dataclass(frozen=True, slots=True)
class EvaluationSourceLine:
    page: int
    line: int
    text: str

    @property
    def position(self) -> SourcePosition:
        return SourcePosition(self.page, self.line)


@dataclass(frozen=True, slots=True)
class EvaluationSelection:
    start: SourcePosition
    end: SourcePosition
    lines: tuple[EvaluationSourceLine, ...]


@dataclass(frozen=True, slots=True)
class ExpectedObligation:
    obligation_id: str
    description: str
    evidence: tuple[EvaluationRange, ...]
    context_loss: bool


@dataclass(frozen=True, slots=True)
class ForbiddenClaim:
    claim_id: str
    description: str


@dataclass(frozen=True, slots=True)
class ExpectedLineAssessment:
    page: int
    line: int
    role: str


@dataclass(frozen=True, slots=True)
class AnalystDecision:
    version: int
    status: str
    comment: str


@dataclass(frozen=True, slots=True)
class QualityCase:
    case_id: str
    source_hash: str
    reason: str
    comparison_selection: EvaluationSelection
    baseline_selection: EvaluationSelection
    free_selection: EvaluationSelection
    expected_outcome: str
    expected_obligations: tuple[ExpectedObligation, ...]
    forbidden_claims: tuple[ForbiddenClaim, ...]
    allowed_groupings: tuple[tuple[str, ...], ...]
    line_expectations: tuple[ExpectedLineAssessment, ...]
    analyst_decision: AnalystDecision


@dataclass(frozen=True, slots=True)
class QualityCorpus:
    schema_version: int
    version: str
    cases: tuple[QualityCase, ...]
    minimum_full_run_cases: int = 12

    @property
    def ready_for_full_run(self) -> bool:
        return len(self.cases) >= self.minimum_full_run_cases

    @property
    def missing_case_count(self) -> int:
        return max(0, self.minimum_full_run_cases - len(self.cases))


class CorpusValidationError(ValueError):
    pass


__all__ = [
    "AnalystDecision",
    "CorpusValidationError",
    "EvaluationRange",
    "EvaluationSelection",
    "EvaluationSourceLine",
    "ExpectedLineAssessment",
    "ExpectedObligation",
    "ForbiddenClaim",
    "QualityCase",
    "QualityCorpus",
]
