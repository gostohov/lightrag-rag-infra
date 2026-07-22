from __future__ import annotations

from dataclasses import dataclass, replace

from .models import EvaluationSelection, QualityCase


@dataclass(frozen=True, slots=True)
class EvaluatedEvidence:
    page: int
    line_start: int
    line_end: int
    quote: str
    origin: str


@dataclass(frozen=True, slots=True)
class EvaluatedSkeleton:
    skeleton_id: str
    evidence: tuple[EvaluatedEvidence, ...]
    field_evidence_complete: bool


@dataclass(frozen=True, slots=True)
class EvaluatedOutput:
    outcome: str
    parent_status: str
    domain_mutation_count: int
    skeletons: tuple[EvaluatedSkeleton, ...]
    line_assessments: tuple[tuple[int, int, str], ...]
    range_review_success: bool

    def with_skeleton_evidence(
        self,
        evidence: tuple[EvaluatedEvidence, ...],
    ) -> EvaluatedOutput:
        if not self.skeletons:
            raise ValueError("В output нет skeleton")
        return replace(
            self,
            skeletons=(replace(self.skeletons[0], evidence=evidence),)
            + self.skeletons[1:],
        )

    def with_line_assessments(
        self,
        values: tuple[tuple[int, int, str], ...],
    ) -> EvaluatedOutput:
        return replace(self, line_assessments=values)

    def with_parent(
        self,
        status: str,
        domain_mutation_count: int,
    ) -> EvaluatedOutput:
        return replace(
            self,
            parent_status=status,
            domain_mutation_count=domain_mutation_count,
        )


@dataclass(frozen=True, slots=True)
class StructuredAnalystAssessment:
    matched_obligations: tuple[tuple[str, tuple[str, ...]], ...]
    erroneous_merge_skeleton_ids: tuple[str, ...]
    erroneous_split_obligation_ids: tuple[str, ...]
    duplicate_skeleton_ids: tuple[str, ...]
    ungrounded_skeleton_ids: tuple[str, ...]
    critical_forbidden_claim_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class HardGateCounts:
    evidence_outside_selection: int
    source_quote_mismatch: int
    line_assessment_partition: int
    partial_domain_mutation: int
    generated_as_source_evidence: int
    critical_forbidden_claim: int

    @property
    def passed(self) -> bool:
        return all(
            value == 0
            for value in (
                self.evidence_outside_selection,
                self.source_quote_mismatch,
                self.line_assessment_partition,
                self.partial_domain_mutation,
                self.generated_as_source_evidence,
                self.critical_forbidden_claim,
            )
        )


@dataclass(frozen=True, slots=True)
class DecompositionMetricCounts:
    expected_obligations: int
    matched_obligations: int
    missing_obligations: int
    expected_context_loss_obligations: int
    matched_context_loss_obligations: int
    erroneous_merges: int
    erroneous_splits: int
    duplicates: int
    ungrounded_skeletons: int
    incomplete_field_evidence: int
    line_assessment_mismatches: int
    range_review_success: bool

    @property
    def semantic_recall(self) -> float:
        if self.expected_obligations == 0:
            return 1.0
        return self.matched_obligations / self.expected_obligations

    @property
    def context_loss_recall(self) -> float:
        if self.expected_context_loss_obligations == 0:
            return 1.0
        return (
            self.matched_context_loss_obligations
            / self.expected_context_loss_obligations
        )


@dataclass(frozen=True, slots=True)
class QualityEvaluation:
    hard_gates: HardGateCounts
    metrics: DecompositionMetricCounts


class QualityMetrics:
    @classmethod
    def evaluate(
        cls,
        *,
        case: QualityCase,
        selection: EvaluationSelection,
        output: EvaluatedOutput,
        analyst: StructuredAnalystAssessment,
    ) -> QualityEvaluation:
        cls._validate_analyst(case, output, analyst)
        source = {
            (line.page, line.line): line.text
            for line in selection.lines
        }
        outside = 0
        quote_mismatch = 0
        generated = 0
        for skeleton in output.skeletons:
            for evidence in skeleton.evidence:
                coordinates = [
                    (evidence.page, line)
                    for line in range(evidence.line_start, evidence.line_end + 1)
                ]
                if not coordinates or any(item not in source for item in coordinates):
                    outside += 1
                else:
                    expected_quote = "\n".join(source[item] for item in coordinates)
                    if evidence.quote != expected_quote:
                        quote_mismatch += 1
                if evidence.origin != "source":
                    generated += 1
        expected_positions = set(source)
        actual_positions = [
            (page, line) for page, line, _role in output.line_assessments
        ]
        partition_error = int(
            set(actual_positions) != expected_positions
            or len(actual_positions) != len(set(actual_positions))
        )
        partial_mutation = int(
            output.parent_status != "completed"
            and output.domain_mutation_count != 0
        )
        hard_gates = HardGateCounts(
            evidence_outside_selection=outside,
            source_quote_mismatch=quote_mismatch,
            line_assessment_partition=partition_error,
            partial_domain_mutation=partial_mutation,
            generated_as_source_evidence=generated,
            critical_forbidden_claim=len(
                analyst.critical_forbidden_claim_ids
            ),
        )
        matched = {
            obligation_id
            for _skeleton_id, obligation_ids in analyst.matched_obligations
            for obligation_id in obligation_ids
        }
        expected = {
            obligation.obligation_id
            for obligation in case.expected_obligations
        }
        expected_context_loss = {
            obligation.obligation_id
            for obligation in case.expected_obligations
            if obligation.context_loss
        }
        expected_roles = {
            (item.page, item.line): item.role
            for item in case.line_expectations
            if (item.page, item.line) in source
        }
        actual_roles = {
            (page, line): role
            for page, line, role in output.line_assessments
        }
        line_mismatches = sum(
            actual_roles.get(position) != role
            for position, role in expected_roles.items()
        )
        metrics = DecompositionMetricCounts(
            expected_obligations=len(expected),
            matched_obligations=len(expected & matched),
            missing_obligations=len(expected - matched),
            expected_context_loss_obligations=len(expected_context_loss),
            matched_context_loss_obligations=len(
                expected_context_loss & matched
            ),
            erroneous_merges=len(analyst.erroneous_merge_skeleton_ids),
            erroneous_splits=len(analyst.erroneous_split_obligation_ids),
            duplicates=len(analyst.duplicate_skeleton_ids),
            ungrounded_skeletons=len(analyst.ungrounded_skeleton_ids),
            incomplete_field_evidence=sum(
                not skeleton.field_evidence_complete
                for skeleton in output.skeletons
            ),
            line_assessment_mismatches=line_mismatches,
            range_review_success=output.range_review_success,
        )
        return QualityEvaluation(hard_gates, metrics)

    @staticmethod
    def _validate_analyst(
        case: QualityCase,
        output: EvaluatedOutput,
        analyst: StructuredAnalystAssessment,
    ) -> None:
        skeleton_ids = {item.skeleton_id for item in output.skeletons}
        obligation_ids = {
            item.obligation_id for item in case.expected_obligations
        }
        forbidden_ids = {item.claim_id for item in case.forbidden_claims}
        matched_skeletons = [
            skeleton_id
            for skeleton_id, _obligations in analyst.matched_obligations
        ]
        if len(matched_skeletons) != len(set(matched_skeletons)):
            raise ValueError("Один skeleton размечен obligations несколько раз")
        referenced_skeletons = set(matched_skeletons)
        referenced_skeletons.update(analyst.erroneous_merge_skeleton_ids)
        referenced_skeletons.update(analyst.duplicate_skeleton_ids)
        referenced_skeletons.update(analyst.ungrounded_skeleton_ids)
        if not referenced_skeletons <= skeleton_ids:
            raise ValueError("Analyst assessment содержит неизвестный skeleton")
        matched_obligations = {
            obligation
            for _skeleton, obligations in analyst.matched_obligations
            for obligation in obligations
        }
        referenced_obligations = (
            matched_obligations
            | set(analyst.erroneous_split_obligation_ids)
        )
        if not referenced_obligations <= obligation_ids:
            raise ValueError("Analyst assessment содержит неизвестный obligation")
        if not set(analyst.critical_forbidden_claim_ids) <= forbidden_ids:
            raise ValueError(
                "Analyst assessment содержит неизвестный forbidden claim"
            )


__all__ = [
    "DecompositionMetricCounts",
    "EvaluatedEvidence",
    "EvaluatedOutput",
    "EvaluatedSkeleton",
    "HardGateCounts",
    "QualityEvaluation",
    "QualityMetrics",
    "StructuredAnalystAssessment",
]
