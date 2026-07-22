from __future__ import annotations

import hashlib
from dataclasses import dataclass

from .metrics import QualityEvaluation
from .runner import (
    ExperimentManifest,
    ExperimentPair,
    ExperimentStage,
    RunRecord,
)


@dataclass(frozen=True, slots=True)
class BlindReviewPacket:
    review_id: str
    case_id: str
    output_reference: str

    def to_dict(self) -> dict[str, object]:
        return {
            "review_id": self.review_id,
            "case_id": self.case_id,
            "output_reference": self.output_reference,
        }


class BlindReviewBuilder:
    @staticmethod
    def build(
        pair: ExperimentPair,
        record: RunRecord,
    ) -> BlindReviewPacket:
        if record.pair_id != pair.pair_id:
            raise ValueError("Review record ссылается на другую pair")
        review_id = "REVIEW_" + hashlib.sha256(
            f"{pair.pair_id}:{record.run_id}".encode("utf-8")
        ).hexdigest()[:16].upper()
        return BlindReviewPacket(
            review_id,
            pair.case_id,
            record.diagnostic_path,
        )


@dataclass(frozen=True, slots=True)
class VariantAggregate:
    variant: str
    total_runs: int
    valid_runs: int
    invalid_runs: int
    hard_gate_failures: int
    expected_obligations: int
    matched_obligations: int
    missing_obligations: int
    expected_context_loss_obligations: int
    matched_context_loss_obligations: int
    erroneous_merges: int
    erroneous_splits: int
    duplicates: int
    ungrounded_skeletons: int
    successful_range_reviews: int

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

    def to_dict(self) -> dict[str, object]:
        return {
            "variant": self.variant,
            "total_runs": self.total_runs,
            "valid_runs": self.valid_runs,
            "invalid_runs": self.invalid_runs,
            "hard_gate_failures": self.hard_gate_failures,
            "expected_obligations": self.expected_obligations,
            "matched_obligations": self.matched_obligations,
            "missing_obligations": self.missing_obligations,
            "expected_context_loss_obligations": (
                self.expected_context_loss_obligations
            ),
            "matched_context_loss_obligations": (
                self.matched_context_loss_obligations
            ),
            "erroneous_merges": self.erroneous_merges,
            "erroneous_splits": self.erroneous_splits,
            "duplicates": self.duplicates,
            "ungrounded_skeletons": self.ungrounded_skeletons,
            "successful_range_reviews": self.successful_range_reviews,
            "semantic_recall": self.semantic_recall,
            "context_loss_recall": self.context_loss_recall,
        }


@dataclass(frozen=True, slots=True)
class PairReport:
    pair_id: str
    case_id: str
    mode: str
    baseline: VariantAggregate
    candidate: VariantAggregate
    maximum_variability: float

    def to_dict(self) -> dict[str, object]:
        return {
            "pair_id": self.pair_id,
            "case_id": self.case_id,
            "mode": self.mode,
            "baseline": self.baseline.to_dict(),
            "candidate": self.candidate.to_dict(),
            "maximum_variability": self.maximum_variability,
        }


@dataclass(frozen=True, slots=True)
class ExperimentReport:
    experiment_id: str
    stage: str
    total_runs: int
    invalid_runs: int
    raw_diagnostics: tuple[str, ...]
    aggregates: tuple[VariantAggregate, ...]
    pairs: tuple[PairReport, ...]
    decision: str

    def to_dict(self) -> dict[str, object]:
        return {
            "experiment_id": self.experiment_id,
            "stage": self.stage,
            "total_runs": self.total_runs,
            "invalid_runs": self.invalid_runs,
            "raw_diagnostics": list(self.raw_diagnostics),
            "aggregates": [item.to_dict() for item in self.aggregates],
            "pairs": [item.to_dict() for item in self.pairs],
            "decision": self.decision,
        }

    def render_markdown(self) -> str:
        lines = [
            f"# Experiment {self.experiment_id}",
            "",
            f"Stage: `{self.stage}`",
            f"Decision: `{self.decision}`",
            f"Runs: {self.total_runs}; invalid: {self.invalid_runs}",
            "",
            "## Raw diagnostics",
            "",
        ]
        lines.extend(f"- `{path}`" for path in self.raw_diagnostics)
        for aggregate in self.aggregates:
            lines.extend(
                [
                    "",
                    f"## {aggregate.variant}",
                    "",
                    f"- Runs: {aggregate.total_runs}",
                    f"- Invalid: {aggregate.invalid_runs}",
                    f"- Hard gate failures: {aggregate.hard_gate_failures}",
                    f"- Semantic recall: {aggregate.semantic_recall:.4f}",
                    f"- Missing obligations: {aggregate.missing_obligations}",
                    f"- Erroneous merges: {aggregate.erroneous_merges}",
                    f"- Erroneous splits: {aggregate.erroneous_splits}",
                    f"- Duplicates: {aggregate.duplicates}",
                    f"- Ungrounded: {aggregate.ungrounded_skeletons}",
                ]
            )
        lines.extend(["", "## Cases", ""])
        for pair in self.pairs:
            lines.append(
                f"- `{pair.case_id}` / `{pair.mode}` / "
                f"variability={pair.maximum_variability:.4f}"
            )
        return "\n".join(lines) + "\n"


class ExperimentReportBuilder:
    @classmethod
    def build(
        cls,
        manifest: ExperimentManifest,
        records: tuple[RunRecord, ...],
        evaluations: dict[str, QualityEvaluation],
    ) -> ExperimentReport:
        ordered_records = tuple(sorted(records, key=lambda item: item.run_id))
        known_pairs = {pair.pair_id for pair in manifest.pairs}
        if any(record.pair_id not in known_pairs for record in ordered_records):
            raise ValueError("Report содержит run неизвестной pair")
        valid_ids = {
            record.run_id
            for record in ordered_records
            if record.status == "valid"
        }
        if set(evaluations) != valid_ids:
            raise ValueError(
                "Каждый valid run должен иметь ровно одну evaluation"
            )
        if manifest.stage is ExperimentStage.FULL:
            expected_slots = {
                (pair.pair_id, variant, repeat)
                for pair in manifest.pairs
                for variant in ("baseline", "candidate")
                for repeat in range(
                    1,
                    pair.baseline.configuration.repeats + 1,
                )
            }
            actual_slots = {
                (record.pair_id, record.variant, record.repeat)
                for record in ordered_records
            }
            if not expected_slots <= actual_slots:
                raise ValueError(
                    "Full report не содержит все заявленные повторы"
                )
        aggregates = tuple(
            cls._aggregate(variant, ordered_records, evaluations)
            for variant in ("baseline", "candidate")
        )
        pair_reports = tuple(
            cls._pair_report(pair, ordered_records, evaluations)
            for pair in sorted(manifest.pairs, key=lambda item: item.pair_id)
        )
        invalid_runs = sum(
            record.status == "invalid" for record in ordered_records
        )
        decision = cls._decision(
            manifest,
            aggregates,
            pair_reports,
            invalid_runs,
        )
        return ExperimentReport(
            manifest.experiment_id,
            manifest.stage.value,
            len(ordered_records),
            invalid_runs,
            tuple(record.diagnostic_path for record in ordered_records),
            aggregates,
            pair_reports,
            decision,
        )

    @staticmethod
    def _aggregate(
        variant: str,
        records: tuple[RunRecord, ...],
        evaluations: dict[str, QualityEvaluation],
    ) -> VariantAggregate:
        selected = tuple(item for item in records if item.variant == variant)
        valid = tuple(item for item in selected if item.status == "valid")
        results = tuple(evaluations[item.run_id] for item in valid)
        return VariantAggregate(
            variant=variant,
            total_runs=len(selected),
            valid_runs=len(valid),
            invalid_runs=len(selected) - len(valid),
            hard_gate_failures=sum(
                not result.hard_gates.passed for result in results
            ),
            expected_obligations=sum(
                result.metrics.expected_obligations for result in results
            ),
            matched_obligations=sum(
                result.metrics.matched_obligations for result in results
            ),
            missing_obligations=sum(
                result.metrics.missing_obligations for result in results
            ),
            expected_context_loss_obligations=sum(
                result.metrics.expected_context_loss_obligations
                for result in results
            ),
            matched_context_loss_obligations=sum(
                result.metrics.matched_context_loss_obligations
                for result in results
            ),
            erroneous_merges=sum(
                result.metrics.erroneous_merges for result in results
            ),
            erroneous_splits=sum(
                result.metrics.erroneous_splits for result in results
            ),
            duplicates=sum(
                result.metrics.duplicates for result in results
            ),
            ungrounded_skeletons=sum(
                result.metrics.ungrounded_skeletons for result in results
            ),
            successful_range_reviews=sum(
                result.metrics.range_review_success for result in results
            ),
        )

    @classmethod
    def _pair_report(
        cls,
        pair: ExperimentPair,
        records: tuple[RunRecord, ...],
        evaluations: dict[str, QualityEvaluation],
    ) -> PairReport:
        selected = tuple(
            record for record in records if record.pair_id == pair.pair_id
        )
        baseline = cls._aggregate("baseline", selected, evaluations)
        candidate = cls._aggregate("candidate", selected, evaluations)
        variability = 0.0
        for variant in ("baseline", "candidate"):
            recalls = [
                evaluations[record.run_id].metrics.semantic_recall
                for record in selected
                if record.variant == variant
                and record.status == "valid"
            ]
            if recalls:
                variability = max(
                    variability,
                    max(recalls) - min(recalls),
                )
        return PairReport(
            pair.pair_id,
            pair.case_id,
            pair.mode.value,
            baseline,
            candidate,
            variability,
        )

    @staticmethod
    def _decision(
        manifest: ExperimentManifest,
        aggregates: tuple[VariantAggregate, ...],
        pairs: tuple[PairReport, ...],
        invalid_runs: int,
    ) -> str:
        if manifest.stage is ExperimentStage.PILOT:
            return "pilot_only"
        thresholds = manifest.thresholds
        if thresholds is None:
            return "invalid_manifest"
        candidate = next(
            item for item in aggregates if item.variant == "candidate"
        )
        hard_failure = any(item.hard_gate_failures for item in aggregates)
        variability_failure = any(
            pair.maximum_variability > thresholds.maximum_variability
            for pair in pairs
        )
        semantic_failure = (
            candidate.semantic_recall < thresholds.minimum_semantic_recall
            or candidate.missing_obligations > thresholds.maximum_missing
            or candidate.erroneous_merges > thresholds.maximum_merges
            or candidate.erroneous_splits > thresholds.maximum_splits
            or candidate.duplicates > thresholds.maximum_duplicates
            or candidate.ungrounded_skeletons > thresholds.maximum_ungrounded
        )
        regression = False
        if pairs and pairs[0].mode == "experiment_a_windowing":
            regression = any(
                pair.candidate.missing_obligations
                > pair.baseline.missing_obligations
                or pair.candidate.erroneous_merges
                > pair.baseline.erroneous_merges
                or pair.candidate.erroneous_splits
                > pair.baseline.erroneous_splits
                or pair.candidate.duplicates > pair.baseline.duplicates
                or pair.candidate.ungrounded_skeletons
                > pair.baseline.ungrounded_skeletons
                or pair.candidate.successful_range_reviews
                < pair.baseline.successful_range_reviews
                for pair in pairs
            )
        elif pairs:
            baseline = next(
                item for item in aggregates if item.variant == "baseline"
            )
            regression = (
                candidate.context_loss_recall
                < thresholds.minimum_semantic_recall
                or candidate.missing_obligations
                >= baseline.missing_obligations
                or candidate.ungrounded_skeletons
                > baseline.ungrounded_skeletons
                or candidate.successful_range_reviews
                < baseline.successful_range_reviews
            )
        if (
            invalid_runs
            or hard_failure
            or variability_failure
            or semantic_failure
            or regression
        ):
            return "исправить и повторить"
        return "rollout"


__all__ = [
    "BlindReviewBuilder",
    "BlindReviewPacket",
    "ExperimentReport",
    "ExperimentReportBuilder",
    "PairReport",
    "VariantAggregate",
]
