from __future__ import annotations

import unittest

from pmi_generator.workbench.application.evaluation import (
    EvaluatedEvidence,
    EvaluatedOutput,
    EvaluatedSkeleton,
    QualityCorpusCodec,
    QualityMetrics,
    StructuredAnalystAssessment,
)

from tests.test_workbench_quality_corpus import corpus_fixture


def valid_output() -> EvaluatedOutput:
    return EvaluatedOutput(
        outcome="skeletons_created",
        parent_status="completed",
        domain_mutation_count=1,
        skeletons=(
            EvaluatedSkeleton(
                skeleton_id="SK_001",
                evidence=(
                    EvaluatedEvidence(
                        page=1,
                        line_start=2,
                        line_end=3,
                        quote="synthetic line 2\nsynthetic line 3",
                        origin="source",
                    ),
                ),
                field_evidence_complete=True,
            ),
        ),
        line_assessments=((1, 1, "context"), (1, 2, "evidence"), (1, 3, "evidence")),
        range_review_success=True,
    )


def assessment(**changes: object) -> StructuredAnalystAssessment:
    values: dict[str, object] = {
        "matched_obligations": (("SK_001", ("OBL_001",)),),
        "erroneous_merge_skeleton_ids": (),
        "erroneous_split_obligation_ids": (),
        "duplicate_skeleton_ids": (),
        "ungrounded_skeleton_ids": (),
        "critical_forbidden_claim_ids": (),
    }
    values.update(changes)
    return StructuredAnalystAssessment(**values)  # type: ignore[arg-type]


class QualityMetricsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.case = QualityCorpusCodec.load(corpus_fixture()).cases[0]

    def evaluate(
        self,
        output: EvaluatedOutput,
        analyst: StructuredAnalystAssessment | None = None,
    ):
        return QualityMetrics.evaluate(
            case=self.case,
            selection=self.case.free_selection,
            output=output,
            analyst=analyst or assessment(),
        )

    def test_valid_result_passes_all_hard_gates(self) -> None:
        result = self.evaluate(valid_output())
        self.assertTrue(result.hard_gates.passed)
        self.assertEqual(result.metrics.missing_obligations, 0)
        self.assertEqual(result.metrics.semantic_recall, 1.0)

    def test_each_deterministic_hard_gate_is_independent(self) -> None:
        base = valid_output()
        outside = EvaluatedEvidence(2, 1, 1, "outside", "source")
        bad_quote = EvaluatedEvidence(1, 2, 3, "wrong quote", "source")
        generated = EvaluatedEvidence(
            1,
            2,
            3,
            "synthetic line 2\nsynthetic line 3",
            "generated",
        )
        cases = {
            "evidence_outside_selection": (
                base.with_skeleton_evidence((outside,)),
                assessment(),
            ),
            "source_quote_mismatch": (
                base.with_skeleton_evidence((bad_quote,)),
                assessment(),
            ),
            "line_assessment_partition": (
                base.with_line_assessments(((1, 1, "context"),)),
                assessment(),
            ),
            "partial_domain_mutation": (
                base.with_parent("failed", 1),
                assessment(),
            ),
            "generated_as_source_evidence": (
                base.with_skeleton_evidence((generated,)),
                assessment(),
            ),
            "critical_forbidden_claim": (
                base,
                assessment(critical_forbidden_claim_ids=("CLAIM_001",)),
            ),
        }

        for gate, (output, analyst) in cases.items():
            with self.subTest(gate=gate):
                result = self.evaluate(output, analyst)
                self.assertFalse(result.hard_gates.passed)
                self.assertEqual(getattr(result.hard_gates, gate), 1)

    def test_semantic_counters_are_counted_independently(self) -> None:
        result = self.evaluate(
            valid_output(),
            assessment(
                matched_obligations=(),
                erroneous_merge_skeleton_ids=("SK_001",),
                erroneous_split_obligation_ids=("OBL_001",),
                duplicate_skeleton_ids=("SK_001",),
                ungrounded_skeleton_ids=("SK_001",),
            ),
        )

        self.assertEqual(result.metrics.missing_obligations, 1)
        self.assertEqual(result.metrics.erroneous_merges, 1)
        self.assertEqual(result.metrics.erroneous_splits, 1)
        self.assertEqual(result.metrics.duplicates, 1)
        self.assertEqual(result.metrics.ungrounded_skeletons, 1)
        self.assertEqual(result.metrics.semantic_recall, 0.0)

    def test_unknown_analyst_ids_are_rejected_without_text_heuristics(self) -> None:
        with self.assertRaisesRegex(ValueError, "неизвестный"):
            self.evaluate(
                valid_output(),
                assessment(
                    critical_forbidden_claim_ids=("UNKNOWN_CLAIM",),
                ),
            )


if __name__ == "__main__":
    unittest.main()
