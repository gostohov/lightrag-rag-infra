from __future__ import annotations

import copy
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path

from pmi_generator.workbench.application.evaluation import (
    BlindReviewBuilder,
    ExperimentMode,
    ExperimentReportBuilder,
    ExperimentStage,
    ExperimentStore,
    PairValidationError,
    PairedExperimentRunner,
    PilotReceipt,
    QualityCorpusCodec,
    QualityMetrics,
    RunRecord,
    SemanticThresholds,
)
from pmi_generator.workbench.application.prompting import default_policy
from pmi_generator.workbench.domain import (
    SourceDocument,
    SourceMetadata,
    SourcePage,
    SourceSection,
)
from pmi_generator.workbench.infrastructure.mock_mode import MockLlmTransport
from pmi_generator.workbench.infrastructure.quality import (
    QualityExperimentExecutor,
)

from tests.test_workbench_quality_corpus import corpus_fixture
from tests.test_workbench_quality_metrics import assessment, valid_output
from tests.test_workbench_quality_runner import configuration


def full_corpus():
    payload = corpus_fixture()
    base = payload["cases"][0]  # type: ignore[index]
    payload["cases"] = []
    for index in range(12):
        item = copy.deepcopy(base)
        item["case_id"] = f"SYNTHETIC_CASE_{index + 1:03d}"
        payload["cases"].append(item)  # type: ignore[union-attr]
    return QualityCorpusCodec.load(payload)


class QualityReportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.corpus = full_corpus()
        self.runner = PairedExperimentRunner(self.corpus)
        self.case = self.corpus.cases[0]
        self.a_pair = self.runner.default_pair(
            ExperimentMode.WINDOWING,
            self.case,
            configuration(),
        )
        self.b_pair = self.runner.default_pair(
            ExperimentMode.SELECTION,
            self.case,
            configuration(),
        )

    def test_blind_review_packet_does_not_disclose_variant(self) -> None:
        record = RunRecord(
            "RUN_A",
            self.a_pair.pair_id,
            "baseline",
            1,
            "valid",
            "raw/RUN_A.json",
            None,
        )

        packet = BlindReviewBuilder.build(self.a_pair, record)

        self.assertNotIn("baseline", packet.to_dict().values())
        self.assertNotIn("candidate", packet.to_dict().values())
        self.assertEqual(packet.case_id, self.case.case_id)

    def test_pilot_requires_exactly_four_valid_variant_runs(self) -> None:
        manifest = self.runner.manifest(
            ExperimentStage.PILOT,
            (self.a_pair, self.b_pair),
        )
        with tempfile.TemporaryDirectory() as directory:
            store = ExperimentStore(Path(directory))
            store.create(manifest)
            for index, (pair, variant) in enumerate(
                (
                    (self.a_pair, "baseline"),
                    (self.a_pair, "candidate"),
                    (self.b_pair, "baseline"),
                    (self.b_pair, "candidate"),
                ),
                start=1,
            ):
                run_id = f"RUN_{index}"
                store.record(
                    manifest,
                    RunRecord(
                        run_id,
                        pair.pair_id,
                        variant,
                        1,
                        "valid",
                        f"raw/{run_id}.json",
                        None,
                    ),
                    {"synthetic": run_id},
                )

            evaluation = QualityMetrics.evaluate(
                case=self.case,
                selection=self.case.free_selection,
                output=valid_output(),
                analyst=assessment(),
            )
            receipt = store.pilot_receipt(
                manifest,
                {f"RUN_{index}": evaluation for index in range(1, 5)},
            )

            self.assertTrue(receipt.successful)
            self.assertEqual(len(receipt.run_ids), 4)

    def test_full_manifest_requires_ready_corpus_pilot_and_thresholds(self) -> None:
        pilot = self.runner.manifest(
            ExperimentStage.PILOT,
            (self.a_pair, self.b_pair),
        )
        with tempfile.TemporaryDirectory() as directory:
            store = ExperimentStore(Path(directory))
            store.create(pilot)
            with self.assertRaisesRegex(PairValidationError, "pilot"):
                self.runner.full_manifest(
                    (self.a_pair,),
                    pilot_receipt=store.pilot_receipt(pilot, {}),
                    thresholds=SemanticThresholds.strict_example(),
                )

    def test_threshold_change_creates_new_full_experiment_id(self) -> None:
        receipt = PilotReceipt(
            "PILOT_OK",
            ("1", "2", "3", "4"),
            True,
        )
        pairs = tuple(
            self.runner.default_pair(
                ExperimentMode.WINDOWING,
                case,
                configuration(),
            )
            for case in self.corpus.cases
        )
        first = self.runner.full_manifest(
            pairs,
            pilot_receipt=receipt,
            thresholds=SemanticThresholds.strict_example(),
        )
        changed = self.runner.full_manifest(
            pairs,
            pilot_receipt=receipt,
            thresholds=SemanticThresholds(
                minimum_semantic_recall=0.75,
                maximum_variability=0.25,
                maximum_missing=1,
                maximum_merges=0,
                maximum_splits=0,
                maximum_duplicates=0,
                maximum_ungrounded=0,
            ),
        )
        self.assertNotEqual(first.experiment_id, changed.experiment_id)

        with self.assertRaisesRegex(ValueError, "все заявленные повторы"):
            ExperimentReportBuilder.build(first, (), {})

    def test_report_keeps_invalid_denominator_raw_links_and_order_invariance(
        self,
    ) -> None:
        manifest = self.runner.manifest(
            ExperimentStage.PILOT,
            (self.a_pair,),
        )
        valid = RunRecord(
            "RUN_1",
            self.a_pair.pair_id,
            "baseline",
            1,
            "valid",
            "raw/RUN_1.json",
            None,
        )
        invalid = RunRecord(
            "RUN_2",
            self.a_pair.pair_id,
            "candidate",
            1,
            "invalid",
            "raw/RUN_2.json",
            "synthetic failure",
        )
        evaluation = QualityMetrics.evaluate(
            case=self.case,
            selection=self.case.comparison_selection,
            output=valid_output(),
            analyst=assessment(),
        )

        first = ExperimentReportBuilder.build(
            manifest,
            (valid, invalid),
            {"RUN_1": evaluation},
        )
        second = ExperimentReportBuilder.build(
            manifest,
            (invalid, valid),
            {"RUN_1": evaluation},
        )

        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertEqual(first.total_runs, 2)
        self.assertEqual(first.invalid_runs, 1)
        self.assertEqual(
            set(first.raw_diagnostics),
            {"raw/RUN_1.json", "raw/RUN_2.json"},
        )


class QualityPilotExecutorTests(unittest.IsolatedAsyncioTestCase):
    async def test_runs_four_variants_through_real_prompt_flows(self) -> None:
        corpus = QualityCorpusCodec.load(corpus_fixture())
        runner = PairedExperimentRunner(corpus)
        case = corpus.cases[0]
        manifest = runner.manifest(
            ExperimentStage.PILOT,
            (
                runner.default_pair(
                    ExperimentMode.WINDOWING,
                    case,
                    configuration(),
                ),
                runner.default_pair(
                    ExperimentMode.SELECTION,
                    case,
                    configuration(),
                ),
            ),
        )
        document = SourceDocument(
            pages=(
                SourcePage(
                    1,
                    "1",
                    (
                        "synthetic line 1",
                        "synthetic line 2",
                        "synthetic line 3",
                    ),
                ),
            ),
            sections=(
                SourceSection("root", "1", "Synthetic", ("1",), (1,)),
            ),
            metadata=SourceMetadata(
                "synthetic.pdf",
                "a" * 64,
                "synthetic",
                "1",
                datetime(2026, 1, 1, tzinfo=UTC),
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            executor = QualityExperimentExecutor(
                document=document,
                policy=default_policy(),
                transport_factory=lambda: MockLlmTransport(delay=0),
                output_root=Path(directory),
                evaluation_route_max_lines=1,
                evaluation_primary_max_lines=1,
                evaluation_overlap_lines=0,
            )

            records = await executor.execute(manifest)

            self.assertEqual(len(records), 4)
            self.assertTrue(all(record.status == "valid" for record in records))
            self.assertEqual(
                len(
                    list(
                        (
                            Path(directory)
                            / manifest.experiment_id
                            / "raw"
                        ).glob("*.json")
                    )
                ),
                4,
            )

    def test_full_schedule_contains_three_repeats_for_every_variant(self) -> None:
        corpus = full_corpus()
        runner = PairedExperimentRunner(corpus)
        pairs = tuple(
            runner.default_pair(
                ExperimentMode.WINDOWING,
                case,
                configuration(),
            )
            for case in corpus.cases
        )
        manifest = runner.full_manifest(
            pairs,
            pilot_receipt=PilotReceipt(
                "PILOT_OK",
                ("1", "2", "3", "4"),
                True,
            ),
            thresholds=SemanticThresholds.strict_example(),
        )

        schedule = QualityExperimentExecutor.schedule(manifest)

        self.assertEqual(len(schedule), 72)
        self.assertEqual(
            {repeat for _pair, _variant, repeat in schedule},
            {1, 2, 3},
        )


if __name__ == "__main__":
    unittest.main()
