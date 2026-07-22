from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pmi_generator.workbench.application.evaluation import (
    ExperimentMode,
    ExperimentStage,
    ExperimentStore,
    PairValidationError,
    PairedExperimentRunner,
    PinnedConfiguration,
    QualityCorpusCodec,
    RunRecord,
)

from tests.test_workbench_quality_corpus import corpus_fixture


def configuration(
    *,
    model_id: str = "synthetic-model",
    prompt_version: str = "prompt-1",
    policy_version: str = "policy-1",
) -> PinnedConfiguration:
    return PinnedConfiguration.build(
        model_id=model_id,
        server_configuration={"endpoint": "synthetic"},
        prompt_version=prompt_version,
        policy_version=policy_version,
        sampling={"temperature": 0},
        budgets={"input": 12000, "output": 4096},
        repeats=3,
    )


class PairedExperimentRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.corpus = QualityCorpusCodec.load(corpus_fixture())
        self.case = self.corpus.cases[0]

    def test_experiment_a_requires_exact_same_source_selection(self) -> None:
        runner = PairedExperimentRunner(self.corpus)
        with self.assertRaisesRegex(PairValidationError, "одинаковый selection"):
            runner.build_pair(
                mode=ExperimentMode.WINDOWING,
                case=self.case,
                baseline_selection=self.case.baseline_selection,
                candidate_selection=self.case.comparison_selection,
                baseline_mode="single_call",
                candidate_mode="windowed",
                baseline_configuration=configuration(),
                candidate_configuration=configuration(),
                baseline_tool_schema="decomposition-1",
                candidate_tool_schema="window-1",
                baseline_workflow="single-1",
                candidate_workflow="window-1",
            )

    def test_experiment_b_rejects_mode_or_model_difference(self) -> None:
        runner = PairedExperimentRunner(self.corpus)
        with self.assertRaisesRegex(PairValidationError, "Prompt 1 mode"):
            runner.build_pair(
                mode=ExperimentMode.SELECTION,
                case=self.case,
                baseline_selection=self.case.baseline_selection,
                candidate_selection=self.case.free_selection,
                baseline_mode="single_call",
                candidate_mode="windowed",
                baseline_configuration=configuration(),
                candidate_configuration=configuration(),
                baseline_tool_schema="decomposition-1",
                candidate_tool_schema="decomposition-1",
                baseline_workflow="single-1",
                candidate_workflow="single-1",
            )
        with self.assertRaisesRegex(PairValidationError, "configuration"):
            runner.build_pair(
                mode=ExperimentMode.SELECTION,
                case=self.case,
                baseline_selection=self.case.baseline_selection,
                candidate_selection=self.case.free_selection,
                baseline_mode="single_call",
                candidate_mode="single_call",
                baseline_configuration=configuration(),
                candidate_configuration=configuration(model_id="other-model"),
                baseline_tool_schema="decomposition-1",
                candidate_tool_schema="decomposition-1",
                baseline_workflow="single-1",
                candidate_workflow="single-1",
            )

    def test_manifest_id_changes_with_prompt_or_policy_version(self) -> None:
        runner = PairedExperimentRunner(self.corpus)
        pair = runner.default_pair(
            ExperimentMode.WINDOWING,
            self.case,
            configuration(),
        )
        changed_pair = runner.default_pair(
            ExperimentMode.WINDOWING,
            self.case,
            configuration(prompt_version="prompt-2"),
        )

        manifest = runner.manifest(ExperimentStage.PILOT, (pair,))
        changed = runner.manifest(ExperimentStage.PILOT, (changed_pair,))

        self.assertNotEqual(manifest.experiment_id, changed.experiment_id)
        self.assertEqual(manifest.corpus_version, self.corpus.version)

    def test_store_keeps_raw_valid_invalid_and_repeated_runs(self) -> None:
        runner = PairedExperimentRunner(self.corpus)
        pair = runner.default_pair(
            ExperimentMode.WINDOWING,
            self.case,
            configuration(),
        )
        manifest = runner.manifest(ExperimentStage.PILOT, (pair,))
        with tempfile.TemporaryDirectory() as directory:
            store = ExperimentStore(Path(directory))
            store.create(manifest)
            loaded = store.load_manifest(manifest.experiment_id)
            self.assertEqual(loaded, manifest)
            first = RunRecord(
                run_id="RUN_001",
                pair_id=pair.pair_id,
                variant="baseline",
                repeat=1,
                status="valid",
                diagnostic_path="raw/RUN_001.json",
                error=None,
            )
            invalid = RunRecord(
                run_id="RUN_002",
                pair_id=pair.pair_id,
                variant="candidate",
                repeat=1,
                status="invalid",
                diagnostic_path="raw/RUN_002.json",
                error="synthetic transport failure",
            )
            store.record(manifest, first, {"raw": "first"})
            store.record(manifest, invalid, {"raw": "invalid"})

            records = store.records(manifest)
            self.assertEqual([item.run_id for item in records], ["RUN_001", "RUN_002"])
            self.assertEqual(records[1].status, "invalid")
            raw = json.loads(
                (
                    Path(directory)
                    / manifest.experiment_id
                    / "raw"
                    / "RUN_002.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(raw, {"raw": "invalid"})

            manifest_path = (
                Path(directory) / manifest.experiment_id / "manifest.json"
            )
            tampered = json.loads(manifest_path.read_text(encoding="utf-8"))
            tampered["pairs"][0]["baseline"]["prompt_1_mode"] = "tampered"
            manifest_path.write_text(
                json.dumps(tampered, ensure_ascii=False),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                PairValidationError,
                "experiment ID повреждён",
            ):
                store.load_manifest(manifest.experiment_id)


if __name__ == "__main__":
    unittest.main()
