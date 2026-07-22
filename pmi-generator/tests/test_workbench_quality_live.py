from __future__ import annotations

import os
import json
import unittest
from datetime import UTC, datetime
from pathlib import Path

from pmi_generator.workbench.application.evaluation import (
    ExperimentMode,
    ExperimentStage,
    PairedExperimentRunner,
    PilotReceipt,
    PinnedConfiguration,
    QualityCorpusCodec,
    SemanticThresholds,
)
from pmi_generator.workbench.application.prompting import PromptId, default_policy
from pmi_generator.workbench.infrastructure.llm import (
    OpenAICompatibleTransport,
    OpenAITransportSettings,
)
from pmi_generator.workbench.infrastructure.quality import (
    QualityExperimentExecutor,
)
from pmi_generator.workbench.infrastructure.source import PypdfSourceExtractor


@unittest.skipUnless(
    os.getenv("PMI_QUALITY_LIVE") == "1",
    "VDI-only decomposition quality pilot",
)
class LiveDecompositionQualityPilot(unittest.IsolatedAsyncioTestCase):
    async def test_capture_four_pilot_variants(self) -> None:
        corpus_path = Path(os.environ["PMI_QUALITY_CORPUS"])
        pdf_path = Path(os.environ["PMI_QUALITY_PDF"])
        output_root = Path(os.environ["PMI_QUALITY_OUTPUT"])
        stage = os.getenv("PMI_QUALITY_STAGE", "pilot")
        llm_url = os.environ["PMI_LLM_URL"]
        llm_model = os.environ["PMI_LLM_MODEL"]
        route_lines = int(os.getenv("PMI_QUALITY_ROUTE_LINES", "40"))
        primary_lines = int(os.getenv("PMI_QUALITY_PRIMARY_LINES", "40"))
        overlap_lines = int(os.getenv("PMI_QUALITY_OVERLAP_LINES", "20"))

        corpus = QualityCorpusCodec.load_file(corpus_path)
        cases = {case.case_id: case for case in corpus.cases}
        document = PypdfSourceExtractor().extract(
            pdf_path,
            original_name=pdf_path.name,
            created_at=datetime.now(UTC),
        )
        policy = default_policy()
        prompt = policy.prompts[PromptId.DECOMPOSITION]
        configuration = PinnedConfiguration.build(
            model_id=llm_model,
            server_configuration={"base_url": llm_url},
            prompt_version=prompt.version,
            policy_version=policy.version,
            sampling=dict(prompt.generation_parameters),
            budgets={
                "single_input": prompt.input_budget.max_estimated_tokens
                if prompt.input_budget is not None
                else None,
                "evaluation_route_lines": route_lines,
                "evaluation_primary_lines": primary_lines,
                "evaluation_overlap_lines": overlap_lines,
            },
            repeats=3,
        )
        runner = PairedExperimentRunner(corpus)
        if stage == "pilot":
            case_a_id = os.environ["PMI_QUALITY_CASE_A"]
            case_b_id = os.environ["PMI_QUALITY_CASE_B"]
            self.assertIn(case_a_id, cases)
            self.assertIn(case_b_id, cases)
            manifest = runner.manifest(
                ExperimentStage.PILOT,
                (
                    runner.default_pair(
                        ExperimentMode.WINDOWING,
                        cases[case_a_id],
                        configuration,
                    ),
                    runner.default_pair(
                        ExperimentMode.SELECTION,
                        cases[case_b_id],
                        configuration,
                    ),
                ),
            )
        elif stage in {"full-a", "full-b"}:
            mode = (
                ExperimentMode.WINDOWING
                if stage == "full-a"
                else ExperimentMode.SELECTION
            )
            receipt_payload = json.loads(
                Path(os.environ["PMI_QUALITY_PILOT_RECEIPT"]).read_text(
                    encoding="utf-8"
                )
            )
            threshold_payload = json.loads(
                Path(os.environ["PMI_QUALITY_THRESHOLDS"]).read_text(
                    encoding="utf-8"
                )
            )
            receipt = PilotReceipt(
                pilot_experiment_id=str(
                    receipt_payload["pilot_experiment_id"]
                ),
                run_ids=tuple(receipt_payload["run_ids"]),
                successful=bool(receipt_payload["successful"]),
            )
            thresholds = SemanticThresholds(**threshold_payload)
            manifest = runner.full_manifest(
                tuple(
                    runner.default_pair(mode, case, configuration)
                    for case in corpus.cases
                ),
                pilot_receipt=receipt,
                thresholds=thresholds,
            )
        else:
            self.fail(f"Неизвестный PMI_QUALITY_STAGE={stage}")
        transport_settings = OpenAITransportSettings(
            base_url=llm_url,
            model=llm_model,
            api_key=os.getenv("PMI_LLM_API_KEY"),
            timeout=float(os.getenv("PMI_TIMEOUT", "600")),
            verify_ssl=os.getenv("PMI_INSECURE", "0")
            not in {"1", "true", "yes"},
            no_proxy=os.getenv("PMI_NO_PROXY", "0")
            in {"1", "true", "yes"},
        )
        executor = QualityExperimentExecutor(
            document=document,
            policy=policy,
            transport_factory=lambda: OpenAICompatibleTransport(
                transport_settings
            ),
            output_root=output_root,
            evaluation_route_max_lines=route_lines,
            evaluation_primary_max_lines=primary_lines,
            evaluation_overlap_lines=overlap_lines,
        )

        records = await executor.execute(manifest)

        self.assertEqual(
            len(records),
            4 if stage == "pilot" else len(corpus.cases) * 2 * 3,
        )
        self.assertTrue(
            all(record.status == "valid" for record in records),
            [record.to_dict() for record in records],
        )
        print(output_root / manifest.experiment_id)


if __name__ == "__main__":
    unittest.main()
