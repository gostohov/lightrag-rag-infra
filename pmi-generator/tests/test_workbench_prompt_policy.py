from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from pmi_generator.workbench.application.prompting import (
    PromptId,
    PromptInputBudget,
    PromptPolicy,
    PromptPolicyError,
    RegressionCase,
    RegressionHarness,
    ScriptedPromptModel,
    default_policy,
    load_regression_cases,
    run_live_regression,
)


class PromptPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = default_policy()

    def test_policy_and_prompt_versions_are_in_every_call_description(self) -> None:
        for prompt_id in PromptId:
            with self.subTest(prompt=prompt_id):
                call = self.policy.build_call(prompt_id, self._context(prompt_id))
                self.assertEqual(call.policy_version, self.policy.version)
                self.assertTrue(call.prompt_version)
                self.assertTrue(call.fingerprint)
                self.assertTrue(call.rule_ids)

    def test_every_required_rule_has_regression_example(self) -> None:
        cases = load_regression_cases()
        covered = {rule_id for case in cases for rule_id in case.rule_ids}

        self.assertEqual(set(self.policy.rules), covered)

    def test_each_prompt_receives_only_allowed_rules_and_context(self) -> None:
        for prompt_id in PromptId:
            with self.subTest(prompt=prompt_id):
                spec = self.policy.prompts[prompt_id]
                call = self.policy.build_call(prompt_id, self._context(prompt_id))
                self.assertEqual(set(call.rule_ids), set(spec.rule_ids))
                self.assertEqual(set(call.context), set(spec.allowed_context))

                with self.assertRaisesRegex(PromptPolicyError, "не разрешены"):
                    self.policy.build_call(
                        prompt_id,
                        {**self._context(prompt_id), "чужое_поле": "нельзя"},
                    )

    def test_every_prompt_has_an_explicit_output_token_budget(self) -> None:
        expected = {
            PromptId.DECOMPOSITION: 4096,
            PromptId.DECOMPOSITION_WINDOW: 4096,
            PromptId.DECOMPOSITION_WINDOW_SEMANTIC: 8192,
            PromptId.DECOMPOSITION_SEMANTIC_SYNTHESIS: 4096,
            PromptId.DECOMPOSITION_RECONCILIATION: 2048,
            PromptId.POPULATION: 4096,
            PromptId.GAP_RESEARCH: 1024,
            PromptId.REFINEMENT: 2048,
            PromptId.SELECTION_REVIEW: 4096,
            PromptId.CONVERSATION: 1024,
        }

        for prompt_id, max_tokens in expected.items():
            with self.subTest(prompt=prompt_id):
                call = self.policy.build_call(prompt_id, self._context(prompt_id))
                self.assertEqual(call.generation_parameters["max_tokens"], max_tokens)

        self.assertEqual(
            self.policy.prompts[
                PromptId.DECOMPOSITION_WINDOW_SEMANTIC
            ].length_retry_max_tokens,
            12_288,
        )
        self.assertTrue(
            all(
                spec.length_retry_max_tokens is None
                for prompt_id, spec in self.policy.prompts.items()
                if prompt_id is not PromptId.DECOMPOSITION_WINDOW_SEMANTIC
            )
        )

    def test_length_retry_profile_is_validated_and_fingerprinted(self) -> None:
        semantic = self.policy.prompts[
            PromptId.DECOMPOSITION_WINDOW_SEMANTIC
        ]
        prompts = tuple(self.policy.prompts.values())

        with self.assertRaisesRegex(
            PromptPolicyError,
            "должен превышать",
        ):
            PromptPolicy(
                version=self.policy.version,
                rules=tuple(self.policy.rules.values()),
                prompts=tuple(
                    replace(
                        spec,
                        length_retry_max_tokens=4096,
                    )
                    if spec.prompt_id
                    is PromptId.DECOMPOSITION_WINDOW_SEMANTIC
                    else spec
                    for spec in prompts
                ),
            )

        changed = PromptPolicy(
            version=self.policy.version,
            rules=tuple(self.policy.rules.values()),
            prompts=tuple(
                replace(semantic, length_retry_max_tokens=16_384)
                if spec.prompt_id
                is PromptId.DECOMPOSITION_WINDOW_SEMANTIC
                else spec
                for spec in prompts
            ),
        )
        original_call = self.policy.build_call(
            PromptId.DECOMPOSITION_WINDOW_SEMANTIC,
            self._context(PromptId.DECOMPOSITION_WINDOW_SEMANTIC),
        )
        changed_call = changed.build_call(
            PromptId.DECOMPOSITION_WINDOW_SEMANTIC,
            self._context(PromptId.DECOMPOSITION_WINDOW_SEMANTIC),
        )

        self.assertNotEqual(original_call.fingerprint, changed_call.fingerprint)

    def test_decomposition_has_an_explicit_single_call_input_budget(self) -> None:
        budget = self.policy.prompts[PromptId.DECOMPOSITION].input_budget

        self.assertIsNotNone(budget)
        self.assertEqual(budget.max_lines, 240)
        self.assertEqual(budget.max_estimated_tokens, 12_000)
        self.assertTrue(
            all(
                spec.input_budget is None
                for prompt_id, spec in self.policy.prompts.items()
                if prompt_id is not PromptId.DECOMPOSITION
            )
        )

    def test_input_budget_change_changes_prompt_fingerprint(self) -> None:
        original = self.policy.build_call(
            PromptId.DECOMPOSITION,
            self._context(PromptId.DECOMPOSITION),
        )
        prompts = tuple(
            replace(
                spec,
                input_budget=PromptInputBudget(
                    max_lines=120,
                    max_estimated_tokens=6_000,
                    estimator="decomposition-json-utf8-div4-v1",
                ),
            )
            if spec.prompt_id is PromptId.DECOMPOSITION
            else spec
            for spec in self.policy.prompts.values()
        )
        changed = PromptPolicy(
            version=self.policy.version,
            rules=tuple(self.policy.rules.values()),
            prompts=prompts,
        )

        updated = changed.build_call(
            PromptId.DECOMPOSITION,
            self._context(PromptId.DECOMPOSITION),
        )

        self.assertNotEqual(original.fingerprint, updated.fingerprint)

    def test_prompt_text_change_changes_fingerprint(self) -> None:
        original = self.policy.build_call(PromptId.DECOMPOSITION, self._context(PromptId.DECOMPOSITION))
        changed = self.policy.with_prompt_text(
            PromptId.DECOMPOSITION,
            self.policy.prompts[PromptId.DECOMPOSITION].instruction + "\nНовое правило.",
        )

        updated = changed.build_call(PromptId.DECOMPOSITION, self._context(PromptId.DECOMPOSITION))

        self.assertNotEqual(original.fingerprint, updated.fingerprint)

    def test_prompt_call_can_narrow_tools_and_fingerprint_tracks_wire_contract(self) -> None:
        context = self._context(PromptId.GAP_RESEARCH)
        full = self.policy.build_call(PromptId.GAP_RESEARCH, context)
        narrowed = self.policy.build_call(
            PromptId.GAP_RESEARCH,
            context,
            allowed_tools=("ask_lightrag",),
        )

        self.assertEqual(narrowed.allowed_tools, ("ask_lightrag",))
        self.assertNotEqual(full.fingerprint, narrowed.fingerprint)
        with self.assertRaisesRegex(PromptPolicyError, "не разрешены tools"):
            self.policy.build_call(
                PromptId.GAP_RESEARCH,
                context,
                allowed_tools=("unknown_tool",),
            )
        with self.assertRaisesRegex(PromptPolicyError, "не может быть пустым"):
            self.policy.build_call(
                PromptId.GAP_RESEARCH,
                context,
                allowed_tools=(),
            )
        with self.assertRaisesRegex(PromptPolicyError, "должны быть уникальными"):
            self.policy.build_call(
                PromptId.GAP_RESEARCH,
                context,
                allowed_tools=("ask_lightrag", "ask_lightrag"),
            )

    def test_decomposition_explicitly_forbids_guessing_not_equal_value(self) -> None:
        instruction = self.policy.prompts[PromptId.DECOMPOSITION].instruction

        self.assertIn("не равно X", instruction)
        self.assertIn("input_value=null", instruction)
        self.assertIn("gap kind=input_value", instruction)

    def test_decomposition_preserves_compound_conditions_and_accounts_for_lines(self) -> None:
        instruction = self.policy.prompts[PromptId.DECOMPOSITION].instruction

        self.assertIn("A И B", instruction)
        self.assertIn("не создавай отдельные каркасы", instruction)
        self.assertIn("каждую строку", instruction)
        self.assertIn("line_assessments", instruction)
        self.assertIn("action_ranges", instruction)
        self.assertIn("changed_factor_ranges", instruction)

    def test_window_decomposition_preserves_candidate_invariants(self) -> None:
        instruction = self.policy.prompts[
            PromptId.DECOMPOSITION_WINDOW
        ].instruction

        self.assertIn("input_value=null", instruction)
        self.assertIn("gap kind=input_value", instruction)
        self.assertIn("action=null", instruction)
        self.assertIn("gap kind=action", instruction)
        self.assertIn("Не возвращай primary_line_assessments", instruction)
        self.assertIn("validated source ranges candidates", instruction)
        self.assertIn("outcome=no_local_testable_behavior", instruction)
        self.assertIn("outcome=boundary_dependency", instruction)
        self.assertIn("outcome=candidates", instruction)
        self.assertIn("primary=false", instruction)
        self.assertIn("хотя бы одной primary=true", instruction)
        self.assertIn("только в primary=false overlap", instruction)
        self.assertIn("нужная информация есть в overlap", instruction)
        self.assertIn("достигающим указанной границы окна", instruction)
        self.assertIn("Нумерация строк начинается заново", instruction)
        self.assertIn("отдельный source range", instruction)

    def test_semantic_window_prompt_excludes_domain_structure(self) -> None:
        instruction = self.policy.prompts[
            PromptId.DECOMPOSITION_WINDOW_SEMANTIC
        ].instruction

        self.assertIn("смысловые фрагменты", instruction)
        self.assertIn("opaque line_ids", instruction)
        self.assertIn("Не возвращай roles", instruction)
        self.assertIn("missing_parts", instruction)
        self.assertIn("boundary_needs", instruction)
        self.assertIn("coordinates", instruction)
        self.assertIn("target_paths", instruction)
        self.assertIn("line assessments", instruction)
        self.assertNotIn("input_value=null", instruction)

    def test_semantic_synthesis_owns_card_content_not_technical_fields(
        self,
    ) -> None:
        instruction = self.policy.prompts[
            PromptId.DECOMPOSITION_SEMANTIC_SYNTHESIS
        ].instruction

        self.assertIn("condition", instruction)
        self.assertIn("changed_factor", instruction)
        self.assertIn("fact_ids", instruction)
        self.assertIn("target_facts", instruction)
        self.assertIn("supporting_facts", instruction)
        self.assertNotIn("primary_fragment_ids", instruction)
        self.assertIn("конкретным action", instruction)
        self.assertIn("application создаст typed gap", instruction)
        self.assertIn("Не возвращай missing_parts", instruction)
        self.assertIn("target_paths", instruction)

    def test_reconciliation_returns_only_semantic_case_decision(
        self,
    ) -> None:
        instruction = self.policy.prompts[
            PromptId.DECOMPOSITION_RECONCILIATION
        ].instruction

        self.assertIn("candidate_pair", instruction)
        self.assertIn("сравни A и B", instruction)
        self.assertIn("без candidate IDs", instruction)
        self.assertIn("submit_reconciliation_case", instruction)
        self.assertNotIn("accepted_candidate_ids", instruction)

    def test_population_explains_grounded_values_derivations_and_gaps(self) -> None:
        instruction = self.policy.prompts[PromptId.POPULATION].instruction

        self.assertIn("source_values", instruction)
        self.assertNotIn("analyst_values", instruction)
        self.assertIn("derivations", instruction)
        self.assertIn("evidence_id", instruction)
        self.assertNotIn("analyst_message_id", instruction)
        self.assertIn("подтверждаемый conversation", instruction)
        self.assertIn("gap", instruction)
        self.assertIn("не означает", instruction)
        self.assertIn("одном разделе", instruction)
        self.assertIn("каждое обязательное поле", instruction)
        self.assertIn("хотя бы один ожидаемый результат", instruction)
        self.assertIn("resolution_targets", instruction)
        self.assertIn("Source facts не объединяй", instruction)
        self.assertIn("exact, finite_set или deterministic_rule", instruction)

    def test_conversation_routes_research_by_typed_gap_mode(self) -> None:
        instruction = self.policy.prompts[PromptId.CONVERSATION].instruction

        self.assertIn("LightRAG через research_gap для source_fact", instruction)
        self.assertIn(
            "Не утверждай, что LightRAG технически недоступен",
            instruction,
        )
        self.assertIn("Для design_decision", instruction)
        self.assertIn("propose_design_decision", instruction)

    def test_gap_research_forbids_tool_call_ids_as_provenance(self) -> None:
        instruction = self.policy.prompts[PromptId.GAP_RESEARCH].instruction

        self.assertIn("observations.evidence_ids", instruction)
        self.assertNotIn("evidence или observations.evidence_ids", instruction)
        self.assertIn("ни в evidence_id, ни в analyst_message_id", instruction)
        self.assertIn("analyst_message_id всегда null", instruction)
        self.assertIn("call_id разрешён только для expand_lightrag", instruction)

    @staticmethod
    def _context(prompt_id: PromptId) -> dict[str, object]:
        return {
            key: f"значение {key}"
            for key in default_policy().prompts[prompt_id].allowed_context
        }


class RegressionHarnessTests(unittest.IsolatedAsyncioTestCase):
    async def test_harness_distinguishes_structural_and_semantic_failures(self) -> None:
        cases = [
            RegressionCase(
                case_id="STRUCTURE",
                prompt_id=PromptId.DECOMPOSITION,
                rule_ids=("PMI-INFO-001",),
                context={"selection": "текст"},
                expected_tool="submit_decomposition",
                required_arguments=("outcome",),
                assertions=(),
                scripted_response={"tool_name": "wrong_tool", "arguments": {}},
            ),
            RegressionCase(
                case_id="SEMANTICS",
                prompt_id=PromptId.DECOMPOSITION,
                rule_ids=("PMI-INFO-001",),
                context={"selection": "текст"},
                expected_tool="submit_decomposition",
                required_arguments=("outcome",),
                assertions=({"path": "arguments.outcome", "equals": "no_testable_behavior"},),
                scripted_response={
                    "tool_name": "submit_decomposition",
                    "arguments": {"outcome": "submitted", "skeletons": []},
                },
            ),
        ]
        harness = RegressionHarness(default_policy(), ScriptedPromptModel(cases))

        report = await harness.run(cases)

        self.assertEqual(report.results[0].status, "structural_failure")
        self.assertEqual(report.results[1].status, "semantic_failure")

    async def test_offline_suite_uses_only_scripted_tool_calls(self) -> None:
        cases = load_regression_cases()
        model = ScriptedPromptModel(cases)

        report = await RegressionHarness(default_policy(), model).run(cases)

        self.assertTrue(report.success)
        self.assertEqual(len(model.calls), len(cases))
        self.assertTrue(all(result.status == "passed" for result in report.results))

    async def test_nested_list_assertions_cover_live_no_guess_shape(self) -> None:
        case = next(
            item for item in load_regression_cases()
            if item.case_id == "not_equal_value_without_guess"
        )

        report = await RegressionHarness(
            default_policy(),
            ScriptedPromptModel([case]),
        ).run([case])

        self.assertTrue(report.success)

    async def test_opt_in_run_saves_machine_readable_report(self) -> None:
        case = load_regression_cases()[0]
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "live-report.json"
            report = await run_live_regression(
                default_policy(),
                ScriptedPromptModel([case]),
                [case],
                output=output,
            )

            saved = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(report.mode, "live")
        self.assertEqual(saved["mode"], "live")
        self.assertEqual(saved["policy_version"], default_policy().version)


if __name__ == "__main__":
    unittest.main()
