from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pmi_generator.workbench.application.decomposition import (
    DecompositionRoute,
    WindowChildStatus,
    WindowedAttemptError,
    WindowedAttemptState,
    WindowedAttemptStatus,
    WindowPlanError,
    WindowPlanner,
    WindowedDecompositionStore,
    default_windowing_policy,
)
from pmi_generator.workbench.application.source import SavedSelection
from pmi_generator.workbench.application.prompting import default_policy
from pmi_generator.workbench.domain.source import (
    SourceDocument,
    SourcePage,
    SourcePosition,
    SourceSection,
    TextSelection,
)
from pmi_generator.workbench.infrastructure.storage import SqliteUnitOfWork


def selection(line_count: int, *, line_text: str = "source") -> TextSelection:
    positions = tuple(SourcePosition(1, line) for line in range(1, line_count + 1))
    return TextSelection(
        start=positions[0],
        end=positions[-1],
        positions=positions,
        text="\n".join(line_text for _position in positions),
    )


def source_document(
    line_count: int,
    *,
    line_text: str = "source",
) -> SourceDocument:
    split = max(1, line_count // 2)
    return SourceDocument(
        pages=(
            SourcePage(
                1,
                "1",
                tuple(f"{line_text} {line}" for line in range(1, split + 1)),
            ),
            SourcePage(
                2,
                "2",
                tuple(
                    f"{line_text} {line}"
                    for line in range(split + 1, line_count + 1)
                ),
            ),
        ),
        sections=(
            SourceSection("root", "1", "Root", ("1",), (1,)),
            SourceSection(
                "child",
                "1.1",
                "Child",
                ("1", "1.1"),
                (2,),
                parent_section_id="root",
            ),
        ),
    )


def saved_selection(document: SourceDocument) -> SavedSelection:
    selected = document.select(document.positions[0], document.positions[-1])
    return SavedSelection(
        "SELECTION_1",
        "root",
        selected,
        document.metadata.document_version,
        "root",
    )


class WindowingBudgetPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = default_windowing_policy(default_policy())

    def test_route_is_decided_without_an_llm_call(self) -> None:
        single = self.policy.assess(selection(10))
        windowed = self.policy.assess(selection(241))
        hard_limit = self.policy.assess(selection(self.policy.hard_max_lines + 1))

        self.assertEqual(single.route, DecompositionRoute.SINGLE_CALL)
        self.assertEqual(windowed.route, DecompositionRoute.WINDOWED)
        self.assertEqual(hard_limit.route, DecompositionRoute.HARD_LIMIT)

    def test_window_and_hard_limits_are_bounded_and_derived_from_prompt_policy(
        self,
    ) -> None:
        prompt_budget = default_policy().prompts["prompt_1"].input_budget

        self.assertEqual(self.policy.single_call_max_lines, prompt_budget.max_lines)
        self.assertEqual(
            self.policy.single_call_max_estimated_tokens,
            prompt_budget.max_estimated_tokens,
        )
        self.assertEqual(
            self.policy.child_output_tokens,
            8192,
        )
        self.assertEqual(self.policy.child_length_retry_max_tokens, 12_288)
        self.assertEqual(self.policy.child_output_budget_tokens, 3072)
        self.assertEqual(self.policy.semantic_split_max_depth, 2)
        self.assertEqual(self.policy.semantic_split_min_primary_lines, 6)
        self.assertEqual(
            self.policy.semantic_split_max_generation_requests,
            14,
        )
        self.assertLess(
            self.policy.primary_max_lines,
            self.policy.single_call_max_lines // 2,
        )
        self.assertLessEqual(
            self.policy.estimate_child_output(
                self.policy.primary_max_lines,
            ),
            self.policy.child_output_budget_tokens,
        )
        self.assertGreaterEqual(
            self.policy.max_windows * self.policy.primary_max_lines,
            self.policy.hard_max_lines,
        )
        self.assertEqual(self.policy.reconciliation_output_tokens, 2048)
        self.assertEqual(self.policy.reconciliation_max_groups, 64)
        self.assertEqual(self.policy.max_repair_attempts, 1)
        self.assertTrue(self.policy.fingerprint)

    def test_token_budget_can_choose_windowing_before_line_budget(self) -> None:
        decision = self.policy.assess(selection(2, line_text="я" * 30_000))

        self.assertEqual(decision.route, DecompositionRoute.WINDOWED)
        self.assertGreater(
            decision.budget.estimated_tokens,
            decision.budget.max_estimated_tokens,
        )

    def test_output_budget_routes_vdi_229_line_selection_to_windowing(
        self,
    ) -> None:
        decision = self.policy.assess(
            selection(229, line_text="Требование и ожидаемое поведение")
        )

        self.assertEqual(decision.route, DecompositionRoute.WINDOWED)
        self.assertLessEqual(
            decision.budget.estimated_tokens,
            decision.budget.max_estimated_tokens,
        )


class WindowedAttemptStateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.state = WindowedAttemptState.planned(
            parent_attempt_id="ATTEMPT_PARENT",
            selection_id="SELECTION_1",
            document_version="sha256:" + "1" * 64,
            expected_workflow_revision="workflow-revision-1",
            policy_version="windowing-policy-1",
            prompt_version="1.6.0",
            schema_version="semantic-window-result-1",
            window_plan_hash="plan-hash",
            window_ids=("WINDOW_0001", "WINDOW_0002"),
        )

    def test_parent_runs_children_then_reconciliation_and_completion(self) -> None:
        state = self.state.start()
        state = state.start_child("WINDOW_0001", "ATTEMPT_CHILD_1")
        state = state.complete_child("WINDOW_0001", "ATTEMPT_CHILD_1")
        state = state.start_child("WINDOW_0002", "ATTEMPT_CHILD_2")
        state = state.complete_child("WINDOW_0002", "ATTEMPT_CHILD_2")
        state = state.begin_reconciliation()
        state = state.complete()

        self.assertEqual(state.status, WindowedAttemptStatus.COMPLETED)
        self.assertEqual(
            tuple(child.status for child in state.children),
            (WindowChildStatus.COMPLETED, WindowChildStatus.COMPLETED),
        )

    def test_reconciliation_requires_every_child_and_matching_attempt_id(self) -> None:
        state = self.state.start().start_child(
            "WINDOW_0001",
            "ATTEMPT_CHILD_1",
        )

        with self.assertRaisesRegex(WindowedAttemptError, "не совпадает"):
            state.complete_child("WINDOW_0001", "OTHER")
        with self.assertRaisesRegex(WindowedAttemptError, "не завершены"):
            state.begin_reconciliation()

    def test_cancel_is_terminal_and_marks_unfinished_children_cancelled(self) -> None:
        state = self.state.start().start_child(
            "WINDOW_0001",
            "ATTEMPT_CHILD_1",
        )
        cancelled = state.cancel("Отменено аналитиком")

        self.assertEqual(cancelled.status, WindowedAttemptStatus.CANCELLED)
        self.assertEqual(cancelled.stop_reason, "Отменено аналитиком")
        self.assertEqual(
            tuple(child.status for child in cancelled.children),
            (WindowChildStatus.CANCELLED, WindowChildStatus.CANCELLED),
        )
        with self.assertRaisesRegex(WindowedAttemptError, "терминальном"):
            cancelled.start()

    def test_failure_is_terminal_without_completed_parent_result(self) -> None:
        failed = self.state.start().fail("Invalid child result")

        self.assertEqual(failed.status, WindowedAttemptStatus.FAILED)
        self.assertEqual(failed.stop_reason, "Invalid child result")
        with self.assertRaisesRegex(WindowedAttemptError, "reconciliation"):
            failed.complete()

    def test_child_failure_is_bound_to_its_attempt_and_terminates_parent(self) -> None:
        running = self.state.start().start_child(
            "WINDOW_0001",
            "ATTEMPT_CHILD_1",
        )

        with self.assertRaisesRegex(WindowedAttemptError, "не совпадает"):
            running.fail_child("WINDOW_0001", "OTHER", "invalid")

        failed = running.fail_child(
            "WINDOW_0001",
            "ATTEMPT_CHILD_1",
            "Invalid child result",
        )

        self.assertEqual(failed.status, WindowedAttemptStatus.FAILED)
        self.assertEqual(failed.children[0].status, WindowChildStatus.FAILED)
        self.assertIn("WINDOW_0001", failed.stop_reason)

    def test_technical_recovery_can_replace_only_running_child_attempt(self) -> None:
        running = self.state.start().start_child(
            "WINDOW_0001",
            "ATTEMPT_INTERRUPTED",
        )

        recovered = running.recover_child(
            "WINDOW_0001",
            previous_attempt_id="ATTEMPT_INTERRUPTED",
            recovery_attempt_id="ATTEMPT_RECOVERY",
        )

        self.assertEqual(
            recovered.children[0].status,
            WindowChildStatus.RUNNING,
        )
        self.assertEqual(
            recovered.children[0].attempt_id,
            "ATTEMPT_RECOVERY",
        )
        with self.assertRaisesRegex(WindowedAttemptError, "не совпадает"):
            running.recover_child(
                "WINDOW_0001",
                previous_attempt_id="ATTEMPT_OTHER",
                recovery_attempt_id="ATTEMPT_RECOVERY",
            )


class WindowPlannerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = default_windowing_policy(default_policy())

    def test_plan_is_deterministic_and_primary_lines_cover_selection_once(self) -> None:
        document = source_document(300)
        saved = saved_selection(document)
        planner = WindowPlanner(document, self.policy)

        first = planner.build(saved)
        second = planner.build(saved)

        self.assertEqual(first, second)
        self.assertEqual(first.plan_hash, second.plan_hash)
        self.assertEqual(
            len(first.windows),
            (
                len(saved.selection.positions)
                + self.policy.primary_max_lines
                - 1
            )
            // self.policy.primary_max_lines,
        )
        primary = tuple(
            line.position
            for window in first.windows
            for line in window.lines
            if line.primary
        )
        self.assertEqual(primary, saved.selection.positions)
        self.assertEqual(len(primary), len(set(primary)))
        self.assertTrue(
            any(
                not line.primary
                for window in first.windows
                for line in window.lines
            )
        )

    def test_vdi_regression_sizes_batches_for_required_child_output(self) -> None:
        document = source_document(376, line_text="Строка раздела PUT DATA")
        plan = WindowPlanner(document, self.policy).build(
            saved_selection(document)
        )

        self.assertEqual(
            self.policy.CHILD_OUTPUT_FIXED_RESERVE_TOKENS,
            2048,
        )
        self.assertEqual(self.policy.primary_max_lines, 25)
        self.assertEqual(len(plan.windows), 16)
        self.assertEqual(
            sum(len(window.primary_positions) for window in plan.windows),
            376,
        )
        for window in plan.windows:
            self.assertLessEqual(len(window.primary_positions), 25)
            self.assertLessEqual(
                window.estimated_output_tokens,
                window.output_budget_tokens,
            )
            self.assertEqual(window.output_max_tokens, 8192)

    def test_each_window_payload_stays_inside_single_call_budget(self) -> None:
        document = source_document(260, line_text="длинная строка " + "я" * 80)
        saved = saved_selection(document)
        plan = WindowPlanner(document, self.policy).build(saved)

        for window in plan.windows:
            with self.subTest(window=window.window_id):
                assessment = self.policy.assess_window(
                    window.as_selection(),
                    window_id=window.window_id,
                )
                self.assertEqual(
                    assessment.route,
                    DecompositionRoute.SINGLE_CALL,
                )
                self.assertLessEqual(
                    len(window.primary_positions),
                    self.policy.primary_max_lines,
                )
                self.assertLessEqual(
                    len(window.lines),
                    self.policy.single_call_max_lines,
                )

    def test_single_source_line_larger_than_window_budget_is_rejected(self) -> None:
        document = source_document(2, line_text="я" * 40_000)

        with self.assertRaisesRegex(WindowPlanError, "одна строка"):
            WindowPlanner(document, self.policy).build(saved_selection(document))

    def test_plan_carries_exact_source_and_outline_context(self) -> None:
        document = source_document(300)
        plan = WindowPlanner(document, self.policy).build(
            saved_selection(document)
        )
        second_page_window = next(
            window
            for window in plan.windows
            if window.primary_positions[0].page_index == 2
        )

        self.assertEqual(second_page_window.outline_node_id, "child")
        self.assertEqual(second_page_window.outline_path, ("1", "1.1"))
        for line in second_page_window.lines:
            self.assertEqual(line.text, document.line(line.position))

    def test_window_budget_uses_actual_stable_window_id(self) -> None:
        document = source_document(260, line_text="строка " + "я" * 80)
        saved = SavedSelection(
            "SELECTION_" + "X" * 80,
            "root",
            document.select(document.positions[0], document.positions[-1]),
            document.metadata.document_version,
            "root",
        )

        plan = WindowPlanner(document, self.policy).build(saved)

        for window in plan.windows:
            assessment = self.policy.assess_window(
                window.as_selection(),
                window_id=window.window_id,
            )
            self.assertEqual(assessment.route, DecompositionRoute.SINGLE_CALL)
            self.assertEqual(
                window.estimated_tokens,
                assessment.budget.estimated_tokens,
            )


class WindowPlanPersistenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.database_path = Path(self.temporary.name) / "workbench.sqlite3"
        self.document = source_document(300)
        self.saved = saved_selection(self.document)
        self.policy = default_windowing_policy(default_policy())
        self.plan = WindowPlanner(self.document, self.policy).build(self.saved)
        self.store = WindowedDecompositionStore(
            lambda: SqliteUnitOfWork(self.database_path)
        )
        self.state = WindowedAttemptState.planned(
            parent_attempt_id="ATTEMPT_PARENT",
            selection_id=self.saved.selection_id,
            document_version=self.saved.document_version,
            expected_workflow_revision="workflow-revision-1",
            policy_version=self.policy.fingerprint,
            prompt_version=self.policy.prompt_version,
            schema_version=self.policy.candidate_schema_version,
            window_plan_hash=self.plan.plan_hash,
            window_ids=tuple(window.window_id for window in self.plan.windows),
        )

    def test_parent_and_plan_survive_sqlite_restart(self) -> None:
        progressed = (
            self.state.start()
            .start_child(self.plan.windows[0].window_id, "ATTEMPT_CHILD_1")
            .complete_child(self.plan.windows[0].window_id, "ATTEMPT_CHILD_1")
        )
        self.store.save(progressed, self.plan)

        restarted = WindowedDecompositionStore(
            lambda: SqliteUnitOfWork(self.database_path)
        )
        restored_state, restored_plan = restarted.load(
            "ATTEMPT_PARENT",
            self.saved,
        )

        self.assertEqual(restored_state, progressed)
        self.assertEqual(restored_plan, self.plan)

    def test_stale_source_version_rejects_saved_plan(self) -> None:
        self.store.save(self.state, self.plan)
        stale = SavedSelection(
            self.saved.selection_id,
            self.saved.section_id,
            self.saved.selection,
            "sha256:" + "2" * 64,
            self.saved.anchor_outline_node_id,
        )

        with self.assertRaisesRegex(WindowPlanError, "document_version"):
            self.store.load("ATTEMPT_PARENT", stale)

    def test_tampered_plan_hash_and_unknown_schema_are_rejected(self) -> None:
        self.store.save(self.state, self.plan)
        with SqliteUnitOfWork(self.database_path) as uow:
            record = uow.records.get(
                "decomposition_windowed_attempt",
                "ATTEMPT_PARENT",
            )
            payload = dict(record.payload)
            payload["plan"] = {**payload["plan"], "plan_hash": "tampered"}
            uow.records.save(
                type(record)(record.kind, record.record_id, payload)
            )

        with self.assertRaisesRegex(WindowPlanError, "hash"):
            self.store.load("ATTEMPT_PARENT", self.saved)

        self.store.save(self.state, self.plan)
        with SqliteUnitOfWork(self.database_path) as uow:
            record = uow.records.get(
                "decomposition_windowed_attempt",
                "ATTEMPT_PARENT",
            )
            payload = dict(record.payload)
            payload["storage_schema_version"] = 999
            uow.records.save(
                type(record)(record.kind, record.record_id, payload)
            )

        with self.assertRaisesRegex(WindowPlanError, "schema"):
            self.store.load("ATTEMPT_PARENT", self.saved)


if __name__ == "__main__":
    unittest.main()
