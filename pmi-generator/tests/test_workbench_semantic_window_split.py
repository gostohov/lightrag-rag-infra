from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from pmi_generator.workbench.application.decomposition.windowing.plan import (
    DecompositionWindow,
    WindowPlan,
    WindowSourceLine,
)
from pmi_generator.workbench.application.decomposition.windowing.semantic_split import (
    SemanticSubwindowError,
    SemanticSubwindowPlanner,
    SemanticSubwindowState,
    SemanticSubwindowStatus,
    SemanticSubwindowStore,
)
from pmi_generator.workbench.application.decomposition.windowing.semantic import (
    semantic_window_context,
)
from pmi_generator.workbench.domain.source import SourcePosition
from pmi_generator.workbench.infrastructure.storage import (
    InMemoryDatabase,
    InMemoryUnitOfWork,
    SqliteUnitOfWork,
)


def logical_window() -> DecompositionWindow:
    return DecompositionWindow(
        window_id="SELECTION_1:WINDOW:0006",
        index=5,
        lines=tuple(
            WindowSourceLine(
                SourcePosition(310 + ((line - 1) // 50), ((line - 1) % 50) + 1),
                f"source line {line}",
                61 <= line <= 85,
            )
            for line in range(1, 146)
        ),
        global_start=SourcePosition(307, 19),
        global_end=SourcePosition(313, 39),
        outline_node_id="4.21",
        outline_label="4.21. UPDATE RECORD",
        outline_path=("4", "4.21"),
        input_max_lines=240,
        input_max_estimated_tokens=12_000,
        estimated_tokens=6367,
        output_max_tokens=8192,
        output_budget_tokens=3072,
        estimated_output_tokens=3048,
        policy_version="window-policy",
    )


def window_plan(window: DecompositionWindow) -> WindowPlan:
    provisional = WindowPlan(
        schema_version="window-plan-2",
        selection_id="SELECTION_1",
        document_version="document-v1",
        selection_start=window.global_start,
        selection_end=window.global_end,
        policy_version="window-policy",
        windows=(window,),
        plan_hash="pending",
    )
    return replace(provisional, plan_hash=provisional.recompute_hash())


def planner() -> SemanticSubwindowPlanner:
    return SemanticSubwindowPlanner(
        max_depth=2,
        min_primary_lines=6,
        max_generation_requests=14,
        policy_version="semantic-split-policy-1",
        prompt_version="2.1.0",
        contract_version="semantic-window-facts-4",
    )


class SemanticSubwindowPlanTests(unittest.TestCase):
    def setUp(self) -> None:
        self.window = logical_window()
        self.parent_plan = window_plan(self.window)
        self.plan = planner().build(
            parent_attempt_id="ATTEMPT_PARENT",
            logical_child_attempt_id="ATTEMPT_WINDOW_LOGICAL",
            parent_plan=self.parent_plan,
            window=self.window,
        )

    def test_midpoint_tree_is_bounded_and_partitions_primary_positions(self) -> None:
        root = self.plan.root
        first, second = self.plan.children(root.node_id)

        self.assertEqual(len(root.primary_positions), 25)
        self.assertEqual(
            [len(first.primary_positions), len(second.primary_positions)],
            [12, 13],
        )
        self.assertEqual(
            [
                len(item.primary_positions)
                for item in (
                    *self.plan.children(first.node_id),
                    *self.plan.children(second.node_id),
                )
            ],
            [6, 6, 6, 7],
        )
        self.assertEqual(
            first.primary_positions + second.primary_positions,
            root.primary_positions,
        )
        self.assertEqual(self.plan.max_depth, 2)
        self.assertEqual(self.plan.min_primary_lines, 6)
        self.assertEqual(self.plan.max_generation_requests, 14)
        self.assertEqual(self.plan.recompute_hash(), self.plan.plan_hash)

    def test_masked_subwindow_preserves_full_context_and_line_ids(self) -> None:
        first = self.plan.children(self.plan.root.node_id)[0]

        masked = self.plan.masked_window(self.window, first.node_id)

        self.assertEqual(len(masked.lines), 145)
        self.assertEqual(
            [(line.position, line.text) for line in masked.lines],
            [(line.position, line.text) for line in self.window.lines],
        )
        self.assertEqual(masked.primary_positions, first.primary_positions)
        self.assertEqual(
            sum(line.primary for line in masked.lines),
            12,
        )
        context = semantic_window_context(masked)
        self.assertEqual(
            set(context),
            {"outline", "primary_line_ids", "lines"},
        )
        self.assertNotIn("node_id", context["outline"])
        self.assertTrue(
            all(
                set(line) == {"line_id", "text", "primary"}
                for line in context["lines"]
            )
        )

    def test_plan_rejects_tampering_and_wrong_parent_binding(self) -> None:
        payload = self.plan.to_dict()
        payload["max_depth"] = 3

        with self.assertRaisesRegex(
            SemanticSubwindowError,
            "hash",
        ):
            type(self.plan).from_dict(payload)

        with self.assertRaisesRegex(
            SemanticSubwindowError,
            "parent",
        ):
            self.plan.validate_binding(
                parent_attempt_id="OTHER_PARENT",
                logical_child_attempt_id="ATTEMPT_WINDOW_LOGICAL",
                parent_plan=self.parent_plan,
                window=self.window,
            )


class SemanticSubwindowStateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.window = logical_window()
        self.parent_plan = window_plan(self.window)
        self.plan = planner().build(
            parent_attempt_id="ATTEMPT_PARENT",
            logical_child_attempt_id="ATTEMPT_WINDOW_LOGICAL",
            parent_plan=self.parent_plan,
            window=self.window,
        )
        self.state = SemanticSubwindowState.started(
            self.plan,
            root_generation_attempt_id="ATTEMPT_GENERATION_ROOT",
            consumed_generation_requests=2,
        )

    def test_state_reserves_two_requests_per_node_and_is_bounded(self) -> None:
        root = self.plan.root
        first, second = self.plan.children(root.node_id)
        state = self.state.start_node(
            first.node_id,
            "ATTEMPT_SUB_1",
            plan=self.plan,
        )

        self.assertEqual(state.reserved_generation_requests, 4)
        self.assertEqual(
            state.node(first.node_id).status,
            SemanticSubwindowStatus.RUNNING,
        )
        state = state.complete_node(first.node_id, "ATTEMPT_SUB_1")
        state = state.start_node(
            second.node_id,
            "ATTEMPT_SUB_2",
            plan=self.plan,
        )
        state = state.split_node(
            second.node_id,
            "ATTEMPT_SUB_2",
            plan=self.plan,
        )

        self.assertEqual(
            state.node(second.node_id).status,
            SemanticSubwindowStatus.SPLIT,
        )
        self.assertEqual(state.reserved_generation_requests, 6)

    def test_state_rejects_leaf_split_and_generation_budget_overflow(self) -> None:
        first = self.plan.children(self.plan.root.node_id)[0]
        leaf = self.plan.children(first.node_id)[0]
        state = replace(
            self.state,
            reserved_generation_requests=(
                self.plan.max_generation_requests
            ),
        )

        with self.assertRaisesRegex(
            SemanticSubwindowError,
            "generation",
        ):
            state.start_node(
                leaf.node_id,
                "ATTEMPT_OVER_BUDGET",
                plan=self.plan,
            )

        running = self.state.start_node(
            leaf.node_id,
            "ATTEMPT_LEAF",
            plan=self.plan,
        )
        with self.assertRaisesRegex(
            SemanticSubwindowError,
            "делить",
        ):
            running.split_node(
                leaf.node_id,
                "ATTEMPT_LEAF",
                plan=self.plan,
            )


class SemanticSubwindowStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.window = logical_window()
        self.parent_plan = window_plan(self.window)
        self.plan = planner().build(
            parent_attempt_id="ATTEMPT_PARENT",
            logical_child_attempt_id="ATTEMPT_WINDOW_LOGICAL",
            parent_plan=self.parent_plan,
            window=self.window,
        )
        self.state = SemanticSubwindowState.started(
            self.plan,
            root_generation_attempt_id="ATTEMPT_GENERATION_ROOT",
            consumed_generation_requests=2,
        )

    def test_sqlite_round_trip_preserves_immutable_plan_and_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "workbench.sqlite3"
            store = SemanticSubwindowStore(
                lambda: SqliteUnitOfWork(path)
            )

            store.save(self.state, self.plan)
            state, plan = store.load(
                parent_attempt_id="ATTEMPT_PARENT",
                logical_window_id=self.window.window_id,
                parent_plan=self.parent_plan,
                window=self.window,
            )

            self.assertEqual(state, self.state)
            self.assertEqual(plan, self.plan)

    def test_compare_and_swap_rejects_stale_state(self) -> None:
        database = InMemoryDatabase()
        store = SemanticSubwindowStore(
            lambda: InMemoryUnitOfWork(database)
        )
        store.save(self.state, self.plan)
        first = self.plan.children(self.plan.root.node_id)[0]
        updated = self.state.start_node(
            first.node_id,
            "ATTEMPT_SUB_1",
            plan=self.plan,
        )
        store.save(updated, self.plan, expected_state=self.state)

        with self.assertRaisesRegex(
            SemanticSubwindowError,
            "изменился",
        ):
            store.save(
                self.state.cancel("stale"),
                self.plan,
                expected_state=self.state,
            )


if __name__ == "__main__":
    unittest.main()
