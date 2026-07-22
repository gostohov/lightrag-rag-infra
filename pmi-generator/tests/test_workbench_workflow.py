from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pmi_generator.workbench.application.workflow import (
    CommandKind,
    WorkflowCommand,
    WorkflowError,
    WorkflowStage,
    WorkflowState,
    apply_command,
)
from pmi_generator.workbench.infrastructure.workflow import SqliteWorkflowRuntime


def command(kind: CommandKind, **payload: object) -> WorkflowCommand:
    return WorkflowCommand(kind=kind, payload=payload)


def state_after_decomposition() -> WorkflowState:
    state = apply_command(
        WorkflowState.empty(),
        command(CommandKind.CONFIRM_SELECTION, selection_id="SELECTION_0001"),
    )
    state = apply_command(
        state,
        command(
            CommandKind.BEGIN_ATTEMPT,
            attempt_id="ATTEMPT_DECOMPOSE",
            attempt_kind="prompt_1",
        ),
    )
    return apply_command(
        state,
        command(
            CommandKind.APPLY_DECOMPOSITION,
            skeleton_ids=["SKELETON_0001", "SKELETON_0002"],
        ),
    )


class WorkflowTransitionTests(unittest.TestCase):
    def test_happy_path_reaches_export_gate(self) -> None:
        state = state_after_decomposition()
        state = apply_command(
            state,
            command(
                CommandKind.TAKE_SKELETON,
                skeleton_id="SKELETON_0001",
                card_id="CARD_0001",
            ),
        )
        state = apply_command(
            state,
            command(CommandKind.EXCLUDE_SKELETON, skeleton_id="SKELETON_0002"),
        )
        state = apply_command(
            state,
            command(
                CommandKind.BEGIN_ATTEMPT,
                attempt_id="ATTEMPT_0001",
                attempt_kind="prompt_2",
                card_id="CARD_0001",
            ),
        )
        state = apply_command(
            state,
            command(
                CommandKind.APPLY_ATTEMPT_RESULT,
                attempt_id="ATTEMPT_0001",
                revision=1,
                gap_statuses={"GAP_0001": "open"},
                outcome="populated",
            ),
        )
        state = apply_command(
            state,
            command(
                CommandKind.BEGIN_ATTEMPT,
                attempt_id="ATTEMPT_0002",
                attempt_kind="prompt_3",
                card_id="CARD_0001",
                gap_id="GAP_0001",
            ),
        )
        state = apply_command(
            state,
            command(
                CommandKind.APPLY_ATTEMPT_RESULT,
                attempt_id="ATTEMPT_0002",
                revision=2,
                gap_statuses={"GAP_0001": "resolved"},
                outcome="resolved",
            ),
        )
        state = apply_command(
            state,
            command(
                CommandKind.DECIDE_CARD,
                card_id="CARD_0001",
                decision="include",
                revision=2,
            ),
        )
        state = apply_command(
            state,
            command(
                CommandKind.BEGIN_ATTEMPT,
                attempt_id="ATTEMPT_REVIEW",
                attempt_kind="prompt_4",
            ),
        )
        state = apply_command(state, command(CommandKind.SAVE_RANGE_REVIEW, warnings=[]))
        state = apply_command(state, command(CommandKind.REQUEST_EXPORT))

        self.assertEqual(state.stage, WorkflowStage.EXPORT_ALLOWED)
        self.assertTrue(state.export_allowed)

    def test_forbidden_transition_does_not_mutate_state(self) -> None:
        state = state_after_decomposition()
        original = state.to_dict()

        with self.assertRaisesRegex(WorkflowError, "Промпт 2"):
            apply_command(
                state,
                command(
                    CommandKind.BEGIN_ATTEMPT,
                    attempt_id="ATTEMPT_0001",
                    attempt_kind="prompt_2",
                    card_id="CARD_0001",
                ),
            )

        self.assertEqual(state.to_dict(), original)

    def test_cancelled_attempt_rejects_late_result(self) -> None:
        state = state_after_decomposition()
        state = apply_command(
            state,
            command(
                CommandKind.TAKE_SKELETON,
                skeleton_id="SKELETON_0001",
                card_id="CARD_0001",
            ),
        )
        state = apply_command(
            state,
            command(
                CommandKind.BEGIN_ATTEMPT,
                attempt_id="ATTEMPT_0001",
                attempt_kind="prompt_2",
                card_id="CARD_0001",
            ),
        )
        state = apply_command(
            state,
            command(CommandKind.CANCEL_ATTEMPT, attempt_id="ATTEMPT_0001"),
        )

        with self.assertRaisesRegex(WorkflowError, "не является активной"):
            apply_command(
                state,
                command(
                    CommandKind.APPLY_ATTEMPT_RESULT,
                    attempt_id="ATTEMPT_0001",
                    gaps=[],
                ),
            )

        self.assertIn("ATTEMPT_0001", state.cancelled_attempt_ids)

    def test_card_revision_invalidates_decision_and_range_review(self) -> None:
        state = self._reviewed_state()
        state = apply_command(
            state,
            command(
                CommandKind.BEGIN_ATTEMPT,
                attempt_id="ATTEMPT_REFINE",
                attempt_kind="refinement",
                card_id="CARD_0001",
            ),
        )

        updated = apply_command(
            state,
            command(
                CommandKind.REFINE_CARD,
                card_id="CARD_0001",
                revision=2,
                outcome="updated",
                gap_statuses={},
            ),
        )

        self.assertIsNone(updated.cards["CARD_0001"].decision)
        self.assertIsNone(updated.range_review)
        self.assertFalse(updated.export_allowed)
        self.assertEqual(updated.cards["CARD_0002"], state.cards["CARD_0002"])

    def test_review_with_warnings_requires_explicit_analyst_decision(self) -> None:
        state = self._decided_state()
        state = apply_command(
            state,
            command(
                CommandKind.BEGIN_ATTEMPT,
                attempt_id="ATTEMPT_REVIEW",
                attempt_kind="prompt_4",
            ),
        )
        state = apply_command(
            state,
            command(CommandKind.SAVE_RANGE_REVIEW, warnings=["Не покрыто условие"]),
        )

        with self.assertRaisesRegex(WorkflowError, "замечания"):
            apply_command(state, command(CommandKind.REQUEST_EXPORT))

        state = apply_command(state, command(CommandKind.CONTINUE_WITH_ISSUES))
        state = apply_command(state, command(CommandKind.REQUEST_EXPORT))
        self.assertTrue(state.export_allowed)

    def _decided_state(self) -> WorkflowState:
        state = state_after_decomposition()
        for skeleton_id, card_id in (
            ("SKELETON_0001", "CARD_0001"),
            ("SKELETON_0002", "CARD_0002"),
        ):
            state = apply_command(
                state,
                command(
                    CommandKind.TAKE_SKELETON,
                    skeleton_id=skeleton_id,
                    card_id=card_id,
                ),
            )
            state = apply_command(
                state,
                command(
                    CommandKind.BEGIN_ATTEMPT,
                    attempt_id=f"ATTEMPT_{card_id}",
                    attempt_kind="prompt_2",
                    card_id=card_id,
                ),
            )
            state = apply_command(
                state,
                command(
                    CommandKind.APPLY_ATTEMPT_RESULT,
                    attempt_id=f"ATTEMPT_{card_id}",
                    revision=1,
                    gap_statuses={},
                    outcome="populated",
                ),
            )
            state = apply_command(
                state,
                command(
                    CommandKind.DECIDE_CARD,
                    card_id=card_id,
                    decision="include",
                    revision=1,
                ),
            )
        return state

    def _reviewed_state(self) -> WorkflowState:
        state = apply_command(
            self._decided_state(),
            command(
                CommandKind.BEGIN_ATTEMPT,
                attempt_id="ATTEMPT_REVIEW",
                attempt_kind="prompt_4",
            ),
        )
        return apply_command(
            state,
            command(CommandKind.SAVE_RANGE_REVIEW, warnings=[]),
        )


class WorkflowCheckpointTests(unittest.TestCase):
    def test_sqlite_checkpoint_resumes_same_stable_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            database = Path(tmp) / "workbench.sqlite3"
            with SqliteWorkflowRuntime(database) as runtime:
                runtime.execute(
                    "SESSION_0001",
                    command(CommandKind.CONFIRM_SELECTION, selection_id="SELECTION_0001"),
                )
                runtime.execute(
                    "SESSION_0001",
                    command(
                        CommandKind.BEGIN_ATTEMPT,
                        attempt_id="ATTEMPT_0001",
                        attempt_kind="prompt_1",
                    ),
                )
                runtime.execute(
                    "SESSION_0001",
                    command(
                        CommandKind.APPLY_DECOMPOSITION,
                        skeleton_ids=["SKELETON_0001"],
                    ),
                )

            with SqliteWorkflowRuntime(database) as runtime:
                restored = runtime.current_state("SESSION_0001")

            self.assertEqual(restored.stage, WorkflowStage.DECOMPOSITION_REVIEW)
            self.assertEqual(restored.skeletons["SKELETON_0001"].status, "pending")

    def test_rejected_command_is_journaled_without_state_change(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            database = Path(tmp) / "workbench.sqlite3"
            with SqliteWorkflowRuntime(database) as runtime:
                state = runtime.execute(
                    "SESSION_0001",
                    command(CommandKind.CONFIRM_SELECTION, selection_id="SELECTION_0001"),
                )
                with self.assertRaises(WorkflowError):
                    runtime.execute(
                        "SESSION_0001",
                        command(CommandKind.REQUEST_EXPORT),
                    )
                restored = runtime.current_state("SESSION_0001")
                journal = runtime.journal("SESSION_0001")

            self.assertEqual(restored, state)
            self.assertEqual(journal[-1]["result"], "rejected")

    def test_human_stage_uses_langgraph_interrupt_and_resume(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            database = Path(tmp) / "workbench.sqlite3"
            with SqliteWorkflowRuntime(database) as runtime:
                runtime.execute(
                    "SELECTION_0001",
                    command(CommandKind.CONFIRM_SELECTION, selection_id="SELECTION_0001"),
                )
                runtime.execute(
                    "SELECTION_0001",
                    command(
                        CommandKind.BEGIN_ATTEMPT,
                        attempt_id="ATTEMPT_0001",
                        attempt_kind="prompt_1",
                    ),
                )
                runtime.execute(
                    "SELECTION_0001",
                    command(
                        CommandKind.APPLY_DECOMPOSITION,
                        skeleton_ids=["SKELETON_0001"],
                    ),
                )

                self.assertTrue(runtime.waiting_for_input("SELECTION_0001"))
                runtime.execute(
                    "SELECTION_0001",
                    command(
                        CommandKind.TAKE_SKELETON,
                        skeleton_id="SKELETON_0001",
                        card_id="CARD_0001",
                    ),
                )
                self.assertTrue(runtime.waiting_for_input("SELECTION_0001"))
                state = runtime.execute(
                    "SELECTION_0001",
                    command(
                        CommandKind.BEGIN_ATTEMPT,
                        attempt_id="ATTEMPT_0002",
                        attempt_kind="prompt_2",
                        card_id="CARD_0001",
                    ),
                )

                self.assertFalse(runtime.waiting_for_input("SELECTION_0001"))
                self.assertEqual(state.stage, WorkflowStage.POPULATING_CARD)

if __name__ == "__main__":
    unittest.main()
