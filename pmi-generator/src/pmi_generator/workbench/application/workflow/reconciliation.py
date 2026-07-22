from __future__ import annotations

from collections.abc import Callable

from ...domain import CardDecisionKind, GapStatus
from ..repositories import UnitOfWork
from ..review_status import selection_review_is_current
from .commands import CommandKind, WorkflowCommand
from .models import WorkflowStage
from .ports import WorkflowRuntime


class WorkflowConsistencyError(RuntimeError):
    pass


DECISIONS = {
    CardDecisionKind.INCLUDE: "include",
    CardDecisionKind.INCLUDE_INCOMPLETE: "include_incomplete",
    CardDecisionKind.EXCLUDE: "exclude",
}

GAPS = {
    GapStatus.OPEN: "open",
    GapStatus.RESOLVED: "resolved",
    GapStatus.LEFT_OPEN: "left_open",
}


class WorkflowReconciler:
    def __init__(
        self,
        *,
        uow_factory: Callable[[], UnitOfWork],
        workflow: WorkflowRuntime,
    ) -> None:
        self.uow_factory = uow_factory
        self.workflow = workflow

    def restore_if_empty(self, selection_id: str) -> bool:
        if self.workflow.current_state(selection_id).stage is not WorkflowStage.EMPTY:
            return False
        with self.uow_factory() as uow:
            selection = uow.records.get("source_selection", selection_id)
            decomposition = uow.records.get("decomposition", selection_id)
            skeletons = sorted(
                (
                    record
                    for record in uow.records.list_kind("card_skeleton")
                    if record.payload.get("selection_id") == selection_id
                ),
                key=lambda item: item.record_id,
            )
            cards = {
                card.card_id: card
                for card in uow.cards.list_all()
                if card.selection_id == selection_id
            }
            review = uow.records.get("selection_review", selection_id)
        if selection is None:
            raise WorkflowConsistencyError(
                f"Нельзя восстановить workflow {selection_id}: selection отсутствует"
            )

        self.workflow.execute(
            selection_id,
            WorkflowCommand(
                CommandKind.CONFIRM_SELECTION,
                {"selection_id": selection_id},
            ),
        )
        if decomposition is None:
            self.assert_consistent(selection_id)
            return True

        recovery_attempt = f"RECOVERY_PROMPT_1:{selection_id}"
        self.workflow.execute(
            selection_id,
            WorkflowCommand(
                CommandKind.BEGIN_ATTEMPT,
                {"attempt_id": recovery_attempt, "attempt_kind": "prompt_1"},
            ),
        )
        self.workflow.execute(
            selection_id,
            WorkflowCommand(
                CommandKind.APPLY_DECOMPOSITION,
                {
                    "outcome": str(decomposition.payload.get("outcome")),
                    "skeleton_ids": [item.record_id for item in skeletons],
                },
            ),
        )
        for skeleton in skeletons:
            decision = skeleton.payload.get("decision")
            if decision == "selected":
                card_id = str(skeleton.payload.get("card_id"))
                self.workflow.execute(
                    selection_id,
                    WorkflowCommand(
                        CommandKind.TAKE_SKELETON,
                        {"skeleton_id": skeleton.record_id, "card_id": card_id},
                    ),
                )
            elif decision == "excluded":
                self.workflow.execute(
                    selection_id,
                    WorkflowCommand(
                        CommandKind.EXCLUDE_SKELETON,
                        {"skeleton_id": skeleton.record_id},
                    ),
                )

        for card_id, card in sorted(cards.items()):
            if card.revision > 0:
                attempt_id = f"RECOVERY_PROMPT_2:{card_id}"
                self.workflow.execute(
                    selection_id,
                    WorkflowCommand(
                        CommandKind.BEGIN_ATTEMPT,
                        {
                            "attempt_id": attempt_id,
                            "attempt_kind": "prompt_2",
                            "card_id": card_id,
                        },
                    ),
                )
                self.workflow.execute(
                    selection_id,
                    WorkflowCommand(
                        CommandKind.APPLY_ATTEMPT_RESULT,
                        {
                            "attempt_id": attempt_id,
                            "revision": card.revision,
                            "gap_statuses": {
                                gap_id: GAPS[gap.status]
                                for gap_id, gap in card.gaps.items()
                            },
                            "outcome": "populated",
                        },
                    ),
                )
            if card.decision is not None:
                self.workflow.execute(
                    selection_id,
                    WorkflowCommand(
                        CommandKind.DECIDE_CARD,
                        {
                            "card_id": card_id,
                            "decision": DECISIONS[card.decision.kind],
                            "revision": card.decision.revision,
                        },
                    ),
                )

        current_revisions = {card_id: card.revision for card_id, card in cards.items()}
        if selection_review_is_current(review, cards.values(), selection_id):
            attempt_id = f"RECOVERY_PROMPT_4:{selection_id}"
            warnings = [
                str(item["issue_id"])
                for item in review.payload.get("issues", [])
            ]
            self.workflow.execute(
                selection_id,
                WorkflowCommand(
                    CommandKind.BEGIN_ATTEMPT,
                    {"attempt_id": attempt_id, "attempt_kind": "prompt_4"},
                ),
            )
            self.workflow.execute(
                selection_id,
                WorkflowCommand(CommandKind.SAVE_RANGE_REVIEW, {"warnings": warnings}),
            )
            if warnings and review.payload.get("analyst_decision"):
                self.workflow.execute(
                    selection_id,
                    WorkflowCommand(CommandKind.CONTINUE_WITH_ISSUES, {}),
                )

        self.assert_consistent(selection_id)
        return True

    def assert_consistent(self, selection_id: str) -> None:
        state = self.workflow.current_state(selection_id)
        with self.uow_factory() as uow:
            selection = uow.records.get("source_selection", selection_id)
            decomposition = uow.records.get("decomposition", selection_id)
            skeletons = {
                record.record_id: record
                for record in uow.records.list_kind("card_skeleton")
                if record.payload.get("selection_id") == selection_id
            }
            cards = {
                card.card_id: card
                for card in uow.cards.list_all()
                if card.selection_id == selection_id
            }
            review = uow.records.get("selection_review", selection_id)

        mismatches: list[str] = []
        if selection is None:
            mismatches.append("канонический selection отсутствует")
        if state.selection_id != selection_id:
            mismatches.append("checkpoint относится к другому selection")

        expected_outcome = (
            str(decomposition.payload.get("outcome")) if decomposition else None
        )
        if state.decomposition_outcome != expected_outcome:
            mismatches.append("исход декомпозиции расходится")
        if set(state.skeletons) != set(skeletons):
            mismatches.append("набор каркасов расходится")
        for skeleton_id, record in skeletons.items():
            checkpoint = state.skeletons.get(skeleton_id)
            if checkpoint is None:
                continue
            decision = record.payload.get("decision")
            expected_status = {
                None: "pending",
                "selected": "selected",
                "excluded": "excluded",
            }.get(decision)
            if (
                checkpoint.status != expected_status
                or checkpoint.card_id != record.payload.get("card_id")
            ):
                mismatches.append(f"каркас {skeleton_id} расходится")

        if set(state.cards) != set(cards):
            mismatches.append("набор карточек расходится")
        for card_id, card in cards.items():
            checkpoint = state.cards.get(card_id)
            if checkpoint is None:
                continue
            expected_gaps = tuple(
                sorted((gap_id, GAPS[gap.status]) for gap_id, gap in card.gaps.items())
            )
            expected_decision = DECISIONS[card.decision.kind] if card.decision else None
            if (
                checkpoint.revision != card.revision
                or checkpoint.populated != (card.revision > 0)
                or tuple(sorted(checkpoint.gaps)) != expected_gaps
                or checkpoint.decision != expected_decision
                or checkpoint.decision_revision
                != (card.decision.revision if card.decision else None)
            ):
                mismatches.append(f"карточка {card_id} расходится")

        current_revisions = {card_id: card.revision for card_id, card in cards.items()}
        review_current = selection_review_is_current(
            review,
            cards.values(),
            selection_id,
        )
        if review_current != (state.range_review is not None):
            mismatches.append("проверка диапазона расходится")
        elif review_current and state.range_review is not None and review is not None:
            expected_warnings = tuple(
                str(item.get("issue_id")) for item in review.payload.get("issues", [])
            )
            accepted = bool(review.payload.get("analyst_decision"))
            if (
                state.range_review.revisions
                != tuple(sorted(current_revisions.items()))
                or state.range_review.warnings != expected_warnings
                or state.range_review.accepted_with_issues != accepted
            ):
                mismatches.append("содержимое проверки диапазона расходится")

        if mismatches:
            raise WorkflowConsistencyError(
                f"Workflow checkpoint {selection_id} расходится с domain state: "
                + "; ".join(mismatches)
            )


__all__ = ["WorkflowConsistencyError", "WorkflowReconciler"]
