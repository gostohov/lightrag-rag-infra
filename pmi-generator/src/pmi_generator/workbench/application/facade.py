from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Generic, Protocol, TypeVar

from ..domain import (
    CardDecision,
    CardDecisionKind,
    GapResolutionMode,
    GapStatus,
    RelatedGap,
    SourceDocument,
    TestCard,
    TextSelection,
)
from .card_population import PopulationResult, PopulationService
from .card_population import AnalystMessage
from .card_history import save_card_revision
from .conversation import (
    AnalystProposalService,
    ConversationAction,
    ConversationAgent,
    ConversationAgentError,
    ConversationContext,
    ConversationGapClosureContext,
    ConversationGapContext,
    ConversationProposalContext,
    ConversationToolCall,
    ConversationToolResult,
    ConversationTurnKind,
    ConversationTurnResult,
    action_effect,
    action_user_label,
    requires_confirmation,
)
from .decomposition import (
    DecompositionBudget,
    DecompositionBudgetPolicy,
    DecompositionRoute,
    DecompositionService,
    WindowPlanError,
    WindowingDecision,
    WindowingPolicy,
    default_windowing_policy,
)
from .exporting import FIELD_LABELS, FullPmiExportService, MarkdownCardRenderer
from .metrics import export_metrics
from .prompting import default_policy
from .gap_investigation import (
    AnalystConfirmation,
    GapArguments,
    GapInvestigationService,
)
from .llm import AttemptDiscardedError
from .range_workspace import (
    RangeWorkspaceController,
    RangeWorkspaceService,
    RangeWorkspaceState,
)
from .refinement import (
    CardDecisionService,
    CardRefinementService,
    RefinementArguments,
    RefinementProposalResult,
)
from .repositories import UnitOfWork
from .selection_review import SelectionReviewService
from .session import SessionEvent, SessionEventKind, SessionService
from .session.diagnostics import (
    export_selection_diagnostics,
    export_session_diagnostics,
)
from .source import SavedSelection, SelectionRangeSummary, SelectionService
from .state import AttemptRecord, AttemptStatus, StoredRecord
from .workflow import (
    CommandKind,
    WorkflowCommand,
    WorkflowReconciler,
    WorkflowRuntime,
    WorkflowStage,
)
from .worker_ports import PromptWorkers


ResultT = TypeVar("ResultT")


@dataclass(frozen=True, slots=True)
class WorkbenchOperation(Generic[ResultT]):
    awaitable: Awaitable[ResultT]
    cancel: Callable[[], object]
    progress: Callable[[], object | None] = lambda: None


class WorkbenchFacade(Protocol):
    document: SourceDocument
    run_dir: Path

    def selections(self) -> tuple[StoredRecord, ...]: ...

    def selection_ranges(self) -> tuple[SelectionRangeSummary, ...]: ...

    def workspace(self, selection_id: str) -> RangeWorkspaceState: ...

    def range_controller(self, selection_id: str) -> RangeWorkspaceController: ...

    def save_selection(
        self,
        section_id: str,
        selection: TextSelection,
        *,
        supersede_selection_ids: tuple[str, ...] = (),
    ) -> SavedSelection: ...

    def assess_decomposition(self, selection: TextSelection) -> DecompositionBudget: ...

    def assess_decomposition_route(
        self,
        selection: TextSelection,
    ) -> WindowingDecision: ...

    def load_selection(self, selection_id: str) -> SavedSelection: ...

    def decompose(self, selection: SavedSelection) -> WorkbenchOperation[Any]: ...

    def skeletons(self, skeleton_ids: tuple[str, ...]) -> tuple[StoredRecord, ...]: ...

    def skeleton(self, skeleton_id: str) -> StoredRecord | None: ...

    def take_skeleton(self, selection_id: str, skeleton_id: str) -> str: ...

    def exclude_skeleton(self, selection_id: str, skeleton_id: str, reason: str) -> None: ...

    def open_card_session(self, selection_id: str, card_id: str) -> str: ...

    def session_for_card(self, selection_id: str, card_id: str) -> str: ...

    def ensure_card_session(self, selection_id: str, card_id: str) -> tuple[str, bool]: ...

    def continuation(self, session_id: str) -> str: ...

    def conversation_context(
        self,
        session_id: str,
        card_id: str,
    ) -> ConversationContext: ...

    def dispatch_conversation_tool(
        self,
        selection: SavedSelection,
        skeleton_id: str,
        session_id: str,
        card_id: str,
        message_id: str,
        tool_call: ConversationToolCall,
    ) -> ConversationToolResult: ...

    def conversation_turn(
        self,
        selection: SavedSelection,
        skeleton_id: str,
        session_id: str,
        card_id: str,
        message_id: str,
    ) -> WorkbenchOperation[ConversationTurnResult]: ...

    def repair_card_coverage(
        self,
        session_id: str,
        card_id: str,
    ) -> PopulationResult: ...

    def populate(
        self,
        selection: SavedSelection,
        skeleton_id: str,
        session_id: str,
        card_id: str,
    ) -> WorkbenchOperation[Any]: ...

    def open_gap_ids(self, card_id: str) -> tuple[str, ...]: ...

    def investigate_gap(
        self,
        selection: SavedSelection,
        session_id: str,
        card_id: str,
        gap_id: str,
        research_question: str | None = None,
        research_message_id: str | None = None,
    ) -> WorkbenchOperation[Any]: ...

    def card(self, card_id: str) -> TestCard | None: ...

    def working_card_snapshot(self, card_id: str) -> str: ...

    def include_card(self, card_id: str) -> object: ...

    def exclude_card(self, card_id: str) -> object: ...

    def propose_refinement(
        self,
        session_id: str,
        card_id: str,
        message_id: str,
        expected_revision: int,
    ) -> WorkbenchOperation[RefinementProposalResult]: ...

    def review_selection(self, selection_id: str) -> WorkbenchOperation[Any]: ...

    def review_record(self, selection_id: str) -> StoredRecord | None: ...

    def accept_review_issues(self, selection_id: str) -> None: ...

    def export_full(self) -> Path: ...

    def export_selection(self, selection_id: str) -> tuple[Path, Path]: ...

    def export_diagnostics(self, session_id: str, card_id: str) -> Path: ...

    def append(
        self,
        session_id: str,
        kind: SessionEventKind,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> int: ...

    def history(self, session_id: str) -> list[SessionEvent]: ...

    def active_attempt(self, session_id: str) -> AttemptRecord | None: ...

    def cancel_operation(self, session_id: str, attempt_id: str) -> None: ...


class WorkbenchApplication:
    def __init__(
        self,
        *,
        document: SourceDocument,
        run_dir: Path,
        uow_factory: Callable[[], UnitOfWork],
        workflow: WorkflowRuntime,
        sessions: SessionService,
        workers: PromptWorkers,
        next_id: Callable[[str], str],
        conversation_agent: ConversationAgent | None = None,
        decomposition_budget_policy: DecompositionBudgetPolicy | None = None,
        windowing_policy: WindowingPolicy | None = None,
    ) -> None:
        self.document = document
        self.run_dir = run_dir
        self.uow_factory = uow_factory
        self.workflow = workflow
        self.workers = workers
        self.next_id = next_id
        self.sessions = sessions
        self.conversation_agent = conversation_agent
        self.decomposition_budget_policy = (
            decomposition_budget_policy
            or DecompositionBudgetPolicy.from_prompt_policy(default_policy())
        )
        self.windowing_policy = (
            windowing_policy
            or default_windowing_policy(default_policy())
        )
        self.analyst_proposals = AnalystProposalService(
            uow_factory=uow_factory,
            next_id=next_id,
            clock=sessions.clock,
        )
        self.workspace_service = RangeWorkspaceService(uow_factory=uow_factory)
        self.reconciler = WorkflowReconciler(
            uow_factory=uow_factory,
            workflow=workflow,
        )

    def selections(self) -> tuple[StoredRecord, ...]:
        with self.uow_factory() as uow:
            return tuple(uow.records.list_kind("source_selection"))

    def selection_ranges(self) -> tuple[SelectionRangeSummary, ...]:
        with self.uow_factory() as uow:
            return SelectionService(uow, document=self.document).ranges()

    def recover_workflows(self) -> tuple[str, ...]:
        recovered: list[str] = []
        for selection in self.selections():
            selection_id = selection.record_id
            state = self.workflow.current_state(selection_id)
            if state.stage is WorkflowStage.EMPTY:
                self.reconciler.restore_if_empty(selection_id)
                state = self.workflow.current_state(selection_id)
            active = state.active_attempt
            if active is None:
                continue
            with self.uow_factory() as uow:
                attempt = uow.attempts.get(active.attempt_id)
                windowed_parent = uow.records.get(
                    "decomposition_windowed_attempt",
                    active.attempt_id,
                )
            if (
                attempt is None
                and active.kind == "prompt_1"
                and windowed_parent is not None
                and dict(windowed_parent.payload.get("parent", {})).get(
                    "status"
                )
                == "completed"
            ):
                self.workflow.execute(
                    selection_id,
                    self._completed_attempt_command(selection_id),
                )
                self.reconciler.assert_consistent(selection_id)
                recovered.append(active.attempt_id)
                continue
            if attempt is not None and attempt.status is AttemptStatus.COMPLETED:
                self.workflow.execute(
                    selection_id,
                    self._completed_attempt_command(selection_id),
                )
                self.reconciler.assert_consistent(selection_id)
                recovered.append(active.attempt_id)
                continue
            if attempt is not None and attempt.status in {
                AttemptStatus.ACTIVE,
                AttemptStatus.RESULT_READY,
                AttemptStatus.APPLYING,
            }:
                continue
            kind = (
                CommandKind.CANCEL_ATTEMPT
                if attempt is not None
                and attempt.status in {AttemptStatus.CANCELLED, AttemptStatus.DISCARDED}
                else CommandKind.FAIL_ATTEMPT
            )
            self.workflow.execute(
                selection_id,
                WorkflowCommand(kind, {"attempt_id": active.attempt_id}),
            )
            recovered.append(active.attempt_id)
        return tuple(sorted(recovered))

    def _completed_attempt_command(self, selection_id: str) -> WorkflowCommand:
        state = self.workflow.current_state(selection_id)
        active = state.active_attempt
        if active is None:
            raise RuntimeError("В workflow нет активной попытки для восстановления")
        with self.uow_factory() as uow:
            if active.kind == "prompt_1":
                decomposition = uow.records.get("decomposition", selection_id)
                if decomposition is None:
                    raise RuntimeError("Завершённый Промпт 1 не сохранил декомпозицию")
                skeleton_ids = sorted(
                    record.record_id
                    for record in uow.records.list_kind("card_skeleton")
                    if record.payload.get("selection_id") == selection_id
                )
                return WorkflowCommand(
                    CommandKind.APPLY_DECOMPOSITION,
                    {
                        "outcome": str(decomposition.payload.get("outcome")),
                        "skeleton_ids": skeleton_ids,
                    },
                )

            card = uow.cards.get(active.card_id) if active.card_id else None
            if active.kind in {"prompt_2", "prompt_3", "refinement"} and card is None:
                raise RuntimeError("Завершённая попытка не сохранила карточку")
            if active.kind == "prompt_2":
                return WorkflowCommand(
                    CommandKind.APPLY_ATTEMPT_RESULT,
                    {
                        "attempt_id": active.attempt_id,
                        "revision": card.revision,
                        "gap_statuses": self._gap_statuses(card),
                        "outcome": "populated",
                    },
                )
            if active.kind == "prompt_3":
                gap_result = uow.records.get(
                    "gap_result",
                    f"{active.card_id}:{active.gap_id}",
                )
                if gap_result is None:
                    raise RuntimeError("Завершённый Промпт 3 не сохранил результат пробела")
                return WorkflowCommand(
                    CommandKind.APPLY_ATTEMPT_RESULT,
                    {
                        "attempt_id": active.attempt_id,
                        "revision": card.revision,
                        "gap_statuses": self._gap_statuses(card),
                        "outcome": str(gap_result.payload.get("outcome")),
                    },
                )
            if active.kind == "refinement":
                checkpoint = state.cards[active.card_id]
                return WorkflowCommand(
                    CommandKind.REFINE_CARD,
                    {
                        "card_id": active.card_id,
                        "revision": card.revision,
                        "outcome": (
                            "no_change"
                            if card.revision == checkpoint.revision
                            else "updated"
                        ),
                        "gap_statuses": self._gap_statuses(card),
                    },
                )
            if active.kind == "prompt_4":
                review = uow.records.get("selection_review", selection_id)
                if review is None:
                    raise RuntimeError("Завершённый Промпт 4 не сохранил проверку")
                return WorkflowCommand(
                    CommandKind.SAVE_RANGE_REVIEW,
                    {
                        "warnings": [
                            str(item["issue_id"])
                            for item in review.payload.get("issues", [])
                        ]
                    },
                )
        raise RuntimeError(f"Неизвестный вид завершённой попытки: {active.kind}")

    def workspace(self, selection_id: str) -> RangeWorkspaceState:
        return self.workspace_service.load(selection_id)

    def range_controller(self, selection_id: str) -> RangeWorkspaceController:
        return RangeWorkspaceController(self.workspace_service, selection_id)

    def save_selection(
        self,
        section_id: str,
        selection: TextSelection,
        *,
        supersede_selection_ids: tuple[str, ...] = (),
    ) -> SavedSelection:
        decision = self.windowing_policy.assess(selection)
        if decision.route is DecompositionRoute.HARD_LIMIT:
            raise WindowPlanError(
                "Selection превышает hard limit windowed Prompt 1: "
                f"{decision.budget.line_count}/{decision.hard_max_lines} строк, "
                f"{decision.budget.estimated_tokens}/"
                f"{decision.hard_max_estimated_tokens} оценочных токенов"
            )
        selection_id = self.next_id("SELECTION")
        with self.uow_factory() as uow:
            SelectionService(uow, document=self.document).save(
                selection_id,
                section_id,
                selection,
                supersede_selection_ids=supersede_selection_ids,
            )
        self._ensure_workflow_selection(selection_id)
        self.reconciler.assert_consistent(selection_id)
        return SavedSelection(
            selection_id,
            section_id,
            selection,
            self.document.metadata.document_version,
            section_id,
        )

    def assess_decomposition(self, selection: TextSelection) -> DecompositionBudget:
        return self.windowing_policy.assess(selection).budget

    def assess_decomposition_route(
        self,
        selection: TextSelection,
    ) -> WindowingDecision:
        return self.windowing_policy.assess(selection)

    def load_selection(self, selection_id: str) -> SavedSelection:
        with self.uow_factory() as uow:
            selection = SelectionService(uow, document=self.document).get(
                selection_id
            )
        if selection is None:
            raise ValueError(f"Диапазон {selection_id} не найден")
        return selection

    def decompose(self, selection: SavedSelection) -> WorkbenchOperation[Any]:
        decision = self.windowing_policy.assess(
            selection.selection,
            selection_id=selection.selection_id,
        )
        if decision.route is DecompositionRoute.HARD_LIMIT:
            raise WindowPlanError(
                "Selection превышает hard limit windowed Prompt 1"
            )
        self._ensure_workflow_selection(selection.selection_id)
        self.reconciler.assert_consistent(selection.selection_id)
        attempt_id = self.next_id("ATTEMPT")
        self._begin_attempt(selection.selection_id, attempt_id, "prompt_1")
        try:
            worker = self.workers.decompose(selection, attempt_id)
        except Exception:
            self._fail_attempt_if_active(selection.selection_id, attempt_id)
            raise

        async def execute() -> object:
            try:
                result = await worker.awaitable
                self.workflow.execute(
                    selection.selection_id,
                    WorkflowCommand(
                        CommandKind.APPLY_DECOMPOSITION,
                        {"outcome": result.outcome, "skeleton_ids": list(result.skeleton_ids)},
                    ),
                )
                self.reconciler.assert_consistent(selection.selection_id)
                return result
            except Exception:
                self._fail_attempt_if_active(selection.selection_id, attempt_id)
                raise

        return WorkbenchOperation(
            execute(),
            lambda: self._cancel_attempt(
                selection.selection_id,
                attempt_id,
                worker.cancel,
            ),
            worker.progress,
        )

    def skeletons(self, skeleton_ids: tuple[str, ...]) -> tuple[StoredRecord, ...]:
        with self.uow_factory() as uow:
            records = tuple(uow.records.get("card_skeleton", item) for item in skeleton_ids)
        return tuple(item for item in records if item is not None)

    def skeleton(self, skeleton_id: str) -> StoredRecord | None:
        with self.uow_factory() as uow:
            return uow.records.get("card_skeleton", skeleton_id)

    def take_skeleton(self, selection_id: str, skeleton_id: str) -> str:
        self.reconciler.assert_consistent(selection_id)
        card_id = self._decomposition_service().take(skeleton_id, author="Аналитик")
        self.workflow.execute(
            selection_id,
            WorkflowCommand(
                CommandKind.TAKE_SKELETON,
                {"skeleton_id": skeleton_id, "card_id": card_id},
            ),
        )
        self.reconciler.assert_consistent(selection_id)
        return card_id

    def exclude_skeleton(self, selection_id: str, skeleton_id: str, reason: str) -> None:
        self.reconciler.assert_consistent(selection_id)
        self._decomposition_service().exclude(
            skeleton_id,
            author="Аналитик",
            reason=reason,
        )
        self.workflow.execute(
            selection_id,
            WorkflowCommand(CommandKind.EXCLUDE_SKELETON, {"skeleton_id": skeleton_id}),
        )
        self.reconciler.assert_consistent(selection_id)

    def open_card_session(self, selection_id: str, card_id: str) -> str:
        session_id = self.next_id("SESSION")
        self.sessions.open(session_id, selection_id, card_id)
        return session_id

    def session_for_card(self, selection_id: str, card_id: str) -> str:
        return self.workspace_service.session_for_card(selection_id, card_id)

    def ensure_card_session(self, selection_id: str, card_id: str) -> tuple[str, bool]:
        try:
            return self.session_for_card(selection_id, card_id), False
        except ValueError:
            return self.open_card_session(selection_id, card_id), True

    def continuation(self, session_id: str) -> str:
        with self.uow_factory() as uow:
            session = uow.sessions.get(session_id)
            card = (
                uow.cards.get(session.card_id)
                if session is not None and session.card_id is not None
                else None
            )
        if card is not None:
            return self._card_continuation(card)
        return self.sessions.resume_route(session_id)

    def conversation_context(
        self,
        session_id: str,
        card_id: str,
    ) -> ConversationContext:
        with self.uow_factory() as uow:
            session = uow.sessions.get(session_id)
            card = uow.cards.get(card_id)
        if session is None or card is None or session.card_id != card_id:
            raise ValueError("Conversation context не относится к текущей карточке")
        open_gap = next(
            (
                gap
                for gap in card.gaps.values()
                if gap.status is GapStatus.OPEN
            ),
            None,
        )
        actions: list[ConversationAction] = [
            ConversationAction.EXPORT_DIAGNOSTICS,
        ]
        continuation = self._card_continuation(card)
        if continuation != "card_decision":
            actions.append(ConversationAction.RESUME)
        if open_gap is not None:
            if open_gap.resolution_mode is GapResolutionMode.SOURCE_FACT:
                actions.append(ConversationAction.RESEARCH_GAP)
            elif (
                open_gap.resolution_mode
                is GapResolutionMode.DESIGN_DECISION
            ):
                actions.append(
                    ConversationAction.PROPOSE_DESIGN_DECISION
                )
            actions.extend(
                (
                    ConversationAction.SUBMIT_ANALYST_ANSWER,
                    ConversationAction.CHANGE_GAP_MODE,
                    ConversationAction.LEAVE_GAP,
                )
            )
        elif card.revision > 0:
            actions.append(ConversationAction.REFINE_CARD)
        pending_proposal = self._pending_analyst_proposal(
            session_id,
            card,
            open_gap,
        )
        if pending_proposal is not None:
            actions.extend(
                (
                    ConversationAction.CONFIRM_ANALYST_ANSWER,
                    ConversationAction.REJECT_ANALYST_ANSWER,
                )
            )
        if card.revision > 0:
            actions.extend(
                (
                    ConversationAction.INCLUDE_CARD,
                    ConversationAction.EXCLUDE_CARD,
                    ConversationAction.EXPORT_PMI,
                )
            )
        recent_events = tuple(
            {
                "kind": event.kind.value,
                "text": event.text,
                "metadata": {
                    key: value
                    for key, value in event.metadata.items()
                    if key
                    in {
                        "gap_id",
                        "outcome",
                        "resolution_mode",
                        "conversation_action",
                    }
                },
            }
            for event in self.sessions.history(session_id)[-12:]
        )
        return ConversationContext(
            session_id=session_id,
            card_id=card.card_id,
            card_revision=card.revision,
            stage=session.current_stage,
            continuation=continuation,
            fields={
                path: {
                    "status": field.status.value,
                    "value": field.value,
                }
                for path, field in card.fields.items()
            },
            open_gap=(
                ConversationGapContext(
                    gap_id=open_gap.gap_id,
                    question=open_gap.question,
                    blocking_reason=open_gap.blocking_reason,
                    allowed_paths=open_gap.allowed_paths,
                    resolution_mode=open_gap.resolution_mode.value,
                    closure_schema_version=(
                        open_gap.closure_contract.schema_version
                    ),
                    closure_requirements=tuple(
                        ConversationGapClosureContext(
                            path=requirement.path,
                            accepted_forms=tuple(
                                form.value
                                for form in requirement.accepted_forms
                            ),
                            residual_question=requirement.residual_question,
                        )
                        for requirement
                        in open_gap.closure_contract.requirements
                    ),
                )
                if open_gap is not None
                else None
            ),
            available_actions=tuple(actions),
            pending_proposal=(
                ConversationProposalContext(
                    proposal_id=pending_proposal.record_id,
                    gap_id=(
                        str(pending_proposal.payload["gap_id"])
                        if pending_proposal.payload.get("gap_id")
                        is not None
                        else None
                    ),
                    source_message_id=str(
                        pending_proposal.payload["source_message_id"]
                    ),
                    expected_revision=int(
                        pending_proposal.payload["expected_revision"]
                    ),
                    values=tuple(
                        dict(item)
                        for item in pending_proposal.payload["values"]
                    ),
                    proposal_kind=str(
                        pending_proposal.payload.get(
                            "proposal_kind",
                            "gap_answer",
                        )
                    ),
                    refinement_arguments=(
                        dict(
                            pending_proposal.payload[
                                "refinement_arguments"
                            ]
                        )
                        if isinstance(
                            pending_proposal.payload.get(
                                "refinement_arguments"
                            ),
                            dict,
                        )
                        else None
                    ),
                    closure_evaluation=(
                        dict(
                            pending_proposal.payload[
                                "closure_evaluation"
                            ]
                        )
                        if isinstance(
                            pending_proposal.payload.get(
                                "closure_evaluation"
                            ),
                            dict,
                        )
                        else None
                    ),
                )
                if pending_proposal is not None
                else None
            ),
            recent_events=recent_events,
        )

    def dispatch_conversation_tool(
        self,
        selection: SavedSelection,
        skeleton_id: str,
        session_id: str,
        card_id: str,
        message_id: str,
        tool_call: ConversationToolCall,
    ) -> ConversationToolResult:
        context = self.conversation_context(session_id, card_id)
        action = tool_call.action
        arguments = tool_call.arguments
        if not isinstance(arguments, dict):
            raise ValueError("Аргументы conversation tool должны быть объектом")
        if action is ConversationAction.CONFIRM_ANALYST_ANSWER:
            repeated = self._confirmed_analyst_answer_retry(
                session_id,
                card_id,
                message_id,
                arguments,
            )
            if repeated is not None:
                return repeated
        if action not in context.available_actions:
            raise ValueError(
                f"Действие «{action_user_label(action)}» недоступно"
            )

        if action is ConversationAction.RESUME:
            self._conversation_keys(arguments, {"expected_revision"})
            self._assert_card_revision(card_id, arguments["expected_revision"])
            route = context.continuation
            if route == "population":
                operation = self.populate(
                    selection,
                    skeleton_id,
                    session_id,
                    card_id,
                )
                return self._conversation_operation(
                    action,
                    "Продолжаю первоначальное заполнение карточки.",
                    operation,
                )
            if route == "gap_investigation":
                gap = self._required_open_gap(card_id)
                if gap.resolution_mode is not GapResolutionMode.SOURCE_FACT:
                    return ConversationToolResult(
                        action,
                        action_effect(action),
                        "Для текущего пробела требуется решение или ответ аналитика.",
                    )
                operation = self.investigate_gap(
                    selection,
                    session_id,
                    card_id,
                    gap.gap_id,
                )
                return self._conversation_operation(
                    action,
                    f"Продолжаю исследование {gap.gap_id}.",
                    operation,
                )
            if route == "coverage_repair":
                self.repair_card_coverage(session_id, card_id)
                return ConversationToolResult(
                    action,
                    action_effect(action),
                    "Восстановлены блокирующие пробелы карточки.",
                )
            return ConversationToolResult(
                action,
                action_effect(action),
                "Карточка ожидает решения о включении или исключении.",
            )

        if action is ConversationAction.RESEARCH_GAP:
            self._conversation_keys(
                arguments,
                {"gap_id", "question", "expected_revision"},
            )
            self._assert_card_revision(card_id, arguments["expected_revision"])
            gap = self._required_open_gap(card_id, str(arguments["gap_id"]))
            question = str(arguments["question"]).strip()
            if not question:
                raise ValueError("Research требует непустой предметный вопрос")
            if gap.resolution_mode is not GapResolutionMode.SOURCE_FACT:
                raise ValueError("Research доступен только для source_fact")
            operation = self.investigate_gap(
                selection,
                session_id,
                card_id,
                gap.gap_id,
                question,
                message_id,
            )
            return self._conversation_operation(
                action,
                f"Исследую {gap.gap_id}: {question}",
                operation,
            )

        if action is ConversationAction.PROPOSE_DESIGN_DECISION:
            self._conversation_keys(arguments, set())
            gap = self._required_open_gap(card_id)
            if (
                gap.resolution_mode
                is not GapResolutionMode.DESIGN_DECISION
            ):
                raise ValueError(
                    "Обсуждение проектного решения доступно только для "
                    "design_decision gap"
                )
            return ConversationToolResult(
                action,
                action_effect(action),
                (
                    "LightRAG доступен в Workbench для поиска фактов "
                    "источника, но не может выбрать проектное значение для "
                    f"текущего пробела {gap.gap_id}. "
                    f"Нужно решение аналитика: {gap.question}"
                ),
            )

        if action is ConversationAction.SUBMIT_ANALYST_ANSWER:
            self._conversation_keys(
                arguments,
                {"gap_id", "values", "expected_revision"},
            )
            self._assert_card_revision(card_id, arguments["expected_revision"])
            proposal = self._propose_analyst_answer(
                session_id,
                card_id,
                str(arguments["gap_id"]),
                message_id,
                arguments["values"],
                int(arguments["expected_revision"]),
            )
            return ConversationToolResult(
                action,
                action_effect(action),
                self._proposal_text(proposal),
            )

        if action is ConversationAction.CONFIRM_ANALYST_ANSWER:
            self._conversation_keys(
                arguments,
                {
                    "proposal_id",
                    "expected_revision",
                    "confirmation_message_id",
                },
            )
            self._assert_card_revision(card_id, arguments["expected_revision"])
            self._require_confirmation(action, arguments, message_id)
            proposal = self._required_pending_analyst_proposal(
                session_id,
                card_id,
                str(arguments["proposal_id"]),
                int(arguments["expected_revision"]),
            )
            if proposal.payload.get("source_message_id") == message_id:
                raise ValueError(
                    "Подтверждение требует отдельное сообщение аналитика"
                )
            self._analyst_message(session_id, card_id, message_id)
            is_refinement = (
                proposal.payload.get(
                    "proposal_kind",
                    "gap_answer",
                )
                == "refinement"
            )
            if is_refinement:
                operation = lambda attempt_id: self._apply_refinement_proposal(
                    session_id,
                    card_id,
                    proposal,
                    message_id,
                    attempt_id,
                )
            else:
                operation = lambda attempt_id: self._resolve_gap_from_analyst(
                    session_id,
                    card_id,
                    str(proposal.payload["gap_id"]),
                    str(proposal.payload["source_message_id"]),
                    proposal.payload["values"],
                    attempt_id,
                    proposal_id=proposal.record_id,
                    confirmation_message_id=message_id,
                    expected_revision=int(
                        proposal.payload["expected_revision"]
                    ),
                )
            result = self._run_conversation_mutation(
                action,
                session_id,
                card_id,
                operation,
            )
            return ConversationToolResult(
                action,
                action_effect(action),
                (
                    "Предложенная доработка подтверждена и применена."
                    if is_refinement
                    else (
                        (
                            "Интерпретация ответа аналитика подтверждена и "
                            f"применена к {result.gap_id}. "
                            "Пробел остаётся открытым. "
                            + " ".join(result.remaining_questions)
                        )
                        if result.remaining_questions
                        else (
                            "Интерпретация ответа аналитика подтверждена и "
                            f"применена к {result.gap_id}."
                        )
                    )
                ),
            )

        if action is ConversationAction.REJECT_ANALYST_ANSWER:
            self._conversation_keys(
                arguments,
                {
                    "proposal_id",
                    "expected_revision",
                    "rejection_message_id",
                },
            )
            self._assert_card_revision(card_id, arguments["expected_revision"])
            if arguments["rejection_message_id"] != message_id:
                raise ValueError(
                    "Отказ требует текущее сообщение аналитика"
                )
            self._analyst_message(session_id, card_id, message_id)
            proposal = self._required_pending_analyst_proposal(
                session_id,
                card_id,
                str(arguments["proposal_id"]),
                int(arguments["expected_revision"]),
            )
            if proposal.payload.get("source_message_id") == message_id:
                raise ValueError(
                    "Отказ требует отдельное сообщение аналитика"
                )
            self.analyst_proposals.transition(
                proposal.record_id,
                "rejected",
                message_id=message_id,
            )
            return ConversationToolResult(
                action,
                action_effect(action),
                "Предложенная интерпретация отклонена; карточка не изменена.",
            )

        if action is ConversationAction.CHANGE_GAP_MODE:
            self._conversation_keys(
                arguments,
                {"gap_id", "resolution_mode", "expected_revision"},
            )
            self._assert_card_revision(card_id, arguments["expected_revision"])
            mode = GapResolutionMode(str(arguments["resolution_mode"]))
            gap_id = str(arguments["gap_id"])
            self._run_conversation_mutation(
                action,
                session_id,
                card_id,
                lambda attempt_id: self._change_gap_mode(
                    session_id,
                    card_id,
                    gap_id,
                    message_id,
                    mode,
                    attempt_id,
                ),
            )
            return ConversationToolResult(
                action,
                action_effect(action),
                f"Тип разрешения {gap_id} изменён на {mode.value}.",
            )

        if action is ConversationAction.LEAVE_GAP:
            self._conversation_keys(
                arguments,
                {
                    "gap_id",
                    "reason",
                    "expected_revision",
                    "confirmation_message_id",
                },
            )
            self._assert_card_revision(card_id, arguments["expected_revision"])
            self._require_confirmation(action, arguments, message_id)
            gap_id = str(arguments["gap_id"])
            reason = str(arguments["reason"]).strip()
            if not reason:
                raise ValueError("Оставленный пробел требует основание")
            self._run_conversation_mutation(
                action,
                session_id,
                card_id,
                lambda attempt_id: self._leave_gap(
                    session_id,
                    card_id,
                    gap_id,
                    message_id,
                    reason,
                    attempt_id,
                ),
            )
            return ConversationToolResult(
                action,
                action_effect(action),
                f"Пробел {gap_id} оставлен открытым: {reason}",
            )

        if action is ConversationAction.REFINE_CARD:
            self._conversation_keys(arguments, {"expected_revision"})
            self._assert_card_revision(card_id, arguments["expected_revision"])
            operation = self.propose_refinement(
                session_id,
                card_id,
                message_id,
                int(arguments["expected_revision"]),
            )
            return self._conversation_operation(
                action,
                "Готовлю предложение доработки карточки.",
                operation,
            )

        if action in {
            ConversationAction.INCLUDE_CARD,
            ConversationAction.EXCLUDE_CARD,
        }:
            self._conversation_keys(
                arguments,
                {"expected_revision", "confirmation_message_id"},
            )
            self._assert_card_revision(card_id, arguments["expected_revision"])
            self._require_confirmation(action, arguments, message_id)
            self._analyst_message(session_id, card_id, message_id)
            decision = (
                self.include_card(card_id)
                if action is ConversationAction.INCLUDE_CARD
                else self.exclude_card(card_id)
            )
            return ConversationToolResult(
                action,
                action_effect(action),
                f"Решение сохранено: {decision.kind.value}.",
            )

        if action is ConversationAction.EXPORT_DIAGNOSTICS:
            self._conversation_keys(arguments, set())
            path = self.export_diagnostics(session_id, card_id)
            return ConversationToolResult(
                action,
                action_effect(action),
                f"Диагностика сессии обновлена:\n{path}",
            )

        if action is ConversationAction.EXPORT_PMI:
            self._conversation_keys(arguments, set())
            path = self.export_full()
            return ConversationToolResult(
                action,
                action_effect(action),
                f"ПМИ экспортирован:\n{path}",
            )

        raise ValueError(
            f"Действие «{action_user_label(action)}» не имеет application handler"
        )

    def conversation_turn(
        self,
        selection: SavedSelection,
        skeleton_id: str,
        session_id: str,
        card_id: str,
        message_id: str,
    ) -> WorkbenchOperation[ConversationTurnResult]:
        if self.conversation_agent is None:
            raise RuntimeError("Conversation agent не настроен")
        message = self._analyst_message(
            session_id,
            card_id,
            message_id,
        )
        attempt_id = self.next_id("ATTEMPT")
        self.sessions.start_operation(
            session_id,
            attempt_id,
            operation="Conversation agent обрабатывает сообщение",
            attempt_number=1,
        )
        operation_lock = Lock()
        planner_active = True
        cancel_requested = False
        child_cancel: Callable[[], object] | None = None

        async def execute() -> ConversationTurnResult:
            nonlocal planner_active, child_cancel
            try:
                context = self.conversation_context(session_id, card_id)
                decision = await self.conversation_agent.decide(
                    context=context,
                    message_id=message_id,
                    user_text=message.text,
                )
                if decision.kind in {
                    ConversationTurnKind.ANSWER,
                    ConversationTurnKind.CLARIFICATION,
                }:
                    with operation_lock:
                        accepted = self.sessions.complete_operation(
                            session_id,
                            attempt_id,
                            summary="Conversation agent выбрал следующий ход",
                            result_event=(
                                SessionEventKind.ASSISTANT,
                                decision.text,
                                {
                                    "attempt_id": attempt_id,
                                    "conversation_response": True,
                                    "turn_kind": decision.kind.value,
                                    "message_id": message_id,
                                },
                            ),
                        )
                        planner_active = False
                    if not accepted:
                        raise AttemptDiscardedError(
                            "Решение conversation agent отброшено после отмены"
                        )
                    return ConversationTurnResult(decision)

                if decision.tool_call is None:
                    raise ConversationAgentError(
                        "Conversation tool decision не содержит tool call"
                    )
                with operation_lock:
                    accepted = self.sessions.complete_operation(
                        session_id,
                        attempt_id,
                        summary="Conversation agent выбрал следующий ход",
                        result_event=(
                            SessionEventKind.WORKBENCH,
                            decision.text,
                            {
                                "attempt_id": attempt_id,
                                "conversation_action": decision.tool_call.action.value,
                                "conversation_arguments": decision.tool_call.arguments,
                                "message_id": message_id,
                            },
                        ),
                    )
                    planner_active = False
                if not accepted:
                    raise AttemptDiscardedError(
                        "Решение conversation agent отброшено после отмены"
                    )
                tool_result = self.dispatch_conversation_tool(
                    selection,
                    skeleton_id,
                    session_id,
                    card_id,
                    message_id,
                    decision.tool_call,
                )
                operation_result: object = None
                if tool_result.awaitable is not None:
                    assert tool_result.cancel is not None
                    with operation_lock:
                        child_cancel = tool_result.cancel
                        cancel_after_handoff = cancel_requested
                    if cancel_after_handoff:
                        tool_result.cancel()
                    operation_result = await tool_result.awaitable
                    with operation_lock:
                        child_cancel = None
                self.sessions.append(
                    session_id,
                    SessionEventKind.WORKBENCH,
                    tool_result.text,
                    {
                        "conversation_action": tool_result.action.value,
                        "effect": tool_result.effect.value,
                        "message_id": message_id,
                        "completed": True,
                    },
                )
                return ConversationTurnResult(
                    decision,
                    tool_result,
                    operation_result,
                )
            except AttemptDiscardedError:
                raise
            except Exception as error:
                active = self.sessions.active_attempt(session_id)
                if active is not None and active.attempt_id == attempt_id:
                    self.sessions.fail_operation(
                        session_id,
                        attempt_id,
                        error=f"Conversation agent не завершил ход: {error}",
                    )
                elif active is None:
                    self.sessions.append(
                        session_id,
                        SessionEventKind.ERROR,
                        f"Conversation tool не завершил ход: {error}",
                        {
                            "attempt_id": attempt_id,
                            "message_id": message_id,
                        },
                    )
                raise

        def cancel() -> None:
            nonlocal planner_active, cancel_requested
            with operation_lock:
                cancel_requested = True
                if planner_active:
                    self.sessions.cancel_operation(session_id, attempt_id)
                    planner_active = False
                    return
                current_child_cancel = child_cancel
            if current_child_cancel is not None:
                current_child_cancel()

        return WorkbenchOperation(execute(), cancel)

    def repair_card_coverage(
        self,
        session_id: str,
        card_id: str,
    ) -> PopulationResult:
        card = self._required_card(card_id)
        self.reconciler.assert_consistent(card.selection_id)
        attempt_id = self.next_id("ATTEMPT")
        self._begin_attempt(
            card.selection_id,
            attempt_id,
            "refinement",
            card_id=card_id,
        )
        domain_applied = False
        try:
            service = PopulationService(
                uow_factory=self.uow_factory,
                next_id=self.next_id,
            )
            with self.uow_factory() as uow:
                result = service.repair_coverage(card_id, uow=uow)
                uow.attempts.save(
                    AttemptRecord(
                        attempt_id=attempt_id,
                        session_id=session_id,
                        stage="coverage_repair",
                        status=AttemptStatus.COMPLETED,
                        payload={"card_id": card_id},
                        updated_at=service.clock(),
                    )
                )
            domain_applied = True
            current = self._required_card(card_id)
            self.workflow.execute(
                card.selection_id,
                WorkflowCommand(
                    CommandKind.REFINE_CARD,
                    {
                        "card_id": card_id,
                        "revision": current.revision,
                        "outcome": "gaps_created",
                        "gap_statuses": self._gap_statuses(current),
                    },
                ),
            )
            self.reconciler.assert_consistent(card.selection_id)
            self.sessions.set_stage(
                session_id,
                "восстановлены блокирующие пробелы",
                active_intent=None,
                continuation="gap_investigation",
            )
            self.sessions.append(
                session_id,
                SessionEventKind.WORKBENCH,
                (
                    "Восстановлены блокирующие пробелы карточки.\n"
                    f"Добавлено: {len(result.open_gap_ids)}"
                ),
            )
            return result
        except Exception:
            if not domain_applied:
                self._fail_attempt_if_active(card.selection_id, attempt_id)
            raise

    def populate(
        self,
        selection: SavedSelection,
        skeleton_id: str,
        session_id: str,
        card_id: str,
    ) -> WorkbenchOperation[Any]:
        self.reconciler.assert_consistent(selection.selection_id)
        attempt_id = self.next_id("ATTEMPT")
        self._begin_attempt(
            selection.selection_id,
            attempt_id,
            "prompt_2",
            card_id=card_id,
        )
        self.sessions.set_stage(
            session_id,
            "выполняется Промпт 2",
            active_intent={"kind": "prompt_2", "attempt_id": attempt_id},
            continuation="population",
        )
        try:
            worker = self.workers.populate(
                selection,
                skeleton_id,
                session_id,
                card_id,
                attempt_id,
            )
        except Exception:
            self._fail_attempt_if_active(selection.selection_id, attempt_id)
            self.sessions.set_stage(
                session_id,
                "первоначальное заполнение не запущено",
                active_intent=None,
                continuation="population",
            )
            raise

        async def execute() -> object:
            try:
                result = await worker.awaitable
                self._apply_card_result(
                    selection.selection_id,
                    attempt_id,
                    card_id,
                    outcome="populated",
                )
                current = self._required_card(card_id)
                self.sessions.set_stage(
                    session_id,
                    "первоначальное заполнение завершено",
                    active_intent=None,
                    continuation=self._card_continuation(current),
                )
                return result
            except Exception:
                self._fail_attempt_if_active(selection.selection_id, attempt_id)
                self.sessions.set_stage(
                    session_id,
                    "первоначальное заполнение не завершено",
                    active_intent=None,
                    continuation="population",
                )
                raise

        return WorkbenchOperation(
            execute(),
            lambda: self._cancel_session_attempt(
                selection.selection_id,
                attempt_id,
                session_id,
                "population",
                worker.cancel,
            ),
        )

    def propose_refinement(
        self,
        session_id: str,
        card_id: str,
        message_id: str,
        expected_revision: int,
    ) -> WorkbenchOperation[RefinementProposalResult]:
        card = self._required_card(card_id)
        if card.revision != expected_revision:
            raise ValueError("Устаревшая ревизия карточки")
        self.reconciler.assert_consistent(card.selection_id)
        attempt_id = self.next_id("ATTEMPT")
        self._begin_attempt(
            card.selection_id,
            attempt_id,
            "refinement",
            card_id=card_id,
        )
        self.sessions.set_stage(
            session_id,
            "готовится предложение доработки",
            active_intent={
                "kind": "refinement_proposal",
                "attempt_id": attempt_id,
            },
            continuation=self._card_continuation(card),
        )
        try:
            worker = self.workers.plan_refinement(
                session_id,
                card_id,
                message_id,
                attempt_id,
                expected_revision,
            )
        except Exception:
            self._fail_attempt_if_active(card.selection_id, attempt_id)
            self.sessions.set_stage(
                session_id,
                "предложение доработки не построено",
                active_intent=None,
                continuation=self._card_continuation(card),
            )
            raise

        async def execute() -> RefinementProposalResult:
            try:
                result = await worker.awaitable
                current = self._required_card(card_id)
                if current.revision != expected_revision:
                    raise ValueError(
                        "Карточка изменилась до сохранения proposal"
                    )
                self.workflow.execute(
                    card.selection_id,
                    WorkflowCommand(
                        CommandKind.REFINE_CARD,
                        {
                            "card_id": card_id,
                            "revision": current.revision,
                            "outcome": "no_change",
                            "gap_statuses": self._gap_statuses(current),
                        },
                    ),
                )
                self.reconciler.assert_consistent(card.selection_id)
                if result.proposal_id is None:
                    self.sessions.append(
                        session_id,
                        SessionEventKind.ASSISTANT,
                        "Доработка не требуется.",
                        {
                            "refinement_outcome": "no_change",
                            "message_id": message_id,
                        },
                    )
                    stage = "доработка не требуется"
                else:
                    with self.uow_factory() as uow:
                        proposal = uow.records.get(
                            "analyst_answer_proposal",
                            result.proposal_id,
                        )
                    if proposal is None:
                        raise ValueError(
                            "Сохранённая refinement proposal не найдена"
                        )
                    self.sessions.append(
                        session_id,
                        SessionEventKind.ASSISTANT,
                        self._proposal_text(proposal),
                        {
                            "proposal_id": proposal.record_id,
                            "proposal_kind": "refinement",
                            "message_id": message_id,
                        },
                    )
                    stage = "ожидается подтверждение доработки"
                self.sessions.set_stage(
                    session_id,
                    stage,
                    active_intent=None,
                    continuation=self._card_continuation(current),
                )
                return result
            except Exception:
                self._fail_attempt_if_active(
                    card.selection_id,
                    attempt_id,
                )
                self.sessions.set_stage(
                    session_id,
                    "предложение доработки не завершено",
                    active_intent=None,
                    continuation=self._card_continuation(
                        self._required_card(card_id)
                    ),
                )
                raise

        return WorkbenchOperation(
            execute(),
            lambda: self._cancel_session_attempt(
                card.selection_id,
                attempt_id,
                session_id,
                self._card_continuation(card),
                worker.cancel,
            ),
        )

    def open_gap_ids(self, card_id: str) -> tuple[str, ...]:
        card = self.card(card_id)
        if card is None:
            raise ValueError(f"Карточка {card_id} не найдена")
        return tuple(
            gap.gap_id for gap in card.gaps.values() if gap.status is GapStatus.OPEN
        )

    def investigate_gap(
        self,
        selection: SavedSelection,
        session_id: str,
        card_id: str,
        gap_id: str,
        research_question: str | None = None,
        research_message_id: str | None = None,
    ) -> WorkbenchOperation[Any]:
        card = self.card(card_id)
        if card is None or gap_id not in card.gaps:
            raise ValueError(f"Пробел {gap_id} не найден")
        if card.gaps[gap_id].resolution_mode is not GapResolutionMode.SOURCE_FACT:
            raise ValueError(
                "LightRAG доступен только для пробела типа source_fact"
            )
        self.reconciler.assert_consistent(selection.selection_id)
        attempt_id = self.next_id("ATTEMPT")
        self._begin_attempt(
            selection.selection_id,
            attempt_id,
            "prompt_3",
            card_id=card_id,
            gap_id=gap_id,
        )
        self.sessions.set_stage(
            session_id,
            "исследуется связанный пробел",
            active_intent={
                "kind": "prompt_3",
                "attempt_id": attempt_id,
                "gap_id": gap_id,
            },
            continuation="gap_investigation",
        )
        try:
            worker = self.workers.investigate_gap(
                selection,
                session_id,
                card_id,
                gap_id,
                attempt_id,
                research_question,
                research_message_id,
            )
        except Exception:
            self._fail_attempt_if_active(selection.selection_id, attempt_id)
            self.sessions.set_stage(
                session_id,
                "исследование пробела не запущено",
                active_intent=None,
                continuation="gap_investigation",
            )
            raise

        async def execute() -> object:
            try:
                result = await worker.awaitable
                self._apply_card_result(
                    selection.selection_id,
                    attempt_id,
                    card_id,
                    outcome=result.outcome,
                )
                current = self._required_card(card_id)
                self.sessions.set_stage(
                    session_id,
                    (
                        "нужно решение аналитика"
                        if result.outcome != "resolved"
                        else "исследование пробела завершено"
                    ),
                    active_intent=None,
                    continuation=self._card_continuation(current),
                )
                return result
            except Exception:
                self._fail_attempt_if_active(selection.selection_id, attempt_id)
                self.sessions.set_stage(
                    session_id,
                    "исследование пробела не завершено",
                    active_intent=None,
                    continuation="gap_investigation",
                )
                raise

        return WorkbenchOperation(
            execute(),
            lambda: self._cancel_session_attempt(
                selection.selection_id,
                attempt_id,
                session_id,
                "gap_investigation",
                worker.cancel,
            ),
        )

    def card(self, card_id: str) -> TestCard | None:
        with self.uow_factory() as uow:
            return uow.cards.get(card_id)

    def working_card_snapshot(self, card_id: str) -> str:
        card = self._required_card(card_id)
        return MarkdownCardRenderer().render_working(card)

    def include_card(self, card_id: str) -> object:
        card = self._required_card(card_id)
        self.reconciler.assert_consistent(card.selection_id)
        decision = CardDecisionService(uow_factory=self.uow_factory).include(
            card_id,
            author="Аналитик",
        )
        self._save_card_decision(card.selection_id, decision)
        self.reconciler.assert_consistent(card.selection_id)
        self._mark_decision_stage(card.selection_id, card_id, decision.kind.value)
        return decision

    def exclude_card(self, card_id: str) -> object:
        card = self._required_card(card_id)
        self.reconciler.assert_consistent(card.selection_id)
        decision = CardDecisionService(uow_factory=self.uow_factory).exclude(
            card_id,
            author="Аналитик",
        )
        self._save_card_decision(card.selection_id, decision)
        self.reconciler.assert_consistent(card.selection_id)
        self._mark_decision_stage(card.selection_id, card_id, decision.kind.value)
        return decision

    def review_selection(self, selection_id: str) -> WorkbenchOperation[Any]:
        self.reconciler.assert_consistent(selection_id)
        selection = self.load_selection(selection_id)
        attempt_id = self.next_id("ATTEMPT")
        self._begin_attempt(selection_id, attempt_id, "prompt_4")
        try:
            worker = self.workers.review_selection(selection, attempt_id)
        except Exception:
            self._fail_attempt_if_active(selection_id, attempt_id)
            raise

        async def execute() -> object:
            try:
                result = await worker.awaitable
                record = self.review_record(selection_id)
                if record is None:
                    raise RuntimeError("Prompt 4 не сохранил результат проверки")
                self.workflow.execute(
                    selection_id,
                    WorkflowCommand(
                        CommandKind.SAVE_RANGE_REVIEW,
                        {
                            "warnings": [
                                str(item["issue_id"])
                                for item in record.payload.get("issues", [])
                            ]
                        },
                    ),
                )
                self.reconciler.assert_consistent(selection_id)
                return result
            except Exception:
                self._fail_attempt_if_active(selection_id, attempt_id)
                raise

        return WorkbenchOperation(
            execute(),
            lambda: self._cancel_attempt(
                selection_id,
                attempt_id,
                worker.cancel,
            ),
        )

    def review_record(self, selection_id: str) -> StoredRecord | None:
        with self.uow_factory() as uow:
            return uow.records.get("selection_review", selection_id)

    def accept_review_issues(self, selection_id: str) -> None:
        self.reconciler.assert_consistent(selection_id)
        self._review_service().proceed(
            selection_id,
            author="Аналитик",
            reason="Замечания просмотрены и приняты",
        )
        self.workflow.execute(
            selection_id,
            WorkflowCommand(CommandKind.CONTINUE_WITH_ISSUES, {}),
        )
        self.reconciler.assert_consistent(selection_id)

    def export_full(self) -> Path:
        touched = [
            record.record_id
            for record in self.selections()
            if self.workspace(record.record_id).review_current
        ]
        for selection_id in touched:
            self.reconciler.assert_consistent(selection_id)
            self.workflow.execute(
                selection_id,
                WorkflowCommand(CommandKind.REQUEST_EXPORT, {}),
            )
        return FullPmiExportService(
            run_dir=self.run_dir,
            uow_factory=self.uow_factory,
            reviews=self._review_service(),
            renderer=MarkdownCardRenderer(),
        ).export_full()

    def export_selection(self, selection_id: str) -> tuple[Path, Path]:
        self.reconciler.assert_consistent(selection_id)
        self.workflow.execute(
            selection_id,
            WorkflowCommand(CommandKind.REQUEST_EXPORT, {}),
        )
        markdown = FullPmiExportService(
            run_dir=self.run_dir,
            uow_factory=self.uow_factory,
            reviews=self._review_service(),
            renderer=MarkdownCardRenderer(),
        ).export_selection(selection_id)
        diagnostics = export_selection_diagnostics(
            self.run_dir,
            selection_id,
            self.uow_factory,
        )
        export_metrics(self.run_dir, self.uow_factory)
        return markdown, diagnostics

    def export_diagnostics(self, session_id: str, card_id: str) -> Path:
        path = export_session_diagnostics(
            self.run_dir,
            session_id,
            card_id,
            self.uow_factory,
        )
        export_metrics(self.run_dir, self.uow_factory)
        return path

    def append(
        self,
        session_id: str,
        kind: SessionEventKind,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        return self.sessions.append(session_id, kind, text, metadata)

    def history(self, session_id: str) -> list[SessionEvent]:
        return self.sessions.history(session_id)

    def active_attempt(self, session_id: str) -> AttemptRecord | None:
        return self.sessions.active_attempt(session_id)

    def cancel_operation(self, session_id: str, attempt_id: str) -> None:
        self.sessions.cancel_operation(session_id, attempt_id)

    @staticmethod
    def _conversation_keys(
        arguments: dict[str, Any],
        expected: set[str],
    ) -> None:
        if set(arguments) != expected:
            raise ValueError(
                "Аргументы conversation tool имеют неверную структуру"
            )

    @staticmethod
    def _conversation_operation(
        action: ConversationAction,
        text: str,
        operation: WorkbenchOperation[Any],
    ) -> ConversationToolResult:
        return ConversationToolResult(
            action,
            action_effect(action),
            text,
            awaitable=operation.awaitable,
            cancel=operation.cancel,
        )

    def _assert_card_revision(self, card_id: str, expected: object) -> TestCard:
        card = self._required_card(card_id)
        if isinstance(expected, bool) or not isinstance(expected, int):
            raise ValueError("expected_revision должен быть целым числом")
        if card.revision != expected:
            raise ValueError(
                f"Устаревшая ревизия карточки: ожидалась {expected}, "
                f"текущая {card.revision}"
            )
        return card

    def _required_open_gap(
        self,
        card_id: str,
        gap_id: str | None = None,
    ) -> RelatedGap:
        card = self._required_card(card_id)
        if gap_id is not None:
            gap = card.gaps.get(gap_id)
            if gap is None or gap.status is not GapStatus.OPEN:
                raise ValueError(f"Нет открытого пробела {gap_id}")
            return gap
        gap = next(
            (
                item
                for item in card.gaps.values()
                if item.status is GapStatus.OPEN
            ),
            None,
        )
        if gap is None:
            raise ValueError("Карточка не содержит открытого пробела")
        return gap

    def _analyst_message(
        self,
        session_id: str,
        card_id: str,
        message_id: str,
    ) -> AnalystMessage:
        event = next(
            (
                item
                for item in self.sessions.history(session_id)
                if item.kind is SessionEventKind.ANALYST
                and str(
                    item.metadata.get("message_id")
                    or f"MSG_{item.sequence:06d}"
                )
                == message_id
            ),
            None,
        )
        if event is None:
            raise ValueError(
                f"Сообщение аналитика {message_id} недоступно карточке"
            )
        return AnalystMessage(
            message_id=message_id,
            card_id=card_id,
            author=str(event.metadata.get("author") or "Аналитик"),
            text=event.text,
            created_at=event.created_at,
        )

    def _pending_analyst_proposal(
        self,
        session_id: str,
        card: TestCard,
        open_gap: RelatedGap | None,
    ) -> StoredRecord | None:
        return self.analyst_proposals.pending(
            session_id=session_id,
            card_id=card.card_id,
            card_revision=card.revision,
            open_gap_id=open_gap.gap_id if open_gap is not None else None,
        )

    def _confirmed_analyst_answer_retry(
        self,
        session_id: str,
        card_id: str,
        message_id: str,
        arguments: dict[str, object],
    ) -> ConversationToolResult | None:
        expected_keys = {
            "proposal_id",
            "expected_revision",
            "confirmation_message_id",
        }
        if set(arguments) != expected_keys:
            return None
        proposal_id = str(arguments["proposal_id"])
        with self.uow_factory() as uow:
            proposal = uow.records.get(
                AnalystProposalService.KIND,
                proposal_id,
            )
            card = uow.cards.get(card_id)
        if proposal is None or proposal.payload.get("status") != "confirmed":
            return None
        confirmation_message_id = str(
            arguments["confirmation_message_id"]
        )
        exact_retry = (
            proposal.payload.get("proposal_kind", "gap_answer")
            == "gap_answer"
            and proposal.payload.get("session_id") == session_id
            and proposal.payload.get("card_id") == card_id
            and proposal.payload.get("expected_revision")
            == arguments["expected_revision"]
            and proposal.payload.get("confirmation_message_id")
            == confirmation_message_id
            and confirmation_message_id == message_id
            and card is not None
            and proposal.payload.get("applied_revision") == card.revision
        )
        if not exact_retry:
            raise ValueError(
                "Повторное подтверждение не совпадает с применённым proposal"
            )
        self._analyst_message(session_id, card_id, message_id)
        return ConversationToolResult(
            ConversationAction.CONFIRM_ANALYST_ANSWER,
            action_effect(ConversationAction.CONFIRM_ANALYST_ANSWER),
            "Эта интерпретация уже применена; карточка не изменена.",
        )

    def _required_pending_analyst_proposal(
        self,
        session_id: str,
        card_id: str,
        proposal_id: str,
        expected_revision: int,
    ) -> StoredRecord:
        proposal = self.analyst_proposals.require_pending(
            proposal_id=proposal_id,
            session_id=session_id,
            card_id=card_id,
            expected_revision=expected_revision,
        )
        self._analyst_message(
            session_id,
            card_id,
            str(proposal.payload.get("source_message_id")),
        )
        if (
            proposal.payload.get("proposal_kind", "gap_answer")
            == "refinement"
        ):
            raw_arguments = proposal.payload.get(
                "refinement_arguments"
            )
            if not isinstance(raw_arguments, dict):
                raise ValueError(
                    "Refinement proposal не содержит typed arguments"
                )
            RefinementArguments(
                outcome=str(raw_arguments.get("outcome")),
                updates=list(raw_arguments.get("updates") or []),
                gaps=list(raw_arguments.get("gaps") or []),
                reason=str(raw_arguments.get("reason") or ""),
            )
            return proposal
        gap = self._required_open_gap(
            card_id,
            str(proposal.payload.get("gap_id")),
        )
        self._validate_analyst_values(gap, proposal.payload.get("values"))
        return proposal

    def _propose_analyst_answer(
        self,
        session_id: str,
        card_id: str,
        gap_id: str,
        source_message_id: str,
        raw_values: object,
        expected_revision: int,
    ) -> StoredRecord:
        gap = self._required_open_gap(card_id, gap_id)
        values = self._validate_analyst_values(gap, raw_values)
        evaluation = gap.closure_contract.evaluate(
            {
                str(item["path"]): item["value"]
                for item in values
            },
            source_confirmed=False,
            previously_satisfied=gap.closure_satisfied_paths,
        )
        self._analyst_message(session_id, card_id, source_message_id)
        return self.analyst_proposals.create(
            session_id=session_id,
            card_id=card_id,
            gap_id=gap_id,
            source_message_id=source_message_id,
            expected_revision=expected_revision,
            values=values,
            closure_evaluation={
                "outcome": evaluation.outcome.value,
                "satisfied_paths": list(evaluation.satisfied_paths),
                "remaining_paths": list(evaluation.remaining_paths),
                "remaining_questions": list(
                    evaluation.remaining_questions
                ),
            },
        )

    @staticmethod
    def _validate_analyst_values(
        gap: RelatedGap,
        raw_values: object,
    ) -> list[dict[str, object]]:
        if not isinstance(raw_values, list) or not raw_values:
            raise ValueError("Ответ аналитика требует непустой values")
        values: list[dict[str, object]] = []
        paths: set[str] = set()
        for item in raw_values:
            if not isinstance(item, dict) or set(item) != {"path", "value"}:
                raise ValueError("Элемент values имеет неверную структуру")
            path = str(item["path"])
            if path in paths:
                raise ValueError(f"Путь {path} повторён в values")
            paths.add(path)
            values.append(
                {
                    "path": path,
                    "value": item["value"],
                }
            )
        gap.assert_allows([str(item["path"]) for item in values])
        gap.closure_contract.evaluate(
            {
                str(item["path"]): item["value"]
                for item in values
            },
            source_confirmed=False,
            previously_satisfied=gap.closure_satisfied_paths,
        )
        return values

    @staticmethod
    def _proposal_text(proposal: StoredRecord) -> str:
        if (
            proposal.payload.get("proposal_kind", "gap_answer")
            == "refinement"
        ):
            raw = proposal.payload.get("refinement_arguments")
            if not isinstance(raw, dict):
                raise ValueError(
                    "Refinement proposal не содержит typed arguments"
                )
            lines = [
                "Предлагаю следующую доработку карточки:",
                "",
            ]
            for item in raw.get("updates") or []:
                path = str(item["path"])
                value = item["value"]
                rendered = (
                    ", ".join(str(part) for part in value)
                    if isinstance(value, list)
                    else json.dumps(value, ensure_ascii=False)
                    if isinstance(value, dict)
                    else str(value)
                )
                lines.append(f"- {FIELD_LABELS[path]}: {rendered}")
            for item in raw.get("gaps") or []:
                lines.append(
                    f"- Новый пробел: {item['question']}"
                )
            lines.extend(
                [
                    "",
                    "Подтвердите эту доработку или отклоните её.",
                ]
            )
            return "\n".join(lines)
        lines = [
            "Предлагаю применить следующую интерпретацию ответа аналитика:",
            "",
        ]
        for item in proposal.payload["values"]:
            path = str(item["path"])
            value = item["value"]
            rendered = (
                ", ".join(str(part) for part in value)
                if isinstance(value, list)
                else json.dumps(value, ensure_ascii=False)
                if isinstance(value, dict)
                else str(value)
            )
            lines.append(f"- {FIELD_LABELS[path]}: {rendered}")
        closure = proposal.payload.get("closure_evaluation")
        if (
            isinstance(closure, dict)
            and closure.get("outcome") != "satisfied"
        ):
            lines.extend(
                [
                    "",
                    "После подтверждения пробел останется открытым.",
                ]
            )
            lines.extend(
                f"- {question}"
                for question in closure.get("remaining_questions", [])
            )
        lines.extend(
            [
                "",
                "Подтвердите эту интерпретацию или укажите исправление.",
            ]
        )
        return "\n".join(lines)

    def _resolve_gap_from_analyst(
        self,
        session_id: str,
        card_id: str,
        gap_id: str,
        message_id: str,
        raw_values: object,
        attempt_id: str,
        *,
        proposal_id: str | None = None,
        confirmation_message_id: str | None = None,
        expected_revision: int | None = None,
    ) -> object:
        gap = self._required_open_gap(card_id, gap_id)
        typed_values = self._validate_analyst_values(gap, raw_values)
        closure_evaluation = gap.closure_contract.evaluate(
            {
                str(item["path"]): item["value"]
                for item in typed_values
            },
            source_confirmed=False,
            previously_satisfied=gap.closure_satisfied_paths,
        )
        resolved = closure_evaluation.outcome.value == "satisfied"
        values = [
            {
                **item,
                "evidence_id": None,
                "analyst_message_id": message_id,
            }
            for item in typed_values
        ]
        message = self._analyst_message(session_id, card_id, message_id)
        with self.uow_factory() as uow:
            if expected_revision is not None:
                current = uow.cards.get(card_id)
                if current is None or current.revision != expected_revision:
                    raise ValueError(
                        "Ревизия карточки изменилась до подтверждения "
                        "интерпретации"
                    )
            result = GapInvestigationService(
                uow_factory=self.uow_factory,
                next_id=self.next_id,
            ).apply(
                card_id,
                gap_id,
                GapArguments(
                    outcome=(
                        "resolved"
                        if resolved
                        else "partially_resolved"
                    ),
                    updates=values,
                    unknown_fields=(
                        []
                        if resolved
                        else list(closure_evaluation.remaining_paths)
                    ),
                    missing_fact=(
                        None
                        if resolved
                        else " ".join(
                            closure_evaluation.remaining_questions
                        )
                    ),
                    summary=(
                        "Пробел закрыт ответом аналитика"
                        if resolved
                        else "Ответ аналитика применён частично"
                    ),
                    contradictions=[],
                ),
                available_evidence=(),
                analyst_messages=(message,),
                analyst_confirmation=(
                    AnalystConfirmation(
                        proposal_id=proposal_id,
                        source_message_id=message_id,
                        confirmation_message_id=str(
                            confirmation_message_id
                        ),
                        gap_id=gap_id,
                        expected_revision=int(expected_revision),
                        values=tuple(typed_values),
                    )
                    if proposal_id is not None
                    and confirmation_message_id is not None
                    and expected_revision is not None
                    else None
                ),
                uow=uow,
            )
            if proposal_id is not None:
                self.analyst_proposals.transition(
                    proposal_id,
                    "confirmed",
                    message_id=confirmation_message_id,
                    applied_revision=result.revision,
                    uow=uow,
                )
            self._complete_conversation_attempt(uow, attempt_id)
        return result

    def _apply_refinement_proposal(
        self,
        session_id: str,
        card_id: str,
        proposal: StoredRecord,
        confirmation_message_id: str,
        attempt_id: str,
    ) -> object:
        raw_arguments = proposal.payload.get("refinement_arguments")
        if not isinstance(raw_arguments, dict) or set(raw_arguments) != {
            "outcome",
            "updates",
            "gaps",
            "reason",
        }:
            raise ValueError(
                "Refinement proposal содержит неверные arguments"
            )
        source_message_id = str(
            proposal.payload["source_message_id"]
        )
        source_message = self._analyst_message(
            session_id,
            card_id,
            source_message_id,
        )
        arguments = RefinementArguments(
            outcome=str(raw_arguments["outcome"]),
            updates=list(raw_arguments["updates"]),
            gaps=list(raw_arguments["gaps"]),
            reason=str(raw_arguments["reason"]),
        )
        expected_revision = int(
            proposal.payload["expected_revision"]
        )
        with self.uow_factory() as uow:
            card = uow.cards.get(card_id)
            if card is None or card.revision != expected_revision:
                raise ValueError(
                    "Ревизия карточки изменилась до подтверждения "
                    "доработки"
                )
            result = CardRefinementService(
                uow_factory=self.uow_factory,
                next_id=self.next_id,
            ).apply(
                card_id,
                arguments,
                analyst_messages=(source_message,),
                confirmation_message_id=confirmation_message_id,
                proposal_id=proposal.record_id,
                expected_revision=expected_revision,
                uow=uow,
            )
            self.analyst_proposals.transition(
                proposal.record_id,
                "confirmed",
                message_id=confirmation_message_id,
                applied_revision=result.revision,
                uow=uow,
            )
            self._complete_conversation_attempt(uow, attempt_id)
        return result

    def _change_gap_mode(
        self,
        session_id: str,
        card_id: str,
        gap_id: str,
        message_id: str,
        mode: GapResolutionMode,
        attempt_id: str,
    ) -> None:
        message = self._analyst_message(session_id, card_id, message_id)
        with self.uow_factory() as uow:
            card = uow.cards.get(card_id)
            if card is None:
                raise ValueError(f"Карточка {card_id} не найдена")
            card.change_gap_resolution_mode(gap_id, mode)
            uow.cards.save(card)
            save_card_revision(
                uow,
                card,
                reason=f"изменён тип разрешения {gap_id}",
            )
            uow.records.save(
                StoredRecord(
                    "gap_mode_decision",
                    f"{card_id}:{gap_id}:r{card.revision:06d}",
                    {
                        "card_id": card_id,
                        "gap_id": gap_id,
                        "resolution_mode": mode.value,
                        "analyst_message_id": message.message_id,
                        "reason": message.text,
                        "revision": card.revision,
                    },
                )
            )
            uow.events.append(
                card_id,
                "изменён тип разрешения пробела",
                {
                    "gap_id": gap_id,
                    "resolution_mode": mode.value,
                    "analyst_message_id": message.message_id,
                    "revision": card.revision,
                },
            )
            self._complete_conversation_attempt(uow, attempt_id)

    def _leave_gap(
        self,
        session_id: str,
        card_id: str,
        gap_id: str,
        message_id: str,
        reason: str,
        attempt_id: str,
    ) -> None:
        message = self._analyst_message(session_id, card_id, message_id)
        with self.uow_factory() as uow:
            card = uow.cards.get(card_id)
            if card is None:
                raise ValueError(f"Карточка {card_id} не найдена")
            card.leave_gap_open(gap_id)
            uow.cards.save(card)
            save_card_revision(
                uow,
                card,
                reason=f"пробел {gap_id} оставлен открытым",
            )
            uow.records.save(
                StoredRecord(
                    "gap_leave_decision",
                    f"{card_id}:{gap_id}:r{card.revision:06d}",
                    {
                        "card_id": card_id,
                        "gap_id": gap_id,
                        "reason": reason,
                        "analyst_message_id": message.message_id,
                        "analyst_text": message.text,
                        "revision": card.revision,
                    },
                )
            )
            uow.events.append(
                card_id,
                "пробел оставлен открытым",
                {
                    "gap_id": gap_id,
                    "reason": reason,
                    "analyst_message_id": message.message_id,
                    "revision": card.revision,
                },
            )
            self._complete_conversation_attempt(uow, attempt_id)

    def _run_conversation_mutation(
        self,
        action: ConversationAction,
        session_id: str,
        card_id: str,
        operation: Callable[[str], ResultT],
    ) -> ResultT:
        card = self._required_card(card_id)
        self.reconciler.assert_consistent(card.selection_id)
        attempt_id = self.next_id("ATTEMPT")
        self._begin_attempt(
            card.selection_id,
            attempt_id,
            "refinement",
            card_id=card_id,
        )
        now = self.sessions.clock()
        try:
            with self.uow_factory() as uow:
                uow.attempts.save(
                    AttemptRecord(
                        attempt_id=attempt_id,
                        session_id=session_id,
                        stage=f"conversation:{action.value}",
                        status=AttemptStatus.ACTIVE,
                        payload={
                            "conversation_action": action.value,
                            "card_id": card_id,
                            "expected_revision": card.revision,
                        },
                        updated_at=now,
                    )
                )
                uow.events.append(
                    attempt_id,
                    "conversation mutation выполняется",
                    {
                        "action": action.value,
                        "card_id": card_id,
                        "revision": card.revision,
                    },
                )
            self.sessions.set_stage(
                session_id,
                action_user_label(action),
                active_intent={
                    "kind": "conversation_tool",
                    "attempt_id": attempt_id,
                    "action": action.value,
                },
                continuation=self._card_continuation(card),
            )
            result = operation(attempt_id)
        except Exception:
            self._fail_conversation_mutation(
                card.selection_id,
                session_id,
                attempt_id,
                card_id,
            )
            raise
        self._sync_conversation_mutation(
            session_id,
            card_id,
            outcome="updated",
        )
        return result

    def _complete_conversation_attempt(
        self,
        uow: UnitOfWork,
        attempt_id: str,
    ) -> None:
        attempt = uow.attempts.get(attempt_id)
        if attempt is None or attempt.status is not AttemptStatus.ACTIVE:
            raise ValueError("Conversation mutation attempt не является активной")
        uow.attempts.save(
            attempt.with_status(AttemptStatus.COMPLETED, self.sessions.clock())
        )
        uow.events.append(
            attempt_id,
            "conversation mutation завершена",
            {},
        )

    def _fail_conversation_mutation(
        self,
        selection_id: str,
        session_id: str,
        attempt_id: str,
        card_id: str,
    ) -> None:
        with self.uow_factory() as uow:
            attempt = uow.attempts.get(attempt_id)
            if attempt is not None and attempt.status is AttemptStatus.ACTIVE:
                uow.attempts.save(
                    attempt.with_status(
                        AttemptStatus.FAILED,
                        self.sessions.clock(),
                    )
                )
                uow.events.append(
                    attempt_id,
                    "conversation mutation завершилась ошибкой",
                    {},
                )
        self._fail_attempt_if_active(selection_id, attempt_id)
        self.sessions.set_stage(
            session_id,
            "conversation tool не применён",
            active_intent=None,
            continuation=self._card_continuation(self._required_card(card_id)),
        )

    def _sync_conversation_mutation(
        self,
        session_id: str,
        card_id: str,
        *,
        outcome: str,
    ) -> None:
        card = self._required_card(card_id)
        self.workflow.execute(
            card.selection_id,
            WorkflowCommand(
                CommandKind.REFINE_CARD,
                {
                    "card_id": card_id,
                    "revision": card.revision,
                    "outcome": outcome,
                    "gap_statuses": self._gap_statuses(card),
                },
            ),
        )
        self.reconciler.assert_consistent(card.selection_id)
        self.sessions.set_stage(
            session_id,
            "conversation tool применён",
            active_intent=None,
            continuation=self._card_continuation(card),
        )

    def _require_confirmation(
        self,
        action: ConversationAction,
        arguments: dict[str, Any],
        message_id: str,
    ) -> None:
        if not requires_confirmation(action):
            return
        if arguments.get("confirmation_message_id") != message_id:
            raise ValueError(
                f"Действие «{action_user_label(action)}» требует подтверждение "
                "текущим сообщением"
            )

    def _begin_attempt(
        self,
        selection_id: str,
        attempt_id: str,
        attempt_kind: str,
        *,
        card_id: str | None = None,
        gap_id: str | None = None,
    ) -> None:
        payload: dict[str, object] = {
            "attempt_id": attempt_id,
            "attempt_kind": attempt_kind,
        }
        if card_id is not None:
            payload["card_id"] = card_id
        if gap_id is not None:
            payload["gap_id"] = gap_id
        self.workflow.execute(
            selection_id,
            WorkflowCommand(CommandKind.BEGIN_ATTEMPT, payload),
        )

    def _cancel_attempt(
        self,
        selection_id: str,
        attempt_id: str,
        cancel: Callable[[], object],
    ) -> None:
        cancel()
        self.workflow.execute(
            selection_id,
            WorkflowCommand(CommandKind.CANCEL_ATTEMPT, {"attempt_id": attempt_id}),
        )

    def _cancel_session_attempt(
        self,
        selection_id: str,
        attempt_id: str,
        session_id: str,
        continuation: str,
        cancel: Callable[[], object],
    ) -> None:
        self._cancel_attempt(selection_id, attempt_id, cancel)
        self.sessions.set_stage(
            session_id,
            "операция прервана аналитиком",
            active_intent=None,
            continuation=continuation,
        )

    def _fail_attempt_if_active(self, selection_id: str, attempt_id: str) -> None:
        state = self.workflow.current_state(selection_id)
        if state.active_attempt is None or state.active_attempt.attempt_id != attempt_id:
            return
        self.workflow.execute(
            selection_id,
            WorkflowCommand(CommandKind.FAIL_ATTEMPT, {"attempt_id": attempt_id}),
        )

    def _apply_card_result(
        self,
        selection_id: str,
        attempt_id: str,
        card_id: str,
        *,
        outcome: str,
    ) -> None:
        card = self._required_card(card_id)
        self.workflow.execute(
            selection_id,
            WorkflowCommand(
                CommandKind.APPLY_ATTEMPT_RESULT,
                {
                    "attempt_id": attempt_id,
                    "revision": card.revision,
                    "gap_statuses": self._gap_statuses(card),
                    "outcome": outcome,
                },
            ),
        )
        self.reconciler.assert_consistent(selection_id)

    def _save_card_decision(
        self,
        selection_id: str,
        decision: CardDecision,
    ) -> None:
        kinds = {
            CardDecisionKind.INCLUDE: "include",
            CardDecisionKind.INCLUDE_INCOMPLETE: "include_incomplete",
            CardDecisionKind.EXCLUDE: "exclude",
        }
        self.workflow.execute(
            selection_id,
            WorkflowCommand(
                CommandKind.DECIDE_CARD,
                {
                    "card_id": decision.card_id,
                    "decision": kinds[decision.kind],
                    "revision": decision.revision,
                },
            ),
        )

    def _required_card(self, card_id: str) -> TestCard:
        card = self.card(card_id)
        if card is None:
            raise ValueError(f"Карточка {card_id} не найдена")
        return card

    def _mark_decision_stage(
        self,
        selection_id: str,
        card_id: str,
        decision: str,
    ) -> None:
        try:
            session_id = self.session_for_card(selection_id, card_id)
        except ValueError:
            return
        self.sessions.set_stage(
            session_id,
            f"решение сохранено: {decision}",
            active_intent=None,
            continuation="card_decision",
        )

    @staticmethod
    def _gap_statuses(card: TestCard) -> dict[str, str]:
        statuses = {
            GapStatus.OPEN: "open",
            GapStatus.RESOLVED: "resolved",
            GapStatus.LEFT_OPEN: "left_open",
        }
        return {gap_id: statuses[gap.status] for gap_id, gap in card.gaps.items()}

    @staticmethod
    def _card_continuation(card: TestCard) -> str:
        if card.revision == 0:
            return "population"
        if any(gap.status is GapStatus.OPEN for gap in card.gaps.values()):
            return "gap_investigation"
        if (
            not card.is_ready
            and not any(
                gap.status is GapStatus.LEFT_OPEN
                for gap in card.gaps.values()
            )
        ):
            return "coverage_repair"
        return "card_decision"

    def _decomposition_service(self) -> DecompositionService:
        return DecompositionService(
            document=self.document,
            uow_factory=self.uow_factory,
            next_id=self.next_id,
        )

    def _review_service(self) -> SelectionReviewService:
        return SelectionReviewService(
            uow_factory=self.uow_factory,
            workspace=self.workspace_service,
            next_id=self.next_id,
        )

    def _ensure_workflow_selection(self, selection_id: str) -> None:
        state = self.workflow.current_state(selection_id)
        if state.stage is WorkflowStage.EMPTY:
            self.reconciler.restore_if_empty(selection_id)
            return
        if state.selection_id != selection_id:
            raise RuntimeError(f"Workflow {selection_id} относится к другому диапазону")
