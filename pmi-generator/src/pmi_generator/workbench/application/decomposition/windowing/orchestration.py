from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, replace
from threading import Event, Lock
from typing import NoReturn

from ....domain.source import SourceDocument
from ...llm import (
    AttemptDiscardedError,
    GenerationLengthError,
    LlmToolRuntime,
)
from ...prompting import PromptPolicy
from ...repositories import UnitOfWork
from ...source import SavedSelection
from ...state import StoredRecord
from ..models import DecompositionResult
from .assembly import WindowAssemblyService
from .candidates import WindowCandidateResult
from .conflicts import ConflictPlan, ConflictPlanner
from .models import (
    DecompositionRoute,
    WindowChildStatus,
    WindowedAttemptState,
    WindowedAttemptStatus,
)
from .persistence import WindowedDecompositionStore
from .plan import DecompositionWindow, WindowPlan, WindowPlanner
from .policy import WindowingPolicy
from .reconciliation import (
    ReconciliationDecision,
)
from .reconciliation_case_flow import ReconciliationCaseFlow
from .reconciliation_cases import (
    ReconciliationCasePlanner,
    ReconciliationCaseService,
)
from .semantic_flow import SemanticWindowFlow
from .semantic_coordinator import SemanticSubwindowCoordinator
from .semantic_split import (
    SemanticCoordinatorStatus,
    SemanticSubwindowError,
    SemanticSubwindowStore,
)
from .semantic import SemanticWindowResult
from .semantic_service import SemanticWindowService
from .synthesis import SemanticSynthesisArguments
from .synthesis_flow import SemanticSynthesisFlow
from .synthesis_service import SemanticSynthesisService


@dataclass(frozen=True, slots=True)
class DecompositionProgress:
    route: str
    completed_windows: int
    total_windows: int
    completed_conflicts: int
    total_conflicts: int
    stage: str


class WindowedDecompositionFlow:
    def __init__(
        self,
        *,
        document: SourceDocument,
        policy: PromptPolicy,
        windowing_policy: WindowingPolicy,
        runtime: LlmToolRuntime,
        uow_factory: Callable[[], UnitOfWork],
        next_id: Callable[[str], str],
    ) -> None:
        self.document = document
        self.policy = policy
        self.windowing_policy = windowing_policy
        self.runtime = runtime
        self.uow_factory = uow_factory
        self.next_id = next_id
        self.store = WindowedDecompositionStore(uow_factory)
        self.fact_service = SemanticWindowService(
            document=document,
            uow_factory=uow_factory,
        )
        self.synthesis_service = SemanticSynthesisService(
            document=document,
            uow_factory=uow_factory,
        )
        self.reconciliation_service = ReconciliationCaseService(
            document=document,
            uow_factory=uow_factory,
        )
        self.reconciliation_case_planner = ReconciliationCasePlanner(
            max_cases=windowing_policy.reconciliation_max_cases,
        )
        self._lock = Lock()
        self._explicit_cancel = Event()
        self._active_attempt_id: str | None = None
        self._selection: SavedSelection | None = None
        self._progress = DecompositionProgress(
            route=DecompositionRoute.WINDOWED.value,
            completed_windows=0,
            total_windows=0,
            completed_conflicts=0,
            total_conflicts=0,
            stage="планирование",
        )

    def progress(self) -> DecompositionProgress:
        with self._lock:
            return self._progress

    async def run(
        self,
        *,
        parent_attempt_id: str,
        session_id: str,
        selection: SavedSelection,
        expected_workflow_revision: str,
    ) -> DecompositionResult:
        self._selection = selection
        if self._explicit_cancel.is_set():
            raise AttemptDiscardedError(
                f"Parent attempt {parent_attempt_id} отменён до запуска"
            )
        decision = self.windowing_policy.assess(
            selection.selection,
            selection_id=selection.selection_id,
        )
        if decision.route is not DecompositionRoute.WINDOWED:
            raise ValueError(
                "Windowed flow доступен только для windowed route"
            )
        restored = self.store.load_optional(parent_attempt_id, selection)
        if restored is None:
            plan = WindowPlanner(
                self.document,
                self.windowing_policy,
            ).build(selection)
            state = WindowedAttemptState.planned(
                parent_attempt_id=parent_attempt_id,
                selection_id=selection.selection_id,
                document_version=selection.document_version,
                expected_workflow_revision=expected_workflow_revision,
                policy_version=self.windowing_policy.fingerprint,
                prompt_version=self.windowing_policy.prompt_version,
                schema_version=self.windowing_policy.candidate_schema_version,
                window_plan_hash=plan.plan_hash,
                window_ids=tuple(
                    window.window_id for window in plan.windows
                ),
            ).start()
            self.store.save(state, plan)
        else:
            state, plan = restored
            self._validate_recovery(
                state,
                expected_workflow_revision=expected_workflow_revision,
            )
        self._set_progress(
            total_windows=len(plan.windows),
            completed_windows=sum(
                child.status is WindowChildStatus.COMPLETED
                for child in state.children
            ),
            stage="анализ диапазона",
        )
        try:
            state, fact_results = await self._run_windows(
                state=state,
                plan=plan,
                session_id=session_id,
            )
            if state.status is WindowedAttemptStatus.RUNNING:
                updated = state.begin_reconciliation()
                self.store.save(
                    updated,
                    plan,
                    expected_state=state,
                )
                state = updated
            self._set_progress(stage="сборка смысловых каркасов")
            results = await self._run_synthesis(
                state=state,
                plan=plan,
                fact_results=fact_results,
                session_id=session_id,
            )
            conflict_plan = ConflictPlanner(
                self.document,
                self.windowing_policy,
            ).build(plan, results)
            self._set_progress(
                stage="согласование результатов",
                total_conflicts=len(conflict_plan.groups),
            )
            decisions = await self._run_reconciliation(
                state=state,
                plan=plan,
                conflict_plan=conflict_plan,
                session_id=session_id,
            )
            self._set_progress(stage="сборка результата")
            current, current_plan = self.store.load(
                parent_attempt_id,
                selection,
            )
            if current != state or current_plan != plan:
                raise AttemptDiscardedError(
                    "Parent attempt изменился до финальной assembly"
                )
            outcome = WindowAssemblyService(
                document=self.document,
                uow_factory=self.uow_factory,
                next_id=self.next_id,
            ).apply(
                selection=selection,
                parent=state,
                plan=plan,
                results=results,
                conflict_plan=conflict_plan,
                decisions=decisions,
            )
            self._set_progress(stage="завершено")
            return outcome.decomposition
        except asyncio.CancelledError:
            raise
        except AttemptDiscardedError:
            raise
        except Exception as error:
            self._fail_parent(parent_attempt_id, selection, str(error))
            raise
        finally:
            with self._lock:
                self._active_attempt_id = None

    def cancel(self, parent_attempt_id: str) -> None:
        self._explicit_cancel.set()
        selection = self._selection
        if selection is not None:
            restored = self.store.load_optional(
                parent_attempt_id,
                selection,
            )
            if restored is not None:
                state, plan = restored
                if state.status not in {
                    WindowedAttemptStatus.COMPLETED,
                    WindowedAttemptStatus.FAILED,
                    WindowedAttemptStatus.CANCELLED,
                }:
                    self.store.save(
                        state.cancel("Отменено аналитиком"),
                        plan,
                        expected_state=state,
                    )
                    split_store = SemanticSubwindowStore(self.uow_factory)
                    for window in plan.windows:
                        restored_split = split_store.load_optional(
                            parent_attempt_id=parent_attempt_id,
                            logical_window_id=window.window_id,
                            parent_plan=plan,
                            window=window,
                        )
                        if restored_split is None:
                            continue
                        split_state, split_plan = restored_split
                        if (
                            split_state.status
                            is SemanticCoordinatorStatus.RUNNING
                        ):
                            split_store.save(
                                split_state.cancel(
                                    "Отменено аналитиком"
                                ),
                                split_plan,
                                expected_state=split_state,
                            )
        with self._lock:
            active_attempt_id = self._active_attempt_id
        if active_attempt_id is not None:
            try:
                self.runtime.cancel(active_attempt_id)
            except AttemptDiscardedError:
                pass

    async def _run_windows(
        self,
        *,
        state: WindowedAttemptState,
        plan: WindowPlan,
        session_id: str,
    ) -> tuple[WindowedAttemptState, tuple[SemanticWindowResult, ...]]:
        results: list[SemanticWindowResult] = []
        for window in plan.windows:
            self._raise_if_cancelled(state.parent_attempt_id)
            child = next(
                item
                for item in state.children
                if item.window_id == window.window_id
            )
            existing = self.fact_service.load(
                window.window_id,
                parent=state,
                plan=plan,
            )
            if existing is not None:
                if child.attempt_id != existing.child_attempt_id:
                    raise ValueError(
                        "Сохранённый child result имеет другой attempt ID"
                    )
                if child.status is WindowChildStatus.RUNNING:
                    updated = state.complete_child(
                        window.window_id,
                        existing.child_attempt_id,
                    )
                    self.store.save(
                        updated,
                        plan,
                        expected_state=state,
                    )
                    state = updated
                elif child.status is not WindowChildStatus.COMPLETED:
                    raise ValueError(
                        "Сохранённый child result несовместим с parent state"
                    )
                results.append(existing)
                self._set_progress(
                    completed_windows=len(results),
                )
                continue

            child_attempt_id = self.next_id("ATTEMPT_WINDOW")
            if child.status is WindowChildStatus.PLANNED:
                updated = state.start_child(
                    window.window_id,
                    child_attempt_id,
                )
            elif (
                child.status is WindowChildStatus.RUNNING
                and child.attempt_id is not None
            ):
                child_attempt_id = child.attempt_id
                updated = state
            else:
                raise ValueError(
                    f"Окно {window.window_id} нельзя продолжить"
                )
            if updated != state:
                self.store.save(updated, plan, expected_state=state)
                state = updated
            semantic_flow = SemanticWindowFlow(
                policy=self.policy,
                runtime=self.runtime,
                service=self.fact_service,
            )
            coordinator = SemanticSubwindowCoordinator(
                policy=self.windowing_policy,
                semantic_flow=semantic_flow,
                service=self.fact_service,
                uow_factory=self.uow_factory,
                next_id=self.next_id,
                set_active_attempt=self._set_active_attempt,
                raise_if_cancelled=lambda: self._raise_if_cancelled(
                    state.parent_attempt_id
                ),
            )
            try:
                split_restored = coordinator.state_store.load_optional(
                    parent_attempt_id=state.parent_attempt_id,
                    logical_window_id=window.window_id,
                    parent_plan=plan,
                    window=window,
                )
            except SemanticSubwindowError as error:
                self._raise_semantic_split_failure(
                    parent=state,
                    logical_window=window,
                    error=error,
                )
            if split_restored is not None:
                result = await self._run_split_coordinator(
                    coordinator=coordinator,
                    parent=state,
                    parent_plan=plan,
                    logical_window=window,
                    logical_child_attempt_id=child_attempt_id,
                    root_generation_attempt_id=None,
                    session_id=session_id,
                )
            else:
                generation_attempt_id = self.next_id(
                    "ATTEMPT_WINDOW_GENERATION"
                )
                self._set_active_attempt(generation_attempt_id)
                try:
                    result = await semantic_flow.run(
                        parent=state,
                        plan=plan,
                        window_id=window.window_id,
                        child_attempt_id=child_attempt_id,
                        generation_attempt_id=generation_attempt_id,
                        session_id=session_id,
                    )
                except GenerationLengthError:
                    self._set_active_attempt(None)
                    result = await self._run_split_coordinator(
                        coordinator=coordinator,
                        parent=state,
                        parent_plan=plan,
                        logical_window=window,
                        logical_child_attempt_id=child_attempt_id,
                        root_generation_attempt_id=generation_attempt_id,
                        session_id=session_id,
                    )
                finally:
                    self._set_active_attempt(None)
            current, _current_plan = self.store.load(
                state.parent_attempt_id,
                self._required_selection(),
            )
            if current != state:
                raise AttemptDiscardedError(
                    "Parent attempt изменился во время child call"
                )
            updated = state.complete_child(
                window.window_id,
                child_attempt_id,
            )
            self.store.save(updated, plan, expected_state=state)
            state = updated
            results.append(result)
            self._set_progress(completed_windows=len(results))
        return state, tuple(results)

    async def _run_split_coordinator(
        self,
        *,
        coordinator: SemanticSubwindowCoordinator,
        parent: WindowedAttemptState,
        parent_plan: WindowPlan,
        logical_window: DecompositionWindow,
        logical_child_attempt_id: str,
        root_generation_attempt_id: str | None,
        session_id: str,
    ) -> SemanticWindowResult:
        try:
            return await coordinator.run(
                parent=parent,
                parent_plan=parent_plan,
                logical_window=logical_window,
                logical_child_attempt_id=logical_child_attempt_id,
                root_generation_attempt_id=root_generation_attempt_id,
                session_id=session_id,
            )
        except SemanticSubwindowError as error:
            self._raise_semantic_split_failure(
                parent=parent,
                logical_window=logical_window,
                error=error,
            )

    def _raise_semantic_split_failure(
        self,
        *,
        parent: WindowedAttemptState,
        logical_window: DecompositionWindow,
        error: SemanticSubwindowError,
    ) -> NoReturn:
        record_id = (
            f"{parent.parent_attempt_id}:{logical_window.window_id}"
        )
        with self.uow_factory() as uow:
            uow.records.save(
                StoredRecord(
                    "decomposition_semantic_subwindow_failure",
                    record_id,
                    {
                        "parent_attempt_id": parent.parent_attempt_id,
                        "logical_window_id": logical_window.window_id,
                        "error": str(error),
                    },
                )
            )
        raise ValueError(
            "Semantic-дробление окна достигло технического предела; "
            "подробности сохранены в диагностике"
        ) from error

    async def _run_synthesis(
        self,
        *,
        state: WindowedAttemptState,
        plan: WindowPlan,
        fact_results: tuple[SemanticWindowResult, ...],
        session_id: str,
    ) -> tuple[WindowCandidateResult, ...]:
        results: list[WindowCandidateResult] = []
        facts_by_window = {
            result.window_id: result for result in fact_results
        }
        for window in plan.windows:
            self._raise_if_cancelled(state.parent_attempt_id)
            existing = self.synthesis_service.load(
                parent=state,
                plan=plan,
                target_window_id=window.window_id,
                fact_results=fact_results,
            )
            if existing is None:
                attempt_id = self.next_id("ATTEMPT_SYNTHESIS")
                if not facts_by_window[window.window_id].fragments:
                    existing = self.synthesis_service.accept(
                        parent=state,
                        plan=plan,
                        target_window_id=window.window_id,
                        attempt_id=attempt_id,
                        fact_results=fact_results,
                        arguments=SemanticSynthesisArguments(candidates=[]),
                        raw_arguments={"candidates": []},
                    )
                else:
                    self._set_active_attempt(attempt_id)
                    existing = await SemanticSynthesisFlow(
                        policy=self.policy,
                        runtime=self.runtime,
                        service=self.synthesis_service,
                    ).run(
                        parent=state,
                        plan=plan,
                        target_window_id=window.window_id,
                        attempt_id=attempt_id,
                        fact_results=fact_results,
                        session_id=session_id,
                    )
                    self._set_active_attempt(None)
            results.append(existing)
        return tuple(results)

    async def _run_reconciliation(
        self,
        *,
        state: WindowedAttemptState,
        plan: WindowPlan,
        conflict_plan: ConflictPlan,
        session_id: str,
    ) -> tuple[ReconciliationDecision, ...]:
        decisions: list[ReconciliationDecision] = []
        for group in conflict_plan.groups:
            self._raise_if_cancelled(state.parent_attempt_id)
            cases = self.reconciliation_case_planner.build(group)
            existing = self.reconciliation_service.load(
                parent_attempt_id=state.parent_attempt_id,
                group=group,
                cases=cases,
                plan_hash=plan.plan_hash,
            )
            if existing is None:
                case_decisions = []
                for case in cases:
                    self._raise_if_cancelled(state.parent_attempt_id)
                    case_decision = self.reconciliation_service.load_case(
                        parent_attempt_id=state.parent_attempt_id,
                        group=group,
                        case=case,
                        plan_hash=plan.plan_hash,
                    )
                    if case_decision is None:
                        case_attempt_id = self.next_id(
                            "ATTEMPT_RECONCILIATION_CASE"
                        )
                        self._set_active_attempt(case_attempt_id)
                        try:
                            case_decision = await ReconciliationCaseFlow(
                                policy=self.policy,
                                runtime=self.runtime,
                                service=self.reconciliation_service,
                            ).run(
                                parent=state,
                                plan_hash=plan.plan_hash,
                                group=group,
                                case=case,
                                attempt_id=case_attempt_id,
                                session_id=session_id,
                            )
                        finally:
                            self._set_active_attempt(None)
                    case_decisions.append(case_decision)
                self._raise_if_cancelled(state.parent_attempt_id)
                existing = self.reconciliation_service.assemble(
                    parent=state,
                    plan_hash=plan.plan_hash,
                    group=group,
                    attempt_id=self.next_id(
                        "ATTEMPT_RECONCILIATION_COORDINATOR"
                    ),
                    cases=cases,
                    case_decisions=tuple(case_decisions),
                )
            decisions.append(existing)
            self._set_progress(completed_conflicts=len(decisions))
        return tuple(decisions)

    def _validate_recovery(
        self,
        state: WindowedAttemptState,
        *,
        expected_workflow_revision: str,
    ) -> None:
        if (
            state.expected_workflow_revision
            != expected_workflow_revision
            or state.policy_version != self.windowing_policy.fingerprint
            or state.prompt_version != self.windowing_policy.prompt_version
            or state.schema_version
            != self.windowing_policy.candidate_schema_version
        ):
            raise ValueError("Windowed recovery имеет stale version binding")
        if state.status not in {
            WindowedAttemptStatus.RUNNING,
            WindowedAttemptStatus.RECONCILING,
        }:
            raise AttemptDiscardedError(
                f"Parent attempt имеет терминальный статус {state.status.value}"
            )

    def _fail_parent(
        self,
        parent_attempt_id: str,
        selection: SavedSelection,
        reason: str,
    ) -> None:
        restored = self.store.load_optional(parent_attempt_id, selection)
        if restored is None:
            return
        state, plan = restored
        if state.status in {
            WindowedAttemptStatus.COMPLETED,
            WindowedAttemptStatus.FAILED,
            WindowedAttemptStatus.CANCELLED,
        }:
            return
        running_child = next(
            (
                child
                for child in state.children
                if child.status is WindowChildStatus.RUNNING
                and child.attempt_id is not None
            ),
            None,
        )
        failed = (
            state.fail_child(
                running_child.window_id,
                running_child.attempt_id,
                reason,
            )
            if (
                state.status is WindowedAttemptStatus.RUNNING
                and running_child is not None
            )
            else state.fail(reason)
        )
        self.store.save(
            failed,
            plan,
            expected_state=state,
        )

    def _raise_if_cancelled(self, parent_attempt_id: str) -> None:
        if self._explicit_cancel.is_set():
            raise AttemptDiscardedError(
                f"Parent attempt {parent_attempt_id} отменён"
            )

    def _required_selection(self) -> SavedSelection:
        if self._selection is None:
            raise RuntimeError("Windowed flow не получил selection")
        return self._selection

    def _set_active_attempt(self, attempt_id: str | None) -> None:
        with self._lock:
            self._active_attempt_id = attempt_id

    def _set_progress(self, **changes: object) -> None:
        with self._lock:
            self._progress = replace(self._progress, **changes)
