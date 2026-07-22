from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass

from ....domain.source import SourceDocument, SourcePosition
from ...repositories import UnitOfWork
from ...source import SavedSelection
from ...state import StoredRecord
from ..models import DecompositionArguments, DecompositionResult
from ..service import DecompositionService
from ..validation import DecompositionValidator
from .candidates import (
    PrimaryLineAssessment,
    ValidatedWindowCandidate,
    WindowCandidateResult,
)
from .conflicts import ConflictPlan
from .models import (
    WindowChildStatus,
    WindowedAttemptState,
    WindowedAttemptStatus,
)
from .persistence import WindowedDecompositionStore
from .plan import WindowPlan
from .reconciliation import ReconciliationDecision


class WindowAssemblyError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class WindowAssemblyOutcome:
    decomposition: DecompositionResult
    parent: WindowedAttemptState
    arguments: DecompositionArguments


class WindowAssemblyService:
    RECORD_KIND = "decomposition_window_assembly"
    SCHEMA_VERSION = 1

    def __init__(
        self,
        *,
        document: SourceDocument,
        uow_factory: Callable[[], UnitOfWork],
        next_id: Callable[[str], str],
    ) -> None:
        self.document = document
        self.uow_factory = uow_factory
        self.validator = DecompositionValidator(document)
        self.decomposition_service = DecompositionService(
            document=document,
            uow_factory=uow_factory,
            next_id=next_id,
        )
        self.attempt_store = WindowedDecompositionStore(uow_factory)

    def build(
        self,
        *,
        selection: SavedSelection,
        parent: WindowedAttemptState,
        plan: WindowPlan,
        results: tuple[WindowCandidateResult, ...],
        conflict_plan: ConflictPlan,
        decisions: tuple[ReconciliationDecision, ...],
    ) -> DecompositionArguments:
        ordered_results = self._validate_binding(
            selection=selection,
            parent=parent,
            plan=plan,
            results=results,
            conflict_plan=conflict_plan,
        )
        candidates = self._candidate_map(ordered_results)
        accepted, rejected = self._resolve_candidates(
            parent=parent,
            plan=plan,
            conflict_plan=conflict_plan,
            decisions=decisions,
            candidates=candidates,
            results=ordered_results,
        )
        accepted_candidates = tuple(
            sorted(
                (candidates[candidate_id] for candidate_id in accepted),
                key=self._candidate_sort_key,
            )
        )
        skeletons = [
            self._validated_skeleton(selection, candidate)
            for candidate in accepted_candidates
        ]
        line_assessments = self._line_assessments(
            selection=selection,
            results=ordered_results,
            accepted_candidates=accepted_candidates,
            rejected_candidates=tuple(
                candidates[candidate_id] for candidate_id in rejected
            ),
        )
        dependencies = tuple(
            dependency
            for result in ordered_results
            for dependency in result.boundary_dependencies
        )
        if skeletons:
            outcome = "skeletons_created"
            explanation = (
                "Каркасы собраны из валидированных результатов всех окон"
            )
        elif dependencies:
            outcome = "insufficient_selection"
            explanation = (
                "После обработки всех окон остались только граничные "
                "зависимости без полного каркаса"
            )
        else:
            outcome = "no_testable_behavior"
            explanation = (
                "Во всех окнах выбранного диапазона тестируемое поведение "
                "не найдено"
            )
        arguments = DecompositionArguments(
            outcome=outcome,
            explanation=explanation,
            skeletons=skeletons,
            line_assessments=line_assessments,
        )
        try:
            self.validator.validate(selection.selection, arguments)
        except ValueError as error:
            raise WindowAssemblyError(
                f"Итоговая декомпозиция невалидна: {error}"
            ) from error
        return arguments

    def apply(
        self,
        *,
        selection: SavedSelection,
        parent: WindowedAttemptState,
        plan: WindowPlan,
        results: tuple[WindowCandidateResult, ...],
        conflict_plan: ConflictPlan,
        decisions: tuple[ReconciliationDecision, ...],
    ) -> WindowAssemblyOutcome:
        arguments = self.build(
            selection=selection,
            parent=parent,
            plan=plan,
            results=results,
            conflict_plan=conflict_plan,
            decisions=decisions,
        )
        completed_parent = parent.complete()
        try:
            with self.uow_factory() as uow:
                decomposition = self.decomposition_service.apply(
                    selection,
                    arguments,
                    uow=uow,
                )
                self.attempt_store.save(
                    completed_parent,
                    plan,
                    uow=uow,
                    expected_state=parent,
                )
                self._save_diagnostics(
                    uow=uow,
                    parent=completed_parent,
                    results=results,
                    conflict_plan=conflict_plan,
                    decisions=decisions,
                    arguments=arguments,
                )
        except WindowAssemblyError:
            raise
        except ValueError as error:
            raise WindowAssemblyError(
                f"Атомарное применение assembly отклонено: {error}"
            ) from error
        return WindowAssemblyOutcome(
            decomposition=decomposition,
            parent=completed_parent,
            arguments=arguments,
        )

    def _validate_binding(
        self,
        *,
        selection: SavedSelection,
        parent: WindowedAttemptState,
        plan: WindowPlan,
        results: tuple[WindowCandidateResult, ...],
        conflict_plan: ConflictPlan,
    ) -> tuple[WindowCandidateResult, ...]:
        try:
            plan.validate(selection)
        except ValueError as error:
            raise WindowAssemblyError(str(error)) from error
        if parent.status is not WindowedAttemptStatus.RECONCILING:
            raise WindowAssemblyError(
                "Parent attempt не находится в стадии reconciliation"
            )
        if (
            parent.parent_attempt_id != conflict_plan.parent_attempt_id
            or parent.selection_id != selection.selection_id
            or parent.document_version != selection.document_version
            or parent.window_plan_hash != plan.plan_hash
            or conflict_plan.plan_hash != plan.plan_hash
        ):
            raise WindowAssemblyError(
                "Assembly inputs имеют stale или несовместимую binding"
            )
        expected_windows = tuple(window.window_id for window in plan.windows)
        if tuple(child.window_id for child in parent.children) != expected_windows:
            raise WindowAssemblyError(
                "Parent children не совпадают с immutable window plan"
            )
        if any(
            child.status is not WindowChildStatus.COMPLETED
            for child in parent.children
        ):
            raise WindowAssemblyError(
                "Assembly требует завершения всех child attempts"
            )
        by_window = {result.window_id: result for result in results}
        if len(by_window) != len(results) or set(by_window) != set(expected_windows):
            raise WindowAssemblyError(
                "Window results не покрывают immutable plan ровно один раз"
            )
        ordered = tuple(by_window[window_id] for window_id in expected_windows)
        for child, result in zip(parent.children, ordered, strict=True):
            if (
                result.parent_attempt_id != parent.parent_attempt_id
                or result.child_attempt_id != child.attempt_id
                or result.plan_hash != plan.plan_hash
            ):
                raise WindowAssemblyError(
                    "Window result имеет stale parent, child или plan binding"
                )
        primary_positions = tuple(
            assessment.position
            for result in ordered
            for assessment in result.primary_line_assessments
        )
        if primary_positions != selection.selection.positions:
            raise WindowAssemblyError(
                "Primary assessments не покрывают selection ровно один раз"
            )
        return ordered

    @staticmethod
    def _candidate_map(
        results: tuple[WindowCandidateResult, ...],
    ) -> dict[str, ValidatedWindowCandidate]:
        candidates = {
            candidate.candidate_id: candidate
            for result in results
            for candidate in result.candidates
        }
        if len(candidates) != sum(len(result.candidates) for result in results):
            raise WindowAssemblyError("Candidate IDs должны быть уникальными")
        return candidates

    def _resolve_candidates(
        self,
        *,
        parent: WindowedAttemptState,
        plan: WindowPlan,
        conflict_plan: ConflictPlan,
        decisions: tuple[ReconciliationDecision, ...],
        candidates: dict[str, ValidatedWindowCandidate],
        results: tuple[WindowCandidateResult, ...],
    ) -> tuple[set[str], set[str]]:
        if set(conflict_plan.all_candidate_ids) != set(candidates):
            raise WindowAssemblyError(
                "Conflict plan не соответствует полному набору candidates"
            )
        if (
            len(conflict_plan.all_candidate_ids)
            != len(set(conflict_plan.all_candidate_ids))
            or len(conflict_plan.unconflicted_candidate_ids)
            != len(set(conflict_plan.unconflicted_candidate_ids))
        ):
            raise WindowAssemblyError(
                "Conflict plan содержит duplicate candidate IDs"
            )
        grouped_candidates: list[str] = []
        grouped_dependencies: list[str] = []
        result_dependencies = {
            dependency.dependency_id: dependency
            for result in results
            for dependency in result.boundary_dependencies
        }
        if len(result_dependencies) != sum(
            len(result.boundary_dependencies) for result in results
        ):
            raise WindowAssemblyError(
                "Boundary dependency IDs должны быть уникальными"
            )
        for group in conflict_plan.groups:
            if (
                group.parent_attempt_id != parent.parent_attempt_id
                or group.plan_hash != plan.plan_hash
            ):
                raise WindowAssemblyError(
                    "Conflict group имеет stale parent или plan binding"
                )
            if (
                not set(group.candidate_ids) <= set(candidates)
                or not set(group.dependency_ids) <= set(result_dependencies)
            ):
                raise WindowAssemblyError(
                    "Conflict group ссылается на неизвестные IDs"
                )
            grouped_candidates.extend(group.candidate_ids)
            grouped_dependencies.extend(group.dependency_ids)
            if tuple(group.candidates) != tuple(
                candidates[candidate_id]
                for candidate_id in group.candidate_ids
            ):
                raise WindowAssemblyError(
                    "Conflict group содержит изменённые candidates"
                )
            if tuple(group.dependencies) != tuple(
                result_dependencies[dependency_id]
                for dependency_id in group.dependency_ids
            ):
                raise WindowAssemblyError(
                    "Conflict group содержит изменённые dependencies"
                )
        if (
            len(grouped_candidates) != len(set(grouped_candidates))
            or set(grouped_candidates)
            | set(conflict_plan.unconflicted_candidate_ids)
            != set(candidates)
            or set(grouped_candidates)
            & set(conflict_plan.unconflicted_candidate_ids)
        ):
            raise WindowAssemblyError(
                "Conflict plan не образует точную candidate partition"
            )
        if (
            len(grouped_dependencies) != len(set(grouped_dependencies))
            or set(grouped_dependencies) != set(result_dependencies)
        ):
            raise WindowAssemblyError(
                "Conflict plan не покрывает boundary dependencies"
            )

        by_group = {decision.group_id: decision for decision in decisions}
        if len(by_group) != len(decisions) or set(by_group) != {
            group.group_id for group in conflict_plan.groups
        }:
            raise WindowAssemblyError(
                "Reconciliation decisions не покрывают conflict groups"
            )
        accepted = set(conflict_plan.unconflicted_candidate_ids)
        rejected: set[str] = set()
        for group in conflict_plan.groups:
            decision = by_group[group.group_id]
            if decision.outcome != "resolved":
                raise WindowAssemblyError(
                    f"Conflict group {group.group_id} остался unresolved"
                )
            if (
                decision.parent_attempt_id != parent.parent_attempt_id
                or decision.plan_hash != plan.plan_hash
                or not decision.attempt_id.strip()
            ):
                raise WindowAssemblyError(
                    "Reconciliation decision имеет stale binding"
                )
            decision_candidates = (
                set(decision.accepted_candidate_ids)
                | set(decision.rejected_candidate_ids)
            )
            if (
                set(decision.accepted_candidate_ids)
                & set(decision.rejected_candidate_ids)
                or decision_candidates != set(group.candidate_ids)
                or set(decision.resolved_dependency_ids)
                != set(group.dependency_ids)
            ):
                raise WindowAssemblyError(
                    "Reconciliation decision не закрывает conflict group"
                )
            if any(
                not set(relation.get("candidate_ids", []))
                <= set(group.candidate_ids)
                for relation in decision.relations
            ):
                raise WindowAssemblyError(
                    "Reconciliation relation содержит неизвестный candidate"
                )
            accepted.update(decision.accepted_candidate_ids)
            rejected.update(decision.rejected_candidate_ids)
        if accepted & rejected or accepted | rejected != set(candidates):
            raise WindowAssemblyError(
                "Assembly не получила точную candidate partition"
            )
        return accepted, rejected

    def _validated_skeleton(
        self,
        selection: SavedSelection,
        candidate: ValidatedWindowCandidate,
    ) -> dict[str, object]:
        raw = self._raw_skeleton(candidate.payload)
        try:
            normalized, positions = self.validator.validate_skeleton(
                selection.selection,
                raw,
            )
        except ValueError as error:
            raise WindowAssemblyError(
                f"Candidate {candidate.candidate_id} невалиден: {error}"
            ) from error
        if normalized != candidate.payload or positions != set(
            candidate.evidence_positions
        ):
            raise WindowAssemblyError(
                f"Candidate {candidate.candidate_id} изменён после validation"
            )
        return raw

    @staticmethod
    def _raw_skeleton(payload: dict[str, object]) -> dict[str, object]:
        def ranges(raw: object) -> list[dict[str, int]]:
            if not isinstance(raw, list) or any(
                not isinstance(item, dict)
                or not {"page", "line_start", "line_end"} <= set(item)
                for item in raw
            ):
                raise WindowAssemblyError(
                    "Validated evidence имеет неверную структуру"
                )
            return [
                {
                    "page": int(item["page"]),
                    "line_start": int(item["line_start"]),
                    "line_end": int(item["line_end"]),
                }
                for item in raw
            ]

        consequences = payload.get("consequences")
        if not isinstance(consequences, list) or any(
            not isinstance(item, dict)
            or not {"text", "evidence"} <= set(item)
            for item in consequences
        ):
            raise WindowAssemblyError(
                "Validated consequences имеют неверную структуру"
            )
        return {
            "title": payload.get("title"),
            "condition": payload.get("condition"),
            "changed_factor": payload.get("changed_factor"),
            "input_value": payload.get("input_value"),
            "action": payload.get("action"),
            "condition_ranges": ranges(payload.get("condition_evidence")),
            "changed_factor_ranges": ranges(
                payload.get("changed_factor_evidence")
            ),
            "input_value_ranges": ranges(
                payload.get("input_value_evidence")
            ),
            "action_ranges": ranges(payload.get("action_evidence")),
            "consequences": [
                {
                    "text": item["text"],
                    "evidence_ranges": ranges(item["evidence"]),
                }
                for item in consequences
            ],
            "gaps": payload.get("gaps"),
        }

    def _line_assessments(
        self,
        *,
        selection: SavedSelection,
        results: tuple[WindowCandidateResult, ...],
        accepted_candidates: tuple[ValidatedWindowCandidate, ...],
        rejected_candidates: tuple[ValidatedWindowCandidate, ...],
    ) -> list[dict[str, object]]:
        primary = {
            assessment.position: assessment
            for result in results
            for assessment in result.primary_line_assessments
        }
        accepted_positions = {
            position
            for candidate in accepted_candidates
            for position in candidate.evidence_positions
        }
        rejected_positions = {
            position
            for candidate in rejected_candidates
            for position in candidate.evidence_positions
        }
        result: list[dict[str, object]] = []
        for position in selection.selection.positions:
            if position in accepted_positions:
                role = "evidence"
                reason = "Строка используется принятым candidate"
            elif position in rejected_positions:
                role = "context"
                reason = "Строка использована только отклонённым candidate"
            else:
                assessment: PrimaryLineAssessment = primary[position]
                role = "context"
                reason = assessment.reason
            result.append(
                {
                    "page": position.page_index,
                    "line": position.line_number,
                    "role": role,
                    "reason": reason,
                }
            )
        return result

    def _candidate_sort_key(
        self,
        candidate: ValidatedWindowCandidate,
    ) -> tuple[int, str]:
        return (
            min(
                self.document.position_index(position)
                for position in candidate.evidence_positions
            ),
            candidate.candidate_id,
        )

    def _save_diagnostics(
        self,
        *,
        uow: UnitOfWork,
        parent: WindowedAttemptState,
        results: tuple[WindowCandidateResult, ...],
        conflict_plan: ConflictPlan,
        decisions: tuple[ReconciliationDecision, ...],
        arguments: DecompositionArguments,
    ) -> None:
        payload = json.loads(
            json.dumps(
                {
                    "schema_version": self.SCHEMA_VERSION,
                    "parent": {
                        "parent_attempt_id": parent.parent_attempt_id,
                        "status": parent.status.value,
                        "plan_hash": parent.window_plan_hash,
                    },
                    "window_result_ids": [
                        {
                            "window_id": result.window_id,
                            "child_attempt_id": result.child_attempt_id,
                            "candidate_ids": [
                                candidate.candidate_id
                                for candidate in result.candidates
                            ],
                        }
                        for result in sorted(
                            results,
                            key=lambda item: item.window_id,
                        )
                    ],
                    "conflict_groups": [
                        {
                            "group_id": group.group_id,
                            "candidate_ids": list(group.candidate_ids),
                            "dependency_ids": list(group.dependency_ids),
                        }
                        for group in conflict_plan.groups
                    ],
                    "decisions": [
                        decision.to_dict()
                        for decision in sorted(
                            decisions,
                            key=lambda item: item.group_id,
                        )
                    ],
                    "arguments": asdict(arguments),
                },
                ensure_ascii=False,
            )
        )
        fingerprint = hashlib.sha256(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        payload["fingerprint"] = fingerprint
        existing = uow.records.get(
            self.RECORD_KIND,
            parent.parent_attempt_id,
        )
        if existing is not None:
            if existing.payload.get("fingerprint") != fingerprint:
                raise WindowAssemblyError(
                    "Для parent attempt уже сохранена другая assembly"
                )
            return
        uow.records.save(
            StoredRecord(
                self.RECORD_KIND,
                parent.parent_attempt_id,
                payload,
            )
        )
        uow.events.append(
            parent.parent_attempt_id,
            "windowed decomposition собрана",
            {
                "outcome": arguments.outcome,
                "skeleton_count": len(arguments.skeletons),
            },
        )
