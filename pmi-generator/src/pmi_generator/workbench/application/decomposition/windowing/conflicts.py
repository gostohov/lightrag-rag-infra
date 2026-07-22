from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from itertools import combinations

from ....domain.source import SourceDocument, SourcePosition
from .candidates import (
    ValidatedBoundaryDependency,
    ValidatedWindowCandidate,
    WindowCandidateResult,
)
from .plan import WindowPlan
from .policy import WindowingPolicy


class ConflictPlanError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ConflictSourceLine:
    position: SourcePosition
    text: str

    def to_dict(self) -> dict[str, object]:
        return {
            "page": self.position.page_index,
            "line": self.position.line_number,
            "text": self.text,
        }


@dataclass(frozen=True, slots=True)
class ConflictGroup:
    group_id: str
    parent_attempt_id: str
    plan_hash: str
    candidate_ids: tuple[str, ...]
    dependency_ids: tuple[str, ...]
    reasons: tuple[str, ...]
    candidates: tuple[ValidatedWindowCandidate, ...]
    dependencies: tuple[ValidatedBoundaryDependency, ...]
    source_lines: tuple[ConflictSourceLine, ...]
    estimated_tokens: int

    def to_context(self) -> dict[str, object]:
        return {
            "group_id": self.group_id,
            "plan_hash": self.plan_hash,
            "candidate_ids": list(self.candidate_ids),
            "dependency_ids": list(self.dependency_ids),
            "reasons": list(self.reasons),
            "candidates": [
                {
                    "candidate_id": candidate.candidate_id,
                    "window_id": candidate.window_id,
                    "payload": _reconciliation_candidate_payload(
                        candidate
                    ),
                    "evidence_positions": [
                        {
                            "page": position.page_index,
                            "line": position.line_number,
                        }
                        for position in candidate.evidence_positions
                    ],
                }
                for candidate in self.candidates
            ],
            "dependencies": [
                dependency.to_dict()
                for dependency in self.dependencies
            ],
            "source_lines": [
                line.to_dict() for line in self.source_lines
            ],
        }


def _reconciliation_candidate_payload(
    candidate: ValidatedWindowCandidate,
) -> dict[str, object]:
    payload = candidate.payload
    return {
        "title": payload["title"],
        "condition": payload["condition"],
        "changed_factor": payload["changed_factor"],
        "input_value": payload["input_value"],
        "action": payload["action"],
        "consequences": [
            item["text"]
            for item in payload["consequences"]  # type: ignore[union-attr]
        ],
    }


@dataclass(frozen=True, slots=True)
class ConflictPlan:
    plan_hash: str
    parent_attempt_id: str
    groups: tuple[ConflictGroup, ...]
    unconflicted_candidate_ids: tuple[str, ...]
    all_candidate_ids: tuple[str, ...]


class ConflictPlanner:
    def __init__(
        self,
        document: SourceDocument,
        policy: WindowingPolicy,
    ) -> None:
        self.document = document
        self.policy = policy

    def build(
        self,
        plan: WindowPlan,
        results: tuple[WindowCandidateResult, ...],
    ) -> ConflictPlan:
        ordered_results = self._validate_results(plan, results)
        parent_attempt_id = ordered_results[0].parent_attempt_id
        candidates = {
            candidate.candidate_id: candidate
            for result in ordered_results
            for candidate in result.candidates
        }
        dependencies = {
            dependency.dependency_id: dependency
            for result in ordered_results
            for dependency in result.boundary_dependencies
        }
        if len(candidates) != sum(
            len(result.candidates) for result in ordered_results
        ):
            raise ConflictPlanError("Candidate IDs должны быть уникальными")
        if len(dependencies) != sum(
            len(result.boundary_dependencies)
            for result in ordered_results
        ):
            raise ConflictPlanError("Dependency IDs должны быть уникальными")
        if set(candidates) & set(dependencies):
            raise ConflictPlanError(
                "Candidate и dependency IDs не должны пересекаться"
            )

        nodes = {
            **{candidate_id: candidate_id for candidate_id in candidates},
            **{dependency_id: dependency_id for dependency_id in dependencies},
        }
        reasons: dict[str, set[str]] = {node: set() for node in nodes}

        def find(node: str) -> str:
            root = node
            while nodes[root] != root:
                root = nodes[root]
            while nodes[node] != node:
                parent = nodes[node]
                nodes[node] = root
                node = parent
            return root

        def connect(left: str, right: str, reason: str) -> None:
            left_root = find(left)
            right_root = find(right)
            if left_root != right_root:
                first, second = sorted((left_root, right_root))
                nodes[second] = first
            reasons[left].add(reason)
            reasons[right].add(reason)

        def mark(node: str, reason: str) -> None:
            reasons[node].add(reason)

        by_position: dict[SourcePosition, list[str]] = {}
        for candidate_id, item in candidates.items():
            for position in item.evidence_positions:
                by_position.setdefault(position, []).append(candidate_id)
        for candidate_ids in by_position.values():
            for left, right in combinations(sorted(set(candidate_ids)), 2):
                connect(left, right, "overlapping_evidence")

        owner_assessments = {
            assessment.position: assessment.role
            for result in ordered_results
            for assessment in result.primary_line_assessments
        }
        for candidate_id, item in candidates.items():
            if any(
                owner_assessments.get(position) == "context"
                for position in item.evidence_positions
            ):
                mark(candidate_id, "primary_owner_context")

        window_indexes = {
            window.window_id: window.index for window in plan.windows
        }
        candidates_by_window: dict[str, tuple[ValidatedWindowCandidate, ...]] = {
            window.window_id: tuple(
                sorted(
                    (
                        candidate
                        for candidate in candidates.values()
                        if candidate.window_id == window.window_id
                    ),
                    key=lambda item: item.candidate_id,
                )
            )
            for window in plan.windows
        }
        result_by_window = {
            result.window_id: result for result in ordered_results
        }
        dependency_windows = {
            dependency.dependency_id: result.window_id
            for result in ordered_results
            for dependency in result.boundary_dependencies
        }
        dependency_positions = {
            dependency_id: self._dependency_positions(dependency)
            for dependency_id, dependency in dependencies.items()
        }
        for left_id, right_id in combinations(sorted(dependencies), 2):
            left = dependencies[left_id]
            right = dependencies[right_id]
            left_window = window_indexes[dependency_windows[left_id]]
            right_window = window_indexes[dependency_windows[right_id]]
            opposing_adjacent = (
                abs(left_window - right_window) == 1
                and left.missing_field == right.missing_field
                and {
                    left.direction,
                    right.direction,
                }
                == {"before", "after"}
            )
            if (
                dependency_positions[left_id]
                & dependency_positions[right_id]
                or opposing_adjacent
            ):
                connect(left_id, right_id, "boundary_dependency")
        for window_id, result in result_by_window.items():
            for dependency in result.boundary_dependencies:
                mark(dependency.dependency_id, "boundary_dependency")
                if dependency.candidate_id is not None:
                    if dependency.candidate_id not in candidates:
                        raise ConflictPlanError(
                            "Dependency ссылается на неизвестный candidate"
                        )
                    connect(
                        dependency.dependency_id,
                        dependency.candidate_id,
                        "boundary_dependency",
                    )
                target_index = window_indexes[window_id] + (
                    -1 if dependency.direction == "before" else 1
                )
                if not 0 <= target_index < len(plan.windows):
                    continue
                target_window = plan.windows[target_index]
                positions = dependency_positions[dependency.dependency_id]
                for candidate in candidates_by_window[target_window.window_id]:
                    if positions & set(
                        candidate.evidence_positions
                    ):
                        connect(
                            dependency.dependency_id,
                            candidate.candidate_id,
                            "boundary_dependency",
                        )

        conflict_nodes = {
            node for node, node_reasons in reasons.items() if node_reasons
        }
        components: dict[str, set[str]] = {}
        for node in conflict_nodes:
            components.setdefault(find(node), set()).add(node)

        groups = tuple(
            sorted(
                (
                    self._group(
                        plan,
                        parent_attempt_id,
                        component,
                        reasons,
                        candidates,
                        dependencies,
                    )
                    for component in components.values()
                ),
                key=lambda group: group.group_id,
            )
        )
        if len(groups) > self.policy.reconciliation_max_groups:
            raise ConflictPlanError(
                "Conflict plan превышает reconciliation group budget"
            )
        conflicted_candidates = {
            candidate_id
            for group in groups
            for candidate_id in group.candidate_ids
        }
        all_candidate_ids = tuple(sorted(candidates))
        return ConflictPlan(
            plan_hash=plan.plan_hash,
            parent_attempt_id=parent_attempt_id,
            groups=groups,
            unconflicted_candidate_ids=tuple(
                candidate_id
                for candidate_id in all_candidate_ids
                if candidate_id not in conflicted_candidates
            ),
            all_candidate_ids=all_candidate_ids,
        )

    def _group(
        self,
        plan: WindowPlan,
        parent_attempt_id: str,
        nodes: set[str],
        reasons_by_node: dict[str, set[str]],
        candidates: dict[str, ValidatedWindowCandidate],
        dependencies: dict[str, ValidatedBoundaryDependency],
    ) -> ConflictGroup:
        candidate_ids = tuple(sorted(nodes & set(candidates)))
        dependency_ids = tuple(sorted(nodes & set(dependencies)))
        if len(candidate_ids) > self.policy.reconciliation_max_candidates:
            raise ConflictPlanError(
                "Conflict group превышает reconciliation candidate budget"
            )
        candidate_items = tuple(candidates[item] for item in candidate_ids)
        dependency_items = tuple(
            dependencies[item] for item in dependency_ids
        )
        positions = {
            position
            for item in candidate_items
            for position in item.evidence_positions
        }
        for dependency in dependency_items:
            positions.update(self._dependency_positions(dependency))
        if not positions:
            raise ConflictPlanError(
                "Conflict group не содержит source coordinates"
            )
        ordered_positions = tuple(
            sorted(positions, key=self.document.position_index)
        )
        if len(ordered_positions) > self.policy.reconciliation_max_source_lines:
            raise ConflictPlanError(
                "Conflict group превышает reconciliation source budget"
            )
        group_reasons = tuple(
            sorted(
                {
                    reason
                    for node in nodes
                    for reason in reasons_by_node[node]
                }
            )
        )
        identity = {
            "plan_hash": plan.plan_hash,
            "candidate_ids": candidate_ids,
            "dependency_ids": dependency_ids,
            "reasons": group_reasons,
            "positions": [
                [position.page_index, position.line_number]
                for position in ordered_positions
            ],
        }
        digest = hashlib.sha256(
            json.dumps(identity, sort_keys=True).encode("utf-8")
        ).hexdigest()[:16].upper()
        group = ConflictGroup(
            group_id=f"CONFLICT_{digest}",
            parent_attempt_id=parent_attempt_id,
            plan_hash=plan.plan_hash,
            candidate_ids=candidate_ids,
            dependency_ids=dependency_ids,
            reasons=group_reasons,
            candidates=candidate_items,
            dependencies=dependency_items,
            source_lines=tuple(
                ConflictSourceLine(position, self.document.line(position))
                for position in ordered_positions
            ),
            estimated_tokens=0,
        )
        serialized = json.dumps(
            group.to_context(),
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        estimated_tokens = max(1, (len(serialized) + 3) // 4)
        if estimated_tokens > self.policy.reconciliation_max_estimated_tokens:
            raise ConflictPlanError(
                "Conflict group превышает reconciliation token budget"
            )
        return replace(group, estimated_tokens=estimated_tokens)

    def _validate_results(
        self,
        plan: WindowPlan,
        results: tuple[WindowCandidateResult, ...],
    ) -> tuple[WindowCandidateResult, ...]:
        by_window = {result.window_id: result for result in results}
        if len(by_window) != len(results):
            raise ConflictPlanError("Window results содержат duplicate window")
        expected = tuple(window.window_id for window in plan.windows)
        if set(by_window) != set(expected):
            raise ConflictPlanError(
                "Window results не покрывают immutable plan"
            )
        ordered = tuple(by_window[window_id] for window_id in expected)
        parent_ids = {result.parent_attempt_id for result in ordered}
        if len(parent_ids) != 1:
            raise ConflictPlanError(
                "Window results относятся к разным parent attempts"
            )
        for window, result in zip(plan.windows, ordered, strict=True):
            if result.plan_hash != plan.plan_hash:
                raise ConflictPlanError(
                    "Window result относится к другому plan hash"
                )
            if tuple(
                item.position for item in result.primary_line_assessments
            ) != window.primary_positions:
                raise ConflictPlanError(
                    "Window result нарушает primary ownership"
                )
            if any(
                candidate.window_id != window.window_id
                for candidate in result.candidates
            ):
                raise ConflictPlanError(
                    "Candidate относится к другому окну"
                )
            allowed_window_positions = {
                line.position for line in window.lines
            }
            if any(
                not set(candidate.evidence_positions)
                <= allowed_window_positions
                for candidate in result.candidates
            ):
                raise ConflictPlanError(
                    "Candidate evidence выходит за immutable window"
                )
        return ordered

    def _dependency_positions(
        self,
        dependency: ValidatedBoundaryDependency,
    ) -> set[SourcePosition]:
        result: set[SourcePosition] = set()
        for item in dependency.source:
            excerpt = self.document.select(
                SourcePosition(
                    int(item["page"]),
                    int(item["line_start"]),
                ),
                SourcePosition(
                    int(item["page"]),
                    int(item["line_end"]),
                ),
            )
            result.update(excerpt.positions)
        return result
