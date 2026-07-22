from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass

from ....domain.source import SourceDocument, SourcePosition
from ...repositories import UnitOfWork
from ...state import StoredRecord
from .candidates import (
    ValidatedBoundaryDependency,
    ValidatedWindowCandidate,
)
from .conflicts import ConflictGroup, ConflictSourceLine
from .models import WindowedAttemptState, WindowedAttemptStatus
from .reconciliation import (
    ReconciliationDecision,
    ReconciliationError,
)


PAIR_DECISIONS = {
    "duplicate_keep_a",
    "duplicate_keep_b",
    "keep_separate",
    "unresolved",
}
REVIEW_DECISIONS = {"keep", "reject", "unresolved"}
DEPENDENCY_DECISIONS = {"resolved", "unresolved"}


@dataclass(frozen=True, slots=True)
class ReconciliationCaseArguments:
    decision: str
    reason: str


@dataclass(frozen=True, slots=True)
class ReconciliationCase:
    case_id: str
    group_id: str
    parent_attempt_id: str
    plan_hash: str
    kind: str
    candidate_ids: tuple[str, ...]
    dependency_ids: tuple[str, ...]
    reasons: tuple[str, ...]
    candidates: tuple[ValidatedWindowCandidate, ...]
    dependencies: tuple[ValidatedBoundaryDependency, ...]
    source_lines: tuple[ConflictSourceLine, ...]

    def to_context(self) -> dict[str, object]:
        context: dict[str, object] = {
            "case_kind": self.kind,
            "conflict_reasons": list(self.reasons),
            "source_text": [line.text for line in self.source_lines],
        }
        if self.kind == "candidate_pair":
            context["candidate_a"] = _candidate_context(self.candidates[0])
            context["candidate_b"] = _candidate_context(self.candidates[1])
        elif self.kind == "candidate_review":
            context["candidate"] = _candidate_context(self.candidates[0])
        elif self.kind == "dependency_review":
            dependency = self.dependencies[0]
            context["dependency"] = {
                "direction": dependency.direction,
                "missing_field": dependency.missing_field,
                "reason": dependency.reason,
            }
            context["related_candidates"] = [
                _candidate_context(candidate)
                for candidate in self.candidates
            ]
        else:
            raise ReconciliationError(
                f"Неизвестный reconciliation case kind {self.kind}"
            )
        return context

    def fingerprint(self) -> str:
        return _fingerprint(
            {
                "case_id": self.case_id,
                "group_id": self.group_id,
                "parent_attempt_id": self.parent_attempt_id,
                "plan_hash": self.plan_hash,
                "kind": self.kind,
                "candidate_ids": list(self.candidate_ids),
                "dependency_ids": list(self.dependency_ids),
                "reasons": list(self.reasons),
                "context": self.to_context(),
            }
        )


@dataclass(frozen=True, slots=True)
class ReconciliationCaseDecision:
    parent_attempt_id: str
    attempt_id: str
    group_id: str
    case_id: str
    plan_hash: str
    decision: str
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "parent_attempt_id": self.parent_attempt_id,
            "attempt_id": self.attempt_id,
            "group_id": self.group_id,
            "case_id": self.case_id,
            "plan_hash": self.plan_hash,
            "decision": self.decision,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(
        cls,
        value: dict[str, object],
    ) -> ReconciliationCaseDecision:
        return cls(
            parent_attempt_id=str(value["parent_attempt_id"]),
            attempt_id=str(value["attempt_id"]),
            group_id=str(value["group_id"]),
            case_id=str(value["case_id"]),
            plan_hash=str(value["plan_hash"]),
            decision=str(value["decision"]),
            reason=str(value["reason"]),
        )


class ReconciliationCasePlanner:
    def __init__(self, *, max_cases: int) -> None:
        if max_cases < 1:
            raise ValueError("Reconciliation case budget должен быть положительным")
        self.max_cases = max_cases

    def build(
        self,
        group: ConflictGroup,
    ) -> tuple[ReconciliationCase, ...]:
        candidates = {
            candidate.candidate_id: candidate
            for candidate in group.candidates
        }
        dependencies = {
            dependency.dependency_id: dependency
            for dependency in group.dependencies
        }
        if set(candidates) != set(group.candidate_ids):
            raise ReconciliationError(
                "Conflict group не содержит payload всех candidates"
            )
        if set(dependencies) != set(group.dependency_ids):
            raise ReconciliationError(
                "Conflict group не содержит payload всех dependencies"
            )

        pairs = self._overlap_pairs(candidates)
        covered = {
            candidate_id
            for pair in pairs
            for candidate_id in pair
        }
        cases = [
            self._case(
                group,
                kind="candidate_pair",
                candidate_ids=pair,
                dependency_ids=(),
            )
            for pair in pairs
        ]
        cases.extend(
            self._case(
                group,
                kind="candidate_review",
                candidate_ids=(candidate_id,),
                dependency_ids=(),
            )
            for candidate_id in sorted(set(candidates) - covered)
        )
        cases.extend(
            self._case(
                group,
                kind="dependency_review",
                candidate_ids=tuple(
                    sorted(
                        candidate_id
                        for candidate_id in (
                            dependencies[dependency_id].candidate_id,
                        )
                        if candidate_id in candidates
                    )
                ),
                dependency_ids=(dependency_id,),
            )
            for dependency_id in sorted(dependencies)
        )
        ordered = tuple(
            sorted(
                cases,
                key=lambda case: (
                    case.kind,
                    case.candidate_ids,
                    case.dependency_ids,
                ),
            )
        )
        if len(ordered) > self.max_cases:
            raise ReconciliationError(
                "Reconciliation comparison plan превышает case budget"
            )
        if group.candidate_ids or group.dependency_ids:
            if not ordered:
                raise ReconciliationError(
                    "Reconciliation comparison plan оказался пустым"
                )
        return ordered

    @staticmethod
    def _overlap_pairs(
        candidates: dict[str, ValidatedWindowCandidate],
    ) -> tuple[tuple[str, str], ...]:
        pairs: list[tuple[str, str]] = []
        ordered_ids = sorted(candidates)
        for index, left_id in enumerate(ordered_ids):
            left_positions = set(candidates[left_id].evidence_positions)
            for right_id in ordered_ids[index + 1 :]:
                if left_positions.intersection(
                    candidates[right_id].evidence_positions
                ):
                    pairs.append((left_id, right_id))
        return tuple(pairs)

    @staticmethod
    def _case(
        group: ConflictGroup,
        *,
        kind: str,
        candidate_ids: tuple[str, ...],
        dependency_ids: tuple[str, ...],
    ) -> ReconciliationCase:
        identity = {
            "group_id": group.group_id,
            "plan_hash": group.plan_hash,
            "kind": kind,
            "candidate_ids": candidate_ids,
            "dependency_ids": dependency_ids,
        }
        digest = hashlib.sha256(
            json.dumps(identity, sort_keys=True).encode("utf-8")
        ).hexdigest()[:16].upper()
        candidate_map = {
            candidate.candidate_id: candidate
            for candidate in group.candidates
        }
        dependency_map = {
            dependency.dependency_id: dependency
            for dependency in group.dependencies
        }
        selected_candidates = tuple(
            candidate_map[item] for item in candidate_ids
        )
        selected_dependencies = tuple(
            dependency_map[item] for item in dependency_ids
        )
        return ReconciliationCase(
            case_id=f"RECONCILIATION_CASE_{digest}",
            group_id=group.group_id,
            parent_attempt_id=group.parent_attempt_id,
            plan_hash=group.plan_hash,
            kind=kind,
            candidate_ids=candidate_ids,
            dependency_ids=dependency_ids,
            reasons=group.reasons,
            candidates=selected_candidates,
            dependencies=selected_dependencies,
            source_lines=_case_source_lines(
                group,
                selected_candidates,
                selected_dependencies,
            ),
        )


class ReconciliationCaseService:
    CASE_RECORD_KIND = "decomposition_reconciliation_case"
    GROUP_RECORD_KIND = "decomposition_reconciliation"
    CASE_SCHEMA_VERSION = 1
    GROUP_SCHEMA_VERSION = 3

    def __init__(
        self,
        *,
        document: SourceDocument,
        uow_factory: Callable[[], UnitOfWork],
    ) -> None:
        self.document = document
        self.uow_factory = uow_factory

    def validate_case(
        self,
        *,
        parent: WindowedAttemptState,
        plan_hash: str,
        group: ConflictGroup,
        case: ReconciliationCase,
        attempt_id: str,
        arguments: ReconciliationCaseArguments,
    ) -> ReconciliationCaseDecision:
        self._validate_binding(parent, plan_hash, group)
        if not attempt_id.strip():
            raise ReconciliationError(
                "Reconciliation case требует attempt ID"
            )
        if (
            case.parent_attempt_id != parent.parent_attempt_id
            or case.group_id != group.group_id
            or case.plan_hash != plan_hash
        ):
            raise ReconciliationError(
                "Reconciliation case имеет stale binding"
            )
        allowed = {
            "candidate_pair": PAIR_DECISIONS,
            "candidate_review": REVIEW_DECISIONS,
            "dependency_review": DEPENDENCY_DECISIONS,
        }.get(case.kind)
        if allowed is None or arguments.decision not in allowed:
            raise ReconciliationError(
                f"Недопустимое решение {arguments.decision} "
                f"для case {case.kind}"
            )
        if not arguments.reason.strip():
            raise ReconciliationError(
                "Reconciliation case требует объяснение"
            )
        return ReconciliationCaseDecision(
            parent_attempt_id=parent.parent_attempt_id,
            attempt_id=attempt_id,
            group_id=group.group_id,
            case_id=case.case_id,
            plan_hash=plan_hash,
            decision=arguments.decision,
            reason=arguments.reason,
        )

    def accept_case(
        self,
        *,
        parent: WindowedAttemptState,
        plan_hash: str,
        group: ConflictGroup,
        case: ReconciliationCase,
        attempt_id: str,
        arguments: ReconciliationCaseArguments,
        raw_arguments: dict[str, object],
        uow: UnitOfWork | None = None,
    ) -> ReconciliationCaseDecision:
        if raw_arguments != asdict(arguments):
            raise ReconciliationError(
                "Raw reconciliation case не соответствует typed arguments"
            )
        decision = self.validate_case(
            parent=parent,
            plan_hash=plan_hash,
            group=group,
            case=case,
            attempt_id=attempt_id,
            arguments=arguments,
        )
        if uow is None:
            with self.uow_factory() as owned_uow:
                self._save_case(case, decision, raw_arguments, owned_uow)
        else:
            self._save_case(case, decision, raw_arguments, uow)
        return decision

    def load_case(
        self,
        *,
        parent_attempt_id: str,
        group: ConflictGroup,
        case: ReconciliationCase,
        plan_hash: str,
    ) -> ReconciliationCaseDecision | None:
        record_id = self._case_record_id(
            parent_attempt_id,
            group.group_id,
            case.case_id,
        )
        with self.uow_factory() as uow:
            record = uow.records.get(self.CASE_RECORD_KIND, record_id)
        if record is None:
            return None
        payload = record.payload
        if int(payload.get("schema_version", -1)) != self.CASE_SCHEMA_VERSION:
            raise ReconciliationError(
                "Неизвестная reconciliation case storage schema"
            )
        self._verify_fingerprint(payload, "Reconciliation case")
        if payload.get("case_fingerprint") != case.fingerprint():
            raise ReconciliationError(
                "Reconciliation case относится к другому comparison plan"
            )
        decision = ReconciliationCaseDecision.from_dict(
            dict(payload["decision"])
        )
        if (
            decision.parent_attempt_id != parent_attempt_id
            or decision.group_id != group.group_id
            or decision.case_id != case.case_id
            or decision.plan_hash != plan_hash
        ):
            raise ReconciliationError(
                "Reconciliation case decision имеет stale binding"
            )
        return decision

    def assemble(
        self,
        *,
        parent: WindowedAttemptState,
        plan_hash: str,
        group: ConflictGroup,
        attempt_id: str,
        cases: tuple[ReconciliationCase, ...],
        case_decisions: tuple[ReconciliationCaseDecision, ...],
    ) -> ReconciliationDecision:
        self._validate_binding(parent, plan_hash, group)
        if not attempt_id.strip():
            raise ReconciliationError(
                "Reconciliation coordinator требует attempt ID"
            )
        case_map = {case.case_id: case for case in cases}
        decision_map = {
            decision.case_id: decision
            for decision in case_decisions
        }
        if (
            len(case_map) != len(cases)
            or len(decision_map) != len(case_decisions)
            or set(case_map) != set(decision_map)
        ):
            raise ReconciliationError(
                "Case decisions не покрывают comparison plan"
            )
        for decision in case_decisions:
            if (
                decision.parent_attempt_id != parent.parent_attempt_id
                or decision.group_id != group.group_id
                or decision.plan_hash != plan_hash
                or not decision.attempt_id.strip()
            ):
                raise ReconciliationError(
                    "Case decision имеет stale binding"
                )

        if any(
            decision.decision == "unresolved"
            for decision in case_decisions
        ):
            result = ReconciliationDecision(
                parent_attempt_id=parent.parent_attempt_id,
                attempt_id=attempt_id,
                group_id=group.group_id,
                plan_hash=plan_hash,
                outcome="unresolved",
                accepted_candidate_ids=(),
                rejected_candidate_ids=(),
                resolved_dependency_ids=(),
                relations=(),
                explanation="Один или несколько semantic cases не разрешены",
            )
            self._save_group(cases, result)
            return result

        covered_candidates: set[str] = set()
        covered_dependencies: set[str] = set()
        rejected: set[str] = set()
        duplicate_choices: dict[str, set[str]] = {}
        relations: list[dict[str, object]] = []
        explanations: list[str] = []
        for case in cases:
            case_decision = decision_map[case.case_id]
            covered_candidates.update(case.candidate_ids)
            covered_dependencies.update(case.dependency_ids)
            explanations.append(case_decision.reason)
            if case.kind == "candidate_pair":
                left_id, right_id = case.candidate_ids
                if case_decision.decision == "duplicate_keep_a":
                    rejected.add(right_id)
                    self._add_duplicate_choice(
                        duplicate_choices,
                        loser=right_id,
                        winner=left_id,
                    )
                    relation_kind = "merge"
                elif case_decision.decision == "duplicate_keep_b":
                    rejected.add(left_id)
                    self._add_duplicate_choice(
                        duplicate_choices,
                        loser=left_id,
                        winner=right_id,
                    )
                    relation_kind = "merge"
                elif case_decision.decision == "keep_separate":
                    relation_kind = "split"
                else:
                    raise ReconciliationError(
                        "Pair case содержит недопустимое resolved решение"
                    )
                relations.append(
                    self._relation(
                        group,
                        case.candidate_ids,
                        relation_kind,
                        case_decision.reason,
                    )
                )
            elif case.kind == "candidate_review":
                if case_decision.decision == "reject":
                    rejected.add(case.candidate_ids[0])
                elif case_decision.decision != "keep":
                    raise ReconciliationError(
                        "Review case содержит недопустимое resolved решение"
                    )
            elif (
                case.kind == "dependency_review"
                and case_decision.decision != "resolved"
            ):
                raise ReconciliationError(
                    "Dependency case содержит недопустимое resolved решение"
                )

        self._validate_duplicate_choices(duplicate_choices)
        if covered_candidates != set(group.candidate_ids):
            raise ReconciliationError(
                "Comparison plan не покрывает candidates"
            )
        if covered_dependencies != set(group.dependency_ids):
            raise ReconciliationError(
                "Comparison plan не покрывает dependencies"
            )
        accepted = tuple(sorted(set(group.candidate_ids) - rejected))
        result = ReconciliationDecision(
            parent_attempt_id=parent.parent_attempt_id,
            attempt_id=attempt_id,
            group_id=group.group_id,
            plan_hash=plan_hash,
            outcome="resolved",
            accepted_candidate_ids=accepted,
            rejected_candidate_ids=tuple(sorted(rejected)),
            resolved_dependency_ids=tuple(sorted(group.dependency_ids)),
            relations=tuple(relations),
            explanation="; ".join(explanations),
        )
        self._save_group(cases, result)
        return result

    @staticmethod
    def _add_duplicate_choice(
        choices: dict[str, set[str]],
        *,
        loser: str,
        winner: str,
    ) -> None:
        choices.setdefault(loser, set()).add(winner)

    @staticmethod
    def _validate_duplicate_choices(
        choices: dict[str, set[str]],
    ) -> None:
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(candidate_id: str) -> None:
            if candidate_id in visiting:
                raise ReconciliationError(
                    "Semantic duplicate decisions образуют цикл"
                )
            if candidate_id in visited:
                return
            visiting.add(candidate_id)
            for winner in choices.get(candidate_id, set()):
                visit(winner)
            visiting.remove(candidate_id)
            visited.add(candidate_id)

        for candidate_id in choices:
            visit(candidate_id)

    def load(
        self,
        *,
        parent_attempt_id: str,
        group: ConflictGroup,
        cases: tuple[ReconciliationCase, ...],
        plan_hash: str,
    ) -> ReconciliationDecision | None:
        record_id = self._group_record_id(
            parent_attempt_id,
            group.group_id,
        )
        with self.uow_factory() as uow:
            record = uow.records.get(self.GROUP_RECORD_KIND, record_id)
        if record is None:
            return None
        payload = record.payload
        if int(payload.get("schema_version", -1)) != self.GROUP_SCHEMA_VERSION:
            raise ReconciliationError(
                "Неизвестная reconciliation group storage schema"
            )
        self._verify_fingerprint(payload, "Reconciliation group")
        expected_case_fingerprints = [
            case.fingerprint() for case in cases
        ]
        if payload.get("case_fingerprints") != expected_case_fingerprints:
            raise ReconciliationError(
                "Reconciliation group относится к другому comparison plan"
            )
        decision = ReconciliationDecision.from_dict(
            dict(payload["decision"])
        )
        if (
            decision.parent_attempt_id != parent_attempt_id
            or decision.group_id != group.group_id
            or decision.plan_hash != plan_hash
        ):
            raise ReconciliationError(
                "Reconciliation decision имеет stale binding"
            )
        return decision

    def _save_case(
        self,
        case: ReconciliationCase,
        decision: ReconciliationCaseDecision,
        raw_arguments: dict[str, object],
        uow: UnitOfWork,
    ) -> None:
        payload: dict[str, object] = {
            "schema_version": self.CASE_SCHEMA_VERSION,
            "case_fingerprint": case.fingerprint(),
            "raw_arguments": raw_arguments,
            "decision": decision.to_dict(),
        }
        payload = json.loads(json.dumps(payload, ensure_ascii=False))
        payload["fingerprint"] = _fingerprint(payload)
        record_id = self._case_record_id(
            decision.parent_attempt_id,
            decision.group_id,
            decision.case_id,
        )
        existing = uow.records.get(self.CASE_RECORD_KIND, record_id)
        if existing is not None:
            if existing.payload.get("fingerprint") != payload["fingerprint"]:
                raise ReconciliationError(
                    "Для reconciliation case уже сохранено другое решение"
                )
            return
        uow.records.save(
            StoredRecord(self.CASE_RECORD_KIND, record_id, payload)
        )
        uow.events.append(
            decision.parent_attempt_id,
            "reconciliation case сохранён",
            {
                "group_id": decision.group_id,
                "case_id": decision.case_id,
                "decision": decision.decision,
            },
        )

    def _save_group(
        self,
        cases: tuple[ReconciliationCase, ...],
        decision: ReconciliationDecision,
    ) -> None:
        payload: dict[str, object] = {
            "schema_version": self.GROUP_SCHEMA_VERSION,
            "case_fingerprints": [
                case.fingerprint() for case in cases
            ],
            "decision": decision.to_dict(),
        }
        payload = json.loads(json.dumps(payload, ensure_ascii=False))
        payload["fingerprint"] = _fingerprint(payload)
        record_id = self._group_record_id(
            decision.parent_attempt_id,
            decision.group_id,
        )
        with self.uow_factory() as uow:
            existing = uow.records.get(self.GROUP_RECORD_KIND, record_id)
            if existing is not None:
                if existing.payload.get("fingerprint") != payload["fingerprint"]:
                    raise ReconciliationError(
                        "Для conflict group уже сохранено другое решение"
                    )
                return
            uow.records.save(
                StoredRecord(self.GROUP_RECORD_KIND, record_id, payload)
            )
            uow.events.append(
                decision.parent_attempt_id,
                "reconciliation decision сохранено",
                {
                    "group_id": decision.group_id,
                    "outcome": decision.outcome,
                    "cases": len(cases),
                },
            )

    def _relation(
        self,
        group: ConflictGroup,
        candidate_ids: tuple[str, ...],
        kind: str,
        reason: str,
    ) -> dict[str, object]:
        candidates = {
            candidate.candidate_id: candidate
            for candidate in group.candidates
        }
        allowed_positions = {
            line.position for line in group.source_lines
        }
        positions = {
            position
            for candidate_id in candidate_ids
            for position in candidates[candidate_id].evidence_positions
        }
        if not positions or not positions <= allowed_positions:
            raise ReconciliationError(
                "Comparison relation выходит за bounded conflict context"
            )
        return {
            "kind": kind,
            "candidate_ids": list(candidate_ids),
            "source": self._source_ranges(positions),
            "reason": reason,
        }

    def _source_ranges(
        self,
        positions: set[SourcePosition],
    ) -> list[dict[str, object]]:
        ordered = sorted(
            positions,
            key=lambda item: (item.page_index, item.line_number),
        )
        ranges: list[tuple[SourcePosition, SourcePosition]] = []
        start = ordered[0]
        end = ordered[0]
        for position in ordered[1:]:
            if (
                position.page_index == end.page_index
                and position.line_number == end.line_number + 1
            ):
                end = position
                continue
            ranges.append((start, end))
            start = position
            end = position
        ranges.append((start, end))
        return [
            {
                "page": start.page_index,
                "line_start": start.line_number,
                "line_end": end.line_number,
                "quote": self.document.select(start, end).text,
            }
            for start, end in ranges
        ]

    @staticmethod
    def _validate_binding(
        parent: WindowedAttemptState,
        plan_hash: str,
        group: ConflictGroup,
    ) -> None:
        if parent.status is not WindowedAttemptStatus.RECONCILING:
            raise ReconciliationError(
                "Parent attempt не находится в стадии reconciliation"
            )
        if (
            parent.window_plan_hash != plan_hash
            or group.plan_hash != plan_hash
            or group.parent_attempt_id != parent.parent_attempt_id
        ):
            raise ReconciliationError(
                "Reconciliation имеет stale parent/group/plan binding"
            )

    @staticmethod
    def _verify_fingerprint(
        payload: dict[str, object],
        label: str,
    ) -> None:
        expected = payload.get("fingerprint")
        actual = _fingerprint(
            {
                key: value
                for key, value in payload.items()
                if key != "fingerprint"
            }
        )
        if expected != actual:
            raise ReconciliationError(
                f"{label} fingerprint не соответствует содержимому"
            )

    @staticmethod
    def _case_record_id(
        parent_attempt_id: str,
        group_id: str,
        case_id: str,
    ) -> str:
        return f"{parent_attempt_id}:{group_id}:{case_id}"

    @staticmethod
    def _group_record_id(
        parent_attempt_id: str,
        group_id: str,
    ) -> str:
        return f"{parent_attempt_id}:{group_id}"


def _candidate_context(
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


def _case_source_lines(
    group: ConflictGroup,
    candidates: tuple[ValidatedWindowCandidate, ...],
    dependencies: tuple[ValidatedBoundaryDependency, ...],
) -> tuple[ConflictSourceLine, ...]:
    positions = {
        position
        for candidate in candidates
        for position in candidate.evidence_positions
    }
    for dependency in dependencies:
        for source_range in dependency.source:
            page = int(source_range["page"])
            line_start = int(source_range["line_start"])
            line_end = int(source_range["line_end"])
            positions.update(
                SourcePosition(page, line)
                for line in range(line_start, line_end + 1)
            )
    selected = tuple(
        line for line in group.source_lines if line.position in positions
    )
    if not selected:
        raise ReconciliationError(
            "Reconciliation case не содержит bounded source context"
        )
    return selected


def _fingerprint(payload: dict[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
