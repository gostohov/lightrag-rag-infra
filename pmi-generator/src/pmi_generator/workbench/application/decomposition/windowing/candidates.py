from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass

from ....domain.source import SourceDocument, SourcePosition, TextSelection
from ...repositories import UnitOfWork
from ...state import StoredRecord
from ..models import DecompositionError
from ..validation import DecompositionValidator
from .models import (
    WINDOW_CANDIDATE_SCHEMA_VERSION,
    WindowChildStatus,
    WindowedAttemptState,
    WindowedAttemptStatus,
)
from .plan import DecompositionWindow, WindowPlan


class WindowCandidateError(ValueError):
    pass


def _window_coordinate_help(window: DecompositionWindow) -> str:
    page_lines: dict[int, list[int]] = {}
    for line in window.lines:
        page_lines.setdefault(line.position.page_index, []).append(
            line.position.line_number
        )
    ranges: list[str] = []
    for page, numbers in sorted(page_lines.items()):
        ordered = sorted(set(numbers))
        start = previous = ordered[0]
        spans: list[str] = []
        for number in ordered[1:]:
            if number == previous + 1:
                previous = number
                continue
            spans.append(
                str(start) if start == previous else f"{start}-{previous}"
            )
            start = previous = number
        spans.append(str(start) if start == previous else f"{start}-{previous}")
        ranges.append(f"{page}:{','.join(spans)}")
    return (
        "Допустимые координаты текущего окна: "
        + "; ".join(ranges)
        + ". Нумерация строк начинается заново на каждой странице; "
        "продолжение на следующей странице задай отдельным source range"
    )


@dataclass(frozen=True, slots=True)
class WindowCandidateArguments:
    outcome: str
    explanation: str
    candidates: list[dict[str, object]]
    boundary_dependencies: list[dict[str, object]]


@dataclass(frozen=True, slots=True)
class ValidatedWindowCandidate:
    candidate_id: str
    local_candidate_id: str
    window_id: str
    payload: dict[str, object]
    evidence_positions: tuple[SourcePosition, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "local_candidate_id": self.local_candidate_id,
            "window_id": self.window_id,
            "payload": self.payload,
            "evidence_positions": [
                _position_dict(position)
                for position in self.evidence_positions
            ],
        }

    @classmethod
    def from_dict(
        cls,
        value: dict[str, object],
    ) -> ValidatedWindowCandidate:
        return cls(
            candidate_id=str(value["candidate_id"]),
            local_candidate_id=str(value["local_candidate_id"]),
            window_id=str(value["window_id"]),
            payload=dict(value["payload"]),
            evidence_positions=tuple(
                _position_from_dict(dict(item))
                for item in value["evidence_positions"]  # type: ignore[union-attr]
            ),
        )


@dataclass(frozen=True, slots=True)
class ValidatedBoundaryDependency:
    dependency_id: str
    local_dependency_id: str
    candidate_id: str | None
    direction: str
    missing_field: str
    source: tuple[dict[str, object], ...]
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "dependency_id": self.dependency_id,
            "local_dependency_id": self.local_dependency_id,
            "candidate_id": self.candidate_id,
            "direction": self.direction,
            "missing_field": self.missing_field,
            "source": list(self.source),
            "reason": self.reason,
        }

    @classmethod
    def from_dict(
        cls,
        value: dict[str, object],
    ) -> ValidatedBoundaryDependency:
        return cls(
            dependency_id=str(value["dependency_id"]),
            local_dependency_id=str(value["local_dependency_id"]),
            candidate_id=(
                str(value["candidate_id"])
                if value.get("candidate_id") is not None
                else None
            ),
            direction=str(value["direction"]),
            missing_field=str(value["missing_field"]),
            source=tuple(
                dict(item)
                for item in value["source"]  # type: ignore[union-attr]
            ),
            reason=str(value["reason"]),
        )


@dataclass(frozen=True, slots=True)
class PrimaryLineAssessment:
    position: SourcePosition
    role: str
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            **_position_dict(self.position),
            "role": self.role,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> PrimaryLineAssessment:
        return cls(
            position=_position_from_dict(value),
            role=str(value["role"]),
            reason=str(value["reason"]),
        )


@dataclass(frozen=True, slots=True)
class WindowCandidateResult:
    parent_attempt_id: str
    child_attempt_id: str
    window_id: str
    plan_hash: str
    outcome: str
    explanation: str
    candidates: tuple[ValidatedWindowCandidate, ...]
    boundary_dependencies: tuple[ValidatedBoundaryDependency, ...]
    primary_line_assessments: tuple[PrimaryLineAssessment, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "parent_attempt_id": self.parent_attempt_id,
            "child_attempt_id": self.child_attempt_id,
            "window_id": self.window_id,
            "plan_hash": self.plan_hash,
            "outcome": self.outcome,
            "explanation": self.explanation,
            "candidates": [item.to_dict() for item in self.candidates],
            "boundary_dependencies": [
                item.to_dict() for item in self.boundary_dependencies
            ],
            "primary_line_assessments": [
                item.to_dict() for item in self.primary_line_assessments
            ],
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> WindowCandidateResult:
        return cls(
            parent_attempt_id=str(value["parent_attempt_id"]),
            child_attempt_id=str(value["child_attempt_id"]),
            window_id=str(value["window_id"]),
            plan_hash=str(value["plan_hash"]),
            outcome=str(value["outcome"]),
            explanation=str(value["explanation"]),
            candidates=tuple(
                ValidatedWindowCandidate.from_dict(dict(item))
                for item in value["candidates"]  # type: ignore[union-attr]
            ),
            boundary_dependencies=tuple(
                ValidatedBoundaryDependency.from_dict(dict(item))
                for item in value["boundary_dependencies"]  # type: ignore[union-attr]
            ),
            primary_line_assessments=tuple(
                PrimaryLineAssessment.from_dict(dict(item))
                for item in value["primary_line_assessments"]  # type: ignore[union-attr]
            ),
        )


class WindowCandidateService:
    RECORD_KIND = "decomposition_window_result"
    SCHEMA_VERSION = 1
    MAX_CANDIDATES = 64
    MAX_BOUNDARY_DEPENDENCIES = 64

    def __init__(
        self,
        *,
        document: SourceDocument,
        uow_factory: Callable[[], UnitOfWork],
    ) -> None:
        self.document = document
        self.uow_factory = uow_factory
        self.validator = DecompositionValidator(document)

    def validate(
        self,
        *,
        parent: WindowedAttemptState,
        plan: WindowPlan,
        window_id: str,
        child_attempt_id: str,
        arguments: WindowCandidateArguments,
    ) -> WindowCandidateResult:
        window = validate_window_binding(
            parent,
            plan,
            window_id,
            child_attempt_id,
            expected_schema_version=WINDOW_CANDIDATE_SCHEMA_VERSION,
        )
        outcome = arguments.outcome
        if outcome not in {
            "candidates",
            "no_local_testable_behavior",
            "boundary_dependency",
        }:
            raise WindowCandidateError(
                f"Неизвестный локальный outcome {outcome}"
            )
        if not arguments.explanation.strip():
            raise WindowCandidateError(
                "Локальный outcome требует объяснение"
            )

        candidates = self._validate_candidates(
            window,
            arguments.candidates,
        )
        dependencies = self._validate_dependencies(
            window,
            arguments.boundary_dependencies,
            candidates,
        )
        if outcome == "candidates" and not candidates:
            raise WindowCandidateError(
                "outcome=candidates требует непустой candidates; "
                "для dependency без candidates используй "
                "outcome=boundary_dependency"
            )
        if outcome == "no_local_testable_behavior" and (
            candidates or dependencies
        ):
            raise WindowCandidateError(
                "outcome=no_local_testable_behavior требует candidates=[] "
                "и boundary_dependencies=[]; при наличии candidates используй "
                "outcome=candidates, при наличии только dependency — "
                "outcome=boundary_dependency"
            )
        if outcome == "boundary_dependency" and (
            candidates or not dependencies
        ):
            raise WindowCandidateError(
                "outcome=boundary_dependency требует candidates=[] и непустой "
                "boundary_dependencies; при наличии candidates используй "
                "outcome=candidates"
            )

        evidence_positions = {
            position
            for candidate in candidates
            for position in candidate.evidence_positions
            if position in set(window.primary_positions)
        }
        assessments = self._derive_primary_assessments(
            window,
            evidence_positions,
        )
        return WindowCandidateResult(
            parent_attempt_id=parent.parent_attempt_id,
            child_attempt_id=child_attempt_id,
            window_id=window_id,
            plan_hash=plan.plan_hash,
            outcome=outcome,
            explanation=arguments.explanation,
            candidates=candidates,
            boundary_dependencies=dependencies,
            primary_line_assessments=assessments,
        )

    def accept(
        self,
        *,
        parent: WindowedAttemptState,
        plan: WindowPlan,
        window_id: str,
        child_attempt_id: str,
        arguments: WindowCandidateArguments,
        raw_arguments: dict[str, object],
        uow: UnitOfWork | None = None,
    ) -> WindowCandidateResult:
        if raw_arguments != asdict(arguments):
            raise WindowCandidateError(
                "Raw window arguments не соответствуют typed arguments"
            )
        result = self.validate(
            parent=parent,
            plan=plan,
            window_id=window_id,
            child_attempt_id=child_attempt_id,
            arguments=arguments,
        )
        if uow is not None:
            self._save(result, raw_arguments, uow)
        else:
            with self.uow_factory() as owned_uow:
                self._save(result, raw_arguments, owned_uow)
        return result

    def load(
        self,
        window_id: str,
        *,
        parent_attempt_id: str,
        plan_hash: str,
    ) -> WindowCandidateResult | None:
        with self.uow_factory() as uow:
            record = uow.records.get(
                self.RECORD_KIND,
                self._record_id(parent_attempt_id, window_id),
            )
        if record is None:
            return None
        if int(record.payload.get("schema_version", -1)) != self.SCHEMA_VERSION:
            raise WindowCandidateError(
                "Неизвестная schema validated window result"
            )
        fingerprint_payload = {
            key: value
            for key, value in record.payload.items()
            if key != "fingerprint"
        }
        if (
            self._fingerprint(fingerprint_payload)
            != record.payload.get("fingerprint")
        ):
            raise WindowCandidateError(
                "Validated window result fingerprint не соответствует содержимому"
            )
        result = WindowCandidateResult.from_dict(
            dict(record.payload["validated"])
        )
        if result.parent_attempt_id != parent_attempt_id:
            raise WindowCandidateError(
                "Window result относится к другому parent attempt"
            )
        if result.plan_hash != plan_hash:
            raise WindowCandidateError(
                "Window result относится к другому window plan"
            )
        if result.window_id != window_id:
            raise WindowCandidateError(
                "Window result относится к другому окну"
            )
        return result

    def _validate_candidates(
        self,
        window: DecompositionWindow,
        raw_candidates: object,
    ) -> tuple[ValidatedWindowCandidate, ...]:
        if not isinstance(raw_candidates, list):
            raise WindowCandidateError("candidates должен быть списком")
        if len(raw_candidates) > self.MAX_CANDIDATES:
            raise WindowCandidateError(
                "Окно превышает лимит candidates"
            )
        local_ids: set[str] = set()
        validated: list[ValidatedWindowCandidate] = []
        for index, raw in enumerate(raw_candidates, start=1):
            if not isinstance(raw, dict):
                raise WindowCandidateError(
                    "Candidate должен быть объектом"
                )
            item = dict(raw)
            local_id = str(item.pop("local_candidate_id", ""))
            if not local_id.strip() or local_id in local_ids:
                raise WindowCandidateError(
                    "Local candidate IDs должны быть непустыми и уникальными"
                )
            local_ids.add(local_id)
            try:
                payload, positions = self.validator.validate_skeleton(
                    window.as_selection(),
                    item,
                )
            except (DecompositionError, ValueError) as error:
                raise WindowCandidateError(
                    "Candidate содержит evidence вне окна или невалиден: "
                    f"{error}. {_window_coordinate_help(window)}"
                ) from error
            if not set(positions).intersection(window.primary_positions):
                raise WindowCandidateError(
                    "Candidate должен содержать evidence хотя бы одной строки "
                    "primary=true; поведение только из overlap "
                    "принадлежит другому окну"
                )
            validated.append(
                ValidatedWindowCandidate(
                    candidate_id=(
                        f"{window.window_id}:CANDIDATE:{index:04d}"
                    ),
                    local_candidate_id=local_id,
                    window_id=window.window_id,
                    payload=payload,
                    evidence_positions=tuple(sorted(positions)),
                )
            )
        return tuple(validated)

    def _validate_dependencies(
        self,
        window: DecompositionWindow,
        raw_dependencies: object,
        candidates: tuple[ValidatedWindowCandidate, ...],
    ) -> tuple[ValidatedBoundaryDependency, ...]:
        if not isinstance(raw_dependencies, list):
            raise WindowCandidateError(
                "boundary_dependencies должен быть списком"
            )
        if len(raw_dependencies) > self.MAX_BOUNDARY_DEPENDENCIES:
            raise WindowCandidateError(
                "Окно превышает лимит boundary dependencies"
            )
        candidate_ids = {
            item.local_candidate_id: item.candidate_id
            for item in candidates
        }
        dependency_ids: set[str] = set()
        result: list[ValidatedBoundaryDependency] = []
        for index, raw in enumerate(raw_dependencies, start=1):
            if not isinstance(raw, dict) or set(raw) != {
                "local_dependency_id",
                "local_candidate_id",
                "direction",
                "missing_field",
                "source_ranges",
                "reason",
            }:
                raise WindowCandidateError(
                    "Boundary dependency имеет неверную структуру"
                )
            local_id = str(raw["local_dependency_id"])
            if not local_id.strip() or local_id in dependency_ids:
                raise WindowCandidateError(
                    "Local dependency IDs должны быть непустыми и уникальными"
                )
            dependency_ids.add(local_id)
            local_candidate_id = raw["local_candidate_id"]
            candidate_id: str | None = None
            if local_candidate_id is not None:
                candidate_id = candidate_ids.get(str(local_candidate_id))
                if candidate_id is None:
                    raise WindowCandidateError(
                        "Boundary dependency ссылается на неизвестный candidate"
                    )
            direction = str(raw["direction"])
            if direction not in {"before", "after"}:
                raise WindowCandidateError(
                    f"Неизвестное направление dependency: {direction}"
                )
            missing_field = str(raw["missing_field"])
            if missing_field not in {
                "condition",
                "changed_factor",
                "input_value",
                "action",
                "consequence",
            }:
                raise WindowCandidateError(
                    f"Неизвестное missing field: {missing_field}"
                )
            reason = str(raw["reason"])
            if not reason.strip():
                raise WindowCandidateError(
                    "Boundary dependency требует причину"
                )
            try:
                source = self.validator.evidence(
                    window.as_selection(),
                    raw["source_ranges"],
                )
            except (DecompositionError, ValueError) as error:
                raise WindowCandidateError(
                    "Boundary dependency выходит за пределы окна: "
                    f"{error}. {_window_coordinate_help(window)}"
                ) from error
            source_positions = self.validator.evidence_positions(source)
            ordered_positions = tuple(
                line.position for line in window.lines
            )
            index_by_position = {
                position: index
                for index, position in enumerate(ordered_positions)
            }
            primary_start = index_by_position[window.primary_positions[0]]
            primary_end = index_by_position[window.primary_positions[-1]]
            source_indexes = {
                index_by_position[position]
                for position in source_positions
            }
            touches_boundary = (
                min(source_indexes) <= primary_start
                if direction == "before"
                else max(source_indexes) >= primary_end
            )
            if not touches_boundary:
                raise WindowCandidateError(
                    "Boundary dependency не достигает указанной границы окна"
                )
            result.append(
                ValidatedBoundaryDependency(
                    dependency_id=(
                        f"{window.window_id}:DEPENDENCY:{index:04d}"
                    ),
                    local_dependency_id=local_id,
                    candidate_id=candidate_id,
                    direction=direction,
                    missing_field=missing_field,
                    source=tuple(source),
                    reason=reason,
                )
            )
        return tuple(result)

    @staticmethod
    def _derive_primary_assessments(
        window: DecompositionWindow,
        evidence_positions: set[SourcePosition],
    ) -> tuple[PrimaryLineAssessment, ...]:
        return tuple(
            PrimaryLineAssessment(
                position,
                "evidence",
                "Строка включена в source ranges local candidate",
            )
            if position in evidence_positions
            else PrimaryLineAssessment(
                position,
                "context",
                "Строка не включена в source ranges local candidate",
            )
            for position in window.primary_positions
        )

    @staticmethod
    def _validate_binding(
        parent: WindowedAttemptState,
        plan: WindowPlan,
        window_id: str,
        child_attempt_id: str,
    ) -> DecompositionWindow:
        return validate_window_binding(
            parent,
            plan,
            window_id,
            child_attempt_id,
            expected_schema_version=WINDOW_CANDIDATE_SCHEMA_VERSION,
        )

    def _save(
        self,
        result: WindowCandidateResult,
        raw_arguments: dict[str, object],
        uow: UnitOfWork,
    ) -> None:
        payload = json.loads(
            json.dumps(
                {
                    "schema_version": self.SCHEMA_VERSION,
                    "raw_arguments": raw_arguments,
                    "validated": result.to_dict(),
                },
                ensure_ascii=False,
            )
        )
        fingerprint = self._fingerprint(payload)
        payload["fingerprint"] = fingerprint
        record_id = self._record_id(
            result.parent_attempt_id,
            result.window_id,
        )
        existing = uow.records.get(self.RECORD_KIND, record_id)
        if existing is not None:
            if existing.payload.get("fingerprint") != fingerprint:
                raise WindowCandidateError(
                    "Для окна уже сохранён другой validated result"
                )
            return
        uow.records.save(
            StoredRecord(self.RECORD_KIND, record_id, payload)
        )
        uow.events.append(
            result.parent_attempt_id,
            "результат окна сохранён",
            {
                "window_id": result.window_id,
                "child_attempt_id": result.child_attempt_id,
                "outcome": result.outcome,
            },
        )

    @staticmethod
    def _record_id(parent_attempt_id: str, window_id: str) -> str:
        return f"{parent_attempt_id}:{window_id}"

    @staticmethod
    def _fingerprint(payload: dict[str, object]) -> str:
        return hashlib.sha256(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()


def window_context(window: DecompositionWindow) -> dict[str, object]:
    return {
        "window_id": window.window_id,
        "index": window.index,
        "global_start": _position_dict(window.global_start),
        "global_end": _position_dict(window.global_end),
        "outline": {
            "node_id": window.outline_node_id,
            "label": window.outline_label,
            "path": list(window.outline_path),
        },
        "lines": [
            {
                "page": line.position.page_index,
                "line": line.position.line_number,
                "text": line.text,
                "primary": line.primary,
            }
            for line in window.lines
        ],
    }


def _position_dict(position: SourcePosition) -> dict[str, int]:
    return {
        "page": position.page_index,
        "line": position.line_number,
    }


def _position_from_dict(value: dict[str, object]) -> SourcePosition:
    return SourcePosition(int(value["page"]), int(value["line"]))


def validate_window_binding(
    parent: WindowedAttemptState,
    plan: WindowPlan,
    window_id: str,
    child_attempt_id: str,
    *,
    expected_schema_version: str,
) -> DecompositionWindow:
    if parent.status is not WindowedAttemptStatus.RUNNING:
        raise WindowCandidateError("Parent attempt не выполняется")
    if parent.window_plan_hash != plan.plan_hash:
        raise WindowCandidateError(
            "Parent attempt ссылается на другой window plan"
        )
    if plan.recompute_hash() != plan.plan_hash:
        raise WindowCandidateError(
            "Window plan hash не соответствует содержимому"
        )
    if parent.selection_id != plan.selection_id:
        raise WindowCandidateError(
            "Parent attempt относится к другому selection"
        )
    if parent.document_version != plan.document_version:
        raise WindowCandidateError(
            "Parent attempt содержит другой document_version"
        )
    if parent.policy_version != plan.policy_version:
        raise WindowCandidateError(
            "Parent attempt содержит другую policy version"
        )
    if parent.schema_version != expected_schema_version:
        raise WindowCandidateError(
            "Parent attempt содержит другую candidate schema version"
        )
    try:
        child = next(
            item for item in parent.children if item.window_id == window_id
        )
        window = next(
            item for item in plan.windows if item.window_id == window_id
        )
    except StopIteration as error:
        raise WindowCandidateError(f"Неизвестное окно {window_id}") from error
    if (
        child.status is not WindowChildStatus.RUNNING
        or child.attempt_id != child_attempt_id
    ):
        raise WindowCandidateError(
            "Child result не соответствует активному attempt"
        )
    return window
