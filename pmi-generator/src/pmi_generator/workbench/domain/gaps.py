from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any, Mapping

from .enums import GapResolutionMode, GapStatus
from .errors import DomainValidationError, PathNotAllowedError
from .schema import CARD_FIELD_PATHS


class GapValueForm(StrEnum):
    CONFIRMED_VALUE = "confirmed_value"
    EXACT_VALUE = "exact"
    FINITE_SET = "finite_set"
    DETERMINISTIC_RULE = "deterministic_rule"


class GapClosureOutcome(StrEnum):
    SATISFIED = "satisfied"
    PARTIALLY_SATISFIED = "partially_satisfied"
    INSUFFICIENT = "insufficient"


@dataclass(frozen=True, slots=True)
class GapPathClosure:
    path: str
    accepted_forms: tuple[GapValueForm, ...]
    residual_question: str

    def __post_init__(self) -> None:
        if self.path not in CARD_FIELD_PATHS:
            raise DomainValidationError(
                f"Closure contract содержит неизвестный путь {self.path}"
            )
        if not self.accepted_forms or len(self.accepted_forms) != len(
            set(self.accepted_forms)
        ):
            raise DomainValidationError(
                "Closure requirement требует уникальные допустимые формы"
            )
        if any(not isinstance(item, GapValueForm) for item in self.accepted_forms):
            raise DomainValidationError(
                "Closure requirement содержит неизвестную форму значения"
            )
        if not self.residual_question.strip():
            raise DomainValidationError(
                "Closure requirement требует остаточный вопрос"
            )


@dataclass(frozen=True, slots=True)
class GapClosureEvaluation:
    outcome: GapClosureOutcome
    satisfied_paths: tuple[str, ...]
    remaining_paths: tuple[str, ...]
    remaining_questions: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class GapClosureContract:
    requirements: tuple[GapPathClosure, ...]
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise DomainValidationError(
                f"Неподдерживаемая версия closure contract: {self.schema_version}"
            )
        if not self.requirements:
            raise DomainValidationError(
                "Closure contract требует хотя бы одно поле"
            )
        paths = tuple(item.path for item in self.requirements)
        if len(paths) != len(set(paths)):
            raise DomainValidationError(
                "Closure contract содержит повторяющиеся пути"
            )

    @classmethod
    def legacy(
        cls,
        allowed_paths: tuple[str, ...],
        *,
        question: str,
    ) -> GapClosureContract:
        return cls(
            requirements=tuple(
                GapPathClosure(
                    path=path,
                    accepted_forms=tuple(GapValueForm),
                    residual_question=question,
                )
                for path in allowed_paths
            )
        )

    def evaluate(
        self,
        values: Mapping[str, Any],
        *,
        source_confirmed: bool,
        previously_satisfied: tuple[str, ...] = (),
    ) -> GapClosureEvaluation:
        known_paths = {item.path for item in self.requirements}
        unknown = set(values) - known_paths
        if unknown:
            raise DomainValidationError(
                f"Closure evaluation содержит неизвестные пути: {sorted(unknown)}"
            )
        satisfied = set(previously_satisfied) - set(values)
        if satisfied - known_paths:
            raise DomainValidationError(
                "Closure progress содержит неизвестные пути"
            )
        for requirement in self.requirements:
            if requirement.path not in values:
                continue
            if source_confirmed and values[requirement.path] is None:
                raise DomainValidationError(
                    "Source-confirmed closure value не может быть null"
                )
            form = (
                GapValueForm.EXACT_VALUE
                if source_confirmed
                else _submitted_value_form(values[requirement.path])
            )
            if form in requirement.accepted_forms:
                satisfied.add(requirement.path)

        ordered_satisfied = tuple(
            item.path for item in self.requirements if item.path in satisfied
        )
        remaining = tuple(
            item.path for item in self.requirements if item.path not in satisfied
        )
        questions = tuple(
            item.residual_question
            for item in self.requirements
            if item.path not in satisfied
        )
        if not remaining:
            outcome = GapClosureOutcome.SATISFIED
        elif ordered_satisfied:
            outcome = GapClosureOutcome.PARTIALLY_SATISFIED
        else:
            outcome = GapClosureOutcome.INSUFFICIENT
        return GapClosureEvaluation(
            outcome=outcome,
            satisfied_paths=ordered_satisfied,
            remaining_paths=remaining,
            remaining_questions=questions,
        )

    def normalize_values(
        self,
        values: Mapping[str, Any],
        *,
        source_confirmed: bool,
    ) -> dict[str, Any]:
        unknown = set(values) - {
            item.path for item in self.requirements
        }
        if unknown:
            raise DomainValidationError(
                f"Closure normalization содержит неизвестные пути: {sorted(unknown)}"
            )
        return {
            path: (
                value
                if source_confirmed
                else _normalize_submitted_value(value)
            )
            for path, value in values.items()
        }


def _submitted_value_form(value: Any) -> GapValueForm:
    if not isinstance(value, dict) or "kind" not in value:
        if value is None:
            raise DomainValidationError(
                "Подтверждённое значение closure contract не может быть null"
            )
        return GapValueForm.CONFIRMED_VALUE

    try:
        form = GapValueForm(str(value["kind"]))
    except ValueError as error:
        raise DomainValidationError(
            f"Неизвестная форма closure value: {value['kind']}"
        ) from error
    if form is GapValueForm.CONFIRMED_VALUE:
        if set(value) != {"kind", "value"} or value["value"] is None:
            raise DomainValidationError(
                "Форма confirmed_value требует ровно одно непустое value"
            )
    elif form is GapValueForm.EXACT_VALUE:
        if set(value) != {"kind", "value"} or value["value"] is None:
            raise DomainValidationError(
                "Форма exact требует ровно одно непустое value"
            )
    elif form is GapValueForm.FINITE_SET:
        values = value.get("values")
        if (
            set(value) != {"kind", "values"}
            or not isinstance(values, list)
            or not values
        ):
            raise DomainValidationError(
                "Форма finite_set требует непустой список values"
            )
    else:
        if (
            set(value) != {"kind", "rule", "parameters"}
            or not isinstance(value.get("rule"), str)
            or not value["rule"].strip()
            or not isinstance(value.get("parameters"), dict)
        ):
            raise DomainValidationError(
                "Форма deterministic_rule требует rule и parameters"
            )
    return form


def _normalize_submitted_value(value: Any) -> Any:
    form = _submitted_value_form(value)
    if form is GapValueForm.CONFIRMED_VALUE:
        return (
            value["value"]
            if isinstance(value, dict) and "kind" in value
            else value
        )
    if form is GapValueForm.EXACT_VALUE:
        return value["value"]
    if form is GapValueForm.FINITE_SET:
        return list(value["values"])
    return {
        "rule": value["rule"],
        "parameters": dict(value["parameters"]),
    }


@dataclass(frozen=True, slots=True)
class RelatedGap:
    gap_id: str
    card_id: str
    question: str
    blocking_reason: str
    allowed_paths: tuple[str, ...]
    dependencies: tuple[str, ...]
    closure_criterion: str
    closure_contract: GapClosureContract | None = None
    closure_satisfied_paths: tuple[str, ...] = ()
    resolution_mode: GapResolutionMode = GapResolutionMode.SOURCE_FACT
    status: GapStatus = GapStatus.OPEN

    def __post_init__(self) -> None:
        if not self.gap_id.strip() or not self.card_id.strip():
            raise DomainValidationError("Пробел должен иметь ID и card_id")
        if not self.question.strip() or not self.blocking_reason.strip():
            raise DomainValidationError("Пробел должен иметь один вопрос и причину блокировки")
        if not self.closure_criterion.strip() or not self.allowed_paths:
            raise DomainValidationError("Пробел должен иметь критерий закрытия и разрешённые поля")
        if not isinstance(self.resolution_mode, GapResolutionMode):
            raise DomainValidationError("Пробел должен иметь допустимый resolution_mode")
        unknown = (set(self.allowed_paths) | set(self.dependencies)) - CARD_FIELD_PATHS
        if unknown:
            raise DomainValidationError(f"Неизвестные пути карточки: {sorted(unknown)}")
        contract = self.closure_contract or GapClosureContract.legacy(
            self.allowed_paths,
            question=self.question,
        )
        if set(item.path for item in contract.requirements) != set(
            self.allowed_paths
        ):
            raise DomainValidationError(
                "Closure contract должен покрывать allowed_paths gap"
            )
        if (
            len(self.closure_satisfied_paths)
            != len(set(self.closure_satisfied_paths))
            or set(self.closure_satisfied_paths) - set(self.allowed_paths)
        ):
            raise DomainValidationError(
                "Closure progress содержит неверные пути"
            )
        object.__setattr__(self, "closure_contract", contract)

    def assert_allows(self, paths: tuple[str, ...] | list[str]) -> None:
        forbidden = set(paths) - set(self.allowed_paths)
        if forbidden:
            raise PathNotAllowedError(
                f"Пробел {self.gap_id} не разрешает изменять поля: {sorted(forbidden)}"
            )

    def resolve(self) -> RelatedGap:
        required = {
            item.path for item in self.closure_contract.requirements
        }
        if set(self.closure_satisfied_paths) != required:
            raise DomainValidationError(
                "Пробел нельзя закрыть до выполнения closure contract"
            )
        return replace(self, status=GapStatus.RESOLVED)

    def with_closure_progress(
        self,
        satisfied_paths: tuple[str, ...],
    ) -> RelatedGap:
        ordered = tuple(
            item.path
            for item in self.closure_contract.requirements
            if item.path in set(satisfied_paths)
        )
        if len(ordered) != len(set(satisfied_paths)):
            raise DomainValidationError(
                "Closure progress содержит неизвестные или повторные пути"
            )
        return replace(self, closure_satisfied_paths=ordered)

    def with_resolution_mode(self, mode: GapResolutionMode) -> RelatedGap:
        return replace(self, resolution_mode=mode)

    def leave_open(self) -> RelatedGap:
        return replace(self, status=GapStatus.LEFT_OPEN)
