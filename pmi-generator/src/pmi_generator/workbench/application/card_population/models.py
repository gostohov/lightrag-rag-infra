from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from ...domain.schema import EXPECTED_RESULT_PATHS, REQUIRED_FIELD_PATHS


@dataclass(slots=True)
class PopulationArguments:
    source_values: list[dict[str, Any]]
    derivations: list[dict[str, Any]]
    not_applicable: list[dict[str, Any]]
    gaps: list[dict[str, Any]]
    analyst_values: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.analyst_values:
            raise ValueError(
                "Prompt 2 не принимает analyst_values; "
                "используйте подтверждаемый conversation proposal"
            )
        occupied: dict[str, str] = {}
        duplicates: list[tuple[str, str, str]] = []
        populated: set[str] = set()
        gap_paths: set[str] = set()

        def claim(path: str, section: str) -> None:
            previous = occupied.get(path)
            if previous is not None:
                duplicates.append((path, previous, section))
            else:
                occupied[path] = section

        for section, items in (
            ("source_values", self.source_values),
            ("derivations", self.derivations),
            ("not_applicable", self.not_applicable),
        ):
            for item in items:
                path = str(item["path"])
                claim(path, section)
                if section != "not_applicable":
                    populated.add(path)
        for gap in self.gaps:
            for path in gap["allowed_paths"]:
                normalized = str(path)
                claim(normalized, "gaps.allowed_paths")
                gap_paths.add(normalized)

        covered = populated | gap_paths
        missing_required = sorted(REQUIRED_FIELD_PATHS - covered)
        violations = [
            (
                f"Путь {path} присутствует в нескольких разделах: "
                f"{previous}, {section}"
            )
            for path, previous, section in duplicates
        ]
        if missing_required:
            coverage = ", ".join(
                (
                    f"{path}=not_applicable (недопустимо)"
                    if occupied.get(path) == "not_applicable"
                    else f"{path}={occupied.get(path, 'не покрыто')}"
                )
                for path in sorted(REQUIRED_FIELD_PATHS)
            )
            violations.append(
                "Результат Prompt 2 не покрывает обязательные поля: "
                + ", ".join(missing_required)
                + ". Текущее покрытие: "
                + coverage
                + ". При исправлении сохрани корректно покрытые обязательные поля"
            )
        if violations:
            raise ValueError("; ".join(violations))
        if not EXPECTED_RESULT_PATHS & covered:
            raise ValueError(
                "Результат Prompt 2 не содержит ожидаемый результат или связанный gap"
            )

    def as_dict(self) -> dict[str, object]:
        return {
            "source_values": self.source_values,
            "derivations": self.derivations,
            "not_applicable": self.not_applicable,
            "gaps": self.gaps,
        }


@dataclass(frozen=True, slots=True)
class AnalystMessage:
    message_id: str
    card_id: str
    author: str
    text: str
    created_at: datetime


@dataclass(frozen=True, slots=True)
class PopulationResult:
    card_id: str
    revision: int
    open_gap_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PopulationStart:
    attempt_id: str
    instruction: str | None


class PopulationError(ValueError):
    pass
