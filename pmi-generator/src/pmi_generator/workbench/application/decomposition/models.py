from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DecompositionArguments:
    outcome: str
    explanation: str
    skeletons: list[dict[str, object]]
    line_assessments: list[dict[str, object]]


@dataclass(frozen=True, slots=True)
class DecompositionResult:
    selection_id: str
    outcome: str
    explanation: str
    skeleton_ids: tuple[str, ...]


class DecompositionError(ValueError):
    pass
