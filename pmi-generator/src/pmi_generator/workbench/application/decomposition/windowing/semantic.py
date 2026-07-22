from __future__ import annotations

from dataclasses import dataclass

from ....domain.source import SourcePosition
from .plan import DecompositionWindow
from .semantic_versions import (
    SEMANTIC_CANONICAL_MAPPING_VERSION,
    SEMANTIC_SYNTHESIS_SCHEMA_VERSION,
    SEMANTIC_WINDOW_SCHEMA_VERSION,
)

@dataclass(frozen=True, slots=True)
class SemanticWindowArguments:
    behaviors: list[dict[str, object]]


@dataclass(frozen=True, slots=True)
class SemanticFact:
    fact_id: str
    text: str
    positions: tuple[SourcePosition, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "fact_id": self.fact_id,
            "text": self.text,
            "positions": [
                {
                    "page": position.page_index,
                    "line": position.line_number,
                }
                for position in self.positions
            ],
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> SemanticFact:
        return cls(
            fact_id=str(value["fact_id"]),
            text=str(value["text"]),
            positions=tuple(
                SourcePosition(int(item["page"]), int(item["line"]))
                for item in value["positions"]  # type: ignore[union-attr]
            ),
        )


@dataclass(frozen=True, slots=True)
class SemanticBehaviorFragment:
    fragment_id: str
    window_id: str
    title: str
    summary: str
    facts: tuple[SemanticFact, ...]

    @property
    def positions(self) -> tuple[SourcePosition, ...]:
        return tuple(
            dict.fromkeys(
                position
                for fact in self.facts
                for position in fact.positions
            )
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "fragment_id": self.fragment_id,
            "window_id": self.window_id,
            "title": self.title,
            "summary": self.summary,
            "facts": [fact.to_dict() for fact in self.facts],
        }

    @classmethod
    def from_dict(
        cls,
        value: dict[str, object],
    ) -> SemanticBehaviorFragment:
        return cls(
            fragment_id=str(value["fragment_id"]),
            window_id=str(value["window_id"]),
            title=str(value["title"]),
            summary=str(value["summary"]),
            facts=tuple(
                SemanticFact.from_dict(dict(item))
                for item in value["facts"]  # type: ignore[union-attr]
            ),
        )


@dataclass(frozen=True, slots=True)
class SemanticWindowResult:
    parent_attempt_id: str
    child_attempt_id: str
    window_id: str
    plan_hash: str
    fragments: tuple[SemanticBehaviorFragment, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "parent_attempt_id": self.parent_attempt_id,
            "child_attempt_id": self.child_attempt_id,
            "window_id": self.window_id,
            "plan_hash": self.plan_hash,
            "fragments": [item.to_dict() for item in self.fragments],
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> SemanticWindowResult:
        return cls(
            parent_attempt_id=str(value["parent_attempt_id"]),
            child_attempt_id=str(value["child_attempt_id"]),
            window_id=str(value["window_id"]),
            plan_hash=str(value["plan_hash"]),
            fragments=tuple(
                SemanticBehaviorFragment.from_dict(dict(item))
                for item in value["fragments"]  # type: ignore[union-attr]
            ),
        )


def semantic_window_context(
    window: DecompositionWindow,
) -> dict[str, object]:
    return {
        "outline": {
            "label": window.outline_label,
            "path": list(window.outline_path),
        },
        "primary_line_ids": [
            _line_id(index)
            for index, line in enumerate(window.lines, start=1)
            if line.primary
        ],
        "lines": [
            {
                "line_id": _line_id(index),
                "text": line.text,
                "primary": line.primary,
            }
            for index, line in enumerate(window.lines, start=1)
        ],
    }


def _line_id(index: int) -> str:
    return f"L{index:04d}"
