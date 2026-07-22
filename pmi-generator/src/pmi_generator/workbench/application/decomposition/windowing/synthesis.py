from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ....domain.source import SourceDocument, SourcePosition
from .plan import WindowPlan
from .semantic import (
    SemanticBehaviorFragment,
    SemanticFact,
    SemanticWindowResult,
)


SYNTHESIS_SINGLETON_FIELDS = (
    "condition",
    "changed_factor",
    "input_value",
    "action",
)
SYNTHESIS_REQUIRED_FIELDS = ("condition", "changed_factor", "consequences")


@dataclass(frozen=True, slots=True)
class SemanticSynthesisArguments:
    candidates: list[dict[str, object]]


class SemanticFactScope(str, Enum):
    PRIMARY = "primary"
    CONTEXT = "context"


def semantic_fact_scopes(
    *,
    plan: WindowPlan,
    target_window_id: str,
    results: tuple[SemanticWindowResult, ...],
) -> dict[str, SemanticFactScope]:
    target_window = next(
        window
        for window in plan.windows
        if window.window_id == target_window_id
    )
    primary_positions = set(target_window.primary_positions)
    facts = tuple(
        fact
        for _fragment, fragment_facts in _context_fragments(
            plan,
            target_window_id,
            results,
        )
        for fact in fragment_facts
    )
    fact_ids = tuple(fact.fact_id for fact in facts)
    if len(fact_ids) != len(set(fact_ids)):
        raise ValueError("Semantic synthesis context содержит duplicate fact ID")
    return {
        fact.fact_id: _fact_scope(fact, primary_positions)
        for fact in facts
    }


def semantic_synthesis_context(
    *,
    document: SourceDocument,
    plan: WindowPlan,
    target_window_id: str,
    results: tuple[SemanticWindowResult, ...],
) -> dict[str, object]:
    fragments = _context_fragments(
        plan,
        target_window_id,
        results,
    )
    scopes = semantic_fact_scopes(
        plan=plan,
        target_window_id=target_window_id,
        results=results,
    )
    return {
        "target_window_id": target_window_id,
        "target_fragments": [
            {
                "fragment_id": fragment.fragment_id,
                "title": fragment.title,
                "summary": fragment.summary,
                "target_facts": [
                    _fact_context(document, fact)
                    for fact in fragment_facts
                    if scopes[fact.fact_id] is SemanticFactScope.PRIMARY
                ],
                "supporting_facts": [
                    _fact_context(document, fact)
                    for fact in fragment_facts
                    if scopes[fact.fact_id] is SemanticFactScope.CONTEXT
                ],
            }
            for fragment, fragment_facts in fragments
        ],
    }


def _context_window_ids(
    plan: WindowPlan,
    target_window_id: str,
) -> set[str]:
    target_index = next(
        window.index
        for window in plan.windows
        if window.window_id == target_window_id
    )
    return {
        window.window_id
        for window in plan.windows
        if abs(window.index - target_index) <= 1
    }


def _context_fragments(
    plan: WindowPlan,
    target_window_id: str,
    results: tuple[SemanticWindowResult, ...],
) -> tuple[
    tuple[SemanticBehaviorFragment, tuple[SemanticFact, ...]],
    ...,
]:
    target_window = next(
        window
        for window in plan.windows
        if window.window_id == target_window_id
    )
    allowed_positions = {line.position for line in target_window.lines}
    context_window_ids = _context_window_ids(plan, target_window_id)
    results_by_window = {result.window_id: result for result in results}
    expected_window_ids = {window.window_id for window in plan.windows}
    if (
        len(results_by_window) != len(results)
        or set(results_by_window) != expected_window_ids
    ):
        raise ValueError(
            "Semantic synthesis facts не покрывают window plan ровно один раз"
        )
    fragments: list[
        tuple[SemanticBehaviorFragment, tuple[SemanticFact, ...]]
    ] = []
    for window in plan.windows:
        if window.window_id not in context_window_ids:
            continue
        result = results_by_window[window.window_id]
        for fragment in result.fragments:
            facts = tuple(
                fact
                for fact in fragment.facts
                if set(fact.positions) <= allowed_positions
            )
            if facts and any(
                set(fact.positions).intersection(
                    target_window.primary_positions
                )
                for fact in facts
            ):
                fragments.append((fragment, facts))
    return tuple(fragments)


def _fact_context(
    document: SourceDocument,
    fact: SemanticFact,
) -> dict[str, object]:
    return {
        "fact_id": fact.fact_id,
        "text": fact.text,
        "source_lines": [
            {
                "page": position.page_index,
                "line": position.line_number,
                "text": document.line(position),
            }
            for position in fact.positions
        ],
    }


def _fact_scope(
    fact: SemanticFact,
    primary_positions: set[SourcePosition],
) -> SemanticFactScope:
    return (
        SemanticFactScope.PRIMARY
        if set(fact.positions).intersection(primary_positions)
        else SemanticFactScope.CONTEXT
    )
