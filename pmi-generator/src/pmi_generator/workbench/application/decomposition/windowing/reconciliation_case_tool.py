from __future__ import annotations

from typing import Any

from ...llm import ToolContractError, ToolSpec
from .reconciliation_cases import (
    DEPENDENCY_DECISIONS,
    PAIR_DECISIONS,
    REVIEW_DECISIONS,
    ReconciliationCaseArguments,
)


def _schema(decisions: tuple[str, ...] | None = None) -> dict[str, Any]:
    decision: dict[str, Any] = {"type": "string"}
    if decisions is not None:
        decision["enum"] = list(decisions)
    return {
        "type": "object",
        "properties": {
            "decision": decision,
            "reason": {"type": "string", "minLength": 1},
        },
        "required": ["decision", "reason"],
        "additionalProperties": False,
    }


def _contextual_schema(context: dict[str, Any]) -> dict[str, Any]:
    case = context.get("case")
    if not isinstance(case, dict):
        raise ToolContractError(
            "Reconciliation case tool требует case context"
        )
    case_kind = str(case.get("case_kind", ""))
    decisions = {
        "candidate_pair": PAIR_DECISIONS,
        "candidate_review": REVIEW_DECISIONS,
        "dependency_review": DEPENDENCY_DECISIONS,
    }.get(case_kind)
    if decisions is None:
        raise ToolContractError(
            f"Неизвестный reconciliation case kind {case_kind}"
        )
    return _schema(tuple(sorted(decisions)))


def reconciliation_case_tool() -> ToolSpec:
    return ToolSpec(
        name="submit_reconciliation_case",
        description=(
            "Принять одно semantic merge/split/duplicate решение для "
            "application-bound reconciliation case без технических IDs"
        ),
        arguments_type=ReconciliationCaseArguments,
        json_schema=_schema(),
        contextual_schema=_contextual_schema,
    )
