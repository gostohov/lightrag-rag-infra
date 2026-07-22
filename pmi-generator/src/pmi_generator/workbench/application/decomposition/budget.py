from __future__ import annotations

import json
from dataclasses import dataclass

from ...domain.source import TextSelection
from ..prompting import PromptId, PromptPolicy


@dataclass(frozen=True, slots=True)
class DecompositionBudget:
    line_count: int
    estimated_tokens: int
    max_lines: int
    max_estimated_tokens: int
    estimator: str = "decomposition-json-utf8-div4-v1"

    @property
    def within_single_call(self) -> bool:
        return (
            self.line_count <= self.max_lines
            and self.estimated_tokens <= self.max_estimated_tokens
        )


class DecompositionBudgetExceededError(ValueError):
    def __init__(self, budget: DecompositionBudget) -> None:
        self.budget = budget
        super().__init__(
            "Диапазон превышает технический бюджет Prompt 1: "
            f"{budget.line_count}/{budget.max_lines} строк, "
            f"{budget.estimated_tokens}/{budget.max_estimated_tokens} "
            "оценочных токенов"
        )


class DecompositionBudgetPolicy:
    ESTIMATOR = "decomposition-json-utf8-div4-v1"

    def __init__(
        self,
        *,
        max_lines: int,
        max_estimated_tokens: int,
        estimator: str = ESTIMATOR,
    ) -> None:
        if max_lines < 1 or max_estimated_tokens < 1:
            raise ValueError("Бюджет Prompt 1 должен быть положительным")
        if estimator != self.ESTIMATOR:
            raise ValueError(f"Неизвестный estimator Prompt 1: {estimator}")
        self.max_lines = max_lines
        self.max_estimated_tokens = max_estimated_tokens
        self.estimator = estimator

    @classmethod
    def from_prompt_policy(cls, policy: PromptPolicy) -> DecompositionBudgetPolicy:
        budget = policy.prompts[PromptId.DECOMPOSITION].input_budget
        if budget is None:
            raise ValueError("Prompt 1 не содержит input budget")
        return cls(
            max_lines=budget.max_lines,
            max_estimated_tokens=budget.max_estimated_tokens,
            estimator=budget.estimator,
        )

    def assess(
        self,
        selection: TextSelection,
        *,
        selection_id: str = "<selection-id>",
    ) -> DecompositionBudget:
        payload = decomposition_selection_context(
            selection,
            selection_id=selection_id,
        )
        serialized = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        return DecompositionBudget(
            line_count=len(selection.positions),
            estimated_tokens=max(1, (len(serialized) + 3) // 4),
            max_lines=self.max_lines,
            max_estimated_tokens=self.max_estimated_tokens,
            estimator=self.estimator,
        )

    def require_single_call(
        self,
        selection: TextSelection,
        *,
        selection_id: str = "<selection-id>",
    ) -> DecompositionBudget:
        budget = self.assess(selection, selection_id=selection_id)
        if not budget.within_single_call:
            raise DecompositionBudgetExceededError(budget)
        return budget


def decomposition_selection_context(
    selection: TextSelection,
    *,
    selection_id: str,
) -> dict[str, object]:
    lines = selection.text.split("\n")
    if len(lines) != len(selection.positions):
        raise ValueError("Текст selection не соответствует списку координат")
    return {
        "selection_id": selection_id,
        "start": {
            "page": selection.start.page_index,
            "line": selection.start.line_number,
        },
        "end": {
            "page": selection.end.page_index,
            "line": selection.end.line_number,
        },
        "text": selection.text,
        "lines": [
            {
                "page": position.page_index,
                "line": position.line_number,
                "text": text,
            }
            for position, text in zip(selection.positions, lines, strict=True)
        ],
    }
