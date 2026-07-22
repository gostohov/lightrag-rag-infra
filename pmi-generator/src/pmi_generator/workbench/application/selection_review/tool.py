from __future__ import annotations

from ..llm import ToolSpec
from .models import SelectionReviewArguments


def selection_review_tool() -> ToolSpec:
    return ToolSpec(
        name="submit_selection_review",
        description="Завершить независимую проверку выбранного диапазона.",
        arguments_type=SelectionReviewArguments,
        json_schema={
            "type": "object",
            "properties": {
                "outcome": {"type": "string", "enum": ["approved", "issues_found"]},
                "issues": {"type": "array"},
            },
            "required": ["outcome", "issues"],
            "additionalProperties": False,
        },
    )

