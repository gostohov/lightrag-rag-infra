from __future__ import annotations

from ..llm import ToolSpec
from ..tool_schemas import grounded_field_update_schema, related_gap_schema
from .models import RefinementArguments


def refinement_tool() -> ToolSpec:
    return ToolSpec(
        name="submit_card_refinement",
        description="Атомарно доработать готовую карточку по сообщению аналитика.",
        arguments_type=RefinementArguments,
        json_schema={
            "type": "object",
            "properties": {
                "outcome": {"type": "string", "enum": ["updated", "gaps_created", "no_change"]},
                "updates": {"type": "array", "items": grounded_field_update_schema()},
                "gaps": {"type": "array", "items": related_gap_schema()},
                "reason": {"type": "string", "minLength": 1},
            },
            "required": ["outcome", "updates", "gaps", "reason"],
            "additionalProperties": False,
        },
    )
