from __future__ import annotations

from ..llm import ToolSpec
from .models import DecompositionArguments


def _line_range_schema() -> dict[str, object]:
    return {
        "type": "object",
        "properties": {
            "page": {"type": "integer"},
            "line_start": {"type": "integer"},
            "line_end": {"type": "integer"},
        },
        "required": ["page", "line_start", "line_end"],
        "additionalProperties": False,
    }


def decomposition_tool() -> ToolSpec:
    range_schema = _line_range_schema()
    skeleton_schema = {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "condition": {"type": "string"},
            "changed_factor": {"type": "string"},
            "input_value": {"type": ["string", "null"]},
            "action": {"type": ["string", "null"]},
            "condition_ranges": {"type": "array", "items": range_schema},
            "changed_factor_ranges": {"type": "array", "items": range_schema},
            "input_value_ranges": {"type": "array", "items": range_schema},
            "action_ranges": {"type": "array", "items": range_schema},
            "consequences": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "evidence_ranges": {"type": "array", "items": range_schema},
                    },
                    "required": ["text", "evidence_ranges"],
                    "additionalProperties": False,
                },
            },
            "gaps": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "kind": {"type": "string"},
                        "question": {"type": "string"},
                        "target_paths": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["kind", "question", "target_paths"],
                    "additionalProperties": False,
                },
            },
        },
        "required": [
            "title",
            "condition",
            "changed_factor",
            "input_value",
            "action",
            "condition_ranges",
            "changed_factor_ranges",
            "input_value_ranges",
            "action_ranges",
            "consequences",
            "gaps",
        ],
        "additionalProperties": False,
    }
    return ToolSpec(
        name="submit_decomposition",
        description="Атомарно вернуть результат декомпозиции выбранного диапазона",
        arguments_type=DecompositionArguments,
        json_schema={
            "type": "object",
            "properties": {
                "outcome": {
                    "type": "string",
                    "enum": [
                        "skeletons_created",
                        "no_testable_behavior",
                        "insufficient_selection",
                    ],
                },
                "explanation": {"type": "string"},
                "skeletons": {"type": "array", "items": skeleton_schema},
                "line_assessments": {
                    "type": "array",
                    "description": (
                        "Ровно одна запись для каждой строки selection. Evidence "
                        "допустим только для строки, использованной координатами каркаса."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "page": {"type": "integer"},
                            "line": {"type": "integer"},
                            "role": {
                                "type": "string",
                                "enum": ["evidence", "context"],
                            },
                            "reason": {"type": "string", "minLength": 1},
                        },
                        "required": ["page", "line", "role", "reason"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": [
                "outcome",
                "explanation",
                "skeletons",
                "line_assessments",
            ],
            "additionalProperties": False,
        },
    )
