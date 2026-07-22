from __future__ import annotations

from typing import Any

from ...llm import ToolContractError, ToolSpec
from .semantic import SemanticWindowArguments


def _line_ids(context: dict[str, Any]) -> tuple[str, ...]:
    window = context.get("window")
    if not isinstance(window, dict):
        raise ToolContractError(
            "Semantic window tool требует window context"
        )
    lines = window.get("lines")
    if not isinstance(lines, list) or not lines:
        raise ToolContractError(
            "Semantic window tool требует непустые source lines"
        )
    result: list[str] = []
    for line in lines:
        if not isinstance(line, dict) or not str(line.get("line_id", "")).strip():
            raise ToolContractError(
                "Semantic window context содержит строку без line_id"
            )
        result.append(str(line["line_id"]))
    if len(result) != len(set(result)):
        raise ToolContractError(
            "Semantic window context содержит повторяющиеся line_id"
        )
    return tuple(result)


def _schema(*, line_ids: tuple[str, ...] | None = None) -> dict[str, Any]:
    line_id_schema: dict[str, Any] = {"type": "string"}
    if line_ids is not None:
        line_id_schema["enum"] = list(line_ids)
    line_ids_schema = {
        "type": "array",
        "items": line_id_schema,
        "minItems": 1,
    }
    fact_schema = {
        "type": "object",
        "properties": {
            "text": {"type": "string", "minLength": 1},
            "line_ids": line_ids_schema,
        },
        "required": ["text", "line_ids"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "behaviors": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string", "minLength": 1},
                        "summary": {"type": "string", "minLength": 1},
                        "facts": {
                            "type": "array",
                            "minItems": 1,
                            "items": fact_schema,
                        },
                    },
                    "required": [
                        "title",
                        "summary",
                        "facts",
                    ],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["behaviors"],
        "additionalProperties": False,
    }


def semantic_window_tool() -> ToolSpec:
    return ToolSpec(
        name="submit_semantic_window_result",
        description=(
            "Вернуть только смысловые фрагменты, атомарные факты и line_id "
            "из immutable окна"
        ),
        arguments_type=SemanticWindowArguments,
        json_schema=_schema(),
        contextual_schema=lambda context: _schema(
            line_ids=_line_ids(context)
        ),
    )
