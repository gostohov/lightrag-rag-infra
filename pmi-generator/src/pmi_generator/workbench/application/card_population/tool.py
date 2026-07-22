from __future__ import annotations

from ..llm import ToolSpec
from ..tool_schemas import (
    card_field_path_schema,
    non_required_card_field_path_schema,
    related_gap_schema,
)
from .models import PopulationArguments


def population_tool() -> ToolSpec:
    return ToolSpec(
        name="submit_card_population",
        description="Атомарно заполнить одну выбранную карточку",
        arguments_type=PopulationArguments,
        json_schema={
            "type": "object",
            "properties": {
                "source_values": {
                    "type": "array",
                    "description": (
                        "Только значения, прямо подтверждённые source evidence. "
                        "Выведенные и экспертные значения сюда не помещать."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "path": card_field_path_schema(),
                            "value": {},
                            "evidence_id": {"type": "string", "minLength": 1},
                        },
                        "required": ["path", "value", "evidence_id"],
                        "additionalProperties": False,
                    },
                },
                "derivations": {
                    "type": "array",
                    "description": (
                        "Только выведенные значения с source_evidence_ids, "
                        "явным rule и scope."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "path": card_field_path_schema(),
                            "value": {},
                            "source_evidence_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                                "minItems": 1,
                                "uniqueItems": True,
                            },
                            "rule": {"type": "string", "minLength": 1},
                            "scope": {"type": "string", "minLength": 1},
                        },
                        "required": [
                            "path",
                            "value",
                            "source_evidence_ids",
                            "rule",
                            "scope",
                        ],
                        "additionalProperties": False,
                    },
                },
                "not_applicable": {
                    "type": "array",
                    "description": (
                        "Только поля, которые действительно не относятся к типу "
                        "проверки; отсутствие значения или описания в источнике "
                        "не означает неприменимость."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "path": non_required_card_field_path_schema(),
                            "reason": {"type": "string", "minLength": 1},
                        },
                        "required": ["path", "reason"],
                        "additionalProperties": False,
                    },
                },
                "gaps": {
                    "type": "array",
                    "description": (
                        "Блокирующие пробелы, когда для поля не хватает основания "
                        "или конкретного значения."
                    ),
                    "items": related_gap_schema(),
                },
            },
            "required": [
                "source_values",
                "derivations",
                "not_applicable",
                "gaps",
            ],
            "additionalProperties": False,
        },
    )
