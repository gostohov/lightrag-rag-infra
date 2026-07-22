from __future__ import annotations

from copy import deepcopy

from ...llm import ToolSpec
from .reconciliation import ReconciliationArguments


def _schema() -> dict[str, object]:
    return {
        "type": "object",
        "properties": {
            "outcome": {
                "type": "string",
                "enum": ["resolved", "unresolved"],
            },
            "accepted_candidate_ids": {
                "type": "array",
                "items": {"type": "string"},
            },
            "rejected_candidate_ids": {
                "type": "array",
                "items": {"type": "string"},
            },
            "resolved_dependency_ids": {
                "type": "array",
                "items": {"type": "string"},
            },
            "relations": {
                "type": "array",
                "maxItems": 64,
                "items": {
                    "type": "object",
                    "properties": {
                        "kind": {
                            "type": "string",
                            "enum": ["merge", "split"],
                        },
                        "candidate_ids": {
                            "type": "array",
                            "minItems": 2,
                            "items": {"type": "string"},
                        },
                        "reason": {"type": "string", "minLength": 1},
                    },
                    "required": [
                        "kind",
                        "candidate_ids",
                        "reason",
                    ],
                    "additionalProperties": False,
                },
            },
            "explanation": {"type": "string", "minLength": 1},
        },
        "required": [
            "outcome",
            "accepted_candidate_ids",
            "rejected_candidate_ids",
            "resolved_dependency_ids",
            "relations",
            "explanation",
        ],
        "additionalProperties": False,
    }


def _contextual_schema(context: dict[str, object]) -> dict[str, object]:
    schema = deepcopy(_schema())
    group = context.get("group")
    if not isinstance(group, dict):
        raise ValueError("Reconciliation context не содержит group")
    candidate_ids = group.get("candidate_ids")
    dependency_ids = group.get("dependency_ids")
    if not isinstance(candidate_ids, list) or not isinstance(
        dependency_ids,
        list,
    ):
        raise ValueError(
            "Reconciliation group не содержит typed IDs"
        )
    properties = schema["properties"]
    assert isinstance(properties, dict)
    for name in ("accepted_candidate_ids", "rejected_candidate_ids"):
        property_schema = properties[name]
        assert isinstance(property_schema, dict)
        property_schema["items"] = {
            "type": "string",
            "enum": [str(item) for item in candidate_ids],
        }
    dependencies_schema = properties["resolved_dependency_ids"]
    assert isinstance(dependencies_schema, dict)
    dependencies_schema["items"] = {
        "type": "string",
        "enum": [str(item) for item in dependency_ids],
    }
    if len(candidate_ids) < 2:
        properties["relations"] = {
            "type": "array",
            "enum": [[]],
        }
        return schema
    relations = properties["relations"]
    assert isinstance(relations, dict)
    relation_items = relations["items"]
    assert isinstance(relation_items, dict)
    relation_properties = relation_items["properties"]
    assert isinstance(relation_properties, dict)
    relation_candidates = relation_properties["candidate_ids"]
    assert isinstance(relation_candidates, dict)
    relation_candidates["items"] = {
        "type": "string",
        "enum": [str(item) for item in candidate_ids],
    }
    return schema


def reconciliation_tool() -> ToolSpec:
    return ToolSpec(
        name="submit_reconciliation",
        description=(
            "Разрешить один bounded conflict group только по известным "
            "candidate/dependency IDs и source coordinates"
        ),
        arguments_type=ReconciliationArguments,
        json_schema=_schema(),
        contextual_schema=_contextual_schema,
    )
