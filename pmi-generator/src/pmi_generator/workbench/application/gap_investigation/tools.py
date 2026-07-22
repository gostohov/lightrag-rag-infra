from __future__ import annotations

from copy import deepcopy
from typing import Any

from ..llm import ToolSpec
from ..tool_schemas import card_field_path_schema, grounded_field_update_schema
from .models import AskLightRagArguments, ExpandLightRagArguments, GapArguments


def ask_lightrag_tool() -> ToolSpec:
    return ToolSpec(
        name="ask_lightrag",
        description="Задать LightRAG один короткий вопрос об одном факте.",
        arguments_type=AskLightRagArguments,
        json_schema={
            "type": "object",
            "properties": {"question": {"type": "string"}},
            "required": ["question"],
            "additionalProperties": False,
        },
    )


def expand_lightrag_tool() -> ToolSpec:
    return ToolSpec(
        name="expand_lightrag",
        description="Повторить прежний вопрос с расширенным retrieval-профилем.",
        arguments_type=ExpandLightRagArguments,
        json_schema={
            "type": "object",
            "properties": {
                "call_id": {"type": "string"},
                "reason": {"type": "string"},
            },
            "required": ["call_id", "reason"],
            "additionalProperties": False,
        },
    )


def _submit_gap_result_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "outcome": {"type": "string", "enum": ["resolved", "not_found", "contradiction"]},
            "updates": {"type": "array", "items": grounded_field_update_schema()},
            "unknown_fields": {
                "type": "array",
                "items": card_field_path_schema(),
                "uniqueItems": True,
            },
            "missing_fact": {
                "type": ["string", "object", "null"],
                "properties": {
                    "field": card_field_path_schema(),
                    "description": {"type": "string", "minLength": 1},
                },
                "required": ["field", "description"],
                "additionalProperties": False,
            },
            "summary": {"type": "string"},
            "contradictions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "statement": {"type": "string", "minLength": 1},
                        "evidence_id": {"type": "string", "minLength": 1},
                    },
                    "required": ["statement", "evidence_id"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["outcome", "updates", "unknown_fields", "missing_fact", "summary", "contradictions"],
        "additionalProperties": False,
    }


def _contextual_submit_gap_result_schema(
    context: dict[str, Any],
) -> dict[str, Any]:
    schema = deepcopy(_submit_gap_result_schema())
    properties = schema["properties"]
    gap = context.get("gap")
    allowed_paths = (
        gap.get("allowed_paths", [])
        if isinstance(gap, dict)
        else []
    )
    valid_paths = sorted(
        {
            str(path)
            for path in allowed_paths
            if str(path) in card_field_path_schema()["enum"]
        }
    )
    context_evidence_ids = {
        str(item["evidence_id"])
        for item in context.get("evidence", [])
        if isinstance(item, dict) and item.get("evidence_id")
    }
    observation_evidence_ids = {
        str(evidence_id)
        for observation in context.get("observations", [])
        if isinstance(observation, dict)
        for evidence_id in observation.get("evidence_ids", [])
        if evidence_id
    }
    evidence_ids = sorted(
        {
            evidence_id
            for evidence_id in observation_evidence_ids
            if evidence_id in context_evidence_ids
        }
    )
    analyst_message_ids = sorted(
        {
            str(item["message_id"])
            for item in context.get("analyst_messages", [])
            if isinstance(item, dict) and item.get("message_id")
        }
    )

    if analyst_message_ids:
        outcomes = ["resolved", "not_found"]
    elif evidence_ids:
        outcomes = ["resolved", "not_found", "contradiction"]
    else:
        outcomes = ["not_found"]
    properties["outcome"]["enum"] = outcomes

    update_properties = properties["updates"]["items"]["properties"]
    if valid_paths:
        update_properties["path"] = {"type": "string", "enum": valid_paths}
        properties["unknown_fields"]["items"] = {
            "type": "string",
            "enum": valid_paths,
        }
        properties["missing_fact"]["properties"]["field"] = {
            "type": "string",
            "enum": valid_paths,
        }
    if analyst_message_ids:
        update_properties["evidence_id"] = {"type": "null"}
        update_properties["analyst_message_id"] = {
            "type": "string",
            "enum": analyst_message_ids,
        }
    elif evidence_ids:
        update_properties["evidence_id"] = {
            "type": "string",
            "enum": evidence_ids,
        }
        update_properties["analyst_message_id"] = {"type": "null"}
    else:
        update_properties["evidence_id"] = {"type": "null"}
        update_properties["analyst_message_id"] = {"type": "null"}
    if evidence_ids:
        properties["contradictions"]["items"]["properties"]["evidence_id"] = {
            "type": "string",
            "enum": evidence_ids,
        }
    return schema


def submit_gap_result_tool() -> ToolSpec:
    return ToolSpec(
        name="submit_gap_result",
        description="Завершить исследование одного пробела.",
        arguments_type=GapArguments,
        json_schema=_submit_gap_result_schema(),
        contextual_schema=_contextual_submit_gap_result_schema,
    )
