from __future__ import annotations

from typing import Any

from ...llm import ToolContractError, ToolSpec
from .synthesis import SemanticSynthesisArguments


def _fact_ids(context: dict[str, Any]) -> tuple[str, ...]:
    synthesis = context.get("synthesis")
    if not isinstance(synthesis, dict):
        raise ToolContractError(
            "Semantic synthesis tool требует synthesis context"
        )
    fragments = synthesis.get("target_fragments")
    if not isinstance(fragments, list) or not fragments:
        raise ToolContractError(
            "Semantic synthesis tool требует непустые target_fragments"
        )
    result: list[str] = []
    for fragment in fragments:
        if not isinstance(fragment, dict):
            raise ToolContractError(
                "Semantic synthesis context содержит неверный fragment"
            )
        target_facts = fragment.get("target_facts")
        supporting_facts = fragment.get("supporting_facts")
        if not isinstance(target_facts, list) or not target_facts:
            raise ToolContractError(
                "Semantic synthesis fragment не содержит target_facts"
            )
        if not isinstance(supporting_facts, list):
            raise ToolContractError(
                "Semantic synthesis fragment содержит неверные "
                "supporting_facts"
            )
        for facts in (target_facts, supporting_facts):
            for fact in facts:
                if not isinstance(fact, dict) or not str(
                    fact.get("fact_id", "")
                ).strip():
                    raise ToolContractError(
                        "Semantic synthesis context содержит fact без ID"
                    )
                result.append(str(fact["fact_id"]))
    if len(result) != len(set(result)):
        raise ToolContractError(
            "Semantic synthesis context содержит повторяющиеся fact ID"
        )
    return tuple(result)


def _schema(*, fact_ids: tuple[str, ...] | None = None) -> dict[str, Any]:
    fact_id: dict[str, Any] = {"type": "string"}
    if fact_ids is not None:
        fact_id["enum"] = list(fact_ids)
    slot = {
        "type": "object",
        "properties": {
            "text": {"type": "string", "minLength": 1},
            "fact_ids": {
                "type": "array",
                "minItems": 1,
                "items": fact_id,
            },
        },
        "required": ["text", "fact_ids"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "candidates": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string", "minLength": 1},
                        "condition": slot,
                        "changed_factor": slot,
                        "input_value": slot,
                        "action": slot,
                        "consequences": {
                            "type": "array",
                            "minItems": 1,
                            "items": slot,
                        },
                    },
                    "required": [
                        "title",
                        "condition",
                        "changed_factor",
                        "consequences",
                    ],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["candidates"],
        "additionalProperties": False,
    }


def semantic_synthesis_tool() -> ToolSpec:
    return ToolSpec(
        name="submit_semantic_synthesis",
        description=(
            "Собрать именованные semantic slots каркасов только из "
            "подтверждённых facts; отсутствие optional slot означает пробел"
        ),
        arguments_type=SemanticSynthesisArguments,
        json_schema=_schema(),
        contextual_schema=lambda context: _schema(
            fact_ids=_fact_ids(context)
        ),
    )
