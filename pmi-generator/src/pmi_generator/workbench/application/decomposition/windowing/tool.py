from __future__ import annotations

from ...llm import ToolSpec
from .candidates import WindowCandidateArguments, WindowCandidateService


def _range_schema() -> dict[str, object]:
    return {
        "type": "object",
        "properties": {
            "page": {"type": "integer", "minimum": 1},
            "line_start": {"type": "integer", "minimum": 1},
            "line_end": {"type": "integer", "minimum": 1},
        },
        "required": ["page", "line_start", "line_end"],
        "additionalProperties": False,
    }


def _candidate_schema() -> dict[str, object]:
    range_schema = _range_schema()
    return {
        "type": "object",
        "properties": {
            "local_candidate_id": {"type": "string", "minLength": 1},
            "title": {"type": "string"},
            "condition": {"type": "string"},
            "changed_factor": {"type": "string"},
            "input_value": {
                "type": ["string", "null"],
                "description": (
                    "Конкретный тестовый вход из source. Если null, gaps обязан "
                    "содержать kind=input_value."
                ),
            },
            "action": {
                "type": ["string", "null"],
                "description": (
                    "Тестовое воздействие из source. Если null, gaps обязан "
                    "содержать kind=action."
                ),
            },
            "condition_ranges": {"type": "array", "items": range_schema},
            "changed_factor_ranges": {
                "type": "array",
                "items": range_schema,
            },
            "input_value_ranges": {"type": "array", "items": range_schema},
            "action_ranges": {"type": "array", "items": range_schema},
            "consequences": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "evidence_ranges": {
                            "type": "array",
                            "items": range_schema,
                        },
                    },
                    "required": ["text", "evidence_ranges"],
                    "additionalProperties": False,
                },
            },
            "gaps": {
                "type": "array",
                "description": (
                    "Blocking gaps каркаса. input_value=null требует "
                    "kind=input_value; action=null требует kind=action."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "kind": {"type": "string"},
                        "question": {"type": "string"},
                        "target_paths": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["kind", "question", "target_paths"],
                    "additionalProperties": False,
                },
            },
        },
        "required": [
            "local_candidate_id",
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


def window_candidates_tool() -> ToolSpec:
    range_schema = _range_schema()
    return ToolSpec(
        name="submit_window_candidates",
        description=(
            "Вернуть локальные candidates и boundary dependencies одного "
            "immutable окна Prompt 1"
        ),
        arguments_type=WindowCandidateArguments,
        json_schema={
            "type": "object",
            "properties": {
                "outcome": {
                    "type": "string",
                    "description": (
                        "candidates: candidates непустой, dependencies допустимы; "
                        "boundary_dependency: candidates пустой, dependencies "
                        "непустой; no_local_testable_behavior: оба массива пусты."
                    ),
                    "enum": [
                        "candidates",
                        "no_local_testable_behavior",
                        "boundary_dependency",
                    ],
                },
                "explanation": {"type": "string", "minLength": 1},
                "candidates": {
                    "type": "array",
                    "description": (
                        "Каждый candidate обязан использовать evidence хотя бы "
                        "одной строки primary=true. Не возвращай самостоятельные "
                        "candidates только по primary=false overlap."
                    ),
                    "maxItems": WindowCandidateService.MAX_CANDIDATES,
                    "items": _candidate_schema(),
                },
                "boundary_dependencies": {
                    "type": "array",
                    "description": (
                        "Используй только если обязательной части поведения нет "
                        "ни в primary, ни в доступном overlap текущего окна. "
                        "source_ranges должны достигать заявленной границы."
                    ),
                    "maxItems": (
                        WindowCandidateService.MAX_BOUNDARY_DEPENDENCIES
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "local_dependency_id": {
                                "type": "string",
                                "minLength": 1,
                            },
                            "local_candidate_id": {
                                "type": ["string", "null"],
                            },
                            "direction": {
                                "type": "string",
                                "enum": ["before", "after"],
                            },
                            "missing_field": {
                                "type": "string",
                                "enum": [
                                    "condition",
                                    "changed_factor",
                                    "input_value",
                                    "action",
                                    "consequence",
                                ],
                            },
                            "source_ranges": {
                                "type": "array",
                                "items": range_schema,
                            },
                            "reason": {"type": "string", "minLength": 1},
                        },
                        "required": [
                            "local_dependency_id",
                            "local_candidate_id",
                            "direction",
                            "missing_field",
                            "source_ranges",
                            "reason",
                        ],
                        "additionalProperties": False,
                    },
                },
            },
            "required": [
                "outcome",
                "explanation",
                "candidates",
                "boundary_dependencies",
            ],
            "additionalProperties": False,
        },
    )
