from __future__ import annotations

from typing import Any

from ..domain import GapResolutionMode, GapValueForm
from ..domain.schema import CARD_FIELD_PATHS, REQUIRED_FIELD_PATHS


def card_field_path_schema() -> dict[str, Any]:
    return {
        "type": "string",
        "enum": sorted(CARD_FIELD_PATHS),
    }


def non_required_card_field_path_schema() -> dict[str, Any]:
    return {
        "type": "string",
        "enum": sorted(CARD_FIELD_PATHS - REQUIRED_FIELD_PATHS),
    }


def grounded_field_update_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "path": card_field_path_schema(),
            "value": {},
            "evidence_id": {"type": ["string", "null"]},
            "analyst_message_id": {"type": ["string", "null"]},
        },
        "required": ["path", "value", "evidence_id", "analyst_message_id"],
        "oneOf": [
            {
                "properties": {
                    "evidence_id": {"type": "string", "minLength": 1},
                    "analyst_message_id": {"type": "null"},
                }
            },
            {
                "properties": {
                    "evidence_id": {"type": "null"},
                    "analyst_message_id": {"type": "string", "minLength": 1},
                }
            },
        ],
        "additionalProperties": False,
    }


def related_gap_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "question": {"type": "string", "minLength": 1},
            "blocking_reason": {"type": "string", "minLength": 1},
            "allowed_paths": {
                "type": "array",
                "items": card_field_path_schema(),
                "minItems": 1,
                "uniqueItems": True,
            },
            "dependencies": {
                "type": "array",
                "items": card_field_path_schema(),
                "uniqueItems": True,
            },
            "closure_criterion": {"type": "string", "minLength": 1},
            "resolution_mode": {
                "type": "string",
                "description": (
                    "Режим однородного gap. Если переданы resolution_targets, "
                    "это compatibility-поле не определяет режим отдельных paths."
                ),
                "enum": [mode.value for mode in GapResolutionMode],
            },
            "resolution_targets": {
                "type": "array",
                "description": (
                    "Если allowed_paths требуют разных epistemic origins или "
                    "closure forms, перечислить каждый path ровно один раз. "
                    "Application разделит spec на независимые gaps."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "path": card_field_path_schema(),
                        "resolution_mode": {
                            "type": "string",
                            "enum": [
                                mode.value for mode in GapResolutionMode
                            ],
                        },
                        "accepted_forms": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": [
                                    form.value for form in GapValueForm
                                ],
                            },
                            "minItems": 1,
                            "uniqueItems": True,
                        },
                        "residual_question": {
                            "type": "string",
                            "minLength": 1,
                        },
                    },
                    "required": [
                        "path",
                        "resolution_mode",
                        "accepted_forms",
                        "residual_question",
                    ],
                    "additionalProperties": False,
                },
            },
        },
        "required": [
            "question",
            "blocking_reason",
            "allowed_paths",
            "dependencies",
            "closure_criterion",
            "resolution_mode",
        ],
        "additionalProperties": False,
    }
