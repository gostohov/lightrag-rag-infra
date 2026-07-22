from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class RawCompletion:
    finish_reason: str
    tool_calls: tuple[dict[str, Any], ...]
    usage: dict[str, Any]
    model: str
    content: str = ""
    response_preview: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class DecodedToolCall:
    call_id: str
    name: str
    arguments: object
    raw_arguments: dict[str, Any]
