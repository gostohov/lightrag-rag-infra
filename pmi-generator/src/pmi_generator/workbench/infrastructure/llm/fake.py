from __future__ import annotations

import asyncio
from typing import Any

from ...application.llm import RawCompletion
from ...application.prompting import PromptCall


class ScriptedLlmTransport:
    def __init__(
        self,
        responses: list[RawCompletion | Exception | tuple[float, RawCompletion | Exception]],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._responses = list(responses)
        self._metadata = dict(metadata or {"model": "scripted"})
        self.calls: list[dict[str, Any]] = []

    async def complete(
        self,
        call: PromptCall,
        tools: list[dict[str, Any]],
    ) -> RawCompletion:
        self.calls.append({"call": call, "tools": tools})
        if not self._responses:
            raise RuntimeError("Для fake LLM не задан следующий ответ")
        scripted = self._responses.pop(0)
        if isinstance(scripted, tuple):
            delay, scripted = scripted
            await asyncio.sleep(delay)
        if isinstance(scripted, Exception):
            raise scripted
        return scripted

    def public_metadata(self) -> dict[str, Any]:
        return dict(self._metadata)
