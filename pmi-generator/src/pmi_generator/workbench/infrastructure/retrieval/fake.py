from __future__ import annotations

import asyncio

from ...application.gap_investigation import (
    RetrievalCall,
    RetrievalProfile,
    RetrievalResponse,
    TechnicalRetrievalError,
)


class ScriptedRetrieval:
    def __init__(self, scripted: list[object]) -> None:
        self.scripted = list(scripted)
        self.calls: list[RetrievalCall] = []
        self.delay = 0.0

    async def query(self, question: str, profile: RetrievalProfile) -> RetrievalResponse:
        self.calls.append(RetrievalCall(f"fake-{len(self.calls) + 1}", question, profile))
        if self.delay:
            await asyncio.sleep(self.delay)
        if not self.scripted:
            raise TechnicalRetrievalError("Для fake retrieval не задан ответ")
        item = self.scripted.pop(0)
        if isinstance(item, Exception):
            raise item
        if not isinstance(item, RetrievalResponse):
            raise TechnicalRetrievalError("Fake retrieval получил неверный сценарий")
        return item

