from __future__ import annotations

import asyncio
from collections import deque
from typing import Any


class ScriptedClient:
    """Управляемый клиент для тестов application-слоя без внешних сервисов."""

    def __init__(self, responses: list[dict[str, Any] | Exception]) -> None:
        self._responses = deque(responses)
        self._lock = asyncio.Lock()
        self.calls: list[dict[str, Any]] = []

    async def invoke(self, request: dict[str, Any]) -> dict[str, Any]:
        async with self._lock:
            self.calls.append(request)
            if not self._responses:
                raise RuntimeError("Для fake-клиента не задан следующий ответ")
            response = self._responses.popleft()

        if isinstance(response, Exception):
            raise response
        return response
