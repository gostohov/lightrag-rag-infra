from __future__ import annotations

from typing import Any, Protocol


class AsyncService(Protocol):
    async def invoke(self, request: dict[str, Any]) -> dict[str, Any]: ...
