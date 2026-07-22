from __future__ import annotations

from typing import Any, Protocol

from ..prompting import PromptCall
from .models import RawCompletion


class LlmTransport(Protocol):
    async def complete(
        self,
        call: PromptCall,
        tools: list[dict[str, Any]],
    ) -> RawCompletion: ...

    def public_metadata(self) -> dict[str, Any]: ...
