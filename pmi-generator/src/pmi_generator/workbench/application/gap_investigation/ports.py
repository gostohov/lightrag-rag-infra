from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Protocol

from ..prompting import PromptCall
from .models import GapArguments, RetrievalProfile, RetrievalResponse


class GapAgentStepLimitError(RuntimeError):
    pass


class RetrievalPort(Protocol):
    async def query(self, question: str, profile: RetrievalProfile) -> RetrievalResponse: ...


class GapAgentPort(Protocol):
    async def run(
        self,
        *,
        attempt_id: str,
        session_id: str,
        call_factory: Callable[[], PromptCall],
        ask_lightrag: Callable[[str, str], Awaitable[object]],
        expand_lightrag: Callable[[str, str, str], Awaitable[object]],
        validate_result: Callable[[GapArguments], None],
        submit_result: Callable[[GapArguments], object],
        child_started: Callable[[str], None],
        child_finished: Callable[[], None],
        max_steps: int,
    ) -> object: ...
