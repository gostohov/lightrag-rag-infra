from __future__ import annotations

from ..llm import LlmToolRuntime
from ..prompting import PromptId, PromptPolicy
from ..source import SavedSelection
from .budget import decomposition_selection_context
from .models import DecompositionArguments, DecompositionError, DecompositionResult
from .service import DecompositionService


class DecompositionFlow:
    def __init__(
        self,
        *,
        policy: PromptPolicy,
        runtime: LlmToolRuntime,
        service: DecompositionService,
    ) -> None:
        self.policy = policy
        self.runtime = runtime
        self.service = service

    async def run(
        self,
        *,
        attempt_id: str,
        session_id: str,
        selection: SavedSelection,
        existing_card_summaries: list[dict[str, object]] | None = None,
    ) -> DecompositionResult:
        call = self.policy.build_call(
            PromptId.DECOMPOSITION,
            {
                "selection": decomposition_selection_context(
                    selection.selection,
                    selection_id=selection.selection_id,
                ),
                "existing_card_summaries": existing_card_summaries or [],
            },
        )
        result = await self.runtime.invoke(attempt_id, session_id, call)
        if result.name != "submit_decomposition" or not isinstance(
            result.arguments, DecompositionArguments
        ):
            raise DecompositionError("Runtime вернул неожиданный результат Промпта 1")
        return self.runtime.apply_result(
            attempt_id,
            lambda uow: self.service.apply(selection, result.arguments, uow=uow),
        )
