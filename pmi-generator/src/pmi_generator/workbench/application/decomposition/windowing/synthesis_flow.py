from __future__ import annotations

from ...llm import DecodedToolCall, LlmToolRuntime, ToolContractError
from ...prompting import PromptId, PromptPolicy
from .canonicalization import SemanticWindowError
from .candidates import WindowCandidateResult
from .models import WindowedAttemptState
from .plan import WindowPlan
from .semantic import SemanticWindowResult
from .synthesis import (
    SemanticSynthesisArguments,
    semantic_synthesis_context,
)
from .synthesis_service import SemanticSynthesisService


class SemanticSynthesisFlow:
    def __init__(
        self,
        *,
        policy: PromptPolicy,
        runtime: LlmToolRuntime,
        service: SemanticSynthesisService,
    ) -> None:
        self.policy = policy
        self.runtime = runtime
        self.service = service

    async def run(
        self,
        *,
        parent: WindowedAttemptState,
        plan: WindowPlan,
        target_window_id: str,
        attempt_id: str,
        fact_results: tuple[SemanticWindowResult, ...],
        session_id: str,
    ) -> WindowCandidateResult:
        call = self.policy.build_call(
            PromptId.DECOMPOSITION_SEMANTIC_SYNTHESIS,
            {
                "synthesis": semantic_synthesis_context(
                    document=self.service.canonicalizer.document,
                    plan=plan,
                    target_window_id=target_window_id,
                    results=fact_results,
                )
            },
        )

        def validate_result(decoded: DecodedToolCall) -> None:
            if (
                decoded.name != "submit_semantic_synthesis"
                or not isinstance(
                    decoded.arguments,
                    SemanticSynthesisArguments,
                )
            ):
                raise ToolContractError(
                    "Runtime вернул неожиданный semantic synthesis"
                )
            try:
                self.service.validate(
                    parent=parent,
                    plan=plan,
                    target_window_id=target_window_id,
                    attempt_id=attempt_id,
                    fact_results=fact_results,
                    arguments=decoded.arguments,
                )
            except (SemanticWindowError, ValueError) as error:
                raise ToolContractError(str(error)) from error

        decoded = await self.runtime.invoke(
            attempt_id,
            session_id,
            call,
            validate_result=validate_result,
        )
        assert isinstance(decoded.arguments, SemanticSynthesisArguments)
        return self.runtime.apply_result(
            attempt_id,
            lambda uow: self.service.accept(
                parent=parent,
                plan=plan,
                target_window_id=target_window_id,
                attempt_id=attempt_id,
                fact_results=fact_results,
                arguments=decoded.arguments,
                raw_arguments=decoded.raw_arguments,
                uow=uow,
            ),
        )
