from __future__ import annotations

from ...llm import DecodedToolCall, LlmToolRuntime, ToolContractError
from ...prompting import PromptId, PromptPolicy
from .candidates import (
    WindowCandidateArguments,
    WindowCandidateError,
    WindowCandidateResult,
    WindowCandidateService,
    window_context,
)
from .models import WindowedAttemptState
from .plan import WindowPlan


class WindowCandidateFlow:
    def __init__(
        self,
        *,
        policy: PromptPolicy,
        runtime: LlmToolRuntime,
        service: WindowCandidateService,
    ) -> None:
        self.policy = policy
        self.runtime = runtime
        self.service = service

    async def run(
        self,
        *,
        parent: WindowedAttemptState,
        plan: WindowPlan,
        window_id: str,
        child_attempt_id: str,
        session_id: str,
    ) -> WindowCandidateResult:
        try:
            window = next(
                item for item in plan.windows
                if item.window_id == window_id
            )
        except StopIteration as error:
            raise WindowCandidateError(
                f"Неизвестное окно {window_id}"
            ) from error
        call = self.policy.build_call(
            PromptId.DECOMPOSITION_WINDOW,
            {"window": window_context(window)},
        )

        def validate_result(decoded: DecodedToolCall) -> None:
            if (
                decoded.name != "submit_window_candidates"
                or not isinstance(
                    decoded.arguments,
                    WindowCandidateArguments,
                )
            ):
                raise WindowCandidateError(
                    "Runtime вернул неожиданный child result Prompt 1"
                )
            try:
                self.service.validate(
                    parent=parent,
                    plan=plan,
                    window_id=window_id,
                    child_attempt_id=child_attempt_id,
                    arguments=decoded.arguments,
                )
            except (WindowCandidateError, ValueError) as error:
                raise ToolContractError(str(error)) from error

        decoded = await self.runtime.invoke(
            child_attempt_id,
            session_id,
            call,
            validate_result=validate_result,
        )
        assert isinstance(decoded.arguments, WindowCandidateArguments)
        return self.runtime.apply_result(
            child_attempt_id,
            lambda uow: self.service.accept(
                parent=parent,
                plan=plan,
                window_id=window_id,
                child_attempt_id=child_attempt_id,
                arguments=decoded.arguments,
                raw_arguments=decoded.raw_arguments,
                uow=uow,
            ),
        )
