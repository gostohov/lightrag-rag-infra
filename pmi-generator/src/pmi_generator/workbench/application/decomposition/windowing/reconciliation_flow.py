from __future__ import annotations

from ...llm import DecodedToolCall, LlmToolRuntime, ToolContractError
from ...prompting import PromptId, PromptPolicy
from .conflicts import ConflictGroup
from .models import WindowedAttemptState
from .reconciliation import (
    ReconciliationArguments,
    ReconciliationDecision,
    ReconciliationError,
    ReconciliationService,
    reconciliation_context,
)


class ReconciliationFlow:
    def __init__(
        self,
        *,
        policy: PromptPolicy,
        runtime: LlmToolRuntime,
        service: ReconciliationService,
    ) -> None:
        self.policy = policy
        self.runtime = runtime
        self.service = service

    async def run(
        self,
        *,
        parent: WindowedAttemptState,
        plan_hash: str,
        group: ConflictGroup,
        attempt_id: str,
        session_id: str,
    ) -> ReconciliationDecision:
        call = self.policy.build_call(
            PromptId.DECOMPOSITION_RECONCILIATION,
            {"group": reconciliation_context(group)},
        )

        def validate_result(decoded: DecodedToolCall) -> None:
            if (
                decoded.name != "submit_reconciliation"
                or not isinstance(
                    decoded.arguments,
                    ReconciliationArguments,
                )
            ):
                raise ToolContractError(
                    "Runtime вернул неожиданный reconciliation result"
                )
            try:
                self.service.validate(
                    parent=parent,
                    plan_hash=plan_hash,
                    group=group,
                    attempt_id=attempt_id,
                    arguments=decoded.arguments,
                )
            except (ReconciliationError, ValueError) as error:
                raise ToolContractError(str(error)) from error

        decoded = await self.runtime.invoke(
            attempt_id,
            session_id,
            call,
            validate_result=validate_result,
        )
        assert isinstance(decoded.arguments, ReconciliationArguments)
        return self.runtime.apply_result(
            attempt_id,
            lambda uow: self.service.accept(
                parent=parent,
                plan_hash=plan_hash,
                group=group,
                attempt_id=attempt_id,
                arguments=decoded.arguments,
                raw_arguments=decoded.raw_arguments,
                uow=uow,
            ),
        )
