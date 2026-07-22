from __future__ import annotations

from ...llm import DecodedToolCall, LlmToolRuntime, ToolContractError
from ...prompting import PromptId, PromptPolicy
from .conflicts import ConflictGroup
from .models import WindowedAttemptState
from .reconciliation import ReconciliationError
from .reconciliation_cases import (
    ReconciliationCase,
    ReconciliationCaseArguments,
    ReconciliationCaseDecision,
    ReconciliationCaseService,
)


class ReconciliationCaseFlow:
    def __init__(
        self,
        *,
        policy: PromptPolicy,
        runtime: LlmToolRuntime,
        service: ReconciliationCaseService,
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
        case: ReconciliationCase,
        attempt_id: str,
        session_id: str,
    ) -> ReconciliationCaseDecision:
        call = self.policy.build_call(
            PromptId.DECOMPOSITION_RECONCILIATION,
            {"case": case.to_context()},
        )

        def validate_result(decoded: DecodedToolCall) -> None:
            if (
                decoded.name != "submit_reconciliation_case"
                or not isinstance(
                    decoded.arguments,
                    ReconciliationCaseArguments,
                )
            ):
                raise ToolContractError(
                    "Runtime вернул неожиданный reconciliation case"
                )
            try:
                self.service.validate_case(
                    parent=parent,
                    plan_hash=plan_hash,
                    group=group,
                    case=case,
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
        assert isinstance(decoded.arguments, ReconciliationCaseArguments)
        return self.runtime.apply_result(
            attempt_id,
            lambda uow: self.service.accept_case(
                parent=parent,
                plan_hash=plan_hash,
                group=group,
                case=case,
                attempt_id=attempt_id,
                arguments=decoded.arguments,
                raw_arguments=decoded.raw_arguments,
                uow=uow,
            ),
        )
