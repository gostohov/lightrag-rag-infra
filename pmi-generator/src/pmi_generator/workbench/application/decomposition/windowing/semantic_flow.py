from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace

from ...llm import (
    DecodedToolCall,
    LlmToolRuntime,
    ToolContractError,
)
from ...prompting import PromptCall, PromptId, PromptPolicy
from ...repositories import UnitOfWork
from .canonicalization import SemanticWindowError
from .models import WindowedAttemptState
from .plan import DecompositionWindow, WindowPlan
from .semantic import (
    SemanticWindowArguments,
    SemanticWindowResult,
    semantic_window_context,
)
from .semantic_service import SemanticWindowService


_MAX_REPAIR_PAYLOAD_CHARS = 24_000
_MAX_REPAIR_ERROR_CHARS = 4_000


class SemanticWindowFlow:
    def __init__(
        self,
        *,
        policy: PromptPolicy,
        runtime: LlmToolRuntime,
        service: SemanticWindowService,
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
        generation_attempt_id: str | None = None,
    ) -> SemanticWindowResult:
        try:
            window = next(
                item for item in plan.windows if item.window_id == window_id
            )
        except StopIteration as error:
            raise SemanticWindowError(
                f"Неизвестное окно {window_id}"
            ) from error
        return await self._invoke(
            source_window=window,
            generation_attempt_id=(
                generation_attempt_id or child_attempt_id
            ),
            session_id=session_id,
            validate=lambda arguments: self.service.validate(
                parent=parent,
                plan=plan,
                window_id=window_id,
                child_attempt_id=child_attempt_id,
                arguments=arguments,
            ),
            accept=lambda arguments, raw, uow: self.service.accept(
                parent=parent,
                plan=plan,
                window_id=window_id,
                child_attempt_id=child_attempt_id,
                arguments=arguments,
                raw_arguments=raw,
                uow=uow,
            ),
        )

    async def run_subwindow(
        self,
        *,
        parent: WindowedAttemptState,
        plan: WindowPlan,
        logical_window_id: str,
        logical_child_attempt_id: str,
        source_window: DecompositionWindow,
        generation_attempt_id: str,
        session_id: str,
        accept: Callable[
            [
                SemanticWindowArguments,
                dict[str, object],
                SemanticWindowResult,
                UnitOfWork,
            ],
            SemanticWindowResult,
        ],
    ) -> SemanticWindowResult:
        def validate(arguments: SemanticWindowArguments) -> SemanticWindowResult:
            return self.service.canonicalizer.canonicalize_subwindow(
                parent=parent,
                plan=plan,
                logical_window_id=logical_window_id,
                logical_child_attempt_id=logical_child_attempt_id,
                subwindow=source_window,
                generation_attempt_id=generation_attempt_id,
                arguments=arguments,
            )

        return await self._invoke(
            source_window=source_window,
            generation_attempt_id=generation_attempt_id,
            session_id=session_id,
            validate=validate,
            accept=lambda arguments, raw, uow: accept(
                arguments,
                raw,
                validate(arguments),
                uow,
            ),
        )

    async def _invoke(
        self,
        *,
        source_window: DecompositionWindow,
        generation_attempt_id: str,
        session_id: str,
        validate: Callable[[SemanticWindowArguments], SemanticWindowResult],
        accept: Callable[
            [SemanticWindowArguments, dict[str, object], UnitOfWork],
            SemanticWindowResult,
        ],
    ) -> SemanticWindowResult:
        call = self.policy.build_call(
            PromptId.DECOMPOSITION_WINDOW_SEMANTIC,
            {"window": semantic_window_context(source_window)},
        )

        def validate_result(decoded: DecodedToolCall) -> None:
            if (
                decoded.name != "submit_semantic_window_result"
                or not isinstance(decoded.arguments, SemanticWindowArguments)
            ):
                raise SemanticWindowError(
                    "Runtime вернул неожиданный semantic child result"
                )
            try:
                validate(decoded.arguments)
            except (SemanticWindowError, ValueError) as error:
                raise ToolContractError(str(error)) from error

        decoded = await self.runtime.invoke(
            generation_attempt_id,
            session_id,
            call,
            validate_result=validate_result,
            repair_prompt_builder=_semantic_repair_prompt,
        )
        assert isinstance(decoded.arguments, SemanticWindowArguments)
        return self.runtime.apply_result(
            generation_attempt_id,
            lambda uow: accept(
                decoded.arguments,
                decoded.raw_arguments,
                uow,
            ),
        )


def _semantic_repair_prompt(
    call: PromptCall,
    error: str,
    rejected_tool_calls: tuple[dict[str, object], ...],
) -> PromptCall:
    rejected = (
        rejected_tool_calls[-1].get("arguments", {})
        if rejected_tool_calls
        else {}
    )
    serialized = json.dumps(
        rejected,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    if len(serialized) > _MAX_REPAIR_PAYLOAD_CHARS:
        serialized = (
            serialized[:_MAX_REPAIR_PAYLOAD_CHARS]
            + "...<bounded payload preview>"
        )
    error_detail = error[:_MAX_REPAIR_ERROR_CHARS]
    return replace(
        call,
        system_prompt=(
            f"{call.system_prompt}\n\n"
            "Предыдущий semantic tool call отклонён application validation.\n"
            "Отклонённые аргументы:\n"
            f"{serialized}\n"
            "Все обнаруженные нарушения:\n"
            f"{error_detail}\n"
            "Верни заново ровно один полный submit_semantic_window_result. "
            "Не добавляй технические поля и не возвращай Markdown."
        ),
    )
