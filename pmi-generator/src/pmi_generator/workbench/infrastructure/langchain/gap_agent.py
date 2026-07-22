from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Sequence
from typing import Annotated, Any

from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool, InjectedToolCallId, StructuredTool
from pydantic import ConfigDict, PrivateAttr

from ...application.gap_investigation import (
    GapAgentStepLimitError,
    GapArguments,
    submit_gap_result_tool,
)
from ...application.llm import DecodedToolCall, LlmToolRuntime, TechnicalLlmError
from ...application.prompting import PromptCall


class _RuntimeChatModel(BaseChatModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    runtime: LlmToolRuntime
    session_id: str
    attempt_id: str
    call_factory: Callable[[], PromptCall]
    child_started: Callable[[str], None]
    child_finished: Callable[[], None]
    validate_result: Callable[[DecodedToolCall], None]
    max_steps: int
    _calls: int = PrivateAttr(default=0)

    @property
    def _llm_type(self) -> str:
        return "pmi-typed-tool-runtime"

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        raise RuntimeError("Prompt 3 agent поддерживает только async execution")

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self._calls >= self.max_steps:
            raise GapAgentStepLimitError("Достигнут технический предел Prompt 3")
        self._calls += 1
        child_id = f"{self.attempt_id}-LLM-{self._calls:03d}"
        self.child_started(child_id)
        try:
            try:
                decoded = await self.runtime.invoke(
                    child_id,
                    f"{self.session_id}:{self.attempt_id}",
                    self.call_factory(),
                    validate_result=self.validate_result,
                )
            except TechnicalLlmError as error:
                if getattr(error, "finish_reason", None) == "length":
                    raise GapAgentStepLimitError(
                        "Prompt 3 достиг лимита выходных токенов"
                    ) from error
                raise
            decoded = self.runtime.apply_result(child_id, lambda _uow: decoded)
        finally:
            self.child_finished()
        message = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": decoded.name,
                    "args": decoded.raw_arguments,
                    "id": decoded.call_id,
                    "type": "tool_call",
                }
            ],
        )
        return ChatResult(generations=[ChatGeneration(message=message)])


class LangChainGapAgent:
    def __init__(self, runtime: LlmToolRuntime) -> None:
        self.runtime = runtime

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
    ) -> object:
        submitted: dict[str, object] = {}

        async def ask(
            question: str,
            tool_call_id: Annotated[str, InjectedToolCallId],
        ) -> str:
            observation = await ask_lightrag(tool_call_id, question)
            return json.dumps(observation, ensure_ascii=False, default=str)

        async def expand(
            call_id: str,
            reason: str,
            tool_call_id: Annotated[str, InjectedToolCallId],
        ) -> str:
            observation = await expand_lightrag(tool_call_id, call_id, reason)
            return json.dumps(observation, ensure_ascii=False, default=str)

        async def submit(
            outcome: str,
            updates: list[dict[str, object]],
            unknown_fields: list[str],
            missing_fact: object,
            summary: str,
            contradictions: list[dict[str, object]],
        ) -> str:
            result = submit_result(
                GapArguments(
                    outcome=outcome,
                    updates=updates,
                    unknown_fields=unknown_fields,
                    missing_fact=missing_fact,
                    summary=summary,
                    contradictions=contradictions,
                )
            )
            submitted["result"] = result
            return "Результат исследования атомарно принят"

        submit_spec = submit_gap_result_tool()

        def validate_decoded(decoded: DecodedToolCall) -> None:
            if decoded.name == submit_spec.name:
                validate_result(decoded.arguments)

        tools = (
            StructuredTool.from_function(
                coroutine=ask,
                name="ask_lightrag",
                description="Найти один конкретный факт в LightRAG узким поиском.",
            ),
            StructuredTool.from_function(
                coroutine=expand,
                name="expand_lightrag",
                description="Повторить существующий вопрос расширенным поиском.",
            ),
            StructuredTool.from_function(
                coroutine=submit,
                name=submit_spec.name,
                description=submit_spec.description,
                return_direct=True,
                args_schema=submit_spec.json_schema,
                infer_schema=False,
            ),
        )
        initial_call = call_factory()
        model = _RuntimeChatModel(
            runtime=self.runtime,
            session_id=session_id,
            attempt_id=attempt_id,
            call_factory=call_factory,
            child_started=child_started,
            child_finished=child_finished,
            validate_result=validate_decoded,
            max_steps=max_steps,
        )
        agent = create_agent(
            model=model,
            tools=tools,
            system_prompt=initial_call.system_prompt,
            name="pmi-gap-worker",
        )
        await agent.ainvoke(
            {
                "messages": [
                    HumanMessage(
                        content=json.dumps(
                            initial_call.context,
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                    )
                ]
            },
            config={"recursion_limit": max(8, max_steps * 3 + 2)},
        )
        if "result" not in submitted:
            raise GapAgentStepLimitError("Prompt 3 завершился без submit_gap_result")
        return submitted["result"]


__all__ = ["LangChainGapAgent"]
