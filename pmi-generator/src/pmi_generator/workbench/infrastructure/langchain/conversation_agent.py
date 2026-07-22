from __future__ import annotations

import json
import asyncio
from collections.abc import Callable, Sequence
from functools import reduce
from operator import or_
from typing import Any, Literal

from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool, StructuredTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.checkpoint.base import BaseCheckpointSaver
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    ValidationError,
    create_model,
)

from ...application.conversation import (
    ConversationAction,
    ConversationAgentError,
    ConversationContext,
    ConversationToolCall,
    ConversationTurnDecision,
    ConversationTurnKind,
    user_facing_conversation_text,
)
from ...application.llm import LlmTransport
from ...application.prompting import PromptId, default_policy


_SYSTEM_PROMPT = """\
Ты единый conversation agent PMI Workbench. На каждом ходе выбери ровно один
terminal tool. respond_to_analyst используется для объяснения текущего
состояния без side effects. request_clarification используется при
неоднозначном намерении. Остальные доступные tools являются типизированными
application actions: передавай только аргументы из их schema. Системные
идентификаторы и revision добавляет код. Не объявляй собственный текст evidence
и не обходи application guards. Во всех отображаемых ответах и announcement
используй естественные русские названия действий из
available_action_labels, никогда не показывай action ID или имя terminal tool.
submit_analyst_answer только формирует показываемую pending-интерпретацию и не
изменяет карточку. confirm_analyst_answer выбирай только когда пользователь
явно подтверждает текущую pending-интерпретацию; вопрос, предположение,
инструкция или молчаливый resume подтверждением не являются.
Значение submit_analyst_answer обязано точно соответствовать
open_gap.closure_requirements и schema tool: exact, finite_set и
deterministic_rule передаются явной tagged-формой, а не обычной строкой.
Если ответ можно сохранить, но он не удовлетворяет требуемой конкретности,
используй tagged-форму confirmed_value; не усиливай смысл ответа аналитика.
leave_gap означает только явное решение аналитика оставить вопрос нерешённым.
Никогда не используй leave_gap для закрытия пробела: достаточная подтверждённая
интерпретация закрывается application автоматически.
refine_card также только строит показываемое предложение доработки. Оно не
изменяет карточку до отдельного явного подтверждения.
Workbench имеет typed доступ к LightRAG через действие исследования пробела,
когда оно присутствует в available_actions. Никогда не утверждай, что LightRAG
технически недоступен Workbench. Для design_decision поиск не выбирает
проектное значение: используй доступное действие обсуждения проектного решения
и объясни ограничение именно текущего gap, не отрицая наличие LightRAG.
"""
_POLICY = default_policy()


class _ToolArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    announcement: str


class _ResearchGapArguments(_ToolArguments):
    question: str


class _AnalystValue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1)
    value: Any


_NonNullJsonValue = (
    str | int | float | bool | list[Any] | dict[str, Any]
)


class _ConfirmedAnalystValue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["confirmed_value"]
    value: _NonNullJsonValue


class _ExactAnalystValue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["exact"]
    value: _NonNullJsonValue


class _FiniteSetAnalystValue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["finite_set"]
    values: list[_NonNullJsonValue] = Field(min_length=1)


class _DeterministicRuleAnalystValue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["deterministic_rule"]
    rule: str = Field(min_length=1)
    parameters: dict[str, Any]


class _AnalystAnswerArguments(_ToolArguments):
    values: list[_AnalystValue] = Field(min_length=1)


class _ChangeGapModeArguments(_ToolArguments):
    resolution_mode: Literal[
        "source_fact",
        "design_decision",
        "external_input",
    ]


class _LeaveGapArguments(_ToolArguments):
    decision: Literal["leave_open"]
    reason: str


_ACTION_ARGUMENTS: dict[ConversationAction, type[BaseModel]] = {
    ConversationAction.RESUME: _ToolArguments,
    ConversationAction.RESEARCH_GAP: _ResearchGapArguments,
    ConversationAction.SUBMIT_ANALYST_ANSWER: _AnalystAnswerArguments,
    ConversationAction.CONFIRM_ANALYST_ANSWER: _ToolArguments,
    ConversationAction.REJECT_ANALYST_ANSWER: _ToolArguments,
    ConversationAction.PROPOSE_DESIGN_DECISION: _ToolArguments,
    ConversationAction.CHANGE_GAP_MODE: _ChangeGapModeArguments,
    ConversationAction.LEAVE_GAP: _LeaveGapArguments,
    ConversationAction.REFINE_CARD: _ToolArguments,
    ConversationAction.INCLUDE_CARD: _ToolArguments,
    ConversationAction.EXCLUDE_CARD: _ToolArguments,
    ConversationAction.EXPORT_DIAGNOSTICS: _ToolArguments,
    ConversationAction.EXPORT_PMI: _ToolArguments,
}

_ACTION_DESCRIPTIONS = {
    ConversationAction.RESUME: "Продолжить сохранённую стадию workflow.",
    ConversationAction.RESEARCH_GAP: (
        "Выполнить реальный LightRAG research для текущего source_fact gap "
        "с новым предметным вопросом. Фактический вопрос передаётся "
        "application transport без подмены."
    ),
    ConversationAction.SUBMIT_ANALYST_ANSWER: (
        "Предложить точное отображение ответа аналитика на поля текущего gap."
    ),
    ConversationAction.CONFIRM_ANALYST_ANSWER: (
        "Применить показанную pending-интерпретацию после явного подтверждения."
    ),
    ConversationAction.REJECT_ANALYST_ANSWER: (
        "Отклонить показанную pending-интерпретацию без изменения карточки."
    ),
    ConversationAction.PROPOSE_DESIGN_DECISION: (
        "Объяснить, что LightRAG доступен для source facts, но не может "
        "выбрать проектное значение текущего design_decision gap; обсудить "
        "варианты без изменения карточки."
    ),
    ConversationAction.CHANGE_GAP_MODE: (
        "Изменить тип разрешения текущего gap."
    ),
    ConversationAction.LEAVE_GAP: (
        "Только по явному решению аналитика оставить текущий gap нерешённым. "
        "Это действие не закрывает gap."
    ),
    ConversationAction.REFINE_CARD: (
        "Построить точное предложение доработки карточки без немедленного "
        "изменения."
    ),
    ConversationAction.INCLUDE_CARD: "Включить карточку в итоговый ПМИ.",
    ConversationAction.EXCLUDE_CARD: "Исключить карточку из итогового ПМИ.",
    ConversationAction.EXPORT_DIAGNOSTICS: "Экспортировать диагностику сессии.",
    ConversationAction.EXPORT_PMI: "Экспортировать полный ПМИ.",
}

_REVISION_ACTIONS = frozenset(
    {
        ConversationAction.RESUME,
        ConversationAction.RESEARCH_GAP,
        ConversationAction.SUBMIT_ANALYST_ANSWER,
        ConversationAction.CONFIRM_ANALYST_ANSWER,
        ConversationAction.REJECT_ANALYST_ANSWER,
        ConversationAction.CHANGE_GAP_MODE,
        ConversationAction.LEAVE_GAP,
        ConversationAction.REFINE_CARD,
        ConversationAction.INCLUDE_CARD,
        ConversationAction.EXCLUDE_CARD,
    }
)
_GAP_ACTIONS = frozenset(
    {
        ConversationAction.RESEARCH_GAP,
        ConversationAction.SUBMIT_ANALYST_ANSWER,
        ConversationAction.CHANGE_GAP_MODE,
        ConversationAction.LEAVE_GAP,
    }
)
_CONFIRMED_ACTIONS = frozenset(
    {
        ConversationAction.LEAVE_GAP,
        ConversationAction.CONFIRM_ANALYST_ANSWER,
        ConversationAction.INCLUDE_CARD,
        ConversationAction.EXCLUDE_CARD,
    }
)


class _ConversationChatModel(BaseChatModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    transport: Any
    _bound_tools: tuple[BaseTool, ...] = PrivateAttr(default=())

    @property
    def _llm_type(self) -> str:
        return "pmi-conversation-transport"

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        bound = self.model_copy(deep=False)
        bound._bound_tools = tuple(
            tool for tool in tools if isinstance(tool, BaseTool)
        )
        return bound

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        return asyncio.run(self._agenerate(messages, stop=stop, **kwargs))

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        context = {
            "messages": [
                {
                    "type": message.type,
                    "content": message.content,
                }
                for message in messages
            ]
        }
        schemas = [convert_to_openai_tool(tool) for tool in self._bound_tools]
        call = _POLICY.build_call(PromptId.CONVERSATION, context)
        response = await self.transport.complete(call, schemas)
        if response.finish_reason != "tool_calls" or len(response.tool_calls) != 1:
            raise ConversationAgentError(
                "Conversation turn должен вернуть ровно один terminal tool call"
            )
        raw = response.tool_calls[0]
        tool_name = str(raw.get("name", ""))
        selected_tool = next(
            (tool for tool in self._bound_tools if tool.name == tool_name),
            None,
        )
        if selected_tool is None:
            raise ConversationAgentError(
                "Выбранное действие недоступно в текущем состоянии"
            )
        arguments = raw.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError as error:
                raise ConversationAgentError(
                    "Conversation tool arguments содержат невалидный JSON"
                ) from error
        if not isinstance(arguments, dict):
            raise ConversationAgentError(
                "Conversation tool arguments должны быть объектом"
            )
        args_schema = selected_tool.args_schema
        if args_schema is not None and isinstance(args_schema, type):
            try:
                args_schema.model_validate(arguments)
            except ValidationError as error:
                raise ConversationAgentError(
                    "Аргументы conversation tool имеют неверную структуру"
                ) from error
        message = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": tool_name,
                    "args": arguments,
                    "id": str(raw.get("id", "")),
                    "type": "tool_call",
                }
            ],
        )
        return ChatResult(generations=[ChatGeneration(message=message)])


class LangChainConversationAgent:
    def __init__(
        self,
        *,
        transport: LlmTransport,
        checkpointer: BaseCheckpointSaver,
    ) -> None:
        self.transport = transport
        self.checkpointer = checkpointer

    async def decide(
        self,
        *,
        context: ConversationContext,
        message_id: str,
        user_text: str,
    ) -> ConversationTurnDecision:
        captured: dict[str, ConversationTurnDecision] = {}

        def respond_to_analyst(text: str) -> str:
            text = user_facing_conversation_text(text)
            captured["decision"] = ConversationTurnDecision(
                ConversationTurnKind.ANSWER,
                text,
            )
            return text

        def request_clarification(text: str) -> str:
            text = user_facing_conversation_text(text)
            captured["decision"] = ConversationTurnDecision(
                ConversationTurnKind.CLARIFICATION,
                text,
            )
            return text

        def application_tool(action: ConversationAction) -> StructuredTool:
            def select(**raw_arguments: Any) -> str:
                announcement = user_facing_conversation_text(
                    str(raw_arguments.pop("announcement"))
                )
                arguments = self._dispatch_arguments(
                    context=context,
                    message_id=message_id,
                    action=action,
                    subject_arguments=raw_arguments,
                )
                captured["decision"] = ConversationTurnDecision(
                    ConversationTurnKind.TOOL_CALL,
                    announcement,
                    ConversationToolCall(action, arguments),
                )
                return announcement

            return StructuredTool.from_function(
                func=select,
                name=action.value,
                description=_ACTION_DESCRIPTIONS[action],
                args_schema=(
                    self._analyst_answer_schema(context)
                    if action is ConversationAction.SUBMIT_ANALYST_ANSWER
                    else _ACTION_ARGUMENTS[action]
                ),
                return_direct=True,
            )

        tools = [
            StructuredTool.from_function(
                func=respond_to_analyst,
                name="respond_to_analyst",
                description="Ответить или объяснить состояние без side effects.",
                return_direct=True,
            ),
            StructuredTool.from_function(
                func=request_clarification,
                name="request_clarification",
                description="Задать уточняющий вопрос без side effects.",
                return_direct=True,
            ),
        ]
        tools.extend(
            application_tool(action)
            for action in context.available_actions
        )
        agent = create_agent(
            model=_ConversationChatModel(transport=self.transport),
            tools=tuple(tools),
            system_prompt=_SYSTEM_PROMPT,
            checkpointer=self.checkpointer,
            name="pmi-conversation-agent",
        )
        turn = {
            "message_id": message_id,
            "user_text": user_text,
            "current_context": context.as_dict(),
        }
        await asyncio.to_thread(
            agent.invoke,
            {
                "messages": [
                    HumanMessage(
                        content=json.dumps(
                            turn,
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                    )
                ]
            },
            {
                "configurable": {"thread_id": context.session_id},
                "recursion_limit": 4,
            },
        )
        decision = captured.get("decision")
        if decision is None:
            raise ConversationAgentError(
                "Conversation agent завершился без типизированного решения"
            )
        return decision

    @staticmethod
    def _analyst_answer_schema(
        context: ConversationContext,
    ) -> type[BaseModel]:
        gap = context.open_gap
        if gap is None or not gap.closure_requirements:
            return _AnalystAnswerArguments
        if all(
            requirement.accepted_forms == ("confirmed_value",)
            for requirement in gap.closure_requirements
        ):
            return _AnalystAnswerArguments

        value_models = {
            "confirmed_value": _ConfirmedAnalystValue,
            "exact": _ExactAnalystValue,
            "finite_set": _FiniteSetAnalystValue,
            "deterministic_rule": _DeterministicRuleAnalystValue,
        }
        item_models: list[type[BaseModel]] = []
        for index, requirement in enumerate(gap.closure_requirements):
            accepted_forms = tuple(requirement.accepted_forms)
            forms = tuple(
                dict.fromkeys(("confirmed_value", *accepted_forms))
            )
            selected = tuple(
                value_models[form]
                for form in forms
                if form in value_models
            )
            if len(selected) != len(forms):
                raise ConversationAgentError(
                    "Closure contract содержит неизвестную форму"
                )
            value_type = (
                selected[0]
                if len(selected) == 1
                else reduce(or_, selected)
            )
            item_models.append(
                create_model(
                    f"AnalystValue_{index}",
                    __config__=ConfigDict(extra="forbid"),
                    path=(Literal[requirement.path], ...),
                    value=(value_type, ...),
                )
            )
        item_type = (
            item_models[0]
            if len(item_models) == 1
            else reduce(or_, item_models)
        )
        return create_model(
            "StrictAnalystAnswerArguments",
            __base__=_ToolArguments,
            values=(list[item_type], Field(min_length=1)),
        )

    @staticmethod
    def _dispatch_arguments(
        *,
        context: ConversationContext,
        message_id: str,
        action: ConversationAction,
        subject_arguments: dict[str, Any],
    ) -> dict[str, Any]:
        arguments = {
            key: LangChainConversationAgent._plain_value(value)
            for key, value in subject_arguments.items()
        }
        if action in _REVISION_ACTIONS:
            arguments["expected_revision"] = context.card_revision
        if action in _GAP_ACTIONS:
            if context.open_gap is None:
                raise ConversationAgentError(
                    "Выбранное действие требует открытый пробел"
                )
            arguments["gap_id"] = context.open_gap.gap_id
        if action in _CONFIRMED_ACTIONS:
            arguments["confirmation_message_id"] = message_id
        if action is ConversationAction.LEAVE_GAP:
            arguments.pop("decision")
        if action in {
            ConversationAction.CONFIRM_ANALYST_ANSWER,
            ConversationAction.REJECT_ANALYST_ANSWER,
        }:
            proposal = context.pending_proposal
            if proposal is None:
                raise ConversationAgentError(
                    "Выбранное действие требует ожидающую интерпретацию"
                )
            arguments["proposal_id"] = proposal.proposal_id
        if action is ConversationAction.REJECT_ANALYST_ANSWER:
            arguments["rejection_message_id"] = message_id
        return arguments

    @staticmethod
    def _plain_value(value: Any) -> Any:
        if isinstance(value, BaseModel):
            return {
                key: LangChainConversationAgent._plain_value(item)
                for key, item in value.model_dump().items()
            }
        if isinstance(value, list):
            return [
                LangChainConversationAgent._plain_value(item)
                for item in value
            ]
        if isinstance(value, dict):
            return {
                key: LangChainConversationAgent._plain_value(item)
                for key, item in value.items()
            }
        return value


__all__ = ["LangChainConversationAgent"]
