from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver

from pmi_generator.workbench.application.conversation import (
    ConversationAction,
    ConversationAgentError,
    ConversationContext,
    ConversationGapClosureContext,
    ConversationGapContext,
    ConversationProposalContext,
    ConversationTurnKind,
)
from pmi_generator.workbench.application.llm import RawCompletion
from pmi_generator.workbench.infrastructure.langchain import (
    LangChainConversationAgent,
)
from pmi_generator.workbench.infrastructure.llm import ScriptedLlmTransport


def context(
    session_id: str = "SESSION_1",
    *,
    stage: str = "нужно решение аналитика",
    continuation: str = "gap_investigation",
    available_actions: tuple[ConversationAction, ...] | None = None,
    with_open_gap: bool = True,
    with_pending_proposal: bool = False,
    gap_mode: str = "source_fact",
    accepted_forms: tuple[str, ...] = ("confirmed_value",),
) -> ConversationContext:
    return ConversationContext(
        session_id=session_id,
        card_id="CARD_1",
        card_revision=3,
        stage=stage,
        continuation=continuation,
        fields={
            "test.observation.method": {
                "status": "неизвестно",
                "value": None,
            }
        },
        open_gap=(
            ConversationGapContext(
                gap_id="GAP_1",
                question="Как наблюдать бит PTH?",
                blocking_reason="Метод не найден",
                allowed_paths=("test.observation.method",),
                resolution_mode=gap_mode,
                closure_requirements=(
                    ConversationGapClosureContext(
                        path="test.observation.method",
                        accepted_forms=accepted_forms,
                        residual_question="Как наблюдать бит PTH?",
                    ),
                ),
            )
            if with_open_gap
            else None
        ),
        available_actions=available_actions
        or (
            ConversationAction.RESEARCH_GAP,
            ConversationAction.SUBMIT_ANALYST_ANSWER,
            ConversationAction.LEAVE_GAP,
        ),
        pending_proposal=(
            ConversationProposalContext(
                proposal_id="PROPOSAL_1",
                gap_id="GAP_1",
                source_message_id="MSG_SOURCE",
                expected_revision=3,
                values=(
                    {
                        "path": "test.observation.method",
                        "value": "GET DATA",
                    },
                ),
            )
            if with_pending_proposal
            else None
        ),
    )


def tool_response(name: str, arguments: dict[str, object]) -> RawCompletion:
    return RawCompletion(
        finish_reason="tool_calls",
        tool_calls=(
            {
                "id": "call-1",
                "name": name,
                "arguments": arguments,
            },
        ),
        usage={},
        model="scripted",
    )


class ConversationAgentTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.connection = sqlite3.connect(":memory:", check_same_thread=False)
        self.addCleanup(self.connection.close)

    def agent(
        self,
        responses: list[RawCompletion],
    ) -> tuple[LangChainConversationAgent, ScriptedLlmTransport]:
        transport = ScriptedLlmTransport(responses)
        return (
            LangChainConversationAgent(
                transport=transport,
                checkpointer=SqliteSaver(self.connection),
            ),
            transport,
        )

    async def test_read_only_answer_uses_real_langgraph_agent(self) -> None:
        agent, transport = self.agent(
            [
                tool_response(
                    "respond_to_analyst",
                    {"text": "Ищу метод, потому что без него результат ненаблюдаем."},
                )
            ]
        )

        decision = await agent.decide(
            context=context(),
            message_id="MSG_1",
            user_text="Почему ты ищешь метод наблюдения?",
        )

        self.assertIs(decision.kind, ConversationTurnKind.ANSWER)
        self.assertIn("ненаблюдаем", decision.text)
        call = transport.calls[0]["call"]
        self.assertEqual(call.prompt_id.value, "conversation")
        serialized = str(call.context)
        self.assertIn("CARD_1", serialized)
        self.assertIn("Почему ты ищешь", serialized)
        self.assertNotIn("repository", serialized.casefold())

    async def test_visible_answers_hide_action_ids_for_every_card_state(self) -> None:
        cases = (
            (
                "incomplete",
                context("SESSION_INCOMPLETE"),
                (
                    "Доступны research_gap, submit_analyst_answer "
                    "и leave_gap."
                ),
                (
                    "исследовать пробел по новому вопросу",
                    "использовать ответ аналитика",
                    "оставить пробел открытым",
                ),
            ),
            (
                "complete",
                context(
                    "SESSION_COMPLETE",
                    stage="карточка подготовлена",
                    continuation="card_decision",
                    available_actions=(
                        ConversationAction.REFINE_CARD,
                        ConversationAction.INCLUDE_CARD,
                        ConversationAction.EXCLUDE_CARD,
                    ),
                    with_open_gap=False,
                ),
                "Доступны include_card, exclude_card и refine_card.",
                (
                    "включить карточку в итоговый ПМИ",
                    "исключить карточку из итогового ПМИ",
                    "продолжить доработку карточки",
                ),
            ),
            (
                "excluded",
                context(
                    "SESSION_EXCLUDED",
                    stage="карточка исключена",
                    continuation="card_decision",
                    available_actions=(
                        ConversationAction.INCLUDE_CARD,
                        ConversationAction.EXPORT_DIAGNOSTICS,
                    ),
                    with_open_gap=False,
                ),
                "После exclude_card доступны include_card и export_diagnostics.",
                (
                    "исключить карточку из итогового ПМИ",
                    "включить карточку в итоговый ПМИ",
                    "экспортировать диагностику сессии",
                ),
            ),
        )
        for name, current_context, raw_text, expected_labels in cases:
            with self.subTest(name=name):
                agent, _ = self.agent(
                    [
                        tool_response(
                            "respond_to_analyst",
                            {"text": raw_text},
                        )
                    ]
                )

                decision = await agent.decide(
                    context=current_context,
                    message_id=f"MSG_{name.upper()}",
                    user_text="Что можно сделать дальше?",
                )

                for action in ConversationAction:
                    self.assertNotIn(action.value, decision.text)
                for label in expected_labels:
                    self.assertIn(label, decision.text)

    async def test_tool_announcement_is_natural_but_action_id_is_preserved(
        self,
    ) -> None:
        agent, _ = self.agent(
            [
                tool_response(
                    "include_card",
                    {
                        "announcement": (
                            "Выполняю include_card; затем доступен refine_card."
                        ),
                    },
                )
            ]
        )

        decision = await agent.decide(
            context=context(
                "SESSION_READY",
                stage="карточка подготовлена",
                continuation="card_decision",
                available_actions=(
                    ConversationAction.INCLUDE_CARD,
                    ConversationAction.REFINE_CARD,
                ),
                with_open_gap=False,
            ),
            message_id="MSG_READY",
            user_text="Включи карточку.",
        )

        self.assertIs(
            decision.tool_call.action,
            ConversationAction.INCLUDE_CARD,
        )
        self.assertNotIn("include_card", decision.text)
        self.assertNotIn("refine_card", decision.text)
        self.assertIn("включить карточку в итоговый ПМИ", decision.text)
        self.assertIn("продолжить доработку карточки", decision.text)

    async def test_ambiguous_turn_returns_clarification_without_action(self) -> None:
        agent, _ = self.agent(
            [
                tool_response(
                    "request_clarification",
                    {"text": "Нужно только обсудить вариант или применить его?"},
                )
            ]
        )

        decision = await agent.decide(
            context=context(),
            message_id="MSG_2",
            user_text="Давай с этим разберёмся.",
        )

        self.assertIs(decision.kind, ConversationTurnKind.CLARIFICATION)
        self.assertIsNone(decision.tool_call)

    async def test_agent_selects_only_available_typed_action(self) -> None:
        agent, transport = self.agent(
            [
                tool_response(
                    "research_gap",
                    {
                        "question": "Можно ли прочитать PTH через GET DATA?",
                        "announcement": "Проверю GET DATA в источниках.",
                    },
                )
            ]
        )

        decision = await agent.decide(
            context=context(),
            message_id="MSG_3",
            user_text="Поищи через GET DATA.",
        )

        self.assertIs(decision.kind, ConversationTurnKind.TOOL_CALL)
        self.assertIs(
            decision.tool_call.action,
            ConversationAction.RESEARCH_GAP,
        )
        self.assertEqual(
            decision.tool_call.arguments,
            {
                "gap_id": "GAP_1",
                "question": "Можно ли прочитать PTH через GET DATA?",
                "expected_revision": 3,
            },
        )
        schemas = {
            item["function"]["name"]: item["function"]["parameters"]
            for item in transport.calls[0]["tools"]
        }
        self.assertNotIn("select_application_tool", schemas)
        self.assertEqual(
            set(schemas["research_gap"]["properties"]),
            {"announcement", "question"},
        )
        self.assertNotIn("gap_id", schemas["research_gap"]["properties"])
        self.assertNotIn(
            "expected_revision",
            schemas["research_gap"]["properties"],
        )
        research_tool = next(
            item
            for item in transport.calls[0]["tools"]
            if item["function"]["name"] == "research_gap"
        )
        self.assertIn("LightRAG", research_tool["function"]["description"])
        system_text = " ".join(
            str(
                transport.calls[0]["call"].context["messages"][0][
                    "content"
                ]
            ).split()
        )
        self.assertIn(
            "не утверждай, что LightRAG технически недоступен",
            system_text,
        )

    async def test_design_gap_exposes_explanation_without_research_tool(
        self,
    ) -> None:
        agent, transport = self.agent(
            [
                tool_response(
                    "propose_design_decision",
                    {
                        "announcement": (
                            "Объясню границу поиска и проектного решения."
                        )
                    },
                )
            ]
        )

        decision = await agent.decide(
            context=context(
                "SESSION_DESIGN",
                available_actions=(
                    ConversationAction.PROPOSE_DESIGN_DECISION,
                    ConversationAction.SUBMIT_ANALYST_ANSWER,
                ),
                gap_mode="design_decision",
            ),
            message_id="MSG_DESIGN",
            user_text="Почему ты не ищешь это в LightRAG?",
        )

        self.assertIs(
            decision.tool_call.action,
            ConversationAction.PROPOSE_DESIGN_DECISION,
        )
        tool_names = {
            item["function"]["name"]
            for item in transport.calls[0]["tools"]
        }
        self.assertIn("propose_design_decision", tool_names)
        self.assertNotIn("research_gap", tool_names)

    async def test_analyst_answer_has_exact_nested_openai_schema(self) -> None:
        agent, transport = self.agent(
            [
                tool_response(
                    "submit_analyst_answer",
                    {
                        "announcement": "Применяю ответ аналитика.",
                        "values": [
                            {
                                "path": "test.observation.method",
                                "value": "Get Data",
                            }
                        ],
                    },
                )
            ]
        )

        decision = await agent.decide(
            context=context(),
            message_id="MSG_LIVE_SHAPED",
            user_text="Может Get Data подойдёт?",
        )

        self.assertEqual(
            decision.tool_call.arguments["values"],
            [
                {
                    "path": "test.observation.method",
                    "value": "Get Data",
                }
            ],
        )
        schemas = {
            item["function"]["name"]: item["function"]["parameters"]
            for item in transport.calls[0]["tools"]
        }
        item_schema = schemas["submit_analyst_answer"]["properties"]["values"][
            "items"
        ]
        self.assertEqual(item_schema["additionalProperties"], False)
        self.assertEqual(set(item_schema["required"]), {"path", "value"})
        self.assertEqual(
            set(item_schema["properties"]),
            {"path", "value"},
        )
        values_schema = schemas["submit_analyst_answer"]["properties"]["values"]
        self.assertEqual(values_schema["minItems"], 1)

    async def test_exact_only_gap_requires_explicit_exact_value_form(self) -> None:
        exact_context = context(
            "SESSION_EXACT",
            accepted_forms=("exact",),
        )
        malformed_agent, malformed_transport = self.agent(
            [
                tool_response(
                    "submit_analyst_answer",
                    {
                        "announcement": "Предлагаю точное значение.",
                        "values": [
                            {
                                "path": "test.observation.method",
                                "value": "GET DATA",
                            }
                        ],
                    },
                )
            ]
        )

        with self.assertRaisesRegex(
            ConversationAgentError,
            "неверную структуру",
        ):
            await malformed_agent.decide(
                context=exact_context,
                message_id="MSG_EXACT_RAW",
                user_text="GET DATA",
            )

        schema = next(
            item["function"]["parameters"]
            for item in malformed_transport.calls[0]["tools"]
            if item["function"]["name"] == "submit_analyst_answer"
        )
        serialized_schema = str(schema)
        self.assertIn("exact", serialized_schema)
        self.assertIn("confirmed_value", serialized_schema)
        self.assertNotIn("finite_set", serialized_schema)
        self.assertNotIn("deterministic_rule", serialized_schema)

        confirmed_agent, _ = self.agent(
            [
                tool_response(
                    "submit_analyst_answer",
                    {
                        "announcement": "Сохраняю недостаточный ответ.",
                        "values": [
                            {
                                "path": "test.observation.method",
                                "value": {
                                    "kind": "confirmed_value",
                                    "value": "любой способ",
                                },
                            }
                        ],
                    },
                )
            ]
        )
        confirmed = await confirmed_agent.decide(
            context=exact_context,
            message_id="MSG_CONFIRMED_ONLY",
            user_text="Пусть будет любой способ.",
        )
        self.assertEqual(
            confirmed.tool_call.arguments["values"][0]["value"]["kind"],
            "confirmed_value",
        )

        exact_agent, transport = self.agent(
            [
                tool_response(
                    "submit_analyst_answer",
                    {
                        "announcement": "Предлагаю точное значение.",
                        "values": [
                            {
                                "path": "test.observation.method",
                                "value": {
                                    "kind": "exact",
                                    "value": "GET DATA",
                                },
                            }
                        ],
                    },
                )
            ]
        )

        decision = await exact_agent.decide(
            context=exact_context,
            message_id="MSG_EXACT",
            user_text="GET DATA",
        )

        self.assertEqual(
            decision.tool_call.arguments["values"],
            [
                {
                    "path": "test.observation.method",
                    "value": {
                        "kind": "exact",
                        "value": "GET DATA",
                    },
                }
            ],
        )
        serialized_context = str(transport.calls[0]["call"].context)
        self.assertIn("closure_schema_version", serialized_context)
        self.assertIn("closure_requirements", serialized_context)
        self.assertIn("exact", serialized_context)

    async def test_leave_gap_schema_cannot_express_close_decision(self) -> None:
        agent, transport = self.agent(
            [
                tool_response(
                    "leave_gap",
                    {
                        "announcement": "Закрываю пробел.",
                        "decision": "close",
                        "reason": "Значение подтверждено.",
                    },
                )
            ]
        )

        with self.assertRaisesRegex(
            ConversationAgentError,
            "неверную структуру",
        ):
            await agent.decide(
                context=context(),
                message_id="MSG_CLOSE",
                user_text="Значение корректно, пробел можно закрыть.",
            )

        schema = next(
            item["function"]["parameters"]
            for item in transport.calls[0]["tools"]
            if item["function"]["name"] == "leave_gap"
        )
        self.assertEqual(
            schema["properties"]["decision"]["const"],
            "leave_open",
        )

    async def test_malformed_analyst_answer_is_rejected_before_dispatch(self) -> None:
        cases = (
            (
                "field instead of path",
                [{"field": "test.observation.method", "value": "Get Data"}],
            ),
            ("missing path", [{"value": "Get Data"}]),
            (
                "extra key",
                [
                    {
                        "path": "test.observation.method",
                        "value": "Get Data",
                        "evidence_id": None,
                    }
                ],
            ),
            ("empty values", []),
        )
        for name, values in cases:
            with self.subTest(name=name):
                agent, _ = self.agent(
                    [
                        tool_response(
                            "submit_analyst_answer",
                            {
                                "announcement": "Применяю ответ аналитика.",
                                "values": values,
                            },
                        )
                    ]
                )

                with self.assertRaisesRegex(
                    ConversationAgentError,
                    "неверную структуру",
                ):
                    await agent.decide(
                        context=context(f"SESSION_{name}"),
                        message_id="MSG_MALFORMED",
                        user_text="Может Get Data подойдёт?",
                    )

    async def test_confirmation_uses_persisted_proposal_identifiers(self) -> None:
        agent, transport = self.agent(
            [
                tool_response(
                    "confirm_analyst_answer",
                    {"announcement": "Подтверждаю показанную интерпретацию."},
                )
            ]
        )

        decision = await agent.decide(
            context=context(
                "SESSION_CONFIRM",
                available_actions=(
                    ConversationAction.CONFIRM_ANALYST_ANSWER,
                    ConversationAction.REJECT_ANALYST_ANSWER,
                ),
                with_pending_proposal=True,
            ),
            message_id="MSG_CONFIRM",
            user_text="Да, подтверждаю.",
        )

        self.assertIs(
            decision.tool_call.action,
            ConversationAction.CONFIRM_ANALYST_ANSWER,
        )
        self.assertEqual(
            decision.tool_call.arguments,
            {
                "expected_revision": 3,
                "confirmation_message_id": "MSG_CONFIRM",
                "proposal_id": "PROPOSAL_1",
            },
        )
        schema = next(
            item["function"]["parameters"]
            for item in transport.calls[0]["tools"]
            if item["function"]["name"] == "confirm_analyst_answer"
        )
        self.assertEqual(set(schema["properties"]), {"announcement"})
        serialized_context = str(transport.calls[0]["call"].context)
        self.assertIn("PROPOSAL_1", serialized_context)
        self.assertIn("test.observation.method", serialized_context)

    async def test_unavailable_action_is_rejected_after_model_choice(self) -> None:
        agent, _ = self.agent(
            [
                tool_response(
                    "include_card",
                    {
                        "announcement": "Включу карточку.",
                    },
                )
            ]
        )

        with self.assertRaisesRegex(ConversationAgentError, "недоступно"):
            await agent.decide(
                context=context(),
                message_id="MSG_4",
                user_text="Включи карточку.",
            )

    async def test_multiple_tool_calls_are_rejected_without_loop(self) -> None:
        response = RawCompletion(
            finish_reason="tool_calls",
            tool_calls=(
                {
                    "id": "one",
                    "name": "respond_to_analyst",
                    "arguments": {"text": "A"},
                },
                {
                    "id": "two",
                    "name": "respond_to_analyst",
                    "arguments": {"text": "B"},
                },
            ),
            usage={},
            model="scripted",
        )
        agent, transport = self.agent([response])

        with self.assertRaisesRegex(ConversationAgentError, "ровно один"):
            await agent.decide(
                context=context(),
                message_id="MSG_5",
                user_text="Ответь дважды.",
            )

        self.assertEqual(len(transport.calls), 1)

    async def test_sqlite_checkpoint_restores_same_conversation_after_restart(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "conversation.sqlite3"
            first_connection = sqlite3.connect(path, check_same_thread=False)
            first_transport = ScriptedLlmTransport(
                [tool_response("respond_to_analyst", {"text": "Первый ответ."})]
            )
            first = LangChainConversationAgent(
                transport=first_transport,
                checkpointer=SqliteSaver(first_connection),
            )
            await first.decide(
                context=context("SESSION_RESTART"),
                message_id="MSG_1",
                user_text="Первый вопрос.",
            )
            first_connection.close()

            second_connection = sqlite3.connect(path, check_same_thread=False)
            self.addCleanup(second_connection.close)
            second_transport = ScriptedLlmTransport(
                [tool_response("respond_to_analyst", {"text": "Второй ответ."})]
            )
            second = LangChainConversationAgent(
                transport=second_transport,
                checkpointer=SqliteSaver(second_connection),
            )

            decision = await second.decide(
                context=context("SESSION_RESTART"),
                message_id="MSG_2",
                user_text="Продолжим.",
            )

            self.assertEqual(decision.text, "Второй ответ.")
            serialized = str(second_transport.calls[0]["call"].context)
            self.assertIn("Первый вопрос.", serialized)
            self.assertIn("Продолжим.", serialized)


if __name__ == "__main__":
    unittest.main()
