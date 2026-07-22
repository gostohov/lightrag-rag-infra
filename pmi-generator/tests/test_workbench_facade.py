from __future__ import annotations

import asyncio
import sqlite3
import threading
import tempfile
import unittest
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, Mock

from langgraph.checkpoint.sqlite import SqliteSaver

from pmi_generator.workbench.application.card_population import population_tool
from pmi_generator.workbench.application.conversation import (
    ConversationAction,
    ConversationEffect,
    ConversationToolResult,
    ConversationTurnDecision,
    ConversationTurnKind,
    ConversationToolCall,
)
from pmi_generator.workbench.application.decomposition import (
    DecompositionBudget,
    DecompositionBudgetPolicy,
    DecompositionArguments,
    DecompositionRoute,
    WindowPlanError,
    WindowingDecision,
    decomposition_tool,
)
from pmi_generator.workbench.application.gap_investigation import (
    RetrievalResponse,
    ask_lightrag_tool,
    expand_lightrag_tool,
    submit_gap_result_tool,
)
from pmi_generator.workbench.application.facade import (
    WorkbenchApplication,
    WorkbenchOperation,
)
from pmi_generator.workbench.application.llm import (
    AttemptDiscardedError,
    LlmToolRuntime,
    RawCompletion,
    TypedToolRegistry,
)
from pmi_generator.workbench.application.prompting import default_policy
from pmi_generator.workbench.application.selection_review import selection_review_tool
from pmi_generator.workbench.application.session import (
    SessionEventKind,
    SessionService,
)
from pmi_generator.workbench.application.state import AttemptRecord, AttemptStatus
from pmi_generator.workbench.application.workflow import (
    CommandKind,
    WorkflowCommand,
    WorkflowConsistencyError,
    WorkflowState,
    apply_command,
)
from pmi_generator.workbench.domain import (
    CardMutation,
    ContentField,
    DomainValidationError,
    GapClosureContract,
    GapPathClosure,
    GapResolutionMode,
    GapValueForm,
    PathNotAllowedError,
    RelatedGap,
    SourceDocument,
    SourcePage,
    SourcePosition,
    TestCard,
)
from pmi_generator.workbench.infrastructure.llm import ScriptedLlmTransport
from pmi_generator.workbench.infrastructure.langchain import (
    LangChainConversationAgent,
)
from pmi_generator.workbench.infrastructure.storage import (
    InMemoryDatabase,
    InMemoryUnitOfWork,
)
from pmi_generator.workbench.infrastructure.retrieval import ScriptedRetrieval
from pmi_generator.workbench.infrastructure.workers import ProductionPromptWorkers


class RecordingWorkflowRuntime:
    def __init__(self) -> None:
        self.states: dict[str, WorkflowState] = {}
        self.commands: list[WorkflowCommand] = []

    def execute(self, thread_id: str, command: WorkflowCommand) -> WorkflowState:
        state = apply_command(self.current_state(thread_id), command)
        self.states[thread_id] = state
        self.commands.append(command)
        return state

    def current_state(self, thread_id: str) -> WorkflowState:
        return self.states.get(thread_id, WorkflowState.empty())


def document() -> SourceDocument:
    return SourceDocument(
        pages=(
            SourcePage(
                page_index=1,
                logical_page=1,
                lines=(
                    "Проверить первый байт команды",
                    "Если байт не равен 81, карта возвращает 6987",
                ),
            ),
        ),
        sections=(),
    )


def decomposition() -> DecompositionArguments:
    return DecompositionArguments(
        outcome="skeletons_created",
        explanation="",
        skeletons=[
            {
                "title": "Проверка первого байта",
                "condition": "первый байт не равен 81",
                "changed_factor": "первый байт",
                "input_value": None,
                "action": "отправить команду",
                "condition_ranges": [{"page": 1, "line_start": 1, "line_end": 2}],
                "changed_factor_ranges": [
                    {"page": 1, "line_start": 1, "line_end": 2}
                ],
                "input_value_ranges": [],
                "action_ranges": [{"page": 1, "line_start": 1, "line_end": 2}],
                "consequences": [
                    {
                        "text": "карта возвращает 6987",
                        "evidence_ranges": [
                            {"page": 1, "line_start": 2, "line_end": 2}
                        ],
                    }
                ],
                "gaps": [
                    {
                        "kind": "input_value",
                        "question": "Какое конкретное значение использовать?",
                        "target_paths": ["test_design.input_value"],
                    }
                ],
            }
        ],
        line_assessments=[
            {
                "page": 1,
                "line": line,
                "role": "evidence",
                "reason": "Строка использована каркасом",
            }
            for line in (1, 2)
        ],
    )


class WorkbenchFacadeWorkflowTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.database = InMemoryDatabase()
        self.workflow = RecordingWorkflowRuntime()
        self.counts: dict[str, int] = {}
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        tools = TypedToolRegistry()
        tools.register(decomposition_tool())
        tools.register(population_tool())
        tools.register(selection_review_tool())
        arguments = decomposition()
        runtime = LlmToolRuntime(
            transport=ScriptedLlmTransport(
                [
                    RawCompletion(
                        "tool_calls",
                        (
                            {
                                "id": "call-1",
                                "name": "submit_decomposition",
                                "arguments": {
                                    "outcome": arguments.outcome,
                                    "explanation": arguments.explanation,
                                    "skeletons": arguments.skeletons,
                                    "line_assessments": arguments.line_assessments,
                                },
                            },
                        ),
                        {},
                        "fake",
                    ),
                    RawCompletion(
                        "tool_calls",
                        (
                            {
                                "id": "call-2",
                                "name": "submit_card_population",
                                "arguments": {
                                    "source_values": [
                                        {
                                            "path": "requirement.condition",
                                            "value": "первый байт не равен 81",
                                            "evidence_id": "EVIDENCE_0001",
                                        },
                                        {
                                            "path": "requirement.behavior",
                                            "value": "карта возвращает ошибку",
                                            "evidence_id": "EVIDENCE_0001",
                                        },
                                        {
                                            "path": "test.action",
                                            "value": "отправить команду",
                                            "evidence_id": "EVIDENCE_0001",
                                        },
                                        {
                                            "path": "test.changed_factor",
                                            "value": "первый байт",
                                            "evidence_id": "EVIDENCE_0001",
                                        },
                                        {
                                            "path": "test.expected.status_word",
                                            "value": "6987",
                                            "evidence_id": "EVIDENCE_0001",
                                        },
                                        {
                                            "path": "test.observation.method",
                                            "value": "проверить SW1SW2",
                                            "evidence_id": "EVIDENCE_0001",
                                        },
                                    ],
                                    "derivations": [],
                                    "not_applicable": [],
                                    "gaps": [
                                        {
                                            "question": "Какое контрольное значение использовать?",
                                            "blocking_reason": "Нужна исполнимая параметризация",
                                            "allowed_paths": ["test.control_values"],
                                            "dependencies": ["requirement.condition"],
                                            "closure_criterion": "Указано контрольное значение",
                                            "resolution_mode": "source_fact",
                                        }
                                    ],
                                },
                            },
                        ),
                        {},
                        "fake",
                    ),
                    RawCompletion(
                        "tool_calls",
                        (
                            {
                                "id": "call-3",
                                "name": "submit_selection_review",
                                "arguments": {"outcome": "approved", "issues": []},
                            },
                        ),
                        {},
                        "fake",
                    ),
                ]
            ),
            tools=tools,
            uow_factory=lambda: InMemoryUnitOfWork(self.database),
        )
        sessions = SessionService(
            uow_factory=lambda: InMemoryUnitOfWork(self.database)
        )
        workers = ProductionPromptWorkers(
            document=document(),
            uow_factory=lambda: InMemoryUnitOfWork(self.database),
            policy=default_policy(),
            runtime_factory=lambda: runtime,
            retrieval_factory=lambda _selection: self.fail(
                "retrieval не должен вызываться"
            ),
            sessions=sessions,
            next_id=self.next_id,
        )
        self.facade = WorkbenchApplication(
            document=document(),
            run_dir=Path(self.temporary.name),
            uow_factory=lambda: InMemoryUnitOfWork(self.database),
            workflow=self.workflow,
            sessions=sessions,
            workers=workers,
            next_id=self.next_id,
        )

    def next_id(self, prefix: str) -> str:
        self.counts[prefix] = self.counts.get(prefix, 0) + 1
        return f"{prefix}_{self.counts[prefix]:04d}"

    async def test_prompt_1_and_skeleton_decision_are_owned_by_root_runtime(self) -> None:
        source = document()
        saved = self.facade.save_selection(
            "section-1",
            source.select(SourcePosition(1, 1), SourcePosition(1, 2)),
        )

        result = await self.facade.decompose(saved).awaitable
        card_id = self.facade.take_skeleton(
            saved.selection_id,
            result.skeleton_ids[0],
        )

        self.assertEqual(card_id, "CARD_0001")
        self.assertEqual(
            [item.kind for item in self.workflow.commands],
            [
                CommandKind.CONFIRM_SELECTION,
                CommandKind.BEGIN_ATTEMPT,
                CommandKind.APPLY_DECOMPOSITION,
                CommandKind.TAKE_SKELETON,
            ],
        )

    async def test_prompt_1_hard_limit_is_checked_before_selection_and_attempt(
        self,
    ) -> None:
        source = document()
        selection = source.select(SourcePosition(1, 1), SourcePosition(1, 2))
        self.facade.windowing_policy = Mock(
            assess=Mock(
                return_value=WindowingDecision(
                    route=DecompositionRoute.HARD_LIMIT,
                    budget=DecompositionBudget(
                        line_count=2,
                        estimated_tokens=100,
                        max_lines=1,
                        max_estimated_tokens=50,
                    ),
                    hard_max_lines=1,
                    hard_max_estimated_tokens=50,
                )
            )
        )

        with self.assertRaises(WindowPlanError):
            self.facade.save_selection("section-1", selection)

        self.assertNotIn("SELECTION", self.counts)
        self.assertEqual(self.facade.selections(), ())
        self.assertEqual(self.workflow.commands, [])

    async def test_prompt_1_hard_limit_is_rechecked_before_attempt(self) -> None:
        source = document()
        saved = self.facade.save_selection(
            "section-1",
            source.select(SourcePosition(1, 1), SourcePosition(1, 2)),
        )
        self.facade.windowing_policy = Mock(
            assess=Mock(
                return_value=WindowingDecision(
                    route=DecompositionRoute.HARD_LIMIT,
                    budget=DecompositionBudget(
                        line_count=2,
                        estimated_tokens=100,
                        max_lines=1,
                        max_estimated_tokens=50,
                    ),
                    hard_max_lines=1,
                    hard_max_estimated_tokens=50,
                )
            )
        )
        commands_before = list(self.workflow.commands)

        with self.assertRaises(WindowPlanError):
            self.facade.decompose(saved)

        self.assertNotIn("ATTEMPT", self.counts)
        self.assertEqual(self.workflow.commands, commands_before)

    async def _populated_card(self) -> tuple[object, str, str, str]:
        source = document()
        saved = self.facade.save_selection(
            "section-1",
            source.select(SourcePosition(1, 1), SourcePosition(1, 2)),
        )
        decomposition_result = await self.facade.decompose(saved).awaitable
        skeleton_id = decomposition_result.skeleton_ids[0]
        card_id = self.facade.take_skeleton(saved.selection_id, skeleton_id)
        session_id = self.facade.open_card_session(saved.selection_id, card_id)
        await self.facade.populate(
            saved,
            skeleton_id,
            session_id,
            card_id,
        ).awaitable
        return saved, skeleton_id, session_id, card_id

    async def test_conversation_context_exposes_typed_state_and_actions(self) -> None:
        _saved, _skeleton_id, session_id, card_id = await self._populated_card()

        context = self.facade.conversation_context(session_id, card_id)

        self.assertEqual(context.card_revision, 1)
        self.assertEqual(context.open_gap.gap_id, "GAP_0001")
        self.assertEqual(context.open_gap.resolution_mode, "source_fact")
        self.assertIn(
            ConversationAction.RESEARCH_GAP,
            context.available_actions,
        )
        self.assertIn(
            ConversationAction.SUBMIT_ANALYST_ANSWER,
            context.available_actions,
        )
        self.assertNotIn(
            ConversationAction.REFINE_CARD,
            context.available_actions,
        )

    async def test_conversation_plain_answer_is_persisted_without_card_mutation(
        self,
    ) -> None:
        saved, skeleton_id, session_id, card_id = await self._populated_card()
        sequence = self.facade.append(
            session_id,
            SessionEventKind.ANALYST,
            "Почему нужен этот пробел?",
            {"author": "Аналитик"},
        )
        agent = Mock()
        agent.decide = AsyncMock(
            return_value=ConversationTurnDecision(
                ConversationTurnKind.ANSWER,
                "Пробел связан с отсутствующим контрольным значением.",
            )
        )
        self.facade.conversation_agent = agent
        revision = self.facade.card(card_id).revision

        result = await self.facade.conversation_turn(
            saved,
            skeleton_id,
            session_id,
            card_id,
            f"MSG_{sequence:06d}",
        ).awaitable

        self.assertIs(result.decision.kind, ConversationTurnKind.ANSWER)
        self.assertEqual(self.facade.card(card_id).revision, revision)
        response = self.facade.history(session_id)[-1]
        self.assertIs(response.kind, SessionEventKind.ASSISTANT)
        self.assertTrue(response.metadata["conversation_response"])
        self.assertEqual(response.metadata["turn_kind"], "answer")

    async def test_research_action_excludes_trigger_message_from_analyst_evidence(
        self,
    ) -> None:
        saved, skeleton_id, session_id, card_id = await self._populated_card()
        sequence = self.facade.append(
            session_id,
            SessionEventKind.ANALYST,
            "Поищи, можно ли использовать GET DATA.",
            {"author": "Аналитик"},
        )
        message_id = f"MSG_{sequence:06d}"
        operation = Mock(awaitable=Mock(), cancel=Mock())
        self.facade.investigate_gap = Mock(return_value=operation)  # type: ignore[method-assign]

        result = self.facade.dispatch_conversation_tool(
            saved,
            skeleton_id,
            session_id,
            card_id,
            message_id,
            ConversationToolCall(
                ConversationAction.RESEARCH_GAP,
                {
                    "gap_id": "GAP_0001",
                    "question": "Можно ли использовать GET DATA?",
                    "expected_revision": 1,
                },
            ),
        )

        self.facade.investigate_gap.assert_called_once_with(
            saved,
            session_id,
            card_id,
            "GAP_0001",
            "Можно ли использовать GET DATA?",
            message_id,
        )
        self.assertIs(result.awaitable, operation.awaitable)

    async def test_design_decision_explains_research_scope_without_mutation(
        self,
    ) -> None:
        saved, skeleton_id, session_id, card_id = await self._populated_card()
        with InMemoryUnitOfWork(self.database) as uow:
            card = uow.cards.get(card_id)
            card.change_gap_resolution_mode(
                "GAP_0001",
                GapResolutionMode.DESIGN_DECISION,
            )
            uow.cards.save(card)
        sequence = self.facade.append(
            session_id,
            SessionEventKind.ANALYST,
            "Почему не поискать значение в LightRAG?",
            {"author": "Аналитик"},
        )
        message_id = f"MSG_{sequence:06d}"
        before = self.facade.card(card_id).revision

        context = self.facade.conversation_context(session_id, card_id)
        result = self.facade.dispatch_conversation_tool(
            saved,
            skeleton_id,
            session_id,
            card_id,
            message_id,
            ConversationToolCall(
                ConversationAction.PROPOSE_DESIGN_DECISION,
                {},
            ),
        )

        self.assertIn(
            ConversationAction.PROPOSE_DESIGN_DECISION,
            context.available_actions,
        )
        self.assertNotIn(
            ConversationAction.RESEARCH_GAP,
            context.available_actions,
        )
        self.assertIn("LightRAG доступен", result.text)
        self.assertIn("не может выбрать проектное значение", result.text)
        self.assertNotIn("технически недоступ", result.text)
        self.assertEqual(self.facade.card(card_id).revision, before)

    async def test_live_shaped_research_turn_reaches_application_dispatch(
        self,
    ) -> None:
        saved, skeleton_id, session_id, card_id = await self._populated_card()
        sequence = self.facade.append(
            session_id,
            SessionEventKind.ANALYST,
            "Поищи, можно ли использовать GET DATA.",
            {"author": "Аналитик"},
        )
        message_id = f"MSG_{sequence:06d}"
        connection = sqlite3.connect(":memory:", check_same_thread=False)
        self.addCleanup(connection.close)
        self.facade.conversation_agent = LangChainConversationAgent(
            transport=ScriptedLlmTransport(
                [
                    RawCompletion(
                        "tool_calls",
                        (
                            {
                                "id": "call-research",
                                "name": "research_gap",
                                "arguments": {
                                    "question": "Можно ли прочитать PTH через GET DATA?",
                                    "announcement": "Проверю GET DATA в источниках.",
                                },
                            },
                        ),
                        {},
                        "fake",
                    )
                ]
            ),
            checkpointer=SqliteSaver(connection),
        )
        seen: list[tuple[object, ...]] = []

        def investigate_gap(*args: object) -> WorkbenchOperation[str]:
            seen.append(args)

            async def finish() -> str:
                return "not_found"

            return WorkbenchOperation(finish(), lambda: None)

        self.facade.investigate_gap = investigate_gap  # type: ignore[method-assign]

        result = await self.facade.conversation_turn(
            saved,
            skeleton_id,
            session_id,
            card_id,
            message_id,
        ).awaitable

        self.assertEqual(
            seen,
            [
                (
                    saved,
                    session_id,
                    card_id,
                    "GAP_0001",
                    "Можно ли прочитать PTH через GET DATA?",
                    message_id,
                )
            ],
        )
        self.assertIs(
            result.decision.tool_call.action,
            ConversationAction.RESEARCH_GAP,
        )

    async def test_natural_research_turn_reaches_actual_retrieval_payload(
        self,
    ) -> None:
        saved, skeleton_id, session_id, card_id = await self._populated_card()
        user_text = "Поищи, можно ли прочитать PTH через GET DATA с тегом C7."
        research_question = "Можно ли прочитать PTH через GET DATA с тегом C7?"
        sequence = self.facade.append(
            session_id,
            SessionEventKind.ANALYST,
            user_text,
            {"author": "Аналитик"},
        )
        message_id = f"MSG_{sequence:06d}"
        conversation_connection = sqlite3.connect(
            ":memory:",
            check_same_thread=False,
        )
        self.addCleanup(conversation_connection.close)
        self.facade.conversation_agent = LangChainConversationAgent(
            transport=ScriptedLlmTransport(
                [
                    RawCompletion(
                        "tool_calls",
                        (
                            {
                                "id": "conversation-research",
                                "name": "research_gap",
                                "arguments": {
                                    "question": research_question,
                                    "announcement": "Проверю GET DATA и тег C7.",
                                },
                            },
                        ),
                        {},
                        "fake",
                    )
                ]
            ),
            checkpointer=SqliteSaver(conversation_connection),
        )
        prompt_3_transport = ScriptedLlmTransport(
            [
                RawCompletion(
                    "tool_calls",
                    (
                        {
                            "id": "narrow",
                            "name": "ask_lightrag",
                            "arguments": {
                                "question": "Какое контрольное значение использовать?"
                            },
                        },
                    ),
                    {},
                    "fake",
                ),
                RawCompletion(
                    "tool_calls",
                    (
                        {
                            "id": "broad",
                            "name": "expand_lightrag",
                            "arguments": {
                                "call_id": "narrow",
                                "reason": "Нужен более широкий контекст",
                            },
                        },
                    ),
                    {},
                    "fake",
                ),
                RawCompletion(
                    "tool_calls",
                    (
                        {
                            "id": "submit",
                            "name": "submit_gap_result",
                            "arguments": {
                                "outcome": "not_found",
                                "updates": [],
                                "unknown_fields": ["test.control_values"],
                                "missing_fact": "способ чтения PTH",
                                "summary": "Способ не найден",
                                "contradictions": [],
                            },
                        },
                    ),
                    {},
                    "fake",
                ),
            ]
        )
        registry = TypedToolRegistry()
        for tool in (
            ask_lightrag_tool(),
            expand_lightrag_tool(),
            submit_gap_result_tool(),
        ):
            registry.register(tool)
        runtime = LlmToolRuntime(
            transport=prompt_3_transport,
            tools=registry,
            uow_factory=lambda: InMemoryUnitOfWork(self.database),
        )
        retrieval = ScriptedRetrieval(
            [
                RetrievalResponse("Узкий поиск не дал точного фрагмента", ()),
                RetrievalResponse("Расширенный поиск не дал точного фрагмента", ()),
            ]
        )
        self.facade.workers = ProductionPromptWorkers(
            document=document(),
            uow_factory=lambda: InMemoryUnitOfWork(self.database),
            policy=default_policy(),
            runtime_factory=lambda: runtime,
            retrieval_factory=lambda _selection: retrieval,
            sessions=self.facade.sessions,
            next_id=self.next_id,
        )

        await self.facade.conversation_turn(
            saved,
            skeleton_id,
            session_id,
            card_id,
            message_id,
        ).awaitable

        self.assertEqual(
            [call.question for call in retrieval.calls],
            [research_question, research_question],
        )
        self.assertEqual(
            [call.profile.name for call in retrieval.calls],
            ["узкий поиск", "расширенный поиск"],
        )
        first_context = prompt_3_transport.calls[0]["call"].context
        self.assertEqual(
            first_context["research_question"],
            {"text": research_question},
        )
        self.assertNotIn("analyst_messages", first_context)
        with InMemoryUnitOfWork(self.database) as uow:
            observations = uow.records.list_kind("retrieval_observation")
        self.assertEqual(
            [item.payload["question"] for item in observations],
            [research_question, research_question],
        )
        diagnostics = self.facade.export_diagnostics(session_id, card_id)
        diagnostic_text = diagnostics.read_text(encoding="utf-8")
        self.assertIn(user_text, diagnostic_text)
        self.assertIn(research_question, diagnostic_text)
        self.assertIn('"profile": "узкий поиск"', diagnostic_text)
        self.assertIn('"profile": "расширенный поиск"', diagnostic_text)
        card = self.facade.card(card_id)
        self.assertFalse(
            any(
                evidence.message_id == message_id
                for evidence in card.evidence.values()
            )
        )

    async def test_cancelled_conversation_discards_late_agent_answer(self) -> None:
        saved, skeleton_id, session_id, card_id = await self._populated_card()
        sequence = self.facade.append(
            session_id,
            SessionEventKind.ANALYST,
            "Почему нужен этот пробел?",
            {"author": "Аналитик"},
        )
        started = asyncio.Event()
        release = asyncio.Event()

        async def decide(**_kwargs: object) -> ConversationTurnDecision:
            started.set()
            await release.wait()
            return ConversationTurnDecision(
                ConversationTurnKind.ANSWER,
                "Поздний ответ.",
            )

        agent = Mock()
        agent.decide = decide
        self.facade.conversation_agent = agent
        operation = self.facade.conversation_turn(
            saved,
            skeleton_id,
            session_id,
            card_id,
            f"MSG_{sequence:06d}",
        )
        task = asyncio.create_task(operation.awaitable)
        await started.wait()

        operation.cancel()
        release.set()

        with self.assertRaises(AttemptDiscardedError):
            await task
        self.assertNotIn(
            "Поздний ответ.",
            [event.text for event in self.facade.history(session_id)],
        )

    async def test_cancelled_late_tool_decisions_do_not_create_or_confirm_proposal(
        self,
    ) -> None:
        saved, skeleton_id, session_id, card_id = await self._populated_card()
        source_sequence = self.facade.append(
            session_id,
            SessionEventKind.ANALYST,
            "Использовать 80.",
            {"author": "Аналитик"},
        )
        source_message_id = f"MSG_{source_sequence:06d}"

        async def cancel_late_decision(
            decision: ConversationTurnDecision,
            message_id: str,
        ) -> None:
            started = asyncio.Event()
            release = asyncio.Event()

            async def decide(**_kwargs: object) -> ConversationTurnDecision:
                started.set()
                await release.wait()
                return decision

            agent = Mock()
            agent.decide = decide
            self.facade.conversation_agent = agent
            operation = self.facade.conversation_turn(
                saved,
                skeleton_id,
                session_id,
                card_id,
                message_id,
            )
            task = asyncio.create_task(operation.awaitable)
            await started.wait()
            operation.cancel()
            release.set()
            with self.assertRaises(AttemptDiscardedError):
                await task

        await cancel_late_decision(
            ConversationTurnDecision(
                ConversationTurnKind.TOOL_CALL,
                "Предлагаю значение 80.",
                ConversationToolCall(
                    ConversationAction.SUBMIT_ANALYST_ANSWER,
                    {
                        "gap_id": "GAP_0001",
                        "values": [
                            {
                                "path": "test.control_values",
                                "value": ["80"],
                            }
                        ],
                        "expected_revision": 1,
                    },
                ),
            ),
            source_message_id,
        )
        with InMemoryUnitOfWork(self.database) as uow:
            self.assertEqual(
                uow.records.list_kind("analyst_answer_proposal"),
                [],
            )
        self.assertEqual(self.facade.card(card_id).revision, 1)

        self.facade.dispatch_conversation_tool(
            saved,
            skeleton_id,
            session_id,
            card_id,
            source_message_id,
            ConversationToolCall(
                ConversationAction.SUBMIT_ANALYST_ANSWER,
                {
                    "gap_id": "GAP_0001",
                    "values": [
                        {
                            "path": "test.control_values",
                            "value": ["80"],
                        }
                    ],
                    "expected_revision": 1,
                },
            ),
        )
        pending = self.facade.conversation_context(
            session_id,
            card_id,
        ).pending_proposal
        with self.assertRaisesRegex(
            ValueError,
            "отдельное сообщение",
        ):
            self.facade.dispatch_conversation_tool(
                saved,
                skeleton_id,
                session_id,
                card_id,
                source_message_id,
                ConversationToolCall(
                    ConversationAction.CONFIRM_ANALYST_ANSWER,
                    {
                        "proposal_id": pending.proposal_id,
                        "expected_revision": 1,
                        "confirmation_message_id": source_message_id,
                    },
                ),
            )
        self.assertEqual(self.facade.card(card_id).revision, 1)
        confirmation_sequence = self.facade.append(
            session_id,
            SessionEventKind.ANALYST,
            "Да, подтверждаю.",
            {"author": "Аналитик"},
        )
        confirmation_message_id = f"MSG_{confirmation_sequence:06d}"

        await cancel_late_decision(
            ConversationTurnDecision(
                ConversationTurnKind.TOOL_CALL,
                "Применяю подтверждение.",
                ConversationToolCall(
                    ConversationAction.CONFIRM_ANALYST_ANSWER,
                    {
                        "proposal_id": pending.proposal_id,
                        "expected_revision": 1,
                        "confirmation_message_id": (
                            confirmation_message_id
                        ),
                    },
                ),
            ),
            confirmation_message_id,
        )

        self.assertEqual(self.facade.card(card_id).revision, 1)
        self.assertEqual(
            self.facade.conversation_context(
                session_id,
                card_id,
            ).pending_proposal,
            pending,
        )
        with InMemoryUnitOfWork(self.database) as uow:
            proposal = uow.records.get(
                "analyst_answer_proposal",
                pending.proposal_id,
            )
        self.assertEqual(proposal.payload["status"], "pending")

    async def test_cancel_during_tool_handoff_is_forwarded_to_child(self) -> None:
        saved, skeleton_id, session_id, card_id = await self._populated_card()
        sequence = self.facade.append(
            session_id,
            SessionEventKind.ANALYST,
            "Продолжай.",
            {"author": "Аналитик"},
        )
        revision = self.facade.card(card_id).revision
        agent = Mock()
        agent.decide = AsyncMock(
            return_value=ConversationTurnDecision(
                ConversationTurnKind.TOOL_CALL,
                "Продолжаю сохранённую стадию.",
                ConversationToolCall(
                    ConversationAction.RESUME,
                    {"expected_revision": revision},
                ),
            )
        )
        self.facade.conversation_agent = agent
        child_cancelled = threading.Event()
        operation_holder: dict[str, object] = {}

        async def child() -> None:
            if child_cancelled.is_set():
                raise AttemptDiscardedError("Дочерняя операция отменена")

        def dispatch(*_args: object) -> ConversationToolResult:
            cancellation = threading.Thread(
                target=lambda: operation_holder["operation"].cancel(),
            )
            cancellation.start()
            cancellation.join(timeout=1)
            return ConversationToolResult(
                ConversationAction.RESUME,
                ConversationEffect.EXPENSIVE,
                "Продолжаю сохранённую стадию.",
                child(),
                child_cancelled.set,
            )

        self.facade.dispatch_conversation_tool = dispatch  # type: ignore[method-assign]
        operation = self.facade.conversation_turn(
            saved,
            skeleton_id,
            session_id,
            card_id,
            f"MSG_{sequence:06d}",
        )
        operation_holder["operation"] = operation

        with self.assertRaises(AttemptDiscardedError):
            await operation.awaitable
        self.assertTrue(child_cancelled.is_set())
        self.assertFalse(
            any(
                event.metadata.get("completed")
                for event in self.facade.history(session_id)
            )
        )

    async def test_conversation_analyst_answer_creates_proposal_without_mutation(
        self,
    ) -> None:
        saved, skeleton_id, session_id, card_id = await self._populated_card()
        sequence = self.facade.append(
            session_id,
            SessionEventKind.ANALYST,
            "Использовать контрольное значение 80.",
            {"author": "Аналитик"},
        )
        message_id = f"MSG_{sequence:06d}"

        result = self.facade.dispatch_conversation_tool(
            saved,
            skeleton_id,
            session_id,
            card_id,
            message_id,
            ConversationToolCall(
                ConversationAction.SUBMIT_ANALYST_ANSWER,
                {
                    "gap_id": "GAP_0001",
                    "values": [
                        {
                            "path": "test.control_values",
                            "value": ["80"],
                        }
                    ],
                    "expected_revision": 1,
                },
            ),
        )

        with InMemoryUnitOfWork(self.database) as uow:
            card = uow.cards.get(card_id)
            proposals = uow.records.list_kind("analyst_answer_proposal")
        self.assertIn("Контрольные значения", result.text)
        self.assertIn("80", result.text)
        self.assertNotIn("test.control_values", result.text)
        self.assertEqual(card.revision, 1)
        self.assertEqual(card.gaps["GAP_0001"].status.value, "открыт")
        self.assertIsNone(card.field("test.control_values").value)
        self.assertFalse(
            any(item.message_id == message_id for item in card.evidence.values())
        )
        self.assertEqual(len(proposals), 1)
        self.assertEqual(proposals[0].payload["status"], "pending")
        self.assertEqual(proposals[0].payload["source_message_id"], message_id)
        self.assertEqual(
            proposals[0].payload["values"],
            [{"path": "test.control_values", "value": ["80"]}],
        )
        self.facade.reconciler.assert_consistent(saved.selection_id)

    async def test_confirmed_analyst_proposal_applies_exact_values_once(
        self,
    ) -> None:
        saved, skeleton_id, session_id, card_id = await self._populated_card()
        source_sequence = self.facade.append(
            session_id,
            SessionEventKind.ANALYST,
            "Использовать контрольное значение 80.",
            {"author": "Аналитик"},
        )
        source_message_id = f"MSG_{source_sequence:06d}"
        self.facade.dispatch_conversation_tool(
            saved,
            skeleton_id,
            session_id,
            card_id,
            source_message_id,
            ConversationToolCall(
                ConversationAction.SUBMIT_ANALYST_ANSWER,
                {
                    "gap_id": "GAP_0001",
                    "values": [
                        {"path": "test.control_values", "value": ["80"]}
                    ],
                    "expected_revision": 1,
                },
            ),
        )
        pending = self.facade.conversation_context(
            session_id,
            card_id,
        ).pending_proposal
        confirmation_sequence = self.facade.append(
            session_id,
            SessionEventKind.ANALYST,
            "Да, подтверждаю эту интерпретацию.",
            {"author": "Аналитик"},
        )
        confirmation_message_id = f"MSG_{confirmation_sequence:06d}"

        result = self.facade.dispatch_conversation_tool(
            saved,
            skeleton_id,
            session_id,
            card_id,
            confirmation_message_id,
            ConversationToolCall(
                ConversationAction.CONFIRM_ANALYST_ANSWER,
                {
                    "proposal_id": pending.proposal_id,
                    "expected_revision": 1,
                    "confirmation_message_id": confirmation_message_id,
                },
            ),
        )

        with InMemoryUnitOfWork(self.database) as uow:
            card = uow.cards.get(card_id)
            proposal = uow.records.get(
                "analyst_answer_proposal",
                pending.proposal_id,
            )
        self.assertIn("подтверждена", result.text)
        self.assertEqual(card.revision, 2)
        self.assertEqual(card.field("test.control_values").value, ["80"])
        self.assertEqual(card.gaps["GAP_0001"].status.value, "закрыт")
        evidence = next(
            item
            for item in card.evidence.values()
            if item.message_id == source_message_id
        )
        self.assertEqual(evidence.quote, "Использовать контрольное значение 80.")
        self.assertEqual(
            card.field("test.control_values").status.value,
            "подтверждено аналитиком",
        )
        self.assertEqual(len(card.resolutions), 1)
        resolution = next(iter(card.resolutions.values()))
        self.assertEqual(resolution.proposal_id, pending.proposal_id)
        self.assertEqual(resolution.source_message_id, source_message_id)
        self.assertEqual(
            resolution.confirmation_message_id,
            confirmation_message_id,
        )
        self.assertEqual(resolution.gap_id, "GAP_0001")
        self.assertEqual(resolution.expected_revision, 1)
        self.assertEqual(
            resolution.values,
            (
                {
                    "path": "test.control_values",
                    "value": ["80"],
                },
            ),
        )
        self.assertEqual(proposal.payload["status"], "confirmed")
        self.assertEqual(
            proposal.payload["confirmation_message_id"],
            confirmation_message_id,
        )
        self.assertEqual(
            self.facade.conversation_context(
                session_id,
                card_id,
            ).pending_proposal,
            None,
        )
        with self.assertRaises(ValueError):
            self.facade.dispatch_conversation_tool(
                saved,
                skeleton_id,
                session_id,
                card_id,
                confirmation_message_id,
                ConversationToolCall(
                    ConversationAction.CONFIRM_ANALYST_ANSWER,
                    {
                        "proposal_id": pending.proposal_id,
                        "expected_revision": 2,
                        "confirmation_message_id": confirmation_message_id,
                    },
                ),
            )
        self.assertEqual(self.facade.card(card_id).revision, 2)
        diagnostics = self.facade.export_diagnostics(session_id, card_id)
        diagnostic_text = diagnostics.read_text(encoding="utf-8")
        self.assertIn("## Интерпретации ответов аналитика", diagnostic_text)
        self.assertIn(source_message_id, diagnostic_text)
        self.assertIn(confirmation_message_id, diagnostic_text)
        self.assertIn('"path": "test.control_values"', diagnostic_text)
        self.assertIn('"value": [', diagnostic_text)

    async def test_insufficient_confirmed_answer_updates_field_but_keeps_gap_open(
        self,
    ) -> None:
        saved, skeleton_id, session_id, card_id = await self._populated_card()
        with InMemoryUnitOfWork(self.database) as uow:
            card = uow.cards.get(card_id)
            state = card.snapshot()
            original = state.gaps[0]
            strict_gap = RelatedGap(
                gap_id=original.gap_id,
                card_id=card_id,
                question=(
                    "Какие конкретные данные (тэги/значения) передаются "
                    "в поле Data команды PUT DATA?"
                ),
                blocking_reason="Без конкретных Data тест невоспроизводим",
                allowed_paths=("test.command.data",),
                dependencies=(),
                closure_criterion="Заданы воспроизводимые Data",
                closure_contract=GapClosureContract(
                    requirements=(
                        GapPathClosure(
                            path="test.command.data",
                            accepted_forms=(
                                GapValueForm.EXACT_VALUE,
                                GapValueForm.FINITE_SET,
                                GapValueForm.DETERMINISTIC_RULE,
                            ),
                            residual_question=(
                                "Укажите точные байты, конечный набор или "
                                "детерминированное правило генерации Data."
                            ),
                        ),
                    ),
                ),
                resolution_mode=GapResolutionMode.DESIGN_DECISION,
            )
            uow.cards.save(
                TestCard.restore(replace(state, gaps=(strict_gap,)))
            )

        closure_context = self.facade.conversation_context(
            session_id,
            card_id,
        ).open_gap.closure_requirements
        self.assertEqual(
            closure_context[0].accepted_forms,
            ("exact", "finite_set", "deterministic_rule"),
        )

        source_sequence = self.facade.append(
            session_id,
            SessionEventKind.ANALYST,
            "пускай будут произвольные байты",
            {"author": "Аналитик"},
        )
        source_message_id = f"MSG_{source_sequence:06d}"
        proposal_result = self.facade.dispatch_conversation_tool(
            saved,
            skeleton_id,
            session_id,
            card_id,
            source_message_id,
            ConversationToolCall(
                ConversationAction.SUBMIT_ANALYST_ANSWER,
                {
                    "gap_id": "GAP_0001",
                    "values": [
                        {
                            "path": "test.command.data",
                            "value": {
                                "kind": "confirmed_value",
                                "value": "произвольные байты",
                            },
                        }
                    ],
                    "expected_revision": 1,
                },
            ),
        )
        proposal = self.facade.conversation_context(
            session_id,
            card_id,
        ).pending_proposal
        self.assertIn("останется открытым", proposal_result.text)
        self.assertIn("точные байты", proposal_result.text)

        confirmation_sequence = self.facade.append(
            session_id,
            SessionEventKind.ANALYST,
            "подтверждаю",
            {"author": "Аналитик"},
        )
        confirmation_message_id = f"MSG_{confirmation_sequence:06d}"
        confirmation_call = ConversationToolCall(
            ConversationAction.CONFIRM_ANALYST_ANSWER,
            {
                "proposal_id": proposal.proposal_id,
                "expected_revision": 1,
                "confirmation_message_id": confirmation_message_id,
            },
        )

        result = self.facade.dispatch_conversation_tool(
            saved,
            skeleton_id,
            session_id,
            card_id,
            confirmation_message_id,
            confirmation_call,
        )

        card = self.facade.card(card_id)
        self.assertIn("остаётся открытым", result.text)
        self.assertEqual(card.revision, 2)
        self.assertEqual(
            card.field("test.command.data").value,
            "произвольные байты",
        )
        self.assertEqual(
            card.field("test.command.data").status.value,
            "подтверждено аналитиком",
        )
        self.assertEqual(card.gaps["GAP_0001"].status.value, "открыт")
        self.assertEqual(
            card.gaps["GAP_0001"].closure_satisfied_paths,
            (),
        )
        self.assertEqual(len(card.resolutions), 1)

        retry = self.facade.dispatch_conversation_tool(
            saved,
            skeleton_id,
            session_id,
            card_id,
            confirmation_message_id,
            confirmation_call,
        )
        self.assertIn("уже применена", retry.text)
        self.assertEqual(self.facade.card(card_id).revision, 2)

        exact_sequence = self.facade.append(
            session_id,
            SessionEventKind.ANALYST,
            "Использовать точное значение 80.",
            {"author": "Аналитик"},
        )
        exact_message_id = f"MSG_{exact_sequence:06d}"
        self.facade.dispatch_conversation_tool(
            saved,
            skeleton_id,
            session_id,
            card_id,
            exact_message_id,
            ConversationToolCall(
                ConversationAction.SUBMIT_ANALYST_ANSWER,
                {
                    "gap_id": "GAP_0001",
                    "values": [
                        {
                            "path": "test.command.data",
                            "value": {"kind": "exact", "value": "80"},
                        }
                    ],
                    "expected_revision": 2,
                },
            ),
        )
        exact_proposal = self.facade.conversation_context(
            session_id,
            card_id,
        ).pending_proposal
        exact_confirmation_sequence = self.facade.append(
            session_id,
            SessionEventKind.ANALYST,
            "Да, подтверждаю точное значение.",
            {"author": "Аналитик"},
        )
        exact_confirmation_id = (
            f"MSG_{exact_confirmation_sequence:06d}"
        )
        self.facade.dispatch_conversation_tool(
            saved,
            skeleton_id,
            session_id,
            card_id,
            exact_confirmation_id,
            ConversationToolCall(
                ConversationAction.CONFIRM_ANALYST_ANSWER,
                {
                    "proposal_id": exact_proposal.proposal_id,
                    "expected_revision": 2,
                    "confirmation_message_id": exact_confirmation_id,
                },
            ),
        )

        card = self.facade.card(card_id)
        self.assertEqual(card.revision, 3)
        self.assertEqual(card.field("test.command.data").value, "80")
        self.assertEqual(card.gaps["GAP_0001"].status.value, "закрыт")
        self.assertEqual(
            card.gaps["GAP_0001"].closure_satisfied_paths,
            ("test.command.data",),
        )
        diagnostics = self.facade.export_diagnostics(session_id, card_id)
        diagnostic_text = diagnostics.read_text(encoding="utf-8")
        self.assertIn('"closure_evaluation"', diagnostic_text)
        self.assertIn('"partially_resolved"', diagnostic_text)

    async def test_rejected_and_replaced_analyst_proposals_do_not_mutate_card(
        self,
    ) -> None:
        saved, skeleton_id, session_id, card_id = await self._populated_card()

        def propose(text: str, value: str) -> tuple[str, str]:
            sequence = self.facade.append(
                session_id,
                SessionEventKind.ANALYST,
                text,
                {"author": "Аналитик"},
            )
            message_id = f"MSG_{sequence:06d}"
            self.facade.dispatch_conversation_tool(
                saved,
                skeleton_id,
                session_id,
                card_id,
                message_id,
                ConversationToolCall(
                    ConversationAction.SUBMIT_ANALYST_ANSWER,
                    {
                        "gap_id": "GAP_0001",
                        "values": [
                            {
                                "path": "test.control_values",
                                "value": [value],
                            }
                        ],
                        "expected_revision": 1,
                    },
                ),
            )
            proposal_id = self.facade.conversation_context(
                session_id,
                card_id,
            ).pending_proposal.proposal_id
            return message_id, proposal_id

        _first_message_id, first_proposal_id = propose(
            "Возможно, использовать 80.",
            "80",
        )
        _second_message_id, second_proposal_id = propose(
            "Нет, лучше использовать 7F.",
            "7F",
        )
        rejection_sequence = self.facade.append(
            session_id,
            SessionEventKind.ANALYST,
            "Отклоняю и эту интерпретацию.",
            {"author": "Аналитик"},
        )
        rejection_message_id = f"MSG_{rejection_sequence:06d}"

        self.facade.dispatch_conversation_tool(
            saved,
            skeleton_id,
            session_id,
            card_id,
            rejection_message_id,
            ConversationToolCall(
                ConversationAction.REJECT_ANALYST_ANSWER,
                {
                    "proposal_id": second_proposal_id,
                    "expected_revision": 1,
                    "rejection_message_id": rejection_message_id,
                },
            ),
        )

        with InMemoryUnitOfWork(self.database) as uow:
            first = uow.records.get(
                "analyst_answer_proposal",
                first_proposal_id,
            )
            second = uow.records.get(
                "analyst_answer_proposal",
                second_proposal_id,
            )
        card = self.facade.card(card_id)
        self.assertEqual(first.payload["status"], "replaced")
        self.assertEqual(second.payload["status"], "rejected")
        self.assertEqual(
            second.payload["rejection_message_id"],
            rejection_message_id,
        )
        self.assertEqual(card.revision, 1)
        self.assertIsNone(card.field("test.control_values").value)
        self.assertIsNone(
            self.facade.conversation_context(
                session_id,
                card_id,
            ).pending_proposal
        )

    async def test_card_revision_invalidates_pending_analyst_proposal(self) -> None:
        saved, skeleton_id, session_id, card_id = await self._populated_card()
        source_sequence = self.facade.append(
            session_id,
            SessionEventKind.ANALYST,
            "Использовать 80.",
            {"author": "Аналитик"},
        )
        source_message_id = f"MSG_{source_sequence:06d}"
        self.facade.dispatch_conversation_tool(
            saved,
            skeleton_id,
            session_id,
            card_id,
            source_message_id,
            ConversationToolCall(
                ConversationAction.SUBMIT_ANALYST_ANSWER,
                {
                    "gap_id": "GAP_0001",
                    "values": [
                        {"path": "test.control_values", "value": ["80"]}
                    ],
                    "expected_revision": 1,
                },
            ),
        )
        proposal_id = self.facade.conversation_context(
            session_id,
            card_id,
        ).pending_proposal.proposal_id
        mode_sequence = self.facade.append(
            session_id,
            SessionEventKind.ANALYST,
            "Это внешнее значение стенда.",
            {"author": "Аналитик"},
        )
        self.facade.dispatch_conversation_tool(
            saved,
            skeleton_id,
            session_id,
            card_id,
            f"MSG_{mode_sequence:06d}",
            ConversationToolCall(
                ConversationAction.CHANGE_GAP_MODE,
                {
                    "gap_id": "GAP_0001",
                    "resolution_mode": "external_input",
                    "expected_revision": 1,
                },
            ),
        )

        context = self.facade.conversation_context(session_id, card_id)
        with InMemoryUnitOfWork(self.database) as uow:
            proposal = uow.records.get(
                "analyst_answer_proposal",
                proposal_id,
            )
        self.assertEqual(self.facade.card(card_id).revision, 2)
        self.assertIsNone(context.pending_proposal)
        self.assertNotIn(
            ConversationAction.CONFIRM_ANALYST_ANSWER,
            context.available_actions,
        )
        self.assertEqual(proposal.payload["status"], "invalidated")

    async def test_pending_analyst_proposal_survives_application_restart(
        self,
    ) -> None:
        saved, skeleton_id, session_id, card_id = await self._populated_card()
        sequence = self.facade.append(
            session_id,
            SessionEventKind.ANALYST,
            "Использовать контрольное значение 80.",
            {"author": "Аналитик"},
        )
        self.facade.dispatch_conversation_tool(
            saved,
            skeleton_id,
            session_id,
            card_id,
            f"MSG_{sequence:06d}",
            ConversationToolCall(
                ConversationAction.SUBMIT_ANALYST_ANSWER,
                {
                    "gap_id": "GAP_0001",
                    "values": [
                        {"path": "test.control_values", "value": ["80"]}
                    ],
                    "expected_revision": 1,
                },
            ),
        )
        before = self.facade.conversation_context(
            session_id,
            card_id,
        ).pending_proposal
        restarted = WorkbenchApplication(
            document=self.facade.document,
            run_dir=self.facade.run_dir,
            uow_factory=lambda: InMemoryUnitOfWork(self.database),
            workflow=self.facade.workflow,
            sessions=SessionService(
                uow_factory=lambda: InMemoryUnitOfWork(self.database)
            ),
            workers=self.facade.workers,
            next_id=self.next_id,
        )

        after = restarted.conversation_context(
            session_id,
            card_id,
        ).pending_proposal

        self.assertEqual(after, before)
        self.assertIn(
            ConversationAction.CONFIRM_ANALYST_ANSWER,
            restarted.conversation_context(
                session_id,
                card_id,
            ).available_actions,
        )
        self.assertEqual(restarted.card(card_id).revision, 1)

    async def test_different_open_gap_invalidates_pending_proposal(
        self,
    ) -> None:
        saved, skeleton_id, session_id, card_id = await self._populated_card()
        sequence = self.facade.append(
            session_id,
            SessionEventKind.ANALYST,
            "Использовать 80.",
            {"author": "Аналитик"},
        )
        message_id = f"MSG_{sequence:06d}"
        self.facade.dispatch_conversation_tool(
            saved,
            skeleton_id,
            session_id,
            card_id,
            message_id,
            ConversationToolCall(
                ConversationAction.SUBMIT_ANALYST_ANSWER,
                {
                    "gap_id": "GAP_0001",
                    "values": [
                        {
                            "path": "test.control_values",
                            "value": ["80"],
                        }
                    ],
                    "expected_revision": 1,
                },
            ),
        )
        proposal_id = self.facade.conversation_context(
            session_id,
            card_id,
        ).pending_proposal.proposal_id

        pending = self.facade.analyst_proposals.pending(
            session_id=session_id,
            card_id=card_id,
            card_revision=1,
            open_gap_id="GAP_OTHER",
        )

        self.assertIsNone(pending)
        self.assertEqual(self.facade.card(card_id).revision, 1)
        with InMemoryUnitOfWork(self.database) as uow:
            proposal = uow.records.get(
                "analyst_answer_proposal",
                proposal_id,
            )
        self.assertEqual(proposal.payload["status"], "invalidated")

    async def test_conversation_mutation_rejects_stale_revision(self) -> None:
        saved, skeleton_id, session_id, card_id = await self._populated_card()
        sequence = self.facade.append(
            session_id,
            SessionEventKind.ANALYST,
            "Использовать 80.",
            {"author": "Аналитик"},
        )

        with self.assertRaisesRegex(ValueError, "Устаревшая ревизия"):
            self.facade.dispatch_conversation_tool(
                saved,
                skeleton_id,
                session_id,
                card_id,
                f"MSG_{sequence:06d}",
                ConversationToolCall(
                    ConversationAction.SUBMIT_ANALYST_ANSWER,
                    {
                        "gap_id": "GAP_0001",
                        "values": [
                            {
                                "path": "test.control_values",
                                "value": ["80"],
                            }
                        ],
                        "expected_revision": 0,
                    },
                ),
            )

        self.assertEqual(self.facade.card(card_id).revision, 1)

    async def test_analyst_answer_application_guard_rejects_wire_shape_and_path(
        self,
    ) -> None:
        saved, skeleton_id, session_id, card_id = await self._populated_card()
        sequence = self.facade.append(
            session_id,
            SessionEventKind.ANALYST,
            "Может Get Data подойдёт?",
            {"author": "Аналитик"},
        )
        message_id = f"MSG_{sequence:06d}"
        baseline = self.facade.card(card_id)
        baseline_evidence = dict(baseline.evidence)
        invalid_values = (
            [{"field": "test.control_values", "value": ["80"]}],
            [{"path": "test.observation.method", "value": "Get Data"}],
            [
                {
                    "path": "test.control_values",
                    "value": {
                        "kind": "finite_set",
                        "values": "80",
                    },
                }
            ],
        )

        for values in invalid_values:
            with self.subTest(values=values):
                with self.assertRaises(
                    (
                        ValueError,
                        DomainValidationError,
                        PathNotAllowedError,
                    )
                ):
                    self.facade.dispatch_conversation_tool(
                        saved,
                        skeleton_id,
                        session_id,
                        card_id,
                        message_id,
                        ConversationToolCall(
                            ConversationAction.SUBMIT_ANALYST_ANSWER,
                            {
                                "gap_id": "GAP_0001",
                                "values": values,
                                "expected_revision": 1,
                            },
                        ),
                    )

                card = self.facade.card(card_id)
                self.assertEqual(card.revision, 1)
                self.assertEqual(dict(card.evidence), baseline_evidence)
                self.assertEqual(
                    card.gaps["GAP_0001"].status.value,
                    "открыт",
                )

    async def test_change_mode_and_leave_gap_require_analyst_provenance(
        self,
    ) -> None:
        saved, skeleton_id, session_id, card_id = await self._populated_card()
        mode_sequence = self.facade.append(
            session_id,
            SessionEventKind.ANALYST,
            "Это проектное решение, а не факт спецификации.",
            {"author": "Аналитик"},
        )
        mode_message_id = f"MSG_{mode_sequence:06d}"
        self.facade.dispatch_conversation_tool(
            saved,
            skeleton_id,
            session_id,
            card_id,
            mode_message_id,
            ConversationToolCall(
                ConversationAction.CHANGE_GAP_MODE,
                {
                    "gap_id": "GAP_0001",
                    "resolution_mode": "design_decision",
                    "expected_revision": 1,
                },
            ),
        )
        self.assertIs(
            self.facade.card(card_id).gaps["GAP_0001"].resolution_mode,
            GapResolutionMode.DESIGN_DECISION,
        )
        leave_sequence = self.facade.append(
            session_id,
            SessionEventKind.ANALYST,
            "Оставь пробел, решение уточним у разработчиков.",
            {"author": "Аналитик"},
        )
        leave_message_id = f"MSG_{leave_sequence:06d}"

        with self.assertRaisesRegex(ValueError, "требует подтверждение"):
            self.facade.dispatch_conversation_tool(
                saved,
                skeleton_id,
                session_id,
                card_id,
                leave_message_id,
                ConversationToolCall(
                    ConversationAction.LEAVE_GAP,
                    {
                        "gap_id": "GAP_0001",
                        "reason": "Уточнить у разработчиков",
                        "expected_revision": 2,
                        "confirmation_message_id": "MSG_OTHER",
                    },
                ),
            )

        self.facade.dispatch_conversation_tool(
            saved,
            skeleton_id,
            session_id,
            card_id,
            leave_message_id,
            ConversationToolCall(
                ConversationAction.LEAVE_GAP,
                {
                    "gap_id": "GAP_0001",
                    "reason": "Уточнить у разработчиков",
                    "expected_revision": 2,
                    "confirmation_message_id": leave_message_id,
                },
            ),
        )
        with InMemoryUnitOfWork(self.database) as uow:
            card = uow.cards.get(card_id)
            decision = uow.records.get(
                "gap_leave_decision",
                f"{card_id}:GAP_0001:r{card.revision:06d}",
            )
        self.assertEqual(card.gaps["GAP_0001"].status.value, "оставлен открытым")
        self.assertEqual(
            decision.payload["analyst_message_id"],
            leave_message_id,
        )
        self.assertFalse(
            any(item.message_id == leave_message_id for item in card.evidence.values())
        )

    async def test_card_decision_requires_current_message_confirmation(self) -> None:
        saved, skeleton_id, session_id, card_id = await self._populated_card()
        sequence = self.facade.append(
            session_id,
            SessionEventKind.ANALYST,
            "Включи карточку неполной.",
            {"author": "Аналитик"},
        )
        message_id = f"MSG_{sequence:06d}"

        with self.assertRaisesRegex(ValueError, "требует подтверждение"):
            self.facade.dispatch_conversation_tool(
                saved,
                skeleton_id,
                session_id,
                card_id,
                message_id,
                ConversationToolCall(
                    ConversationAction.INCLUDE_CARD,
                    {
                        "expected_revision": 1,
                        "confirmation_message_id": "MSG_OTHER",
                    },
                ),
            )
        self.assertIsNone(self.facade.card(card_id).decision)

        result = self.facade.dispatch_conversation_tool(
            saved,
            skeleton_id,
            session_id,
            card_id,
            message_id,
            ConversationToolCall(
                ConversationAction.INCLUDE_CARD,
                {
                    "expected_revision": 1,
                    "confirmation_message_id": message_id,
                },
            ),
        )

        self.assertIn("включить неполной", result.text)
        self.assertEqual(
            self.facade.card(card_id).decision.kind.value,
            "включить неполной",
        )

    async def test_conversation_mutation_recovers_after_checkpoint_failure(
        self,
    ) -> None:
        saved, skeleton_id, session_id, card_id = await self._populated_card()
        sequence = self.facade.append(
            session_id,
            SessionEventKind.ANALYST,
            "Считать это проектным решением.",
            {"author": "Аналитик"},
        )
        message_id = f"MSG_{sequence:06d}"
        execute = self.workflow.execute
        failed = False

        def fail_once(
            thread_id: str,
            command: WorkflowCommand,
        ) -> WorkflowState:
            nonlocal failed
            if command.kind is CommandKind.REFINE_CARD and not failed:
                failed = True
                raise RuntimeError("checkpoint unavailable")
            return execute(thread_id, command)

        self.workflow.execute = fail_once  # type: ignore[method-assign]
        with self.assertRaisesRegex(RuntimeError, "checkpoint unavailable"):
            self.facade.dispatch_conversation_tool(
                saved,
                skeleton_id,
                session_id,
                card_id,
                message_id,
                ConversationToolCall(
                    ConversationAction.CHANGE_GAP_MODE,
                    {
                        "gap_id": "GAP_0001",
                        "resolution_mode": "design_decision",
                        "expected_revision": 1,
                    },
                ),
            )

        with InMemoryUnitOfWork(self.database) as uow:
            attempt = uow.attempts.list_for_session(session_id)[-1]
        self.assertIs(attempt.status, AttemptStatus.COMPLETED)
        self.assertIsNotNone(
            self.workflow.current_state(saved.selection_id).active_attempt
        )

        self.workflow.execute = execute  # type: ignore[method-assign]
        recovered = self.facade.recover_workflows()

        self.assertIn(attempt.attempt_id, recovered)
        self.assertIsNone(
            self.workflow.current_state(saved.selection_id).active_attempt
        )
        self.facade.reconciler.assert_consistent(saved.selection_id)

    async def _orphan_card(self) -> tuple[object, str, str]:
        source = document()
        saved = self.facade.save_selection(
            "section-1",
            source.select(SourcePosition(1, 1), SourcePosition(1, 2)),
        )
        decomposition_result = await self.facade.decompose(saved).awaitable
        card_id = self.facade.take_skeleton(
            saved.selection_id,
            decomposition_result.skeleton_ids[0],
        )
        session_id = self.facade.open_card_session(saved.selection_id, card_id)
        await self.facade.populate(
            saved,
            decomposition_result.skeleton_ids[0],
            session_id,
            card_id,
        ).awaitable

        with InMemoryUnitOfWork(self.database) as uow:
            card = uow.cards.get(card_id)
            gap_id = next(iter(card.gaps))
            card.apply(
                CardMutation(
                    fields={
                        "requirement.behavior": ContentField.unknown()
                    },
                    gap_progress={
                        gap_id: card.gaps[gap_id].allowed_paths
                    },
                    resolved_gap_ids=(gap_id,),
                )
            )
            uow.cards.save(card)
        self.workflow.execute(
            saved.selection_id,
            WorkflowCommand(
                CommandKind.BEGIN_ATTEMPT,
                {
                    "attempt_id": "ATTEMPT_ORPHAN_SETUP",
                    "attempt_kind": "refinement",
                    "card_id": card_id,
                },
            ),
        )
        self.workflow.execute(
            saved.selection_id,
            WorkflowCommand(
                CommandKind.REFINE_CARD,
                {
                    "card_id": card_id,
                    "revision": 2,
                    "outcome": "updated",
                    "gap_statuses": {gap_id: "resolved"},
                },
            ),
        )
        return saved, card_id, session_id

    async def test_coverage_repair_syncs_card_workflow_and_session(self) -> None:
        saved, card_id, session_id = await self._orphan_card()

        self.assertEqual(self.facade.continuation(session_id), "coverage_repair")
        result = self.facade.repair_card_coverage(session_id, card_id)

        with InMemoryUnitOfWork(self.database) as uow:
            card = uow.cards.get(card_id)
            session = uow.sessions.get(session_id)
            attempt = uow.attempts.list_for_session(session_id)[-1]
        state = self.workflow.current_state(saved.selection_id)
        self.assertEqual(result.revision, 3)
        self.assertEqual(
            card.gaps[result.open_gap_ids[0]].allowed_paths,
            ("requirement.behavior",),
        )
        self.assertEqual(state.cards[card_id].revision, 3)
        self.assertEqual(
            dict(state.cards[card_id].gaps)[result.open_gap_ids[0]],
            "open",
        )
        self.assertEqual(session.payload["continuation"], "gap_investigation")
        self.assertEqual(attempt.status, AttemptStatus.COMPLETED)
        self.assertEqual(attempt.stage, "coverage_repair")
        self.facade.reconciler.assert_consistent(saved.selection_id)

    async def test_coverage_repair_recovers_after_root_graph_apply_failure(self) -> None:
        saved, card_id, session_id = await self._orphan_card()
        execute = self.workflow.execute

        def fail_repair(thread_id: str, command: WorkflowCommand) -> WorkflowState:
            if (
                command.kind is CommandKind.REFINE_CARD
                and command.payload.get("outcome") == "gaps_created"
            ):
                raise RuntimeError("checkpoint unavailable")
            return execute(thread_id, command)

        self.workflow.execute = fail_repair  # type: ignore[method-assign]
        with self.assertRaisesRegex(RuntimeError, "checkpoint unavailable"):
            self.facade.repair_card_coverage(session_id, card_id)

        with InMemoryUnitOfWork(self.database) as uow:
            attempt = uow.attempts.list_for_session(session_id)[-1]
        state = self.workflow.current_state(saved.selection_id)
        self.assertEqual(attempt.status, AttemptStatus.COMPLETED)
        self.assertIsNotNone(state.active_attempt)

        self.workflow.execute = execute  # type: ignore[method-assign]
        self.assertEqual(
            self.facade.recover_workflows(),
            (attempt.attempt_id,),
        )
        self.facade.reconciler.assert_consistent(saved.selection_id)
        self.assertEqual(self.facade.continuation(session_id), "gap_investigation")

    async def test_ensure_card_session_reuses_history_without_duplicate(self) -> None:
        source = document()
        saved = self.facade.save_selection(
            "section-1",
            source.select(SourcePosition(1, 1), SourcePosition(1, 2)),
        )
        decomposition_result = await self.facade.decompose(saved).awaitable
        card_id = self.facade.take_skeleton(
            saved.selection_id,
            decomposition_result.skeleton_ids[0],
        )
        session_id = self.facade.open_card_session(saved.selection_id, card_id)
        self.facade.append(
            session_id,
            SessionEventKind.WORKBENCH,
            "Сохранённое событие",
        )

        reopened_id, created = self.facade.ensure_card_session(
            saved.selection_id,
            card_id,
        )

        self.assertEqual(reopened_id, session_id)
        self.assertFalse(created)
        self.assertEqual(len(self.database.sessions), 1)
        self.assertEqual(
            [event.text for event in self.facade.history(reopened_id)],
            ["Сохранённое событие"],
        )

    async def test_restart_clears_root_attempt_already_cancelled_in_domain_store(self) -> None:
        source = document()
        saved = self.facade.save_selection(
            "section-1",
            source.select(SourcePosition(1, 1), SourcePosition(1, 2)),
        )
        result = await self.facade.decompose(saved).awaitable
        card_id = self.facade.take_skeleton(saved.selection_id, result.skeleton_ids[0])
        self.workflow.execute(
            saved.selection_id,
            WorkflowCommand(
                CommandKind.BEGIN_ATTEMPT,
                {
                    "attempt_id": "ATTEMPT_INTERRUPTED",
                    "attempt_kind": "prompt_2",
                    "card_id": card_id,
                },
            ),
        )
        with InMemoryUnitOfWork(self.database) as uow:
            uow.attempts.save(
                AttemptRecord(
                    "ATTEMPT_INTERRUPTED",
                    "SESSION_1",
                    "prompt_2",
                    AttemptStatus.CANCELLED,
                    {},
                    datetime.now(UTC),
                )
            )

        recovered = self.facade.recover_workflows()

        self.assertEqual(recovered, ("ATTEMPT_INTERRUPTED",))
        state = self.workflow.current_state(saved.selection_id)
        self.assertIsNone(state.active_attempt)
        self.assertIn("ATTEMPT_INTERRUPTED", state.cancelled_attempt_ids)
        self.facade.reconciler.assert_consistent(saved.selection_id)
        self.facade.reconciler.assert_consistent(saved.selection_id)

    async def test_restart_applies_completed_domain_result_to_root_checkpoint(self) -> None:
        source = document()
        saved = self.facade.save_selection(
            "section-1",
            source.select(SourcePosition(1, 1), SourcePosition(1, 2)),
        )
        result = await self.facade.decompose(saved).awaitable
        skeleton_id = result.skeleton_ids[0]
        card_id = self.facade.take_skeleton(saved.selection_id, skeleton_id)
        session_id = self.facade.open_card_session(saved.selection_id, card_id)
        attempt_id = "ATTEMPT_AFTER_DOMAIN_COMMIT"
        self.workflow.execute(
            saved.selection_id,
            WorkflowCommand(
                CommandKind.BEGIN_ATTEMPT,
                {
                    "attempt_id": attempt_id,
                    "attempt_kind": "prompt_2",
                    "card_id": card_id,
                },
            ),
        )
        worker = self.facade.workers.populate(
            saved,
            skeleton_id,
            session_id,
            card_id,
            attempt_id,
        )
        await worker.awaitable

        self.assertEqual(
            self.workflow.current_state(saved.selection_id).active_attempt.attempt_id,
            attempt_id,
        )
        recovered = self.facade.recover_workflows()

        self.assertEqual(recovered, (attempt_id,))
        state = self.workflow.current_state(saved.selection_id)
        self.assertIsNone(state.active_attempt)
        self.assertEqual(state.cards[card_id].revision, 1)
        self.assertTrue(state.cards[card_id].populated)
        self.facade.reconciler.assert_consistent(saved.selection_id)

    async def test_checkpoint_domain_divergence_blocks_continuation(self) -> None:
        source = document()
        saved = self.facade.save_selection(
            "section-1",
            source.select(SourcePosition(1, 1), SourcePosition(1, 2)),
        )
        result = await self.facade.decompose(saved).awaitable
        card_id = self.facade.take_skeleton(saved.selection_id, result.skeleton_ids[0])
        self.workflow.execute(
            saved.selection_id,
            WorkflowCommand(
                CommandKind.BEGIN_ATTEMPT,
                {
                    "attempt_id": "ATTEMPT_DIVERGED",
                    "attempt_kind": "prompt_2",
                    "card_id": card_id,
                },
            ),
        )
        self.workflow.execute(
            saved.selection_id,
            WorkflowCommand(
                CommandKind.APPLY_ATTEMPT_RESULT,
                {
                    "attempt_id": "ATTEMPT_DIVERGED",
                    "revision": 1,
                    "gap_statuses": {},
                    "outcome": "populated",
                },
            ),
        )

        with self.assertRaisesRegex(WorkflowConsistencyError, "карточка"):
            self.facade.reconciler.assert_consistent(saved.selection_id)

    async def test_empty_checkpoint_is_rebuilt_from_canonical_domain_state(self) -> None:
        source = document()
        saved = self.facade.save_selection(
            "section-1",
            source.select(SourcePosition(1, 1), SourcePosition(1, 2)),
        )
        result = await self.facade.decompose(saved).awaitable
        card_id = self.facade.take_skeleton(saved.selection_id, result.skeleton_ids[0])
        session_id = self.facade.open_card_session(saved.selection_id, card_id)
        await self.facade.populate(
            saved,
            result.skeleton_ids[0],
            session_id,
            card_id,
        ).awaitable
        self.facade.include_card(card_id)
        await self.facade.review_selection(saved.selection_id).awaitable

        restored_workflow = RecordingWorkflowRuntime()
        restored = WorkbenchApplication(
            document=document(),
            run_dir=Path(self.temporary.name),
            uow_factory=lambda: InMemoryUnitOfWork(self.database),
            workflow=restored_workflow,
            sessions=self.facade.sessions,
            workers=self.facade.workers,
            next_id=self.next_id,
        )

        restored.recover_workflows()

        state = restored_workflow.current_state(saved.selection_id)
        self.assertEqual(state.selection_id, saved.selection_id)
        self.assertEqual(state.cards[card_id].revision, 1)
        self.assertEqual(state.cards[card_id].decision, "include_incomplete")
        self.assertIsNotNone(state.range_review)
        restored.reconciler.assert_consistent(saved.selection_id)

    async def test_sync_worker_failure_clears_graph_attempt_and_keeps_retry_route(self) -> None:
        source = document()
        saved = self.facade.save_selection(
            "section-1",
            source.select(SourcePosition(1, 1), SourcePosition(1, 2)),
        )
        result = await self.facade.decompose(saved).awaitable
        card_id = self.facade.take_skeleton(saved.selection_id, result.skeleton_ids[0])
        session_id = self.facade.open_card_session(saved.selection_id, card_id)
        self.facade.workers.populate = Mock(
            side_effect=RuntimeError("worker configuration failed")
        )

        with self.assertRaisesRegex(RuntimeError, "configuration"):
            self.facade.populate(
                saved,
                result.skeleton_ids[0],
                session_id,
                card_id,
            )

        state = self.workflow.current_state(saved.selection_id)
        self.assertIsNone(state.active_attempt)
        self.assertEqual(state.failed_attempt_ids, ("ATTEMPT_0002",))
        with InMemoryUnitOfWork(self.database) as uow:
            session = uow.sessions.get(session_id)
        self.assertEqual(session.payload["continuation"], "population")
        self.assertIsNone(session.payload["active_intent"])
        self.assertEqual(session.current_stage, "первоначальное заполнение не запущено")

    async def test_required_prompt_sequence_reaches_graph_export_gate(self) -> None:
        source = document()
        saved = self.facade.save_selection(
            "section-1",
            source.select(SourcePosition(1, 1), SourcePosition(1, 2)),
        )
        decomposition_result = await self.facade.decompose(saved).awaitable
        card_id = self.facade.take_skeleton(
            saved.selection_id,
            decomposition_result.skeleton_ids[0],
        )
        session_id = self.facade.open_card_session(saved.selection_id, card_id)

        await self.facade.populate(
            saved,
            decomposition_result.skeleton_ids[0],
            session_id,
            card_id,
        ).awaitable
        with InMemoryUnitOfWork(self.database) as uow:
            session = uow.sessions.get(session_id)
        self.assertEqual(session.current_stage, "первоначальное заполнение завершено")
        self.assertEqual(session.payload["continuation"], "gap_investigation")
        self.assertIsNone(session.payload["active_intent"])
        self.facade.include_card(card_id)
        await self.facade.review_selection(saved.selection_id).awaitable
        exported = self.facade.export_full()

        state = self.workflow.current_state(saved.selection_id)
        self.assertTrue(state.export_allowed)
        self.assertEqual(state.cards[card_id].revision, 1)
        self.assertEqual(state.cards[card_id].decision, "include_incomplete")
        self.assertTrue(exported.exists())
        self.assertEqual(
            [item.kind for item in self.workflow.commands],
            [
                CommandKind.CONFIRM_SELECTION,
                CommandKind.BEGIN_ATTEMPT,
                CommandKind.APPLY_DECOMPOSITION,
                CommandKind.TAKE_SKELETON,
                CommandKind.BEGIN_ATTEMPT,
                CommandKind.APPLY_ATTEMPT_RESULT,
                CommandKind.DECIDE_CARD,
                CommandKind.BEGIN_ATTEMPT,
                CommandKind.SAVE_RANGE_REVIEW,
                CommandKind.REQUEST_EXPORT,
            ],
        )

    async def test_conversation_refinement_waits_for_exact_confirmation(
        self,
    ) -> None:
        saved, skeleton_id, session_id, card_id = await self._populated_card()
        leave_sequence = self.facade.append(
            session_id,
            SessionEventKind.ANALYST,
            "Оставить текущий пробел открытым.",
            {"author": "Аналитик"},
        )
        leave_message_id = f"MSG_{leave_sequence:06d}"
        self.facade.dispatch_conversation_tool(
            saved,
            skeleton_id,
            session_id,
            card_id,
            leave_message_id,
            ConversationToolCall(
                ConversationAction.LEAVE_GAP,
                {
                    "gap_id": "GAP_0001",
                    "reason": "Отдельное решение аналитика",
                    "expected_revision": 1,
                    "confirmation_message_id": leave_message_id,
                },
            ),
        )
        source_sequence = self.facade.append(
            session_id,
            SessionEventKind.ANALYST,
            "Замени воздействие на отправку GET DATA.",
            {"author": "Аналитик"},
        )
        source_message_id = f"MSG_{source_sequence:06d}"
        registry = TypedToolRegistry()
        from pmi_generator.workbench.application.refinement import (
            refinement_tool,
        )

        registry.register(refinement_tool())
        refinement_runtime = LlmToolRuntime(
            transport=ScriptedLlmTransport(
                [
                    RawCompletion(
                        "tool_calls",
                        (
                            {
                                "id": "refinement-proposal",
                                "name": "submit_card_refinement",
                                "arguments": {
                                    "outcome": "updated",
                                    "updates": [
                                        {
                                            "path": "test.action",
                                            "value": "Отправить GET DATA",
                                            "evidence_id": None,
                                            "analyst_message_id": (
                                                source_message_id
                                            ),
                                        }
                                    ],
                                    "gaps": [],
                                    "reason": "Уточнение аналитика",
                                },
                            },
                        ),
                        {},
                        "fake",
                    )
                ]
            ),
            tools=registry,
            uow_factory=lambda: InMemoryUnitOfWork(self.database),
        )
        self.facade.workers = ProductionPromptWorkers(
            document=document(),
            uow_factory=lambda: InMemoryUnitOfWork(self.database),
            policy=default_policy(),
            runtime_factory=lambda: refinement_runtime,
            retrieval_factory=lambda _selection: self.fail(
                "retrieval не должен вызываться"
            ),
            sessions=self.facade.sessions,
            next_id=self.next_id,
        )
        self.facade.conversation_agent = Mock(
            decide=AsyncMock(
                return_value=ConversationTurnDecision(
                    ConversationTurnKind.TOOL_CALL,
                    "Предлагаю уточнение карточки.",
                    ConversationToolCall(
                        ConversationAction.REFINE_CARD,
                        {"expected_revision": 2},
                    ),
                )
            )
        )

        await self.facade.conversation_turn(
            saved,
            skeleton_id,
            session_id,
            card_id,
            source_message_id,
        ).awaitable

        card = self.facade.card(card_id)
        self.assertEqual(card.revision, 2)
        self.assertNotEqual(
            card.field("test.action").value,
            "Отправить GET DATA",
        )
        pending = self.facade.conversation_context(
            session_id,
            card_id,
        ).pending_proposal
        self.assertEqual(pending.proposal_kind, "refinement")
        self.assertEqual(
            pending.values,
            (
                {
                    "path": "test.action",
                    "value": "Отправить GET DATA",
                },
            ),
        )
        restarted = WorkbenchApplication(
            document=self.facade.document,
            run_dir=self.facade.run_dir,
            uow_factory=lambda: InMemoryUnitOfWork(self.database),
            workflow=self.facade.workflow,
            sessions=SessionService(
                uow_factory=lambda: InMemoryUnitOfWork(self.database)
            ),
            workers=self.facade.workers,
            next_id=self.next_id,
        )
        self.assertEqual(
            restarted.conversation_context(
                session_id,
                card_id,
            ).pending_proposal,
            pending,
        )
        confirmation_sequence = self.facade.append(
            session_id,
            SessionEventKind.ANALYST,
            "Да, применяй именно это изменение.",
            {"author": "Аналитик"},
        )
        confirmation_message_id = f"MSG_{confirmation_sequence:06d}"
        restarted.conversation_agent = Mock(
            decide=AsyncMock(
                return_value=ConversationTurnDecision(
                    ConversationTurnKind.TOOL_CALL,
                    "Применяю подтверждённое изменение.",
                    ConversationToolCall(
                        ConversationAction.CONFIRM_ANALYST_ANSWER,
                        {
                            "proposal_id": pending.proposal_id,
                            "expected_revision": 2,
                            "confirmation_message_id": (
                                confirmation_message_id
                            ),
                        },
                    ),
                )
            )
        )

        await restarted.conversation_turn(
            saved,
            skeleton_id,
            session_id,
            card_id,
            confirmation_message_id,
        ).awaitable

        card = restarted.card(card_id)
        self.assertEqual(card.revision, 3)
        self.assertEqual(
            card.field("test.action").value,
            "Отправить GET DATA",
        )
        self.assertEqual(
            card.field("test.action").status.value,
            "подтверждено аналитиком",
        )
        resolution = next(
            item
            for item in card.resolutions.values()
            if item.proposal_id == pending.proposal_id
        )
        self.assertEqual(resolution.source_message_id, source_message_id)
        self.assertEqual(
            resolution.confirmation_message_id,
            confirmation_message_id,
        )
        self.assertEqual(resolution.expected_revision, 2)
        self.assertEqual(
            resolution.values,
            (
                {
                    "path": "test.action",
                    "value": "Отправить GET DATA",
                },
            ),
        )
        diagnostics = restarted.export_diagnostics(session_id, card_id)
        diagnostic_text = diagnostics.read_text(encoding="utf-8")
        self.assertIn(pending.proposal_id, diagnostic_text)
        self.assertIn(source_message_id, diagnostic_text)
        self.assertIn(confirmation_message_id, diagnostic_text)
        self.assertIn('"proposal_kind": "refinement"', diagnostic_text)
        self.assertIn('"path": "test.action"', diagnostic_text)
        self.assertIn('"value": "Отправить GET DATA"', diagnostic_text)

    async def test_decision_can_change_after_review_without_checkpoint_divergence(
        self,
    ) -> None:
        source = document()
        saved = self.facade.save_selection(
            "section-1",
            source.select(SourcePosition(1, 1), SourcePosition(1, 2)),
        )
        decomposition_result = await self.facade.decompose(saved).awaitable
        card_id = self.facade.take_skeleton(
            saved.selection_id,
            decomposition_result.skeleton_ids[0],
        )
        session_id = self.facade.open_card_session(saved.selection_id, card_id)
        await self.facade.populate(
            saved,
            decomposition_result.skeleton_ids[0],
            session_id,
            card_id,
        ).awaitable
        self.facade.include_card(card_id)
        await self.facade.review_selection(saved.selection_id).awaitable

        excluded = self.facade.exclude_card(card_id)

        self.assertEqual(excluded.kind.value, "исключить")
        self.assertTrue(self.facade.workspace(saved.selection_id).review_stale)
        self.assertIsNone(self.workflow.current_state(saved.selection_id).range_review)
        self.facade.reconciler.assert_consistent(saved.selection_id)

        included = self.facade.include_card(card_id)

        self.assertEqual(included.kind.value, "включить неполной")
        self.assertTrue(self.facade.workspace(saved.selection_id).review_stale)
        self.facade.reconciler.assert_consistent(saved.selection_id)


if __name__ == "__main__":
    unittest.main()
