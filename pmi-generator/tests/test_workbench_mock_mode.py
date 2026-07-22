from __future__ import annotations

import asyncio
import json
import os
import tempfile
import unittest
from dataclasses import replace
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from prompt_toolkit.formatted_text.utils import fragment_list_to_text
from prompt_toolkit.output import DummyOutput

from pmi_generator.workbench.application.conversation import (
    ConversationAction,
    ConversationContext,
    ConversationGapContext,
    ConversationProposalContext,
    ConversationToolCall,
)
from pmi_generator.workbench.application.range_workspace import (
    RangeWorkspaceState,
    WorkspaceItem,
)
from pmi_generator.workbench.application.state import StoredRecord
from pmi_generator.workbench.application.prompting import PromptId, default_policy
from pmi_generator.workbench.application.source import SavedSelection
from pmi_generator.workbench.domain import (
    ExecutionMode,
    SourceDocument,
    SourcePage,
    SourcePosition,
    SourceSection,
)
from pmi_generator.workbench.infrastructure.mock_mode import (
    MockLlmTransport,
    MockRetrieval,
)
from pmi_generator.workbench.presentation.decomposition import (
    SkeletonDetailScreen,
    SkeletonListScreen,
)
from pmi_generator.workbench.presentation.range_workspace import (
    RangeWorkspaceScreen,
)
from pmi_generator.workbench.presentation.result import ResultScreen
from pmi_generator.workbench.presentation.selection_review import (
    SelectionReviewScreen,
)
from pmi_generator.workbench.presentation.session import TerminalSessionShell
from pmi_generator.workbench.presentation.source import (
    ConfirmationScreen,
    SelectionScreen,
    StructureScreen,
)
from pmi_generator.workbench.presentation.startup import render_startup
from pmi_generator.workbench.presentation.terminal import TerminalWorkbench
from pmi_generator.workbench.application.gap_investigation import (
    RetrievalBudgetPolicy,
)
from pmi_generator.workbench.application.llm import AttemptDiscardedError
from pmi_generator.workbench.application.session import SessionEventKind


def document(
    execution_mode: ExecutionMode = ExecutionMode.PRODUCTION,
) -> SourceDocument:
    metadata = replace(
        SourceDocument(
            pages=(SourcePage(1, "1", ("metadata",)),),
            sections=(),
        ).metadata,
        execution_mode=execution_mode,
    )
    return SourceDocument(
        pages=(
            SourcePage(
                7,
                "A-7",
                (
                    "Произвольная первая строка PDF",
                    "Нейтральное содержание без известных заголовков",
                    "Произвольная последняя строка PDF",
                ),
            ),
        ),
        sections=(
            SourceSection(
                "page-0007",
                "",
                "Страница A-7",
                ("Страница A-7",),
                (7,),
            ),
        ),
        metadata=metadata,
    )


def saved_selection() -> SavedSelection:
    source = document()
    return SavedSelection(
        "SELECTION_MOCK",
        "page-0007",
        source.select(SourcePosition(7, 2), SourcePosition(7, 3)),
    )


def schemas(call: object) -> list[dict[str, object]]:
    return [
        {"function": {"name": name}}
        for name in call.allowed_tools
    ]


class MockLlmTransportTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.policy = default_policy()
        self.transport = MockLlmTransport(delay=0)

    def test_gap_research_uses_conversation_research_question(self) -> None:
        tool, arguments = MockLlmTransport._gap_research(
            {
                "gap": {"question": "Исходный вопрос"},
                "research_question": {
                    "text": "Можно ли наблюдать результат через GET DATA?"
                },
                "observations": [],
            }
        )

        self.assertEqual(tool, "ask_lightrag")
        self.assertEqual(
            arguments["question"],
            "Можно ли наблюдать результат через GET DATA? [mock]",
        )

    def test_empty_window_returns_local_context_result(self) -> None:
        tool, arguments = MockLlmTransport._window_decomposition(
            {
                "window": {
                    "lines": [
                        {
                            "line_id": "L0001",
                            "text": "",
                            "primary": True,
                        }
                    ]
                }
            }
        )

        self.assertEqual(tool, "submit_semantic_window_result")
        self.assertEqual(arguments, {"behaviors": []})
        self.assertNotIn("primary_line_assessments", arguments)

    def test_conversation_accepts_context_with_user_facing_action_labels(
        self,
    ) -> None:
        current_context = ConversationContext(
            session_id="SESSION_MOCK",
            card_id="CARD_MOCK",
            card_revision=2,
            stage="карточка подготовлена",
            continuation="card_decision",
            fields={},
            open_gap=None,
            available_actions=(
                ConversationAction.REFINE_CARD,
                ConversationAction.INCLUDE_CARD,
            ),
        ).as_dict()

        tool, arguments = MockLlmTransport._conversation(
            {
                "messages": [
                    {
                        "type": "human",
                        "content": json.dumps(
                            {
                                "message_id": "MSG_MOCK",
                                "user_text": "работа завершена?",
                                "current_context": current_context,
                            },
                            ensure_ascii=False,
                        ),
                    }
                ]
            }
        )

        self.assertEqual(tool, "respond_to_analyst")
        self.assertNotIn("include_card", arguments["text"])

    def test_conversation_confirms_restored_refinement_proposal(self) -> None:
        current_context = ConversationContext(
            session_id="SESSION_MOCK",
            card_id="CARD_MOCK",
            card_revision=2,
            stage="ожидается подтверждение доработки",
            continuation="card_decision",
            fields={},
            open_gap=None,
            available_actions=(
                ConversationAction.CONFIRM_ANALYST_ANSWER,
                ConversationAction.REJECT_ANALYST_ANSWER,
            ),
            pending_proposal=ConversationProposalContext(
                proposal_id="PROPOSAL_MOCK",
                gap_id=None,
                source_message_id="MSG_SOURCE",
                expected_revision=2,
                values=(
                    {
                        "path": "test.observation.method",
                        "value": "Проверить журнал APDU",
                    },
                ),
                proposal_kind="refinement",
            ),
        ).as_dict()

        tool, _arguments = MockLlmTransport._conversation(
            {
                "messages": [
                    {
                        "type": "human",
                        "content": json.dumps(
                            {
                                "message_id": "MSG_CONFIRM",
                                "user_text": (
                                    "Да, подтверждаю эту доработку"
                                ),
                                "current_context": current_context,
                            },
                            ensure_ascii=False,
                        ),
                    }
                ]
            }
        )

        self.assertEqual(tool, "confirm_analyst_answer")

    def test_conversation_explains_design_research_boundary(self) -> None:
        current_context = ConversationContext(
            session_id="SESSION_DESIGN",
            card_id="CARD_MOCK",
            card_revision=2,
            stage="исследование пробела",
            continuation="gap_decision",
            fields={},
            open_gap=ConversationGapContext(
                gap_id="GAP_DESIGN",
                question="Какие Data использовать в тесте?",
                blocking_reason="Источник не задаёт тестовый вариант",
                allowed_paths=("test.command.data",),
                resolution_mode="design_decision",
            ),
            available_actions=(
                ConversationAction.PROPOSE_DESIGN_DECISION,
                ConversationAction.SUBMIT_ANALYST_ANSWER,
            ),
        ).as_dict()

        tool, _arguments = MockLlmTransport._conversation(
            {
                "messages": [
                    {
                        "type": "human",
                        "content": json.dumps(
                            {
                                "message_id": "MSG_DESIGN",
                                "user_text": (
                                    "Почему не поискать значение в LightRAG?"
                                ),
                                "current_context": current_context,
                            },
                            ensure_ascii=False,
                        ),
                    }
                ]
            }
        )

        self.assertEqual(tool, "propose_design_decision")

    async def test_prompt_1_uses_arbitrary_selection_and_is_repeatable(self) -> None:
        call = self.policy.build_call(
            PromptId.DECOMPOSITION,
            {
                "selection": {
                    "selection_id": "SELECTION_MOCK",
                    "start": {"page": 7, "line": 2},
                    "end": {"page": 7, "line": 3},
                    "text": (
                        "Нейтральное содержание без известных заголовков\n"
                        "Произвольная последняя строка PDF"
                    ),
                    "lines": [
                        {
                            "page": 7,
                            "line": 2,
                            "text": "Нейтральное содержание без известных заголовков",
                        },
                        {
                            "page": 7,
                            "line": 3,
                            "text": "Произвольная последняя строка PDF",
                        },
                    ],
                },
                "existing_card_summaries": [],
            },
        )

        first = await self.transport.complete(call, schemas(call))
        second = await self.transport.complete(call, schemas(call))

        self.assertEqual(first, second)
        arguments = first.tool_calls[0]["arguments"]
        self.assertEqual(first.tool_calls[0]["name"], "submit_decomposition")
        self.assertEqual(len(arguments["skeletons"]), 2)
        first_skeleton, last_skeleton = arguments["skeletons"]
        self.assertEqual(
            first_skeleton["condition_ranges"],
            [{"page": 7, "line_start": 2, "line_end": 2}],
        )
        self.assertEqual(
            last_skeleton["condition_ranges"],
            [{"page": 7, "line_start": 3, "line_end": 3}],
        )
        self.assertIn("Нейтральное содержание", first_skeleton["title"])
        self.assertIn("последняя строка PDF", last_skeleton["title"])
        self.assertNotIn("Каркас 1", str(arguments))
        self.assertNotIn("Условие 1", str(arguments))
        self.assertNotIn("mock-1", str(arguments))

    async def test_prompt_1_creates_one_skeleton_for_one_meaningful_line(
        self,
    ) -> None:
        call = self.policy.build_call(
            PromptId.DECOMPOSITION,
            {
                "selection": {
                    "selection_id": "SELECTION_ONE",
                    "start": {"page": 7, "line": 2},
                    "end": {"page": 7, "line": 2},
                    "text": "Единственное проверяемое утверждение",
                    "lines": [
                        {
                            "page": 7,
                            "line": 2,
                            "text": "Единственное проверяемое утверждение",
                        }
                    ],
                },
                "existing_card_summaries": [],
            },
        )

        response = await self.transport.complete(call, schemas(call))

        skeletons = response.tool_calls[0]["arguments"]["skeletons"]
        self.assertEqual(len(skeletons), 1)
        self.assertIn("Единственное проверяемое утверждение", skeletons[0]["title"])

    async def test_prompt_2_creates_one_source_grounded_gap(self) -> None:
        call = self.policy.build_call(
            PromptId.POPULATION,
            {
                "selection": {"text": "source"},
                "skeleton": {
                    "title": "[mock] Проверка первого байта",
                    "condition": "Первый байт не равен 81",
                    "changed_factor": "первый байт команды",
                    "action": "Передать PUT DATA с первым байтом 80",
                    "consequences": [
                        {"text": "Карта возвращает статус 6987"}
                    ],
                },
                "card": {"card_id": "CARD_1"},
                "evidence": [{"evidence_id": "EVIDENCE_1", "quote": "source"}],
            },
        )

        response = await self.transport.complete(call, schemas(call))
        arguments = response.tool_calls[0]["arguments"]

        self.assertEqual(response.tool_calls[0]["name"], "submit_card_population")
        self.assertEqual(len(arguments["gaps"]), 1)
        self.assertTrue(arguments["source_values"])
        self.assertEqual(
            {
                item["evidence_id"]
                for item in arguments["source_values"]
            },
            {"EVIDENCE_1"},
        )
        values = {
            item["path"]: item["value"]
            for item in arguments["source_values"]
        }
        self.assertEqual(
            values["requirement.condition"],
            "Первый байт не равен 81",
        )
        self.assertEqual(
            values["test.action"],
            "Передать PUT DATA с первым байтом 80",
        )
        self.assertEqual(
            values["test.expected.response_data"],
            "Карта возвращает статус 6987",
        )
        self.assertIn("Карта возвращает статус 6987", arguments["gaps"][0]["question"])
        self.assertNotIn("Условие подтверждено", str(arguments))

    async def test_prompt_3_switches_from_retrieval_to_resolution_by_context(self) -> None:
        base = {
            "selection": {"text": "source"},
            "card": {"card_id": "CARD_1"},
            "gap": {
                "question": "Как наблюдать результат?",
                "allowed_paths": ["test.observation.method"],
            },
            "evidence": [],
        }
        ask = self.policy.build_call(
            PromptId.GAP_RESEARCH,
            {**base, "observations": []},
        )
        submit = self.policy.build_call(
            PromptId.GAP_RESEARCH,
            {
                **base,
                "observations": [
                    {
                        "evidence_ids": ["EVIDENCE_RAG"],
                        "answer": (
                            "[mock] Источник предписывает проверить журнал операций"
                        ),
                    }
                ],
            },
        )

        ask_response = await self.transport.complete(ask, schemas(ask))
        submit_response = await self.transport.complete(submit, schemas(submit))

        self.assertEqual(ask_response.tool_calls[0]["name"], "ask_lightrag")
        self.assertEqual(
            submit_response.tool_calls[0]["name"],
            "submit_gap_result",
        )
        update = submit_response.tool_calls[0]["arguments"]["updates"][0]
        self.assertEqual(update["path"], "test.observation.method")
        self.assertEqual(update["evidence_id"], "EVIDENCE_RAG")
        self.assertIn("проверить журнал операций", update["value"])
        self.assertNotIn("Наблюдение по точному source fragment", update["value"])

    async def test_refinement_and_prompt_4_follow_current_context(self) -> None:
        refinement = self.policy.build_call(
            PromptId.REFINEMENT,
            {
                "card": {"card_id": "CARD_1"},
                "message": {
                    "message_id": "MESSAGE_8",
                    "text": "Использовать журнал операций",
                    "author": "Аналитик",
                },
                "evidence": [],
            },
        )
        review = self.policy.build_call(
            PromptId.SELECTION_REVIEW,
            {
                "selection": {"selection_id": "SELECTION_MOCK"},
                "cards": [{"card_id": "CARD_1"}],
                "skeleton_decisions": [],
            },
        )

        refinement_response = await self.transport.complete(
            refinement,
            schemas(refinement),
        )
        review_response = await self.transport.complete(review, schemas(review))

        update = refinement_response.tool_calls[0]["arguments"]["updates"][0]
        self.assertEqual(update["analyst_message_id"], "MESSAGE_8")
        self.assertIn("Использовать журнал операций", update["value"])
        self.assertEqual(
            review_response.tool_calls[0]["arguments"],
            {"outcome": "approved", "issues": []},
        )


class MockRetrievalTests(unittest.IsolatedAsyncioTestCase):
    async def test_response_is_exact_inside_selection_and_repeatable(self) -> None:
        source = document()
        selection = saved_selection()
        retrieval = MockRetrieval(source, selection, delay=0)
        profile = RetrievalBudgetPolicy.defaults().narrow

        question = "Произвольная последняя строка PDF"
        first = await retrieval.query(question, profile)
        second = await retrieval.query(question, profile)

        self.assertEqual(first, second)
        self.assertIn("[mock]", first.answer)
        fragment = first.fragments[0]
        self.assertTrue(fragment.is_exact)
        self.assertEqual(fragment.page, 7)
        self.assertEqual(fragment.line_start, 3)
        self.assertEqual(fragment.quote, source.line(SourcePosition(7, 3)))
        self.assertIn(fragment.quote, first.answer)
        self.assertIn(
            SourcePosition(fragment.page, fragment.line_start),
            selection.selection.positions,
        )


class MockCompositionTests(unittest.TestCase):
    def test_mock_settings_do_not_require_network_configuration(self) -> None:
        from pmi_generator.workbench.application.settings import WorkbenchSettings

        clean_environment = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith("PMI_")
        }
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.dict(os.environ, clean_environment, clear=True),
        ):
            settings = WorkbenchSettings.from_environment(
                Path(temporary),
                execution_mode=ExecutionMode.MOCK,
            )

        self.assertIs(settings.execution_mode, ExecutionMode.MOCK)
        self.assertIsNone(settings.llm_url)
        self.assertIsNone(settings.lightrag_url)

    def test_manual_delay_defaults_to_one_second_and_can_be_disabled(self) -> None:
        source = document(ExecutionMode.MOCK)
        selection = SavedSelection(
            "SELECTION_MOCK",
            "page-0007",
            source.select(SourcePosition(7, 2), SourcePosition(7, 3)),
        )

        self.assertEqual(MockLlmTransport().delay, 1.0)
        self.assertEqual(MockRetrieval(source, selection).delay, 1.0)
        self.assertEqual(MockLlmTransport(delay=0).delay, 0.0)
        self.assertEqual(
            MockRetrieval(source, selection, delay=0).delay,
            0.0,
        )

    def test_mock_composition_selects_mock_adapters(self) -> None:
        from pmi_generator.workbench.application.settings import WorkbenchSettings
        from pmi_generator.workbench.infrastructure.composition import (
            build_workbench_application,
        )
        from pmi_generator.workbench.infrastructure.storage import (
            workbench_database_path,
        )
        from pmi_generator.workbench.infrastructure.workflow import (
            SqliteWorkflowRuntime,
        )

        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            settings = WorkbenchSettings.from_environment(
                run_dir,
                execution_mode=ExecutionMode.MOCK,
            )
            with (
                patch(
                    "pmi_generator.workbench.infrastructure.composition._production_transport",
                    side_effect=AssertionError("production LLM was constructed"),
                ),
                patch(
                    "pmi_generator.workbench.infrastructure.composition.LightRAGClient",
                    side_effect=AssertionError("LightRAG client was constructed"),
                ),
                SqliteWorkflowRuntime(workbench_database_path(run_dir)) as workflow,
            ):
                facade = build_workbench_application(
                    settings,
                    document(),
                    workflow,
                    mock_delay=0,
                )
                selection = facade.save_selection(
                    "page-0007",
                    document().select(
                        SourcePosition(7, 2),
                        SourcePosition(7, 3),
                    ),
                )

                result = __import__("asyncio").run(
                    facade.decompose(selection).awaitable
                )

        self.assertEqual(len(result.skeleton_ids), 2)

    def test_production_composition_keeps_production_adapter_factories(self) -> None:
        from pmi_generator.workbench.application.settings import WorkbenchSettings
        from pmi_generator.workbench.infrastructure.composition import (
            build_workbench_application,
        )
        from pmi_generator.workbench.infrastructure.storage import (
            workbench_database_path,
        )
        from pmi_generator.workbench.infrastructure.workflow import (
            SqliteWorkflowRuntime,
        )

        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            settings = WorkbenchSettings(
                run_dir=run_dir,
                llm_url="https://llm.invalid/v1",
                llm_model="production-model",
                llm_api_key=None,
                lightrag_url="https://lightrag.invalid",
                lightrag_api_key=None,
                llm_timeout=10,
                retrieval_timeout=10,
                verify_ssl=True,
                no_proxy=False,
            )
            production_transport = MockLlmTransport(delay=0)
            client = Mock()
            with (
                patch(
                    "pmi_generator.workbench.infrastructure.composition._production_transport",
                    return_value=production_transport,
                ) as transport_factory,
                patch(
                    "pmi_generator.workbench.infrastructure.composition.LightRAGClient",
                    return_value=client,
                ) as lightrag_type,
                SqliteWorkflowRuntime(workbench_database_path(run_dir)) as workflow,
            ):
                facade = build_workbench_application(
                    settings,
                    document(),
                    workflow,
                )
                selection = facade.save_selection(
                    "page-0007",
                    document().select(
                        SourcePosition(7, 2),
                        SourcePosition(7, 3),
                    ),
                )
                __import__("asyncio").run(
                    facade.decompose(selection).awaitable
                )
                retrieval = facade.workers.retrieval_factory(selection)

        transport_factory.assert_called_once_with(settings)
        lightrag_type.assert_called_once()
        self.assertIs(retrieval.client, client)


class MockWalkingSkeletonTests(unittest.IsolatedAsyncioTestCase):
    async def test_mock_mode_uses_root_graph_survives_restart_and_exports(
        self,
    ) -> None:
        from pmi_generator.workbench.application.settings import WorkbenchSettings
        from pmi_generator.workbench.infrastructure.composition import (
            build_workbench_application,
        )
        from pmi_generator.workbench.infrastructure.storage import (
            SqliteUnitOfWork,
            workbench_database_path,
        )
        from pmi_generator.workbench.infrastructure.workflow import (
            SqliteWorkflowRuntime,
        )

        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            database_path = workbench_database_path(run_dir)
            source = document(ExecutionMode.MOCK)
            settings = WorkbenchSettings.from_environment(
                run_dir,
                execution_mode=ExecutionMode.MOCK,
            )
            network_guards = (
                patch(
                    "pmi_generator.workbench.infrastructure.composition._production_transport",
                    side_effect=AssertionError("production LLM was constructed"),
                ),
                patch(
                    "pmi_generator.workbench.infrastructure.composition.LightRAGClient",
                    side_effect=AssertionError("LightRAG client was constructed"),
                ),
            )
            with network_guards[0], network_guards[1]:
                with SqliteWorkflowRuntime(database_path) as workflow:
                    facade = build_workbench_application(
                        settings,
                        source,
                        workflow,
                        mock_delay=0,
                    )
                    self.assertEqual(
                        facade.workers.__class__.__name__,
                        "ProductionPromptWorkers",
                    )
                    saved = facade.save_selection(
                        "page-0007",
                        source.select(
                            SourcePosition(7, 2),
                            SourcePosition(7, 3),
                        ),
                    )
                    decomposition = await facade.decompose(saved).awaitable
                    card_id = facade.take_skeleton(
                        saved.selection_id,
                        decomposition.skeleton_ids[0],
                    )
                    facade.exclude_skeleton(
                        saved.selection_id,
                        decomposition.skeleton_ids[1],
                        "[mock] Исключено в ручном сценарии",
                    )
                    session_id = facade.open_card_session(
                        saved.selection_id,
                        card_id,
                    )
                    population = await facade.populate(
                        saved,
                        decomposition.skeleton_ids[0],
                        session_id,
                        card_id,
                    ).awaitable
                    self.assertEqual(len(population.open_gap_ids), 1)
                    gap_id = population.open_gap_ids[0]
                    gap_result = await facade.investigate_gap(
                        saved,
                        session_id,
                        card_id,
                        gap_id,
                    ).awaitable
                    self.assertEqual(gap_result.outcome, "resolved")
                    self.assertEqual(gap_result.observations, 1)

                with SqliteWorkflowRuntime(database_path) as restarted_workflow:
                    facade = build_workbench_application(
                        settings,
                        source,
                        restarted_workflow,
                        mock_delay=0,
                    )
                    self.assertEqual(
                        facade.continuation(session_id),
                        "card_decision",
                    )
                    facade.append(
                        session_id,
                        SessionEventKind.ANALYST,
                        "Использовать журнал операций",
                        {
                            "message_id": "MESSAGE_MOCK",
                            "author": "Аналитик",
                        },
                    )
                    revision = facade.card(card_id).revision
                    refinement = await facade.propose_refinement(
                        session_id,
                        card_id,
                        "MESSAGE_MOCK",
                        revision,
                    ).awaitable
                    self.assertEqual(refinement.outcome, "updated")
                    facade.append(
                        session_id,
                        SessionEventKind.ANALYST,
                        "Да, подтверждаю доработку.",
                        {
                            "message_id": "MESSAGE_CONFIRM_MOCK",
                            "author": "Аналитик",
                        },
                    )
                    facade.dispatch_conversation_tool(
                        saved,
                        decomposition.skeleton_ids[0],
                        session_id,
                        card_id,
                        "MESSAGE_CONFIRM_MOCK",
                        ConversationToolCall(
                            ConversationAction.CONFIRM_ANALYST_ANSWER,
                            {
                                "proposal_id": refinement.proposal_id,
                                "expected_revision": revision,
                                "confirmation_message_id": (
                                    "MESSAGE_CONFIRM_MOCK"
                                ),
                            },
                        ),
                    )
                    facade.include_card(card_id)
                    review = await facade.review_selection(
                        saved.selection_id
                    ).awaitable
                    self.assertEqual(review.outcome, "approved")
                    export = facade.export_full()
                    graph_state = restarted_workflow.current_state(
                        saved.selection_id
                    )

                self.assertTrue(graph_state.export_allowed)
                self.assertIsNone(graph_state.active_attempt)
                self.assertTrue(export.is_file())
                exported = export.read_text(encoding="utf-8")
                self.assertIn("[mock]", exported)
                self.assertIn("Использовать журнал операций", exported)

                with SqliteUnitOfWork(database_path) as uow:
                    diagnostics = uow.records.list_kind("llm_diagnostic")
                    retrieval = uow.records.list_kind(
                        "retrieval_observation"
                    )
                    restored = uow.cards.get(card_id)
                prompt_ids = {
                    str(item.payload["prompt_id"])
                    for item in diagnostics
                }
                self.assertEqual(
                    prompt_ids,
                    {
                        PromptId.DECOMPOSITION.value,
                        PromptId.POPULATION.value,
                        PromptId.GAP_RESEARCH.value,
                        PromptId.REFINEMENT.value,
                        PromptId.SELECTION_REVIEW.value,
                    },
                )
                self.assertEqual(len(retrieval), 1)
                self.assertIn("[mock]", retrieval[0].payload["answer"])
                self.assertEqual(
                    restored.field("test.observation.method").value,
                    "[mock] Использовать журнал операций",
                )
                self.assertEqual(restored.decision.kind.value, "включить")

    async def test_cancelled_mock_prompt_can_be_retried_without_queue_shift(
        self,
    ) -> None:
        from pmi_generator.workbench.application.settings import WorkbenchSettings
        from pmi_generator.workbench.infrastructure.composition import (
            build_workbench_application,
        )
        from pmi_generator.workbench.infrastructure.storage import (
            workbench_database_path,
        )
        from pmi_generator.workbench.infrastructure.workflow import (
            SqliteWorkflowRuntime,
        )

        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            source = document(ExecutionMode.MOCK)
            settings = WorkbenchSettings.from_environment(
                run_dir,
                execution_mode=ExecutionMode.MOCK,
            )
            with SqliteWorkflowRuntime(
                workbench_database_path(run_dir)
            ) as workflow:
                facade = build_workbench_application(
                    settings,
                    source,
                    workflow,
                    mock_delay=0.05,
                )
                saved = facade.save_selection(
                    "page-0007",
                    source.select(
                        SourcePosition(7, 2),
                        SourcePosition(7, 3),
                    ),
                )
                operation = facade.decompose(saved)
                running = asyncio.create_task(operation.awaitable)
                await asyncio.sleep(0.01)

                operation.cancel()

                with self.assertRaises(AttemptDiscardedError):
                    await running
                self.assertIsNone(
                    facade.workspace(saved.selection_id).terminal_status
                )
                self.assertEqual(
                    facade.workspace(saved.selection_id).items,
                    (),
                )
                retry = await facade.decompose(saved).awaitable

        self.assertEqual(len(retry.skeleton_ids), 2)


class MockPresentationLabelTests(unittest.TestCase):
    label = "Тестовый режим: mock"

    def setUp(self) -> None:
        self.source = document(ExecutionMode.MOCK)
        self.selection = SavedSelection(
            "SELECTION_MOCK",
            "page-0007",
            self.source.select(
                SourcePosition(7, 2),
                SourcePosition(7, 3),
            ),
        )
        self.record = StoredRecord(
            "card_skeleton",
            "SKELETON_MOCK",
            {
                "title": "[mock] Каркас",
                "section_number": "Страница A-7",
                "condition": "[mock] Условие",
                "changed_factor": "[mock] Фактор",
                "input_value": "mock",
                "action": "[mock] Действие",
                "condition_evidence": [],
                "consequences": [{"text": "[mock] Последствие"}],
                "gaps": [],
                "decision": None,
            },
        )

    def assert_rendered(self, rendered: object) -> None:
        self.assertIn(self.label, fragment_list_to_text(rendered))

    def test_startup_structure_selection_and_confirmation_show_label(self) -> None:
        startup = render_startup(
            Path("/tmp/mock-run"),
            document=self.source,
            execution_mode=ExecutionMode.MOCK,
        )
        structure = StructureScreen(
            self.source,
            (),
            mode_label=self.label,
        ).render(width=100, height=24)
        selection = SelectionScreen(
            self.source,
            self.source.sections[0],
            mode_label=self.label,
        ).render(width=100, height=24)
        confirmation = ConfirmationScreen(
            self.source.sections[0],
            self.selection.selection,
            source_name="specification.pdf",
            mode_label=self.label,
        ).render(width=100, height=24)

        self.assertIn(self.label, startup)
        self.assert_rendered(structure.fragments)
        self.assert_rendered(selection.fragments)
        self.assert_rendered(confirmation)

    def test_skeleton_list_and_detail_show_label(self) -> None:
        skeletons = SkeletonListScreen(
            (self.record,),
            self.selection,
            mode_label=self.label,
        ).render(width=100, height=24)
        detail = SkeletonDetailScreen(
            self.record,
            mode_label=self.label,
        ).render(width=100, height=24)

        self.assert_rendered(skeletons)
        self.assert_rendered(detail)

    def test_workspace_result_and_prompt_4_show_label(self) -> None:
        state = RangeWorkspaceState(
            selection_id=self.selection.selection_id,
            items=(
                WorkspaceItem(
                    "SKELETON_MOCK",
                    "[mock] Карточка",
                    "готова",
                    "ready",
                    card_id="CARD_MOCK",
                ),
            ),
            can_review=True,
            review_current=False,
            review_stale=False,
            included=1,
            included_incomplete=0,
            excluded=1,
        )
        controller = SimpleNamespace(
            state=state,
            cursor=0,
            rows_count=1,
        )
        workspace = RangeWorkspaceScreen(
            controller,
            self.selection,
            section_number="Страница A-7",
            mode_label=self.label,
        ).render(width=100, height=24)
        result = ResultScreen(
            "PMI Workbench / Результат",
            "[mock] Готово",
            (("back", "Назад"),),
            mode_label=self.label,
        ).render(width=100, height=24)
        review = SelectionReviewScreen(
            [],
            selection=self.selection,
            section_number="Страница A-7",
            mode_label=self.label,
        ).render(width=100, height=24)

        self.assert_rendered(workspace)
        self.assert_rendered(result)
        self.assert_rendered(review)

    def test_session_and_export_message_show_label(self) -> None:
        output = StringIO()
        gateway = Mock()
        gateway.history.return_value = []
        shell = TerminalSessionShell(
            gateway,
            "SESSION_MOCK",
            output=output,
            mode_label=self.label,
            prompt_output=DummyOutput(),
        )
        shell._render_header()

        facade = Mock()
        facade.export_full.return_value = Path("/tmp/mock-run/review/exports/pmi-full.md")
        facade.selections.return_value = []
        terminal = TerminalWorkbench(self.source, facade=facade)
        terminal._export_full()

        self.assertIn(self.label, output.getvalue())
        self.assertIn(self.label, terminal._structure_notice[1])

    def test_terminal_derives_label_from_persisted_mode(self) -> None:
        mock_terminal = TerminalWorkbench(self.source, facade=Mock())
        production_terminal = TerminalWorkbench(document(), facade=Mock())

        self.assertEqual(mock_terminal.mode_label, self.label)
        self.assertIsNone(production_terminal.mode_label)


if __name__ == "__main__":
    unittest.main()
