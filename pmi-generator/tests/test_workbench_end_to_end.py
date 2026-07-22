from __future__ import annotations

import asyncio
import io
import json
import os
import tempfile
import unittest
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import Mock, patch

from pmi_generator.workbench import run_workbench
from pmi_generator.workbench.application.gap_investigation import (
    RetrievalBudgetPolicy,
    RetrievalFragment,
    RetrievalResponse,
    ask_lightrag_tool,
)
from pmi_generator.workbench.application.conversation import (
    ConversationAction,
    ConversationAgentError,
    ConversationToolCall,
)
from pmi_generator.workbench.application.llm import (
    RawCompletion,
    TypedToolRegistry,
)
from pmi_generator.workbench.application.prompting import PromptId, default_policy
from pmi_generator.workbench.application.recovery import RecoveryService
from pmi_generator.workbench.application.session import SessionEventKind
from pmi_generator.workbench.application.settings import WorkbenchSettings
from pmi_generator.workbench.application.state import (
    AttemptRecord,
    AttemptStatus,
    SessionRecord,
)
from pmi_generator.workbench.application.workflow import CommandKind
from pmi_generator.workbench.domain import (
    CardMutation,
    ContentField,
    GapClosureContract,
    GapPathClosure,
    GapResolutionMode,
    GapValueForm,
    RelatedGap,
    SourceDocument,
    SourcePage,
    SourcePosition,
    SourceSection,
    TestCard,
)
from pmi_generator.workbench.infrastructure.composition import (
    build_workbench_application,
)
from pmi_generator.workbench.infrastructure.llm import ScriptedLlmTransport
from pmi_generator.workbench.infrastructure.retrieval import ScriptedRetrieval
from pmi_generator.workbench.infrastructure.storage import (
    SqliteUnitOfWork,
    workbench_database_path,
)
from pmi_generator.workbench.infrastructure.workflow import SqliteWorkflowRuntime
from tests.source_fixture import write_source_snapshot
from pmi_generator.workbench.presentation.terminal import TerminalWorkbench


NOW = datetime(2026, 7, 16, 12, 0, tzinfo=UTC)


def completion(name: str, arguments: dict[str, object], index: int) -> RawCompletion:
    return RawCompletion(
        "tool_calls",
        ({"id": f"call-{index}", "name": name, "arguments": arguments},),
        {"prompt_tokens": 10, "completion_tokens": 5},
        "fake",
    )


def source_document() -> SourceDocument:
    return SourceDocument(
        pages=(
            SourcePage(
                1,
                283,
                (
                    "Требование 4.16.6. Проверить первый байт данных команды.",
                    "Если первый байт не равен 81, карта прекращает PUT DATA",
                    "и возвращает ответ 6987.",
                    "Если CLA не равен 80, команда отклоняется",
                    "с ответом 6E00.",
                ),
            ),
        ),
        sections=(
            SourceSection(
                "section-0270",
                "4.16.5",
                "Обработка команды",
                ("4", "4.16", "4.16.5"),
                (1,),
            ),
        ),
    )


def fake_settings(run_dir: Path) -> WorkbenchSettings:
    return WorkbenchSettings(
        run_dir=run_dir,
        llm_url=None,
        llm_model=None,
        llm_api_key=None,
        lightrag_url=None,
        lightrag_api_key=None,
        llm_timeout=10,
        retrieval_timeout=10,
        verify_ssl=True,
        no_proxy=False,
    )


class WalkingSkeletonTests(unittest.IsolatedAsyncioTestCase):
    async def test_malformed_analyst_answer_then_continue_never_reuses_message(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            run_dir = Path(temp)
            database_path = workbench_database_path(run_dir)
            source = source_document()
            ids: dict[str, int] = {}

            def next_id(prefix: str) -> str:
                ids[prefix] = ids.get(prefix, 0) + 1
                return f"{prefix}_{ids[prefix]:04d}"

            not_found = {
                "outcome": "not_found",
                "updates": [],
                "unknown_fields": ["test.control_values"],
                "missing_fact": "конкретное контрольное значение",
                "summary": "Точное значение не найдено",
                "contradictions": [],
            }
            transport = ScriptedLlmTransport(
                [
                    completion(
                        "submit_decomposition",
                        self._decomposition(),
                        1,
                    ),
                    completion(
                        "submit_card_population",
                        self._population(),
                        2,
                    ),
                    completion(
                        "submit_analyst_answer",
                        {
                            "announcement": "Применяю ответ аналитика.",
                            "values": [
                                {
                                    "field": "test.control_values",
                                    "value": ["Get Data"],
                                }
                            ],
                        },
                        3,
                    ),
                    completion(
                        "ask_lightrag",
                        {
                            "question": (
                                "Какое конкретное неверное значение "
                                "использовать?"
                            )
                        },
                        4,
                    ),
                    completion("submit_gap_result", not_found, 5),
                    completion(
                        "ask_lightrag",
                        {
                            "question": (
                                "Какое конкретное неверное значение "
                                "использовать?"
                            )
                        },
                        6,
                    ),
                    completion("submit_gap_result", not_found, 7),
                ]
            )
            retrieval = ScriptedRetrieval(
                [
                    RetrievalResponse("Точное значение не найдено.", ()),
                    RetrievalResponse("Точное значение не найдено.", ()),
                ]
            )

            with SqliteWorkflowRuntime(database_path) as workflow:
                facade = build_workbench_application(
                    fake_settings(run_dir),
                    source,
                    workflow,
                    transport=transport,
                    retrieval_factory=lambda _selection: retrieval,
                    next_id=next_id,
                )
                selection = source.select(
                    SourcePosition(1, 1),
                    SourcePosition(1, 5),
                )
                saved = facade.save_selection("section-0270", selection)
                decomposition = await facade.decompose(saved).awaitable
                skeleton_id = decomposition.skeleton_ids[0]
                card_id = facade.take_skeleton(
                    saved.selection_id,
                    skeleton_id,
                )
                session_id = facade.open_card_session(
                    saved.selection_id,
                    card_id,
                )
                await facade.populate(
                    saved,
                    skeleton_id,
                    session_id,
                    card_id,
                ).awaitable
                baseline = facade.card(card_id).snapshot()
                sequence = facade.append(
                    session_id,
                    SessionEventKind.ANALYST,
                    "Может Get Data подойдёт?",
                    {"author": "Аналитик"},
                )
                message_id = f"MSG_{sequence:06d}"

                with self.assertRaises(ConversationAgentError):
                    await facade.conversation_turn(
                        saved,
                        skeleton_id,
                        session_id,
                        card_id,
                        message_id,
                    ).awaitable

                for _attempt in range(2):
                    operation = facade.dispatch_conversation_tool(
                        saved,
                        skeleton_id,
                        session_id,
                        card_id,
                        "",
                        ConversationToolCall(
                            ConversationAction.RESUME,
                            {"expected_revision": baseline.revision},
                        ),
                    )
                    self.assertIsNotNone(operation.awaitable)
                    await operation.awaitable
                    self.assertEqual(
                        facade.card(card_id).snapshot(),
                        baseline,
                    )

                card = facade.card(card_id)
                self.assertEqual(card.revision, 1)
                self.assertEqual(
                    card.gaps["GAP_0001"].status.value,
                    "открыт",
                )
                self.assertFalse(
                    any(
                        item.kind.value == "экспертное знание"
                        for item in card.evidence.values()
                    )
                )
                self.assertNotIn(
                    "Анализ ответа APDU",
                    str(card.snapshot()),
                )
                prompt_3_calls = [
                    item["call"]
                    for item in transport.calls
                    if item["call"].prompt_id is PromptId.GAP_RESEARCH
                ]
                self.assertEqual(len(prompt_3_calls), 4)
                self.assertTrue(
                    all(
                        "analyst_messages" not in call.context
                        for call in prompt_3_calls
                    )
                )
                self.assertEqual(
                    [call.question for call in retrieval.calls],
                    [
                        "Какое конкретное неверное значение использовать?",
                        "Какое конкретное неверное значение использовать?",
                    ],
                )
                diagnostic = facade.export_diagnostics(
                    session_id,
                    card_id,
                ).read_text(encoding="utf-8")
                self.assertIn("Может Get Data подойдёт?", diagnostic)
                self.assertIn("неверную структуру", diagnostic)
                self.assertNotIn("human_knowledge", diagnostic)
                self.assertNotIn("Анализ ответа APDU", diagnostic)

    async def test_production_conversation_confirmation_creates_one_revision(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            run_dir = Path(temp)
            database_path = workbench_database_path(run_dir)
            source = source_document()
            ids: dict[str, int] = {}

            def next_id(prefix: str) -> str:
                ids[prefix] = ids.get(prefix, 0) + 1
                return f"{prefix}_{ids[prefix]:04d}"

            transport = ScriptedLlmTransport(
                [
                    completion(
                        "submit_decomposition",
                        self._decomposition(),
                        1,
                    ),
                    completion(
                        "submit_card_population",
                        self._population(),
                        2,
                    ),
                    completion(
                        "submit_analyst_answer",
                        {
                            "announcement": (
                                "Покажу интерпретацию ответа."
                            ),
                            "values": [
                                {
                                    "path": "test.control_values",
                                    "value": {
                                        "kind": "confirmed_value",
                                        "value": ["0b"],
                                    },
                                }
                            ],
                        },
                        3,
                    ),
                    completion(
                        "confirm_analyst_answer",
                        {
                            "announcement": (
                                "Применяю подтверждённую интерпретацию."
                            )
                        },
                        4,
                    ),
                ]
            )

            with SqliteWorkflowRuntime(database_path) as workflow:
                facade = build_workbench_application(
                    fake_settings(run_dir),
                    source,
                    workflow,
                    transport=transport,
                    retrieval_factory=lambda _selection: self.fail(
                        "LightRAG не должен вызываться"
                    ),
                    next_id=next_id,
                )
                saved = facade.save_selection(
                    "section-0270",
                    source.select(
                        SourcePosition(1, 1),
                        SourcePosition(1, 5),
                    ),
                )
                decomposition = await facade.decompose(saved).awaitable
                skeleton_id = decomposition.skeleton_ids[0]
                card_id = facade.take_skeleton(
                    saved.selection_id,
                    skeleton_id,
                )
                session_id = facade.open_card_session(
                    saved.selection_id,
                    card_id,
                )
                await facade.populate(
                    saved,
                    skeleton_id,
                    session_id,
                    card_id,
                ).awaitable
                source_sequence = facade.append(
                    session_id,
                    SessionEventKind.ANALYST,
                    "0b",
                    {"author": "Аналитик"},
                )
                source_message_id = f"MSG_{source_sequence:06d}"

                await facade.conversation_turn(
                    saved,
                    skeleton_id,
                    session_id,
                    card_id,
                    source_message_id,
                ).awaitable
                self.assertEqual(facade.card(card_id).revision, 1)
                pending = facade.conversation_context(
                    session_id,
                    card_id,
                ).pending_proposal
                self.assertEqual(
                    pending.values,
                    (
                        {
                            "path": "test.control_values",
                            "value": {
                                "kind": "confirmed_value",
                                "value": ["0b"],
                            },
                        },
                    ),
                )
                confirmation_sequence = facade.append(
                    session_id,
                    SessionEventKind.ANALYST,
                    "Да, подтверждаю именно это значение.",
                    {"author": "Аналитик"},
                )
                confirmation_message_id = (
                    f"MSG_{confirmation_sequence:06d}"
                )

                await facade.conversation_turn(
                    saved,
                    skeleton_id,
                    session_id,
                    card_id,
                    confirmation_message_id,
                ).awaitable

                card = facade.card(card_id)
                self.assertEqual(card.revision, 2)
                field = card.field("test.control_values")
                self.assertEqual(field.value, ["0b"])
                self.assertEqual(
                    field.status.value,
                    "подтверждено аналитиком",
                )
                self.assertEqual(len(card.resolutions), 1)
                resolution = next(iter(card.resolutions.values()))
                self.assertEqual(
                    resolution.source_message_id,
                    source_message_id,
                )
                self.assertEqual(
                    resolution.confirmation_message_id,
                    confirmation_message_id,
                )
                self.assertEqual(
                    resolution.values,
                    (
                        {
                            "path": "test.control_values",
                            "value": ["0b"],
                        },
                    ),
                )
                diagnostic = facade.export_diagnostics(
                    session_id,
                    card_id,
                ).read_text(encoding="utf-8")
                self.assertIn(source_message_id, diagnostic)
                self.assertIn(confirmation_message_id, diagnostic)
                self.assertIn(pending.proposal_id, diagnostic)
                self.assertIn('"kind": "confirmed_value"', diagnostic)
                self.assertIn('"value": [', diagnostic)

    async def test_partial_gap_closure_survives_sqlite_restart_once(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            run_dir = Path(temp)
            database_path = workbench_database_path(run_dir)
            source = source_document()
            ids: dict[str, int] = {}

            def next_id(prefix: str) -> str:
                ids[prefix] = ids.get(prefix, 0) + 1
                return f"{prefix}_{ids[prefix]:04d}"

            transport = ScriptedLlmTransport(
                [
                    completion(
                        "submit_decomposition",
                        self._decomposition(),
                        1,
                    ),
                    completion(
                        "submit_card_population",
                        self._population(),
                        2,
                    ),
                ]
            )
            builder_options = {
                "transport": transport,
                "retrieval_factory": lambda _selection: self.fail(
                    "LightRAG не должен вызываться для design decision"
                ),
                "next_id": next_id,
            }

            with SqliteWorkflowRuntime(database_path) as workflow:
                facade = build_workbench_application(
                    fake_settings(run_dir),
                    source,
                    workflow,
                    **builder_options,
                )
                saved = facade.save_selection(
                    "section-0270",
                    source.select(
                        SourcePosition(1, 1),
                        SourcePosition(1, 5),
                    ),
                )
                decomposition = await facade.decompose(saved).awaitable
                skeleton_id = decomposition.skeleton_ids[0]
                card_id = facade.take_skeleton(
                    saved.selection_id,
                    skeleton_id,
                )
                session_id = facade.open_card_session(
                    saved.selection_id,
                    card_id,
                )
                await facade.populate(
                    saved,
                    skeleton_id,
                    session_id,
                    card_id,
                ).awaitable

                with SqliteUnitOfWork(database_path) as uow:
                    card = uow.cards.get(card_id)
                    state = card.snapshot()
                    original = state.gaps[0]
                    strict_gap = RelatedGap(
                        gap_id=original.gap_id,
                        card_id=card_id,
                        question=(
                            "Какие конкретные тэги и значения передаются "
                            "в Data команды PUT DATA?"
                        ),
                        blocking_reason=(
                            "Без конкретных Data тест невоспроизводим"
                        ),
                        allowed_paths=("test.command.data",),
                        dependencies=original.dependencies,
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
                                        "Укажите точные байты, конечный набор "
                                        "или правило генерации Data."
                                    ),
                                ),
                            ),
                        ),
                        resolution_mode=GapResolutionMode.DESIGN_DECISION,
                    )
                    uow.cards.save(
                        TestCard.restore(
                            replace(state, gaps=(strict_gap,))
                        )
                    )

                source_sequence = facade.append(
                    session_id,
                    SessionEventKind.ANALYST,
                    "пускай будут произвольные байты",
                    {"author": "Аналитик"},
                )
                source_message_id = f"MSG_{source_sequence:06d}"
                facade.dispatch_conversation_tool(
                    saved,
                    skeleton_id,
                    session_id,
                    card_id,
                    source_message_id,
                    ConversationToolCall(
                        ConversationAction.SUBMIT_ANALYST_ANSWER,
                        {
                            "gap_id": strict_gap.gap_id,
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
                proposal = facade.conversation_context(
                    session_id,
                    card_id,
                ).pending_proposal
                confirmation_sequence = facade.append(
                    session_id,
                    SessionEventKind.ANALYST,
                    "подтверждаю именно эту интерпретацию",
                    {"author": "Аналитик"},
                )
                confirmation_message_id = (
                    f"MSG_{confirmation_sequence:06d}"
                )
                confirmation = ConversationToolCall(
                    ConversationAction.CONFIRM_ANALYST_ANSWER,
                    {
                        "proposal_id": proposal.proposal_id,
                        "expected_revision": 1,
                        "confirmation_message_id": (
                            confirmation_message_id
                        ),
                    },
                )
                facade.dispatch_conversation_tool(
                    saved,
                    skeleton_id,
                    session_id,
                    card_id,
                    confirmation_message_id,
                    confirmation,
                )

                partial = facade.card(card_id)
                self.assertEqual(partial.revision, 2)
                self.assertEqual(
                    partial.gaps[strict_gap.gap_id].status.value,
                    "открыт",
                )
                self.assertEqual(
                    partial.gaps[
                        strict_gap.gap_id
                    ].closure_satisfied_paths,
                    (),
                )
                self.assertEqual(len(partial.resolutions), 1)

            with SqliteWorkflowRuntime(database_path) as restarted_workflow:
                restarted = build_workbench_application(
                    fake_settings(run_dir),
                    source,
                    restarted_workflow,
                    **builder_options,
                )
                restored = restarted.card(card_id)
                restored_gap = restored.gaps[strict_gap.gap_id]

                self.assertEqual(restored.revision, 2)
                self.assertEqual(restored_gap.status.value, "открыт")
                self.assertEqual(
                    restored_gap.closure_contract,
                    strict_gap.closure_contract,
                )
                self.assertEqual(restored_gap.closure_satisfied_paths, ())
                self.assertEqual(len(restored.resolutions), 1)

                retry = restarted.dispatch_conversation_tool(
                    saved,
                    skeleton_id,
                    session_id,
                    card_id,
                    confirmation_message_id,
                    confirmation,
                )

                self.assertIn("уже применена", retry.text)
                self.assertEqual(restarted.card(card_id).revision, 2)
                diagnostic = restarted.export_diagnostics(
                    session_id,
                    card_id,
                ).read_text(encoding="utf-8")
                self.assertIn("произвольные байты", diagnostic)
                self.assertIn(proposal.proposal_id, diagnostic)
                self.assertIn(source_message_id, diagnostic)
                self.assertIn(confirmation_message_id, diagnostic)
                self.assertIn('"partially_resolved"', diagnostic)

    async def test_production_composition_survives_restart_and_exports_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            run_dir = Path(temp)
            database_path = workbench_database_path(run_dir)
            document = source_document()
            ids: dict[str, int] = {}

            def next_id(prefix: str) -> str:
                ids[prefix] = ids.get(prefix, 0) + 1
                return f"{prefix}_{ids[prefix]:04d}"

            transport = ScriptedLlmTransport(
                [
                    completion("submit_decomposition", self._decomposition(), 1),
                    completion("submit_card_population", self._population(), 2),
                    completion(
                        "ask_lightrag",
                        {
                            "question": (
                                "Какое конкретное неверное значение первого байта "
                                "использовать для PUT DATA?"
                            )
                        },
                        3,
                    ),
                    completion("submit_gap_result", self._gap_result(), 4),
                    completion("submit_card_refinement", self._refinement(), 5),
                    completion(
                        "submit_selection_review",
                        {"outcome": "approved", "issues": []},
                        6,
                    ),
                ]
            )
            retrieval = ScriptedRetrieval(
                [
                    RetrievalResponse(
                        "Для проверки условия можно использовать значение 80.",
                        (
                            RetrievalFragment(
                                "spec_2.3.pdf",
                                "2.3",
                                283,
                                1,
                                3,
                                "section-0270",
                                "Допустимо тестовое значение 80.",
                            ),
                        ),
                    )
                ]
            )
            builder_options = {
                "transport": transport,
                "retrieval_factory": lambda _selection: retrieval,
                "next_id": next_id,
            }

            with SqliteWorkflowRuntime(database_path) as workflow:
                facade = build_workbench_application(
                    fake_settings(run_dir),
                    document,
                    workflow,
                    **builder_options,
                )
                selection = document.select(
                    SourcePosition(1, 1),
                    SourcePosition(1, 5),
                )
                saved = facade.save_selection("section-0270", selection)
                decomposition = await facade.decompose(saved).awaitable
                card_id = facade.take_skeleton(
                    saved.selection_id,
                    decomposition.skeleton_ids[0],
                )
                facade.exclude_skeleton(
                    saved.selection_id,
                    decomposition.skeleton_ids[1],
                    "Вне walking skeleton",
                )
                session_id = facade.open_card_session(saved.selection_id, card_id)
                await facade.populate(
                    saved,
                    decomposition.skeleton_ids[0],
                    session_id,
                    card_id,
                ).awaitable
                gap_id = facade.open_gap_ids(card_id)[0]
                await facade.investigate_gap(
                    saved,
                    session_id,
                    card_id,
                    gap_id,
                ).awaitable
                population_context = next(
                    item["call"].context
                    for item in transport.calls
                    if item["call"].prompt_id.value == "prompt_2"
                )
                population_addresses = {
                    (
                        evidence["address"]["line_start"],
                        evidence["address"]["line_end"],
                    )
                    for evidence in population_context["evidence"]
                }
                self.assertEqual(population_addresses, {(1, 2), (2, 3)})
                self.assertNotIn((1, 5), population_addresses)

            self.assertEqual(
                RecoveryService(
                    lambda: SqliteUnitOfWork(database_path),
                    clock=lambda: NOW,
                ).recover(),
                (),
            )

            with SqliteWorkflowRuntime(database_path) as restarted_workflow:
                facade = build_workbench_application(
                    fake_settings(run_dir),
                    document,
                    restarted_workflow,
                    **builder_options,
                )
                self.assertEqual(facade.continuation(session_id), "card_decision")
                facade.append(
                    session_id,
                    SessionEventKind.ANALYST,
                    "Используй значение 7F вместо 80.",
                    {"message_id": "MSG_0001", "author": "Аналитик"},
                )
                revision = facade.card(card_id).revision
                refinement = await facade.propose_refinement(
                    session_id,
                    card_id,
                    "MSG_0001",
                    revision,
                ).awaitable
                facade.append(
                    session_id,
                    SessionEventKind.ANALYST,
                    "Да, подтверждаю эту доработку.",
                    {"message_id": "MSG_0002", "author": "Аналитик"},
                )
                facade.dispatch_conversation_tool(
                    saved,
                    decomposition.skeleton_ids[0],
                    session_id,
                    card_id,
                    "MSG_0002",
                    ConversationToolCall(
                        ConversationAction.CONFIRM_ANALYST_ANSWER,
                        {
                            "proposal_id": refinement.proposal_id,
                            "expected_revision": revision,
                            "confirmation_message_id": "MSG_0002",
                        },
                    ),
                )
                facade.include_card(card_id)
                await facade.review_selection(saved.selection_id).awaitable
                export = facade.export_full()
                diagnostic = facade.export_diagnostics(session_id, card_id)

                state = restarted_workflow.current_state(saved.selection_id)
                self.assertTrue(state.export_allowed)
                self.assertIsNone(state.active_attempt)
                self.assertEqual(
                    [
                        item["command"]["kind"]
                        for item in restarted_workflow.journal(saved.selection_id)
                    ],
                    [
                        CommandKind.CONFIRM_SELECTION.value,
                        CommandKind.BEGIN_ATTEMPT.value,
                        CommandKind.APPLY_DECOMPOSITION.value,
                        CommandKind.TAKE_SKELETON.value,
                        CommandKind.EXCLUDE_SKELETON.value,
                        CommandKind.BEGIN_ATTEMPT.value,
                        CommandKind.APPLY_ATTEMPT_RESULT.value,
                        CommandKind.BEGIN_ATTEMPT.value,
                        CommandKind.APPLY_ATTEMPT_RESULT.value,
                        CommandKind.BEGIN_ATTEMPT.value,
                        CommandKind.REFINE_CARD.value,
                        CommandKind.BEGIN_ATTEMPT.value,
                        CommandKind.REFINE_CARD.value,
                        CommandKind.DECIDE_CARD.value,
                        CommandKind.BEGIN_ATTEMPT.value,
                        CommandKind.SAVE_RANGE_REVIEW.value,
                        CommandKind.REQUEST_EXPORT.value,
                    ],
                )

            rendered = export.read_text(encoding="utf-8")
            diagnostic_text = diagnostic.read_text(encoding="utf-8")
            self.assertIn("Используй значение 7F", rendered)
            self.assertIn("7F", rendered)
            for expected in (
                "## Исходный диапазон",
                "Требование 4.16.6",
                "## Декомпозиция и решения по каркасам",
                "## Версии карточки",
                "### Ревизия 0",
                "### Ревизия 1",
                "### Ревизия 2",
                "### Ревизия 3",
                "## Prompt policy, попытки и tool calls",
                "submit_gap_result",
                "## Запросы LightRAG и evidence",
                "## Детерминированные проверки",
                "## Принятая карточка и итоговый Markdown",
                "## Оценка полноты диапазона",
                "## Экспертная оценка карточки",
                '"result": "не оценено"',
            ):
                with self.subTest(expected=expected):
                    self.assertIn(expected, diagnostic_text)
            self.assertEqual(len(retrieval.calls), 1)
            metrics = run_dir / "review" / "diagnostics" / "workbench-metrics.json"
            metric_values = json.loads(metrics.read_text(encoding="utf-8"))
            self.assertEqual(metric_values["retrieval_calls"], 1)
            self.assertEqual(metric_values["retrieval_unique_questions"], 1)
            self.assertEqual(metric_values["card_revisions_total"], 4)
            self.assertEqual(metric_values["expert_quality"], "не оценено")
            self.assertEqual(
                metric_values["baseline_comparison"],
                "не выполнено",
            )
            with SqliteUnitOfWork(database_path) as restarted:
                restored = restarted.cards.get(card_id)
                self.assertEqual(restored.field("test.control_values").value, ["7F"])
                self.assertEqual(restored.decision.kind.value, "включить")

    @staticmethod
    def _decomposition() -> dict[str, object]:
        return {
            "outcome": "skeletons_created",
            "explanation": "Найдены два явно описанных условия.",
            "skeletons": [
                {
                    "title": "Проверка первого байта",
                    "condition": "первый байт не равен 81",
                    "changed_factor": "первый байт",
                    "input_value": None,
                    "action": "отправить PUT DATA",
                    "condition_ranges": [
                        {"page": 1, "line_start": 1, "line_end": 2}
                    ],
                    "changed_factor_ranges": [
                        {"page": 1, "line_start": 1, "line_end": 2}
                    ],
                    "input_value_ranges": [],
                    "action_ranges": [
                        {"page": 1, "line_start": 1, "line_end": 2}
                    ],
                    "consequences": [
                        {
                            "text": "SW 6987",
                            "evidence_ranges": [
                                {"page": 1, "line_start": 2, "line_end": 3}
                            ],
                        }
                    ],
                    "gaps": [
                        {
                            "kind": "input_value",
                            "question": "Какое конкретное значение использовать?",
                            "target_paths": ["test.control_values"],
                        }
                    ],
                },
                {
                    "title": "Проверка CLA",
                    "condition": "CLA не равен 80",
                    "changed_factor": "CLA",
                    "input_value": None,
                    "action": "отправить команду",
                    "condition_ranges": [
                        {"page": 1, "line_start": 4, "line_end": 4}
                    ],
                    "changed_factor_ranges": [
                        {"page": 1, "line_start": 4, "line_end": 4}
                    ],
                    "input_value_ranges": [],
                    "action_ranges": [
                        {"page": 1, "line_start": 4, "line_end": 4}
                    ],
                    "consequences": [
                        {
                            "text": "SW 6E00",
                            "evidence_ranges": [
                                {"page": 1, "line_start": 4, "line_end": 5}
                            ],
                        }
                    ],
                    "gaps": [
                        {
                            "kind": "input_value",
                            "question": "Какое конкретное значение CLA использовать?",
                            "target_paths": ["test.control_values"],
                        }
                    ],
                },
            ],
            "line_assessments": [
                {
                    "page": 1,
                    "line": line,
                    "role": "evidence",
                    "reason": "Строка использована каркасом",
                }
                for line in range(1, 6)
            ],
        }

    @staticmethod
    def _population() -> dict[str, object]:
        values = {
            "requirement.condition": "первый байт не равен 81",
            "requirement.behavior": "прекратить PUT DATA",
            "test.action": "отправить PUT DATA",
            "test.changed_factor": "первый байт",
            "test.expected.status_word": "6987",
            "test.observation.method": "проверить SW1SW2",
        }
        return {
            "source_values": [
                {
                    "path": path,
                    "value": value,
                    "evidence_id": "EVIDENCE_0001",
                }
                for path, value in values.items()
            ],
            "derivations": [],
            "not_applicable": [],
            "gaps": [
                {
                    "question": "Какое конкретное неверное значение использовать?",
                    "blocking_reason": "Нужно исполнимое значение",
                    "allowed_paths": ["test.control_values"],
                    "dependencies": ["requirement.condition"],
                    "closure_criterion": "указано значение",
                    "resolution_mode": "source_fact",
                }
            ],
        }

    @staticmethod
    def _gap_result() -> dict[str, object]:
        return {
            "outcome": "resolved",
            "updates": [
                {
                    "path": "test.control_values",
                    "value": ["80"],
                    "evidence_id": "EVIDENCE_0003",
                    "analyst_message_id": None,
                }
            ],
            "unknown_fields": [],
            "missing_fact": None,
            "summary": "Значение найдено",
            "contradictions": [],
        }

    @staticmethod
    def _refinement() -> dict[str, object]:
        return {
            "outcome": "updated",
            "updates": [
                {
                    "path": "test.control_values",
                    "value": ["7F"],
                    "evidence_id": None,
                    "analyst_message_id": "MSG_0001",
                }
            ],
            "gaps": [],
            "reason": "Эксперт выбрал другое значение",
        }


class TerminalWorkbenchBoundaryTests(unittest.TestCase):
    def test_headless_terminal_route_reaches_production_export(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            run_dir = Path(temp)
            database_path = workbench_database_path(run_dir)
            source = source_document()
            counts: dict[str, int] = {}

            def next_id(prefix: str) -> str:
                counts[prefix] = counts.get(prefix, 0) + 1
                return f"{prefix}_{counts[prefix]:04d}"

            transport = ScriptedLlmTransport(
                [
                    completion(
                        "submit_decomposition",
                        WalkingSkeletonTests._decomposition(),
                        1,
                    ),
                    completion(
                        "submit_card_population",
                        WalkingSkeletonTests._population(),
                        2,
                    ),
                    completion(
                        "ask_lightrag",
                        {
                            "question": (
                                "Какое конкретное неверное значение первого байта "
                                "использовать для PUT DATA?"
                            )
                        },
                        3,
                    ),
                    completion(
                        "submit_gap_result",
                        WalkingSkeletonTests._gap_result(),
                        4,
                    ),
                    completion(
                        "refine_card",
                        {
                            "announcement": "Применяю уточнение аналитика.",
                        },
                        5,
                    ),
                    completion(
                        "submit_card_refinement",
                        WalkingSkeletonTests._refinement(),
                        6,
                    ),
                    completion(
                        "confirm_analyst_answer",
                        {
                            "announcement": (
                                "Применяю подтверждённую доработку."
                            )
                        },
                        7,
                    ),
                    completion(
                        "submit_selection_review",
                        {"outcome": "approved", "issues": []},
                        8,
                    ),
                ]
            )
            retrieval = ScriptedRetrieval(
                [
                    RetrievalResponse(
                        "Для проверки условия можно использовать значение 80.",
                        (
                            RetrievalFragment(
                                "spec_2.3.pdf",
                                "2.3",
                                283,
                                1,
                                3,
                                "section-0270",
                                "Допустимо тестовое значение 80.",
                            ),
                        ),
                    )
                ]
            )

            with SqliteWorkflowRuntime(database_path) as workflow:
                facade = build_workbench_application(
                    fake_settings(run_dir),
                    source,
                    workflow,
                    transport=transport,
                    retrieval_factory=lambda _selection: retrieval,
                    next_id=next_id,
                )
                terminal = TerminalWorkbench(source, facade=facade)

                def wait_for_operation(
                    _label: str,
                    awaitable: object,
                    cancel: object = None,
                    context: object = None,
                    *,
                    full_screen: bool = False,
                ) -> object:
                    del cancel, context, full_screen
                    return asyncio.run(awaitable)

                terminal._wait = wait_for_operation
                saved = facade.save_selection(
                    "section-0270",
                    source.select(SourcePosition(1, 1), SourcePosition(1, 5)),
                )

                terminal._decide_skeletons = Mock()
                terminal._decompose_selection(saved)
                skeleton_ids = terminal._decide_skeletons.call_args.args[1]

                excluded_detail = Mock()
                excluded_detail.run.return_value = Mock(
                    action="exclude",
                    reason="Вне проверяемого сценария",
                )
                with patch(
                    "pmi_generator.workbench.presentation.terminal.SkeletonDetailScreen",
                    return_value=excluded_detail,
                ):
                    terminal._decide_skeleton(saved, skeleton_ids[1])

                shell_arguments: dict[str, object] = {}
                shell = Mock()

                def build_shell(*_args: object, **kwargs: object) -> Mock:
                    shell_arguments.update(kwargs)
                    return shell

                shell.run.side_effect = lambda: shell_arguments["startup_handler"]()
                selected_detail = Mock()
                selected_detail.run.return_value = Mock(action="take", reason="")
                with (
                    patch(
                        "pmi_generator.workbench.presentation.terminal.SkeletonDetailScreen",
                        return_value=selected_detail,
                    ),
                    patch(
                        "pmi_generator.workbench.presentation.terminal.TerminalSessionShell",
                        side_effect=build_shell,
                    ),
                ):
                    terminal._decide_skeleton(saved, skeleton_ids[0])

                skeleton = facade.skeleton(skeleton_ids[0])
                card_id = str(skeleton.payload["card_id"])
                session_id = facade.session_for_card(saved.selection_id, card_id)
                facade.append(
                    session_id,
                    SessionEventKind.ANALYST,
                    "Используй значение 7F вместо 80.",
                    {
                        "author": "Аналитик",
                        "message_id": "MSG_0001",
                    },
                )
                shell_arguments["message_handler"]("MSG_0001")
                facade.append(
                    session_id,
                    SessionEventKind.ANALYST,
                    "Да, подтверждаю.",
                    {
                        "author": "Аналитик",
                        "message_id": "MSG_0002",
                    },
                )
                shell_arguments["message_handler"]("MSG_0002")
                errors = [
                    event.text
                    for event in facade.history(session_id)
                    if event.kind is SessionEventKind.ERROR
                ]
                self.assertEqual(errors, [])
                shell_arguments["command_handlers"]["/include"]()

                review_screen = Mock()
                review_screen.run.return_value = "back"
                with patch(
                    "pmi_generator.workbench.presentation.terminal.SelectionReviewScreen",
                    return_value=review_screen,
                ):
                    terminal._review_selection(saved.selection_id)
                terminal._export_full()

                state = workflow.current_state(saved.selection_id)
                self.assertTrue(state.export_allowed)
                self.assertEqual(state.cards[card_id].decision, "include")

            exported = run_dir / "review" / "exports" / "pmi-full.md"
            self.assertTrue(exported.exists())
            self.assertIn("7F", exported.read_text(encoding="utf-8"))
            self.assertEqual(len(transport.calls), 8)
            self.assertEqual(len(retrieval.calls), 1)

    @staticmethod
    def _dialog(result: object) -> Mock:
        dialog = Mock()
        dialog.run.return_value = result
        return dialog


class RestartAndBootstrapTests(unittest.TestCase):
    def test_restart_cancels_only_active_attempts_and_keeps_decisions(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = workbench_database_path(Path(temp))
            with SqliteUnitOfWork(path) as uow:
                uow.sessions.save(SessionRecord("S1", "SEL", "CARD", "prompt 3", {}, NOW))
                uow.attempts.save(
                    AttemptRecord("A1", "S1", "prompt 3", AttemptStatus.ACTIVE, {}, NOW)
                )
            recovered = RecoveryService(
                lambda: SqliteUnitOfWork(path), clock=lambda: NOW
            ).recover()
            self.assertEqual(recovered, ("A1",))
            with SqliteUnitOfWork(path) as uow:
                self.assertEqual(uow.attempts.get("A1").status, AttemptStatus.CANCELLED)
                session = uow.sessions.get("S1")
                self.assertEqual(
                    session.current_stage,
                    "операция прервана после перезапуска",
                )
                self.assertEqual(session.payload["continuation"], "population")
                self.assertIsNone(session.payload["active_intent"])

    def test_restart_discards_ready_result_and_fails_interrupted_apply(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = workbench_database_path(Path(temp))
            with SqliteUnitOfWork(path) as uow:
                uow.attempts.save(
                    AttemptRecord(
                        "A_READY",
                        "S1",
                        "prompt 2",
                        AttemptStatus.RESULT_READY,
                        {},
                        NOW,
                    )
                )
                uow.attempts.save(
                    AttemptRecord(
                        "A_APPLYING",
                        "S1",
                        "prompt 2",
                        AttemptStatus.APPLYING,
                        {},
                        NOW,
                    )
                )

            recovered = RecoveryService(
                lambda: SqliteUnitOfWork(path),
                clock=lambda: NOW,
            ).recover()

            self.assertEqual(recovered, ("A_APPLYING", "A_READY"))
            with SqliteUnitOfWork(path) as uow:
                self.assertEqual(
                    uow.attempts.get("A_READY").status,
                    AttemptStatus.CANCELLED,
                )
                self.assertEqual(
                    uow.attempts.get("A_APPLYING").status,
                    AttemptStatus.FAILED,
                )
                diagnostic = uow.records.get("recovery_diagnostic", "A_APPLYING")
                self.assertEqual(
                    diagnostic.payload["reason"],
                    "Workbench перезапущен во время применения результата",
                )

    def test_restart_restores_each_stable_session_continuation_route(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = workbench_database_path(Path(temp))
            cards = {
                "S_POPULATION": self._recovery_card("C_POPULATION"),
                "S_GAP": self._recovery_card("C_GAP", with_open_gap=True),
                "S_DECISION": self._recovery_card("C_DECISION", populated=True),
            }
            with SqliteUnitOfWork(path) as uow:
                for session_id, card in cards.items():
                    uow.cards.save(card)
                    uow.sessions.save(
                        SessionRecord(
                            session_id,
                            "SELECTION_RECOVERY",
                            card.card_id,
                            "операция выполняется",
                            {"active_intent": {"attempt_id": f"A_{session_id}"}},
                            NOW,
                        )
                    )
                    uow.attempts.save(
                        AttemptRecord(
                            f"A_{session_id}",
                            session_id,
                            "prompt",
                            AttemptStatus.ACTIVE,
                            {},
                            NOW,
                        )
                    )

            RecoveryService(
                lambda: SqliteUnitOfWork(path),
                clock=lambda: NOW,
            ).recover()

            with SqliteUnitOfWork(path) as uow:
                routes = {
                    session_id: uow.sessions.get(session_id).payload["continuation"]
                    for session_id in cards
                }
            self.assertEqual(
                routes,
                {
                    "S_POPULATION": "population",
                    "S_GAP": "gap_investigation",
                    "S_DECISION": "card_decision",
                },
            )

    @staticmethod
    def _recovery_card(
        card_id: str,
        *,
        with_open_gap: bool = False,
        populated: bool = False,
    ) -> TestCard:
        card = TestCard.create(
            card_id=card_id,
            selection_id="SELECTION_RECOVERY",
            title="Recovery",
            section_number="4.16",
            changed_factors=("factor",),
            consequences=("result",),
        )
        if with_open_gap:
            card.apply(
                CardMutation(
                    gaps=(
                        RelatedGap(
                            gap_id=f"GAP_{card_id}",
                            card_id=card_id,
                            question="Что проверить?",
                            blocking_reason="Нужен ответ",
                            allowed_paths=("test.control_values",),
                            dependencies=(),
                            closure_criterion="Ответ найден",
                        ),
                    )
                )
            )
        elif populated:
            card.apply(
                CardMutation(
                    fields={
                        "test.command.le": ContentField.not_applicable(
                            "Поле не относится к сценарию"
                        )
                    }
                )
            )
        return card

    def test_review_start_initializes_database_and_reports_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            run_dir = Path(temp)
            document = source_document()
            write_source_snapshot(
                run_dir,
                pages=document.pages,
                sections=document.sections,
                original_name="spec_2.3.pdf",
            )
            output = io.StringIO()
            self.assertEqual(run_workbench(run_dir, output=output), 0)
            self.assertTrue(workbench_database_path(run_dir).exists())
            self.assertIn("PMI Workbench", output.getvalue())


@unittest.skipUnless(
    os.getenv("PMI_WORKBENCH_LIVE_SMOKE") == "1",
    "нужен PMI_WORKBENCH_LIVE_SMOKE=1",
)
class LiveServicesSmokeTest(unittest.IsolatedAsyncioTestCase):
    async def test_vllm_and_lightrag_are_reachable(self) -> None:
        from pmi_generator.clients.lightrag import LightRAGClient
        from pmi_generator.workbench.infrastructure.llm import (
            OpenAICompatibleTransport,
            OpenAITransportSettings,
        )
        from pmi_generator.workbench.infrastructure.retrieval import LightRAGRetrieval

        client = LightRAGClient(
            os.environ["PMI_LIGHTRAG_URL"],
            os.getenv("PMI_LIGHTRAG_API_KEY"),
            float(os.getenv("PMI_LIGHTRAG_QUERY_TIMEOUT", "900")),
            not bool(int(os.getenv("PMI_INSECURE", "0"))),
            None,
            bool(int(os.getenv("PMI_NO_PROXY", "0"))),
        )
        result = await LightRAGRetrieval(client).query(
            "Какой код ответа означает успешное выполнение команды?",
            RetrievalBudgetPolicy.defaults().narrow,
        )
        self.assertTrue(result.answer.strip())
        registry = TypedToolRegistry()
        registry.register(ask_lightrag_tool())
        prompt = default_policy().build_call(
            PromptId.GAP_RESEARCH,
            {
                "selection": {"text": "smoke"},
                "card": {"card_id": "SMOKE"},
                "gap": {"question": "smoke"},
                "evidence": [],
                "analyst_messages": [],
                "observations": [],
            },
        )
        transport = OpenAICompatibleTransport(
            OpenAITransportSettings(
                base_url=os.environ["PMI_LLM_URL"],
                model=os.environ["PMI_LLM_MODEL"],
                api_key=os.getenv("PMI_LLM_API_KEY"),
                timeout=float(os.getenv("PMI_TIMEOUT", "600")),
                verify_ssl=os.getenv("PMI_INSECURE", "0") != "1",
            )
        )
        completion_result = await transport.complete(
            prompt, registry.openai_schemas(("ask_lightrag",))
        )
        self.assertEqual(completion_result.finish_reason, "tool_calls")
