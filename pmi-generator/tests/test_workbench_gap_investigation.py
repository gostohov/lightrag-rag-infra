from __future__ import annotations

import asyncio
import unittest
from datetime import UTC, datetime

from pmi_generator.workbench.application.card_population import AnalystMessage
from pmi_generator.workbench.application.gap_investigation import (
    AnalystConfirmation,
    GapArguments,
    GapInvestigationError,
    GapInvestigationFlow,
    GapInvestigationService,
    RetrievalBudgetPolicy,
    RetrievalFragment,
    RetrievalResponse,
    ask_lightrag_tool,
    expand_lightrag_tool,
    submit_gap_result_tool,
)
from pmi_generator.workbench.application.llm import (
    AttemptDiscardedError,
    LlmToolRuntime,
    RawCompletion,
    ToolContractError,
    TypedToolRegistry,
)
from pmi_generator.workbench.application.prompting import default_policy
from pmi_generator.workbench.application.session import SessionEventKind, SessionService
from pmi_generator.workbench.domain import CardMutation, RelatedGap, TestCard
from pmi_generator.workbench.domain.schema import CARD_FIELD_PATHS
from pmi_generator.workbench.infrastructure.llm import ScriptedLlmTransport
from pmi_generator.workbench.infrastructure.langchain import LangChainGapAgent
from pmi_generator.workbench.infrastructure.retrieval import ScriptedRetrieval
from pmi_generator.workbench.infrastructure.storage import InMemoryDatabase, InMemoryUnitOfWork


NOW = datetime(2026, 7, 16, 12, 0, tzinfo=UTC)


def analyst_confirmation(
    *values: tuple[str, object],
    revision: int = 1,
) -> AnalystConfirmation:
    return AnalystConfirmation(
        proposal_id="PROPOSAL_0001",
        source_message_id="MSG_0001",
        confirmation_message_id="MSG_CONFIRM_0001",
        gap_id="GAP_0002",
        expected_revision=revision,
        values=tuple(
            {"path": path, "value": value}
            for path, value in values
        ),
    )


def make_card(
    card_id: str = "CARD_0001",
    gap_id: str = "GAP_0001",
    allowed_paths: tuple[str, ...] = ("test.observation.method",),
) -> TestCard:
    card = TestCard.create(
        card_id=card_id,
        selection_id="SELECTION_0001",
        title="Проверка уменьшения счётчика",
        section_number="4.16.5",
        changed_factors=("результат проверки MAC",),
        consequences=("счётчик SK SMI уменьшен",),
    )
    card.apply(
        CardMutation(
            gaps=(
                RelatedGap(
                    gap_id=gap_id,
                    card_id=card_id,
                    question="Как наблюдать уменьшение счётчика SK SMI?",
                    blocking_reason="Без способа наблюдения результат непроверяем",
                    allowed_paths=allowed_paths,
                    dependencies=("requirement.consequences",),
                    closure_criterion="указан подтверждённый способ наблюдения",
                ),
            )
        )
    )
    return card


class GapToolContractTests(unittest.TestCase):
    def test_result_paths_are_constrained_to_domain_schema(self) -> None:
        properties = submit_gap_result_tool().json_schema["properties"]
        self.assertEqual(
            set(properties["updates"]["items"]["properties"]["path"]["enum"]),
            set(CARD_FIELD_PATHS),
        )
        self.assertEqual(
            set(properties["unknown_fields"]["items"]["enum"]),
            set(CARD_FIELD_PATHS),
        )

    def test_structured_missing_fact_must_reference_unknown_field(self) -> None:
        with self.assertRaisesRegex(
            GapInvestigationError,
            "отсутствует в неизвестных полях",
        ):
            GapArguments(
                outcome="not_found",
                updates=[],
                unknown_fields=["test.observation.method"],
                missing_fact={
                    "field": "test.command.data",
                    "description": "Конкретное значение команды",
                },
                summary="Значение не найдено",
                contradictions=[],
            )

    def test_resolved_cannot_contain_unknown_fields(self) -> None:
        with self.assertRaisesRegex(
            GapInvestigationError,
            "resolved требует только подтверждённые обновления",
        ):
            GapArguments(
                outcome="resolved",
                updates=[
                    {
                        "path": "test.command.data",
                        "value": "80",
                        "evidence_id": None,
                        "analyst_message_id": "MSG_000173",
                    }
                ],
                unknown_fields=["test.action"],
                missing_fact=None,
                summary="Аналитик указал значение 80",
                contradictions=[],
            )


def raw_tool(name: str, arguments: dict[str, object], call_id: str = "call-1") -> RawCompletion:
    return RawCompletion(
        finish_reason="tool_calls",
        tool_calls=({"id": call_id, "name": name, "arguments": arguments},),
        usage={},
        model="fake",
    )


def exact_fragment() -> RetrievalFragment:
    return RetrievalFragment(
        document_id="spec_2.3.pdf",
        document_version="2.3",
        page=284,
        line_start=10,
        line_end=12,
        chunk_id="section-0270",
        quote="При совпадении MAC необходимо уменьшить счётчик SK SMI.",
    )


class GapInvestigationServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.database = InMemoryDatabase()
        with InMemoryUnitOfWork(self.database) as uow:
            uow.cards.save(make_card())
        self.counter = 0
        self.service = GapInvestigationService(
            uow_factory=lambda: InMemoryUnitOfWork(self.database),
            next_id=self.next_id,
            clock=lambda: NOW,
        )

    def next_id(self, prefix: str) -> str:
        self.counter += 1
        return f"{prefix}_{self.counter:04d}"

    def test_resolved_updates_only_allowed_paths_and_closes_gap(self) -> None:
        evidence = exact_fragment().to_evidence(
            evidence_id="EVIDENCE_0001",
            card_id="CARD_0001",
            selection_id="SELECTION_0001",
            collected_at=NOW,
        )
        result = self.service.apply(
            "CARD_0001",
            "GAP_0001",
            GapArguments(
                outcome="resolved",
                updates=[
                    {
                        "path": "test.observation.method",
                        "value": "прочитать счётчик до и после команды",
                        "evidence_id": "EVIDENCE_0001",
                        "analyst_message_id": None,
                    }
                ],
                unknown_fields=[],
                missing_fact=None,
                summary="Способ найден",
                contradictions=[],
            ),
            available_evidence=(evidence,),
            analyst_messages=(),
        )

        with InMemoryUnitOfWork(self.database) as uow:
            card = uow.cards.get("CARD_0001")
        self.assertEqual(result.outcome, "resolved")
        self.assertEqual(card.field("test.observation.method").value, "прочитать счётчик до и после команды")
        self.assertEqual(card.gaps["GAP_0001"].status.value, "закрыт")

    def test_resolved_must_fill_every_unknown_gap_path(self) -> None:
        with InMemoryUnitOfWork(self.database) as uow:
            uow.cards.save(
                make_card(
                    card_id="CARD_0002",
                    gap_id="GAP_0002",
                    allowed_paths=("test.command.data", "test.action"),
                )
            )
        message = AnalystMessage(
            "MSG_0001",
            "CARD_0002",
            "Аналитик",
            "80",
            NOW,
        )

        with self.assertRaisesRegex(
            GapInvestigationError,
            "test.action",
        ):
            self.service.apply(
                "CARD_0002",
                "GAP_0002",
                GapArguments(
                    outcome="resolved",
                    updates=[
                        {
                            "path": "test.command.data",
                            "value": "80",
                            "evidence_id": None,
                            "analyst_message_id": "MSG_0001",
                        }
                    ],
                    unknown_fields=[],
                    missing_fact=None,
                    summary="Аналитик указал значение",
                    contradictions=[],
                ),
                available_evidence=(),
                analyst_messages=(message,),
            )

    def test_one_analyst_message_becomes_one_evidence_for_multiple_updates(self) -> None:
        with InMemoryUnitOfWork(self.database) as uow:
            uow.cards.save(
                make_card(
                    card_id="CARD_0002",
                    gap_id="GAP_0002",
                    allowed_paths=("test.command.data", "test.action"),
                )
            )
        message = AnalystMessage(
            "MSG_0001",
            "CARD_0002",
            "Аналитик",
            "80",
            NOW,
        )

        self.service.apply(
            "CARD_0002",
            "GAP_0002",
            GapArguments(
                outcome="resolved",
                updates=[
                    {
                        "path": "test.command.data",
                        "value": "80",
                        "evidence_id": None,
                        "analyst_message_id": "MSG_0001",
                    },
                    {
                        "path": "test.action",
                        "value": "Отправить PUT DATA с первым байтом 80",
                        "evidence_id": None,
                        "analyst_message_id": "MSG_0001",
                    },
                ],
                unknown_fields=[],
                missing_fact=None,
                summary="Аналитик указал значение",
                contradictions=[],
            ),
            available_evidence=(),
            analyst_messages=(message,),
            analyst_confirmation=analyst_confirmation(
                ("test.command.data", "80"),
                (
                    "test.action",
                    "Отправить PUT DATA с первым байтом 80",
                ),
            ),
        )

        with InMemoryUnitOfWork(self.database) as uow:
            card = uow.cards.get("CARD_0002")
        data_evidence = card.field("test.command.data").evidence_ids
        action_evidence = card.field("test.action").evidence_ids
        self.assertEqual(data_evidence, action_evidence)
        self.assertEqual(len(card.evidence), 1)
        self.assertEqual(card.evidence[data_evidence[0]].message_id, "MSG_0001")
        resolution = next(iter(card.resolutions.values()))
        self.assertEqual(resolution.proposal_id, "PROPOSAL_0001")
        self.assertEqual(resolution.source_message_id, "MSG_0001")
        self.assertEqual(
            resolution.confirmation_message_id,
            "MSG_CONFIRM_0001",
        )
        self.assertEqual(resolution.gap_id, "GAP_0002")
        self.assertEqual(resolution.expected_revision, 1)
        self.assertEqual(
            resolution.values,
            (
                {"path": "test.command.data", "value": "80"},
                {
                    "path": "test.action",
                    "value": "Отправить PUT DATA с первым байтом 80",
                },
            ),
        )

    def test_forbidden_path_rejects_entire_result(self) -> None:
        arguments = GapArguments(
            outcome="resolved",
            updates=[
                {
                    "path": "test.command.cla",
                    "value": "0C",
                    "evidence_id": "EVIDENCE_0001",
                    "analyst_message_id": None,
                }
            ],
            unknown_fields=[],
            missing_fact=None,
            summary="",
            contradictions=[],
        )
        evidence = exact_fragment().to_evidence(
            evidence_id="EVIDENCE_0001",
            card_id="CARD_0001",
            selection_id="SELECTION_0001",
            collected_at=NOW,
        )
        with self.assertRaises(GapInvestigationError):
            self.service.apply(
                "CARD_0001", "GAP_0001", arguments,
                available_evidence=(evidence,), analyst_messages=(),
            )
        with InMemoryUnitOfWork(self.database) as uow:
            self.assertEqual(uow.cards.get("CARD_0001").revision, 1)

    def test_existing_card_evidence_cannot_resolve_current_research(self) -> None:
        evidence = exact_fragment().to_evidence(
            evidence_id="EVIDENCE_EXISTING",
            card_id="CARD_0001",
            selection_id="SELECTION_0001",
            collected_at=NOW,
        )
        with InMemoryUnitOfWork(self.database) as uow:
            card = uow.cards.get("CARD_0001")
            card.apply(CardMutation(evidence=(evidence,)))
            uow.cards.save(card)

        with self.assertRaisesRegex(
            GapInvestigationError,
            "EVIDENCE_EXISTING",
        ):
            self.service.apply(
                "CARD_0001",
                "GAP_0001",
                GapArguments(
                    outcome="resolved",
                    updates=[
                        {
                            "path": "test.observation.method",
                            "value": "GET DATA",
                            "evidence_id": "EVIDENCE_EXISTING",
                            "analyst_message_id": None,
                        }
                    ],
                    unknown_fields=[],
                    missing_fact=None,
                    summary="Способ найден",
                    contradictions=[],
                ),
                available_evidence=(),
                analyst_messages=(),
            )

    def test_not_found_and_contradiction_do_not_change_card(self) -> None:
        for outcome, contradictions in (
            ("not_found", []),
            (
                "contradiction",
                [
                    {"statement": "значение уменьшается", "evidence_id": "E1"},
                    {"statement": "значение не меняется", "evidence_id": "E2"},
                ],
            ),
        ):
            with self.subTest(outcome=outcome):
                before = self.database.cards["CARD_0001"]
                evidence = (
                    exact_fragment().to_evidence("E1", "CARD_0001", "SELECTION_0001", NOW),
                    exact_fragment().to_evidence("E2", "CARD_0001", "SELECTION_0001", NOW),
                )
                result = self.service.apply(
                    "CARD_0001",
                    "GAP_0001",
                    GapArguments(
                        outcome=outcome,
                        updates=[],
                        unknown_fields=["test.observation.method"] if outcome == "not_found" else [],
                        missing_fact="способ наблюдения" if outcome == "not_found" else None,
                        summary="Результат исследования",
                        contradictions=contradictions,
                    ),
                    available_evidence=evidence,
                    analyst_messages=(),
                )
                self.assertEqual(result.outcome, outcome)
                self.assertEqual(self.database.cards["CARD_0001"], before)

    def test_contradiction_requires_two_evidenced_statements(self) -> None:
        with self.assertRaisesRegex(GapInvestigationError, "два"):
            self.service.apply(
                "CARD_0001",
                "GAP_0001",
                GapArguments("contradiction", [], [], None, "", [{"statement": "одно", "evidence_id": "E1"}]),
                available_evidence=(),
                analyst_messages=(),
            )

    def test_analyst_message_becomes_card_local_evidence_atomically(self) -> None:
        message = AnalystMessage("MSG_0001", "CARD_0001", "Аналитик", "Используй стендовый счётчик.", NOW)
        self.service.apply(
            "CARD_0001",
            "GAP_0001",
            GapArguments(
                "resolved",
                [{"path": "test.observation.method", "value": "стендовый счётчик", "evidence_id": None, "analyst_message_id": "MSG_0001"}],
                [], None, "", [],
            ),
            available_evidence=(),
            analyst_messages=(message,),
            analyst_confirmation=AnalystConfirmation(
                proposal_id="PROPOSAL_0001",
                source_message_id="MSG_0001",
                confirmation_message_id="MSG_CONFIRM_0001",
                gap_id="GAP_0001",
                expected_revision=1,
                values=(
                    {
                        "path": "test.observation.method",
                        "value": "стендовый счётчик",
                    },
                ),
            ),
        )
        with InMemoryUnitOfWork(self.database) as uow:
            card = uow.cards.get("CARD_0001")
        evidence = card.evidence[card.field("test.observation.method").evidence_ids[0]]
        self.assertEqual(evidence.message_id, "MSG_0001")
        self.assertEqual(
            card.field("test.observation.method").status.value,
            "подтверждено аналитиком",
        )
        self.assertEqual(len(card.resolutions), 1)

    def test_analyst_update_requires_matching_confirmation(self) -> None:
        message = AnalystMessage(
            "MSG_0001",
            "CARD_0001",
            "Аналитик",
            "Используй стендовый счётчик.",
            NOW,
        )
        arguments = GapArguments(
            "resolved",
            [
                {
                    "path": "test.observation.method",
                    "value": "стендовый счётчик",
                    "evidence_id": None,
                    "analyst_message_id": "MSG_0001",
                }
            ],
            [],
            None,
            "",
            [],
        )

        with self.assertRaisesRegex(
            GapInvestigationError,
            "подтверждение",
        ):
            self.service.apply(
                "CARD_0001",
                "GAP_0001",
                arguments,
                available_evidence=(),
                analyst_messages=(message,),
            )
        with InMemoryUnitOfWork(self.database) as uow:
            card = uow.cards.get("CARD_0001")
        self.assertEqual(card.revision, 1)
        self.assertEqual(card.resolutions, {})


class GapInvestigationFlowTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.database = InMemoryDatabase()
        with InMemoryUnitOfWork(self.database) as uow:
            uow.cards.save(make_card())
        self.sessions = SessionService(
            uow_factory=lambda: InMemoryUnitOfWork(self.database), clock=lambda: NOW
        )
        self.sessions.open("SESSION_0001", "SELECTION_0001", "CARD_0001")

    def make_flow(
        self,
        llm: list[object],
        retrieval: list[object],
        max_steps: int = 6,
        max_retries: int = 0,
    ):
        registry = TypedToolRegistry()
        for tool in (ask_lightrag_tool(), expand_lightrag_tool(), submit_gap_result_tool()):
            registry.register(tool)
        transport = ScriptedLlmTransport(llm)
        runtime = LlmToolRuntime(
            transport=transport,
            tools=registry,
            uow_factory=lambda: InMemoryUnitOfWork(self.database),
            max_retries=max_retries,
        )
        retrieval_client = ScriptedRetrieval(retrieval)
        counter = {"value": 0}

        def next_id(prefix: str) -> str:
            counter["value"] += 1
            return f"{prefix}_{counter['value']:04d}"

        flow = GapInvestigationFlow(
            policy=default_policy(),
            runtime=runtime,
            agent=LangChainGapAgent(runtime),
            retrieval=retrieval_client,
            budgets=RetrievalBudgetPolicy.defaults(),
            service=GapInvestigationService(
                uow_factory=lambda: InMemoryUnitOfWork(self.database),
                next_id=next_id,
                clock=lambda: NOW,
            ),
            sessions=self.sessions,
            uow_factory=lambda: InMemoryUnitOfWork(self.database),
            next_id=next_id,
            clock=lambda: NOW,
            max_steps=max_steps,
        )
        return flow, transport, retrieval_client

    async def test_short_question_uses_narrow_profile_and_only_gap_context(self) -> None:
        question = "Как наблюдать уменьшение счётчика SK SMI?"
        flow, transport, retrieval = self.make_flow(
            [
                raw_tool("ask_lightrag", {"question": question}),
                raw_tool(
                    "submit_gap_result",
                    {
                        "outcome": "not_found", "updates": [],
                        "unknown_fields": ["test.observation.method"],
                        "missing_fact": "способ чтения", "summary": "Не найдено",
                        "contradictions": [],
                    },
                    "call-2",
                ),
            ],
            [RetrievalResponse("Прямой способ не найден", ())],
        )
        await flow.run(
            attempt_id="RESEARCH_0001", session_id="SESSION_0001",
            card_id="CARD_0001", gap_id="GAP_0001",
            selection={"text": "Требование 4.16.10"},
        )

        self.assertEqual(retrieval.calls[0].profile.name, "узкий поиск")
        self.assertEqual(retrieval.calls[0].question, question)
        context = transport.calls[0]["call"].context
        self.assertEqual(context["gap"]["gap_id"], "GAP_0001")
        self.assertNotIn("session_history", context)
        self.assertEqual(
            transport.calls[0]["call"].allowed_tools,
            ("ask_lightrag",),
        )
        self.assertEqual(
            transport.calls[1]["call"].allowed_tools,
            ("expand_lightrag", "submit_gap_result"),
        )
        stop = self.sessions.history("SESSION_0001")[-1]
        self.assertIs(stop.kind, SessionEventKind.ASSISTANT)
        self.assertIn("требуется решение аналитика", stop.text)
        self.assertIn("Как наблюдать уменьшение счётчика SK SMI?", stop.text)
        self.assertIn("способ чтения", stop.text)
        self.assertIn("Не найдено", stop.text)
        self.assertIn("test.observation.method", stop.text)

    async def test_research_question_is_context_not_analyst_evidence(self) -> None:
        instruction = "Можно ли прочитать PTH через GET DATA с тегом C7?"
        self.sessions.append(
            "SESSION_0001",
            SessionEventKind.ASSISTANT,
            "Требуется решение аналитика.",
            {"gap_id": "GAP_0001", "outcome": "not_found"},
        )
        sequence = self.sessions.append(
            "SESSION_0001",
            SessionEventKind.ANALYST,
            instruction,
            {"author": "Аналитик"},
        )
        message_id = f"MSG_{sequence:06d}"
        flow, transport, _ = self.make_flow(
            [
                raw_tool(
                    "ask_lightrag",
                    {"question": "Как наблюдать уменьшение счётчика SK SMI?"},
                    "narrow",
                ),
                raw_tool(
                    "expand_lightrag",
                    {"call_id": "narrow", "reason": "Нужен более широкий контекст"},
                    "broad",
                ),
                raw_tool(
                    "submit_gap_result",
                    {
                        "outcome": "not_found",
                        "updates": [],
                        "unknown_fields": ["test.observation.method"],
                        "missing_fact": "метод наблюдения",
                        "summary": "Метод наблюдения не найден",
                        "contradictions": [],
                    },
                    "submit",
                ),
            ],
            [
                RetrievalResponse("Нет точного метода", ()),
                RetrievalResponse("В расширенном поиске метод не найден", ()),
            ],
        )

        await flow.run(
            attempt_id="RESEARCH_WITH_INSTRUCTION",
            session_id="SESSION_0001",
            card_id="CARD_0001",
            gap_id="GAP_0001",
            selection={"text": "Требование 4.16.10"},
            research_question=instruction,
            research_message_id=message_id,
        )

        context = transport.calls[0]["call"].context
        self.assertEqual(context["research_question"], {"text": instruction})
        self.assertNotIn("analyst_messages", context)
        retrieval = flow.retrieval
        self.assertEqual(
            [call.question for call in retrieval.calls],
            [instruction, instruction],
        )
        self.assertEqual(
            [call.profile.name for call in retrieval.calls],
            ["узкий поиск", "расширенный поиск"],
        )

    async def test_first_question_falls_back_to_gap_question(self) -> None:
        flow, _, retrieval = self.make_flow(
            [
                raw_tool(
                    "ask_lightrag",
                    {"question": "Модель предложила другую формулировку"},
                ),
                raw_tool(
                    "submit_gap_result",
                    {
                        "outcome": "not_found",
                        "updates": [],
                        "unknown_fields": ["test.observation.method"],
                        "missing_fact": "метод наблюдения",
                        "summary": "Метод наблюдения не найден",
                        "contradictions": [],
                    },
                    "submit",
                ),
            ],
            [RetrievalResponse("Нет точного метода", ())],
        )

        await flow.run(
            attempt_id="RESEARCH_WITH_FALLBACK",
            session_id="SESSION_0001",
            card_id="CARD_0001",
            gap_id="GAP_0001",
            selection={"text": "Требование 4.16.10"},
        )

        self.assertEqual(
            retrieval.calls[0].question,
            "Как наблюдать уменьшение счётчика SK SMI?",
        )

    async def test_structured_missing_fact_reaches_submit_callback(self) -> None:
        question = "Как прочитать значение счётчика SK SMI?"
        flow, _, _ = self.make_flow(
            [
                raw_tool("ask_lightrag", {"question": question}),
                raw_tool(
                    "submit_gap_result",
                    {
                        "outcome": "not_found",
                        "updates": [],
                        "unknown_fields": ["test.observation.method"],
                        "missing_fact": {
                            "field": "test.observation.method",
                            "description": "Подтверждённый способ чтения счётчика",
                        },
                        "summary": "Способ не найден в источниках",
                        "contradictions": [],
                    },
                    "call-2",
                ),
            ],
            [RetrievalResponse("Прямой способ не найден", ())],
        )

        result = await flow.run(
            attempt_id="RESEARCH_STRUCTURED_MISSING",
            session_id="SESSION_0001",
            card_id="CARD_0001",
            gap_id="GAP_0001",
            selection={"text": "Требование 4.16.10"},
        )

        self.assertEqual(result.outcome, "not_found")
        stop = self.sessions.history("SESSION_0001")[-1]
        self.assertIn(
            "Не удалось определить: Подтверждённый способ чтения счётчика",
            stop.text,
        )
        self.assertNotIn("{'field':", stop.text)

    async def test_resume_ignores_latest_analyst_message_and_retrieves(self) -> None:
        self.sessions.append(
            "SESSION_0001",
            SessionEventKind.ANALYST,
            "Старое сообщение не по текущей остановке",
            {"author": "Аналитик"},
        )
        self.sessions.append(
            "SESSION_0001",
            SessionEventKind.ASSISTANT,
            (
                "Исследование остановлено: требуется решение аналитика.\n"
                "Вопрос: Как наблюдать уменьшение счётчика SK SMI?"
            ),
            {"gap_id": "GAP_0001", "outcome": "not_found"},
        )
        self.sessions.append(
            "SESSION_0001",
            SessionEventKind.ANALYST,
            "Считать значение командой GET DATA",
            {"author": "Аналитик"},
        )
        flow, transport, retrieval = self.make_flow(
            [
                raw_tool(
                    "ask_lightrag",
                    {"question": "Как наблюдать уменьшение счётчика SK SMI?"},
                    "ask-1",
                ),
                raw_tool(
                    "submit_gap_result",
                    {
                        "outcome": "not_found",
                        "updates": [],
                        "unknown_fields": ["test.observation.method"],
                        "missing_fact": "Подтверждённый способ наблюдения",
                        "summary": "Способ не найден в источниках",
                        "contradictions": [],
                    },
                    "submit-1",
                )
            ],
            [RetrievalResponse("Способ не найден", ())],
        )

        result = await flow.run(
            attempt_id="RESEARCH_ANALYST_ANSWER",
            session_id="SESSION_0001",
            card_id="CARD_0001",
            gap_id="GAP_0001",
            selection={"text": "Требование 4.16.10"},
        )

        self.assertEqual(result.outcome, "not_found")
        self.assertEqual(len(retrieval.calls), 1)
        call = transport.calls[0]["call"]
        self.assertEqual(call.allowed_tools, ("ask_lightrag",))
        self.assertNotIn("analyst_messages", call.context)
        with InMemoryUnitOfWork(self.database) as uow:
            card = uow.cards.get("CARD_0001")
        self.assertEqual(card.revision, 1)
        self.assertIsNone(card.field("test.observation.method").value)

    async def test_resume_does_not_apply_tentative_question_as_evidence(
        self,
    ) -> None:
        self.sessions.append(
            "SESSION_0001",
            SessionEventKind.ASSISTANT,
            "Исследование остановлено: требуется решение аналитика.",
            {"gap_id": "GAP_0001", "outcome": "not_found"},
        )
        self.sessions.append(
            "SESSION_0001",
            SessionEventKind.ANALYST,
            "Может Get Data подойдёт?",
            {"author": "Аналитик"},
        )
        flow, transport, retrieval = self.make_flow(
            [
                raw_tool(
                    "ask_lightrag",
                    {"question": "Как наблюдать уменьшение счётчика SK SMI?"},
                    "ask-1",
                ),
                raw_tool(
                    "submit_gap_result",
                    {
                        "outcome": "not_found",
                        "updates": [],
                        "unknown_fields": ["test.observation.method"],
                        "missing_fact": "Подтверждённый способ наблюдения",
                        "summary": "Способ не найден в источниках",
                        "contradictions": [],
                    },
                    "submit-1",
                )
            ],
            [RetrievalResponse("Способ не найден", ())],
        )

        result = await flow.run(
            attempt_id="RESEARCH_TENTATIVE_RESUME",
            session_id="SESSION_0001",
            card_id="CARD_0001",
            gap_id="GAP_0001",
            selection={"text": "Требование 4.16.10"},
        )

        with InMemoryUnitOfWork(self.database) as uow:
            card = uow.cards.get("CARD_0001")
        self.assertEqual(result.outcome, "not_found")
        self.assertEqual(len(retrieval.calls), 1)
        self.assertNotIn(
            "analyst_messages",
            transport.calls[0]["call"].context,
        )
        self.assertEqual(card.revision, 1)
        self.assertIsNone(card.field("test.observation.method").value)
        self.assertFalse(
            any(
                item.quote == "Может Get Data подойдёт?"
                for item in card.evidence.values()
            )
        )

    async def test_resume_does_not_expand_short_analyst_answer(
        self,
    ) -> None:
        with InMemoryUnitOfWork(self.database) as uow:
            uow.cards.save(
                make_card(
                    allowed_paths=("test.initial_state", "test.preconditions"),
                )
            )
        self.sessions.append(
            "SESSION_0001",
            SessionEventKind.ASSISTANT,
            "Исследование остановлено: требуется решение аналитика.",
            {"gap_id": "GAP_0001", "outcome": "not_found"},
        )
        self.sessions.append(
            "SESSION_0001",
            SessionEventKind.ANALYST,
            "0b",
            {"author": "Аналитик"},
        )
        flow, transport, retrieval = self.make_flow(
            [
                raw_tool(
                    "ask_lightrag",
                    {"question": "Как наблюдать уменьшение счётчика SK SMI?"},
                    "ask-1",
                ),
                raw_tool(
                    "submit_gap_result",
                    {
                        "outcome": "not_found",
                        "updates": [],
                        "unknown_fields": [
                            "test.initial_state",
                            "test.preconditions",
                        ],
                        "missing_fact": "Подтверждённое начальное состояние",
                        "summary": "Значение не найдено в источниках",
                        "contradictions": [],
                    },
                    "submit-1",
                )
            ],
            [RetrievalResponse("Значение не найдено", ())],
        )

        result = await flow.run(
            attempt_id="RESEARCH_SHORT_RESUME",
            session_id="SESSION_0001",
            card_id="CARD_0001",
            gap_id="GAP_0001",
            selection={"text": "Требование 4.17.3"},
        )

        with InMemoryUnitOfWork(self.database) as uow:
            card = uow.cards.get("CARD_0001")
        self.assertEqual(result.outcome, "not_found")
        self.assertEqual(len(retrieval.calls), 1)
        self.assertNotIn(
            "analyst_messages",
            transport.calls[0]["call"].context,
        )
        self.assertEqual(card.revision, 1)
        self.assertIsNone(card.field("test.initial_state").value)
        self.assertIsNone(card.field("test.preconditions").value)
        self.assertFalse(any(item.quote == "0b" for item in card.evidence.values()))

    async def test_mixed_resolved_result_is_repaired_before_tool_callback(self) -> None:
        flow, transport, retrieval = self.make_flow(
            [
                raw_tool(
                    "ask_lightrag",
                    {"question": "Как наблюдать уменьшение счётчика SK SMI?"},
                    "ask",
                ),
                raw_tool(
                    "submit_gap_result",
                    {
                        "outcome": "resolved",
                        "updates": [
                            {
                                "path": "test.observation.method",
                                "value": "Сравнить счётчик до и после команды",
                                "evidence_id": "EVIDENCE_0001",
                                "analyst_message_id": None,
                            }
                        ],
                        "unknown_fields": ["test.observation.method"],
                        "missing_fact": None,
                        "summary": "Источник описывает изменение счётчика",
                        "contradictions": [],
                    },
                    "invalid",
                ),
                raw_tool(
                    "submit_gap_result",
                    {
                        "outcome": "resolved",
                        "updates": [
                            {
                                "path": "test.observation.method",
                                "value": "Сравнить счётчик до и после команды",
                                "evidence_id": "EVIDENCE_0001",
                                "analyst_message_id": None,
                            }
                        ],
                        "unknown_fields": [],
                        "missing_fact": None,
                        "summary": "Источник описывает изменение счётчика",
                        "contradictions": [],
                    },
                    "repaired",
                ),
            ],
            [RetrievalResponse("Счётчик уменьшается после команды", (exact_fragment(),))],
            max_retries=1,
        )

        result = await flow.run(
            attempt_id="RESEARCH_REPAIRED_RESOLVED",
            session_id="SESSION_0001",
            card_id="CARD_0001",
            gap_id="GAP_0001",
            selection={"text": "Требование 4.16.10"},
        )

        self.assertEqual(result.outcome, "resolved")
        self.assertEqual(len(retrieval.calls), 1)
        self.assertEqual(len(transport.calls), 3)
        repair_prompt = transport.calls[2]["call"].system_prompt
        self.assertIn(
            "resolved требует только подтверждённые обновления",
            repair_prompt,
        )

    async def test_answer_before_latest_gap_stop_is_not_reused(self) -> None:
        self.sessions.append(
            "SESSION_0001",
            SessionEventKind.ANALYST,
            "Считать значение командой GET DATA",
            {"author": "Аналитик"},
        )
        self.sessions.append(
            "SESSION_0001",
            SessionEventKind.ASSISTANT,
            (
                "Исследование остановлено: требуется решение аналитика.\n"
                "Вопрос: Как наблюдать уменьшение счётчика SK SMI?"
            ),
            {"gap_id": "GAP_0001", "outcome": "not_found"},
        )
        flow, transport, retrieval = self.make_flow(
            [
                raw_tool(
                    "ask_lightrag",
                    {"question": "Как прочитать значение счётчика SK SMI?"},
                    "call-1",
                ),
                raw_tool(
                    "submit_gap_result",
                    {
                        "outcome": "not_found",
                        "updates": [],
                        "unknown_fields": ["test.observation.method"],
                        "missing_fact": "Подтверждённый способ чтения",
                        "summary": "Способ не найден",
                        "contradictions": [],
                    },
                    "call-2",
                ),
            ],
            [RetrievalResponse("Прямой способ не найден", ())],
        )

        result = await flow.run(
            attempt_id="RESEARCH_AFTER_NEW_STOP",
            session_id="SESSION_0001",
            card_id="CARD_0001",
            gap_id="GAP_0001",
            selection={"text": "Требование 4.16.10"},
        )

        self.assertEqual(result.outcome, "not_found")
        self.assertEqual(len(retrieval.calls), 1)
        call = transport.calls[0]["call"]
        self.assertEqual(call.allowed_tools, ("ask_lightrag",))
        self.assertNotIn("analyst_messages", call.context)

    async def test_expand_repeats_same_question_with_broad_profile(self) -> None:
        question = "Как наблюдать уменьшение счётчика SK SMI?"
        flow, transport, retrieval = self.make_flow(
            [
                raw_tool("ask_lightrag", {"question": question}, "q1"),
                raw_tool("expand_lightrag", {"call_id": "q1", "reason": "нет точного адреса"}, "q2"),
                raw_tool("submit_gap_result", {
                    "outcome": "not_found", "updates": [],
                    "unknown_fields": ["test.observation.method"], "missing_fact": "способ чтения",
                    "summary": "Не найдено", "contradictions": [],
                }, "done"),
            ],
            [RetrievalResponse("мало данных", ()), RetrievalResponse("тоже не найдено", ())],
        )
        await flow.run(
            attempt_id="RESEARCH_0001", session_id="SESSION_0001",
            card_id="CARD_0001", gap_id="GAP_0001", selection={"text": "фрагмент"},
        )
        self.assertEqual([call.question for call in retrieval.calls], [question, question])
        self.assertEqual([call.profile.name for call in retrieval.calls], ["узкий поиск", "расширенный поиск"])
        self.assertEqual(
            transport.calls[1]["call"].allowed_tools,
            ("expand_lightrag", "submit_gap_result"),
        )
        self.assertEqual(
            transport.calls[2]["call"].allowed_tools,
            ("submit_gap_result",),
        )
        lightrag_results = [
            event
            for event in self.sessions.history("SESSION_0001")
            if event.metadata.get("lightrag_result")
        ]
        self.assertEqual(len(lightrag_results), 2)
        self.assertEqual(
            [event.metadata["profile"] for event in lightrag_results],
            ["узкий поиск", "расширенный поиск"],
        )
        self.assertTrue(
            all(
                event.metadata["markdown_body_start_line"] == 5
                for event in lightrag_results
            )
        )
        self.assertIn("Профиль: узкий поиск", lightrag_results[0].text)
        self.assertIn("Профиль: расширенный поиск", lightrag_results[1].text)

    async def test_broad_evidence_cannot_expand_same_call_again(self) -> None:
        question = "Как прочитать значение счётчика SK SMI?"
        flow, transport, _ = self.make_flow(
            [
                raw_tool("ask_lightrag", {"question": question}, "q1"),
                raw_tool(
                    "expand_lightrag",
                    {"call_id": "q1", "reason": "нужен точный адрес"},
                    "q2",
                ),
                raw_tool("submit_gap_result", {
                    "outcome": "not_found", "updates": [],
                    "unknown_fields": ["test.observation.method"],
                    "missing_fact": "способ чтения", "summary": "Не найдено",
                    "contradictions": [],
                }, "done"),
            ],
            [
                RetrievalResponse("мало данных", ()),
                RetrievalResponse("Найден точный фрагмент", (exact_fragment(),)),
            ],
        )

        await flow.run(
            attempt_id="RESEARCH_0001", session_id="SESSION_0001",
            card_id="CARD_0001", gap_id="GAP_0001", selection={"text": "фрагмент"},
        )

        self.assertEqual(
            transport.calls[2]["call"].allowed_tools,
            ("ask_lightrag", "submit_gap_result"),
        )

    async def test_duplicate_question_and_profile_becomes_technical_limit(self) -> None:
        question = "Как наблюдать уменьшение счётчика SK SMI?"
        flow, _, retrieval = self.make_flow(
            [
                raw_tool("ask_lightrag", {"question": question}, "q1"),
                raw_tool(
                    "ask_lightrag",
                    {"question": "  как НАБЛЮДАТЬ уменьшение СЧЁТЧИКА sk smi?  "},
                    "q2",
                ),
            ],
            [RetrievalResponse("Найден прямой фрагмент", (exact_fragment(),))],
        )

        result = await flow.run(
            attempt_id="RESEARCH_0001", session_id="SESSION_0001",
            card_id="CARD_0001", gap_id="GAP_0001", selection={"text": "фрагмент"},
        )

        self.assertEqual(result.outcome, "technical_limit")
        self.assertEqual(len(retrieval.calls), 1)
        history = self.sessions.history("SESSION_0001")
        self.assertFalse(any(event.kind is SessionEventKind.ERROR for event in history))
        self.assertIn("повтор одинакового вопроса", history[-1].text)
        self.assertIn("/continue", history[-1].text)

    async def test_response_without_exact_fragment_cannot_be_used_as_evidence(self) -> None:
        flow, _, _ = self.make_flow(
            [
                raw_tool("ask_lightrag", {"question": "Как прочитать счётчик?"}, "q1"),
                raw_tool("submit_gap_result", {
                    "outcome": "resolved",
                    "updates": [{"path": "test.observation.method", "value": "GET DATA", "evidence_id": "EVIDENCE_0001", "analyst_message_id": None}],
                    "unknown_fields": [], "missing_fact": None, "summary": "", "contradictions": [],
                }, "done"),
            ],
            [RetrievalResponse("Используйте GET DATA", ())],
        )
        with self.assertRaisesRegex(ToolContractError, "not_found"):
            await flow.run(
                attempt_id="RESEARCH_0001", session_id="SESSION_0001",
                card_id="CARD_0001", gap_id="GAP_0001", selection={"text": "фрагмент"},
            )

    async def test_tool_call_id_used_as_evidence_is_repaired_before_tool_node(self) -> None:
        question = "Как прочитать счётчик?"
        wrong_evidence_id = "chatcmpl-tool-b31f3dfcc842b2fc"
        flow, transport, _ = self.make_flow(
            [
                raw_tool(
                    "ask_lightrag",
                    {"question": question},
                    wrong_evidence_id,
                ),
                raw_tool(
                    "submit_gap_result",
                    {
                        "outcome": "resolved",
                        "updates": [
                            {
                                "path": "test.observation.method",
                                "value": "GET DATA",
                                "evidence_id": wrong_evidence_id,
                                "analyst_message_id": None,
                            }
                        ],
                        "unknown_fields": [],
                        "missing_fact": None,
                        "summary": "Способ найден",
                        "contradictions": [],
                    },
                    "invalid-submit",
                ),
                raw_tool(
                    "submit_gap_result",
                    {
                        "outcome": "resolved",
                        "updates": [
                            {
                                "path": "test.observation.method",
                                "value": "GET DATA",
                                "evidence_id": "EVIDENCE_0001",
                                "analyst_message_id": None,
                            }
                        ],
                        "unknown_fields": [],
                        "missing_fact": None,
                        "summary": "Способ найден",
                        "contradictions": [],
                    },
                    "repaired-submit",
                ),
            ],
            [RetrievalResponse("Используйте GET DATA", (exact_fragment(),))],
            max_retries=1,
        )

        result = await flow.run(
            attempt_id="RESEARCH_TOOL_ID_EVIDENCE",
            session_id="SESSION_0001",
            card_id="CARD_0001",
            gap_id="GAP_0001",
            selection={"text": "фрагмент"},
        )

        self.assertEqual(result.outcome, "resolved")
        self.assertEqual(len(transport.calls), 3)
        self.assertIn(
            wrong_evidence_id,
            transport.calls[2]["call"].system_prompt,
        )
        with InMemoryUnitOfWork(self.database) as uow:
            card = uow.cards.get("CARD_0001")
            diagnostics = [
                item
                for item in uow.records.list_kind("llm_diagnostic")
                if item.payload.get("prompt_id") == "prompt_3"
            ]
        self.assertEqual(
            card.field("test.observation.method").evidence_ids,
            ("EVIDENCE_0001",),
        )
        self.assertTrue(
            any(
                wrong_evidence_id in " ".join(item.payload.get("errors", []))
                for item in diagnostics
            )
        )

    async def test_tool_call_id_cannot_be_used_as_analyst_message(self) -> None:
        wrong_message_id = "chatcmpl-tool-943a09a1d726f3c0"
        flow, transport, _ = self.make_flow(
            [
                raw_tool(
                    "ask_lightrag",
                    {"question": "Как наблюдать уменьшение счётчика SK SMI?"},
                    "ask",
                ),
                raw_tool(
                    "submit_gap_result",
                    {
                        "outcome": "resolved",
                        "updates": [
                            {
                                "path": "test.observation.method",
                                "value": "GET DATA",
                                "evidence_id": None,
                                "analyst_message_id": wrong_message_id,
                            }
                        ],
                        "unknown_fields": [],
                        "missing_fact": None,
                        "summary": "Способ найден",
                        "contradictions": [],
                    },
                    "invalid-submit",
                ),
                raw_tool(
                    "submit_gap_result",
                    {
                        "outcome": "not_found",
                        "updates": [],
                        "unknown_fields": ["test.observation.method"],
                        "missing_fact": "Подтверждённый способ наблюдения",
                        "summary": "Способ не найден в источниках",
                        "contradictions": [],
                    },
                    "repaired-submit",
                ),
            ],
            [RetrievalResponse("Способ не найден", ())],
            max_retries=1,
        )

        result = await flow.run(
            attempt_id="RESEARCH_TOOL_ID_ANALYST",
            session_id="SESSION_0001",
            card_id="CARD_0001",
            gap_id="GAP_0001",
            selection={"text": "фрагмент"},
        )

        self.assertEqual(result.outcome, "not_found")
        self.assertEqual(len(transport.calls), 3)
        submit_schema = next(
            tool["function"]["parameters"]
            for tool in transport.calls[1]["tools"]
            if tool["function"]["name"] == "submit_gap_result"
        )
        update_properties = submit_schema["properties"]["updates"]["items"][
            "properties"
        ]
        self.assertEqual(
            submit_schema["properties"]["outcome"]["enum"],
            ["not_found"],
        )
        self.assertEqual(update_properties["path"]["enum"], ["test.observation.method"])
        self.assertEqual(update_properties["evidence_id"], {"type": "null"})
        self.assertEqual(update_properties["analyst_message_id"], {"type": "null"})
        self.assertIn(
            "'resolved' is not one of ['not_found']",
            transport.calls[2]["call"].system_prompt,
        )
        with InMemoryUnitOfWork(self.database) as uow:
            card = uow.cards.get("CARD_0001")
        self.assertFalse(
            any(item.message_id == wrong_message_id for item in card.evidence.values())
        )

    async def test_observation_without_exact_evidence_cannot_resolve_with_null_provenance(
        self,
    ) -> None:
        existing_evidence = exact_fragment().to_evidence(
            evidence_id="EVIDENCE_EXISTING",
            card_id="CARD_0001",
            selection_id="SELECTION_0001",
            collected_at=NOW,
        )
        with InMemoryUnitOfWork(self.database) as uow:
            card = uow.cards.get("CARD_0001")
            card.apply(CardMutation(evidence=(existing_evidence,)))
            uow.cards.save(card)
        invalid_update = {
            "path": "test.observation.method",
            "value": "GET DATA",
            "evidence_id": None,
            "analyst_message_id": None,
        }
        flow, transport, _ = self.make_flow(
            [
                raw_tool(
                    "ask_lightrag",
                    {"question": "Как прочитать счётчик?"},
                    "chatcmpl-tool-bd5fbef1e23c180e",
                ),
                raw_tool(
                    "submit_gap_result",
                    {
                        "outcome": "resolved",
                        "updates": [invalid_update],
                        "unknown_fields": [],
                        "missing_fact": None,
                        "summary": "Способ найден",
                        "contradictions": [],
                    },
                    "invalid-submit",
                ),
                raw_tool(
                    "submit_gap_result",
                    {
                        "outcome": "not_found",
                        "updates": [],
                        "unknown_fields": ["test.observation.method"],
                        "missing_fact": "Точный фрагмент со способом наблюдения",
                        "summary": "LightRAG не вернул exact evidence",
                        "contradictions": [],
                    },
                    "repaired-submit",
                ),
            ],
            [RetrievalResponse("Используйте GET DATA", ())],
            max_retries=1,
        )

        result = await flow.run(
            attempt_id="RESEARCH_NULL_PROVENANCE",
            session_id="SESSION_0001",
            card_id="CARD_0001",
            gap_id="GAP_0001",
            selection={"text": "фрагмент"},
        )

        self.assertEqual(result.outcome, "not_found")
        submit_schema = next(
            tool["function"]["parameters"]
            for tool in transport.calls[1]["tools"]
            if tool["function"]["name"] == "submit_gap_result"
        )
        self.assertEqual(
            submit_schema["properties"]["outcome"]["enum"],
            ["not_found"],
        )
        update_properties = submit_schema["properties"]["updates"]["items"][
            "properties"
        ]
        self.assertEqual(update_properties["evidence_id"], {"type": "null"})
        self.assertEqual(update_properties["analyst_message_id"], {"type": "null"})
        self.assertNotIn(
            "EVIDENCE_EXISTING",
            str(update_properties),
        )

    async def test_cancel_discards_late_retrieval_result(self) -> None:
        flow, _, retrieval = self.make_flow(
            [raw_tool("ask_lightrag", {"question": "Как прочитать счётчик?"}, "q1")],
            [RetrievalResponse("Ответ", (exact_fragment(),))],
        )
        retrieval.delay = 0.05
        task = asyncio.create_task(flow.run(
            attempt_id="RESEARCH_0001", session_id="SESSION_0001",
            card_id="CARD_0001", gap_id="GAP_0001", selection={"text": "фрагмент"},
        ))
        await asyncio.sleep(0.01)
        flow.cancel("RESEARCH_0001")
        with self.assertRaises(AttemptDiscardedError):
            await task
        with InMemoryUnitOfWork(self.database) as uow:
            self.assertEqual(uow.cards.get("CARD_0001").revision, 1)

    async def test_technical_limit_can_continue_with_new_attempt(self) -> None:
        flow, _, _ = self.make_flow(
            [raw_tool("ask_lightrag", {"question": "Как прочитать счётчик?"}, "q1")],
            [RetrievalResponse("нет", ())],
            max_steps=1,
        )
        result = await flow.run(
            attempt_id="RESEARCH_0001", session_id="SESSION_0001",
            card_id="CARD_0001", gap_id="GAP_0001", selection={"text": "фрагмент"},
        )
        self.assertEqual(result.outcome, "technical_limit")
        stopped = self.sessions.history("SESSION_0001")[-1]
        self.assertIn("достигнут технический предел", stopped.text)
        self.assertIn("Как наблюдать уменьшение счётчика SK SMI?", stopped.text)
        self.assertIn("/continue", stopped.text)

        second, _, _ = self.make_flow(
            [
                raw_tool(
                    "ask_lightrag",
                    {"question": "Как прочитать счётчик?"},
                    "q2",
                ),
                raw_tool("submit_gap_result", {
                    "outcome": "not_found", "updates": [],
                    "unknown_fields": ["test.observation.method"], "missing_fact": "способ",
                    "summary": "Не найдено", "contradictions": [],
                }),
            ],
            [RetrievalResponse("нет", ())],
        )
        continued = await second.run(
            attempt_id="RESEARCH_0002", session_id="SESSION_0001",
            card_id="CARD_0001", gap_id="GAP_0001", selection={"text": "фрагмент"},
        )
        self.assertEqual(continued.outcome, "not_found")

    async def test_output_token_limit_becomes_visible_technical_limit(self) -> None:
        flow, _, retrieval = self.make_flow(
            [
                RawCompletion(
                    finish_reason="length",
                    tool_calls=(),
                    usage={"completion_tokens": 1024},
                    model="fake",
                )
            ],
            [],
        )

        result = await flow.run(
            attempt_id="RESEARCH_LENGTH",
            session_id="SESSION_0001",
            card_id="CARD_0001",
            gap_id="GAP_0001",
            selection={"text": "фрагмент"},
        )

        self.assertEqual(result.outcome, "technical_limit")
        self.assertFalse(retrieval.calls)
        stopped = self.sessions.history("SESSION_0001")[-1]
        self.assertIn("достигнут технический предел", stopped.text)
        self.assertIn("/continue", stopped.text)
