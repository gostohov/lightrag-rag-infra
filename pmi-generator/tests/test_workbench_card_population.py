from __future__ import annotations

import asyncio
import threading
import unittest
from datetime import UTC, datetime

from pmi_generator.workbench.application.card_population import (
    CardPopulationFlow,
    PopulationArguments,
    PopulationError,
    PopulationService,
    PopulationStartController,
    population_tool,
)
from pmi_generator.workbench.application.llm import (
    AttemptDiscardedError,
    LlmToolRuntime,
    RawCompletion,
    ToolContractError,
    TypedToolRegistry,
)
from pmi_generator.workbench.application.prompting import default_policy
from pmi_generator.workbench.application.session import (
    SessionEventKind,
    SessionService,
)
from pmi_generator.workbench.domain import (
    CardMutation,
    ContentField,
    Evidence,
    SourceAddress,
    TestCard,
)
from pmi_generator.workbench.domain.schema import (
    CARD_FIELD_PATHS,
    REQUIRED_FIELD_PATHS,
)
from pmi_generator.workbench.infrastructure.llm import ScriptedLlmTransport
from pmi_generator.workbench.infrastructure.storage import InMemoryDatabase, InMemoryUnitOfWork


NOW = datetime(2026, 7, 16, 12, 0, tzinfo=UTC)


def make_card(card_id: str = "CARD_0001") -> TestCard:
    return TestCard.create(
        card_id=card_id,
        selection_id="SELECTION_0001",
        title="Проверка первого байта",
        section_number="4.16.5",
        changed_factors=("первый байт",),
        consequences=("возврат 6987",),
    )


def source_evidence(card_id: str = "CARD_0001") -> Evidence:
    return Evidence.source_fragment(
        evidence_id="EVIDENCE_0001",
        card_id=card_id,
        selection_id="SELECTION_0001",
        quote="Если первый байт не равен 81, карта возвращает 6987.",
        address=SourceAddress(
            document_id="spec_2.3.pdf",
            document_version="2.3",
            page=283,
            line_start=1,
            line_end=2,
            chunk_id="section-0270",
        ),
        collected_at=NOW,
    )


def apdu_source_evidence(card_id: str = "CARD_0001") -> Evidence:
    return Evidence.source_fragment(
        evidence_id="EVIDENCE_APDU",
        card_id=card_id,
        selection_id="SELECTION_0001",
        quote=(
            "CLA имеет значение 0C, INS имеет значение DA, Lc равно длине "
            "поля Data, Le имеет значение 00."
        ),
        address=SourceAddress(
            document_id="spec_2.3.pdf",
            document_version="2.3",
            page=279,
            line_start=10,
            line_end=13,
            chunk_id="section-4.16.1",
        ),
        collected_at=NOW,
    )


def valid_arguments() -> PopulationArguments:
    return PopulationArguments(
        source_values=[
            {
                "path": "requirement.condition",
                "value": "первый байт не равен 81",
                "evidence_id": "EVIDENCE_0001",
            },
            {
                "path": "requirement.behavior",
                "value": "карта прекращает выполнение команды",
                "evidence_id": "EVIDENCE_0001",
            },
            {
                "path": "test.action",
                "value": "отправить PUT DATA с неверным первым байтом",
                "evidence_id": "EVIDENCE_0001",
            },
            {
                "path": "test.observation.method",
                "value": "проверить SW1SW2",
                "evidence_id": "EVIDENCE_0001",
            },
            {
                "path": "test.expected.status_word",
                "value": "6987",
                "evidence_id": "EVIDENCE_0001",
            },
        ],
        analyst_values=[],
        derivations=[
            {
                "path": "test.changed_factor",
                "value": "первый байт данных",
                "source_evidence_ids": ["EVIDENCE_0001"],
                "rule": "из условия меняется только первый байт",
                "scope": "текущая карточка",
            }
        ],
        not_applicable=[
            {"path": "test.command.le", "reason": "Поле Le не относится к проверке"}
        ],
        gaps=[
            {
                "question": "Какое конкретное неверное значение использовать?",
                "blocking_reason": "Без значения нельзя выполнить команду",
                "allowed_paths": ["test.control_values"],
                "dependencies": ["requirement.condition"],
                "closure_criterion": "выбрано одно подтверждённое значение",
                "resolution_mode": "source_fact",
            }
        ],
    )


class PopulationServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.database = InMemoryDatabase()
        with InMemoryUnitOfWork(self.database) as uow:
            uow.cards.save(make_card())
        self.counts: dict[str, int] = {"EVIDENCE": 1}
        self.service = PopulationService(
            uow_factory=lambda: InMemoryUnitOfWork(self.database),
            next_id=self.next_id,
            clock=lambda: NOW,
        )

    def next_id(self, prefix: str) -> str:
        self.counts[prefix] = self.counts.get(prefix, 0) + 1
        return f"{prefix}_{self.counts[prefix]:04d}"

    def test_four_valid_sections_apply_as_one_revision(self) -> None:
        result = self.service.apply(
            "CARD_0001",
            valid_arguments(),
            available_evidence=(source_evidence(),),
        )

        with InMemoryUnitOfWork(self.database) as uow:
            card = uow.cards.get("CARD_0001")
        self.assertEqual(result.revision, 1)
        self.assertEqual(card.revision, 1)
        self.assertEqual(card.field("requirement.condition").value, "первый байт не равен 81")
        self.assertEqual(card.field("test.command.le").status.value, "не применимо")
        self.assertEqual(tuple(card.gaps), ("GAP_0001",))

    def test_mixed_resolution_targets_become_independent_typed_gaps(
        self,
    ) -> None:
        payload = valid_arguments()
        payload.not_applicable = []
        source_paths = {
            "test.command.cla",
            "test.command.ins",
            "test.command.lc",
            "test.command.le",
        }
        design_paths = {
            "test.command.p1",
            "test.command.p2",
            "test.command.data",
        }
        all_paths = (
            "test.command.cla",
            "test.command.ins",
            "test.command.p1",
            "test.command.p2",
            "test.command.lc",
            "test.command.data",
            "test.command.le",
        )
        payload.gaps.append(
            {
                "question": "Какие поля APDU использовать?",
                "blocking_reason": "Без APDU команда невоспроизводима",
                "allowed_paths": list(all_paths),
                "dependencies": ("test.action",),
                "closure_criterion": "Все поля APDU заданы",
                "resolution_mode": "design_decision",
                "resolution_targets": [
                    {
                        "path": path,
                        "resolution_mode": (
                            "source_fact"
                            if path in source_paths
                            else "design_decision"
                        ),
                        "accepted_forms": (
                            ["exact"]
                            if path in source_paths
                            else [
                                "exact",
                                "finite_set",
                                "deterministic_rule",
                            ]
                        ),
                        "residual_question": f"Укажите {path}.",
                    }
                    for path in all_paths
                ],
            }
        )

        result = self.service.apply(
            "CARD_0001",
            payload,
            available_evidence=(source_evidence(),),
        )

        with InMemoryUnitOfWork(self.database) as uow:
            card = uow.cards.get("CARD_0001")
        split = [
            gap
            for gap in card.gaps.values()
            if set(gap.allowed_paths) & set(all_paths)
        ]
        self.assertEqual(result.revision, 1)
        self.assertEqual(len(split), len(all_paths))
        self.assertEqual(
            {gap.allowed_paths[0] for gap in split},
            set(all_paths),
        )
        self.assertEqual(
            {
                gap.allowed_paths[0]
                for gap in split
                if gap.resolution_mode.value == "source_fact"
            },
            source_paths,
        )
        self.assertEqual(
            {
                gap.allowed_paths[0]
                for gap in split
                if gap.resolution_mode.value == "design_decision"
            },
            design_paths,
        )
        self.assertEqual(
            [gap.resolution_mode.value for gap in split],
            [
                "source_fact",
                "source_fact",
                "source_fact",
                "source_fact",
                "design_decision",
                "design_decision",
                "design_decision",
            ],
        )
        data_gap = next(
            gap
            for gap in split
            if gap.allowed_paths == ("test.command.data",)
        )
        self.assertEqual(
            {item.value for item in data_gap.closure_contract.requirements[0].accepted_forms},
            {"exact", "finite_set", "deterministic_rule"},
        )

    def test_resolution_targets_must_cover_allowed_paths_exactly(self) -> None:
        payload = valid_arguments()
        payload.gaps[0]["resolution_targets"] = [
            {
                "path": "test.command.data",
                "resolution_mode": "design_decision",
                "accepted_forms": ["exact"],
                "residual_question": "Укажите Data.",
            }
        ]

        with self.assertRaisesRegex(
            PopulationError,
            "resolution targets.*allowed_paths",
        ):
            self.service.apply(
                "CARD_0001",
                payload,
                available_evidence=(source_evidence(),),
            )
        with InMemoryUnitOfWork(self.database) as uow:
            self.assertEqual(uow.cards.get("CARD_0001").revision, 0)

    def test_multi_path_gap_without_typed_targets_is_rejected(self) -> None:
        payload = valid_arguments()
        payload.gaps[0]["allowed_paths"] = [
            "test.control_values",
            "test.command.data",
        ]

        with self.assertRaisesRegex(
            PopulationError,
            "Multi-path gap требует typed resolution targets",
        ):
            self.service.apply(
                "CARD_0001",
                payload,
                available_evidence=(source_evidence(),),
            )
        with InMemoryUnitOfWork(self.database) as uow:
            self.assertEqual(uow.cards.get("CARD_0001").revision, 0)

    def test_source_apdu_facts_remain_distinct_from_design_targets(
        self,
    ) -> None:
        payload = valid_arguments()
        payload.not_applicable = []
        payload.source_values.extend(
            [
                {
                    "path": "test.command.cla",
                    "value": "0C",
                    "evidence_id": "EVIDENCE_APDU",
                },
                {
                    "path": "test.command.ins",
                    "value": "DA",
                    "evidence_id": "EVIDENCE_APDU",
                },
                {
                    "path": "test.command.lc",
                    "value": {
                        "rule": "length_of",
                        "field": "test.command.data",
                    },
                    "evidence_id": "EVIDENCE_APDU",
                },
                {
                    "path": "test.command.le",
                    "value": "00",
                    "evidence_id": "EVIDENCE_APDU",
                },
            ]
        )
        design_paths = (
            "test.command.p1",
            "test.command.p2",
            "test.command.data",
        )
        payload.gaps.append(
            {
                "question": "Какой тестовый вариант APDU использовать?",
                "blocking_reason": "Источник не выбирает тестовый объект",
                "allowed_paths": list(design_paths),
                "dependencies": ["test.action"],
                "closure_criterion": "Выбран воспроизводимый вариант",
                "resolution_mode": "design_decision",
                "resolution_targets": [
                    {
                        "path": path,
                        "resolution_mode": "design_decision",
                        "accepted_forms": [
                            "exact",
                            "finite_set",
                            "deterministic_rule",
                        ],
                        "residual_question": f"Выберите {path}.",
                    }
                    for path in design_paths
                ],
            }
        )

        self.service.apply(
            "CARD_0001",
            payload,
            available_evidence=(
                source_evidence(),
                apdu_source_evidence(),
            ),
        )

        with InMemoryUnitOfWork(self.database) as uow:
            card = uow.cards.get("CARD_0001")
        for path, expected in {
            "test.command.cla": "0C",
            "test.command.ins": "DA",
            "test.command.le": "00",
        }.items():
            self.assertEqual(card.field(path).value, expected)
            self.assertEqual(
                card.field(path).status.value,
                "подтверждено источником",
            )
            evidence = card.evidence[
                card.field(path).evidence_ids[0]
            ]
            self.assertEqual(evidence.quote, apdu_source_evidence().quote)
            self.assertEqual(evidence.address.page, 279)
            self.assertEqual(evidence.address.line_start, 10)
        self.assertEqual(
            card.field("test.command.lc").value,
            {
                "rule": "length_of",
                "field": "test.command.data",
            },
        )
        self.assertEqual(
            {
                gap.allowed_paths[0]
                for gap in card.gaps.values()
                if gap.resolution_mode.value == "design_decision"
            },
            set(design_paths),
        )

    def test_path_cannot_appear_in_two_sections(self) -> None:
        payload = valid_arguments()
        payload.not_applicable.append(
            {"path": "requirement.condition", "reason": "конфликт"}
        )

        with self.assertRaisesRegex(PopulationError, "нескольких разделах"):
            self.service.apply(
                "CARD_0001", payload, available_evidence=(source_evidence(),)
            )

    def test_unknown_evidence_rejects_entire_result(self) -> None:
        payload = valid_arguments()
        payload.source_values[0]["evidence_id"] = "EVIDENCE_UNKNOWN"

        with self.assertRaisesRegex(PopulationError, "evidence"):
            self.service.apply("CARD_0001", payload, available_evidence=())
        with InMemoryUnitOfWork(self.database) as uow:
            self.assertEqual(uow.cards.get("CARD_0001").revision, 0)

    def test_derivation_requires_sources_rule_and_scope(self) -> None:
        for field in ("source_evidence_ids", "rule", "scope"):
            with self.subTest(field=field):
                payload = valid_arguments()
                payload.derivations[0][field] = [] if field == "source_evidence_ids" else ""
                with self.assertRaises(PopulationError):
                    self.service.apply(
                        "CARD_0001",
                        payload,
                        available_evidence=(source_evidence(),),
                    )

    def test_gap_requires_mode_closure_criterion_and_allowed_paths(self) -> None:
        for field in ("resolution_mode", "closure_criterion", "allowed_paths"):
            with self.subTest(field=field):
                payload = valid_arguments()
                payload.gaps[0][field] = [] if field == "allowed_paths" else ""
                with self.assertRaises(PopulationError):
                    self.service.apply(
                        "CARD_0001",
                        payload,
                        available_evidence=(source_evidence(),),
                    )

    def test_analyst_values_are_rejected_without_mutation(self) -> None:
        payload = valid_arguments()
        payload.analyst_values.append(
            {
                "path": "test.control_values",
                "value": ["80"],
                "analyst_message_id": "MSG_0001",
            }
        )
        payload.gaps = []
        with self.assertRaisesRegex(
            PopulationError,
            "analyst_values",
        ):
            self.service.apply(
                "CARD_0001",
                payload,
                available_evidence=(source_evidence(),),
            )

        with InMemoryUnitOfWork(self.database) as uow:
            card = uow.cards.get("CARD_0001")
        self.assertEqual(card.revision, 0)
        self.assertFalse(card.evidence)

    def test_repair_coverage_creates_gaps_for_uncovered_blocking_fields(self) -> None:
        card = make_card()
        evidence = source_evidence()
        card.apply(
            CardMutation(
                evidence=(evidence,),
                fields={
                    "requirement.condition": ContentField.confirmed(
                        "первый байт не равен 81",
                        (evidence.evidence_id,),
                    ),
                    "test.action": ContentField.confirmed(
                        "отправить PUT DATA",
                        (evidence.evidence_id,),
                    ),
                    "test.changed_factor": ContentField.confirmed(
                        "первый байт",
                        (evidence.evidence_id,),
                    ),
                    "test.expected.status_word": ContentField.confirmed(
                        "6987",
                        (evidence.evidence_id,),
                    ),
                },
            )
        )
        with InMemoryUnitOfWork(self.database) as uow:
            uow.cards.save(card)

        result = self.service.repair_coverage("CARD_0001")

        with InMemoryUnitOfWork(self.database) as uow:
            repaired = uow.cards.get("CARD_0001")
        self.assertEqual(result.revision, 2)
        self.assertEqual(len(result.open_gap_ids), 2)
        allowed_paths = {
            gap.allowed_paths
            for gap in repaired.gaps.values()
        }
        self.assertEqual(
            allowed_paths,
            {
                ("requirement.behavior",),
                ("test.observation.method",),
            },
        )

    def test_invalid_source_result_rolls_back_card(self) -> None:
        payload = valid_arguments()
        payload.derivations[0]["source_evidence_ids"] = ["UNKNOWN"]

        with self.assertRaises(PopulationError):
            self.service.apply("CARD_0001", payload, available_evidence=())

        with InMemoryUnitOfWork(self.database) as uow:
            card = uow.cards.get("CARD_0001")
        self.assertEqual(card.revision, 0)
        self.assertFalse(card.evidence)


class PopulationToolContractTests(unittest.TestCase):
    def test_prompt_2_wire_contract_has_no_analyst_values(self) -> None:
        schema = population_tool().openai_schema()["function"]["parameters"]

        self.assertNotIn("analyst_values", schema["properties"])
        self.assertNotIn("analyst_values", schema["required"])

    def test_all_card_paths_are_constrained_to_domain_schema(self) -> None:
        schema = population_tool().json_schema
        properties = schema["properties"]
        path_schemas = (
            properties["source_values"]["items"]["properties"]["path"],
            properties["derivations"]["items"]["properties"]["path"],
            properties["gaps"]["items"]["properties"]["allowed_paths"]["items"],
            properties["gaps"]["items"]["properties"]["dependencies"]["items"],
            properties["gaps"]["items"]["properties"]["resolution_targets"][
                "items"
            ]["properties"]["path"],
        )

        for path_schema in path_schemas:
            self.assertEqual(set(path_schema["enum"]), set(CARD_FIELD_PATHS))

        not_applicable_paths = set(
            properties["not_applicable"]["items"]["properties"]["path"]["enum"]
        )
        self.assertEqual(
            not_applicable_paths,
            set(CARD_FIELD_PATHS) - set(REQUIRED_FIELD_PATHS),
        )

    def test_wire_schema_avoids_vllm_unsupported_array_keywords(self) -> None:
        wire_schema = population_tool().openai_schema()["function"]["parameters"]
        wire_text = str(wire_schema)
        source_item = wire_schema["properties"]["source_values"]["items"]

        self.assertNotIn("uniqueItems", wire_text)
        self.assertNotIn("minItems", wire_text)
        self.assertNotIn("minLength", wire_text)
        self.assertNotIn("oneOf", wire_text)
        self.assertEqual(
            set(source_item["required"]),
            {"path", "value", "evidence_id"},
        )
        properties = wire_schema["properties"]
        self.assertNotIn("null", str(source_item))
        self.assertIn("прямо подтверждённые", properties["source_values"]["description"])
        self.assertIn("выведенные", properties["derivations"]["description"])
        self.assertIn("отсутствие значения", properties["not_applicable"]["description"])
        self.assertIn("не хватает основания", properties["gaps"]["description"])

    def test_live_prompt_2_unknown_paths_are_rejected_before_domain_apply(self) -> None:
        tools = TypedToolRegistry()
        tools.register(population_tool())
        arguments = {
            "source_values": [
                {
                    "path": "condition",
                    "value": "Первый байт не равен 81",
                    "evidence_id": "EVIDENCE_0001",
                },
                {
                    "path": "input_value",
                    "value": "00",
                    "evidence_id": None,
                },
                {
                    "path": "consequences[0].text",
                    "value": "Карта прекращает выполнение команды",
                    "evidence_id": "EVIDENCE_0001",
                },
            ],
            "derivations": [],
            "not_applicable": [],
            "gaps": [],
        }

        with self.assertRaises(ToolContractError):
            tools.decode(
                {
                    "id": "live-call",
                    "name": "submit_card_population",
                    "arguments": arguments,
                },
                ("submit_card_population",),
            )

    def test_confirmed_value_requires_exactly_one_grounding_reference(self) -> None:
        tools = TypedToolRegistry()
        tools.register(population_tool())
        arguments = {
            "source_values": [
                {
                    "path": "test.control_values",
                    "value": "00",
                    "evidence_id": None,
                }
            ],
            "derivations": [],
            "not_applicable": [],
            "gaps": [],
        }

        with self.assertRaises(ToolContractError):
            tools.decode(
                {
                    "id": "ungrounded-call",
                    "name": "submit_card_population",
                    "arguments": arguments,
                },
                ("submit_card_population",),
            )

    def test_population_result_must_cover_every_blocking_field(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "requirement.behavior.*test.observation.method",
        ):
            PopulationArguments(
                source_values=[
                    {
                        "path": "requirement.condition",
                        "value": "первый байт не равен 81",
                        "evidence_id": "EVIDENCE_0001",
                    },
                    {
                        "path": "test.action",
                        "value": "отправить PUT DATA",
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
                ],
                analyst_values=[],
                derivations=[],
                not_applicable=[],
                gaps=[],
            )

    def test_duplicate_paths_across_sections_are_rejected_at_tool_boundary(
        self,
    ) -> None:
        for label, mutate in (
            (
                "duplicate_not_applicable",
                lambda payload: payload["not_applicable"].append(
                    {
                        "path": "test.command.le",
                        "reason": "Повторная классификация того же поля",
                    }
                ),
            ),
            (
                "not_applicable_and_gap",
                lambda payload: payload["not_applicable"].append(
                    {
                        "path": "test.control_values",
                        "reason": "Значение отсутствует в источнике",
                    }
                ),
            ),
        ):
            with self.subTest(label=label):
                tools = TypedToolRegistry()
                tools.register(population_tool())
                arguments = valid_arguments().as_dict()
                mutate(arguments)

                with self.assertRaisesRegex(
                    ToolContractError,
                    "присутствует в нескольких разделах",
                ):
                    tools.decode(
                        {
                            "id": f"{label}-call",
                            "name": "submit_card_population",
                            "arguments": arguments,
                        },
                        ("submit_card_population",),
                    )


class PopulationFlowTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.database = InMemoryDatabase()
        with InMemoryUnitOfWork(self.database) as uow:
            uow.cards.save(make_card())
        self.session = SessionService(
            uow_factory=lambda: InMemoryUnitOfWork(self.database), clock=lambda: NOW
        )
        self.session.open("SESSION_0001", "SELECTION_0001", "CARD_0001")

    def make_flow(self, scripted: list[object]) -> tuple[CardPopulationFlow, ScriptedLlmTransport]:
        tools = TypedToolRegistry()
        tools.register(population_tool())
        transport = ScriptedLlmTransport(scripted)
        runtime = LlmToolRuntime(
            transport=transport,
            tools=tools,
            uow_factory=lambda: InMemoryUnitOfWork(self.database),
        )
        service = PopulationService(
            uow_factory=lambda: InMemoryUnitOfWork(self.database),
            next_id=lambda prefix: f"{prefix}_0001",
            clock=lambda: NOW,
        )
        return (
            CardPopulationFlow(
                policy=default_policy(),
                runtime=runtime,
                service=service,
                sessions=self.session,
            ),
            transport,
        )

    async def test_prompt_context_contains_only_current_card_and_no_lightrag(self) -> None:
        self.session.append(
            "SESSION_0001",
            SessionEventKind.ANALYST,
            "Может использовать 80?",
            {"message_id": "MSG_TENTATIVE", "author": "Аналитик"},
        )
        payload = valid_arguments()
        response = RawCompletion(
            finish_reason="tool_calls",
            tool_calls=(
                {
                    "id": "call-1",
                    "name": "submit_card_population",
                    "arguments": payload.as_dict(),
                },
            ),
            usage={},
            model="fake",
        )
        flow, transport = self.make_flow([response])

        await flow.run(
            attempt_id="ATTEMPT_0001",
            session_id="SESSION_0001",
            card_id="CARD_0001",
            selection={"text": "выбранный текст"},
            skeleton={"title": "каркас"},
            available_evidence=(source_evidence(),),
        )

        context = transport.calls[0]["call"].context
        self.assertEqual(context["card"]["card_id"], "CARD_0001")
        self.assertNotIn("other_cards", context)
        self.assertNotIn("lightrag", context)
        self.assertNotIn("analyst_messages", context)

    async def test_live_ungrounded_action_is_repaired_before_population_apply(
        self,
    ) -> None:
        rejected_arguments = valid_arguments().as_dict()
        rejected_arguments["source_values"] = [
            item
            for item in rejected_arguments["source_values"]
            if item["path"] != "test.action"
        ]
        rejected_arguments["source_values"].extend(
            [
                {
                    "path": "requirement.consequences",
                    "value": ["Карта возвращает 6987"],
                    "evidence_id": "EVIDENCE_0001",
                },
                {
                    "path": "test.action",
                    "value": (
                        "Выполнить команду PUT DATA с первым байтом данных, "
                        "отличным от '81'"
                    ),
                    "evidence_id": None,
                },
            ]
        )
        rejected = RawCompletion(
            finish_reason="tool_calls",
            tool_calls=(
                {
                    "id": "live-call",
                    "name": "submit_card_population",
                    "arguments": rejected_arguments,
                },
            ),
            usage={},
            model="fake",
        )
        accepted = RawCompletion(
            finish_reason="tool_calls",
            tool_calls=(
                {
                    "id": "repair-call",
                    "name": "submit_card_population",
                    "arguments": valid_arguments().as_dict(),
                },
            ),
            usage={},
            model="fake",
        )
        flow, transport = self.make_flow([rejected, accepted])

        result = await flow.run(
            attempt_id="ATTEMPT_LIVE_REPAIR",
            session_id="SESSION_0001",
            card_id="CARD_0001",
            selection={"text": "выбранный текст"},
            skeleton={"title": "каркас"},
            available_evidence=(source_evidence(),),
        )

        self.assertEqual(result.revision, 1)
        self.assertEqual(len(transport.calls), 2)
        self.assertIn(
            "source_values",
            transport.calls[1]["call"].system_prompt,
        )
        diagnostic = self.database.records[
            ("llm_diagnostic", "ATTEMPT_LIVE_REPAIR")
        ]
        self.assertEqual(diagnostic.payload["retry"], 1)
        self.assertEqual(len(diagnostic.payload["rejected_tool_calls"]), 1)

    async def test_incomplete_population_is_repaired_before_domain_apply(self) -> None:
        rejected_arguments = valid_arguments().as_dict()
        rejected_arguments["source_values"] = [
            item
            for item in rejected_arguments["source_values"]
            if item["path"] not in {"requirement.behavior", "test.observation.method"}
        ]
        rejected = RawCompletion(
            finish_reason="tool_calls",
            tool_calls=(
                {
                    "id": "incomplete-call",
                    "name": "submit_card_population",
                    "arguments": rejected_arguments,
                },
            ),
            usage={},
            model="fake",
        )
        accepted = RawCompletion(
            finish_reason="tool_calls",
            tool_calls=(
                {
                    "id": "repair-call",
                    "name": "submit_card_population",
                    "arguments": valid_arguments().as_dict(),
                },
            ),
            usage={},
            model="fake",
        )
        flow, transport = self.make_flow([rejected, accepted])

        result = await flow.run(
            attempt_id="ATTEMPT_COVERAGE_REPAIR",
            session_id="SESSION_0001",
            card_id="CARD_0001",
            selection={"text": "выбранный текст"},
            skeleton={"title": "каркас"},
            available_evidence=(source_evidence(),),
        )

        self.assertEqual(result.revision, 1)
        self.assertEqual(len(transport.calls), 2)
        self.assertIn(
            "не покрывает обязательные поля",
            transport.calls[1]["call"].system_prompt,
        )
        repair_prompt = transport.calls[1]["call"].system_prompt
        self.assertIn("requirement.condition=source_values", repair_prompt)
        self.assertIn("requirement.behavior=не покрыто", repair_prompt)
        self.assertIn("test.observation.method=не покрыто", repair_prompt)
        self.assertIn(
            "сохрани корректно покрытые обязательные поля",
            repair_prompt,
        )

    async def test_duplicate_live_paths_are_repaired_before_population_apply(
        self,
    ) -> None:
        rejected_arguments = valid_arguments().as_dict()
        rejected_arguments["source_values"] = [
            item
            for item in rejected_arguments["source_values"]
            if item["path"] != "test.observation.method"
        ]
        rejected_arguments["not_applicable"].extend(
            [
                {
                    "path": "test.command.ins",
                    "reason": "Команда PUT DATA не имеет поля INS",
                },
                {
                    "path": "test.command.ins",
                    "reason": "Инструкция команды фиксирована",
                },
                {
                    "path": "test.control_values",
                    "reason": "Конкретное значение отсутствует в источнике",
                },
            ]
        )
        rejected = RawCompletion(
            finish_reason="tool_calls",
            tool_calls=(
                {
                    "id": "live-duplicate-call",
                    "name": "submit_card_population",
                    "arguments": rejected_arguments,
                },
            ),
            usage={},
            model="fake",
        )
        accepted = RawCompletion(
            finish_reason="tool_calls",
            tool_calls=(
                {
                    "id": "repair-call",
                    "name": "submit_card_population",
                    "arguments": valid_arguments().as_dict(),
                },
            ),
            usage={},
            model="fake",
        )
        flow, transport = self.make_flow([rejected, accepted])

        result = await flow.run(
            attempt_id="ATTEMPT_DUPLICATE_REPAIR",
            session_id="SESSION_0001",
            card_id="CARD_0001",
            selection={"text": "выбранный текст"},
            skeleton={"title": "каркас"},
            available_evidence=(source_evidence(),),
        )

        self.assertEqual(result.revision, 1)
        self.assertEqual(len(transport.calls), 2)
        self.assertIn(
            "присутствует в нескольких разделах",
            transport.calls[1]["call"].system_prompt,
        )
        self.assertIn(
            "test.observation.method=не покрыто",
            transport.calls[1]["call"].system_prompt,
        )

    async def test_cancelled_attempt_does_not_create_initial_revision(self) -> None:
        response = RawCompletion(
            finish_reason="tool_calls",
            tool_calls=(
                {
                    "id": "call-1",
                    "name": "submit_card_population",
                    "arguments": valid_arguments().as_dict(),
                },
            ),
            usage={},
            model="fake",
        )
        flow, _ = self.make_flow([(0.05, response)])
        pending = asyncio.create_task(
            flow.run(
                attempt_id="ATTEMPT_CANCEL",
                session_id="SESSION_0001",
                card_id="CARD_0001",
                selection={},
                skeleton={},
                available_evidence=(source_evidence(),),
            )
        )
        await asyncio.sleep(0.01)
        flow.cancel("ATTEMPT_CANCEL")

        with self.assertRaises(AttemptDiscardedError):
            await pending
        with InMemoryUnitOfWork(self.database) as uow:
            self.assertEqual(uow.cards.get("CARD_0001").revision, 0)

    def test_cancel_after_response_ready_wins_before_domain_apply(self) -> None:
        response = RawCompletion(
            finish_reason="tool_calls",
            tool_calls=(
                {
                    "id": "call-1",
                    "name": "submit_card_population",
                    "arguments": valid_arguments().as_dict(),
                },
            ),
            usage={},
            model="fake",
        )
        flow, _ = self.make_flow([response])
        reached_apply = threading.Event()
        release_apply = threading.Event()
        apply_result = flow.runtime.apply_result

        def blocked_apply_result(attempt_id: str, operation: object) -> object:
            reached_apply.set()
            if not release_apply.wait(1):
                raise TimeoutError("Тест не освободил apply_result")
            return apply_result(attempt_id, operation)  # type: ignore[arg-type]

        flow.runtime.apply_result = blocked_apply_result  # type: ignore[method-assign]
        errors: list[BaseException] = []

        def execute() -> None:
            try:
                asyncio.run(
                    flow.run(
                        attempt_id="ATTEMPT_READY_CANCEL",
                        session_id="SESSION_0001",
                        card_id="CARD_0001",
                        selection={},
                        skeleton={},
                        available_evidence=(source_evidence(),),
                    )
                )
            except BaseException as error:
                errors.append(error)

        worker = threading.Thread(target=execute)
        worker.start()
        self.assertTrue(reached_apply.wait(1))

        flow.cancel("ATTEMPT_READY_CANCEL")
        release_apply.set()
        worker.join(1)

        self.assertFalse(worker.is_alive())
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], AttemptDiscardedError)
        with InMemoryUnitOfWork(self.database) as uow:
            self.assertEqual(uow.cards.get("CARD_0001").revision, 0)

    async def test_continue_and_instruction_allocate_new_attempt_ids(self) -> None:
        attempt_ids = iter(("A1", "A2", "A3"))
        controller = PopulationStartController(lambda: next(attempt_ids))

        first = controller.start()
        second = controller.continue_without_message()
        third = controller.with_instruction("Уточни параметр")

        self.assertEqual((first.attempt_id, second.attempt_id, third.attempt_id), ("A1", "A2", "A3"))
        self.assertIsNone(second.instruction)
        self.assertEqual(third.instruction, "Уточни параметр")


if __name__ == "__main__":
    unittest.main()
