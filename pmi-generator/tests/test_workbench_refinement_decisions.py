from __future__ import annotations

import unittest
from datetime import UTC, datetime

from pmi_generator.workbench.application.card_population import AnalystMessage
from pmi_generator.workbench.application.refinement import (
    CardDecisionService,
    CardRefinementService,
    RefinementArguments,
    RefinementError,
    refinement_tool,
)
from pmi_generator.workbench.application.session import SessionEventKind, SessionService
from pmi_generator.workbench.domain import (
    AnalystResolution,
    CardMutation,
    ContentField,
    Evidence,
    RelatedGap,
    SourceAddress,
    TestCard,
)
from pmi_generator.workbench.domain.schema import CARD_FIELD_PATHS
from pmi_generator.workbench.infrastructure.storage import InMemoryDatabase, InMemoryUnitOfWork
from pmi_generator.workbench.presentation.session import TerminalSessionShell


NOW = datetime(2026, 7, 16, 12, 0, tzinfo=UTC)


def make_card() -> TestCard:
    card = TestCard.create(
        card_id="CARD_0001",
        selection_id="SELECTION_0001",
        title="Проверка первого байта",
        section_number="4.16.5",
        changed_factors=("первый байт",),
        consequences=("SW 6987",),
    )
    evidence = Evidence.source_fragment(
        evidence_id="EVIDENCE_SOURCE",
        card_id=card.card_id,
        selection_id=card.selection_id,
        quote="Карта возвращает 6987.",
        address=SourceAddress("spec.pdf", "2.3", 283, 1, 2, "chunk-1"),
        collected_at=NOW,
    )
    card.apply(
        CardMutation(
            evidence=(evidence,),
            fields={"test.expected.status_word": ContentField.confirmed("6987", (evidence.evidence_id,))},
        )
    )
    return card


def add_observation_gap(card: TestCard) -> None:
    card.apply(
        CardMutation(
            gaps=(
                RelatedGap(
                    gap_id="GAP_0001",
                    card_id=card.card_id,
                    question="Как наблюдать результат?",
                    blocking_reason="Нет способа наблюдения",
                    allowed_paths=("test.observation.method",),
                    dependencies=(),
                    closure_criterion="Указан способ наблюдения",
                ),
            )
        )
    )


class RefinementTests(unittest.TestCase):
    def setUp(self) -> None:
        self.database = InMemoryDatabase()
        with InMemoryUnitOfWork(self.database) as uow:
            uow.cards.save(make_card())
        self.counter = 0
        self.service = CardRefinementService(
            uow_factory=lambda: InMemoryUnitOfWork(self.database),
            next_id=self.next_id,
            clock=lambda: NOW,
        )
        self.message = AnalystMessage(
            "MSG_0001", "CARD_0001", "Аналитик", "Ожидаемый SW должен быть 9000.", NOW
        )

    def next_id(self, prefix: str) -> str:
        self.counter += 1
        return f"{prefix}_{self.counter:04d}"

    def confirmation(self) -> dict[str, object]:
        with InMemoryUnitOfWork(self.database) as uow:
            revision = uow.cards.get("CARD_0001").revision
        return {
            "confirmation_message_id": "MSG_CONFIRM_0001",
            "proposal_id": "PROPOSAL_0001",
            "expected_revision": revision,
        }

    def test_mutating_refinement_requires_confirmation_chain(self) -> None:
        with self.assertRaisesRegex(
            RefinementError,
            "подтверждённый proposal",
        ):
            self.service.apply(
                "CARD_0001",
                RefinementArguments(
                    outcome="updated",
                    updates=[
                        {
                            "path": "test.control_values",
                            "value": ["80"],
                            "evidence_id": None,
                            "analyst_message_id": "MSG_0001",
                        }
                    ],
                    gaps=[],
                    reason="Уточнение аналитика",
                ),
                analyst_messages=(self.message,),
            )

        with InMemoryUnitOfWork(self.database) as uow:
            card = uow.cards.get("CARD_0001")
        self.assertEqual(card.revision, 1)
        self.assertIsNone(card.field("test.control_values").value)
        self.assertFalse(
            any(
                evidence.message_id == "MSG_0001"
                for evidence in card.evidence.values()
            )
        )

    def test_updated_replaces_full_value_and_creates_human_evidence(self) -> None:
        result = self.service.apply(
            "CARD_0001",
            RefinementArguments(
                outcome="updated",
                updates=[
                    {
                        "path": "test.control_values",
                        "value": ["80", "00"],
                        "evidence_id": None,
                        "analyst_message_id": "MSG_0001",
                    }
                ],
                gaps=[],
                reason="Уточнение аналитика",
            ),
            analyst_messages=(self.message,),
            **self.confirmation(),
        )
        with InMemoryUnitOfWork(self.database) as uow:
            card = uow.cards.get("CARD_0001")
        self.assertTrue(result.changed)
        self.assertEqual(card.field("test.control_values").value, ["80", "00"])
        self.assertEqual(
            card.field("test.control_values").status.value,
            "подтверждено аналитиком",
        )
        evidence = card.evidence[card.field("test.control_values").evidence_ids[0]]
        self.assertEqual(evidence.message_id, "MSG_0001")

    def test_one_analyst_message_becomes_one_evidence_for_multiple_updates(self) -> None:
        self.service.apply(
            "CARD_0001",
            RefinementArguments(
                outcome="updated",
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
                gaps=[],
                reason="Уточнение аналитика",
            ),
            analyst_messages=(self.message,),
            **self.confirmation(),
        )

        with InMemoryUnitOfWork(self.database) as uow:
            card = uow.cards.get("CARD_0001")
        data_evidence = card.field("test.command.data").evidence_ids
        action_evidence = card.field("test.action").evidence_ids
        self.assertEqual(data_evidence, action_evidence)
        human_evidence = [
            evidence
            for evidence in card.evidence.values()
            if evidence.message_id == "MSG_0001"
        ]
        self.assertEqual(len(human_evidence), 1)

    def test_unknown_path_rejects_atomic_update(self) -> None:
        with self.assertRaises(RefinementError):
            self.service.apply(
                "CARD_0001",
                RefinementArguments(
                    "updated",
                    [{"path": "card.title", "value": "Новое", "evidence_id": None, "analyst_message_id": "MSG_0001"}],
                    [], "",
                ),
                analyst_messages=(self.message,),
                **self.confirmation(),
            )
        with InMemoryUnitOfWork(self.database) as uow:
            self.assertEqual(uow.cards.get("CARD_0001").revision, 1)

    def test_gaps_created_adds_typed_gap(self) -> None:
        result = self.service.apply(
            "CARD_0001",
            RefinementArguments(
                "gaps_created",
                [],
                [{
                    "question": "Как наблюдать изменение состояния?",
                    "blocking_reason": "Нет наблюдаемого результата",
                    "allowed_paths": ["test.observation.method"],
                    "dependencies": ["test.expected.status_word"],
                    "closure_criterion": "указан способ наблюдения",
                    "resolution_mode": "design_decision",
                }],
                "Нужно исследование",
            ),
            analyst_messages=(self.message,),
            **self.confirmation(),
        )
        self.assertEqual(len(result.gap_ids), 1)
        with InMemoryUnitOfWork(self.database) as uow:
            gap = uow.cards.get("CARD_0001").gaps[result.gap_ids[0]]
        self.assertEqual(gap.resolution_mode.value, "design_decision")

    def test_refinement_splits_mixed_resolution_targets(self) -> None:
        result = self.service.apply(
            "CARD_0001",
            RefinementArguments(
                "gaps_created",
                [],
                [
                    {
                        "question": "Какие поля команды использовать?",
                        "blocking_reason": "Команда неполна",
                        "allowed_paths": [
                            "test.command.cla",
                            "test.command.data",
                        ],
                        "dependencies": ["test.action"],
                        "closure_criterion": "Поля заданы",
                        "resolution_mode": "design_decision",
                        "resolution_targets": [
                            {
                                "path": "test.command.cla",
                                "resolution_mode": "source_fact",
                                "accepted_forms": ["exact"],
                                "residual_question": (
                                    "Какое CLA задаёт источник?"
                                ),
                            },
                            {
                                "path": "test.command.data",
                                "resolution_mode": "design_decision",
                                "accepted_forms": [
                                    "exact",
                                    "deterministic_rule",
                                ],
                                "residual_question": (
                                    "Какие воспроизводимые Data использовать?"
                                ),
                            },
                        ],
                    }
                ],
                "Нужно разделить происхождение значений",
            ),
            analyst_messages=(self.message,),
            **self.confirmation(),
        )

        with InMemoryUnitOfWork(self.database) as uow:
            card = uow.cards.get("CARD_0001")
        gaps = [card.gaps[gap_id] for gap_id in result.gap_ids]
        self.assertEqual(len(gaps), 2)
        self.assertEqual(
            {
                gap.allowed_paths[0]: gap.resolution_mode.value
                for gap in gaps
            },
            {
                "test.command.cla": "source_fact",
                "test.command.data": "design_decision",
            },
        )

    def test_no_change_preserves_revision_and_decision(self) -> None:
        decisions = CardDecisionService(
            uow_factory=lambda: InMemoryUnitOfWork(self.database), clock=lambda: NOW
        )
        with InMemoryUnitOfWork(self.database) as uow:
            card = uow.cards.get("CARD_0001")
            add_observation_gap(card)
            uow.cards.save(card)
        decisions.include("CARD_0001", author="Аналитик")
        with InMemoryUnitOfWork(self.database) as uow:
            before = uow.cards.get("CARD_0001")
        result = self.service.apply(
            "CARD_0001", RefinementArguments("no_change", [], [], "Изменение не требуется"),
            analyst_messages=(self.message,),
        )
        with InMemoryUnitOfWork(self.database) as uow:
            after = uow.cards.get("CARD_0001")
        self.assertFalse(result.changed)
        self.assertEqual(after.revision, before.revision)
        self.assertEqual(after.decision, before.decision)

    def test_identical_updated_value_is_normalized_to_no_change(self) -> None:
        with InMemoryUnitOfWork(self.database) as uow:
            card = uow.cards.get("CARD_0001")
            evidence_id = card.field("test.expected.status_word").evidence_ids[0]
            revision = card.revision

        result = self.service.apply(
            "CARD_0001",
            RefinementArguments(
                "updated",
                [
                    {
                        "path": "test.expected.status_word",
                        "value": "6987",
                        "evidence_id": evidence_id,
                        "analyst_message_id": None,
                    }
                ],
                [],
                "Значение уже актуально",
            ),
            analyst_messages=(self.message,),
            **self.confirmation(),
        )

        self.assertEqual(result.outcome, "no_change")
        self.assertFalse(result.changed)
        self.assertEqual(result.revision, revision)

    def test_content_change_invalidates_decision_and_selection_review(self) -> None:
        with InMemoryUnitOfWork(self.database) as uow:
            card = uow.cards.get("CARD_0001")
            card.include_incomplete(author="Аналитик", reason="Допустимо для PoC", at=NOW)
            card.mark_selection_review_current()
            uow.cards.save(card)
        self.service.apply(
            "CARD_0001",
            RefinementArguments(
                "updated",
                [{"path": "test.action", "value": "Отправить APDU", "evidence_id": None, "analyst_message_id": "MSG_0001"}],
                [], "Уточнение",
            ),
            analyst_messages=(self.message,),
            **self.confirmation(),
        )
        with InMemoryUnitOfWork(self.database) as uow:
            card = uow.cards.get("CARD_0001")
        self.assertIsNone(card.decision)
        self.assertFalse(card.selection_review_current)

    def test_conflicting_human_value_is_kept_in_diagnostic(self) -> None:
        self.service.apply(
            "CARD_0001",
            RefinementArguments(
                "updated",
                [{"path": "test.expected.status_word", "value": "9000", "evidence_id": None, "analyst_message_id": "MSG_0001"}],
                [], "Экспертное уточнение",
            ),
            analyst_messages=(self.message,),
            **self.confirmation(),
        )
        with InMemoryUnitOfWork(self.database) as uow:
            card = uow.cards.get("CARD_0001")
            diagnostics = uow.records.list_kind("refinement_conflict")
        self.assertEqual(card.field("test.expected.status_word").value, "9000")
        self.assertEqual(len(diagnostics), 1)


class DecisionAndShellTests(unittest.TestCase):
    def setUp(self) -> None:
        self.database = InMemoryDatabase()
        with InMemoryUnitOfWork(self.database) as uow:
            uow.cards.save(make_card())
        self.decisions = CardDecisionService(
            uow_factory=lambda: InMemoryUnitOfWork(self.database), clock=lambda: NOW
        )

    def test_include_incomplete_and_exclude_are_explicit_current_revision_decisions(self) -> None:
        with InMemoryUnitOfWork(self.database) as uow:
            card = uow.cards.get("CARD_0001")
            add_observation_gap(card)
            uow.cards.save(card)
        included = self.decisions.include("CARD_0001", author="Аналитик")
        self.assertEqual(included.kind.value, "включить неполной")
        self.assertEqual(included.revision, 2)
        excluded = self.decisions.exclude("CARD_0001", author="Аналитик")
        self.assertEqual(excluded.kind.value, "исключить")
        self.assertEqual(excluded.revision, 2)

    def test_incomplete_card_without_explicit_gap_cannot_be_included(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "не содержит явного блокирующего пробела",
        ):
            self.decisions.include("CARD_0001", author="Аналитик")

    def test_ready_card_gets_regular_include_decision(self) -> None:
        with InMemoryUnitOfWork(self.database) as uow:
            card = uow.cards.get("CARD_0001")
            evidence_id = "EVIDENCE_SOURCE"
            card.apply(
                CardMutation(
                    fields={
                        "requirement.condition": ContentField.confirmed("условие", (evidence_id,)),
                        "requirement.behavior": ContentField.confirmed("поведение", (evidence_id,)),
                        "test.action": ContentField.confirmed("отправить APDU", (evidence_id,)),
                        "test.changed_factor": ContentField.confirmed("первый байт", (evidence_id,)),
                        "test.observation.method": ContentField.confirmed("проверить SW", (evidence_id,)),
                    }
                )
            )
            uow.cards.save(card)
        decision = self.decisions.include("CARD_0001", author="Аналитик")
        self.assertEqual(decision.kind.value, "включить")

    def test_session_commands_do_not_close_shell(self) -> None:
        sessions = SessionService(
            uow_factory=lambda: InMemoryUnitOfWork(self.database), clock=lambda: NOW
        )
        sessions.open("SESSION_0001", "SELECTION_0001", "CARD_0001")
        called: list[str] = []
        shell = TerminalSessionShell(
            sessions,
            "SESSION_0001",
            command_handlers={
                "/include": lambda: called.append("include"),
                "/exclude": lambda: called.append("exclude"),
            },
        )
        shell.handle_command("/include")
        shell.handle_command("/exclude")
        self.assertEqual(called, ["include", "exclude"])
        self.assertFalse(shell.controller.should_exit)
        self.assertIn("/include", shell.completer.commands)

    def test_old_revision_decision_is_not_current_after_change(self) -> None:
        with InMemoryUnitOfWork(self.database) as uow:
            card = uow.cards.get("CARD_0001")
            add_observation_gap(card)
            uow.cards.save(card)
        decision = self.decisions.include("CARD_0001", author="Аналитик")
        with InMemoryUnitOfWork(self.database) as uow:
            card = uow.cards.get("CARD_0001")
            human = Evidence.human_knowledge(
                evidence_id="EVIDENCE_HUMAN", card_id=card.card_id,
                selection_id=card.selection_id, quote="Новое", author="Аналитик",
                message_id="MSG", collected_at=NOW,
            )
            resolution = AnalystResolution(
                resolution_id="RESOLUTION_MANUAL",
                card_id=card.card_id,
                author="Аналитик",
                created_at=NOW,
                reason="Явное изменение",
                target_paths=("test.action",),
                evidence_ids=(human.evidence_id,),
                source_message_id=human.message_id,
                confirmation_message_id=human.message_id,
                values=(
                    {"path": "test.action", "value": "Новое"},
                ),
            )
            card.apply(
                CardMutation(
                    evidence=(human,),
                    fields={
                        "test.action": ContentField.analyst_confirmed(
                            "Новое",
                            (human.evidence_id,),
                        )
                    },
                    resolutions=(resolution,),
                )
            )
            uow.cards.save(card)
        self.assertFalse(self.decisions.is_current("CARD_0001", decision))


class RefinementToolTests(unittest.TestCase):
    def test_tool_contract_is_strict(self) -> None:
        spec = refinement_tool()
        self.assertEqual(spec.name, "submit_card_refinement")
        self.assertFalse(spec.json_schema["additionalProperties"])
        properties = spec.json_schema["properties"]
        self.assertEqual(
            set(properties["updates"]["items"]["properties"]["path"]["enum"]),
            set(CARD_FIELD_PATHS),
        )
        gap = properties["gaps"]["items"]["properties"]
        self.assertEqual(set(gap["allowed_paths"]["items"]["enum"]), set(CARD_FIELD_PATHS))
        self.assertEqual(set(gap["dependencies"]["items"]["enum"]), set(CARD_FIELD_PATHS))
        self.assertEqual(
            set(gap["resolution_mode"]["enum"]),
            {"source_fact", "design_decision", "external_input"},
        )
        target = gap["resolution_targets"]["items"]["properties"]
        self.assertEqual(
            set(target["path"]["enum"]),
            set(CARD_FIELD_PATHS),
        )
        self.assertEqual(
            set(target["accepted_forms"]["items"]["enum"]),
            {
                "confirmed_value",
                "exact",
                "finite_set",
                "deterministic_rule",
            },
        )
