from __future__ import annotations

import unittest
from datetime import UTC, datetime

from pmi_generator.workbench.domain import (
    AnalystResolution,
    CardDecisionKind,
    CardMutation,
    ContentField,
    Derivation,
    DomainValidationError,
    Evidence,
    EvidenceScopeError,
    EpistemicStatus,
    GapResolutionMode,
    PathNotAllowedError,
    RelatedGap,
    SourceAddress,
    TestCard,
)


NOW = datetime(2026, 7, 16, 12, 0, tzinfo=UTC)


def source_evidence(
    evidence_id: str = "EVIDENCE_0001",
    card_id: str = "CARD_0001",
    selection_id: str = "SELECTION_0001",
) -> Evidence:
    return Evidence.source_fragment(
        evidence_id=evidence_id,
        card_id=card_id,
        selection_id=selection_id,
        quote="Если MAC совпал, уменьшить счетчик SK SMI.",
        address=SourceAddress(
            document_id="spec_2.3.pdf",
            document_version="2.3",
            page=284,
            line_start=31,
            line_end=34,
            chunk_id="section-0270",
        ),
        collected_at=NOW,
    )


def card(card_id: str = "CARD_0001") -> TestCard:
    return TestCard.create(
        card_id=card_id,
        selection_id="SELECTION_0001",
        title="Уменьшение счетчика SK SMI",
        section_number="4.16.5",
        changed_factors=("корректность MAC",),
        consequences=("счетчик SK SMI уменьшен", "команда продолжает обработку"),
    )


class ContentFieldTests(unittest.TestCase):
    def test_confirmed_field_requires_evidence_reference(self) -> None:
        with self.assertRaisesRegex(DomainValidationError, "evidence"):
            ContentField.confirmed("MAC совпал", evidence_ids=())

    def test_derived_field_requires_derivation_reference(self) -> None:
        with self.assertRaisesRegex(DomainValidationError, "вывода"):
            ContentField.derived("PUT DATA", derivation_id="")

    def test_analyst_confirmed_field_has_distinct_status(self) -> None:
        field = ContentField.analyst_confirmed(
            "80",
            evidence_ids=("EVIDENCE_HUMAN_0001",),
        )

        self.assertIs(
            field.status,
            EpistemicStatus.ANALYST_CONFIRMED,
        )


class TestCardInvariantTests(unittest.TestCase):
    def test_multiple_consequences_are_allowed_for_one_changed_factor(self) -> None:
        test_card = card()

        self.assertEqual(len(test_card.consequences), 2)
        self.assertEqual(test_card.changed_factor, "корректность MAC")

    def test_two_changed_factors_are_rejected(self) -> None:
        with self.assertRaisesRegex(DomainValidationError, "изменяемый фактор"):
            TestCard.create(
                card_id="CARD_0001",
                selection_id="SELECTION_0001",
                title="Неатомарная проверка",
                section_number="4.16.5",
                changed_factors=("MAC", "Lc"),
                consequences=("команда отклонена",),
            )

    def test_foreign_card_evidence_cannot_support_field(self) -> None:
        test_card = card()
        foreign = source_evidence(card_id="CARD_0002")
        mutation = CardMutation(
            evidence=(foreign,),
            fields={
                "requirement.condition": ContentField.confirmed(
                    "MAC совпал",
                    evidence_ids=(foreign.evidence_id,),
                )
            },
        )

        with self.assertRaises(EvidenceScopeError):
            test_card.apply(mutation)

    def test_selection_evidence_can_support_another_card_in_same_selection(self) -> None:
        test_card = card("CARD_0002")
        shared = Evidence.selection_fragment(
            evidence_id="EVIDENCE_SELECTION_0001",
            selection_id=test_card.selection_id,
            quote="Команда возвращает 6987.",
            address=SourceAddress(
                document_id="spec_2.3.pdf",
                document_version="2.3",
                page=284,
                line_start=31,
                line_end=34,
            ),
            collected_at=NOW,
        )

        changed = test_card.apply(
            CardMutation(
                evidence=(shared,),
                fields={
                    "test.expected.status_word": ContentField.confirmed(
                        "6987",
                        evidence_ids=(shared.evidence_id,),
                    )
                },
            )
        )

        self.assertTrue(changed)

    def test_gap_rejects_update_outside_allowed_paths(self) -> None:
        test_card = card()
        gap = RelatedGap(
            gap_id="GAP_0001",
            card_id=test_card.card_id,
            question="Как наблюдать уменьшение счетчика?",
            blocking_reason="Без наблюдения результат нельзя проверить",
            allowed_paths=("test.observation.method",),
            dependencies=("test.expected.state_change",),
            closure_criterion="Указан подтвержденный способ наблюдения",
        )
        test_card.apply(CardMutation(gaps=(gap,)))

        with self.assertRaises(PathNotAllowedError):
            test_card.resolve_gap(
                "GAP_0001",
                CardMutation(fields={"test.action": ContentField.unknown()}),
                closure_satisfied_paths=("test.observation.method",),
            )

    def test_open_gap_resolution_mode_change_increments_revision(self) -> None:
        test_card = card()
        gap = RelatedGap(
            gap_id="GAP_0001",
            card_id=test_card.card_id,
            question="Как наблюдать изменение?",
            blocking_reason="Способ не выбран",
            allowed_paths=("test.observation.method",),
            dependencies=(),
            closure_criterion="Способ выбран",
        )
        test_card.apply(CardMutation(gaps=(gap,)))
        revision = test_card.revision

        changed = test_card.change_gap_resolution_mode(
            gap.gap_id,
            GapResolutionMode.DESIGN_DECISION,
        )

        self.assertTrue(changed)
        self.assertEqual(test_card.revision, revision + 1)
        self.assertIs(
            test_card.gaps[gap.gap_id].resolution_mode,
            GapResolutionMode.DESIGN_DECISION,
        )
        self.assertFalse(
            test_card.change_gap_resolution_mode(
                gap.gap_id,
                GapResolutionMode.DESIGN_DECISION,
            )
        )

    def test_analyst_resolution_is_card_local(self) -> None:
        first = card("CARD_0001")
        second = card("CARD_0002")
        knowledge = Evidence.human_knowledge(
            evidence_id="EVIDENCE_HUMAN_0001",
            card_id=first.card_id,
            selection_id=first.selection_id,
            quote="Для проверки использовать первый байт 80.",
            author="Аналитик",
            message_id="MESSAGE_0001",
            collected_at=NOW,
        )
        resolution = AnalystResolution(
            resolution_id="RESOLUTION_0001",
            card_id=first.card_id,
            author="Аналитик",
            created_at=NOW,
            reason="Экспертное знание",
            target_paths=("test.command.data",),
            evidence_ids=(knowledge.evidence_id,),
        )
        mutation = CardMutation(
            evidence=(knowledge,),
            fields={
                "test.command.data": ContentField.analyst_confirmed(
                    "80",
                    evidence_ids=(knowledge.evidence_id,),
                )
            },
        )

        first.apply_analyst_resolution(resolution, mutation)
        with self.assertRaises(EvidenceScopeError):
            second.apply_analyst_resolution(resolution, mutation)

        self.assertEqual(first.field("test.command.data").value, "80")

    def test_analyst_resolution_requires_human_knowledge(self) -> None:
        test_card = card()
        evidence = source_evidence()
        resolution = AnalystResolution(
            resolution_id="RESOLUTION_0001",
            card_id=test_card.card_id,
            author="Аналитик",
            created_at=NOW,
            reason="Экспертное решение",
            target_paths=("test.command.data",),
            evidence_ids=(evidence.evidence_id,),
        )

        with self.assertRaisesRegex(DomainValidationError, "экспертное знание"):
            test_card.apply_analyst_resolution(
                resolution,
                CardMutation(
                    evidence=(evidence,),
                    fields={
                        "test.command.data": ContentField.analyst_confirmed(
                            "80",
                            evidence_ids=(evidence.evidence_id,),
                        )
                    },
                ),
            )

    def test_source_confirmed_field_rejects_human_knowledge(self) -> None:
        test_card = card()
        knowledge = Evidence.human_knowledge(
            evidence_id="EVIDENCE_HUMAN_0001",
            card_id=test_card.card_id,
            selection_id=test_card.selection_id,
            quote="Использовать 80.",
            author="Аналитик",
            message_id="MSG_0001",
            collected_at=NOW,
        )

        with self.assertRaisesRegex(
            DomainValidationError,
            "источник",
        ):
            test_card.apply(
                CardMutation(
                    evidence=(knowledge,),
                    fields={
                        "test.command.data": ContentField.confirmed(
                            "80",
                            (knowledge.evidence_id,),
                        )
                    },
                )
            )

    def test_confirmed_status_cannot_bypass_required_evidence(self) -> None:
        test_card = card()

        with self.assertRaisesRegex(
            DomainValidationError,
            "требует evidence",
        ):
            test_card.apply(
                CardMutation(
                    fields={
                        "test.action": ContentField(
                            status=EpistemicStatus.SOURCE_CONFIRMED,
                            value="APDU",
                        )
                    }
                )
            )

    def test_analyst_confirmed_field_requires_matching_resolution(self) -> None:
        test_card = card()
        knowledge = Evidence.human_knowledge(
            evidence_id="EVIDENCE_HUMAN_0001",
            card_id=test_card.card_id,
            selection_id=test_card.selection_id,
            quote="Использовать 80.",
            author="Аналитик",
            message_id="MSG_0001",
            collected_at=NOW,
        )

        with self.assertRaisesRegex(
            DomainValidationError,
            "решение аналитика",
        ):
            test_card.apply(
                CardMutation(
                    evidence=(knowledge,),
                    fields={
                        "test.command.data": ContentField.analyst_confirmed(
                            "80",
                            (knowledge.evidence_id,),
                        )
                    },
                )
            )

    def test_analyst_resolution_rejects_mixed_evidence(self) -> None:
        test_card = card()
        source = source_evidence()
        knowledge = Evidence.human_knowledge(
            evidence_id="EVIDENCE_HUMAN_0001",
            card_id=test_card.card_id,
            selection_id=test_card.selection_id,
            quote="Использовать 80.",
            author="Аналитик",
            message_id="MSG_0001",
            collected_at=NOW,
        )
        resolution = AnalystResolution(
            resolution_id="RESOLUTION_0001",
            card_id=test_card.card_id,
            author="Аналитик",
            created_at=NOW,
            reason="Подтверждение",
            target_paths=("test.control_values",),
            evidence_ids=(knowledge.evidence_id, source.evidence_id),
        )

        with self.assertRaisesRegex(
            DomainValidationError,
            "только экспертное знание",
        ):
            test_card.apply(
                CardMutation(
                    evidence=(source, knowledge),
                    resolutions=(resolution,),
                )
            )

    def test_analyst_confirmed_field_must_match_resolution_value(self) -> None:
        test_card = card()
        knowledge = Evidence.human_knowledge(
            evidence_id="EVIDENCE_HUMAN_0001",
            card_id=test_card.card_id,
            selection_id=test_card.selection_id,
            quote="Использовать 80.",
            author="Аналитик",
            message_id="MSG_0001",
            collected_at=NOW,
        )
        resolution = AnalystResolution(
            resolution_id="RESOLUTION_0001",
            card_id=test_card.card_id,
            author="Аналитик",
            created_at=NOW,
            reason="Подтверждение",
            target_paths=("test.control_values",),
            evidence_ids=(knowledge.evidence_id,),
            values=(
                {"path": "test.control_values", "value": ["80"]},
            ),
        )

        with self.assertRaisesRegex(
            DomainValidationError,
            "решение аналитика",
        ):
            test_card.apply(
                CardMutation(
                    evidence=(knowledge,),
                    fields={
                        "test.control_values": (
                            ContentField.analyst_confirmed(
                                ["00"],
                                (knowledge.evidence_id,),
                            )
                        )
                    },
                    resolutions=(resolution,),
                )
            )

    def test_content_change_increments_revision_and_invalidates_decisions(self) -> None:
        test_card = card()
        gap = RelatedGap(
            gap_id="GAP_0001",
            card_id=test_card.card_id,
            question="Как наблюдать результат?",
            blocking_reason="Нет тестового оракула",
            allowed_paths=("test.observation.method",),
            dependencies=(),
            closure_criterion="Найден способ наблюдения",
        )
        test_card.apply(CardMutation(gaps=(gap,)))
        test_card.include_incomplete(author="Аналитик", reason="Для технического артефакта", at=NOW)
        test_card.mark_selection_review_current()
        previous_revision = test_card.revision

        evidence = source_evidence()
        changed = test_card.apply(
            CardMutation(
                evidence=(evidence,),
                fields={
                    "test.action": ContentField.confirmed(
                        "Выполнить PUT DATA",
                        evidence_ids=(evidence.evidence_id,),
                    )
                },
            )
        )

        self.assertTrue(changed)
        self.assertEqual(test_card.revision, previous_revision + 1)
        self.assertIsNone(test_card.decision)
        self.assertFalse(test_card.selection_review_current)

    def test_no_change_preserves_revision_and_decisions(self) -> None:
        test_card = card()
        test_card.include_incomplete(author="Аналитик", reason="Осознанная неполнота", at=NOW)
        test_card.mark_selection_review_current()
        revision = test_card.revision
        decision = test_card.decision

        changed = test_card.apply(CardMutation())

        self.assertFalse(changed)
        self.assertEqual(test_card.revision, revision)
        self.assertEqual(test_card.decision, decision)
        self.assertTrue(test_card.selection_review_current)

    def test_incomplete_card_can_be_included_only_with_explicit_decision(self) -> None:
        test_card = card()
        gap = RelatedGap(
            gap_id="GAP_0001",
            card_id=test_card.card_id,
            question="Как наблюдать результат?",
            blocking_reason="Нет тестового оракула",
            allowed_paths=("test.observation.method",),
            dependencies=(),
            closure_criterion="Найден способ наблюдения",
        )
        test_card.apply(CardMutation(gaps=(gap,)))

        self.assertFalse(test_card.is_ready)
        with self.assertRaisesRegex(DomainValidationError, "не готова"):
            test_card.include(author="Аналитик", at=NOW)

        test_card.include_incomplete(author="Аналитик", reason="Пробел принят", at=NOW)

        self.assertEqual(test_card.decision.kind, CardDecisionKind.INCLUDE_INCOMPLETE)
        self.assertEqual(test_card.decision.revision, test_card.revision)

    def test_derived_value_requires_existing_derivation_and_evidence(self) -> None:
        test_card = card()
        evidence = source_evidence()
        derivation = Derivation(
            derivation_id="DERIVATION_0001",
            card_id=test_card.card_id,
            source_evidence_ids=(evidence.evidence_id,),
            rule="Если поле CLA равно 0C, используется script processing",
            scope="Текущая команда PUT DATA",
        )

        changed = test_card.apply(
            CardMutation(
                evidence=(evidence,),
                derivations=(derivation,),
                fields={
                    "test.command.cla": ContentField.derived(
                        "0C",
                        derivation_id=derivation.derivation_id,
                    )
                },
            )
        )

        self.assertTrue(changed)
        self.assertEqual(test_card.field("test.command.cla").value, "0C")


if __name__ == "__main__":
    unittest.main()
