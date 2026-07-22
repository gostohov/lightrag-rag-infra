from __future__ import annotations

import json
import unittest

from pmi_generator.workbench.domain import (
    CardMutation,
    GapClosureContract,
    GapClosureOutcome,
    GapPathClosure,
    GapValueForm,
    RelatedGap,
)
from pmi_generator.workbench.domain.errors import DomainValidationError
from pmi_generator.workbench.infrastructure.storage.card_codec import (
    decode_card,
    encode_card,
)
from pmi_generator.workbench.infrastructure.storage import StorageError
from tests.test_workbench_persistence import make_card


class GapClosureContractTests(unittest.TestCase):
    def test_legacy_contract_accepts_confirmed_values(self) -> None:
        contract = GapClosureContract.legacy(
            ("test.control_values",),
            question="Какие контрольные значения использовать?",
        )

        evaluation = contract.evaluate(
            {"test.control_values": ["80"]},
            source_confirmed=False,
        )

        self.assertIs(evaluation.outcome, GapClosureOutcome.SATISFIED)
        self.assertEqual(evaluation.satisfied_paths, ("test.control_values",))
        self.assertEqual(evaluation.remaining_paths, ())

    def test_reproducible_contract_rejects_untyped_placeholder(self) -> None:
        contract = GapClosureContract(
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
        )

        evaluation = contract.evaluate(
            {"test.command.data": "произвольные байты"},
            source_confirmed=False,
        )

        self.assertIs(evaluation.outcome, GapClosureOutcome.INSUFFICIENT)
        self.assertEqual(evaluation.satisfied_paths, ())
        self.assertEqual(
            evaluation.remaining_questions,
            (
                "Укажите точные байты, конечный набор или "
                "детерминированное правило генерации Data.",
            ),
        )

    def test_reproducible_contract_accepts_each_explicit_form(self) -> None:
        contract = GapClosureContract(
            requirements=(
                GapPathClosure(
                    path="test.command.data",
                    accepted_forms=(
                        GapValueForm.EXACT_VALUE,
                        GapValueForm.FINITE_SET,
                        GapValueForm.DETERMINISTIC_RULE,
                    ),
                    residual_question="Нужны воспроизводимые Data.",
                ),
            ),
        )
        values = (
            {"kind": "exact", "value": "80"},
            {"kind": "finite_set", "values": ["80", "7F"]},
            {
                "kind": "deterministic_rule",
                "rule": "Повторить байт 80 восемь раз",
                "parameters": {"byte": "80", "count": 8},
            },
        )

        for value in values:
            with self.subTest(value=value):
                evaluation = contract.evaluate(
                    {"test.command.data": value},
                    source_confirmed=False,
                )
                self.assertIs(
                    evaluation.outcome,
                    GapClosureOutcome.SATISFIED,
                )

    def test_explicit_confirmed_value_is_saved_without_satisfying_exact(self) -> None:
        contract = GapClosureContract(
            requirements=(
                GapPathClosure(
                    path="test.command.data",
                    accepted_forms=(GapValueForm.EXACT_VALUE,),
                    residual_question="Нужно точное значение.",
                ),
            ),
        )
        value = {
            "kind": "confirmed_value",
            "value": "произвольные байты",
        }

        evaluation = contract.evaluate(
            {"test.command.data": value},
            source_confirmed=False,
        )

        self.assertIs(evaluation.outcome, GapClosureOutcome.INSUFFICIENT)
        self.assertEqual(
            contract.normalize_values(
                {"test.command.data": value},
                source_confirmed=False,
            ),
            {"test.command.data": "произвольные байты"},
        )

    def test_malformed_explicit_form_is_rejected(self) -> None:
        contract = GapClosureContract(
            requirements=(
                GapPathClosure(
                    path="test.command.data",
                    accepted_forms=(GapValueForm.DETERMINISTIC_RULE,),
                    residual_question="Нужно правило генерации.",
                ),
            ),
        )

        with self.assertRaisesRegex(
            DomainValidationError,
            "deterministic_rule требует rule и parameters",
        ):
            contract.evaluate(
                {
                    "test.command.data": {
                        "kind": "deterministic_rule",
                        "rule": "",
                    }
                },
                source_confirmed=False,
            )

    def test_source_confirmed_value_satisfies_reproducible_contract(self) -> None:
        contract = GapClosureContract(
            requirements=(
                GapPathClosure(
                    path="test.command.cla",
                    accepted_forms=(GapValueForm.EXACT_VALUE,),
                    residual_question="Нужно точное CLA.",
                ),
            ),
        )

        evaluation = contract.evaluate(
            {"test.command.cla": "0C"},
            source_confirmed=True,
        )

        self.assertIs(evaluation.outcome, GapClosureOutcome.SATISFIED)

    def test_replacing_satisfied_path_with_insufficient_value_removes_progress(
        self,
    ) -> None:
        contract = GapClosureContract(
            requirements=(
                GapPathClosure(
                    path="test.command.data",
                    accepted_forms=(GapValueForm.EXACT_VALUE,),
                    residual_question="Нужно точное значение Data.",
                ),
                GapPathClosure(
                    path="test.command.p1",
                    accepted_forms=(GapValueForm.EXACT_VALUE,),
                    residual_question="Нужно точное значение P1.",
                ),
            ),
        )

        evaluation = contract.evaluate(
            {
                "test.command.data": "произвольные байты",
                "test.command.p1": {"kind": "exact", "value": "BF"},
            },
            source_confirmed=False,
            previously_satisfied=("test.command.data",),
        )

        self.assertIs(
            evaluation.outcome,
            GapClosureOutcome.PARTIALLY_SATISFIED,
        )
        self.assertEqual(
            evaluation.satisfied_paths,
            ("test.command.p1",),
        )
        self.assertEqual(
            evaluation.remaining_paths,
            ("test.command.data",),
        )

    def test_source_confirmed_null_is_rejected(self) -> None:
        contract = GapClosureContract(
            requirements=(
                GapPathClosure(
                    path="test.command.data",
                    accepted_forms=(GapValueForm.EXACT_VALUE,),
                    residual_question="Нужно точное значение Data.",
                ),
            ),
        )

        with self.assertRaisesRegex(
            DomainValidationError,
            "не может быть null",
        ):
            contract.evaluate(
                {"test.command.data": None},
                source_confirmed=True,
            )


class GapClosureCodecTests(unittest.TestCase):
    def test_round_trip_preserves_versioned_contract_and_progress(self) -> None:
        card = make_card()
        gap = RelatedGap(
            gap_id="GAP_TYPED",
            card_id=card.card_id,
            question="Какие Data использовать?",
            blocking_reason="Без Data тест невоспроизводим",
            allowed_paths=("test.command.data",),
            dependencies=(),
            closure_criterion="Заданы воспроизводимые Data",
            closure_contract=GapClosureContract(
                requirements=(
                    GapPathClosure(
                        path="test.command.data",
                        accepted_forms=(
                            GapValueForm.EXACT_VALUE,
                            GapValueForm.DETERMINISTIC_RULE,
                        ),
                        residual_question="Укажите точные Data.",
                    ),
                ),
            ),
            closure_satisfied_paths=(),
        )
        card.apply(CardMutation(gaps=(gap,)))

        restored = decode_card(encode_card(card))

        restored_gap = restored.gaps["GAP_TYPED"]
        self.assertEqual(restored_gap.closure_contract.schema_version, 1)
        self.assertEqual(
            restored_gap.closure_contract.requirements,
            gap.closure_contract.requirements,
        )
        self.assertEqual(restored_gap.closure_satisfied_paths, ())

    def test_legacy_payload_gets_compatible_contract(self) -> None:
        card = make_card()
        card.apply(
            CardMutation(
                gaps=(
                    RelatedGap(
                        gap_id="GAP_LEGACY",
                        card_id=card.card_id,
                        question="Какие контрольные значения использовать?",
                        blocking_reason="Контрольные значения неизвестны",
                        allowed_paths=("test.control_values",),
                        dependencies=(),
                        closure_criterion="Значение подтверждено",
                    ),
                )
            )
        )
        raw = json.loads(encode_card(card))
        raw_gap = raw["gaps"][0]
        raw_gap.pop("closure_contract", None)
        raw_gap.pop("closure_satisfied_paths", None)

        restored = decode_card(json.dumps(raw, ensure_ascii=False))

        gap = next(iter(restored.gaps.values()))
        self.assertEqual(
            gap.closure_contract,
            GapClosureContract.legacy(
                gap.allowed_paths,
                question=gap.question,
            ),
        )
        self.assertEqual(gap.closure_satisfied_paths, ())

    def test_invalid_closure_contract_is_reported_as_storage_error(self) -> None:
        card = make_card()
        card.apply(
            CardMutation(
                gaps=(
                    RelatedGap(
                        gap_id="GAP_INVALID",
                        card_id=card.card_id,
                        question="Какие Data использовать?",
                        blocking_reason="Data неизвестны",
                        allowed_paths=("test.command.data",),
                        dependencies=(),
                        closure_criterion="Data заданы",
                    ),
                )
            )
        )
        raw = json.loads(encode_card(card))
        raw["gaps"][0]["closure_contract"]["schema_version"] = 99

        with self.assertRaisesRegex(
            StorageError,
            "Некорректная запись карточки",
        ):
            decode_card(json.dumps(raw, ensure_ascii=False))


if __name__ == "__main__":
    unittest.main()
