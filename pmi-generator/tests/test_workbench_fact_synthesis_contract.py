from __future__ import annotations

import unittest
from dataclasses import asdict

from pmi_generator.workbench.application.decomposition import (
    SemanticSynthesisArguments,
    SemanticWindowArguments,
    semantic_synthesis_tool,
    semantic_window_context,
    semantic_window_tool,
)
from pmi_generator.workbench.application.decomposition.windowing.plan import (
    DecompositionWindow,
    WindowSourceLine,
)
from pmi_generator.workbench.application.llm import (
    ToolContractError,
    TypedToolRegistry,
)
from pmi_generator.workbench.domain.source import SourcePosition


def window() -> DecompositionWindow:
    return DecompositionWindow(
        window_id="SELECTION_1:WINDOW:0001",
        index=0,
        lines=(
            WindowSourceLine(
                SourcePosition(297, 10),
                "Проверить, не заблокирован ли апплет",
                True,
            ),
            WindowSourceLine(
                SourcePosition(297, 11),
                "Если бит Application Blocked равен 1b,",
                True,
            ),
            WindowSourceLine(
                SourcePosition(297, 12),
                "карта возвращает SW1 SW2 = 6283.",
                True,
            ),
        ),
        global_start=SourcePosition(297, 10),
        global_end=SourcePosition(297, 12),
        outline_node_id="4.19",
        outline_label="4.19 SELECT",
        outline_path=("4", "4.19"),
        input_max_lines=256,
        input_max_estimated_tokens=12_000,
        estimated_tokens=100,
        output_max_tokens=4096,
        output_budget_tokens=3072,
        estimated_output_tokens=500,
        policy_version="policy",
    )


def facts() -> SemanticWindowArguments:
    return SemanticWindowArguments(
        behaviors=[
            {
                "title": "Выбор заблокированного приложения",
                "summary": (
                    "SELECT заблокированного приложения завершается 6283"
                ),
                "facts": [
                    {
                        "text": "Бит Application Blocked равен 1b",
                        "line_ids": ["L0002"],
                    },
                    {
                        "text": "Карта возвращает SW1 SW2 = 6283",
                        "line_ids": ["L0003"],
                    },
                ],
            }
        ]
    )


def synthesis_context() -> dict[str, object]:
    return {
        "synthesis": {
            "target_window_id": "SELECTION_1:WINDOW:0001",
            "target_fragments": [
                {
                    "fragment_id": "FRAGMENT_1",
                    "title": "Выбор заблокированного приложения",
                    "summary": (
                        "SELECT заблокированного приложения завершается 6283"
                    ),
                    "target_facts": [
                        {
                            "fact_id": "FACT_1",
                            "text": "Бит Application Blocked равен 1b",
                        },
                        {
                            "fact_id": "FACT_2",
                            "text": "Карта возвращает SW1 SW2 = 6283",
                        },
                    ],
                    "supporting_facts": [],
                }
            ],
        }
    }


def synthesis() -> SemanticSynthesisArguments:
    return SemanticSynthesisArguments(
        candidates=[
            {
                "title": "SELECT заблокированного приложения",
                "condition": {
                    "text": "Выбираемое приложение заблокировано",
                    "fact_ids": ["FACT_1"],
                },
                "changed_factor": {
                    "text": "Значение бита Application Blocked",
                    "fact_ids": ["FACT_1"],
                },
                "action": {
                    "text": "Выполнить SELECT заблокированного приложения",
                    "fact_ids": ["FACT_1"],
                },
                "consequences": [
                    {
                        "text": "Карта возвращает SW1 SW2 = 6283",
                        "fact_ids": ["FACT_2"],
                    },
                ],
            }
        ]
    )


def property_names(schema: object) -> set[str]:
    result: set[str] = set()
    if isinstance(schema, dict):
        properties = schema.get("properties")
        if isinstance(properties, dict):
            result.update(str(item) for item in properties)
        for value in schema.values():
            result.update(property_names(value))
    elif isinstance(schema, list):
        for value in schema:
            result.update(property_names(value))
    return result


class FactExtractionContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = TypedToolRegistry()
        self.registry.register(semantic_window_tool())
        self.context = {"window": semantic_window_context(window())}

    def test_window_tool_accepts_only_fact_fragments_and_line_ids(self) -> None:
        decoded = self.registry.decode(
            {
                "id": "call-facts",
                "name": "submit_semantic_window_result",
                "arguments": asdict(facts()),
            },
            ("submit_semantic_window_result",),
            context=self.context,
        )

        self.assertEqual(decoded.arguments, facts())
        names = property_names(
            semantic_window_tool().schema_for(self.context)
        )
        self.assertTrue(
            {"behaviors", "title", "summary", "facts", "text", "line_ids"}
            <= names
        )
        self.assertTrue(
            {
                "parts",
                "role",
                "missing_parts",
                "boundary_needs",
                "direction",
                "missing_part",
                "target_paths",
                "page",
                "line",
            }.isdisjoint(names)
        )

    def test_vdi_repair_shape_is_rejected_before_application_logic(self) -> None:
        payload = asdict(facts())
        payload["behaviors"][0]["parts"] = [
            {
                "role": "action",
                "text": "Отправка команды SELECT",
                "line_ids": ["L0001"],
            }
        ]
        payload["behaviors"][0]["missing_parts"] = ["action"]
        payload["behaviors"][0]["boundary_needs"] = [
            {
                "direction": "before",
                "missing_part": "changed_factor",
                "line_ids": ["L0002"],
            }
        ]

        with self.assertRaises(ToolContractError):
            self.registry.decode(
                {
                    "id": "call-old-vdi-repair",
                    "name": "submit_semantic_window_result",
                    "arguments": payload,
                },
                ("submit_semantic_window_result",),
                context=self.context,
            )


class SemanticSynthesisContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = TypedToolRegistry()
        self.registry.register(semantic_synthesis_tool())
        self.context = synthesis_context()

    def test_synthesis_uses_only_semantic_parts_and_known_fact_ids(self) -> None:
        decoded = self.registry.decode(
            {
                "id": "call-synthesis",
                "name": "submit_semantic_synthesis",
                "arguments": asdict(synthesis()),
            },
            ("submit_semantic_synthesis",),
            context=self.context,
        )

        self.assertEqual(decoded.arguments, synthesis())
        names = property_names(
            semantic_synthesis_tool().schema_for(self.context)
        )
        self.assertTrue(
            {
                "candidates",
                "title",
                "condition",
                "changed_factor",
                "input_value",
                "action",
                "consequences",
                "text",
                "fact_ids",
            }
            <= names
        )
        self.assertTrue(
            {
                "parts",
                "role",
                "line_ids",
                "missing_parts",
                "boundary_needs",
                "target_paths",
                "source_ranges",
                "primary_line_assessments",
                "candidate_id",
                "gap_id",
                "page",
                "line",
            }.isdisjoint(names)
        )

    def test_standalone_neighbor_is_not_a_tool_fact_source(self) -> None:
        self.context["synthesis"]["supporting_context"] = [
            {
                "fragment_id": "FRAGMENT_NEIGHBOR",
                "facts": [
                    {
                        "fact_id": "FACT_NEIGHBOR",
                        "text": "Соседняя независимая проверка",
                    }
                ],
            }
        ]
        payload = asdict(synthesis())
        payload["candidates"][0]["condition"]["fact_ids"] = [
            "FACT_NEIGHBOR"
        ]

        with self.assertRaisesRegex(ToolContractError, "FACT_NEIGHBOR"):
            self.registry.decode(
                {
                    "id": "call-neighbor-fact",
                    "name": "submit_semantic_synthesis",
                    "arguments": payload,
                },
                ("submit_semantic_synthesis",),
                context=self.context,
            )

    def test_unknown_fact_id_is_rejected_by_contextual_schema(self) -> None:
        payload = asdict(synthesis())
        payload["candidates"][0]["condition"]["fact_ids"] = ["FACT_UNKNOWN"]

        with self.assertRaisesRegex(ToolContractError, "FACT_UNKNOWN"):
            self.registry.decode(
                {
                    "id": "call-unknown-fact",
                    "name": "submit_semantic_synthesis",
                    "arguments": payload,
                },
                ("submit_semantic_synthesis",),
                context=self.context,
            )

    def test_vdi_duplicate_condition_payload_is_rejected_by_schema(
        self,
    ) -> None:
        payload = {
            "candidates": [
                {
                    "title": "Проверка предусловий UPDATE RECORD",
                    "parts": [
                        {
                            "role": "condition",
                            "text": "Транзакция инициирована через GPO",
                            "fact_ids": ["FACT_1"],
                        },
                        {
                            "role": "condition",
                            "text": "Соблюдены условия доступа",
                            "fact_ids": ["FACT_2"],
                        },
                    ],
                }
            ]
        }

        with self.assertRaises(ToolContractError):
            self.registry.decode(
                {
                    "id": "call-old-duplicate-condition",
                    "name": "submit_semantic_synthesis",
                    "arguments": payload,
                },
                ("submit_semantic_synthesis",),
                context=self.context,
            )

    def test_vdi_compound_condition_uses_both_confirmed_facts(self) -> None:
        payload = asdict(synthesis())
        candidate = payload["candidates"][0]
        candidate["condition"] = {
            "text": (
                "Транзакция инициирована через GPO и соблюдены условия доступа"
            ),
            "fact_ids": ["FACT_1", "FACT_2"],
        }

        decoded = self.registry.decode(
            {
                "id": "call-compound-condition",
                "name": "submit_semantic_synthesis",
                "arguments": payload,
            },
            ("submit_semantic_synthesis",),
            context=self.context,
        )

        self.assertEqual(
            decoded.arguments.candidates[0]["condition"]["fact_ids"],
            ["FACT_1", "FACT_2"],
        )

    def test_required_named_slot_cannot_be_omitted(self) -> None:
        payload = asdict(synthesis())
        del payload["candidates"][0]["changed_factor"]

        with self.assertRaisesRegex(ToolContractError, "changed_factor"):
            self.registry.decode(
                {
                    "id": "call-missing-changed-factor",
                    "name": "submit_semantic_synthesis",
                    "arguments": payload,
                },
                ("submit_semantic_synthesis",),
                context=self.context,
            )

    def test_optional_named_slots_may_be_omitted(self) -> None:
        payload = asdict(synthesis())
        del payload["candidates"][0]["action"]

        decoded = self.registry.decode(
            {
                "id": "call-without-optional-slots",
                "name": "submit_semantic_synthesis",
                "arguments": payload,
            },
            ("submit_semantic_synthesis",),
            context=self.context,
        )

        self.assertNotIn("action", decoded.arguments.candidates[0])


if __name__ == "__main__":
    unittest.main()
