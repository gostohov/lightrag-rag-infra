from __future__ import annotations

import unittest
from dataclasses import asdict

from pmi_generator.workbench.application.decomposition import (
    SemanticWindowArguments,
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
                SourcePosition(274, 38),
                "Карта прекращает выполнение команды",
                True,
            ),
            WindowSourceLine(
                SourcePosition(275, 1),
                "и возвращает ответ об ошибке 6A80",
                True,
            ),
            WindowSourceLine(
                SourcePosition(275, 2),
                "Следующее требование",
                False,
            ),
        ),
        global_start=SourcePosition(274, 38),
        global_end=SourcePosition(275, 2),
        outline_node_id="4.15",
        outline_label="4.15",
        outline_path=("4", "4.15"),
        input_max_lines=256,
        input_max_estimated_tokens=12_000,
        estimated_tokens=100,
        output_max_tokens=4096,
        output_budget_tokens=3072,
        estimated_output_tokens=500,
        policy_version="policy",
    )


def semantic_arguments() -> SemanticWindowArguments:
    return SemanticWindowArguments(
        behaviors=[
            {
                "title": "Отклонение команды при некорректных данных",
                "summary": "Карта прекращает команду и возвращает 6A80",
                "facts": [
                    {
                        "text": "Карта прекращает выполнение команды",
                        "line_ids": ["L0001"],
                    },
                    {
                        "text": "Карта возвращает 6A80",
                        "line_ids": ["L0002"],
                    },
                ],
            }
        ]
    )


class SemanticWindowContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = TypedToolRegistry()
        self.registry.register(semantic_window_tool())
        self.context = {"window": semantic_window_context(window())}

    def decode(self, arguments: dict[str, object]):
        return self.registry.decode(
            {
                "id": "call-semantic",
                "name": "submit_semantic_window_result",
                "arguments": arguments,
            },
            ("submit_semantic_window_result",),
            context=self.context,
        )

    def test_minimal_fact_result_decodes_with_contextual_line_ids(self) -> None:
        decoded = self.decode(asdict(semantic_arguments()))

        self.assertEqual(decoded.arguments, semantic_arguments())

    def test_unknown_line_and_technical_fields_are_rejected(self) -> None:
        unknown = asdict(semantic_arguments())
        unknown["behaviors"][0]["facts"][0]["line_ids"] = ["274:39"]
        with self.assertRaisesRegex(ToolContractError, "274:39"):
            self.decode(unknown)

        technical = asdict(semantic_arguments())
        technical["behaviors"][0]["parts"] = []
        technical["behaviors"][0]["target_paths"] = []
        with self.assertRaises(ToolContractError):
            self.decode(technical)

    def test_cross_page_evidence_uses_opaque_line_ids_only(self) -> None:
        decoded = self.decode(asdict(semantic_arguments()))

        self.assertEqual(
            [
                item
                for fact in decoded.arguments.behaviors[0]["facts"]
                for item in fact["line_ids"]
            ],
            ["L0001", "L0002"],
        )

    def test_empty_behaviors_is_negative_result(self) -> None:
        decoded = self.decode({"behaviors": []})

        self.assertEqual(decoded.arguments.behaviors, [])


if __name__ == "__main__":
    unittest.main()
