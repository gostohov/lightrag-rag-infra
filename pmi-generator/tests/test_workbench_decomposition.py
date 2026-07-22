from __future__ import annotations

import unittest

from prompt_toolkit.formatted_text.utils import fragment_list_to_text
from prompt_toolkit.input.defaults import create_pipe_input
from prompt_toolkit.output import DummyOutput

from pmi_generator.workbench.application.decomposition import (
    DecompositionArguments,
    DecompositionError,
    DecompositionFlow,
    DecompositionService,
    SkeletonDecisionController,
    decomposition_tool,
)
from pmi_generator.workbench.application.llm import LlmToolRuntime, RawCompletion, TypedToolRegistry
from pmi_generator.workbench.application.prompting import default_policy
from pmi_generator.workbench.application.source import SavedSelection
from pmi_generator.workbench.domain.source import (
    SourceDocument,
    SourcePage,
    SourcePosition,
)
from pmi_generator.workbench.infrastructure.storage import InMemoryDatabase, InMemoryUnitOfWork
from pmi_generator.workbench.infrastructure.llm import ScriptedLlmTransport
from pmi_generator.workbench.presentation.decomposition import (
    SkeletonDetailScreen,
    SkeletonListScreen,
    render_decomposition_outcome,
    render_skeletons,
)


def source_document() -> SourceDocument:
    return SourceDocument(
        pages=(
            SourcePage(
                page_index=283,
                logical_page=283,
                lines=(
                    "Требование 4.16.6. Проверить первый байт данных команды",
                    "Если первый байт данных не равен 81, карта прекращает PUT DATA",
                    "Карта устанавливает Script Failed и возвращает 6987",
                ),
            ),
            SourcePage(
                page_index=284,
                logical_page=284,
                lines=("Текст соседней страницы", "Вторая строка", "Третья строка"),
            ),
        ),
        sections=(),
    )


def saved_selection() -> SavedSelection:
    document = source_document()
    return SavedSelection(
        selection_id="SELECTION_0001",
        section_id="section-0270",
        selection=document.select(SourcePosition(283, 1), SourcePosition(283, 3)),
    )


def skeleton(*, title: str = "Проверка первого байта", page: int = 283) -> dict[str, object]:
    return {
        "title": title,
        "condition": "первый байт данных не равен 81",
        "changed_factor": "первый байт данных",
        "input_value": None,
        "action": "отправить PUT DATA",
        "condition_ranges": [{"page": page, "line_start": 1, "line_end": 2}],
        "changed_factor_ranges": [{"page": page, "line_start": 1, "line_end": 2}],
        "input_value_ranges": [],
        "action_ranges": [{"page": page, "line_start": 2, "line_end": 2}],
        "consequences": [
            {
                "text": "карта возвращает 6987",
                "evidence_ranges": [{"page": page, "line_start": 3, "line_end": 3}],
            }
        ],
        "gaps": [
            {
                "kind": "input_value",
                "question": "Какое конкретное значение первого байта использовать?",
                "target_paths": ["test_design.input_value"],
            }
        ],
    }


def arguments(*items: dict[str, object]) -> DecompositionArguments:
    return DecompositionArguments(
        outcome="skeletons_created",
        explanation="",
        skeletons=list(items),
        line_assessments=[
            {
                "page": 283,
                "line": line,
                "role": "evidence",
                "reason": "Строка использована в каркасе",
            }
            for line in range(1, 4)
        ],
    )


class DecompositionServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.database = InMemoryDatabase()
        self.id_counts: dict[str, int] = {}
        self.service = DecompositionService(
            document=source_document(),
            uow_factory=lambda: InMemoryUnitOfWork(self.database),
            next_id=self.next_id,
        )

    def next_id(self, prefix: str) -> str:
        self.id_counts[prefix] = self.id_counts.get(prefix, 0) + 1
        return f"{prefix}_{self.id_counts[prefix]:04d}"

    def test_successful_result_saves_all_skeletons_in_one_transaction(self) -> None:
        result = self.service.apply(
            saved_selection(),
            arguments(skeleton(), skeleton(title="Проверка второго варианта")),
        )

        self.assertEqual(result.skeleton_ids, ("SKELETON_0001", "SKELETON_0002"))
        self.assertEqual(
            len([key for key in self.database.records if key[0] == "card_skeleton"]),
            2,
        )

    def test_invalid_single_skeleton_rolls_back_entire_result(self) -> None:
        invalid = skeleton(title="Некорректный")
        invalid["consequences"] = []

        with self.assertRaises(DecompositionError):
            self.service.apply(saved_selection(), arguments(skeleton(), invalid))

        self.assertFalse(self.database.records)

    def test_evidence_range_outside_selection_is_rejected(self) -> None:
        with self.assertRaisesRegex(DecompositionError, "выбранного диапазона"):
            self.service.apply(saved_selection(), arguments(skeleton(page=284)))

    def test_every_selection_line_requires_exactly_one_assessment(self) -> None:
        proposed = arguments(skeleton())
        proposed.line_assessments.pop()

        with self.assertRaisesRegex(DecompositionError, "не классифицированы"):
            self.service.apply(saved_selection(), proposed)

    def test_context_line_cannot_be_used_as_skeleton_evidence(self) -> None:
        proposed = arguments(skeleton())
        proposed.line_assessments[1]["role"] = "context"
        proposed.line_assessments[1]["reason"] = "Ошибочно признана контекстом"

        with self.assertRaisesRegex(DecompositionError, "Роль evidence"):
            self.service.apply(saved_selection(), proposed)

    def test_action_and_changed_factor_require_source_ranges(self) -> None:
        proposed = skeleton()
        proposed["action_ranges"] = []

        with self.assertRaisesRegex(DecompositionError, "action"):
            self.service.apply(saved_selection(), arguments(proposed))

    def test_quote_is_copied_from_document_and_is_not_accepted_from_model(self) -> None:
        proposed = skeleton()
        proposed["quote"] = "модель подменила цитату"
        spec = decomposition_tool()
        self.assertFalse(spec.json_schema["properties"]["skeletons"]["items"]["additionalProperties"])

        proposed.pop("quote")
        result = self.service.apply(saved_selection(), arguments(proposed))
        record = self.database.records[("card_skeleton", result.skeleton_ids[0])]

        self.assertEqual(
            record.payload["condition_evidence"][0]["quote"],
            "Требование 4.16.6. Проверить первый байт данных команды\n"
            "Если первый байт данных не равен 81, карта прекращает PUT DATA",
        )

    def test_retry_of_same_valid_result_is_idempotent(self) -> None:
        first = self.service.apply(saved_selection(), arguments(skeleton()))
        second = self.service.apply(saved_selection(), arguments(skeleton()))

        self.assertEqual(second, first)
        self.assertEqual(
            len([key for key in self.database.records if key[0] == "card_skeleton"]),
            1,
        )

    def test_excluded_skeleton_is_audited_without_creating_card(self) -> None:
        result = self.service.apply(saved_selection(), arguments(skeleton()))

        self.service.exclude(result.skeleton_ids[0], author="Аналитик", reason="Не нужен")

        record = self.database.records[("card_skeleton", result.skeleton_ids[0])]
        self.assertEqual(record.payload["decision"], "excluded")
        self.assertFalse(self.database.cards)

    def test_take_skeleton_creates_card_and_returns_future_session_target(self) -> None:
        result = self.service.apply(saved_selection(), arguments(skeleton()))

        card_id = self.service.take(result.skeleton_ids[0], author="Аналитик")

        self.assertEqual(card_id, "CARD_0001")
        self.assertIn(card_id, self.database.cards)


class DecompositionPresentationTests(unittest.TestCase):
    def test_all_three_outcomes_have_explicit_ui(self) -> None:
        for outcome, expected in (
            ("skeletons_created", "Каркасы построены"),
            ("no_testable_behavior", "Проверяемое поведение не найдено"),
            ("insufficient_selection", "Недостаточный диапазон"),
        ):
            with self.subTest(outcome=outcome):
                rendered = render_decomposition_outcome(outcome, "Пояснение")
                self.assertIn(expected, rendered)

    def test_decision_moves_cursor_without_opening_next_skeleton(self) -> None:
        controller = SkeletonDecisionController(
            ["SKELETON_0001", "SKELETON_0002", "SKELETON_0003"]
        )
        controller.open_current()

        controller.record_decision("selected")

        self.assertEqual(controller.cursor, 1)
        self.assertEqual(controller.screen, "list")
        self.assertIn("Решения: 1 из 3", render_skeletons(controller))

    def test_full_screen_list_shows_all_decisions_and_skeleton_summary(self) -> None:
        database = InMemoryDatabase()
        ids = iter(("S1", "S2"))
        service = DecompositionService(
            document=source_document(),
            uow_factory=lambda: InMemoryUnitOfWork(database),
            next_id=lambda prefix: next(ids),
        )
        result = service.apply(
            saved_selection(),
            arguments(skeleton(), skeleton(title="Проверка второго варианта")),
        )
        service.exclude(result.skeleton_ids[1], author="Аналитик", reason="Не нужен")
        records = tuple(
            database.records[("card_skeleton", skeleton_id)]
            for skeleton_id in result.skeleton_ids
        )
        screen = SkeletonListScreen(records, saved_selection())

        text = fragment_list_to_text(screen.render(width=48, height=16))

        self.assertIn("Решения: 1 из 2", text)
        self.assertIn("[без решения]", text)
        self.assertIn("[исключён]", text)
        self.assertIn("Обязательные последствия: 1", text)
        self.assertEqual(len(text.splitlines()), 16)

        screen.update(
            tuple(
                database.records[("card_skeleton", skeleton_id)]
                for skeleton_id in result.skeleton_ids
            )
        )
        screen.advance_after_decision("S2")
        self.assertEqual(screen.records[screen.cursor].record_id, "S1")

    def test_skeleton_detail_contains_complete_prompt_1_payload(self) -> None:
        database = InMemoryDatabase()
        service = DecompositionService(
            document=source_document(),
            uow_factory=lambda: InMemoryUnitOfWork(database),
            next_id=lambda prefix: f"{prefix}_0001",
        )
        result = service.apply(saved_selection(), arguments(skeleton()))
        record = database.records[("card_skeleton", result.skeleton_ids[0])]

        screen = SkeletonDetailScreen(record)
        text = fragment_list_to_text(screen.render(width=52, height=24))

        self.assertIn("Проверяемое условие:", text)
        self.assertIn("Изменяемый фактор:", text)
        self.assertIn("Конкретное значение:", text)
        self.assertIn("Воздействие:", text)
        self.assertIn("Обязательные последствия:", text)
        self.assertIn("Источник:", text)

        screen.follow_cursor = True
        actions = fragment_list_to_text(screen.render(width=52, height=24))
        self.assertIn("Взять в работу", actions)

    def test_exclusion_reason_is_collected_inline(self) -> None:
        database = InMemoryDatabase()
        service = DecompositionService(
            document=source_document(),
            uow_factory=lambda: InMemoryUnitOfWork(database),
            next_id=lambda prefix: f"{prefix}_0001",
        )
        result = service.apply(saved_selection(), arguments(skeleton()))
        record = database.records[("card_skeleton", result.skeleton_ids[0])]

        with create_pipe_input() as pipe_input:
            pipe_input.send_text("\x1b[B\rnot applicable\r")
            decision = SkeletonDetailScreen(
                record,
                input=pipe_input,
                output=DummyOutput(),
            ).run()

        self.assertEqual(decision.action, "exclude")
        self.assertEqual(decision.reason, "not applicable")

    def test_selected_skeleton_can_reopen_its_card_session(self) -> None:
        database = InMemoryDatabase()
        service = DecompositionService(
            document=source_document(),
            uow_factory=lambda: InMemoryUnitOfWork(database),
            next_id=lambda prefix: f"{prefix}_0001",
        )
        result = service.apply(saved_selection(), arguments(skeleton()))
        service.take(result.skeleton_ids[0], author="Аналитик")
        record = database.records[("card_skeleton", result.skeleton_ids[0])]

        screen = SkeletonDetailScreen(record)
        screen.follow_cursor = True
        text = fragment_list_to_text(screen.render(width=52, height=24))

        self.assertEqual(
            screen.actions,
            (
                ("open_session", "Открыть сессию"),
                ("back", "Назад к каркасам"),
            ),
        )
        self.assertIn("Открыть сессию", text)


class DecompositionFlowTests(unittest.IsolatedAsyncioTestCase):
    async def test_flow_uses_one_atomic_llm_call(self) -> None:
        database = InMemoryDatabase()
        tools = TypedToolRegistry()
        tools.register(decomposition_tool())
        proposed = arguments(skeleton())
        transport = ScriptedLlmTransport(
            [
                RawCompletion(
                    finish_reason="tool_calls",
                    tool_calls=(
                        {
                            "id": "call-1",
                            "name": "submit_decomposition",
                            "arguments": {
                                "outcome": proposed.outcome,
                                "explanation": proposed.explanation,
                                "skeletons": proposed.skeletons,
                                "line_assessments": proposed.line_assessments,
                            },
                        },
                    ),
                    usage={},
                    model="fake",
                )
            ]
        )
        runtime = LlmToolRuntime(
            transport=transport,
            tools=tools,
            uow_factory=lambda: InMemoryUnitOfWork(database),
        )
        service = DecompositionService(
            document=source_document(),
            uow_factory=lambda: InMemoryUnitOfWork(database),
            next_id=lambda prefix: f"{prefix}_0001",
        )

        result = await DecompositionFlow(
            policy=default_policy(), runtime=runtime, service=service
        ).run(
            attempt_id="ATTEMPT_0001",
            session_id="SESSION_0001",
            selection=saved_selection(),
        )

        self.assertEqual(result.skeleton_ids, ("SKELETON_0001",))
        self.assertEqual(len(transport.calls), 1)
        self.assertEqual(
            transport.calls[0]["call"].context["selection"]["lines"],
            [
                {
                    "page": 283,
                    "line": 1,
                    "text": "Требование 4.16.6. Проверить первый байт данных команды",
                },
                {
                    "page": 283,
                    "line": 2,
                    "text": (
                        "Если первый байт данных не равен 81, "
                        "карта прекращает PUT DATA"
                    ),
                },
                {
                    "page": 283,
                    "line": 3,
                    "text": "Карта устанавливает Script Failed и возвращает 6987",
                },
            ],
        )

    async def test_flow_passes_every_line_of_a_cross_page_selection(self) -> None:
        database = InMemoryDatabase()
        tools = TypedToolRegistry()
        tools.register(decomposition_tool())
        document = source_document()
        selection = SavedSelection(
            selection_id="SELECTION_ALL",
            section_id="section-0270",
            selection=document.select(SourcePosition(283, 1), SourcePosition(284, 3)),
        )
        line_assessments = [
            {
                "page": position.page_index,
                "line": position.line_number,
                "role": "context",
                "reason": "Проверяемое поведение не найдено",
            }
            for position in selection.selection.positions
        ]
        transport = ScriptedLlmTransport(
            [
                RawCompletion(
                    finish_reason="tool_calls",
                    tool_calls=(
                        {
                            "id": "call-all",
                            "name": "submit_decomposition",
                            "arguments": {
                                "outcome": "no_testable_behavior",
                                "explanation": "Нет проверяемого поведения",
                                "skeletons": [],
                                "line_assessments": line_assessments,
                            },
                        },
                    ),
                    usage={},
                    model="fake",
                )
            ]
        )
        runtime = LlmToolRuntime(
            transport=transport,
            tools=tools,
            uow_factory=lambda: InMemoryUnitOfWork(database),
        )

        result = await DecompositionFlow(
            policy=default_policy(),
            runtime=runtime,
            service=DecompositionService(
                document=document,
                uow_factory=lambda: InMemoryUnitOfWork(database),
                next_id=lambda prefix: f"{prefix}_0001",
            ),
        ).run(
            attempt_id="ATTEMPT_ALL",
            session_id="SESSION_ALL",
            selection=selection,
        )

        sent = transport.calls[0]["call"].context["selection"]
        self.assertEqual(result.outcome, "no_testable_behavior")
        self.assertEqual(sent["start"], {"page": 283, "line": 1})
        self.assertEqual(sent["end"], {"page": 284, "line": 3})
        self.assertEqual(len(sent["lines"]), 6)
        self.assertEqual(
            [(item["page"], item["line"]) for item in sent["lines"]],
            [(283, 1), (283, 2), (283, 3), (284, 1), (284, 2), (284, 3)],
        )


if __name__ == "__main__":
    unittest.main()
