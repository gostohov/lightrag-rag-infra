from __future__ import annotations

import copy
import unittest
from dataclasses import asdict, replace

from pmi_generator.workbench.application.decomposition import (
    SEMANTIC_CANONICAL_MAPPING_VERSION,
    SEMANTIC_SYNTHESIS_SCHEMA_VERSION,
    SEMANTIC_WINDOW_SCHEMA_VERSION,
    SemanticSynthesisArguments,
    SemanticSynthesisService,
    SemanticWindowArguments,
    SemanticWindowError,
    SemanticWindowFlow,
    SemanticWindowService,
    WindowPlanner,
    WindowedAttemptState,
    default_windowing_policy,
    semantic_window_tool,
)
from pmi_generator.workbench.application.llm import (
    LlmToolRuntime,
    RawCompletion,
    TypedToolRegistry,
)
from pmi_generator.workbench.application.prompting import (
    PromptId,
    default_policy,
)
from pmi_generator.workbench.application.source import SavedSelection
from pmi_generator.workbench.domain.source import (
    SourceDocument,
    SourcePage,
    SourceSection,
)
from pmi_generator.workbench.infrastructure.llm import ScriptedLlmTransport
from pmi_generator.workbench.infrastructure.storage import (
    InMemoryDatabase,
    InMemoryUnitOfWork,
)


def document() -> SourceDocument:
    return SourceDocument(
        pages=(
            SourcePage(
                1,
                "1",
                tuple(f"Строка источника {line}" for line in range(1, 301)),
            ),
        ),
        sections=(SourceSection("root", "1", "Root", ("1",), (1,)),),
    )


def selection(source: SourceDocument) -> SavedSelection:
    selected = source.select(source.positions[0], source.positions[9])
    return SavedSelection(
        "SELECTION_1",
        "root",
        selected,
        source.metadata.document_version,
        "root",
    )


def arguments() -> SemanticWindowArguments:
    return SemanticWindowArguments(
        behaviors=[
            {
                "title": "Проверка поведения",
                "summary": "Проверяется поведение выбранной строки",
                "facts": [
                    {
                        "text": "Известен исходный факт",
                        "line_ids": ["L0001"],
                    },
                    {
                        "text": "Известно последствие",
                        "line_ids": ["L0002"],
                    },
                ],
            }
        ]
    )


def setup_state():
    source = document()
    saved = selection(source)
    policy = default_policy()
    windowing = default_windowing_policy(policy)
    plan = WindowPlanner(source, windowing).build(saved)
    window = plan.windows[0]
    parent = (
        WindowedAttemptState.planned(
            parent_attempt_id="ATTEMPT_PARENT",
            selection_id=saved.selection_id,
            document_version=saved.document_version,
            expected_workflow_revision="workflow-revision-1",
            policy_version=plan.policy_version,
            prompt_version=policy.prompts[
                PromptId.DECOMPOSITION_WINDOW_SEMANTIC
            ].version,
            schema_version=SEMANTIC_WINDOW_SCHEMA_VERSION,
            window_plan_hash=plan.plan_hash,
            window_ids=tuple(item.window_id for item in plan.windows),
        )
        .start()
        .start_child(window.window_id, "ATTEMPT_CHILD")
    )
    return source, saved, policy, windowing, plan, window, parent


class SemanticWindowFlowTests(unittest.IsolatedAsyncioTestCase):
    async def test_repair_persists_raw_and_validated_fact_views(self) -> None:
        source, saved, policy, _windowing, plan, window, parent = setup_state()
        rejected = asdict(arguments())
        rejected["behaviors"][0]["facts"].append(
            {"text": "Неизвестная строка", "line_ids": ["L9999"]}
        )
        transport = ScriptedLlmTransport(
            [
                RawCompletion(
                    "tool_calls",
                    (
                        {
                            "id": "call-rejected",
                            "name": "submit_semantic_window_result",
                            "arguments": rejected,
                        },
                    ),
                    {"prompt_tokens": 100, "completion_tokens": 50},
                    "scripted",
                ),
                RawCompletion(
                    "tool_calls",
                    (
                        {
                            "id": "call-accepted",
                            "name": "submit_semantic_window_result",
                            "arguments": asdict(arguments()),
                        },
                    ),
                    {"prompt_tokens": 100, "completion_tokens": 50},
                    "scripted",
                ),
            ]
        )
        database = InMemoryDatabase()
        uow_factory = lambda: InMemoryUnitOfWork(database)
        registry = TypedToolRegistry()
        registry.register(semantic_window_tool())
        runtime = LlmToolRuntime(
            transport=transport,
            tools=registry,
            uow_factory=uow_factory,
        )

        result = await SemanticWindowFlow(
            policy=policy,
            runtime=runtime,
            service=SemanticWindowService(
                document=source,
                uow_factory=uow_factory,
            ),
        ).run(
            parent=parent,
            plan=plan,
            window_id=window.window_id,
            child_attempt_id="ATTEMPT_CHILD",
            session_id=saved.selection_id,
        )

        self.assertEqual(len(result.fragments), 1)
        self.assertEqual(len(transport.calls), 2)
        repair = transport.calls[1]["call"].system_prompt
        self.assertIn("Предыдущий semantic tool call", repair)
        self.assertIn("L9999", repair)
        stored = database.records[
            (
                "decomposition_window_semantic_facts",
                f"ATTEMPT_PARENT:{window.window_id}",
            )
        ]
        self.assertEqual(stored.payload["raw_arguments"], asdict(arguments()))
        self.assertEqual(
            stored.payload["validated"]["fragments"][0]["fragment_id"],
            result.fragments[0].fragment_id,
        )
        self.assertNotIn(("decomposition", saved.selection_id), database.records)


class SemanticPersistenceTests(unittest.TestCase):
    def setUp(self) -> None:
        (
            self.source,
            self.saved,
            self.policy,
            self.windowing,
            self.plan,
            self.window,
            self.active_parent,
        ) = setup_state()
        self.database = InMemoryDatabase()
        self.uow_factory = lambda: InMemoryUnitOfWork(self.database)
        self.fact_service = SemanticWindowService(
            document=self.source,
            uow_factory=self.uow_factory,
        )
        self.facts = self.fact_service.accept(
            parent=self.active_parent,
            plan=self.plan,
            window_id=self.window.window_id,
            child_attempt_id="ATTEMPT_CHILD",
            arguments=arguments(),
            raw_arguments=asdict(arguments()),
        )
        self.completed_parent = self.active_parent.complete_child(
            self.window.window_id,
            "ATTEMPT_CHILD",
        ).begin_reconciliation()

    def test_fact_result_survives_restart_and_revalidation(self) -> None:
        loaded = self.fact_service.load(
            self.window.window_id,
            parent=self.completed_parent,
            plan=self.plan,
        )

        self.assertEqual(loaded, self.facts)

    def test_context_only_behavior_is_kept_raw_but_not_validated(
        self,
    ) -> None:
        masked_window = replace(
            self.window,
            lines=tuple(
                replace(line, primary=index <= 2)
                for index, line in enumerate(self.window.lines, start=1)
            ),
        )
        provisional = replace(
            self.plan,
            windows=(masked_window,),
            plan_hash="",
        )
        masked_plan = replace(
            provisional,
            plan_hash=provisional.recompute_hash(),
        )
        parent = (
            WindowedAttemptState.planned(
                parent_attempt_id="ATTEMPT_CONTEXT_OWNER",
                selection_id=self.saved.selection_id,
                document_version=self.saved.document_version,
                expected_workflow_revision="workflow-revision-context",
                policy_version=masked_plan.policy_version,
                prompt_version=self.policy.prompts[
                    PromptId.DECOMPOSITION_WINDOW_SEMANTIC
                ].version,
                schema_version=SEMANTIC_WINDOW_SCHEMA_VERSION,
                window_plan_hash=masked_plan.plan_hash,
                window_ids=(masked_window.window_id,),
            )
            .start()
            .start_child(masked_window.window_id, "ATTEMPT_CONTEXT_CHILD")
        )
        raw = SemanticWindowArguments(
            behaviors=[
                *arguments().behaviors,
                {
                    "title": "Соседнее поведение",
                    "summary": "Поведение относится к следующему owner.",
                    "facts": [
                        {
                            "text": "Контекстный факт",
                            "line_ids": ["L0003"],
                        }
                    ],
                },
            ]
        )

        accepted = self.fact_service.accept(
            parent=parent,
            plan=masked_plan,
            window_id=masked_window.window_id,
            child_attempt_id="ATTEMPT_CONTEXT_CHILD",
            arguments=raw,
            raw_arguments=asdict(raw),
        )
        record = self.database.records[
            (
                "decomposition_window_semantic_facts",
                f"ATTEMPT_CONTEXT_OWNER:{masked_window.window_id}",
            )
        ]

        self.assertEqual(len(accepted.fragments), 1)
        self.assertEqual(len(record.payload["raw_arguments"]["behaviors"]), 2)
        self.assertEqual(record.payload["context_only_behaviors"], 1)
        self.assertEqual(len(record.payload["validated"]["fragments"]), 1)

    def test_recomputed_tamper_is_rejected_against_validated_facts(self) -> None:
        record = self.database.records[
            (
                "decomposition_window_semantic_facts",
                f"ATTEMPT_PARENT:{self.window.window_id}",
            )
        ]
        record.payload["raw_arguments"]["behaviors"][0]["title"] = "Подмена"
        record.payload["fingerprint"] = self.fact_service._fingerprint(
            {
                key: value
                for key, value in record.payload.items()
                if key != "fingerprint"
            }
        )

        with self.assertRaisesRegex(
            SemanticWindowError,
            "не соответствуют raw arguments",
        ):
            self.fact_service.load(
                self.window.window_id,
                parent=self.completed_parent,
                plan=self.plan,
            )

    def test_synthesis_round_trip_binds_exact_fact_set(self) -> None:
        fact_ids = [
            fact.fact_id for fact in self.facts.fragments[0].facts
        ]
        synthesis = SemanticSynthesisArguments(
            candidates=[
                {
                    "title": "Проверка поведения",
                    "condition": {
                        "text": "Условие",
                        "fact_ids": [fact_ids[0]],
                    },
                    "changed_factor": {
                        "text": "Фактор",
                        "fact_ids": [fact_ids[0]],
                    },
                    "consequences": [
                        {
                            "text": "Последствие",
                            "fact_ids": [fact_ids[1]],
                        },
                    ],
                }
            ]
        )
        service = SemanticSynthesisService(
            document=self.source,
            uow_factory=self.uow_factory,
        )
        expected = service.accept(
            parent=self.completed_parent,
            plan=self.plan,
            target_window_id=self.window.window_id,
            attempt_id="ATTEMPT_SYNTHESIS",
            fact_results=(self.facts,),
            arguments=synthesis,
            raw_arguments=asdict(synthesis),
        )

        loaded = service.load(
            parent=self.completed_parent,
            plan=self.plan,
            target_window_id=self.window.window_id,
            fact_results=(self.facts,),
        )

        self.assertEqual(loaded, expected)
        record = self.database.records[
            (
                "decomposition_window_result",
                f"ATTEMPT_PARENT:{self.window.window_id}",
            )
        ]
        self.assertEqual(
            record.payload["contract_version"],
            SEMANTIC_SYNTHESIS_SCHEMA_VERSION,
        )
        self.assertEqual(
            record.payload["mapping_version"],
            SEMANTIC_CANONICAL_MAPPING_VERSION,
        )

    def test_cancelled_parent_cannot_reuse_saved_facts(self) -> None:
        cancelled = self.active_parent.cancel("Отменено аналитиком")

        with self.assertRaisesRegex(SemanticWindowError, "не выполняется"):
            self.fact_service.load(
                self.window.window_id,
                parent=cancelled,
                plan=self.plan,
            )

    def test_windowing_fingerprint_contains_both_semantic_stages(self) -> None:
        self.assertEqual(
            self.windowing.semantic_schema_version,
            SEMANTIC_WINDOW_SCHEMA_VERSION,
        )
        self.assertEqual(
            self.windowing.synthesis_schema_version,
            SEMANTIC_SYNTHESIS_SCHEMA_VERSION,
        )
        self.assertEqual(
            self.windowing.semantic_mapping_version,
            SEMANTIC_CANONICAL_MAPPING_VERSION,
        )


if __name__ == "__main__":
    unittest.main()
