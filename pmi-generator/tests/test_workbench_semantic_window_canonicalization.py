from __future__ import annotations

import unittest
from dataclasses import replace

from pmi_generator.workbench.application.decomposition import (
    SEMANTIC_WINDOW_SCHEMA_VERSION,
    SemanticBehaviorFragment,
    SemanticFact,
    SemanticFactScope,
    SemanticSynthesisArguments,
    SemanticSynthesisCanonicalizer,
    SemanticWindowArguments,
    SemanticWindowCanonicalizer,
    SemanticWindowError,
    SemanticWindowResult,
    WindowPlan,
    WindowedAttemptState,
    semantic_fact_scopes,
    semantic_synthesis_context,
)
from pmi_generator.workbench.application.decomposition.windowing.plan import (
    DecompositionWindow,
    WindowSourceLine,
)
from pmi_generator.workbench.domain.source import (
    SourceDocument,
    SourcePage,
    SourcePosition,
    SourceSection,
)


def document() -> SourceDocument:
    return SourceDocument(
        pages=(
            SourcePage(
                297,
                "297",
                (
                    "Проверить, не заблокирован ли апплет",
                    "Если бит Application Blocked равен 1b,",
                    "карта возвращает SW1 SW2 = 6283.",
                ),
            ),
            SourcePage(
                298,
                "298",
                (
                    "CLA 00, INS A4",
                    "Data содержит AID приложения",
                ),
            ),
        ),
        sections=(
            SourceSection("root", "4.19", "SELECT", ("4", "4.19"), (297, 298)),
        ),
    )


def window() -> DecompositionWindow:
    return DecompositionWindow(
        window_id="SELECTION_1:WINDOW:0001",
        index=0,
        lines=(
            WindowSourceLine(
                SourcePosition(297, 1),
                "Проверить, не заблокирован ли апплет",
                True,
            ),
            WindowSourceLine(
                SourcePosition(297, 2),
                "Если бит Application Blocked равен 1b,",
                True,
            ),
            WindowSourceLine(
                SourcePosition(297, 3),
                "карта возвращает SW1 SW2 = 6283.",
                True,
            ),
            WindowSourceLine(
                SourcePosition(298, 1),
                "CLA 00, INS A4",
                True,
            ),
            WindowSourceLine(
                SourcePosition(298, 2),
                "Data содержит AID приложения",
                True,
            ),
        ),
        global_start=SourcePosition(297, 1),
        global_end=SourcePosition(298, 2),
        outline_node_id="root",
        outline_label="SELECT",
        outline_path=("4", "4.19"),
        input_max_lines=256,
        input_max_estimated_tokens=12_000,
        estimated_tokens=100,
        output_max_tokens=4096,
        output_budget_tokens=3072,
        estimated_output_tokens=500,
        policy_version="policy",
    )


def plan() -> WindowPlan:
    provisional = WindowPlan(
        schema_version="window-plan-2",
        selection_id="SELECTION_1",
        document_version=document().metadata.document_version,
        selection_start=SourcePosition(297, 1),
        selection_end=SourcePosition(298, 2),
        policy_version="policy",
        windows=(window(),),
        plan_hash="",
    )
    return replace(provisional, plan_hash=provisional.recompute_hash())


def active_parent() -> WindowedAttemptState:
    current_plan = plan()
    return (
        WindowedAttemptState.planned(
            parent_attempt_id="ATTEMPT_PARENT",
            selection_id=current_plan.selection_id,
            document_version=current_plan.document_version,
            expected_workflow_revision="workflow-revision-1",
            policy_version=current_plan.policy_version,
            prompt_version="semantic-window-facts",
            schema_version=SEMANTIC_WINDOW_SCHEMA_VERSION,
            window_plan_hash=current_plan.plan_hash,
            window_ids=(window().window_id,),
        )
        .start()
        .start_child(window().window_id, "ATTEMPT_CHILD")
    )


def fact_arguments() -> SemanticWindowArguments:
    return SemanticWindowArguments(
        behaviors=[
            {
                "title": "Выбор заблокированного приложения",
                "summary": "SELECT заблокированного приложения возвращает 6283",
                "facts": [
                    {
                        "text": "Бит Application Blocked равен 1b",
                        "line_ids": ["L0002"],
                    },
                    {
                        "text": "Карта возвращает SW1 SW2 = 6283",
                        "line_ids": ["L0003"],
                    },
                    {
                        "text": "Команда передаёт AID приложения",
                        "line_ids": ["L0004", "L0005"],
                    },
                ],
            }
        ]
    )


def synthesis_candidate(
    title: str,
    *parts: tuple[str, str, list[str]],
) -> dict[str, object]:
    candidate: dict[str, object] = {
        "title": title,
        "consequences": [],
    }
    consequences: list[dict[str, object]] = []
    for role, text, fact_ids in parts:
        slot = {"text": text, "fact_ids": fact_ids}
        if role == "consequence":
            consequences.append(slot)
        else:
            candidate[role] = slot
    candidate["consequences"] = consequences
    return candidate


class SemanticFactCanonicalizationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.canonicalizer = SemanticWindowCanonicalizer(document())

    def extract(self, arguments: SemanticWindowArguments):
        return self.canonicalizer.canonicalize(
            parent=active_parent(),
            plan=plan(),
            window_id=window().window_id,
            child_attempt_id="ATTEMPT_CHILD",
            arguments=arguments,
        )

    def test_vdi_content_is_accepted_without_card_roles_or_boundaries(
        self,
    ) -> None:
        result = self.extract(fact_arguments())

        self.assertEqual(len(result.fragments), 1)
        fragment = result.fragments[0]
        self.assertEqual(
            fragment.title,
            "Выбор заблокированного приложения",
        )
        self.assertEqual(len(fragment.facts), 3)
        self.assertTrue(
            all(":FACT:" in fact.fact_id for fact in fragment.facts)
        )

    def test_unknown_line_and_overlap_only_fragment_are_rejected(self) -> None:
        invalid = fact_arguments()
        invalid.behaviors[0]["facts"][0]["line_ids"] = ["L9999"]

        with self.assertRaisesRegex(SemanticWindowError, "L9999"):
            self.extract(invalid)

    def test_context_only_behaviors_are_routed_out_of_subwindow(self) -> None:
        logical_window = window()
        subwindow = replace(
            logical_window,
            window_id="SELECTION_1:WINDOW:0001:SUB:0001",
            lines=tuple(
                replace(line, primary=index <= 2)
                for index, line in enumerate(logical_window.lines, start=1)
            ),
        )
        behaviors = [
            {
                "title": "Primary behavior",
                "summary": "Поведение принадлежит этой части.",
                "facts": [
                    {"text": "Primary fact", "line_ids": ["L0001"]}
                ],
            },
            *[
                {
                    "title": f"Context behavior {index}",
                    "summary": "Поведение принадлежит соседней части.",
                    "facts": [
                        {
                            "text": f"Context fact {index}",
                            "line_ids": [line_id],
                        }
                    ],
                }
                for index, line_id in enumerate(
                    ("L0003", "L0004", "L0005", "L0003"),
                    start=1,
                )
            ],
            {
                "title": "Boundary behavior",
                "summary": "Поведение пересекает границу частей.",
                "facts": [
                    {
                        "text": "Boundary fact",
                        "line_ids": ["L0002", "L0003"],
                    }
                ],
            },
        ]
        arguments = SemanticWindowArguments(behaviors=behaviors)

        result = self.canonicalizer.canonicalize_subwindow(
            parent=active_parent(),
            plan=plan(),
            logical_window_id=logical_window.window_id,
            logical_child_attempt_id="ATTEMPT_CHILD",
            subwindow=subwindow,
            generation_attempt_id="ATTEMPT_GENERATION",
            arguments=arguments,
        )
        owned = self.canonicalizer.owned_arguments(subwindow, arguments)

        self.assertEqual(
            [fragment.title for fragment in result.fragments],
            ["Primary behavior", "Boundary behavior"],
        )
        self.assertEqual(
            [behavior["title"] for behavior in owned.behaviors],
            ["Primary behavior", "Boundary behavior"],
        )

    def test_unknown_line_in_context_only_behavior_is_rejected(self) -> None:
        logical_window = window()
        subwindow = replace(
            logical_window,
            window_id="SELECTION_1:WINDOW:0001:SUB:0001",
            lines=tuple(
                replace(line, primary=index <= 2)
                for index, line in enumerate(logical_window.lines, start=1)
            ),
        )
        invalid = SemanticWindowArguments(
            behaviors=[
                {
                    "title": "Invalid context behavior",
                    "summary": "Неизвестная строка не становится допустимой.",
                    "facts": [
                        {
                            "text": "Invalid fact",
                            "line_ids": ["L9999"],
                        }
                    ],
                }
            ]
        )

        with self.assertRaisesRegex(SemanticWindowError, "L9999"):
            self.canonicalizer.canonicalize_subwindow(
                parent=active_parent(),
                plan=plan(),
                logical_window_id=logical_window.window_id,
                logical_child_attempt_id="ATTEMPT_CHILD",
                subwindow=subwindow,
                generation_attempt_id="ATTEMPT_GENERATION",
                arguments=invalid,
            )

    def test_fact_order_does_not_change_canonical_result(self) -> None:
        original = fact_arguments()
        reordered = fact_arguments()
        reordered.behaviors[0]["facts"] = list(
            reversed(reordered.behaviors[0]["facts"])
        )

        self.assertEqual(self.extract(original), self.extract(reordered))

    def test_fact_count_is_bounded_before_synthesis(self) -> None:
        excessive = fact_arguments()
        excessive.behaviors[0]["facts"] = [
            {
                "text": f"Факт {index}",
                "line_ids": ["L0001"],
            }
            for index in range(33)
        ]

        with self.assertRaisesRegex(SemanticWindowError, "лимит 32"):
            self.extract(excessive)


class SemanticFactScopePolicyTests(unittest.TestCase):
    def test_first_middle_and_last_window_use_source_position_ownership(
        self,
    ) -> None:
        lines = window().lines
        windows = (
            replace(
                window(),
                window_id="WINDOW_1",
                index=0,
                lines=(
                    replace(lines[0], primary=True),
                    replace(lines[1], primary=True),
                    replace(lines[2], primary=False),
                ),
            ),
            replace(
                window(),
                window_id="WINDOW_2",
                index=1,
                lines=(
                    replace(lines[1], primary=False),
                    replace(lines[2], primary=True),
                    replace(lines[3], primary=True),
                    replace(lines[4], primary=False),
                ),
            ),
            replace(
                window(),
                window_id="WINDOW_3",
                index=2,
                lines=(
                    replace(lines[3], primary=False),
                    replace(lines[4], primary=True),
                ),
            ),
        )
        scoped_plan = WindowPlan(
            schema_version="window-plan-2",
            selection_id="SELECTION_SCOPE",
            document_version=document().metadata.document_version,
            selection_start=lines[0].position,
            selection_end=lines[-1].position,
            policy_version="policy",
            windows=windows,
            plan_hash="scope-plan",
        )

        def fact(
            fact_id: str,
            *positions: SourcePosition,
        ) -> SemanticFact:
            return SemanticFact(fact_id, fact_id, positions)

        def result(
            window_id: str,
            *facts: SemanticFact,
        ) -> SemanticWindowResult:
            return SemanticWindowResult(
                parent_attempt_id="PARENT",
                child_attempt_id=f"CHILD_{window_id}",
                window_id=window_id,
                plan_hash="scope-plan",
                fragments=(
                    SemanticBehaviorFragment(
                        fragment_id=f"FRAGMENT_{window_id}",
                        window_id=window_id,
                        title=window_id,
                        summary=window_id,
                        facts=facts,
                    ),
                ),
            )

        results = (
            result(
                "WINDOW_1",
                fact("W1_OWN", lines[1].position),
                fact("W1_DUPLICATE", lines[2].position),
            ),
            result(
                "WINDOW_2",
                fact("W2_LEFT_CONTEXT", lines[1].position),
                fact("W2_OWN", lines[2].position),
                fact(
                    "W2_MIXED",
                    lines[1].position,
                    lines[2].position,
                ),
                fact("W2_RIGHT_CONTEXT", lines[4].position),
            ),
            result(
                "WINDOW_3",
                fact("W3_LEFT_CONTEXT", lines[3].position),
                fact("W3_OWN", lines[4].position),
            ),
        )
        expected_primary = {
            "WINDOW_1": {
                "W1_OWN",
                "W2_LEFT_CONTEXT",
                "W2_MIXED",
            },
            "WINDOW_2": {
                "W1_DUPLICATE",
                "W2_OWN",
                "W2_MIXED",
                "W3_LEFT_CONTEXT",
            },
            "WINDOW_3": {
                "W2_RIGHT_CONTEXT",
                "W3_OWN",
            },
        }

        for target_window_id, expected in expected_primary.items():
            with self.subTest(target_window_id=target_window_id):
                scopes = semantic_fact_scopes(
                    plan=scoped_plan,
                    target_window_id=target_window_id,
                    results=results,
                )
                self.assertEqual(
                    {
                        fact_id
                        for fact_id, scope in scopes.items()
                        if scope is SemanticFactScope.PRIMARY
                    },
                    expected,
                )

        self.assertEqual(
            semantic_synthesis_context(
                document=document(),
                plan=scoped_plan,
                target_window_id="WINDOW_2",
                results=results,
            ),
            semantic_synthesis_context(
                document=document(),
                plan=scoped_plan,
                target_window_id="WINDOW_2",
                results=tuple(reversed(results)),
            ),
        )


class SemanticSynthesisCanonicalizationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = document()
        self.current_plan = plan()
        self.facts = SemanticWindowCanonicalizer(self.source).canonicalize(
            parent=active_parent(),
            plan=self.current_plan,
            window_id=window().window_id,
            child_attempt_id="ATTEMPT_CHILD",
            arguments=fact_arguments(),
        )
        self.parent = active_parent().complete_child(
            window().window_id,
            "ATTEMPT_CHILD",
        ).begin_reconciliation()
        self.canonicalizer = SemanticSynthesisCanonicalizer(self.source)
        self.fact_ids = [
            fact.fact_id
            for fact in self.facts.fragments[0].facts
        ]

    def synthesize(self, arguments: SemanticSynthesisArguments):
        return self.canonicalizer.canonicalize(
            parent=self.parent,
            plan=self.current_plan,
            target_window_id=window().window_id,
            attempt_id="ATTEMPT_SYNTHESIS",
            fact_results=(self.facts,),
            arguments=arguments,
        )

    def test_builds_card_content_and_technical_fields_in_application(
        self,
    ) -> None:
        result = self.synthesize(
            SemanticSynthesisArguments(
                candidates=[
                    synthesis_candidate(
                        "SELECT заблокированного приложения",
                        (
                            "condition",
                            "Приложение заблокировано",
                            [self.fact_ids[0]],
                        ),
                        (
                            "changed_factor",
                            "Бит Application Blocked",
                            [self.fact_ids[0]],
                        ),
                        (
                            "consequence",
                            "Карта возвращает 6283",
                            [self.fact_ids[1]],
                        ),
                    )
                ]
            )
        )

        candidate = result.candidates[0]
        self.assertEqual(candidate.payload["condition"], "Приложение заблокировано")
        self.assertEqual(
            candidate.payload["gaps"],
            [
                {
                    "kind": "input_value",
                    "question": (
                        "Какое конкретное входное значение использовать для "
                        "теста «SELECT заблокированного приложения»?"
                    ),
                    "target_paths": ["test_design.input_value"],
                },
                {
                    "kind": "action",
                    "question": (
                        "Какое конкретное воздействие выполнить для теста "
                        "«SELECT заблокированного приложения»?"
                    ),
                    "target_paths": ["test_design.action"],
                },
            ],
        )
        self.assertEqual(
            candidate.payload["consequences"][0]["evidence"][0]["quote"],
            "карта возвращает SW1 SW2 = 6283.",
        )

    def test_cross_page_fact_ids_become_page_local_exact_evidence(self) -> None:
        result = self.synthesize(
            SemanticSynthesisArguments(
                candidates=[
                    synthesis_candidate(
                        "SELECT по AID",
                        (
                            "condition",
                            "Приложение выбирается по AID",
                            [self.fact_ids[2]],
                        ),
                        (
                            "changed_factor",
                            "AID приложения",
                            [self.fact_ids[2]],
                        ),
                        (
                            "action",
                            "Выполнить SELECT по AID",
                            [self.fact_ids[2]],
                        ),
                        (
                            "consequence",
                            "Карта обрабатывает выбранный AID",
                            [self.fact_ids[2]],
                        ),
                    )
                ]
            )
        )

        evidence = result.candidates[0].payload["condition_evidence"]
        self.assertEqual(
            [(item["page"], item["line_start"], item["line_end"]) for item in evidence],
            [(298, 1, 2)],
        )

    def test_multiple_named_consequences_keep_their_own_evidence(
        self,
    ) -> None:
        result = self.synthesize(
            SemanticSynthesisArguments(
                candidates=[
                    synthesis_candidate(
                        "Несколько последствий",
                        (
                            "condition",
                            "Приложение заблокировано",
                            [self.fact_ids[0]],
                        ),
                        (
                            "changed_factor",
                            "Состояние приложения",
                            [self.fact_ids[0]],
                        ),
                        (
                            "consequence",
                            "Карта возвращает 6283",
                            [self.fact_ids[1]],
                        ),
                        (
                            "consequence",
                            "Выбор приложения не выполняется",
                            [self.fact_ids[0]],
                        ),
                    )
                ]
            )
        )

        consequences = result.candidates[0].payload["consequences"]
        self.assertEqual(len(consequences), 2)
        self.assertEqual(
            {
                item["text"]: item["evidence"][0]["quote"]
                for item in consequences
            },
            {
                "Карта возвращает 6283": (
                    "карта возвращает SW1 SW2 = 6283."
                ),
                "Выбор приложения не выполняется": (
                    "Если бит Application Blocked равен 1b,"
                ),
            },
        )

    def test_synthesis_requires_primary_fact_and_required_card_parts(
        self,
    ) -> None:
        with self.assertRaises(SemanticWindowError) as captured:
            self.synthesize(
                SemanticSynthesisArguments(
                    candidates=[
                        {
                            "title": "Неполный сценарий",
                            "condition": {
                                "text": "Условие",
                                "fact_ids": [self.fact_ids[0]],
                            },
                        }
                    ]
                )
            )

        message = str(captured.exception)
        self.assertIn("changed_factor", message)
        self.assertIn("consequences", message)

    def test_fact_scope_is_derived_per_fact_inside_mixed_fragment(
        self,
    ) -> None:
        scoped_window = replace(
            window(),
            lines=(
                replace(window().lines[0], primary=False),
                *window().lines[1:],
            ),
        )
        provisional = WindowPlan(
            schema_version="window-plan-2",
            selection_id="SELECTION_1",
            document_version=self.source.metadata.document_version,
            selection_start=SourcePosition(297, 1),
            selection_end=SourcePosition(298, 2),
            policy_version="policy",
            windows=(scoped_window,),
            plan_hash="",
        )
        scoped_plan = replace(
            provisional,
            plan_hash=provisional.recompute_hash(),
        )
        parent = (
            WindowedAttemptState.planned(
                parent_attempt_id="ATTEMPT_SCOPED",
                selection_id=scoped_plan.selection_id,
                document_version=scoped_plan.document_version,
                expected_workflow_revision="workflow-revision-1",
                policy_version=scoped_plan.policy_version,
                prompt_version="semantic-window-facts",
                schema_version=SEMANTIC_WINDOW_SCHEMA_VERSION,
                window_plan_hash=scoped_plan.plan_hash,
                window_ids=(scoped_window.window_id,),
            )
            .start()
            .start_child(scoped_window.window_id, "ATTEMPT_SCOPED_CHILD")
        )
        facts = SemanticWindowCanonicalizer(self.source).canonicalize(
            parent=parent,
            plan=scoped_plan,
            window_id=scoped_window.window_id,
            child_attempt_id="ATTEMPT_SCOPED_CHILD",
            arguments=SemanticWindowArguments(
                behaviors=[
                    {
                        "title": "Коды ответа SELECT",
                        "summary": "Успех и блокировка приложения",
                        "facts": [
                            {
                                "text": "Overlap-only fact",
                                "line_ids": ["L0001"],
                            },
                            {
                                "text": "Primary-only fact",
                                "line_ids": ["L0002"],
                            },
                            {
                                "text": "Mixed fact",
                                "line_ids": ["L0001", "L0002"],
                            },
                        ],
                    }
                ]
            ),
        )
        context = semantic_synthesis_context(
            document=self.source,
            plan=scoped_plan,
            target_window_id=scoped_window.window_id,
            results=(facts,),
        )
        context_facts = {
            item["text"]: item
            for fragment in context["target_fragments"]
            for group in ("target_facts", "supporting_facts")
            for item in fragment[group]
        }

        self.assertNotIn("primary_fragment_ids", context)
        self.assertNotIn("primary_fact_ids", context)
        self.assertEqual(
            {
                item["text"]
                for fragment in context["target_fragments"]
                for item in fragment["target_facts"]
            },
            {"Primary-only fact", "Mixed fact"},
        )
        self.assertEqual(
            {
                item["text"]
                for fragment in context["target_fragments"]
                for item in fragment["supporting_facts"]
            },
            {"Overlap-only fact"},
        )

        ready = parent.complete_child(
            scoped_window.window_id,
            "ATTEMPT_SCOPED_CHILD",
        ).begin_reconciliation()
        result = self.canonicalizer.canonicalize(
            parent=ready,
            plan=scoped_plan,
            target_window_id=scoped_window.window_id,
            attempt_id="ATTEMPT_SCOPED_SYNTHESIS",
            fact_results=(facts,),
            arguments=SemanticSynthesisArguments(
                candidates=[
                    synthesis_candidate(
                        "Overlap candidate",
                        (
                            "condition",
                            "Условие overlap",
                            [context_facts["Overlap-only fact"]["fact_id"]],
                        ),
                        (
                            "changed_factor",
                            "Фактор overlap",
                            [context_facts["Overlap-only fact"]["fact_id"]],
                        ),
                        (
                            "consequence",
                            "Результат overlap",
                            [context_facts["Overlap-only fact"]["fact_id"]],
                        ),
                    ),
                    synthesis_candidate(
                        "Primary candidate",
                        (
                            "condition",
                            "Условие primary",
                            [context_facts["Primary-only fact"]["fact_id"]],
                        ),
                        (
                            "changed_factor",
                            "Фактор primary",
                            [context_facts["Primary-only fact"]["fact_id"]],
                        ),
                        (
                            "consequence",
                            "Результат primary",
                            [context_facts["Primary-only fact"]["fact_id"]],
                        ),
                    ),
                    synthesis_candidate(
                        "Mixed candidate",
                        (
                            "condition",
                            "Условие mixed",
                            [context_facts["Mixed fact"]["fact_id"]],
                        ),
                        (
                            "changed_factor",
                            "Фактор mixed",
                            [context_facts["Mixed fact"]["fact_id"]],
                        ),
                        (
                            "consequence",
                            "Результат mixed",
                            [context_facts["Mixed fact"]["fact_id"]],
                        ),
                    ),
                ]
            ),
        )

        self.assertEqual(
            {item.payload["title"] for item in result.candidates},
            {"Primary candidate", "Mixed candidate"},
        )
        self.assertIn("исключил 1 context-only", result.explanation)
        assessments = {
            item.position: item.role
            for item in result.primary_line_assessments
        }
        self.assertNotIn(SourcePosition(297, 1), assessments)
        self.assertEqual(assessments[SourcePosition(297, 2)], "evidence")

        with self.assertRaisesRegex(
            SemanticWindowError,
            "не вернул candidate для primary facts",
        ):
            self.canonicalizer.canonicalize(
                parent=ready,
                plan=scoped_plan,
                target_window_id=scoped_window.window_id,
                attempt_id="ATTEMPT_SUPPORT_ONLY",
                fact_results=(facts,),
                arguments=SemanticSynthesisArguments(
                    candidates=[
                        synthesis_candidate(
                            "Support-only candidate",
                            (
                                "condition",
                                "Условие overlap",
                                [context_facts["Overlap-only fact"]["fact_id"]],
                            ),
                            (
                                "changed_factor",
                                "Фактор overlap",
                                [context_facts["Overlap-only fact"]["fact_id"]],
                            ),
                            (
                                "consequence",
                                "Результат overlap",
                                [context_facts["Overlap-only fact"]["fact_id"]],
                            ),
                        )
                    ]
                ),
            )

        with self.assertRaisesRegex(SemanticWindowError, "FACT_UNKNOWN"):
            self.canonicalizer.canonicalize(
                parent=ready,
                plan=scoped_plan,
                target_window_id=scoped_window.window_id,
                attempt_id="ATTEMPT_UNKNOWN_FACT",
                fact_results=(facts,),
                arguments=SemanticSynthesisArguments(
                    candidates=[
                        synthesis_candidate(
                            "Unknown fact candidate",
                            ("condition", "Условие", ["FACT_UNKNOWN"]),
                            (
                                "changed_factor",
                                "Фактор",
                                ["FACT_UNKNOWN"],
                            ),
                            (
                                "consequence",
                                "Результат",
                                ["FACT_UNKNOWN"],
                            ),
                        )
                    ]
                ),
            )

    def test_synthesis_excludes_standalone_neighbor_behavior(
        self,
    ) -> None:
        first_window = replace(
            window(),
            lines=(
                *window().lines[:3],
                replace(window().lines[3], primary=False),
            ),
        )
        second_window = replace(
            window(),
            window_id="SELECTION_1:WINDOW:0002",
            index=1,
            lines=(
                replace(window().lines[2], primary=False),
                *window().lines[3:],
            ),
        )
        provisional = WindowPlan(
            schema_version="window-plan-2",
            selection_id="SELECTION_1",
            document_version=self.source.metadata.document_version,
            selection_start=SourcePosition(297, 1),
            selection_end=SourcePosition(298, 2),
            policy_version="policy",
            windows=(first_window, second_window),
            plan_hash="",
        )
        multi_plan = replace(
            provisional,
            plan_hash=provisional.recompute_hash(),
        )
        parent = (
            WindowedAttemptState.planned(
                parent_attempt_id="ATTEMPT_MULTI",
                selection_id=multi_plan.selection_id,
                document_version=multi_plan.document_version,
                expected_workflow_revision="workflow-revision-1",
                policy_version=multi_plan.policy_version,
                prompt_version="semantic-window-facts",
                schema_version=SEMANTIC_WINDOW_SCHEMA_VERSION,
                window_plan_hash=multi_plan.plan_hash,
                window_ids=(
                    first_window.window_id,
                    second_window.window_id,
                ),
            )
            .start()
            .start_child(first_window.window_id, "ATTEMPT_FIRST")
        )
        fact_canonicalizer = SemanticWindowCanonicalizer(self.source)
        first_facts = fact_canonicalizer.canonicalize(
            parent=parent,
            plan=multi_plan,
            window_id=first_window.window_id,
            child_attempt_id="ATTEMPT_FIRST",
            arguments=SemanticWindowArguments(
                behaviors=[
                    {
                        "title": "Заблокированное приложение",
                        "summary": "Заблокированное приложение возвращает 6283",
                        "facts": [
                            {
                                "text": "Application Blocked равен 1b",
                                "line_ids": ["L0002"],
                            },
                            {
                                "text": "Карта возвращает 6283",
                                "line_ids": ["L0003"],
                            },
                        ],
                    }
                ]
            ),
        )
        parent = (
            parent.complete_child(
                first_window.window_id,
                "ATTEMPT_FIRST",
            ).start_child(second_window.window_id, "ATTEMPT_SECOND")
        )
        second_facts = fact_canonicalizer.canonicalize(
            parent=parent,
            plan=multi_plan,
            window_id=second_window.window_id,
            child_attempt_id="ATTEMPT_SECOND",
            arguments=SemanticWindowArguments(
                behaviors=[
                    {
                        "title": "Выбор по AID",
                        "summary": "Команда SELECT передаёт AID приложения",
                        "facts": [
                            {
                                "text": "CLA 00, INS A4",
                                "line_ids": ["L0002"],
                            },
                            {
                                "text": "Data содержит AID приложения",
                                "line_ids": ["L0003"],
                            },
                        ],
                    }
                ]
            ),
        )
        parent = parent.complete_child(
            second_window.window_id,
            "ATTEMPT_SECOND",
        ).begin_reconciliation()
        neighbor_ids = [
            fact.fact_id
            for fact in second_facts.fragments[0].facts
        ]
        context = semantic_synthesis_context(
            document=self.source,
            plan=multi_plan,
            target_window_id=first_window.window_id,
            results=(first_facts, second_facts),
        )
        available_ids = {
            fact["fact_id"]
            for fragment in context["target_fragments"]
            for group in ("target_facts", "supporting_facts")
            for fact in fragment[group]
        }
        self.assertNotIn(neighbor_ids[0], available_ids)
        self.assertNotIn(neighbor_ids[1], available_ids)


if __name__ == "__main__":
    unittest.main()
