from __future__ import annotations

import unittest
from dataclasses import asdict, replace

from pmi_generator.workbench.application.decomposition import (
    SEMANTIC_WINDOW_SCHEMA_VERSION,
    ConflictPlanError,
    ConflictPlanner,
    ConflictSourceLine,
    PrimaryLineAssessment,
    ReconciliationArguments,
    ReconciliationCaseArguments,
    ReconciliationCasePlanner,
    ReconciliationCaseService,
    ReconciliationError,
    ReconciliationService,
    SemanticSynthesisArguments,
    SemanticSynthesisCanonicalizer,
    SemanticWindowArguments,
    SemanticWindowCanonicalizer,
    WindowAssemblyError,
    WindowAssemblyService,
    ValidatedBoundaryDependency,
    ValidatedWindowCandidate,
    WindowCandidateResult,
    WindowPlanner,
    WindowedAttemptState,
    WindowedDecompositionStore,
    default_windowing_policy,
    reconciliation_tool,
    reconciliation_case_tool,
)
from pmi_generator.workbench.application.llm import (
    ToolContractError,
    TypedToolRegistry,
)
from pmi_generator.workbench.application.prompting import default_policy
from pmi_generator.workbench.application.source import SavedSelection
from pmi_generator.workbench.domain.source import (
    SourceDocument,
    SourcePage,
    SourcePosition,
    SourceSection,
)
from pmi_generator.workbench.infrastructure.storage import (
    InMemoryDatabase,
    InMemoryUnitOfWork,
)


def source_document() -> SourceDocument:
    return SourceDocument(
        pages=(
            SourcePage(
                1,
                "1",
                tuple(f"Строка источника {line}" for line in range(1, 151)),
            ),
            SourcePage(
                2,
                "2",
                tuple(f"Строка источника {line}" for line in range(151, 301)),
            ),
        ),
        sections=(
            SourceSection("root", "1", "Root", ("1",), (1,)),
            SourceSection(
                "child",
                "1.1",
                "Child",
                ("1", "1.1"),
                (2,),
                parent_section_id="root",
            ),
        ),
    )


def setup_plan():
    document = source_document()
    selection = SavedSelection(
        "SELECTION_1",
        "root",
        document.select(document.positions[0], document.positions[-1]),
        document.metadata.document_version,
        "root",
    )
    policy = default_windowing_policy(default_policy())
    plan = WindowPlanner(document, policy).build(selection)
    return document, selection, policy, plan


def candidate(
    window_id: str,
    index: int,
    *positions: SourcePosition,
    text: str = "Одинаковая формулировка",
) -> ValidatedWindowCandidate:
    evidence = [
        {
            "page": position.page_index,
            "line_start": position.line_number,
            "line_end": position.line_number,
            "quote": f"Строка источника {position.line_number}",
        }
        for position in positions
    ]
    return ValidatedWindowCandidate(
        candidate_id=f"{window_id}:CANDIDATE:{index:04d}",
        local_candidate_id=f"local-{index}",
        window_id=window_id,
        payload={
            "title": text,
            "condition": text,
            "changed_factor": text,
            "input_value": None,
            "action": text,
            "condition_evidence": evidence,
            "changed_factor_evidence": evidence,
            "input_value_evidence": [],
            "action_evidence": evidence,
            "consequences": [
                {
                    "text": text,
                    "evidence": evidence,
                }
            ],
            "gaps": [
                {
                    "kind": "input_value",
                    "question": "Какое входное значение использовать?",
                    "target_paths": ["test_design.input_value"],
                }
            ],
        },
        evidence_positions=tuple(positions),
    )


def result(
    plan,
    window_index: int,
    *,
    candidates=(),
    dependencies=(),
    evidence_positions=(),
) -> WindowCandidateResult:
    window = plan.windows[window_index]
    evidence = set(evidence_positions)
    return WindowCandidateResult(
        parent_attempt_id="ATTEMPT_PARENT",
        child_attempt_id=f"ATTEMPT_CHILD_{window_index + 1}",
        window_id=window.window_id,
        plan_hash=plan.plan_hash,
        outcome=(
            "candidates"
            if candidates
            else "boundary_dependency"
            if dependencies
            else "no_local_testable_behavior"
        ),
        explanation="Локальный результат",
        candidates=tuple(candidates),
        boundary_dependencies=tuple(dependencies),
        primary_line_assessments=tuple(
            PrimaryLineAssessment(
                position,
                "evidence" if position in evidence else "context",
                "Локальная классификация",
            )
            for position in window.primary_positions
        ),
    )


def complete_results(
    plan,
    *overrides: WindowCandidateResult,
) -> tuple[WindowCandidateResult, ...]:
    by_window = {item.window_id: item for item in overrides}
    return tuple(
        by_window.get(window.window_id, result(plan, index))
        for index, window in enumerate(plan.windows)
    )


def reconciling_parent(plan, policy) -> WindowedAttemptState:
    state = WindowedAttemptState.planned(
        parent_attempt_id="ATTEMPT_PARENT",
        selection_id=plan.selection_id,
        document_version=plan.document_version,
        expected_workflow_revision="workflow-revision-1",
        policy_version=policy.fingerprint,
        prompt_version=policy.prompt_version,
        schema_version=policy.candidate_schema_version,
        window_plan_hash=plan.plan_hash,
        window_ids=tuple(window.window_id for window in plan.windows),
    ).start()
    for index, window in enumerate(plan.windows, start=1):
        attempt_id = f"ATTEMPT_CHILD_{index}"
        state = state.start_child(window.window_id, attempt_id)
        state = state.complete_child(window.window_id, attempt_id)
    return state.begin_reconciliation()


class ConflictPlannerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.document, self.selection, self.policy, self.plan = setup_plan()
        self.planner = ConflictPlanner(self.document, self.policy)

    def test_overlapping_evidence_builds_one_deterministic_group(self) -> None:
        shared = self.plan.windows[0].primary_positions[-1]
        first = candidate(self.plan.windows[0].window_id, 1, shared)
        second = candidate(self.plan.windows[1].window_id, 1, shared)
        results = complete_results(
            self.plan,
            result(
                self.plan,
                0,
                candidates=(first,),
                evidence_positions=(shared,),
            ),
            result(self.plan, 1, candidates=(second,)),
        )

        forward = self.planner.build(self.plan, results)
        reversed_plan = self.planner.build(self.plan, tuple(reversed(results)))

        self.assertEqual(forward, reversed_plan)
        self.assertEqual(len(forward.groups), 1)
        self.assertEqual(
            set(forward.groups[0].candidate_ids),
            {first.candidate_id, second.candidate_id},
        )
        self.assertIn("overlapping_evidence", forward.groups[0].reasons)

    def test_text_similarity_without_shared_source_does_not_create_group(self) -> None:
        first_position = self.plan.windows[0].primary_positions[0]
        second_position = self.plan.windows[1].primary_positions[0]
        first = candidate(
            self.plan.windows[0].window_id,
            1,
            first_position,
        )
        second = candidate(
            self.plan.windows[1].window_id,
            1,
            second_position,
        )

        conflict_plan = self.planner.build(
            self.plan,
            complete_results(
                self.plan,
                result(
                    self.plan,
                    0,
                    candidates=(first,),
                    evidence_positions=(first_position,),
                ),
                result(
                    self.plan,
                    1,
                    candidates=(second,),
                    evidence_positions=(second_position,),
                ),
            ),
        )

        self.assertEqual(conflict_plan.groups, ())
        self.assertEqual(
            set(conflict_plan.unconflicted_candidate_ids),
            {first.candidate_id, second.candidate_id},
        )

    def test_boundary_dependency_links_only_typed_adjacent_candidate(self) -> None:
        shared = self.plan.windows[0].primary_positions[-1]
        first = candidate(self.plan.windows[0].window_id, 1, shared)
        second = candidate(self.plan.windows[1].window_id, 1, shared)
        dependency = ValidatedBoundaryDependency(
            dependency_id=(
                f"{self.plan.windows[0].window_id}:DEPENDENCY:0001"
            ),
            local_dependency_id="dep-1",
            candidate_id=first.candidate_id,
            direction="after",
            missing_field="consequence",
            source=(
                {
                    "page": 1,
                    "line_start": shared.line_number,
                    "line_end": shared.line_number,
                    "quote": f"Строка источника {shared.line_number}",
                },
            ),
            reason="Последствие продолжается справа",
        )

        conflict_plan = self.planner.build(
            self.plan,
            complete_results(
                self.plan,
                result(
                    self.plan,
                    0,
                    candidates=(first,),
                    dependencies=(dependency,),
                    evidence_positions=(shared,),
                ),
                result(self.plan, 1, candidates=(second,)),
            ),
        )

        self.assertEqual(len(conflict_plan.groups), 1)
        group = conflict_plan.groups[0]
        self.assertEqual(group.dependency_ids, (dependency.dependency_id,))
        self.assertEqual(
            set(group.candidate_ids),
            {first.candidate_id, second.candidate_id},
        )
        self.assertIn("boundary_dependency", group.reasons)

    def test_overlap_candidate_conflicts_with_primary_owner_context(self) -> None:
        overlap = self.plan.windows[0].primary_positions[-1]
        second = candidate(self.plan.windows[1].window_id, 1, overlap)

        conflict_plan = self.planner.build(
            self.plan,
            complete_results(
                self.plan,
                result(self.plan, 0),
                result(self.plan, 1, candidates=(second,)),
            ),
        )

        self.assertEqual(len(conflict_plan.groups), 1)
        self.assertEqual(
            conflict_plan.groups[0].candidate_ids,
            (second.candidate_id,),
        )
        self.assertIn(
            "primary_owner_context",
            conflict_plan.groups[0].reasons,
        )

    def test_opposing_adjacent_dependencies_form_one_typed_group(self) -> None:
        left_position = self.plan.windows[0].primary_positions[-1]
        right_position = self.plan.windows[1].primary_positions[0]
        left = ValidatedBoundaryDependency(
            dependency_id=(
                f"{self.plan.windows[0].window_id}:DEPENDENCY:0001"
            ),
            local_dependency_id="left",
            candidate_id=None,
            direction="after",
            missing_field="consequence",
            source=(
                {
                    "page": 1,
                    "line_start": left_position.line_number,
                    "line_end": left_position.line_number,
                    "quote": f"Строка источника {left_position.line_number}",
                },
            ),
            reason="Продолжение справа",
        )
        right = ValidatedBoundaryDependency(
            dependency_id=(
                f"{self.plan.windows[1].window_id}:DEPENDENCY:0001"
            ),
            local_dependency_id="right",
            candidate_id=None,
            direction="before",
            missing_field="consequence",
            source=(
                {
                    "page": 1,
                    "line_start": right_position.line_number,
                    "line_end": right_position.line_number,
                    "quote": f"Строка источника {right_position.line_number}",
                },
            ),
            reason="Начало слева",
        )

        conflict_plan = self.planner.build(
            self.plan,
            complete_results(
                self.plan,
                result(self.plan, 0, dependencies=(left,)),
                result(self.plan, 1, dependencies=(right,)),
            ),
        )

        self.assertEqual(len(conflict_plan.groups), 1)
        self.assertEqual(
            set(conflict_plan.groups[0].dependency_ids),
            {left.dependency_id, right.dependency_id},
        )
        self.assertEqual(conflict_plan.groups[0].candidate_ids, ())

    def test_group_over_policy_candidate_limit_is_rejected(self) -> None:
        shared = SourcePosition(1, 1)
        candidates = tuple(
            candidate(self.plan.windows[0].window_id, index, shared)
            for index in range(1, self.policy.reconciliation_max_candidates + 2)
        )

        with self.assertRaisesRegex(ConflictPlanError, "candidate budget"):
            self.planner.build(
                self.plan,
                complete_results(
                    self.plan,
                    result(
                        self.plan,
                        0,
                        candidates=candidates,
                        evidence_positions=(shared,),
                    ),
                ),
            )

    def test_group_over_actual_context_token_budget_is_rejected(self) -> None:
        shared = SourcePosition(1, 1)
        first = candidate(
            self.plan.windows[0].window_id,
            1,
            shared,
            text="я" * 50_000,
        )
        second = candidate(
            self.plan.windows[0].window_id,
            2,
            shared,
            text="я" * 50_000,
        )

        with self.assertRaisesRegex(ConflictPlanError, "token budget"):
            self.planner.build(
                self.plan,
                complete_results(
                    self.plan,
                    result(
                        self.plan,
                        0,
                        candidates=(first, second),
                        evidence_positions=(shared,),
                    ),
                ),
            )

    def test_candidate_evidence_outside_immutable_window_is_rejected(self) -> None:
        outside = SourcePosition(2, 150)
        invalid = candidate(
            self.plan.windows[0].window_id,
            1,
            outside,
        )

        with self.assertRaisesRegex(ConflictPlanError, "immutable window"):
            self.planner.build(
                self.plan,
                complete_results(
                    self.plan,
                    result(self.plan, 0, candidates=(invalid,)),
                ),
            )


def resolved_arguments(group) -> ReconciliationArguments:
    return ReconciliationArguments(
        outcome="resolved",
        accepted_candidate_ids=[group.candidate_ids[0]],
        rejected_candidate_ids=list(group.candidate_ids[1:]),
        resolved_dependency_ids=list(group.dependency_ids),
        relations=(
            [
                {
                    "kind": "merge",
                    "candidate_ids": list(group.candidate_ids),
                    "reason": "Один source fragment",
                }
            ]
            if len(group.candidate_ids) > 1
            else []
        ),
        explanation="Конфликт разрешён",
    )


class ReconciliationContractTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.document, self.selection, self.policy, self.plan = setup_plan()
        shared = self.plan.windows[0].primary_positions[-1]
        first = candidate(self.plan.windows[0].window_id, 1, shared)
        second = candidate(self.plan.windows[1].window_id, 1, shared)
        self.conflict_plan = ConflictPlanner(
            self.document,
            self.policy,
        ).build(
            self.plan,
            complete_results(
                self.plan,
                result(
                    self.plan,
                    0,
                    candidates=(first,),
                    evidence_positions=(shared,),
                ),
                result(self.plan, 1, candidates=(second,)),
            ),
        )
        self.group = self.conflict_plan.groups[0]
        self.database = InMemoryDatabase()
        self.parent = reconciling_parent(self.plan, self.policy)
        self.service = ReconciliationService(
            document=self.document,
            uow_factory=lambda: InMemoryUnitOfWork(self.database),
        )

    def test_context_exposes_semantics_but_not_canonical_technical_fields(
        self,
    ) -> None:
        context = self.group.to_context()
        candidate_context = context["candidates"][0]
        payload = candidate_context["payload"]

        self.assertEqual(payload["title"], "Одинаковая формулировка")
        self.assertEqual(
            payload["consequences"],
            ["Одинаковая формулировка"],
        )
        self.assertNotIn("condition_evidence", payload)
        self.assertNotIn("changed_factor_evidence", payload)
        self.assertNotIn("input_value_evidence", payload)
        self.assertNotIn("action_evidence", payload)
        self.assertNotIn("gaps", payload)
        self.assertNotIn("target_paths", str(context))

    async def test_single_candidate_wire_schema_requires_empty_relations(
        self,
    ) -> None:
        candidate_id = self.group.candidate_ids[0]
        single_group = replace(
            self.group,
            candidate_ids=(candidate_id,),
            candidates=(self.group.candidates[0],),
        )
        context = {"group": single_group.to_context()}
        spec = reconciliation_tool()
        wire_schema = spec.openai_schema(context)["function"]["parameters"]
        relations_schema = wire_schema["properties"]["relations"]

        self.assertEqual(
            relations_schema,
            {"type": "array", "enum": [[]]},
        )

        registry = TypedToolRegistry()
        registry.register(spec)
        invalid = ReconciliationArguments(
            outcome="resolved",
            accepted_candidate_ids=[candidate_id],
            rejected_candidate_ids=[],
            resolved_dependency_ids=[],
            relations=[
                {
                    "kind": "merge",
                    "candidate_ids": [candidate_id],
                    "reason": "Candidate связан с самим собой",
                }
            ],
            explanation="Candidate принят",
        )
        with self.assertRaises(ToolContractError):
            registry.decode(
                {
                    "id": "call-single-candidate-relation",
                    "name": "submit_reconciliation",
                    "arguments": asdict(invalid),
                },
                ("submit_reconciliation",),
                context=context,
            )

        valid = replace(invalid, relations=[])
        decoded = registry.decode(
            {
                "id": "call-single-candidate",
                "name": "submit_reconciliation",
                "arguments": asdict(valid),
            },
            ("submit_reconciliation",),
            context=context,
        )
        self.assertEqual(decoded.arguments.relations, [])

    async def test_relation_source_is_derived_across_page_boundary(self) -> None:
        page_end = SourcePosition(1, 150)
        next_page_start = SourcePosition(2, 1)
        cross_page_group = replace(
            self.group,
            candidates=tuple(
                replace(
                    item,
                    evidence_positions=(page_end, next_page_start),
                )
                for item in self.group.candidates
            ),
            source_lines=(
                ConflictSourceLine(page_end, self.document.line(page_end)),
                ConflictSourceLine(
                    next_page_start,
                    self.document.line(next_page_start),
                ),
            ),
        )

        decision = self.service.validate(
            parent=self.parent,
            plan_hash=self.plan.plan_hash,
            group=cross_page_group,
            attempt_id="ATTEMPT_RECONCILE_CROSS_PAGE",
            arguments=resolved_arguments(cross_page_group),
        )

        self.assertEqual(
            [
                (
                    item["page"],
                    item["line_start"],
                    item["line_end"],
                )
                for item in decision.relations[0]["source"]
            ],
            [(1, 150, 150), (2, 1, 1)],
        )
        self.assertEqual(
            decision.relations[0]["source"][0]["quote"],
            "Строка источника 150",
        )
        self.assertEqual(
            decision.relations[0]["source"][1]["quote"],
            "Строка источника 151",
        )

    async def test_relation_source_rejects_evidence_outside_group(self) -> None:
        outside = SourcePosition(2, 1)
        tampered_group = replace(
            self.group,
            candidates=tuple(
                replace(item, evidence_positions=(outside,))
                for item in self.group.candidates
            ),
        )

        with self.assertRaisesRegex(
            ReconciliationError,
            "bounded conflict context",
        ):
            self.service.validate(
                parent=self.parent,
                plan_hash=self.plan.plan_hash,
                group=tampered_group,
                attempt_id="ATTEMPT_RECONCILE_OUTSIDE",
                arguments=resolved_arguments(tampered_group),
            )

    async def test_unresolved_result_is_bounded_and_does_not_mutate_domain(self) -> None:
        unresolved = ReconciliationArguments(
            outcome="unresolved",
            accepted_candidate_ids=[],
            rejected_candidate_ids=[],
            resolved_dependency_ids=[],
            relations=[],
            explanation="Нельзя безопасно выбрать candidate",
        )

        decision = self.service.accept(
            parent=self.parent,
            plan_hash=self.plan.plan_hash,
            group=self.group,
            attempt_id="ATTEMPT_RECONCILE_1",
            arguments=unresolved,
            raw_arguments=asdict(unresolved),
        )

        self.assertEqual(decision.outcome, "unresolved")
        self.assertFalse(
            any(
                kind in {"decomposition", "card_skeleton"}
                for kind, _record_id in self.database.records
            )
        )

    async def test_reconciliation_rejects_parent_before_children_complete(self) -> None:
        running = WindowedAttemptState.planned(
            parent_attempt_id="ATTEMPT_OTHER",
            selection_id=self.plan.selection_id,
            document_version=self.plan.document_version,
            expected_workflow_revision="workflow-revision-1",
            policy_version=self.policy.fingerprint,
            prompt_version=self.policy.prompt_version,
            schema_version=self.policy.candidate_schema_version,
            window_plan_hash=self.plan.plan_hash,
            window_ids=tuple(
                window.window_id for window in self.plan.windows
            ),
        ).start()

        with self.assertRaisesRegex(ReconciliationError, "стадии reconciliation"):
            self.service.validate(
                parent=running,
                plan_hash=self.plan.plan_hash,
                group=self.group,
                attempt_id="ATTEMPT_RECONCILE_EARLY",
                arguments=resolved_arguments(self.group),
            )

    async def test_conflict_group_cannot_be_reused_by_new_parent(self) -> None:
        other_parent = replace(
            self.parent,
            parent_attempt_id="ATTEMPT_OTHER",
        )

        with self.assertRaisesRegex(ReconciliationError, "другому parent"):
            self.service.validate(
                parent=other_parent,
                plan_hash=self.plan.plan_hash,
                group=self.group,
                attempt_id="ATTEMPT_RECONCILE_OTHER",
                arguments=resolved_arguments(self.group),
            )


class WindowAssemblyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.document, self.selection, self.policy, self.plan = setup_plan()
        self.parent = reconciling_parent(self.plan, self.policy)
        self.database = InMemoryDatabase()
        self.id_counter = 0
        self.service = WindowAssemblyService(
            document=self.document,
            uow_factory=lambda: InMemoryUnitOfWork(self.database),
            next_id=self.next_id,
        )
        shared = self.plan.windows[0].primary_positions[-1]
        self.first = candidate(
            self.plan.windows[0].window_id,
            1,
            shared,
            text="Первый candidate",
        )
        self.second = candidate(
            self.plan.windows[1].window_id,
            1,
            shared,
            text="Дубликат из overlap",
        )
        self.results = complete_results(
            self.plan,
            result(
                self.plan,
                0,
                candidates=(self.first,),
                evidence_positions=(shared,),
            ),
            result(self.plan, 1, candidates=(self.second,)),
        )
        self.conflicts = ConflictPlanner(
            self.document,
            self.policy,
        ).build(self.plan, self.results)
        group = self.conflicts.groups[0]
        self.decision = ReconciliationService(
            document=self.document,
            uow_factory=lambda: InMemoryUnitOfWork(self.database),
        ).validate(
            parent=self.parent,
            plan_hash=self.plan.plan_hash,
            group=group,
            attempt_id="ATTEMPT_RECONCILE_1",
            arguments=resolved_arguments(group),
        )

    def next_id(self, prefix: str) -> str:
        self.id_counter += 1
        return f"{prefix}_{self.id_counter:04d}"

    def test_assembly_is_order_independent_and_classifies_every_line_once(self) -> None:
        forward = self.service.build(
            selection=self.selection,
            parent=self.parent,
            plan=self.plan,
            results=self.results,
            conflict_plan=self.conflicts,
            decisions=(self.decision,),
        )
        reversed_result = self.service.build(
            selection=self.selection,
            parent=self.parent,
            plan=self.plan,
            results=tuple(reversed(self.results)),
            conflict_plan=self.conflicts,
            decisions=(self.decision,),
        )

        self.assertEqual(forward, reversed_result)
        self.assertEqual(forward.outcome, "skeletons_created")
        self.assertEqual(len(forward.skeletons), 1)
        self.assertEqual(len(forward.line_assessments), 300)
        evidence = [
            item
            for item in forward.line_assessments
            if item["role"] == "evidence"
        ]
        self.assertEqual(
            evidence,
            [
                {
                    "page": 1,
                    "line": (
                        self.plan.windows[0]
                        .primary_positions[-1]
                        .line_number
                    ),
                    "role": "evidence",
                    "reason": "Строка используется принятым candidate",
                }
            ],
        )

    def test_same_text_with_distinct_evidence_remains_two_skeletons(self) -> None:
        first_position = self.plan.windows[0].primary_positions[0]
        second_position = self.plan.windows[1].primary_positions[0]
        first = candidate(
            self.plan.windows[0].window_id,
            1,
            first_position,
        )
        second = candidate(
            self.plan.windows[1].window_id,
            1,
            second_position,
        )
        results = complete_results(
            self.plan,
            result(
                self.plan,
                0,
                candidates=(first,),
                evidence_positions=(first_position,),
            ),
            result(
                self.plan,
                1,
                candidates=(second,),
                evidence_positions=(second_position,),
            ),
        )
        conflicts = ConflictPlanner(
            self.document,
            self.policy,
        ).build(self.plan, results)

        assembled = self.service.build(
            selection=self.selection,
            parent=self.parent,
            plan=self.plan,
            results=results,
            conflict_plan=conflicts,
            decisions=(),
        )

        self.assertEqual(len(assembled.skeletons), 2)

    def test_empty_outcomes_are_derived_only_after_all_windows(self) -> None:
        no_local_results = tuple(
            result(self.plan, index)
            for index in range(len(self.plan.windows))
        )
        no_local_conflicts = ConflictPlanner(
            self.document,
            self.policy,
        ).build(self.plan, no_local_results)

        no_behavior = self.service.build(
            selection=self.selection,
            parent=self.parent,
            plan=self.plan,
            results=no_local_results,
            conflict_plan=no_local_conflicts,
            decisions=(),
        )

        self.assertEqual(no_behavior.outcome, "no_testable_behavior")

    def test_resolved_boundary_without_candidate_is_insufficient_selection(self) -> None:
        left_position = self.plan.windows[0].primary_positions[-1]
        right_position = self.plan.windows[1].primary_positions[0]
        left = ValidatedBoundaryDependency(
            dependency_id=(
                f"{self.plan.windows[0].window_id}:DEPENDENCY:0001"
            ),
            local_dependency_id="left",
            candidate_id=None,
            direction="after",
            missing_field="consequence",
            source=(
                {
                    "page": 1,
                    "line_start": left_position.line_number,
                    "line_end": left_position.line_number,
                    "quote": f"Строка источника {left_position.line_number}",
                },
            ),
            reason="Продолжение справа",
        )
        right = ValidatedBoundaryDependency(
            dependency_id=(
                f"{self.plan.windows[1].window_id}:DEPENDENCY:0001"
            ),
            local_dependency_id="right",
            candidate_id=None,
            direction="before",
            missing_field="consequence",
            source=(
                {
                    "page": 1,
                    "line_start": right_position.line_number,
                    "line_end": right_position.line_number,
                    "quote": f"Строка источника {right_position.line_number}",
                },
            ),
            reason="Начало слева",
        )
        results = complete_results(
            self.plan,
            result(self.plan, 0, dependencies=(left,)),
            result(self.plan, 1, dependencies=(right,)),
        )
        conflicts = ConflictPlanner(
            self.document,
            self.policy,
        ).build(self.plan, results)
        group = conflicts.groups[0]
        resolution = ReconciliationService(
            document=self.document,
            uow_factory=lambda: InMemoryUnitOfWork(self.database),
        ).validate(
            parent=self.parent,
            plan_hash=self.plan.plan_hash,
            group=group,
            attempt_id="ATTEMPT_RECONCILE_BOUNDARY",
            arguments=ReconciliationArguments(
                outcome="resolved",
                accepted_candidate_ids=[],
                rejected_candidate_ids=[],
                resolved_dependency_ids=list(group.dependency_ids),
                relations=[],
                explanation="Полный candidate не восстановлен",
            ),
        )

        assembled = self.service.build(
            selection=self.selection,
            parent=self.parent,
            plan=self.plan,
            results=results,
            conflict_plan=conflicts,
            decisions=(resolution,),
        )

        self.assertEqual(assembled.outcome, "insufficient_selection")

    def test_unresolved_or_invalid_candidate_never_mutates_domain(self) -> None:
        unresolved = replace(
            self.decision,
            outcome="unresolved",
            accepted_candidate_ids=(),
            rejected_candidate_ids=(),
            relations=(),
            explanation="Нельзя разрешить",
        )
        with self.assertRaisesRegex(WindowAssemblyError, "unresolved"):
            self.service.apply(
                selection=self.selection,
                parent=self.parent,
                plan=self.plan,
                results=self.results,
                conflict_plan=self.conflicts,
                decisions=(unresolved,),
            )

        self.assertFalse(self.database.records)

        invalid_first = replace(
            self.first,
            payload={**self.first.payload, "title": ""},
        )
        invalid_results = (
            result(
                self.plan,
                0,
                candidates=(invalid_first,),
                evidence_positions=self.first.evidence_positions,
            ),
            *self.results[1:],
        )
        with self.assertRaisesRegex(WindowAssemblyError, "изменённые candidates"):
            self.service.apply(
                selection=self.selection,
                parent=self.parent,
                plan=self.plan,
                results=invalid_results,
                conflict_plan=self.conflicts,
                decisions=(self.decision,),
            )

        self.assertFalse(self.database.records)

    def test_persisted_cancel_rejects_late_assembly_atomically(self) -> None:
        store = WindowedDecompositionStore(
            lambda: InMemoryUnitOfWork(self.database)
        )
        store.save(
            self.parent.cancel("Пользователь отменил операцию"),
            self.plan,
        )

        with self.assertRaisesRegex(
            WindowAssemblyError,
            "изменился после начала assembly",
        ):
            self.service.apply(
                selection=self.selection,
                parent=self.parent,
                plan=self.plan,
                results=self.results,
                conflict_plan=self.conflicts,
                decisions=(self.decision,),
            )

        self.assertNotIn(
            ("decomposition", self.selection.selection_id),
            self.database.records,
        )
        self.assertFalse(
            any(
                kind == "card_skeleton"
                for kind, _record_id in self.database.records
            )
        )

    def test_apply_persists_one_domain_result_and_completed_parent_atomically(
        self,
    ) -> None:
        outcome = self.service.apply(
            selection=self.selection,
            parent=self.parent,
            plan=self.plan,
            results=self.results,
            conflict_plan=self.conflicts,
            decisions=(self.decision,),
        )

        self.assertEqual(outcome.parent.status.value, "completed")
        self.assertEqual(len(outcome.decomposition.skeleton_ids), 1)
        self.assertIn(
            ("decomposition", self.selection.selection_id),
            self.database.records,
        )
        self.assertIn(
            (
                "decomposition_windowed_attempt",
                self.parent.parent_attempt_id,
            ),
            self.database.records,
        )
        decomposition_events = [
            event
            for event in self.database.events
            if event.event_type == "декомпозиция сохранена"
        ]
        self.assertEqual(len(decomposition_events), 1)

    def test_semantic_results_reconcile_and_apply_once(self) -> None:
        state = WindowedAttemptState.planned(
            parent_attempt_id="ATTEMPT_SEMANTIC",
            selection_id=self.plan.selection_id,
            document_version=self.plan.document_version,
            expected_workflow_revision="workflow-revision-1",
            policy_version=self.plan.policy_version,
            prompt_version="semantic-window-1",
            schema_version=SEMANTIC_WINDOW_SCHEMA_VERSION,
            window_plan_hash=self.plan.plan_hash,
            window_ids=tuple(
                window.window_id for window in self.plan.windows
            ),
        ).start()
        canonicalizer = SemanticWindowCanonicalizer(self.document)
        fact_results = []
        for index, window in enumerate(self.plan.windows):
            attempt_id = f"ATTEMPT_SEMANTIC_CHILD_{index + 1}"
            state = state.start_child(window.window_id, attempt_id)
            primary_line_id = next(
                f"L{line_index:04d}"
                for line_index, line in enumerate(window.lines, start=1)
                if line.primary
            )
            behaviors = (
                [
                    {
                        "title": "Одинаковая формулировка",
                        "summary": "Одинаковая формулировка",
                        "facts": [
                            {
                                "text": "Одинаковая формулировка",
                                "line_ids": [primary_line_id],
                            }
                        ],
                    }
                ]
                if index < 2
                else []
            )
            fact_results.append(
                canonicalizer.canonicalize(
                    parent=state,
                    plan=self.plan,
                    window_id=window.window_id,
                    child_attempt_id=attempt_id,
                    arguments=SemanticWindowArguments(
                        behaviors=behaviors
                    ),
                )
            )
            state = state.complete_child(window.window_id, attempt_id)
        state = state.begin_reconciliation()
        facts = tuple(fact_results)
        synthesis = SemanticSynthesisCanonicalizer(self.document)
        semantic_results = tuple(
            synthesis.canonicalize(
                parent=state,
                plan=self.plan,
                target_window_id=window.window_id,
                attempt_id=f"ATTEMPT_SYNTHESIS_{index + 1}",
                fact_results=facts,
                arguments=SemanticSynthesisArguments(
                    candidates=(
                        [
                            {
                                "title": "Одинаковая формулировка",
                                "condition": {
                                    "text": "Одинаковая формулировка",
                                    "fact_ids": [
                                        fact_results[index]
                                        .fragments[0]
                                        .facts[0]
                                        .fact_id
                                    ],
                                },
                                "changed_factor": {
                                    "text": "Одинаковая формулировка",
                                    "fact_ids": [
                                        fact_results[index]
                                        .fragments[0]
                                        .facts[0]
                                        .fact_id
                                    ],
                                },
                                "input_value": {
                                    "text": "Одинаковая формулировка",
                                    "fact_ids": [
                                        fact_results[index]
                                        .fragments[0]
                                        .facts[0]
                                        .fact_id
                                    ],
                                },
                                "action": {
                                    "text": "Одинаковая формулировка",
                                    "fact_ids": [
                                        fact_results[index]
                                        .fragments[0]
                                        .facts[0]
                                        .fact_id
                                    ],
                                },
                                "consequences": [
                                    {
                                        "text": "Одинаковая формулировка",
                                        "fact_ids": [
                                            fact_results[index]
                                            .fragments[0]
                                            .facts[0]
                                            .fact_id
                                        ],
                                    }
                                ],
                            }
                        ]
                        if index < 2
                        else []
                    )
                ),
            )
            for index, window in enumerate(self.plan.windows)
        )
        conflicts = ConflictPlanner(
            self.document,
            self.policy,
        ).build(self.plan, semantic_results)

        outcome = self.service.apply(
            selection=self.selection,
            parent=state,
            plan=self.plan,
            results=semantic_results,
            conflict_plan=conflicts,
            decisions=(),
        )

        self.assertEqual(conflicts.groups, ())
        self.assertEqual(len(outcome.decomposition.skeleton_ids), 2)
        decomposition_events = [
            event
            for event in self.database.events
            if event.event_type == "декомпозиция сохранена"
        ]
        self.assertEqual(len(decomposition_events), 1)


class ReconciliationCaseContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.document, self.selection, self.policy, self.plan = setup_plan()
        shared = self.plan.windows[0].primary_positions[-1]
        first = candidate(self.plan.windows[0].window_id, 1, shared)
        second = candidate(self.plan.windows[1].window_id, 1, shared)
        self.group = ConflictPlanner(
            self.document,
            self.policy,
        ).build(
            self.plan,
            complete_results(
                self.plan,
                result(
                    self.plan,
                    0,
                    candidates=(first,),
                    evidence_positions=(shared,),
                ),
                result(self.plan, 1, candidates=(second,)),
            ),
        ).groups[0]
        self.parent = reconciling_parent(self.plan, self.policy)
        self.database = InMemoryDatabase()
        self.planner = ReconciliationCasePlanner(max_cases=128)
        self.service = ReconciliationCaseService(
            document=self.document,
            uow_factory=lambda: InMemoryUnitOfWork(self.database),
        )

    def test_pair_tool_call_contains_only_semantic_decision(self) -> None:
        case = self.planner.build(self.group)[0]
        context = {"case": case.to_context()}
        schema = reconciliation_case_tool().schema_for(context)

        self.assertEqual(
            set(schema["properties"]),
            {"decision", "reason"},
        )
        self.assertNotIn("candidate_id", str(context))
        self.assertNotIn("accepted_candidate_ids", str(schema))
        self.assertNotIn("relations", str(schema))

        registry = TypedToolRegistry()
        registry.register(reconciliation_case_tool())
        decoded = registry.decode(
            {
                "id": "call-case",
                "name": "submit_reconciliation_case",
                "arguments": {
                    "decision": "keep_separate",
                    "reason": "Это независимые проверки",
                },
            },
            ("submit_reconciliation_case",),
            context=context,
        )

        self.assertEqual(
            decoded.arguments,
            ReconciliationCaseArguments(
                decision="keep_separate",
                reason="Это независимые проверки",
            ),
        )
        with self.assertRaises(ToolContractError):
            registry.decode(
                {
                    "id": "call-case-with-technical-id",
                    "name": "submit_reconciliation_case",
                    "arguments": {
                        "decision": "keep_separate",
                        "reason": "Это независимые проверки",
                        "candidate_ids": list(case.candidate_ids),
                    },
                },
                ("submit_reconciliation_case",),
                context=context,
            )

    def test_application_assembles_partition_without_model_ids(self) -> None:
        cases = self.planner.build(self.group)
        self.assertEqual(len(cases), 1)
        case = cases[0]
        case_decision = self.service.validate_case(
            parent=self.parent,
            plan_hash=self.plan.plan_hash,
            group=self.group,
            case=case,
            attempt_id="ATTEMPT_CASE",
            arguments=ReconciliationCaseArguments(
                decision="duplicate_keep_a",
                reason="B является частичным дубликатом A",
            ),
        )

        decision = self.service.assemble(
            parent=self.parent,
            plan_hash=self.plan.plan_hash,
            group=self.group,
            attempt_id="ATTEMPT_COORDINATOR",
            cases=cases,
            case_decisions=(case_decision,),
        )

        self.assertEqual(
            decision.accepted_candidate_ids,
            (case.candidate_ids[0],),
        )
        self.assertEqual(
            decision.rejected_candidate_ids,
            (case.candidate_ids[1],),
        )
        self.assertEqual(decision.relations[0]["kind"], "merge")

    def test_shared_fragment_can_be_covered_by_two_retained_candidates(
        self,
    ) -> None:
        shared = self.group.candidates[0].evidence_positions[0]
        common = candidate(
            self.plan.windows[1].window_id,
            2,
            shared,
            text="Общее поле INS='DC'",
        )
        dense_group = replace(
            self.group,
            candidate_ids=(
                *self.group.candidate_ids,
                common.candidate_id,
            ),
            candidates=(
                *self.group.candidates,
                common,
            ),
        )
        cases = self.planner.build(dense_group)
        full_a, full_b, common_id = dense_group.candidate_ids
        decisions_by_pair = {
            (full_a, full_b): "keep_separate",
            (full_a, common_id): "duplicate_keep_a",
            (full_b, common_id): "duplicate_keep_a",
        }
        case_decisions = tuple(
            self.service.validate_case(
                parent=self.parent,
                plan_hash=self.plan.plan_hash,
                group=dense_group,
                case=case,
                attempt_id=f"ATTEMPT_SHARED_{index}",
                arguments=ReconciliationCaseArguments(
                    decision=decisions_by_pair[case.candidate_ids],
                    reason="Общий fragment покрыт полным сценарием",
                ),
            )
            for index, case in enumerate(cases, start=1)
        )

        decision = self.service.assemble(
            parent=self.parent,
            plan_hash=self.plan.plan_hash,
            group=dense_group,
            attempt_id="ATTEMPT_SHARED_COORDINATOR",
            cases=cases,
            case_decisions=case_decisions,
        )

        self.assertEqual(
            decision.accepted_candidate_ids,
            (full_a, full_b),
        )
        self.assertEqual(
            decision.rejected_candidate_ids,
            (common_id,),
        )
        self.assertEqual(
            [relation["kind"] for relation in decision.relations],
            ["split", "merge", "merge"],
        )

    def test_dense_overlap_plans_every_direct_pair_and_rejects_cycles(
        self,
    ) -> None:
        shared = self.group.candidates[0].evidence_positions[0]
        third = candidate(
            self.plan.windows[1].window_id,
            2,
            shared,
        )
        dense_group = replace(
            self.group,
            candidate_ids=(
                *self.group.candidate_ids,
                third.candidate_id,
            ),
            candidates=(
                *self.group.candidates,
                third,
            ),
        )
        cases = self.planner.build(dense_group)

        self.assertEqual(len(cases), 3)
        self.assertTrue(
            all(case.kind == "candidate_pair" for case in cases)
        )
        decisions_by_pair = {
            cases[0].candidate_ids: "duplicate_keep_a",
            cases[1].candidate_ids: "duplicate_keep_b",
            cases[2].candidate_ids: "duplicate_keep_a",
        }
        case_decisions = tuple(
            self.service.validate_case(
                parent=self.parent,
                plan_hash=self.plan.plan_hash,
                group=dense_group,
                case=case,
                attempt_id=f"ATTEMPT_{index}",
                arguments=ReconciliationCaseArguments(
                    decision=decisions_by_pair[case.candidate_ids],
                    reason="Sanitized cyclic duplicate decision",
                ),
            )
            for index, case in enumerate(cases, start=1)
        )

        with self.assertRaisesRegex(ReconciliationError, "цикл"):
            self.service.assemble(
                parent=self.parent,
                plan_hash=self.plan.plan_hash,
                group=dense_group,
                attempt_id="ATTEMPT_COORDINATOR",
                cases=cases,
                case_decisions=case_decisions,
            )

        with self.assertRaisesRegex(ReconciliationError, "case budget"):
            ReconciliationCasePlanner(max_cases=2).build(dense_group)

    def test_case_and_assembled_group_survive_exact_restart(self) -> None:
        cases = self.planner.build(self.group)
        case = cases[0]
        arguments = ReconciliationCaseArguments(
            decision="keep_separate",
            reason="Проверки имеют разные последствия",
        )
        saved_case = self.service.accept_case(
            parent=self.parent,
            plan_hash=self.plan.plan_hash,
            group=self.group,
            case=case,
            attempt_id="ATTEMPT_CASE_PERSISTED",
            arguments=arguments,
            raw_arguments=asdict(arguments),
        )

        self.assertEqual(
            self.service.load_case(
                parent_attempt_id=self.parent.parent_attempt_id,
                group=self.group,
                case=case,
                plan_hash=self.plan.plan_hash,
            ),
            saved_case,
        )
        saved_group = self.service.assemble(
            parent=self.parent,
            plan_hash=self.plan.plan_hash,
            group=self.group,
            attempt_id="ATTEMPT_COORDINATOR_PERSISTED",
            cases=cases,
            case_decisions=(saved_case,),
        )
        self.assertEqual(
            self.service.load(
                parent_attempt_id=self.parent.parent_attempt_id,
                group=self.group,
                cases=cases,
                plan_hash=self.plan.plan_hash,
            ),
            saved_group,
        )

    def test_tampered_case_fingerprint_is_rejected_on_restart(self) -> None:
        case = self.planner.build(self.group)[0]
        arguments = ReconciliationCaseArguments(
            decision="keep_separate",
            reason="Проверки независимы",
        )
        self.service.accept_case(
            parent=self.parent,
            plan_hash=self.plan.plan_hash,
            group=self.group,
            case=case,
            attempt_id="ATTEMPT_CASE_TAMPERED",
            arguments=arguments,
            raw_arguments=asdict(arguments),
        )
        record_id = (
            f"{self.parent.parent_attempt_id}:"
            f"{self.group.group_id}:{case.case_id}"
        )
        record = self.database.records[
            (
                self.service.CASE_RECORD_KIND,
                record_id,
            )
        ]
        record.payload["decision"]["reason"] = "Подмена"

        with self.assertRaisesRegex(
            ReconciliationError,
            "fingerprint",
        ):
            self.service.load_case(
                parent_attempt_id=self.parent.parent_attempt_id,
                group=self.group,
                case=case,
                plan_hash=self.plan.plan_hash,
            )


if __name__ == "__main__":
    unittest.main()
