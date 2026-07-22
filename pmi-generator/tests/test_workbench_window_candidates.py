from __future__ import annotations

import unittest
from dataclasses import asdict, replace

from pmi_generator.workbench.application.decomposition import (
    WindowCandidateArguments,
    WindowCandidateError,
    WindowCandidateFlow,
    WindowCandidateService,
    WindowPlanner,
    WindowedAttemptState,
    default_windowing_policy,
    window_candidates_tool,
)
from pmi_generator.workbench.application.llm import (
    LlmToolRuntime,
    RawCompletion,
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
from pmi_generator.workbench.infrastructure.llm import ScriptedLlmTransport
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


def saved_selection(document: SourceDocument) -> SavedSelection:
    return SavedSelection(
        "SELECTION_1",
        "root",
        document.select(document.positions[0], document.positions[-1]),
        document.metadata.document_version,
        "root",
    )


def candidate() -> dict[str, object]:
    return {
        "local_candidate_id": "candidate-1",
        "title": "Проверка поведения",
        "condition": "Строка источника 1",
        "changed_factor": "Строка источника 2",
        "input_value": None,
        "action": "Строка источника 2",
        "condition_ranges": [{"page": 1, "line_start": 1, "line_end": 1}],
        "changed_factor_ranges": [
            {"page": 1, "line_start": 2, "line_end": 2}
        ],
        "input_value_ranges": [],
        "action_ranges": [{"page": 1, "line_start": 2, "line_end": 2}],
        "consequences": [
            {
                "text": "Строка источника 3",
                "evidence_ranges": [
                    {"page": 1, "line_start": 3, "line_end": 3}
                ],
            }
        ],
        "gaps": [
            {
                "kind": "input_value",
                "question": "Какое значение использовать?",
                "target_paths": ["test_design.input_value"],
            }
        ],
    }


def arguments() -> WindowCandidateArguments:
    return WindowCandidateArguments(
        outcome="candidates",
        explanation="Локальные candidates построены",
        candidates=[candidate()],
        boundary_dependencies=[],
    )


class WindowCandidateContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.document = source_document()
        self.selection = saved_selection(self.document)
        self.policy = default_windowing_policy(default_policy())
        self.plan = WindowPlanner(self.document, self.policy).build(self.selection)
        self.window = self.plan.windows[0]
        self.state = (
            WindowedAttemptState.planned(
                parent_attempt_id="ATTEMPT_PARENT",
                selection_id=self.selection.selection_id,
                document_version=self.selection.document_version,
                expected_workflow_revision="workflow-revision-1",
                policy_version=self.policy.fingerprint,
                prompt_version=self.policy.prompt_version,
                schema_version="window-candidates-5",
                window_plan_hash=self.plan.plan_hash,
                window_ids=tuple(
                    window.window_id for window in self.plan.windows
                ),
            )
            .start()
            .start_child(self.window.window_id, "ATTEMPT_CHILD_1")
        )
        self.database = InMemoryDatabase()
        self.service = WindowCandidateService(
            document=self.document,
            uow_factory=lambda: InMemoryUnitOfWork(self.database),
        )
        self.primary_end = self.window.primary_positions[-1].line_number

    def test_tool_schema_is_exact_and_forbids_model_domain_ids(self) -> None:
        registry = TypedToolRegistry()
        registry.register(window_candidates_tool())

        decoded = registry.decode(
            {
                "id": "call-1",
                "name": "submit_window_candidates",
                "arguments": asdict(arguments()),
            },
            ("submit_window_candidates",),
        )

        self.assertIsInstance(decoded.arguments, WindowCandidateArguments)
        malformed = candidate()
        malformed["skeleton_id"] = "SKELETON_MODEL_ASSIGNED"
        with self.assertRaises(ToolContractError):
            registry.decode(
                {
                    "id": "call-2",
                    "name": "submit_window_candidates",
                    "arguments": {
                        **asdict(arguments()),
                        "candidates": [malformed],
                    },
                },
                ("submit_window_candidates",),
            )
        unexpected_assessments = asdict(arguments())
        unexpected_assessments["primary_line_assessments"] = []
        with self.assertRaises(ToolContractError):
            registry.decode(
                {
                    "id": "call-3",
                    "name": "submit_window_candidates",
                    "arguments": unexpected_assessments,
                },
                ("submit_window_candidates",),
            )

    def test_validation_copies_source_and_does_not_mutate_domain(self) -> None:
        raw = {
            **asdict(arguments()),
        }

        result = self.service.accept(
            parent=self.state,
            plan=self.plan,
            window_id=self.window.window_id,
            child_attempt_id="ATTEMPT_CHILD_1",
            arguments=arguments(),
            raw_arguments=raw,
        )

        self.assertEqual(
            result.candidates[0].candidate_id,
            f"{self.window.window_id}:CANDIDATE:0001",
        )
        self.assertEqual(
            result.candidates[0].payload["condition_evidence"][0]["quote"],
            "Строка источника 1",
        )
        self.assertNotIn(("decomposition", self.selection.selection_id), self.database.records)
        self.assertFalse(
            any(kind == "card_skeleton" for kind, _record_id in self.database.records)
        )
        stored = self.database.records[
            (
                "decomposition_window_result",
                f"ATTEMPT_PARENT:{self.window.window_id}",
            )
        ]
        self.assertEqual(stored.payload["raw_arguments"], raw)
        self.assertEqual(
            stored.payload["validated"]["candidates"][0]["candidate_id"],
            result.candidates[0].candidate_id,
        )

    def test_candidate_coordinates_must_stay_inside_window(self) -> None:
        proposed = candidate()
        proposed["condition_ranges"] = [
            {"page": 2, "line_start": 31, "line_end": 31}
        ]
        invalid = WindowCandidateArguments(
            outcome="candidates",
            explanation="Candidate построен",
            candidates=[proposed],
            boundary_dependencies=[],
        )

        with self.assertRaisesRegex(WindowCandidateError, "окна"):
            self.service.validate(
                parent=self.state,
                plan=self.plan,
                window_id=self.window.window_id,
                child_attempt_id="ATTEMPT_CHILD_1",
                arguments=invalid,
            )

    def test_cross_page_range_error_lists_exact_window_coordinates(self) -> None:
        window = next(
            item
            for item in self.plan.windows
            if SourcePosition(1, 150) in {line.position for line in item.lines}
            and SourcePosition(2, 1) in {line.position for line in item.lines}
        )
        primary = window.primary_positions[0]
        source_range = {
            "page": primary.page_index,
            "line_start": primary.line_number,
            "line_end": primary.line_number,
        }
        proposed = candidate()
        proposed["condition_ranges"] = [source_range]
        proposed["changed_factor_ranges"] = [source_range]
        proposed["action_ranges"] = [source_range]
        proposed["consequences"][0]["evidence_ranges"] = [
            {"page": 1, "line_start": 150, "line_end": 151}
        ]
        state = (
            WindowedAttemptState.planned(
                parent_attempt_id="ATTEMPT_CROSS_PAGE",
                selection_id=self.selection.selection_id,
                document_version=self.selection.document_version,
                expected_workflow_revision="workflow-revision-1",
                policy_version=self.policy.fingerprint,
                prompt_version=self.policy.prompt_version,
                schema_version="window-candidates-5",
                window_plan_hash=self.plan.plan_hash,
                window_ids=tuple(
                    item.window_id for item in self.plan.windows
                ),
            )
            .start()
            .start_child(window.window_id, "ATTEMPT_CROSS_PAGE_CHILD")
        )

        with self.assertRaisesRegex(
            WindowCandidateError,
            (
                "Допустимые координаты текущего окна: .*1:.*-150; "
                "2:1-.*Нумерация строк начинается заново"
            ),
        ):
            self.service.validate(
                parent=state,
                plan=self.plan,
                window_id=window.window_id,
                child_attempt_id="ATTEMPT_CROSS_PAGE_CHILD",
                arguments=replace(arguments(), candidates=[proposed]),
            )

    def test_candidate_must_include_primary_owned_evidence(self) -> None:
        overlap_line = self.primary_end + 1
        overlap_range = {
            "page": 1,
            "line_start": overlap_line,
            "line_end": overlap_line,
        }
        proposed = candidate()
        proposed["condition_ranges"] = [overlap_range]
        proposed["changed_factor_ranges"] = [overlap_range]
        proposed["action_ranges"] = [overlap_range]
        proposed["consequences"] = [
            {
                "text": "Поведение только из overlap",
                "evidence_ranges": [overlap_range],
            }
        ]
        invalid = WindowCandidateArguments(
            outcome="candidates",
            explanation="Overlap-only candidate",
            candidates=[proposed],
            boundary_dependencies=[],
        )

        with self.assertRaisesRegex(
            WindowCandidateError,
            "хотя бы одной строки primary=true",
        ):
            self.service.validate(
                parent=self.state,
                plan=self.plan,
                window_id=self.window.window_id,
                child_attempt_id="ATTEMPT_CHILD_1",
                arguments=invalid,
            )

    def test_primary_owned_candidate_can_continue_into_overlap(self) -> None:
        primary_range = {
            "page": 1,
            "line_start": self.primary_end,
            "line_end": self.primary_end,
        }
        overlap_line = self.primary_end + 1
        overlap_range = {
            "page": 1,
            "line_start": overlap_line,
            "line_end": overlap_line,
        }
        proposed = candidate()
        proposed["condition_ranges"] = [primary_range]
        proposed["changed_factor_ranges"] = [overlap_range]
        proposed["action_ranges"] = [overlap_range]
        proposed["consequences"] = [
            {
                "text": "Последствие продолжается в overlap",
                "evidence_ranges": [overlap_range],
            }
        ]
        payload = WindowCandidateArguments(
            outcome="candidates",
            explanation="Primary-owned candidate пересекает границу",
            candidates=[proposed],
            boundary_dependencies=[],
        )

        result = self.service.validate(
            parent=self.state,
            plan=self.plan,
            window_id=self.window.window_id,
            child_attempt_id="ATTEMPT_CHILD_1",
            arguments=payload,
        )

        self.assertIn(
            SourcePosition(1, overlap_line),
            result.candidates[0].evidence_positions,
        )

    def test_primary_roles_are_derived_from_candidate_evidence(self) -> None:
        result = self.service.validate(
            parent=self.state,
            plan=self.plan,
            window_id=self.window.window_id,
            child_attempt_id="ATTEMPT_CHILD_1",
            arguments=arguments(),
        )

        roles = {
            item.position.line_number: item.role
            for item in result.primary_line_assessments
        }
        self.assertEqual(
            {line for line, role in roles.items() if role == "evidence"},
            {1, 2, 3},
        )
        self.assertEqual(
            result.primary_line_assessments[0].reason,
            "Строка включена в source ranges local candidate",
        )
        self.assertEqual(
            result.primary_line_assessments[3].reason,
            "Строка не включена в source ranges local candidate",
        )

    def test_new_parent_attempt_does_not_reuse_cancelled_attempt_result(self) -> None:
        self.service.accept(
            parent=self.state,
            plan=self.plan,
            window_id=self.window.window_id,
            child_attempt_id="ATTEMPT_CHILD_1",
            arguments=arguments(),
            raw_arguments=asdict(arguments()),
        )
        second = (
            WindowedAttemptState.planned(
                parent_attempt_id="ATTEMPT_PARENT_2",
                selection_id=self.selection.selection_id,
                document_version=self.selection.document_version,
                expected_workflow_revision="workflow-revision-1",
                policy_version=self.policy.fingerprint,
                prompt_version=self.policy.prompt_version,
                schema_version="window-candidates-5",
                window_plan_hash=self.plan.plan_hash,
                window_ids=tuple(
                    window.window_id for window in self.plan.windows
                ),
            )
            .start()
            .start_child(self.window.window_id, "ATTEMPT_CHILD_2")
        )

        self.service.accept(
            parent=second,
            plan=self.plan,
            window_id=self.window.window_id,
            child_attempt_id="ATTEMPT_CHILD_2",
            arguments=arguments(),
            raw_arguments=asdict(arguments()),
        )

        result_records = [
            record_id
            for kind, record_id in self.database.records
            if kind == "decomposition_window_result"
        ]
        self.assertEqual(len(result_records), 2)
        self.assertTrue(
            any(record_id.startswith("ATTEMPT_PARENT_2:") for record_id in result_records)
        )

    def test_tampered_validated_result_is_rejected_on_recovery(self) -> None:
        self.service.accept(
            parent=self.state,
            plan=self.plan,
            window_id=self.window.window_id,
            child_attempt_id="ATTEMPT_CHILD_1",
            arguments=arguments(),
            raw_arguments=asdict(arguments()),
        )
        record_id = f"ATTEMPT_PARENT:{self.window.window_id}"
        stored = self.database.records[
            ("decomposition_window_result", record_id)
        ]
        stored.payload["validated"]["outcome"] = "tampered"

        with self.assertRaisesRegex(WindowCandidateError, "fingerprint"):
            self.service.load(
                self.window.window_id,
                parent_attempt_id="ATTEMPT_PARENT",
                plan_hash=self.plan.plan_hash,
            )

    def test_boundary_dependency_is_typed_and_not_a_global_outcome(self) -> None:
        boundary = WindowCandidateArguments(
            outcome="boundary_dependency",
            explanation="Последствие находится после primary range",
            candidates=[],
            boundary_dependencies=[
                {
                    "local_dependency_id": "boundary-1",
                    "local_candidate_id": None,
                    "direction": "after",
                    "missing_field": "consequence",
                    "source_ranges": [
                        {
                            "page": 1,
                            "line_start": self.primary_end,
                            "line_end": self.primary_end,
                        }
                    ],
                    "reason": "Условие продолжается в следующем primary range",
                }
            ],
        )

        result = self.service.validate(
            parent=self.state,
            plan=self.plan,
            window_id=self.window.window_id,
            child_attempt_id="ATTEMPT_CHILD_1",
            arguments=boundary,
        )

        self.assertEqual(result.outcome, "boundary_dependency")
        self.assertEqual(result.boundary_dependencies[0].direction, "after")
        self.assertEqual(
            result.boundary_dependencies[0].source[0]["quote"],
            f"Строка источника {self.primary_end}",
        )

    def test_no_local_outcome_with_dependency_reports_required_outcome(self) -> None:
        invalid = WindowCandidateArguments(
            outcome="no_local_testable_behavior",
            explanation="Поведение продолжается за границей",
            candidates=[],
            boundary_dependencies=[
                {
                    "local_dependency_id": "boundary-1",
                    "local_candidate_id": None,
                    "direction": "after",
                    "missing_field": "condition",
                    "source_ranges": [
                        {
                            "page": 1,
                            "line_start": self.primary_end,
                            "line_end": self.primary_end,
                        }
                    ],
                    "reason": "Условие продолжается в следующем primary range",
                }
            ],
        )

        with self.assertRaisesRegex(
            WindowCandidateError,
            "outcome=boundary_dependency",
        ):
            self.service.validate(
                parent=self.state,
                plan=self.plan,
                window_id=self.window.window_id,
                child_attempt_id="ATTEMPT_CHILD_1",
                arguments=invalid,
            )

    def test_boundary_dependency_must_touch_declared_window_edge(self) -> None:
        boundary = WindowCandidateArguments(
            outcome="boundary_dependency",
            explanation="Ошибочная ссылка на середину",
            candidates=[],
            boundary_dependencies=[
                {
                    "local_dependency_id": "boundary-1",
                    "local_candidate_id": None,
                    "direction": "after",
                    "missing_field": "consequence",
                    "source_ranges": [
                        {
                            "page": 1,
                            "line_start": self.primary_end // 2,
                            "line_end": self.primary_end // 2,
                        }
                    ],
                    "reason": "Не достигает правой границы primary",
                }
            ],
        )

        with self.assertRaisesRegex(WindowCandidateError, "границы окна"):
            self.service.validate(
                parent=self.state,
                plan=self.plan,
                window_id=self.window.window_id,
                child_attempt_id="ATTEMPT_CHILD_1",
                arguments=boundary,
            )

    def test_raw_arguments_are_bound_to_exact_typed_result(self) -> None:
        with self.assertRaisesRegex(WindowCandidateError, "Raw window"):
            self.service.accept(
                parent=self.state,
                plan=self.plan,
                window_id=self.window.window_id,
                child_attempt_id="ATTEMPT_CHILD_1",
                arguments=arguments(),
                raw_arguments={"outcome": "different"},
            )

    def test_null_input_without_blocking_gap_is_rejected(self) -> None:
        invalid = candidate()
        invalid["gaps"] = []
        payload = replace(arguments(), candidates=[invalid])

        with self.assertRaisesRegex(
            WindowCandidateError,
            "Отсутствующее входное значение требует блокирующий пробел",
        ):
            self.service.validate(
                parent=self.state,
                plan=self.plan,
                window_id=self.window.window_id,
                child_attempt_id="ATTEMPT_CHILD_1",
                arguments=payload,
            )


class WindowCandidateFlowTests(unittest.IsolatedAsyncioTestCase):
    async def test_child_prompt_is_independent_and_persists_raw_diagnostic(self) -> None:
        document = source_document()
        selection = saved_selection(document)
        policy = default_policy()
        windowing = default_windowing_policy(policy)
        plan = WindowPlanner(document, windowing).build(selection)
        window = plan.windows[0]
        state = (
            WindowedAttemptState.planned(
                parent_attempt_id="ATTEMPT_PARENT",
                selection_id=selection.selection_id,
                document_version=selection.document_version,
                expected_workflow_revision="workflow-revision-1",
                policy_version=windowing.fingerprint,
                prompt_version=windowing.prompt_version,
                schema_version="window-candidates-5",
                window_plan_hash=plan.plan_hash,
                window_ids=tuple(item.window_id for item in plan.windows),
            )
            .start()
            .start_child(window.window_id, "ATTEMPT_CHILD_1")
        )
        response_arguments = arguments()
        invalid_arguments = asdict(response_arguments)
        invalid_arguments["candidates"][0]["condition_ranges"] = [
            {"page": 2, "line_start": 31, "line_end": 31}
        ]
        transport = ScriptedLlmTransport(
            [
                RawCompletion(
                    finish_reason="tool_calls",
                    tool_calls=(
                        {
                            "id": "call-window-invalid",
                            "name": "submit_window_candidates",
                            "arguments": invalid_arguments,
                        },
                    ),
                    usage={"prompt_tokens": 100, "completion_tokens": 50},
                    model="scripted",
                ),
                RawCompletion(
                    finish_reason="tool_calls",
                    tool_calls=(
                        {
                            "id": "call-window-1",
                            "name": "submit_window_candidates",
                            "arguments": asdict(response_arguments),
                        },
                    ),
                    usage={"prompt_tokens": 100, "completion_tokens": 50},
                    model="scripted",
                )
            ]
        )
        database = InMemoryDatabase()
        registry = TypedToolRegistry()
        registry.register(window_candidates_tool())
        runtime = LlmToolRuntime(
            transport=transport,
            tools=registry,
            uow_factory=lambda: InMemoryUnitOfWork(database),
        )
        flow = WindowCandidateFlow(
            policy=policy,
            runtime=runtime,
            service=WindowCandidateService(
                document=document,
                uow_factory=lambda: InMemoryUnitOfWork(database),
            ),
        )

        result = await flow.run(
            parent=state,
            plan=plan,
            window_id=window.window_id,
            child_attempt_id="ATTEMPT_CHILD_1",
            session_id=selection.selection_id,
        )

        self.assertEqual(len(result.candidates), 1)
        self.assertEqual(len(transport.calls), 2)
        call = transport.calls[1]["call"]
        self.assertEqual(set(call.context), {"window"})
        self.assertNotIn("candidates", call.context["window"])
        self.assertIn(
            "вне окна",
            call.system_prompt,
        )
        self.assertIn(
            "Допустимые координаты текущего окна",
            call.system_prompt,
        )
        self.assertIn(
            "Нумерация строк начинается заново",
            call.system_prompt,
        )
        diagnostic = database.records[("llm_diagnostic", "ATTEMPT_CHILD_1")]
        self.assertEqual(
            diagnostic.payload["tool_calls"][0]["arguments"],
            asdict(response_arguments),
        )
        self.assertEqual(len(diagnostic.payload["rejected_tool_calls"]), 1)
        self.assertEqual(
            diagnostic.payload["rejected_tool_calls"][0]["arguments"],
            invalid_arguments,
        )
        self.assertIn(
            (
                "decomposition_window_result",
                f"ATTEMPT_PARENT:{window.window_id}",
            ),
            database.records,
        )

if __name__ == "__main__":
    unittest.main()
