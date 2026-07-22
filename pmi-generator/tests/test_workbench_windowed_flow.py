from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path

from pmi_generator.workbench.application.decomposition import (
    WindowedDecompositionFlow,
    WindowedDecompositionStore,
    default_windowing_policy,
    reconciliation_case_tool,
    semantic_synthesis_tool,
    semantic_window_tool,
)
from pmi_generator.workbench.application.llm import (
    AttemptDiscardedError,
    LlmToolRuntime,
    RawCompletion,
    TechnicalLlmError,
    ToolContractError,
    TypedToolRegistry,
)
from pmi_generator.workbench.application.metrics import collect_metrics
from pmi_generator.workbench.application.facade import WorkbenchApplication
from pmi_generator.workbench.application.prompting import (
    PromptId,
    default_policy,
)
from pmi_generator.workbench.application.source import SavedSelection
from pmi_generator.workbench.application.session import SessionService
from pmi_generator.workbench.application.workflow import (
    CommandKind,
    WorkflowCommand,
    WorkflowState,
    apply_command,
)
from pmi_generator.workbench.domain.source import (
    SourceDocument,
    SourcePage,
    SourceSection,
)
from pmi_generator.workbench.infrastructure.mock_mode import MockLlmTransport
from pmi_generator.workbench.infrastructure.workers import ProductionPromptWorkers
from pmi_generator.workbench.infrastructure.storage import (
    InMemoryDatabase,
    InMemoryUnitOfWork,
    SqliteUnitOfWork,
)


class RecordingWorkflow:
    def __init__(self) -> None:
        self.states: dict[str, WorkflowState] = {}
        self.commands: list[WorkflowCommand] = []

    def execute(
        self,
        thread_id: str,
        command: WorkflowCommand,
    ) -> WorkflowState:
        state = apply_command(self.current_state(thread_id), command)
        self.states[thread_id] = state
        self.commands.append(command)
        return state

    def current_state(self, thread_id: str) -> WorkflowState:
        return self.states.get(thread_id, WorkflowState.empty())


def document() -> SourceDocument:
    return SourceDocument(
        pages=(
            SourcePage(
                1,
                "1",
                tuple(f"Требование строки {line}" for line in range(1, 151)),
            ),
            SourcePage(
                2,
                "2",
                tuple(f"Требование строки {line}" for line in range(151, 301)),
            ),
        ),
        sections=(
            SourceSection("root", "1", "Большой раздел", ("1",), (1, 2)),
        ),
    )


def selection(source: SourceDocument) -> SavedSelection:
    return SavedSelection(
        "SELECTION_LARGE",
        "root",
        source.select(source.positions[0], source.positions[-1]),
        source.metadata.document_version,
        "root",
    )


class BlockingMockTransport(MockLlmTransport):
    def __init__(self, *, pause_on_call: int = 1) -> None:
        super().__init__(delay=0)
        self.pause_on_call = pause_on_call
        self.call_count = 0
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def complete(self, call, tools):  # type: ignore[no-untyped-def]
        self.call_count += 1
        if self.call_count == self.pause_on_call:
            self.started.set()
            await self.release.wait()
        return await super().complete(call, tools)


class FailingSecondWindowTransport(MockLlmTransport):
    def __init__(self) -> None:
        super().__init__(delay=0)
        self.window_calls = 0

    async def complete(self, call, tools):  # type: ignore[no-untyped-def]
        if call.prompt_id is PromptId.DECOMPOSITION_WINDOW_SEMANTIC:
            self.window_calls += 1
            if self.window_calls == 2:
                raise TechnicalLlmError(
                    "Второе окно недоступно",
                    retryable=False,
                )
        return await super().complete(call, tools)


class LengthThenExtendedWindowTransport(MockLlmTransport):
    def __init__(self) -> None:
        super().__init__(delay=0)
        self.semantic_calls = 0

    async def complete(self, call, tools):  # type: ignore[no-untyped-def]
        if call.prompt_id is PromptId.DECOMPOSITION_WINDOW_SEMANTIC:
            self.semantic_calls += 1
            if self.semantic_calls == 1:
                self.calls.append(call)
                return RawCompletion(
                    finish_reason="length",
                    tool_calls=(),
                    usage={
                        "prompt_tokens": 9450,
                        "completion_tokens": 8192,
                    },
                    model="mock-length",
                )
        return await super().complete(call, tools)


class DoubleLengthThenSplitWindowTransport(MockLlmTransport):
    def __init__(self) -> None:
        super().__init__(delay=0)
        self.semantic_calls = 0

    async def complete(self, call, tools):  # type: ignore[no-untyped-def]
        if call.prompt_id is PromptId.DECOMPOSITION_WINDOW_SEMANTIC:
            self.semantic_calls += 1
            if self.semantic_calls <= 2:
                self.calls.append(call)
                return RawCompletion(
                    finish_reason="length",
                    tool_calls=(),
                    usage={
                        "prompt_tokens": 9450,
                        "completion_tokens": call.generation_parameters[
                            "max_tokens"
                        ],
                    },
                    model="mock-double-length",
                )
        return await super().complete(call, tools)


class RecursiveSplitWindowTransport(MockLlmTransport):
    def __init__(self, *, always_length: bool = False) -> None:
        super().__init__(delay=0)
        self.semantic_calls = 0
        self.always_length = always_length

    async def complete(self, call, tools):  # type: ignore[no-untyped-def]
        if call.prompt_id is PromptId.DECOMPOSITION_WINDOW_SEMANTIC:
            self.semantic_calls += 1
            primary_count = sum(
                bool(line["primary"])
                for line in call.context["window"]["lines"]
            )
            should_truncate = (
                self.always_length
                or self.semantic_calls <= 2
                or primary_count == 12
            )
            if should_truncate:
                self.calls.append(call)
                return RawCompletion(
                    finish_reason="length",
                    tool_calls=(),
                    usage={
                        "prompt_tokens": 9450,
                        "completion_tokens": call.generation_parameters[
                            "max_tokens"
                        ],
                    },
                    model="mock-recursive-length",
                )
        return await super().complete(call, tools)


class BlockingSplitWindowTransport(DoubleLengthThenSplitWindowTransport):
    def __init__(self, *, block_semantic_call: int) -> None:
        super().__init__()
        self.block_semantic_call = block_semantic_call
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def complete(self, call, tools):  # type: ignore[no-untyped-def]
        next_semantic_call = self.semantic_calls + 1
        if (
            call.prompt_id is PromptId.DECOMPOSITION_WINDOW_SEMANTIC
            and next_semantic_call == self.block_semantic_call
        ):
            self.started.set()
            await self.release.wait()
        return await super().complete(call, tools)


class FailingSplitWindowTransport(DoubleLengthThenSplitWindowTransport):
    async def complete(self, call, tools):  # type: ignore[no-untyped-def]
        if (
            call.prompt_id is PromptId.DECOMPOSITION_WINDOW_SEMANTIC
            and self.semantic_calls == 2
        ):
            raise TechnicalLlmError(
                "Semantic subwindow недоступно",
                retryable=False,
            )
        return await super().complete(call, tools)


class BoundaryDuplicateSplitTransport(
    DoubleLengthThenSplitWindowTransport
):
    def __init__(self) -> None:
        super().__init__()
        self.boundary_line_ids: tuple[str, str] | None = None

    async def complete(self, call, tools):  # type: ignore[no-untyped-def]
        if (
            call.prompt_id is PromptId.DECOMPOSITION_WINDOW_SEMANTIC
            and self.semantic_calls == 0
        ):
            primary_ids = tuple(
                str(line["line_id"])
                for line in call.context["window"]["lines"]
                if line["primary"]
            )
            self.boundary_line_ids = (
                primary_ids[11],
                primary_ids[12],
            )
        response = await super().complete(call, tools)
        if (
            call.prompt_id is not PromptId.DECOMPOSITION_WINDOW_SEMANTIC
            or self.semantic_calls not in {3, 4}
            or not response.tool_calls
        ):
            return response
        assert self.boundary_line_ids is not None
        arguments = dict(response.tool_calls[0]["arguments"])
        behavior = dict(arguments["behaviors"][0])
        behavior.update(
            {
                "title": "Поведение на границе частей",
                "summary": "Одно поведение использует обе стороны границы.",
                "facts": [
                    {
                        "text": "Связанное условие продолжается через границу.",
                        "line_ids": list(self.boundary_line_ids),
                    }
                ],
            }
        )
        arguments["behaviors"] = [behavior]
        return type(response)(
            response.finish_reason,
            (
                {
                    **response.tool_calls[0],
                    "arguments": arguments,
                },
            ),
            response.usage,
            response.model,
            response.response_preview,
        )


class ContextOnlySplitTransport(DoubleLengthThenSplitWindowTransport):
    async def complete(self, call, tools):  # type: ignore[no-untyped-def]
        response = await super().complete(call, tools)
        if (
            call.prompt_id is not PromptId.DECOMPOSITION_WINDOW_SEMANTIC
            or self.semantic_calls not in {3, 4}
            or not response.tool_calls
        ):
            return response
        context_line = next(
            line
            for line in call.context["window"]["lines"]
            if not line["primary"]
        )
        arguments = dict(response.tool_calls[0]["arguments"])
        arguments["behaviors"] = [
            *arguments["behaviors"],
            {
                "title": f"Context-only behavior {self.semantic_calls}",
                "summary": "Поведение принадлежит соседней части.",
                "facts": [
                    {
                        "text": str(context_line["text"]),
                        "line_ids": [str(context_line["line_id"])],
                    }
                ],
            },
        ]
        return type(response)(
            response.finish_reason,
            (
                {
                    **response.tool_calls[0],
                    "arguments": arguments,
                },
            ),
            response.usage,
            response.model,
            response.response_preview,
        )


class ReconciliationBlockingTransport(MockLlmTransport):
    def __init__(self) -> None:
        super().__init__(delay=0)
        self.window_calls = 0
        self.reconciliation_started = asyncio.Event()
        self.release = asyncio.Event()

    async def complete(self, call, tools):  # type: ignore[no-untyped-def]
        if call.prompt_id is PromptId.DECOMPOSITION_RECONCILIATION:
            self.reconciliation_started.set()
            await self.release.wait()
        response = await super().complete(call, tools)
        if call.prompt_id is not PromptId.DECOMPOSITION_WINDOW_SEMANTIC:
            return response
        self.window_calls += 1
        if self.window_calls != 2:
            return response
        arguments = dict(response.tool_calls[0]["arguments"])
        behavior = dict(arguments["behaviors"][0])
        lines = call.context["window"]["lines"]
        first_primary = next(
            index
            for index, item in enumerate(lines)
            if item["primary"]
        )
        shared_id = str(lines[first_primary - 1]["line_id"])
        primary_id = str(lines[first_primary]["line_id"])
        behavior["facts"] = [
            {
                **fact,
                "line_ids": [shared_id, primary_id],
            }
            for fact in behavior["facts"]
        ]
        arguments["behaviors"] = [behavior]
        return type(response)(
            response.finish_reason,
            (
                {
                    **response.tool_calls[0],
                    "arguments": arguments,
                },
            ),
            response.usage,
            response.model,
            response.response_preview,
        )


class WindowedDecompositionFlowTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.document = document()
        self.selection = selection(self.document)
        self.database = InMemoryDatabase()
        self.policy = default_policy()
        self.windowing = default_windowing_policy(self.policy)
        self.ids: dict[str, int] = {}

    def next_id(self, prefix: str) -> str:
        self.ids[prefix] = self.ids.get(prefix, 0) + 1
        return f"{prefix}_{self.ids[prefix]:04d}"

    def runtime(self, transport: MockLlmTransport) -> LlmToolRuntime:
        tools = TypedToolRegistry()
        tools.register(semantic_window_tool())
        tools.register(semantic_synthesis_tool())
        tools.register(reconciliation_case_tool())
        return LlmToolRuntime(
            transport=transport,
            tools=tools,
            uow_factory=lambda: InMemoryUnitOfWork(self.database),
        )

    def flow(
        self,
        transport: MockLlmTransport,
    ) -> WindowedDecompositionFlow:
        return WindowedDecompositionFlow(
            document=self.document,
            policy=self.policy,
            windowing_policy=self.windowing,
            runtime=self.runtime(transport),
            uow_factory=lambda: InMemoryUnitOfWork(self.database),
            next_id=self.next_id,
        )

    async def test_windowed_route_runs_children_and_one_atomic_assembly(
        self,
    ) -> None:
        transport = MockLlmTransport(delay=0)
        flow = self.flow(transport)

        result = await flow.run(
            parent_attempt_id="ATTEMPT_PARENT",
            session_id=self.selection.selection_id,
            selection=self.selection,
            expected_workflow_revision="workflow-revision-1",
        )

        self.assertEqual(result.outcome, "skeletons_created")
        parent, plan = WindowedDecompositionStore(
            lambda: InMemoryUnitOfWork(self.database)
        ).load("ATTEMPT_PARENT", self.selection)
        self.assertEqual(
            [call.prompt_id for call in transport.calls],
            (
                [PromptId.DECOMPOSITION_WINDOW_SEMANTIC]
                * len(plan.windows)
                + [PromptId.DECOMPOSITION_SEMANTIC_SYNTHESIS]
                * len(plan.windows)
            ),
        )
        progress = flow.progress()
        self.assertEqual(progress.completed_windows, len(plan.windows))
        self.assertEqual(progress.total_windows, len(plan.windows))
        self.assertEqual(progress.stage, "завершено")
        self.assertIn(
            ("decomposition", self.selection.selection_id),
            self.database.records,
        )
        self.assertEqual(parent.status.value, "completed")

    async def test_length_window_retries_once_with_extended_profile(
        self,
    ) -> None:
        transport = LengthThenExtendedWindowTransport()
        flow = self.flow(transport)

        result = await flow.run(
            parent_attempt_id="ATTEMPT_PARENT_LENGTH",
            session_id=self.selection.selection_id,
            selection=self.selection,
            expected_workflow_revision="workflow-revision-length",
        )

        self.assertEqual(result.outcome, "skeletons_created")
        semantic_calls = [
            call
            for call in transport.calls
            if call.prompt_id
            is PromptId.DECOMPOSITION_WINDOW_SEMANTIC
        ]
        self.assertGreaterEqual(len(semantic_calls), 2)
        self.assertEqual(
            [
                semantic_calls[0].generation_parameters["max_tokens"],
                semantic_calls[1].generation_parameters["max_tokens"],
            ],
            [8192, 12_288],
        )
        extended_diagnostics = [
            record.payload
            for (kind, _record_id), record in self.database.records.items()
            if kind == "llm_diagnostic"
            and len(record.payload.get("invocations", [])) == 2
        ]
        self.assertEqual(len(extended_diagnostics), 1)
        self.assertEqual(
            [
                item["finish_reason"]
                for item in extended_diagnostics[0]["invocations"]
            ],
            ["length", "tool_calls"],
        )

    async def test_second_length_splits_logical_window_and_assembles_once(
        self,
    ) -> None:
        transport = DoubleLengthThenSplitWindowTransport()
        flow = self.flow(transport)

        result = await flow.run(
            parent_attempt_id="ATTEMPT_PARENT_SPLIT",
            session_id=self.selection.selection_id,
            selection=self.selection,
            expected_workflow_revision="workflow-revision-split",
        )

        self.assertEqual(result.outcome, "skeletons_created")
        semantic_calls = [
            call
            for call in transport.calls
            if call.prompt_id
            is PromptId.DECOMPOSITION_WINDOW_SEMANTIC
        ]
        self.assertGreaterEqual(len(semantic_calls), 4)
        self.assertEqual(
            [
                semantic_calls[0].generation_parameters["max_tokens"],
                semantic_calls[1].generation_parameters["max_tokens"],
            ],
            [8192, 12_288],
        )
        original_context = semantic_calls[0].context["window"]["lines"]
        split_contexts = [
            call.context["window"]["lines"]
            for call in semantic_calls[2:4]
        ]
        self.assertTrue(
            all(len(item) == len(original_context) for item in split_contexts)
        )
        self.assertEqual(
            [
                sum(bool(line["primary"]) for line in item)
                for item in split_contexts
            ],
            [12, 13],
        )
        self.assertEqual(
            [
                (line["line_id"], line["text"])
                for line in split_contexts[0]
            ],
            [
                (line["line_id"], line["text"])
                for line in original_context
            ],
        )
        self.assertEqual(
            len(
                [
                    key
                    for key in self.database.records
                    if key[0] == "decomposition"
                ]
            ),
            1,
        )
        metrics = collect_metrics(
            lambda: InMemoryUnitOfWork(self.database)
        )
        semantic_invocations = sum(
            len(record.payload.get("invocations", []))
            for (kind, _record_id), record in self.database.records.items()
            if kind == "llm_diagnostic"
            and record.payload.get("prompt_id")
            == PromptId.DECOMPOSITION_WINDOW_SEMANTIC.value
        )
        self.assertGreaterEqual(semantic_invocations, 4)
        self.assertEqual(
            metrics["llm_finish_reason_length"],
            2,
        )

    async def test_recursive_split_uses_midpoint_and_remains_bounded(
        self,
    ) -> None:
        transport = RecursiveSplitWindowTransport()

        result = await self.flow(transport).run(
            parent_attempt_id="ATTEMPT_PARENT_RECURSIVE_SPLIT",
            session_id=self.selection.selection_id,
            selection=self.selection,
            expected_workflow_revision="workflow-revision-recursive",
        )

        self.assertEqual(result.outcome, "skeletons_created")
        semantic_calls = [
            call
            for call in transport.calls
            if call.prompt_id
            is PromptId.DECOMPOSITION_WINDOW_SEMANTIC
        ]
        self.assertEqual(
            [
                sum(
                    bool(line["primary"])
                    for line in call.context["window"]["lines"]
                )
                for call in semantic_calls[:7]
            ],
            [25, 25, 12, 12, 6, 6, 13],
        )
        self.assertTrue(
            all(
                len(call.context["window"]["lines"])
                == len(semantic_calls[0].context["window"]["lines"])
                for call in semantic_calls[:7]
            )
        )

    async def test_boundary_behavior_and_duplicate_evidence_are_preserved(
        self,
    ) -> None:
        result = await self.flow(
            BoundaryDuplicateSplitTransport()
        ).run(
            parent_attempt_id="ATTEMPT_PARENT_SPLIT_BOUNDARY",
            session_id=self.selection.selection_id,
            selection=self.selection,
            expected_workflow_revision="workflow-revision-boundary",
        )

        self.assertEqual(result.outcome, "skeletons_created")
        fact_records = [
            record
            for (kind, _record_id), record in self.database.records.items()
            if kind == "decomposition_window_semantic_facts"
        ]
        boundary_fragments = [
            fragment
            for record in fact_records
            for fragment in record.payload["validated"]["fragments"]
            if fragment["title"] == "Поведение на границе частей"
        ]
        self.assertEqual(len(boundary_fragments), 2)
        self.assertEqual(
            boundary_fragments[0]["facts"][0]["positions"],
            boundary_fragments[1]["facts"][0]["positions"],
        )

    async def test_context_only_subwindow_behaviors_remain_raw_only(
        self,
    ) -> None:
        result = await self.flow(ContextOnlySplitTransport()).run(
            parent_attempt_id="ATTEMPT_PARENT_SPLIT_CONTEXT",
            session_id=self.selection.selection_id,
            selection=self.selection,
            expected_workflow_revision="workflow-revision-context",
        )

        self.assertEqual(result.outcome, "skeletons_created")
        subresults = [
            record.payload
            for (kind, _record_id), record in self.database.records.items()
            if kind == "decomposition_semantic_subwindow_result"
        ]
        self.assertEqual(len(subresults), 2)
        self.assertTrue(
            all(
                payload["context_only_behaviors"] == 1
                for payload in subresults
            )
        )
        self.assertTrue(
            all(
                any(
                    str(behavior["title"]).startswith(
                        "Context-only behavior"
                    )
                    for behavior in payload["raw_arguments"]["behaviors"]
                )
                for payload in subresults
            )
        )
        self.assertTrue(
            all(
                not any(
                    str(behavior["title"]).startswith(
                        "Context-only behavior"
                    )
                    for behavior in payload["owned_arguments"]["behaviors"]
                )
                for payload in subresults
            )
        )
        logical_results = [
            record.payload
            for (kind, _record_id), record in self.database.records.items()
            if kind == "decomposition_window_semantic_facts"
            and record.payload["raw_arguments"]["behaviors"]
        ]
        self.assertTrue(
            all(
                not any(
                    str(behavior["title"]).startswith(
                        "Context-only behavior"
                    )
                    for behavior in payload["raw_arguments"]["behaviors"]
                )
                for payload in logical_results
            )
        )

    async def test_split_limit_fails_closed_without_decomposition(
        self,
    ) -> None:
        transport = RecursiveSplitWindowTransport(always_length=True)

        with self.assertRaisesRegex(
            ValueError,
            "технического предела",
        ) as raised:
            await self.flow(transport).run(
                parent_attempt_id="ATTEMPT_PARENT_SPLIT_LIMIT",
                session_id=self.selection.selection_id,
                selection=self.selection,
                expected_workflow_revision="workflow-revision-limit",
            )
        self.assertNotIn("SEMANTIC_SUB", str(raised.exception))

        self.assertFalse(
            any(
                kind == "decomposition"
                for kind, _record_id in self.database.records
            )
        )
        semantic_calls = [
            call
            for call in transport.calls
            if call.prompt_id
            is PromptId.DECOMPOSITION_WINDOW_SEMANTIC
        ]
        self.assertEqual(
            [
                call.generation_parameters["max_tokens"]
                for call in semantic_calls
            ],
            [8192, 12_288, 8192, 12_288, 8192, 12_288],
        )

    async def test_failed_subwindow_rejects_entire_parent(
        self,
    ) -> None:
        with self.assertRaisesRegex(
            TechnicalLlmError,
            "subwindow недоступно",
        ):
            await self.flow(FailingSplitWindowTransport()).run(
                parent_attempt_id="ATTEMPT_PARENT_SUBWINDOW_FAILURE",
                session_id=self.selection.selection_id,
                selection=self.selection,
                expected_workflow_revision="workflow-revision-failure",
            )

        self.assertFalse(
            any(
                kind == "decomposition"
                for kind, _record_id in self.database.records
            )
        )
        coordinator = next(
            record
            for (kind, _record_id), record in self.database.records.items()
            if kind == "decomposition_semantic_subwindow_attempt"
        )
        self.assertEqual(
            coordinator.payload["state"]["status"],
            "failed",
        )

    async def test_restart_reuses_validated_subresult_only(
        self,
    ) -> None:
        interrupted_transport = BlockingSplitWindowTransport(
            block_semantic_call=4
        )
        interrupted_flow = self.flow(interrupted_transport)
        task = asyncio.create_task(
            interrupted_flow.run(
                parent_attempt_id="ATTEMPT_PARENT_SPLIT_RECOVERY",
                session_id=self.selection.selection_id,
                selection=self.selection,
                expected_workflow_revision="workflow-revision-recovery",
            )
        )
        await interrupted_transport.started.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task

        recovery_transport = MockLlmTransport(delay=0)
        result = await self.flow(recovery_transport).run(
            parent_attempt_id="ATTEMPT_PARENT_SPLIT_RECOVERY",
            session_id=self.selection.selection_id,
            selection=self.selection,
            expected_workflow_revision="workflow-revision-recovery",
        )

        self.assertEqual(result.outcome, "skeletons_created")
        first_recovery_call = next(
            call
            for call in recovery_transport.calls
            if call.prompt_id
            is PromptId.DECOMPOSITION_WINDOW_SEMANTIC
        )
        self.assertEqual(
            sum(
                bool(line["primary"])
                for line in first_recovery_call.context["window"]["lines"]
            ),
            13,
        )

    async def test_explicit_cancel_during_split_cancels_coordinator(
        self,
    ) -> None:
        transport = BlockingSplitWindowTransport(
            block_semantic_call=3
        )
        flow = self.flow(transport)
        task = asyncio.create_task(
            flow.run(
                parent_attempt_id="ATTEMPT_PARENT_SPLIT_CANCEL",
                session_id=self.selection.selection_id,
                selection=self.selection,
                expected_workflow_revision="workflow-revision-cancel",
            )
        )
        await transport.started.wait()
        flow.cancel("ATTEMPT_PARENT_SPLIT_CANCEL")
        transport.release.set()

        with self.assertRaises(AttemptDiscardedError):
            await task

        coordinator_records = [
            record
            for (kind, _record_id), record in self.database.records.items()
            if kind == "decomposition_semantic_subwindow_attempt"
        ]
        self.assertEqual(len(coordinator_records), 1)
        self.assertEqual(
            coordinator_records[0].payload["state"]["status"],
            "cancelled",
        )
        self.assertFalse(
            any(
                kind == "decomposition"
                for kind, _record_id in self.database.records
            )
        )

    async def test_sqlite_restart_recovers_same_split_plan(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            database_path = Path(temporary) / "workbench.sqlite3"
            uow_factory = lambda: SqliteUnitOfWork(database_path)

            def build_flow(
                transport: MockLlmTransport,
            ) -> WindowedDecompositionFlow:
                tools = TypedToolRegistry()
                tools.register(semantic_window_tool())
                tools.register(semantic_synthesis_tool())
                tools.register(reconciliation_case_tool())
                runtime = LlmToolRuntime(
                    transport=transport,
                    tools=tools,
                    uow_factory=uow_factory,
                )
                return WindowedDecompositionFlow(
                    document=self.document,
                    policy=self.policy,
                    windowing_policy=self.windowing,
                    runtime=runtime,
                    uow_factory=uow_factory,
                    next_id=self.next_id,
                )

            interrupted_transport = BlockingSplitWindowTransport(
                block_semantic_call=4
            )
            task = asyncio.create_task(
                build_flow(interrupted_transport).run(
                    parent_attempt_id="ATTEMPT_PARENT_SQLITE_SPLIT",
                    session_id=self.selection.selection_id,
                    selection=self.selection,
                    expected_workflow_revision="workflow-revision-sqlite",
                )
            )
            await interrupted_transport.started.wait()
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task

            with uow_factory() as uow:
                before = uow.records.list_kind(
                    "decomposition_semantic_subwindow_attempt"
                )
            self.assertEqual(len(before), 1)
            split_plan_hash = before[0].payload["plan"]["plan_hash"]

            recovery_transport = MockLlmTransport(delay=0)
            result = await build_flow(recovery_transport).run(
                parent_attempt_id="ATTEMPT_PARENT_SQLITE_SPLIT",
                session_id=self.selection.selection_id,
                selection=self.selection,
                expected_workflow_revision="workflow-revision-sqlite",
            )

            self.assertEqual(result.outcome, "skeletons_created")
            with uow_factory() as uow:
                after = uow.records.list_kind(
                    "decomposition_semantic_subwindow_attempt"
                )
                subresults = uow.records.list_kind(
                    "decomposition_semantic_subwindow_result"
                )
            self.assertEqual(
                after[0].payload["plan"]["plan_hash"],
                split_plan_hash,
            )
            self.assertEqual(after[0].payload["state"]["status"], "completed")
            self.assertEqual(len(subresults), 2)

    async def test_windowed_route_has_no_legacy_tool_fallback(self) -> None:
        runtime = self.runtime(MockLlmTransport(delay=0))

        schemas = runtime.tools.openai_schemas(
            ("submit_semantic_window_result",),
        )

        self.assertEqual(
            schemas[0]["function"]["name"],
            "submit_semantic_window_result",
        )
        with self.assertRaisesRegex(
            ToolContractError,
            "Неизвестный tool submit_window_candidates",
        ):
            runtime.tools.openai_schemas(("submit_window_candidates",))
        with self.assertRaisesRegex(
            ToolContractError,
            "Неизвестный tool submit_reconciliation",
        ):
            runtime.tools.openai_schemas(("submit_reconciliation",))

    async def test_facade_routes_large_selection_without_manual_windows(
        self,
    ) -> None:
        transport = MockLlmTransport(delay=0)
        runtime = self.runtime(transport)
        sessions = SessionService(
            uow_factory=lambda: InMemoryUnitOfWork(self.database)
        )
        workflow = RecordingWorkflow()
        facade = WorkbenchApplication(
            document=self.document,
            run_dir=Path("."),
            uow_factory=lambda: InMemoryUnitOfWork(self.database),
            workflow=workflow,
            sessions=sessions,
            workers=ProductionPromptWorkers(
                document=self.document,
                uow_factory=lambda: InMemoryUnitOfWork(self.database),
                policy=self.policy,
                runtime_factory=lambda: runtime,
                retrieval_factory=lambda _selection: None,  # type: ignore[arg-type,return-value]
                sessions=sessions,
                next_id=self.next_id,
            ),
            next_id=self.next_id,
        )

        saved = facade.save_selection(
            "root",
            self.selection.selection,
        )
        operation = facade.decompose(saved)
        result = await operation.awaitable

        self.assertEqual(result.outcome, "skeletons_created")
        self.assertEqual(
            [command.kind for command in workflow.commands],
            [
                CommandKind.CONFIRM_SELECTION,
                CommandKind.BEGIN_ATTEMPT,
                CommandKind.APPLY_DECOMPOSITION,
            ],
        )
        self.assertEqual(operation.progress().stage, "завершено")

    async def test_failed_child_never_creates_partial_decomposition(self) -> None:
        flow = self.flow(FailingSecondWindowTransport())

        with self.assertRaisesRegex(
            TechnicalLlmError,
            "Второе окно недоступно",
        ):
            await flow.run(
                parent_attempt_id="ATTEMPT_PARENT",
                session_id=self.selection.selection_id,
                selection=self.selection,
                expected_workflow_revision="workflow-revision-1",
            )

        self.assertNotIn(
            ("decomposition", self.selection.selection_id),
            self.database.records,
        )
        parent, _plan = WindowedDecompositionStore(
            lambda: InMemoryUnitOfWork(self.database)
        ).load("ATTEMPT_PARENT", self.selection)
        self.assertEqual(parent.status.value, "failed")
        self.assertEqual(parent.children[1].status.value, "failed")

    async def test_cancel_during_reconciliation_discards_late_result(
        self,
    ) -> None:
        transport = ReconciliationBlockingTransport()
        flow = self.flow(transport)
        task = asyncio.create_task(
            flow.run(
                parent_attempt_id="ATTEMPT_PARENT",
                session_id=self.selection.selection_id,
                selection=self.selection,
                expected_workflow_revision="workflow-revision-1",
            )
        )
        await transport.reconciliation_started.wait()

        flow.cancel("ATTEMPT_PARENT")
        transport.release.set()

        with self.assertRaises(AttemptDiscardedError):
            await task
        self.assertNotIn(
            ("decomposition", self.selection.selection_id),
            self.database.records,
        )

    async def test_completed_windowed_parent_recovers_root_checkpoint(
        self,
    ) -> None:
        transport = MockLlmTransport(delay=0)
        runtime = self.runtime(transport)
        sessions = SessionService(
            uow_factory=lambda: InMemoryUnitOfWork(self.database)
        )
        workflow = RecordingWorkflow()
        workers = ProductionPromptWorkers(
            document=self.document,
            uow_factory=lambda: InMemoryUnitOfWork(self.database),
            policy=self.policy,
            runtime_factory=lambda: runtime,
            retrieval_factory=lambda _selection: None,  # type: ignore[arg-type,return-value]
            sessions=sessions,
            next_id=self.next_id,
        )
        facade = WorkbenchApplication(
            document=self.document,
            run_dir=Path("."),
            uow_factory=lambda: InMemoryUnitOfWork(self.database),
            workflow=workflow,
            sessions=sessions,
            workers=workers,
            next_id=self.next_id,
        )
        saved = facade.save_selection("root", self.selection.selection)
        attempt_id = self.next_id("ATTEMPT")
        facade._begin_attempt(  # noqa: SLF001
            saved.selection_id,
            attempt_id,
            "prompt_1",
        )
        worker = workers.decompose(saved, attempt_id)
        await worker.awaitable

        recovered = facade.recover_workflows()

        self.assertEqual(recovered, (attempt_id,))
        self.assertIsNone(
            workflow.current_state(saved.selection_id).active_attempt
        )
        self.assertEqual(
            workflow.current_state(saved.selection_id).decomposition_outcome,
            "skeletons_created",
        )

    async def test_recovery_reuses_completed_children_of_same_parent(
        self,
    ) -> None:
        first_transport = BlockingMockTransport(pause_on_call=2)
        first_flow = self.flow(first_transport)
        task = asyncio.create_task(
            first_flow.run(
                parent_attempt_id="ATTEMPT_PARENT",
                session_id=self.selection.selection_id,
                selection=self.selection,
                expected_workflow_revision="workflow-revision-1",
            )
        )
        await first_transport.started.wait()
        self.assertEqual(first_flow.progress().completed_windows, 1)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task

        recovered_transport = MockLlmTransport(delay=0)
        recovered_flow = self.flow(recovered_transport)
        result = await recovered_flow.run(
            parent_attempt_id="ATTEMPT_PARENT",
            session_id=self.selection.selection_id,
            selection=self.selection,
            expected_workflow_revision="workflow-revision-1",
        )

        self.assertEqual(result.outcome, "skeletons_created")
        _parent, plan = WindowedDecompositionStore(
            lambda: InMemoryUnitOfWork(self.database)
        ).load("ATTEMPT_PARENT", self.selection)
        self.assertEqual(
            len(
                [
                    call
                    for call in recovered_transport.calls
                    if call.prompt_id
                    is PromptId.DECOMPOSITION_WINDOW_SEMANTIC
                ]
            ),
            len(plan.windows) - 1,
        )

    async def test_explicit_cancel_discards_late_child_and_never_assembles(
        self,
    ) -> None:
        transport = BlockingMockTransport()
        flow = self.flow(transport)
        task = asyncio.create_task(
            flow.run(
                parent_attempt_id="ATTEMPT_PARENT",
                session_id=self.selection.selection_id,
                selection=self.selection,
                expected_workflow_revision="workflow-revision-1",
            )
        )
        await transport.started.wait()

        flow.cancel("ATTEMPT_PARENT")
        transport.release.set()

        with self.assertRaises(AttemptDiscardedError):
            await task
        self.assertNotIn(
            ("decomposition", self.selection.selection_id),
            self.database.records,
        )
        parent, _plan = WindowedDecompositionStore(
            lambda: InMemoryUnitOfWork(self.database)
        ).load("ATTEMPT_PARENT", self.selection)
        self.assertEqual(parent.status.value, "cancelled")


if __name__ == "__main__":
    unittest.main()
