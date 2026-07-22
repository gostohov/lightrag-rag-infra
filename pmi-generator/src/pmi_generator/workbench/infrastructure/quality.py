from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from ..application.decomposition import (
    DecompositionFlow,
    DecompositionService,
    WindowedDecompositionFlow,
    decomposition_tool,
    evaluation_windowing_policy,
    reconciliation_case_tool,
    semantic_synthesis_tool,
    semantic_window_tool,
)
from ..application.evaluation import (
    ExperimentManifest,
    ExperimentStage,
    ExperimentStore,
    ExperimentVariant,
    RunRecord,
)
from ..application.llm import LlmToolRuntime, LlmTransport, TypedToolRegistry
from ..application.prompting import PromptPolicy
from ..application.source import SavedSelection
from ..domain import SourceDocument
from .storage import InMemoryDatabase, InMemoryUnitOfWork


class QualityExperimentExecutor:
    def __init__(
        self,
        *,
        document: SourceDocument,
        policy: PromptPolicy,
        transport_factory: Callable[[], LlmTransport],
        output_root: Path,
        evaluation_route_max_lines: int,
        evaluation_primary_max_lines: int,
        evaluation_overlap_lines: int,
    ) -> None:
        self.document = document
        self.policy = policy
        self.transport_factory = transport_factory
        self.store = ExperimentStore(output_root)
        self.windowing_policy = evaluation_windowing_policy(
            policy,
            route_max_lines=evaluation_route_max_lines,
            primary_max_lines=evaluation_primary_max_lines,
            overlap_lines=evaluation_overlap_lines,
        )

    async def execute(
        self,
        manifest: ExperimentManifest,
    ) -> tuple[RunRecord, ...]:
        records: list[RunRecord] = []
        pairs = {pair.pair_id: pair for pair in manifest.pairs}
        for pair_id, role, repeat in self.schedule(manifest):
            pair = pairs[pair_id]
            if pair.source_hash != self.document.metadata.sha256:
                raise ValueError(
                    f"Case {pair.case_id} имеет другой source hash"
                )
            variant = (
                pair.baseline if role == "baseline" else pair.candidate
            )
            run_id = (
                f"RUN_{pair.pair_id.removeprefix('PAIR_')}_"
                f"{role.upper()}_{repeat:02d}"
            )
            record, raw = await self._execute_variant(
                pair.case_id,
                pair.pair_id,
                run_id,
                role,
                repeat,
                variant,
            )
            self.store.record(manifest, record, raw)
            records.append(record)
        return tuple(records)

    @staticmethod
    def schedule(
        manifest: ExperimentManifest,
    ) -> tuple[tuple[str, str, int], ...]:
        result: list[tuple[str, str, int]] = []
        for pair in manifest.pairs:
            repeats = (
                1
                if manifest.stage is ExperimentStage.PILOT
                else pair.baseline.configuration.repeats
            )
            for role in ("baseline", "candidate"):
                result.extend(
                    (pair.pair_id, role, repeat)
                    for repeat in range(1, repeats + 1)
                )
        return tuple(result)

    async def _execute_variant(
        self,
        case_id: str,
        pair_id: str,
        run_id: str,
        role: str,
        repeat: int,
        variant: ExperimentVariant,
    ) -> tuple[RunRecord, dict[str, object]]:
        database = InMemoryDatabase()
        uow_factory = lambda: InMemoryUnitOfWork(database)
        ids: dict[str, int] = {}

        def next_id(prefix: str) -> str:
            ids[prefix] = ids.get(prefix, 0) + 1
            return f"{prefix}_{run_id}_{ids[prefix]:04d}"

        selection = self.document.select(
            variant.selection.start,
            variant.selection.end,
        )
        actual_lines = tuple(
            (
                position.page_index,
                position.line_number,
                self.document.line(position),
            )
            for position in selection.positions
        )
        expected_lines = tuple(
            (line.page, line.line, line.text)
            for line in variant.selection.lines
        )
        if actual_lines != expected_lines:
            raise ValueError(
                f"Case {case_id} selection snapshot не совпадает с PDF"
            )
        outline = self.document.outline_at(selection.start)
        saved = SavedSelection(
            selection_id=f"SELECTION_{run_id}",
            section_id=outline.section_id,
            selection=selection,
            document_version=self.document.metadata.document_version,
            anchor_outline_node_id=outline.section_id,
        )
        tools = TypedToolRegistry()
        for tool in (
            decomposition_tool(),
            semantic_window_tool(),
            semantic_synthesis_tool(),
            reconciliation_case_tool(),
        ):
            tools.register(tool)
        runtime = LlmToolRuntime(
            transport=self.transport_factory(),
            tools=tools,
            uow_factory=uow_factory,
        )
        attempt_id = f"ATTEMPT_{run_id}"
        error: str | None = None
        try:
            if variant.prompt_1_mode == "single_call":
                result = await DecompositionFlow(
                    policy=self.policy,
                    runtime=runtime,
                    service=DecompositionService(
                        document=self.document,
                        uow_factory=uow_factory,
                        next_id=next_id,
                    ),
                ).run(
                    attempt_id=attempt_id,
                    session_id=run_id,
                    selection=saved,
                )
            elif variant.prompt_1_mode == "windowed":
                result = await WindowedDecompositionFlow(
                    document=self.document,
                    policy=self.policy,
                    windowing_policy=self.windowing_policy,
                    runtime=runtime,
                    uow_factory=uow_factory,
                    next_id=next_id,
                ).run(
                    parent_attempt_id=attempt_id,
                    session_id=run_id,
                    selection=saved,
                    expected_workflow_revision=attempt_id,
                )
            else:
                raise ValueError(
                    f"Неизвестный Prompt 1 mode {variant.prompt_1_mode}"
                )
            outcome = result.outcome
            status = "valid"
        except Exception as caught:  # noqa: BLE001 - invalid live run is an artifact
            outcome = None
            status = "invalid"
            error = f"{type(caught).__name__}: {caught}"
        raw = {
            "case_id": case_id,
            "pair_id": pair_id,
            "run_id": run_id,
            "variant_configuration": variant.to_dict(),
            "source_hash": self.document.metadata.sha256,
            "outcome": outcome,
            "error": error,
            "attempts": [
                {
                    "attempt_id": attempt.attempt_id,
                    "session_id": attempt.session_id,
                    "stage": attempt.stage,
                    "status": attempt.status.value,
                    "payload": attempt.payload,
                    "updated_at": attempt.updated_at.isoformat(),
                }
                for attempt in sorted(
                    database.attempts.values(),
                    key=lambda item: item.attempt_id,
                )
            ],
            "records": [
                {
                    "kind": record.kind,
                    "record_id": record.record_id,
                    "payload": record.payload,
                }
                for record in sorted(
                    database.records.values(),
                    key=lambda item: (item.kind, item.record_id),
                )
            ],
        }
        record = RunRecord(
            run_id=run_id,
            pair_id=pair_id,
            variant=role,
            repeat=repeat,
            status=status,
            diagnostic_path=f"raw/{run_id}.json",
            error=error,
        )
        return record, raw


QualityPilotExecutor = QualityExperimentExecutor


__all__ = ["QualityExperimentExecutor", "QualityPilotExecutor"]
