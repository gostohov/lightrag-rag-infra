from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime

from ..application.card_population import CardPopulationFlow, PopulationService
from ..application.conversation import AnalystProposalService
from ..application.decomposition import (
    DecompositionFlow,
    DecompositionRoute,
    DecompositionService,
    WindowPlanError,
    WindowedDecompositionFlow,
    default_windowing_policy,
)
from ..application.gap_investigation import (
    GapInvestigationFlow,
    GapInvestigationService,
    RetrievalBudgetPolicy,
    RetrievalPort,
)
from ..application.llm import LlmToolRuntime
from ..application.prompting import PromptPolicy
from ..application.range_workspace import RangeWorkspaceService
from ..application.refinement import CardRefinementFlow, CardRefinementService
from ..application.repositories import UnitOfWork
from ..application.selection_review import SelectionReviewFlow, SelectionReviewService
from ..application.session import SessionService
from ..application.source import SavedSelection
from ..application.worker_ports import WorkerOperation
from ..domain import Evidence, SourceAddress, SourceDocument
from .langchain import LangChainGapAgent


class ProductionPromptWorkers:
    def __init__(
        self,
        *,
        document: SourceDocument,
        uow_factory: Callable[[], UnitOfWork],
        policy: PromptPolicy,
        runtime_factory: Callable[[], LlmToolRuntime],
        retrieval_factory: Callable[[SavedSelection], RetrievalPort],
        sessions: SessionService,
        next_id: Callable[[str], str],
    ) -> None:
        self.document = document
        self.uow_factory = uow_factory
        self.policy = policy
        self.runtime_factory = runtime_factory
        self.retrieval_factory = retrieval_factory
        self.sessions = sessions
        self.next_id = next_id
        self.windowing_policy = default_windowing_policy(policy)

    def decompose(self, selection: SavedSelection, attempt_id: str) -> WorkerOperation:
        runtime = self.runtime_factory()
        decision = self.windowing_policy.assess(
            selection.selection,
            selection_id=selection.selection_id,
        )
        if decision.route is DecompositionRoute.WINDOWED:
            flow = WindowedDecompositionFlow(
                document=self.document,
                policy=self.policy,
                windowing_policy=self.windowing_policy,
                runtime=runtime,
                uow_factory=self.uow_factory,
                next_id=self.next_id,
            )
            return WorkerOperation(
                flow.run(
                    parent_attempt_id=attempt_id,
                    session_id=selection.selection_id,
                    selection=selection,
                    expected_workflow_revision=attempt_id,
                ),
                lambda: flow.cancel(attempt_id),
                flow.progress,
            )
        if decision.route is DecompositionRoute.HARD_LIMIT:
            raise WindowPlanError(
                "Selection превышает hard limit windowed Prompt 1"
            )
        flow = DecompositionFlow(
            policy=self.policy,
            runtime=runtime,
            service=DecompositionService(
                document=self.document,
                uow_factory=self.uow_factory,
                next_id=self.next_id,
            ),
        )
        return WorkerOperation(
            flow.run(
                attempt_id=attempt_id,
                session_id=selection.selection_id,
                selection=selection,
            ),
            lambda: runtime.cancel(attempt_id),
        )

    def populate(
        self,
        selection: SavedSelection,
        skeleton_id: str,
        session_id: str,
        card_id: str,
        attempt_id: str,
    ) -> WorkerOperation:
        with self.uow_factory() as uow:
            skeleton = uow.records.get("card_skeleton", skeleton_id)
        if skeleton is None:
            raise ValueError(f"Каркас {skeleton_id} не найден")
        runtime = self.runtime_factory()
        flow = CardPopulationFlow(
            policy=self.policy,
            runtime=runtime,
            service=PopulationService(
                uow_factory=self.uow_factory,
                next_id=self.next_id,
            ),
            sessions=self.sessions,
        )
        return WorkerOperation(
            flow.run(
                attempt_id=attempt_id,
                session_id=session_id,
                card_id=card_id,
                selection={"text": selection.selection.text},
                skeleton=dict(skeleton.payload),
                available_evidence=self._population_evidence(
                    selection,
                    card_id,
                    skeleton.payload,
                ),
            ),
            lambda: flow.cancel(attempt_id),
        )

    def investigate_gap(
        self,
        selection: SavedSelection,
        session_id: str,
        card_id: str,
        gap_id: str,
        attempt_id: str,
        research_question: str | None = None,
        research_message_id: str | None = None,
    ) -> WorkerOperation:
        runtime = self.runtime_factory()
        flow = GapInvestigationFlow(
            policy=self.policy,
            runtime=runtime,
            agent=LangChainGapAgent(runtime),
            retrieval=self.retrieval_factory(selection),
            budgets=RetrievalBudgetPolicy.defaults(),
            service=GapInvestigationService(
                uow_factory=self.uow_factory,
                next_id=self.next_id,
            ),
            sessions=self.sessions,
            uow_factory=self.uow_factory,
            next_id=self.next_id,
        )
        return WorkerOperation(
            flow.run(
                attempt_id=attempt_id,
                session_id=session_id,
                card_id=card_id,
                gap_id=gap_id,
                selection={"text": selection.selection.text},
                research_question=research_question,
                research_message_id=research_message_id,
            ),
            lambda: flow.cancel(attempt_id),
        )

    def plan_refinement(
        self,
        session_id: str,
        card_id: str,
        message_id: str,
        attempt_id: str,
        expected_revision: int,
    ) -> WorkerOperation:
        runtime = self.runtime_factory()
        flow = CardRefinementFlow(
            policy=self.policy,
            runtime=runtime,
            service=CardRefinementService(
                uow_factory=self.uow_factory,
                next_id=self.next_id,
            ),
            sessions=self.sessions,
            proposals=AnalystProposalService(
                uow_factory=self.uow_factory,
                next_id=self.next_id,
                clock=self.sessions.clock,
            ),
        )
        return WorkerOperation(
            flow.plan(
                attempt_id=attempt_id,
                session_id=session_id,
                card_id=card_id,
                message_id=message_id,
                expected_revision=expected_revision,
            ),
            lambda: runtime.cancel(attempt_id),
        )

    def _population_evidence(
        self,
        selection: SavedSelection,
        card_id: str,
        skeleton: dict[str, object],
    ) -> tuple[Evidence, ...]:
        fragments: list[dict[str, object]] = []
        for key in (
            "condition_evidence",
            "changed_factor_evidence",
            "input_value_evidence",
            "action_evidence",
        ):
            value = skeleton.get(key, [])
            if isinstance(value, list):
                fragments.extend(item for item in value if isinstance(item, dict))
        consequences = skeleton.get("consequences", [])
        if isinstance(consequences, list):
            for consequence in consequences:
                if not isinstance(consequence, dict):
                    continue
                value = consequence.get("evidence", [])
                if isinstance(value, list):
                    fragments.extend(item for item in value if isinstance(item, dict))

        unique: dict[tuple[int, int, int, str], dict[str, object]] = {}
        for item in fragments:
            key = (
                int(item["page"]),
                int(item["line_start"]),
                int(item["line_end"]),
                str(item["quote"]),
            )
            unique[key] = item

        result: list[Evidence] = []
        collected_at = datetime.now(UTC)
        for page_index, line_start, line_end, quote in unique:
            result.append(
                Evidence.source_fragment(
                    evidence_id=self.next_id("EVIDENCE"),
                    card_id=card_id,
                    selection_id=selection.selection_id,
                    quote=quote,
                    address=SourceAddress(
                        document_id=self.document.metadata.document_id,
                        document_version=self.document.metadata.document_version,
                        page=page_index,
                        line_start=line_start,
                        line_end=line_end,
                        chunk_id=selection.section_id,
                    ),
                    collected_at=collected_at,
                )
            )
        return tuple(result)

    def review_selection(
        self,
        selection: SavedSelection,
        attempt_id: str,
    ) -> WorkerOperation:
        runtime = self.runtime_factory()
        service = SelectionReviewService(
            uow_factory=self.uow_factory,
            workspace=RangeWorkspaceService(uow_factory=self.uow_factory),
            next_id=self.next_id,
        )
        flow = SelectionReviewFlow(
            policy=self.policy,
            runtime=runtime,
            service=service,
        )
        return WorkerOperation(
            flow.run(attempt_id, selection.selection_id, selection),
            lambda: runtime.cancel(attempt_id),
        )


__all__ = ["ProductionPromptWorkers"]
