from __future__ import annotations

import sqlite3
import uuid
from collections.abc import Callable

from pmi_generator.clients.lightrag import LightRAGClient
from langgraph.checkpoint.sqlite import SqliteSaver

from ..application.card_population import population_tool
from ..application.decomposition import (
    DecompositionBudgetPolicy,
    default_windowing_policy,
    decomposition_tool,
    reconciliation_case_tool,
    semantic_synthesis_tool,
    semantic_window_tool,
)
from ..application.facade import WorkbenchApplication
from ..application.gap_investigation import (
    RetrievalPort,
    ask_lightrag_tool,
    expand_lightrag_tool,
    submit_gap_result_tool,
)
from ..application.llm import LlmToolRuntime, LlmTransport, TypedToolRegistry
from ..application.prompting import default_policy
from ..application.refinement import refinement_tool
from ..application.selection_review import selection_review_tool
from ..application.session import SessionService
from ..application.settings import WorkbenchSettings
from ..application.source import SavedSelection
from ..application.workflow import WorkflowRuntime
from ..domain import ExecutionMode, SourceDocument
from .llm import OpenAICompatibleTransport, OpenAITransportSettings
from .langchain import LangChainConversationAgent
from .mock_mode import MockLlmTransport, MockRetrieval
from .retrieval import LightRAGRetrieval
from .storage import SqliteUnitOfWork, workbench_database_path
from .workers import ProductionPromptWorkers


def build_workbench_application(
    settings: WorkbenchSettings,
    document: SourceDocument,
    workflow: WorkflowRuntime,
    *,
    transport: LlmTransport | None = None,
    retrieval_factory: Callable[[SavedSelection], RetrievalPort] | None = None,
    next_id: Callable[[str], str] | None = None,
    mock_delay: float = 1.0,
) -> WorkbenchApplication:
    database_path = workbench_database_path(settings.run_dir)
    uow_factory = lambda: SqliteUnitOfWork(database_path)
    id_factory = next_id or (
        lambda prefix: f"{prefix}_{uuid.uuid4().hex[:12].upper()}"
    )
    policy = default_policy()
    sessions = SessionService(uow_factory=uow_factory)
    registry = TypedToolRegistry()
    for tool in (
        decomposition_tool(),
        semantic_window_tool(),
        semantic_synthesis_tool(),
        reconciliation_case_tool(),
        population_tool(),
        ask_lightrag_tool(),
        expand_lightrag_tool(),
        submit_gap_result_tool(),
        refinement_tool(),
        selection_review_tool(),
    ):
        registry.register(tool)

    runtime: LlmToolRuntime | None = None
    selected_transport = transport
    if selected_transport is None and settings.execution_mode is ExecutionMode.MOCK:
        selected_transport = MockLlmTransport(delay=mock_delay)
    transport_instance: LlmTransport | None = selected_transport

    def transport_factory() -> LlmTransport:
        nonlocal transport_instance
        if transport_instance is None:
            transport_instance = _production_transport(settings)
        return transport_instance

    def runtime_factory() -> LlmToolRuntime:
        nonlocal runtime
        if runtime is None:
            runtime = LlmToolRuntime(
                transport=transport_factory(),
                tools=registry,
                uow_factory=uow_factory,
            )
        return runtime

    def production_retrieval(_selection: SavedSelection) -> LightRAGRetrieval:
        return LightRAGRetrieval(
            LightRAGClient(
                settings.require_lightrag(),
                settings.lightrag_api_key,
                settings.retrieval_timeout,
                settings.verify_ssl,
                None,
                settings.no_proxy,
            )
        )

    selected_retrieval_factory = retrieval_factory
    if (
        selected_retrieval_factory is None
        and settings.execution_mode is ExecutionMode.MOCK
    ):
        selected_retrieval_factory = lambda selection: MockRetrieval(
            document,
            selection,
            delay=mock_delay,
        )

    workers = ProductionPromptWorkers(
        document=document,
        uow_factory=uow_factory,
        policy=policy,
        runtime_factory=runtime_factory,
        retrieval_factory=(
            selected_retrieval_factory
            if selected_retrieval_factory is not None
            else production_retrieval
        ),
        sessions=sessions,
        next_id=id_factory,
    )
    conversation_connection = sqlite3.connect(
        settings.run_dir / "review" / "conversation-checkpoints.sqlite3",
        check_same_thread=False,
    )
    conversation_agent = LangChainConversationAgent(
        transport=transport_factory(),
        checkpointer=SqliteSaver(conversation_connection),
    )
    application = WorkbenchApplication(
        document=document,
        run_dir=settings.run_dir,
        uow_factory=uow_factory,
        workflow=workflow,
        sessions=sessions,
        workers=workers,
        next_id=id_factory,
        conversation_agent=conversation_agent,
        decomposition_budget_policy=DecompositionBudgetPolicy.from_prompt_policy(policy),
        windowing_policy=default_windowing_policy(policy),
    )
    application.recover_workflows()
    return application


def _production_transport(settings: WorkbenchSettings) -> OpenAICompatibleTransport:
    url, model = settings.require_llm()
    return OpenAICompatibleTransport(
        OpenAITransportSettings(
            base_url=url,
            model=model,
            api_key=settings.llm_api_key,
            timeout=settings.llm_timeout,
            verify_ssl=settings.verify_ssl,
            no_proxy=settings.no_proxy,
        )
    )


__all__ = ["build_workbench_application"]
