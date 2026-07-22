from __future__ import annotations

import sys
from pathlib import Path
from typing import TextIO

from ..domain import ExecutionMode, SourceDocument
from ..infrastructure.composition import build_workbench_application
from ..infrastructure.storage import SqliteUnitOfWork, workbench_database_path
from ..infrastructure.source import load_source_document
from ..infrastructure.workflow import SqliteWorkflowRuntime
from ..presentation.startup import render_startup
from .recovery import RecoveryService
from .settings import WorkbenchSettings


def start_workbench(
    run_dir: Path,
    output: TextIO | None = None,
    *,
    document: SourceDocument | None = None,
    execution_mode: ExecutionMode = ExecutionMode.PRODUCTION,
) -> int:
    stream = output or sys.stdout
    document = document or load_source_document(run_dir)
    if document.metadata.execution_mode is not execution_mode:
        raise ValueError(
            "Execution mode запуска не совпадает с metadata source snapshot"
        )
    settings = WorkbenchSettings.from_environment(
        run_dir,
        execution_mode=execution_mode,
    )
    database_path = workbench_database_path(run_dir)
    uow_factory = lambda: SqliteUnitOfWork(database_path)
    with uow_factory():
        pass
    recovered = RecoveryService(uow_factory).recover()
    startup = render_startup(
        settings.run_dir,
        document=document,
        database_path=database_path,
        recovered_attempts=recovered,
        execution_mode=execution_mode,
    )
    if output is not None or not sys.stdin.isatty() or not sys.stdout.isatty():
        stream.write(startup)
        return 0
    return run_interactive_workbench(settings, document)


def run_interactive_workbench(
    settings: WorkbenchSettings,
    document: SourceDocument,
) -> int:
    from ..presentation.terminal import TerminalWorkbench

    with SqliteWorkflowRuntime(workbench_database_path(settings.run_dir)) as workflow:
        facade = build_workbench_application(settings, document, workflow)
        return TerminalWorkbench(document, facade=facade).run()
