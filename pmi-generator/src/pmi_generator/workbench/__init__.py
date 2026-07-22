from __future__ import annotations

import sys
from pathlib import Path
from typing import TextIO

from .application.bootstrap import start_workbench
from .application.source import SourcePreparationError, SourceRunPreparer
from .domain import ExecutionMode, SourceDocument
from .infrastructure.source import (
    PypdfSourceExtractor,
    SqliteSourceSnapshotRepository,
)


def run_workbench(
    run_dir: Path,
    pdf_path: Path | None = None,
    output: TextIO | None = None,
    *,
    mock: bool = False,
) -> int:
    if mock and pdf_path is None:
        raise SourcePreparationError("Для --mock обязателен --pdf")
    execution_mode = ExecutionMode.MOCK if mock else ExecutionMode.PRODUCTION
    document = _prepare_source_document(
        run_dir,
        pdf_path,
        output=output,
        execution_mode=execution_mode,
    )
    return start_workbench(
        run_dir=run_dir,
        output=output,
        document=document,
        execution_mode=execution_mode,
    )


def _prepare_source_document(
    run_dir: Path,
    pdf_path: Path | None,
    *,
    output: TextIO | None,
    execution_mode: ExecutionMode = ExecutionMode.PRODUCTION,
) -> SourceDocument:
    preparer = SourceRunPreparer(
        extractor=PypdfSourceExtractor(),
        repository=SqliteSourceSnapshotRepository(),
    )
    if pdf_path is None or not _interactive_terminal(output):
        return preparer.prepare(
            run_dir,
            pdf_path,
            execution_mode=execution_mode,
        )

    from .presentation.operation import TerminalOperationRunner

    return TerminalOperationRunner().run_sync(
        "Подготовка источника из PDF",
        lambda: preparer.prepare(
            run_dir,
            pdf_path,
            execution_mode=execution_mode,
        ),
        context=lambda width, height: (
            "PMI Workbench / Подготовка источника\n\n"
            f"PDF: {pdf_path}\n"
            f"Прогон: {run_dir}\n"
            f"Режим: {execution_mode.value}"
        ),
        full_screen=True,
        interruptible=False,
    )


def _interactive_terminal(output: TextIO | None) -> bool:
    return output is None and sys.stdin.isatty() and sys.stdout.isatty()


__all__ = ["run_workbench"]
