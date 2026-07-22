from __future__ import annotations

from pathlib import Path

from ..domain import ExecutionMode, SourceDocument
from .source import render_structure


def render_startup(
    run_dir: Path,
    *,
    document: SourceDocument | None = None,
    database_path: Path | None = None,
    recovered_attempts: tuple[str, ...] = (),
    execution_mode: ExecutionMode = ExecutionMode.PRODUCTION,
) -> str:
    details = [
        f"Режим: {execution_mode.value}",
        f"Прогон: {run_dir}",
    ]
    if execution_mode is ExecutionMode.MOCK:
        details.insert(0, "Тестовый режим: mock")
    if database_path:
        details.append(f"База состояния: {database_path}")
    if recovered_attempts:
        details.append(
            "Прерванные операции восстановлены как отменённые: "
            + ", ".join(recovered_attempts)
        )
    if document is not None:
        details.extend(
            (
                f"Источник: {document.metadata.original_name}",
                f"SHA-256: {document.metadata.sha256}",
                (
                    "Parser: "
                    f"{document.metadata.parser_name} {document.metadata.parser_version}"
                ),
            )
        )
        return "\n".join(details) + f"\n\n{render_structure(document)}\n"
    return "PMI Workbench\n" + "\n".join(details) + "\n"
