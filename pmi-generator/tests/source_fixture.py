from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

from pmi_generator.workbench.domain import (
    ExecutionMode,
    SourceDocument,
    SourceMetadata,
    SourcePage,
    SourceSection,
)
from pmi_generator.workbench.infrastructure.source import (
    save_source_document,
    source_database_path,
    source_pdf_path,
)


FIXTURE_PDF = b"%PDF-1.4\n% PMI Workbench source fixture\n"


def source_metadata(
    *,
    original_name: str = "specification.pdf",
    pdf_bytes: bytes = FIXTURE_PDF,
    execution_mode: ExecutionMode = ExecutionMode.PRODUCTION,
) -> SourceMetadata:
    return SourceMetadata(
        original_name=original_name,
        sha256=hashlib.sha256(pdf_bytes).hexdigest(),
        parser_name="fixture",
        parser_version="1",
        created_at=datetime(2026, 7, 17, tzinfo=UTC),
        execution_mode=execution_mode,
    )


def write_source_snapshot(
    run_dir: Path,
    *,
    pages: tuple[SourcePage, ...],
    sections: tuple[SourceSection, ...],
    original_name: str = "specification.pdf",
    pdf_bytes: bytes = FIXTURE_PDF,
    execution_mode: ExecutionMode = ExecutionMode.PRODUCTION,
) -> SourceDocument:
    document = SourceDocument(
        pages=pages,
        sections=sections,
        metadata=source_metadata(
            original_name=original_name,
            pdf_bytes=pdf_bytes,
            execution_mode=execution_mode,
        ),
    )
    pdf_path = source_pdf_path(run_dir)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.write_bytes(pdf_bytes)
    save_source_document(source_database_path(run_dir), document)
    return document
