from __future__ import annotations

from pathlib import Path

from ...domain import SourceDocument
from .sqlite import (
    file_sha256,
    load_source_document,
    save_source_document,
    source_database_path,
    source_pdf_path,
)


class SqliteSourceSnapshotRepository:
    def load(self, run_dir: Path) -> SourceDocument:
        return load_source_document(run_dir)

    def save(self, run_dir: Path, document: SourceDocument) -> None:
        save_source_document(source_database_path(run_dir), document)

    def pdf_path(self, run_dir: Path) -> Path:
        return source_pdf_path(run_dir)

    def sha256(self, path: Path) -> str:
        return file_sha256(path)
