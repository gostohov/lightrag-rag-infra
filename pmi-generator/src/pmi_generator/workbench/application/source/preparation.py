from __future__ import annotations

import os
import shutil
import uuid
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable, Protocol

from ...domain import ExecutionMode, SourceDocument
from .extraction import SourceExtractor


class SourcePreparationError(ValueError):
    """Run нельзя подготовить из указанного source."""


class SourceSnapshotRepository(Protocol):
    def load(self, run_dir: Path) -> SourceDocument: ...

    def save(self, run_dir: Path, document: SourceDocument) -> None: ...

    def pdf_path(self, run_dir: Path) -> Path: ...

    def sha256(self, path: Path) -> str: ...


class SourceRunPreparer:
    def __init__(
        self,
        *,
        extractor: SourceExtractor,
        repository: SourceSnapshotRepository,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self.extractor = extractor
        self.repository = repository
        self.now = now or (lambda: datetime.now(UTC))

    def prepare(
        self,
        run_dir: Path,
        pdf_path: Path | None = None,
        *,
        execution_mode: ExecutionMode = ExecutionMode.PRODUCTION,
    ) -> SourceDocument:
        if pdf_path is None:
            document = self.repository.load(run_dir)
            self._require_mode(document, execution_mode)
            return document
        if not pdf_path.is_file():
            raise SourcePreparationError(f"PDF-файл не найден: {pdf_path}")
        if run_dir.exists() and not run_dir.is_dir():
            raise SourcePreparationError(
                f"Путь run существует и не является директорией: {run_dir}"
            )
        if run_dir.is_dir() and any(run_dir.iterdir()):
            return self._resume(run_dir, pdf_path, execution_mode)
        return self._create(run_dir, pdf_path, execution_mode)

    def _resume(
        self,
        run_dir: Path,
        pdf_path: Path,
        execution_mode: ExecutionMode,
    ) -> SourceDocument:
        document = self.repository.load(run_dir)
        self._require_mode(document, execution_mode)
        incoming_hash = self._sha256(pdf_path)
        if incoming_hash != document.metadata.sha256:
            raise SourcePreparationError(
                "Указанный PDF не совпадает с source существующего run"
            )
        return document

    def _create(
        self,
        run_dir: Path,
        pdf_path: Path,
        execution_mode: ExecutionMode,
    ) -> SourceDocument:
        run_dir.parent.mkdir(parents=True, exist_ok=True)
        staging = run_dir.parent / f".{run_dir.name}.preparing-{uuid.uuid4().hex}"
        try:
            staging.mkdir()
            staged_pdf = self.repository.pdf_path(staging)
            staged_pdf.parent.mkdir(parents=True)
            shutil.copyfile(pdf_path, staged_pdf)
            document = self.extractor.extract(
                staged_pdf,
                original_name=pdf_path.name,
                created_at=self.now(),
            )
            document = SourceDocument(
                pages=document.pages,
                sections=document.sections,
                metadata=replace(
                    document.metadata,
                    execution_mode=execution_mode,
                ),
            )
            self.repository.save(staging, document)
            validated = self.repository.load(staging)
            os.replace(staging, run_dir)
            return validated
        except (OSError, shutil.Error) as error:
            raise SourcePreparationError(
                f"Не удалось подготовить run {run_dir}: {error}"
            ) from error
        finally:
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)

    def _sha256(self, path: Path) -> str:
        try:
            return self.repository.sha256(path)
        except OSError as error:
            raise SourcePreparationError(
                f"Не удалось прочитать PDF {path}: {error}"
            ) from error

    @staticmethod
    def _require_mode(
        document: SourceDocument,
        requested: ExecutionMode,
    ) -> None:
        actual = document.metadata.execution_mode
        if actual is requested:
            return
        raise SourcePreparationError(
            f"Run создан в режиме {actual.value}; запрошен режим {requested.value}"
        )
