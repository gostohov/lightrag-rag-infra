from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Protocol

from ...domain import SourceDocument


class SourceExtractionError(ValueError):
    """PDF нельзя преобразовать в канонический source snapshot."""


class SourceExtractor(Protocol):
    def extract(
        self,
        pdf_path: Path,
        *,
        original_name: str,
        created_at: datetime,
    ) -> SourceDocument: ...
