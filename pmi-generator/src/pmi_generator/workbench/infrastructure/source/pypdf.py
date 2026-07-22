from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pypdf
from pypdf import PdfReader

from ...application.source import SourceExtractionError
from ...domain import SourceDocument, SourceMetadata, SourcePage, SourceSection


HORIZONTAL_SPACE_RE = re.compile(r"[ \t]+")


@dataclass(frozen=True, slots=True)
class _SectionDraft:
    section_id: str
    number: str
    title: str
    path: tuple[str, ...]
    page_index: int
    parent_section_id: str | None
    origin: str


class PypdfSourceExtractor:
    def extract(
        self,
        pdf_path: Path,
        *,
        original_name: str,
        created_at: datetime,
    ) -> SourceDocument:
        try:
            if not pdf_path.is_file():
                raise SourceExtractionError(f"PDF-файл не найден: {pdf_path}")
            with pdf_path.open("rb") as stream:
                if stream.read(5) != b"%PDF-":
                    raise SourceExtractionError(
                        f"Файл не содержит PDF signature: {pdf_path}"
                    )
            reader = PdfReader(str(pdf_path))
            if reader.is_encrypted:
                raise SourceExtractionError(
                    "Зашифрованный PDF без пароля не поддерживается"
                )
            pages = self._pages(reader)
            if not any(line.strip() for page in pages for line in page.lines):
                raise SourceExtractionError(
                    "PDF не содержит извлекаемого text layer"
                )
            sections = self._outline_sections(reader, len(pages))
            sections = (
                self._with_uncovered_pages(sections, pages)
                if sections
                else self._page_sections(pages)
            )
            metadata = SourceMetadata(
                original_name=original_name,
                sha256=_sha256(pdf_path),
                parser_name="pypdf",
                parser_version=pypdf.__version__,
                created_at=created_at,
            )
            return SourceDocument(
                pages=pages,
                sections=sections,
                metadata=metadata,
            )
        except SourceExtractionError:
            raise
        except Exception as error:
            raise SourceExtractionError(f"Не удалось прочитать PDF: {error}") from error

    @staticmethod
    def _pages(reader: PdfReader) -> tuple[SourcePage, ...]:
        try:
            labels = tuple(reader.page_labels)
        except Exception:
            labels = ()
        result: list[SourcePage] = []
        for page_index, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            lines = tuple(_normalize_line(line) for line in _normalize_newlines(text).splitlines())
            label = labels[page_index - 1] if page_index <= len(labels) else ""
            logical_page = str(label).strip() or None
            result.append(
                SourcePage(
                    page_index=page_index,
                    logical_page=logical_page,
                    lines=lines,
                )
            )
        if not result:
            raise SourceExtractionError("PDF не содержит страниц")
        return tuple(result)

    def _outline_sections(
        self,
        reader: PdfReader,
        pages_total: int,
    ) -> tuple[SourceSection, ...]:
        try:
            outline = reader.outline
        except Exception:
            return ()
        drafts: list[_SectionDraft] = []

        def walk(
            items: list[Any],
            *,
            parent_id: str | None,
            parent_path: tuple[str, ...],
        ) -> None:
            previous_id = parent_id
            previous_path = parent_path
            for item in items:
                if isinstance(item, list):
                    walk(
                        item,
                        parent_id=previous_id,
                        parent_path=previous_path,
                    )
                    continue
                try:
                    page_index = reader.get_destination_page_number(item) + 1
                except Exception:
                    previous_id = parent_id
                    previous_path = parent_path
                    continue
                if page_index < 1 or page_index > pages_total:
                    continue
                title_value = str(getattr(item, "title", item)).strip()
                if not title_value:
                    continue
                section_id = f"outline-{len(drafts) + 1:04d}"
                path = (*parent_path, title_value)
                drafts.append(
                    _SectionDraft(
                        section_id=section_id,
                        number="",
                        title=title_value,
                        path=path,
                        page_index=page_index,
                        parent_section_id=parent_id,
                        origin="outline",
                    )
                )
                previous_id = section_id
                previous_path = path

        walk(list(outline), parent_id=None, parent_path=())
        return _materialize_sections(drafts, pages_total)

    @staticmethod
    def _page_sections(
        pages: tuple[SourcePage, ...],
    ) -> tuple[SourceSection, ...]:
        return tuple(_page_section(page) for page in pages)

    @staticmethod
    def _with_uncovered_pages(
        sections: tuple[SourceSection, ...],
        pages: tuple[SourcePage, ...],
    ) -> tuple[SourceSection, ...]:
        covered = {
            page_index
            for section in sections
            for page_index in section.page_indexes
        }
        uncovered = tuple(page for page in pages if page.page_index not in covered)
        if not uncovered:
            return sections
        first_outline_page = min(
            page_index
            for section in sections
            for page_index in section.page_indexes
        )
        leading = tuple(
            _page_section(page)
            for page in uncovered
            if page.page_index < first_outline_page
        )
        remaining = tuple(
            _page_section(page)
            for page in uncovered
            if page.page_index >= first_outline_page
        )
        return (*leading, *sections, *remaining)


def _normalize_newlines(value: str) -> str:
    return unicodedata.normalize("NFC", value).replace("\r\n", "\n").replace("\r", "\n")


def _normalize_line(value: str) -> str:
    return HORIZONTAL_SPACE_RE.sub(" ", value).strip()


def _page_section(page: SourcePage) -> SourceSection:
    return SourceSection(
        section_id=f"page-{page.page_index:04d}",
        number="",
        title=f"Страница {page.display_number}",
        path=(page.display_number,),
        page_indexes=(page.page_index,),
        origin="page",
        anchor_page_index=page.page_index,
    )


def _materialize_sections(
    drafts: list[_SectionDraft],
    pages_total: int,
) -> tuple[SourceSection, ...]:
    result: list[SourceSection] = []
    for index, draft in enumerate(drafts):
        next_page = (
            drafts[index + 1].page_index
            if index + 1 < len(drafts)
            else pages_total
        )
        end_page = max(draft.page_index, next_page)
        result.append(
            SourceSection(
                section_id=draft.section_id,
                number=draft.number,
                title=draft.title,
                path=draft.path,
                page_indexes=tuple(range(draft.page_index, end_page + 1)),
                parent_section_id=draft.parent_section_id,
                origin=draft.origin,
                anchor_page_index=draft.page_index,
            )
        )
    return tuple(result)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
