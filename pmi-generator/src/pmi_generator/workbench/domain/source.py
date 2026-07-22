from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
import re


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ExecutionMode(StrEnum):
    PRODUCTION = "production"
    MOCK = "mock"


@dataclass(frozen=True, slots=True)
class SourceMetadata:
    original_name: str
    sha256: str
    parser_name: str
    parser_version: str
    created_at: datetime
    execution_mode: ExecutionMode = ExecutionMode.PRODUCTION

    def __post_init__(self) -> None:
        if not isinstance(self.execution_mode, ExecutionMode):
            object.__setattr__(
                self,
                "execution_mode",
                ExecutionMode(self.execution_mode),
            )
        if not self.original_name.strip():
            raise ValueError("Источник должен иметь исходное имя")
        if not SHA256_RE.fullmatch(self.sha256):
            raise ValueError("Источник должен иметь SHA-256 в нижнем регистре")
        if not self.parser_name.strip() or not self.parser_version.strip():
            raise ValueError("Источник должен иметь имя и версию parser")
        if self.created_at.tzinfo is None:
            raise ValueError("Время создания source snapshot должно содержать timezone")

    @property
    def document_id(self) -> str:
        return self.original_name

    @property
    def document_version(self) -> str:
        return f"sha256:{self.sha256}"

    @classmethod
    def transient(cls) -> SourceMetadata:
        return cls(
            original_name="in-memory-source",
            sha256="0" * 64,
            parser_name="in-memory",
            parser_version="1",
            created_at=datetime(1970, 1, 1, tzinfo=UTC),
            execution_mode=ExecutionMode.PRODUCTION,
        )


@dataclass(frozen=True, order=True, slots=True)
class SourcePosition:
    page_index: int
    line_number: int

    def __post_init__(self) -> None:
        if self.page_index < 1 or self.line_number < 1:
            raise ValueError("Координаты источника должны быть положительными")


@dataclass(frozen=True, slots=True)
class SourcePage:
    page_index: int
    logical_page: str | None
    lines: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.page_index < 1:
            raise ValueError("Page index должен быть положительным")
        if self.logical_page is not None:
            object.__setattr__(self, "logical_page", str(self.logical_page))

    @property
    def display_number(self) -> str:
        return self.logical_page or str(self.page_index)


@dataclass(frozen=True, slots=True)
class SourceSection:
    section_id: str
    number: str
    title: str
    path: tuple[str, ...]
    page_indexes: tuple[int, ...]
    parent_section_id: str | None = None
    origin: str = "snapshot"
    anchor_page_index: int | None = None
    anchor_line_number: int | None = None

    def __post_init__(self) -> None:
        if not self.section_id.strip():
            raise ValueError("Outline node должен иметь ID")
        if not self.page_indexes:
            raise ValueError("Outline node должен ссылаться хотя бы на одну страницу")
        if any(page_index < 1 for page_index in self.page_indexes):
            raise ValueError("Страницы outline node должны быть положительными")
        if self.anchor_page_index is None:
            object.__setattr__(self, "anchor_page_index", self.page_indexes[0])
        assert self.anchor_page_index is not None
        if self.anchor_page_index < 1:
            raise ValueError("Anchor page должна быть положительной")
        if self.anchor_page_index not in self.page_indexes:
            raise ValueError("Anchor page должна входить в страницы outline node")
        if self.anchor_line_number is not None and self.anchor_line_number < 1:
            raise ValueError("Anchor line должна быть положительной")

    @property
    def label(self) -> str:
        return " ".join(part.strip() for part in (self.number, self.title) if part.strip())


@dataclass(frozen=True, slots=True)
class TextSelection:
    start: SourcePosition
    end: SourcePosition
    positions: tuple[SourcePosition, ...]
    text: str


class SourceDocument:
    def __init__(
        self,
        pages: tuple[SourcePage, ...],
        sections: tuple[SourceSection, ...],
        metadata: SourceMetadata | None = None,
    ) -> None:
        if not pages:
            raise ValueError("Source snapshot не содержит страниц")
        self.metadata = metadata or SourceMetadata.transient()
        self.pages = pages
        self.sections = sections
        self._pages = {page.page_index: page for page in pages}
        if len(self._pages) != len(pages):
            raise ValueError("Source snapshot содержит повторяющиеся page_index")
        section_ids = [section.section_id for section in sections]
        if len(set(section_ids)) != len(section_ids):
            raise ValueError("Outline содержит повторяющиеся ID")
        preceding_section_ids: set[str] = set()
        for section in sections:
            if (
                section.parent_section_id is not None
                and section.parent_section_id not in preceding_section_ids
            ):
                raise ValueError(
                    f"Outline node {section.section_id} ссылается на неизвестный parent "
                    f"{section.parent_section_id}"
                )
            preceding_section_ids.add(section.section_id)
        self._positions = tuple(
            SourcePosition(page.page_index, number)
            for page in pages
            for number in range(1, len(page.lines) + 1)
        )
        self._order = {position: index for index, position in enumerate(self._positions)}
        self._section_anchors: dict[str, SourcePosition] = {}
        for section in sections:
            for page_index in section.page_indexes:
                self.page(page_index)
            if section.anchor_line_number is not None:
                self.line(
                    SourcePosition(
                        section.anchor_page_index,
                        section.anchor_line_number,
                    )
                )
            self._section_anchors[section.section_id] = self._resolve_anchor(section)

    @property
    def positions(self) -> tuple[SourcePosition, ...]:
        return self._positions

    def page(self, page_index: int) -> SourcePage:
        try:
            return self._pages[page_index]
        except KeyError as error:
            raise ValueError(f"В source snapshot нет страницы {page_index}") from error

    def line(self, position: SourcePosition) -> str:
        page = self.page(position.page_index)
        try:
            return page.lines[position.line_number - 1]
        except IndexError as error:
            raise ValueError(
                f"На странице {position.page_index} нет строки {position.line_number}"
            ) from error

    def positions_for_pages(self, page_indexes: tuple[int, ...]) -> tuple[SourcePosition, ...]:
        requested = set(page_indexes)
        for page_index in page_indexes:
            self.page(page_index)
        return tuple(position for position in self._positions if position.page_index in requested)

    def anchor_position(self, section: SourceSection) -> SourcePosition:
        try:
            return self._section_anchors[section.section_id]
        except KeyError as error:
            raise ValueError(
                f"Outline node {section.section_id} отсутствует в source snapshot"
            ) from error

    def position_index(self, position: SourcePosition) -> int:
        try:
            return self._order[position]
        except KeyError as error:
            raise ValueError(
                f"Координата {position} отсутствует в source snapshot"
            ) from error

    def outline_at(
        self,
        position: SourcePosition,
        *,
        preferred_section_id: str | None = None,
    ) -> SourceSection:
        cursor_index = self.position_index(position)
        candidates: list[tuple[int, SourceSection]] = []
        for section in self.sections:
            anchor_index = self.position_index(self._section_anchors[section.section_id])
            if anchor_index <= cursor_index:
                candidates.append((anchor_index, section))
        if candidates:
            latest_anchor = max(anchor for anchor, _section in candidates)
            tied = tuple(
                section
                for anchor, section in candidates
                if anchor == latest_anchor
            )
            if preferred_section_id is not None:
                preferred = next(
                    (
                        section
                        for section in tied
                        if section.section_id == preferred_section_id
                    ),
                    None,
                )
                if preferred is not None:
                    return preferred
            return tied[-1]
        if not self.sections:
            raise ValueError("Source snapshot не содержит outline nodes")
        return self.sections[0]

    def _resolve_anchor(self, section: SourceSection) -> SourcePosition:
        if section.anchor_line_number is not None:
            position = SourcePosition(
                section.anchor_page_index,
                section.anchor_line_number,
            )
            return position
        for position in self._positions:
            if position.page_index >= section.anchor_page_index:
                return position
        raise ValueError(
            f"После anchor page {section.anchor_page_index} нет извлечённых строк"
        )

    def select(self, first: SourcePosition, second: SourcePosition) -> TextSelection:
        self.line(first)
        self.line(second)
        try:
            left, right = sorted((self._order[first], self._order[second]))
        except KeyError as error:
            raise ValueError(
                f"Координата {error.args[0]} отсутствует в source snapshot"
            ) from error
        positions = self._positions[left : right + 1]
        return TextSelection(
            start=positions[0],
            end=positions[-1],
            positions=positions,
            text="\n".join(self.line(position) for position in positions),
        )
