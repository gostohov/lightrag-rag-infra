from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime
from pathlib import Path

from ...domain import (
    ExecutionMode,
    SourceDocument,
    SourceMetadata,
    SourcePage,
    SourceSection,
)
from .errors import SourceStorageError


SOURCE_SCHEMA_VERSION = 3
SUPPORTED_SOURCE_SCHEMA_VERSIONS = frozenset({2, SOURCE_SCHEMA_VERSION})


def source_directory(run_dir: Path) -> Path:
    return run_dir / "source"


def source_database_path(run_dir: Path) -> Path:
    return source_directory(run_dir) / "document.sqlite3"


def source_pdf_path(run_dir: Path) -> Path:
    return source_directory(run_dir) / "original.pdf"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_source_document(database_path: Path, document: SourceDocument) -> None:
    database_path.parent.mkdir(parents=True, exist_ok=True)
    if database_path.exists():
        raise SourceStorageError(f"Source snapshot уже существует: {database_path}")
    connection = sqlite3.connect(database_path)
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.executescript(
            """
            CREATE TABLE source_schema (
                version INTEGER NOT NULL
            );
            CREATE TABLE source_metadata (
                singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
                original_name TEXT NOT NULL,
                sha256 TEXT NOT NULL,
                parser_name TEXT NOT NULL,
                parser_version TEXT NOT NULL,
                created_at TEXT NOT NULL,
                execution_mode TEXT NOT NULL,
                pages_total INTEGER NOT NULL
            );
            CREATE TABLE source_pages (
                page_index INTEGER PRIMARY KEY,
                logical_page TEXT
            );
            CREATE TABLE source_lines (
                page_index INTEGER NOT NULL,
                line_number INTEGER NOT NULL,
                text TEXT NOT NULL,
                PRIMARY KEY(page_index, line_number),
                FOREIGN KEY(page_index) REFERENCES source_pages(page_index)
            );
            CREATE TABLE source_sections (
                ordinal INTEGER NOT NULL UNIQUE,
                section_id TEXT PRIMARY KEY,
                number TEXT NOT NULL,
                title TEXT NOT NULL,
                parent_section_id TEXT,
                origin TEXT NOT NULL,
                anchor_page_index INTEGER NOT NULL,
                anchor_line_number INTEGER,
                FOREIGN KEY(parent_section_id) REFERENCES source_sections(section_id)
            );
            CREATE TABLE source_section_path (
                section_id TEXT NOT NULL,
                depth INTEGER NOT NULL,
                label TEXT NOT NULL,
                PRIMARY KEY(section_id, depth),
                FOREIGN KEY(section_id) REFERENCES source_sections(section_id)
            );
            CREATE TABLE source_section_pages (
                section_id TEXT NOT NULL,
                ordinal INTEGER NOT NULL,
                page_index INTEGER NOT NULL,
                PRIMARY KEY(section_id, ordinal),
                FOREIGN KEY(section_id) REFERENCES source_sections(section_id),
                FOREIGN KEY(page_index) REFERENCES source_pages(page_index)
            );
            """
        )
        connection.execute(
            "INSERT INTO source_schema(version) VALUES (?)",
            (SOURCE_SCHEMA_VERSION,),
        )
        metadata = document.metadata
        connection.execute(
            """
            INSERT INTO source_metadata(
                singleton, original_name, sha256, parser_name, parser_version,
                created_at, execution_mode, pages_total
            ) VALUES (1, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                metadata.original_name,
                metadata.sha256,
                metadata.parser_name,
                metadata.parser_version,
                metadata.created_at.isoformat(),
                metadata.execution_mode.value,
                len(document.pages),
            ),
        )
        for page in document.pages:
            connection.execute(
                "INSERT INTO source_pages(page_index, logical_page) VALUES (?, ?)",
                (page.page_index, page.logical_page),
            )
            connection.executemany(
                """
                INSERT INTO source_lines(page_index, line_number, text)
                VALUES (?, ?, ?)
                """,
                (
                    (page.page_index, line_number, text)
                    for line_number, text in enumerate(page.lines, start=1)
                ),
            )
        for ordinal, section in enumerate(document.sections, start=1):
            connection.execute(
                """
                INSERT INTO source_sections(
                    ordinal, section_id, number, title, parent_section_id, origin,
                    anchor_page_index, anchor_line_number
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ordinal,
                    section.section_id,
                    section.number,
                    section.title,
                    section.parent_section_id,
                    section.origin,
                    section.anchor_page_index,
                    section.anchor_line_number,
                ),
            )
            connection.executemany(
                """
                INSERT INTO source_section_path(section_id, depth, label)
                VALUES (?, ?, ?)
                """,
                (
                    (section.section_id, depth, label)
                    for depth, label in enumerate(section.path)
                ),
            )
            connection.executemany(
                """
                INSERT INTO source_section_pages(section_id, ordinal, page_index)
                VALUES (?, ?, ?)
                """,
                (
                    (section.section_id, page_ordinal, page_index)
                    for page_ordinal, page_index in enumerate(section.page_indexes)
                ),
            )
        connection.commit()
    except (sqlite3.Error, ValueError) as error:
        connection.rollback()
        raise SourceStorageError(f"Не удалось сохранить source snapshot: {error}") from error
    finally:
        connection.close()


def load_source_document(run_dir: Path) -> SourceDocument:
    database_path = source_database_path(run_dir)
    pdf_path = source_pdf_path(run_dir)
    if not database_path.is_file():
        raise SourceStorageError(
            f"Run несовместим: не найден {database_path.relative_to(run_dir)}"
        )
    if not pdf_path.is_file():
        raise SourceStorageError(
            f"Run повреждён: не найден {pdf_path.relative_to(run_dir)}"
        )
    try:
        connection = sqlite3.connect(
            f"{database_path.resolve().as_uri()}?mode=ro",
            uri=True,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        document = _load(connection)
    except (sqlite3.Error, KeyError, TypeError, ValueError) as error:
        raise SourceStorageError(f"Некорректный source snapshot: {error}") from error
    finally:
        if "connection" in locals():
            connection.close()
    actual_hash = file_sha256(pdf_path)
    if actual_hash != document.metadata.sha256:
        raise SourceStorageError(
            "Run повреждён: SHA-256 source/original.pdf не совпадает с metadata"
        )
    return document


def _load(connection: sqlite3.Connection) -> SourceDocument:
    schema = connection.execute("SELECT version FROM source_schema").fetchall()
    actual_version = int(schema[0]["version"]) if len(schema) == 1 else None
    if actual_version not in SUPPORTED_SOURCE_SCHEMA_VERSIONS:
        actual = schema[0]["version"] if schema else "отсутствует"
        raise SourceStorageError(
            f"Версия source schema {actual} несовместима с {SOURCE_SCHEMA_VERSION}"
        )
    row = connection.execute("SELECT * FROM source_metadata").fetchone()
    if row is None:
        raise SourceStorageError("В source snapshot отсутствует metadata")
    metadata = SourceMetadata(
        original_name=str(row["original_name"]),
        sha256=str(row["sha256"]),
        parser_name=str(row["parser_name"]),
        parser_version=str(row["parser_version"]),
        created_at=datetime.fromisoformat(str(row["created_at"])),
        execution_mode=ExecutionMode(str(row["execution_mode"])),
    )
    page_rows = connection.execute(
        "SELECT page_index, logical_page FROM source_pages ORDER BY page_index"
    ).fetchall()
    pages = tuple(
        SourcePage(
            page_index=int(page["page_index"]),
            logical_page=(
                str(page["logical_page"]) if page["logical_page"] is not None else None
            ),
            lines=tuple(
                str(line["text"])
                for line in connection.execute(
                    """
                    SELECT text FROM source_lines
                    WHERE page_index = ? ORDER BY line_number
                    """,
                    (page["page_index"],),
                ).fetchall()
            ),
        )
        for page in page_rows
    )
    if int(row["pages_total"]) != len(pages):
        raise SourceStorageError("Количество страниц не совпадает с metadata")
    section_rows = connection.execute(
        "SELECT * FROM source_sections ORDER BY ordinal"
    ).fetchall()
    has_anchor_columns = (
        bool(section_rows)
        and "anchor_page_index" in section_rows[0].keys()
        and actual_version >= 3
    )
    sections = tuple(
        SourceSection(
            section_id=str(section["section_id"]),
            number=str(section["number"]),
            title=str(section["title"]),
            path=tuple(
                str(item["label"])
                for item in connection.execute(
                    """
                    SELECT label FROM source_section_path
                    WHERE section_id = ? ORDER BY depth
                    """,
                    (section["section_id"],),
                ).fetchall()
            ),
            page_indexes=tuple(
                int(item["page_index"])
                for item in connection.execute(
                    """
                    SELECT page_index FROM source_section_pages
                    WHERE section_id = ? ORDER BY ordinal
                    """,
                    (section["section_id"],),
                ).fetchall()
            ),
            parent_section_id=(
                str(section["parent_section_id"])
                if section["parent_section_id"] is not None
                else None
            ),
            origin=str(section["origin"]),
            anchor_page_index=(
                int(section["anchor_page_index"])
                if has_anchor_columns
                else None
            ),
            anchor_line_number=(
                int(section["anchor_line_number"])
                if has_anchor_columns and section["anchor_line_number"] is not None
                else None
            ),
        )
        for section in section_rows
    )
    document = SourceDocument(pages=pages, sections=sections, metadata=metadata)
    for section in document.sections:
        for page_index in section.page_indexes:
            document.page(page_index)
    return document
