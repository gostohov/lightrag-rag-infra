from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from pmi_generator.workbench.domain import SourcePage, SourcePosition, SourceSection
from pmi_generator.workbench.infrastructure.source import (
    SourceStorageError,
    load_source_document,
    source_database_path,
    source_pdf_path,
)

from tests.source_fixture import FIXTURE_PDF, write_source_snapshot


class SourceStorageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.run_dir = Path(self.temporary.name)
        self.document = write_source_snapshot(
            self.run_dir,
            pages=(
                SourcePage(1, 283, ("Первая строка", "Вторая строка")),
                SourcePage(2, 284, ("Продолжение",)),
            ),
            sections=(
                SourceSection(
                    "outline-0001",
                    "4.16.5",
                    "Обработка команды",
                    ("4", "4.16", "4.16.5"),
                    (1, 2),
                    origin="outline",
                    anchor_page_index=1,
                    anchor_line_number=2,
                ),
            ),
            original_name="EMV spec 2.3.pdf",
        )

    def test_round_trip_preserves_metadata_pages_lines_and_sections(self) -> None:
        restored = load_source_document(self.run_dir)

        self.assertEqual(restored.metadata, self.document.metadata)
        self.assertEqual(restored.pages, self.document.pages)
        self.assertEqual(restored.sections, self.document.sections)
        self.assertEqual(
            restored.metadata.document_version,
            f"sha256:{restored.metadata.sha256}",
        )
        self.assertEqual(
            restored.anchor_position(restored.sections[0]),
            SourcePosition(1, 2),
        )

    def test_current_v2_snapshot_remains_readable_with_page_only_anchor(self) -> None:
        database = source_database_path(self.run_dir)
        with sqlite3.connect(database) as connection:
            connection.execute("UPDATE source_schema SET version = 2")
            connection.execute(
                """
                CREATE TABLE source_sections_v2 (
                    ordinal INTEGER NOT NULL UNIQUE,
                    section_id TEXT PRIMARY KEY,
                    number TEXT NOT NULL,
                    title TEXT NOT NULL,
                    parent_section_id TEXT,
                    origin TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                INSERT INTO source_sections_v2
                SELECT ordinal, section_id, number, title, parent_section_id, origin
                FROM source_sections
                """
            )
            connection.execute("PRAGMA foreign_keys = OFF")
            connection.execute("DROP TABLE source_sections")
            connection.execute("ALTER TABLE source_sections_v2 RENAME TO source_sections")
            connection.commit()

        restored = load_source_document(self.run_dir)

        self.assertEqual(restored.sections[0].anchor_page_index, 1)
        self.assertIsNone(restored.sections[0].anchor_line_number)
        self.assertEqual(
            restored.anchor_position(restored.sections[0]),
            SourcePosition(1, 1),
        )

    def test_pdf_hash_mismatch_is_rejected(self) -> None:
        source_pdf_path(self.run_dir).write_bytes(FIXTURE_PDF + b"changed")

        with self.assertRaisesRegex(SourceStorageError, "SHA-256"):
            load_source_document(self.run_dir)

    def test_incompatible_schema_is_rejected(self) -> None:
        with sqlite3.connect(source_database_path(self.run_dir)) as connection:
            connection.execute("UPDATE source_schema SET version = 99")

        with self.assertRaisesRegex(SourceStorageError, "несовместима"):
            load_source_document(self.run_dir)

    def test_corrupted_database_is_rejected(self) -> None:
        source_database_path(self.run_dir).write_bytes(b"not a sqlite database")

        with self.assertRaisesRegex(SourceStorageError, "Некорректный"):
            load_source_document(self.run_dir)

    def test_missing_original_pdf_is_rejected(self) -> None:
        source_pdf_path(self.run_dir).unlink()

        with self.assertRaisesRegex(SourceStorageError, "original.pdf"):
            load_source_document(self.run_dir)

    def test_legacy_json_run_is_not_imported(self) -> None:
        legacy_run = self.run_dir / "legacy"
        legacy_run.mkdir()
        (legacy_run / "pages.json").write_text("[]", encoding="utf-8")
        (legacy_run / "structural_chunks.json").write_text("[]", encoding="utf-8")

        with self.assertRaisesRegex(SourceStorageError, "несовместим"):
            load_source_document(legacy_run)


if __name__ == "__main__":
    unittest.main()
