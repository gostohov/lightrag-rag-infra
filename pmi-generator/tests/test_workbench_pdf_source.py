from __future__ import annotations

import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path

from pmi_generator.workbench.application.source import SourceExtractionError
from pmi_generator.workbench.domain import (
    SourceDocument,
    SourcePage,
    SourcePosition,
    SourceSection,
)
from pmi_generator.workbench.infrastructure.source import PypdfSourceExtractor
from tests.pdf_fixture import write_text_pdf


CREATED_AT = datetime(2026, 7, 17, 12, 0, tzinfo=UTC)


class PypdfSourceExtractorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.extractor = PypdfSourceExtractor()

    def extract(self, path: Path):
        return self.extractor.extract(
            path,
            original_name="Specification.pdf",
            created_at=CREATED_AT,
        )

    def test_outline_builds_hierarchy_and_inclusive_page_ranges(self) -> None:
        path = self.root / "outline.pdf"
        write_text_pdf(
            path,
            ("First page\nBody", "Second page\nMore body"),
            outline=(
                ("Topic: alpha/beta?", 0, None),
                ("Appendix [free-form label]", 1, 0),
            ),
        )

        document = self.extract(path)

        self.assertEqual(document.metadata.original_name, "Specification.pdf")
        self.assertEqual(document.metadata.parser_name, "pypdf")
        self.assertEqual(len(document.metadata.sha256), 64)
        self.assertEqual(
            [
                (
                    section.number,
                    section.title,
                    section.parent_section_id,
                    section.page_indexes,
                    section.anchor_page_index,
                    section.anchor_line_number,
                    section.origin,
                )
                for section in document.sections
            ],
            [
                ("", "Topic: alpha/beta?", None, (1, 2), 1, None, "outline"),
                (
                    "",
                    "Appendix [free-form label]",
                    "outline-0001",
                    (2,),
                    2,
                    None,
                    "outline",
                ),
            ],
        )
        self.assertEqual(
            document.anchor_position(document.sections[0]),
            document.positions[0],
        )

    def test_page_only_bookmarks_on_same_page_keep_the_same_safe_anchor(self) -> None:
        path = self.root / "same-page-outline.pdf"
        write_text_pdf(
            path,
            ("First line\nSecond line",),
            outline=(
                ("First bookmark", 0, None),
                ("Second bookmark", 0, None),
            ),
        )

        document = self.extract(path)

        self.assertEqual(len(document.sections), 2)
        self.assertEqual(
            [
                (item.anchor_page_index, item.anchor_line_number)
                for item in document.sections
            ],
            [(1, None), (1, None)],
        )
        self.assertEqual(
            [document.anchor_position(item) for item in document.sections],
            [document.positions[0], document.positions[0]],
        )

    def test_page_only_anchor_skips_an_empty_page_without_fabricating_a_line(self) -> None:
        section = SourceSection(
            "outline-0001",
            "",
            "Empty destination",
            ("Empty destination",),
            (1, 2),
            origin="outline",
            anchor_page_index=1,
        )
        document = SourceDocument(
            pages=(
                SourcePage(1, "1", ()),
                SourcePage(2, "2", ("First available line",)),
            ),
            sections=(section,),
        )

        self.assertIsNone(section.anchor_line_number)
        self.assertEqual(document.anchor_position(section), SourcePosition(2, 1))

    def test_outline_rejects_duplicate_ids_and_unknown_parents(self) -> None:
        page = SourcePage(1, "1", ("Line",))
        duplicate = SourceSection("same", "", "Duplicate", ("Duplicate",), (1,))

        with self.assertRaisesRegex(ValueError, "повторяющиеся ID"):
            SourceDocument(
                pages=(page,),
                sections=(duplicate, duplicate),
            )

        with self.assertRaisesRegex(ValueError, "неизвестный parent"):
            SourceDocument(
                pages=(page,),
                sections=(
                    SourceSection(
                        "child",
                        "",
                        "Child",
                        ("Missing", "Child"),
                        (1,),
                        parent_section_id="missing",
                    ),
                ),
            )

    def test_pages_before_first_outline_destination_remain_navigable(self) -> None:
        path = self.root / "leading-pages.pdf"
        write_text_pdf(
            path,
            ("Cover", "Outline starts here", "Last page"),
            outline=(("First author bookmark", 1, None),),
            page_labels=("cover", "1", "2"),
        )

        document = self.extract(path)

        self.assertEqual(
            [
                (section.origin, section.label, section.page_indexes)
                for section in document.sections
            ],
            [
                ("page", "Страница cover", (1,)),
                ("outline", "First author bookmark", (2, 3)),
            ],
        )
        self.assertEqual(
            {
                page_index
                for section in document.sections
                for page_index in section.page_indexes
            },
            {1, 2, 3},
        )

    def test_pages_are_fallback_without_parsing_document_content(self) -> None:
        path = self.root / "pages.pdf"
        write_text_pdf(
            path,
            ("Looks like a heading", "Arbitrary content with no known grammar"),
        )

        document = self.extract(path)

        self.assertEqual(
            [section.section_id for section in document.sections],
            ["page-0001", "page-0002"],
        )
        self.assertEqual(
            document.page(2).lines,
            ("Arbitrary content with no known grammar",),
        )

    def test_arbitrary_pdf_page_labels_are_preserved_without_parsing(self) -> None:
        path = self.root / "labels.pdf"
        write_text_pdf(
            path,
            ("Cover text", "Appendix text"),
            page_labels=("cover", "A-x"),
        )

        document = self.extract(path)

        self.assertEqual(
            [page.logical_page for page in document.pages],
            ["cover", "A-x"],
        )
        self.assertEqual(
            [section.label for section in document.sections],
            ["Страница cover", "Страница A-x"],
        )

    def test_only_technical_whitespace_normalization_is_applied(self) -> None:
        path = self.root / "spacing.pdf"
        write_text_pdf(path, ("Header   remains\nword- \ncontinuation",))

        document = self.extract(path)

        self.assertEqual(
            document.page(1).lines,
            ("Header remains", "word-", "continuation"),
        )

    def test_pdf_without_text_layer_is_rejected(self) -> None:
        path = self.root / "blank.pdf"
        write_text_pdf(path, ("",))

        with self.assertRaisesRegex(SourceExtractionError, "text layer"):
            self.extract(path)

    def test_corrupted_pdf_is_rejected(self) -> None:
        path = self.root / "broken.pdf"
        path.write_bytes(b"not a pdf")

        with self.assertRaisesRegex(SourceExtractionError, "PDF signature"):
            self.extract(path)


if __name__ == "__main__":
    unittest.main()
