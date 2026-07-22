from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from pmi_generator.workbench import run_workbench
from pmi_generator.workbench.application.source import (
    SourceExtractionError,
    SourcePreparationError,
    SourceRunPreparer,
)
from pmi_generator.workbench.infrastructure.source import (
    PypdfSourceExtractor,
    SourceStorageError,
    SqliteSourceSnapshotRepository,
    load_source_document,
    source_database_path,
    source_pdf_path,
)
from pmi_generator.workbench.infrastructure.storage import workbench_database_path
from pmi_generator.workbench.domain import ExecutionMode
from tests.pdf_fixture import write_text_pdf


class SourceRunPreparerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.pdf = self.root / "input.pdf"
        write_text_pdf(self.pdf, ("Arbitrary source text",))
        self.extractor = Mock(wraps=PypdfSourceExtractor())
        self.repository = SqliteSourceSnapshotRepository()
        self.preparer = SourceRunPreparer(
            extractor=self.extractor,
            repository=self.repository,
        )

    def staging_paths(self) -> tuple[Path, ...]:
        return tuple(self.root.glob(".*.preparing-*"))

    def test_new_run_is_published_only_after_snapshot_validation(self) -> None:
        run_dir = self.root / "new-run"

        document = self.preparer.prepare(run_dir, self.pdf)

        self.assertTrue(source_pdf_path(run_dir).is_file())
        self.assertTrue(source_database_path(run_dir).is_file())
        self.assertEqual(load_source_document(run_dir).metadata, document.metadata)
        self.assertEqual(document.metadata.original_name, self.pdf.name)
        self.assertEqual(self.staging_paths(), ())

    def test_existing_empty_target_is_replaced_by_complete_run(self) -> None:
        run_dir = self.root / "empty-run"
        run_dir.mkdir()

        self.preparer.prepare(run_dir, self.pdf)

        self.assertEqual(
            sorted(path.name for path in run_dir.iterdir()),
            ["source"],
        )
        self.assertEqual(self.staging_paths(), ())

    def test_same_pdf_resumes_without_extraction(self) -> None:
        run_dir = self.root / "resume-run"
        first = self.preparer.prepare(run_dir, self.pdf)
        self.extractor.reset_mock()

        second = self.preparer.prepare(run_dir, self.pdf)

        self.extractor.extract.assert_not_called()
        self.assertEqual(second.metadata, first.metadata)

    def test_execution_mode_is_persisted_and_required_on_resume(self) -> None:
        run_dir = self.root / "mock-run"

        created = self.preparer.prepare(
            run_dir,
            self.pdf,
            execution_mode=ExecutionMode.MOCK,
        )
        loaded = load_source_document(run_dir)

        self.assertIs(created.metadata.execution_mode, ExecutionMode.MOCK)
        self.assertIs(loaded.metadata.execution_mode, ExecutionMode.MOCK)
        before = source_database_path(run_dir).read_bytes()
        with self.assertRaisesRegex(SourcePreparationError, "режиме mock"):
            self.preparer.prepare(
                run_dir,
                self.pdf,
                execution_mode=ExecutionMode.PRODUCTION,
            )
        self.assertEqual(source_database_path(run_dir).read_bytes(), before)

    def test_mock_run_requires_matching_mode_even_without_pdf(self) -> None:
        run_dir = self.root / "mock-run"
        self.preparer.prepare(
            run_dir,
            self.pdf,
            execution_mode=ExecutionMode.MOCK,
        )

        with self.assertRaisesRegex(SourcePreparationError, "режиме mock"):
            self.preparer.prepare(run_dir)

    def test_different_pdf_is_rejected_without_changing_run(self) -> None:
        run_dir = self.root / "conflict-run"
        self.preparer.prepare(run_dir, self.pdf)
        before_pdf = source_pdf_path(run_dir).read_bytes()
        before_db = source_database_path(run_dir).read_bytes()
        other_pdf = self.root / "other.pdf"
        write_text_pdf(other_pdf, ("Different source text",))

        with self.assertRaisesRegex(SourcePreparationError, "не совпадает"):
            self.preparer.prepare(run_dir, other_pdf)

        self.assertEqual(source_pdf_path(run_dir).read_bytes(), before_pdf)
        self.assertEqual(source_database_path(run_dir).read_bytes(), before_db)
        self.assertEqual(self.staging_paths(), ())

    def test_run_without_pdf_loads_existing_snapshot(self) -> None:
        run_dir = self.root / "existing-run"
        expected = self.preparer.prepare(run_dir, self.pdf)
        self.extractor.reset_mock()

        actual = self.preparer.prepare(run_dir)

        self.assertEqual(actual.metadata, expected.metadata)
        self.extractor.extract.assert_not_called()

    def test_invalid_inputs_leave_nonexistent_target_absent(self) -> None:
        inputs = {
            "missing": self.root / "missing.pdf",
            "directory": self.root,
            "corrupt": self.root / "corrupt.pdf",
            "blank": self.root / "blank.pdf",
        }
        inputs["corrupt"].write_bytes(b"not a pdf")
        write_text_pdf(inputs["blank"], ("",))

        for name, pdf_path in inputs.items():
            run_dir = self.root / f"run-{name}"
            with self.subTest(name=name), self.assertRaises(
                (SourceExtractionError, SourcePreparationError)
            ):
                self.preparer.prepare(run_dir, pdf_path)
            self.assertFalse(run_dir.exists())
            self.assertEqual(self.staging_paths(), ())

    def test_extraction_failure_leaves_existing_empty_target_empty(self) -> None:
        run_dir = self.root / "empty-on-error"
        run_dir.mkdir()
        broken = self.root / "broken.pdf"
        broken.write_bytes(b"not a pdf")

        with self.assertRaises(SourceExtractionError):
            self.preparer.prepare(run_dir, broken)

        self.assertEqual(tuple(run_dir.iterdir()), ())
        self.assertEqual(self.staging_paths(), ())

    def test_persistence_failure_removes_staging_and_target(self) -> None:
        run_dir = self.root / "persistence-error"
        repository = Mock(wraps=self.repository)
        repository.save.side_effect = SourceStorageError("write failed")
        preparer = SourceRunPreparer(
            extractor=PypdfSourceExtractor(),
            repository=repository,
        )

        with self.assertRaisesRegex(SourceStorageError, "write failed"):
            preparer.prepare(run_dir, self.pdf)

        self.assertFalse(run_dir.exists())
        self.assertEqual(self.staging_paths(), ())

    def test_publication_failure_keeps_empty_target_and_removes_staging(self) -> None:
        run_dir = self.root / "publication-error"
        run_dir.mkdir()

        with patch(
            "pmi_generator.workbench.application.source.preparation.os.replace",
            side_effect=OSError("rename failed"),
        ):
            with self.assertRaisesRegex(SourcePreparationError, "rename failed"):
                self.preparer.prepare(run_dir, self.pdf)

        self.assertTrue(run_dir.is_dir())
        self.assertEqual(tuple(run_dir.iterdir()), ())
        self.assertEqual(self.staging_paths(), ())

    def test_nonempty_unknown_and_legacy_runs_are_rejected(self) -> None:
        for name, file_name in (
            ("unknown", "unexpected.txt"),
            ("legacy", "pages.json"),
        ):
            run_dir = self.root / name
            run_dir.mkdir()
            (run_dir / file_name).write_text("{}", encoding="utf-8")

            with self.subTest(name=name), self.assertRaises(SourceStorageError):
                self.preparer.prepare(run_dir, self.pdf)

            self.assertFalse(workbench_database_path(run_dir).exists())


class WorkbenchPreparationCompositionTests(unittest.TestCase):
    @patch(
        "pmi_generator.workbench.presentation.operation.TerminalOperationRunner"
    )
    @patch("pmi_generator.workbench._interactive_terminal", return_value=True)
    @patch("pmi_generator.workbench.SourceRunPreparer")
    def test_interactive_pdf_preparation_uses_non_interruptible_loader(
        self,
        preparer_type: Mock,
        interactive_terminal: Mock,
        runner_type: Mock,
    ) -> None:
        from pmi_generator.workbench import _prepare_source_document

        run_dir = Path("/tmp/run")
        pdf_path = Path("/tmp/source.pdf")
        expected = Mock()
        preparer_type.return_value.prepare.return_value = expected
        runner_type.return_value.run_sync.side_effect = (
            lambda label, operation, **kwargs: operation()
        )

        actual = _prepare_source_document(run_dir, pdf_path, output=None)

        self.assertIs(actual, expected)
        interactive_terminal.assert_called_once_with(None)
        preparer_type.return_value.prepare.assert_called_once_with(
            run_dir,
            pdf_path,
            execution_mode=ExecutionMode.PRODUCTION,
        )
        call = runner_type.return_value.run_sync.call_args
        self.assertEqual(call.args[0], "Подготовка источника из PDF")
        self.assertTrue(call.kwargs["full_screen"])
        self.assertFalse(call.kwargs["interruptible"])
        context = call.kwargs["context"](80, 20)
        self.assertIn(str(pdf_path), context)
        self.assertIn(str(run_dir), context)
        self.assertIn("production", context)

    @patch(
        "pmi_generator.workbench.presentation.operation.TerminalOperationRunner"
    )
    def test_pdf_mode_prepares_source_before_workbench_database(
        self,
        runner_type: Mock,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pdf = root / "source.pdf"
            run_dir = root / "run"
            write_text_pdf(pdf, ("Source text",))
            output = io.StringIO()

            result = run_workbench(run_dir, pdf_path=pdf, output=output)

            self.assertEqual(result, 0)
            self.assertTrue(source_database_path(run_dir).is_file())
            self.assertTrue(workbench_database_path(run_dir).is_file())
            self.assertIn("source.pdf", output.getvalue())
            self.assertIn("Режим: production", output.getvalue())
            runner_type.assert_not_called()

    def test_wrong_execution_mode_is_rejected_before_workbench_database(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pdf = root / "source.pdf"
            run_dir = root / "run"
            write_text_pdf(pdf, ("Source text",))

            run_workbench(
                run_dir,
                pdf_path=pdf,
                output=io.StringIO(),
                mock=True,
            )
            database_path = workbench_database_path(run_dir)
            before = database_path.read_bytes()

            with self.assertRaisesRegex(SourcePreparationError, "режиме mock"):
                run_workbench(
                    run_dir,
                    pdf_path=pdf,
                    output=io.StringIO(),
                )

            self.assertEqual(database_path.read_bytes(), before)

    def test_production_run_cannot_be_reopened_as_mock(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pdf = root / "source.pdf"
            run_dir = root / "run"
            write_text_pdf(pdf, ("Source text",))
            run_workbench(run_dir, pdf_path=pdf, output=io.StringIO())
            database_path = workbench_database_path(run_dir)
            before = database_path.read_bytes()

            with self.assertRaisesRegex(
                SourcePreparationError,
                "режиме production",
            ):
                run_workbench(
                    run_dir,
                    pdf_path=pdf,
                    output=io.StringIO(),
                    mock=True,
                )

            self.assertEqual(database_path.read_bytes(), before)

    def test_mock_api_requires_pdf_before_creating_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "missing-run"

            with self.assertRaisesRegex(SourcePreparationError, "обязателен --pdf"):
                run_workbench(run_dir, output=io.StringIO(), mock=True)

            self.assertFalse(run_dir.exists())

    def test_invalid_source_is_rejected_before_workbench_database(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            (run_dir / "pages.json").write_text("{}", encoding="utf-8")

            with self.assertRaises(SourceStorageError):
                run_workbench(run_dir, output=io.StringIO())

            self.assertFalse(workbench_database_path(run_dir).exists())

    def test_pdf_conflict_does_not_change_existing_workbench_database(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first_pdf = root / "first.pdf"
            second_pdf = root / "second.pdf"
            run_dir = root / "run"
            write_text_pdf(first_pdf, ("First source",))
            write_text_pdf(second_pdf, ("Second source",))
            run_workbench(run_dir, pdf_path=first_pdf, output=io.StringIO())
            database_path = workbench_database_path(run_dir)
            before = database_path.read_bytes()

            with self.assertRaisesRegex(SourcePreparationError, "не совпадает"):
                run_workbench(
                    run_dir,
                    pdf_path=second_pdf,
                    output=io.StringIO(),
                )

            self.assertEqual(database_path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
