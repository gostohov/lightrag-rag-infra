from __future__ import annotations

import argparse
import asyncio
import ast
import importlib
import io
import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from pmi_generator.cli import build_parser, main
from pmi_generator.workbench.application.source import SourcePreparationError
from pmi_generator.workbench.domain import ExecutionMode


class WorkbenchFoundationTests(unittest.TestCase):
    def test_cli_accepts_run_with_optional_pdf_without_subcommand(self) -> None:
        args = build_parser().parse_args(["--run", "/tmp/pmi-run"])

        self.assertEqual(args.run, Path("/tmp/pmi-run"))
        self.assertIsNone(args.pdf)
        self.assertEqual(
            vars(args),
            {"run": Path("/tmp/pmi-run"), "pdf": None, "mock": False},
        )

        args = build_parser().parse_args(
            ["--pdf", "/tmp/specification.pdf", "--run", "/tmp/pmi-run"]
        )

        self.assertEqual(args.pdf, Path("/tmp/specification.pdf"))
        self.assertFalse(args.mock)

    def test_cli_accepts_mock_only_with_pdf(self) -> None:
        args = build_parser().parse_args(
            [
                "--pdf",
                "/tmp/specification.pdf",
                "--run",
                "/tmp/pmi-run",
                "--mock",
            ]
        )

        self.assertTrue(args.mock)

        with patch("sys.stderr", new=io.StringIO()) as stderr:
            with self.assertRaises(SystemExit) as raised:
                main(["--run", "/tmp/pmi-run", "--mock"])

        self.assertEqual(raised.exception.code, 2)
        self.assertIn("для --mock обязателен --pdf", stderr.getvalue())

        with patch("sys.stderr", new=io.StringIO()) as stderr:
            with self.assertRaises(SystemExit) as raised:
                main(["--pdf", "/tmp/specification.pdf", "--mock"])

        self.assertEqual(raised.exception.code, 2)
        self.assertIn("required: --run", stderr.getvalue())

    def test_all_subcommands_are_rejected_as_extra_positionals(self) -> None:
        parser = build_parser()

        for command in (
            "review",
            "extract",
            "section",
            "discover",
            "llm-discover",
            "retrieve",
            "generate",
            "render-md",
        ):
            with self.subTest(command=command), self.assertRaises(SystemExit):
                parser.parse_args(["--run", "/tmp/pmi-run", command])

    @patch("pmi_generator.cli.load_env_files")
    @patch("pmi_generator.cli.run_workbench", return_value=17)
    def test_main_calls_workbench_directly(
        self,
        run_workbench: Mock,
        load_env_files: Mock,
    ) -> None:
        result = main(["--run", "/tmp/pmi-run"])

        self.assertEqual(result, 17)
        run_workbench.assert_called_once_with(
            Path("/tmp/pmi-run"),
            pdf_path=None,
            mock=False,
        )
        load_env_files.assert_called_once()

    @patch("pmi_generator.cli.load_env_files")
    @patch("pmi_generator.cli.run_workbench", return_value=17)
    def test_main_passes_pdf_to_workbench(
        self,
        run_workbench: Mock,
        load_env_files: Mock,
    ) -> None:
        result = main(
            [
                "--pdf",
                "/tmp/specification.pdf",
                "--run",
                "/tmp/pmi-run",
            ]
        )

        self.assertEqual(result, 17)
        run_workbench.assert_called_once_with(
            Path("/tmp/pmi-run"),
            pdf_path=Path("/tmp/specification.pdf"),
            mock=False,
        )
        load_env_files.assert_called_once()

    @patch("pmi_generator.cli.load_env_files")
    @patch("pmi_generator.cli.run_workbench", return_value=17)
    def test_main_passes_mock_mode_to_workbench(
        self,
        run_workbench: Mock,
        load_env_files: Mock,
    ) -> None:
        result = main(
            [
                "--pdf",
                "/tmp/specification.pdf",
                "--run",
                "/tmp/pmi-run",
                "--mock",
            ]
        )

        self.assertEqual(result, 17)
        run_workbench.assert_called_once_with(
            Path("/tmp/pmi-run"),
            pdf_path=Path("/tmp/specification.pdf"),
            mock=True,
        )
        load_env_files.assert_called_once()

    @patch("pmi_generator.cli.load_env_files")
    @patch(
        "pmi_generator.cli.run_workbench",
        side_effect=SourcePreparationError("invalid source"),
    )
    def test_main_reports_source_error_through_argument_parser(
        self,
        run_workbench: Mock,
        load_env_files: Mock,
    ) -> None:
        with patch("sys.stderr", new=io.StringIO()) as stderr:
            with self.assertRaises(SystemExit) as raised:
                main(["--pdf", "/tmp/source.pdf", "--run", "/tmp/pmi-run"])

        self.assertEqual(raised.exception.code, 2)
        self.assertIn("invalid source", stderr.getvalue())
        run_workbench.assert_called_once()
        load_env_files.assert_called_once()

    def test_batch_modules_scripts_and_dependencies_are_absent(self) -> None:
        import pmi_generator

        package_root = Path(pmi_generator.__file__).parent
        project_root = Path(__file__).parents[1]
        pyproject = tomllib.loads(
            (project_root / "pyproject.toml").read_text(encoding="utf-8")
        )

        self.assertFalse((package_root / "cli" / "commands.py").exists())
        self.assertFalse(list((package_root / "pipeline").glob("*.py")))
        self.assertFalse(list((package_root / "domain").glob("*.py")))
        self.assertFalse((package_root / "retrieval_defaults.py").exists())
        self.assertFalse((package_root / "clients" / "openai.py").exists())
        self.assertFalse((package_root / "core" / "json_io.py").exists())
        self.assertFalse((package_root / "core" / "observability.py").exists())
        self.assertFalse(list((project_root / "scripts").glob("*")))
        dependencies = " ".join(pyproject["project"]["dependencies"]).casefold()
        self.assertIn("pypdf", dependencies)
        self.assertIn("rich", dependencies)
        self.assertNotIn("langchain-openai", dependencies)

    def test_cli_source_has_no_pipeline_dispatcher(self) -> None:
        import pmi_generator.cli as cli

        source = Path(cli.__file__).read_text(encoding="utf-8")

        self.assertNotIn("pipeline", source)
        self.assertNotIn("dispatch_command", source)

    def test_minimal_workbench_can_start_without_external_services(self) -> None:
        from pmi_generator.workbench import run_workbench
        from pmi_generator.workbench.domain import SourcePage, SourceSection
        from tests.source_fixture import write_source_snapshot

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            write_source_snapshot(
                run_dir,
                pages=(SourcePage(1, 1, ("Текст спецификации",)),),
                sections=(
                    SourceSection(
                        "page-0001",
                        "1",
                        "Страница 1",
                        ("1",),
                        (1,),
                    ),
                ),
            )
            output = io.StringIO()

            result = run_workbench(run_dir, output=output)

        self.assertEqual(result, 0)
        self.assertIn("PMI Workbench", output.getvalue())
        self.assertIn(tmp, output.getvalue())
        self.assertIn("Режим: production", output.getvalue())

    @patch(
        "pmi_generator.workbench.presentation.terminal.TerminalWorkbench"
    )
    @patch(
        "pmi_generator.workbench.application.bootstrap.build_workbench_application"
    )
    @patch(
        "pmi_generator.workbench.application.bootstrap.SqliteWorkflowRuntime"
    )
    def test_interactive_composition_provides_root_workflow_runtime(
        self,
        runtime_type: Mock,
        build_application: Mock,
        terminal_type: Mock,
    ) -> None:
        from pmi_generator.workbench.application.bootstrap import (
            run_interactive_workbench,
        )

        settings = Mock(run_dir=Path("/tmp/pmi-run"))
        document = Mock()
        workflow = runtime_type.return_value.__enter__.return_value
        facade = build_application.return_value
        terminal_type.return_value.run.return_value = 17

        result = run_interactive_workbench(settings, document)

        runtime_type.assert_called_once_with(
            Path("/tmp/pmi-run/review/workbench.sqlite3")
        )
        build_application.assert_called_once_with(settings, document, workflow)
        terminal_type.assert_called_once_with(document, facade=facade)
        terminal_type.return_value.run.assert_called_once_with()
        self.assertEqual(result, 17)

    def test_legacy_review_package_is_removed(self) -> None:
        with self.assertRaises(ModuleNotFoundError):
            importlib.import_module("pmi_generator.review")

    def test_domain_package_has_no_infrastructure_imports(self) -> None:
        import pmi_generator.workbench.domain as domain

        forbidden = ("langchain", "langgraph", "prompt_toolkit", "http", "sqlite")
        root = Path(domain.__file__).parent
        sources = "\n".join(path.read_text(encoding="utf-8") for path in root.glob("*.py"))

        for name in forbidden:
            with self.subTest(name=name):
                self.assertNotIn(name, sources)

    def test_terminal_depends_on_facade_not_concrete_flows_or_adapters(self) -> None:
        from pmi_generator.workbench.presentation import terminal

        source = Path(terminal.__file__).read_text(encoding="utf-8")
        imported = {
            alias.name
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        imported.update(
            node.module or ""
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.ImportFrom)
        )
        forbidden = (
            "pmi_generator.clients",
            "infrastructure",
            "application.card_population",
            "application.decomposition",
            "application.exporting",
            "application.gap_investigation",
            "application.llm",
            "application.refinement",
            "application.selection_review",
            "application.repositories",
            "application.workflow",
        )

        violations = sorted(
            name
            for name in imported
            if any(fragment in name for fragment in forbidden)
        )
        self.assertEqual(violations, [])

    def test_presentation_package_has_no_infrastructure_or_client_imports(self) -> None:
        import pmi_generator.workbench.presentation as presentation

        violations: list[str] = []
        for path in Path(presentation.__file__).parent.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            modules = [
                alias.name
                for node in ast.walk(tree)
                if isinstance(node, ast.Import)
                for alias in node.names
            ]
            modules.extend(
                node.module or ""
                for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom)
            )
            if any(
                "infrastructure" in module or module.startswith("pmi_generator.clients")
                for module in modules
            ):
                violations.append(str(path.relative_to(Path(presentation.__file__).parent)))

        self.assertEqual(violations, [])

    def test_presentation_does_not_capture_terminal_mouse(self) -> None:
        import pmi_generator.workbench.presentation as presentation

        captures: list[str] = []
        root = Path(presentation.__file__).parent
        for path in root.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                for keyword in node.keywords:
                    if (
                        keyword.arg == "mouse_support"
                        and isinstance(keyword.value, ast.Constant)
                        and keyword.value.value is True
                    ):
                        captures.append(str(path.relative_to(root)))

        self.assertEqual(
            sorted(set(captures)),
            [],
            "Terminal mouse selection and scrollback must remain native",
        )

    def test_full_screen_output_does_not_enable_terminal_mouse_tracking(self) -> None:
        from prompt_toolkit.data_structures import Size
        from prompt_toolkit.input.defaults import create_pipe_input
        from prompt_toolkit.output.vt100 import Vt100_Output

        from pmi_generator.workbench.presentation.result import ResultScreen

        stream = io.StringIO()
        output = Vt100_Output(
            stream,
            get_size=lambda: Size(rows=24, columns=80),
            term="xterm",
            enable_cpr=False,
        )
        with create_pipe_input() as pipe_input:
            pipe_input.send_text("\r")
            result = ResultScreen(
                "PMI Workbench",
                "Результат",
                (("close", "Закрыть"),),
                input=pipe_input,
                output=output,
            ).run()

        rendered = stream.getvalue()
        self.assertEqual(result, "close")
        for mode in ("1000", "1003", "1015", "1006"):
            with self.subTest(mode=mode):
                self.assertNotIn(f"\x1b[?{mode}h", rendered)

    def test_application_facade_uses_typed_prompt_worker_port(self) -> None:
        from pmi_generator.workbench.application import facade

        source = Path(facade.__file__).read_text(encoding="utf-8")

        self.assertIn("PromptWorkers", source)
        for concrete in (
            "DecompositionFlow",
            "CardPopulationFlow",
            "GapInvestigationFlow",
            "CardRefinementFlow",
            "SelectionReviewFlow",
            "LlmToolRuntime",
            "RetrievalPort",
        ):
            with self.subTest(concrete=concrete):
                self.assertNotIn(concrete, source)

    def test_terminal_uses_controller_backed_range_and_review_screens(self) -> None:
        from pmi_generator.workbench.presentation import terminal
        from pmi_generator.workbench.presentation.session import renderer

        source = Path(terminal.__file__).read_text(encoding="utf-8")
        renderer_source = Path(renderer.__file__).read_text(encoding="utf-8")

        self.assertIn("RangeWorkspaceScreen(", source)
        self.assertIn("SelectionReviewScreen(", source)
        self.assertNotIn("range_action_values", source)
        self.assertNotIn("OperationLine", renderer_source)

    def test_root_graph_has_no_parallel_fake_or_compatibility_handler(self) -> None:
        from pmi_generator.workbench.infrastructure.workflow import graph

        source = Path(graph.__file__).read_text(encoding="utf-8")
        package = Path(graph.__file__).parent

        self.assertFalse((package / "fakes.py").exists())
        self.assertNotIn("FakeWorkflowWorkers", source)
        self.assertNotIn("apply_workflow_command", source)

    def test_walking_skeleton_uses_production_composition_root(self) -> None:
        source = (
            Path(__file__).with_name("test_workbench_end_to_end.py")
            .read_text(encoding="utf-8")
        )

        self.assertIn("build_workbench_application(", source)
        self.assertIn("SqliteWorkflowRuntime(", source)
        for forbidden in (
            "DecompositionFlow",
            "CardPopulationFlow",
            "GapInvestigationFlow",
            "CardRefinementFlow",
            "SelectionReviewFlow",
            "DecompositionService",
            "PopulationService",
            "GapInvestigationService",
            "CardRefinementService",
            "SelectionReviewService",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)

    def test_prompt_3_uses_langchain_agent_behind_application_port(self) -> None:
        from pmi_generator.workbench.application.gap_investigation import flow
        from pmi_generator.workbench.infrastructure.langchain import gap_agent

        application_source = Path(flow.__file__).read_text(encoding="utf-8")
        adapter_source = Path(gap_agent.__file__).read_text(encoding="utf-8")

        self.assertIn("GapAgentPort", application_source)
        self.assertNotIn("create_agent", application_source)
        self.assertNotIn("for step in range", application_source)
        self.assertIn("create_agent(", adapter_source)
        self.assertIn("return_direct=True", adapter_source)


class ScriptedClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_client_returns_scripted_responses_and_records_calls(self) -> None:
        from pmi_generator.workbench.infrastructure.fakes import ScriptedClient

        client = ScriptedClient([{"result": 1}, {"result": 2}])

        first, second = await asyncio.gather(
            client.invoke({"request": "first"}),
            client.invoke({"request": "second"}),
        )

        self.assertEqual(first, {"result": 1})
        self.assertEqual(second, {"result": 2})
        self.assertEqual(
            client.calls,
            [{"request": "first"}, {"request": "second"}],
        )

    async def test_client_can_raise_scripted_error(self) -> None:
        from pmi_generator.workbench.infrastructure.fakes import ScriptedClient

        client = ScriptedClient([RuntimeError("service unavailable")])

        with self.assertRaisesRegex(RuntimeError, "service unavailable"):
            await client.invoke({"request": "fails"})


if __name__ == "__main__":
    unittest.main()
