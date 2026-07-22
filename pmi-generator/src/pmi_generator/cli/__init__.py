from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from ..core.env import default_env_paths, load_env_files
from ..workbench import run_workbench
from ..workbench.application.source import (
    SourceExtractionError,
    SourcePreparationError,
)
from ..workbench.infrastructure.source import SourceStorageError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pmi-generator",
        description="Открыть рабочее место аналитика ПМИ.",
    )
    parser.add_argument(
        "--run",
        type=Path,
        required=True,
        help="Директория сохранённой работы",
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        help="PDF для подготовки нового run",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Запустить production Workbench с локальными mock-адаптерами",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    load_env_files(default_env_paths(Path(__file__)))
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.mock and args.pdf is None:
        parser.error("для --mock обязателен --pdf")
    try:
        return run_workbench(args.run, pdf_path=args.pdf, mock=args.mock)
    except (SourceExtractionError, SourcePreparationError, SourceStorageError) as error:
        parser.error(str(error))
