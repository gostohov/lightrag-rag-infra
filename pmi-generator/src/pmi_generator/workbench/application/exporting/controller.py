from __future__ import annotations

from .service import FullPmiExportService


class FullExportController:
    def __init__(self, service: FullPmiExportService) -> None:
        self.service = service

    def execute(self) -> str:
        path = self.service.export_full()
        return f"Полный ПМИ сформирован:\n{path}"

