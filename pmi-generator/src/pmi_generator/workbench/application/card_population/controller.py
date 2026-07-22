from __future__ import annotations

from typing import Callable

from .models import PopulationStart


class PopulationStartController:
    def __init__(self, next_attempt_id: Callable[[], str]) -> None:
        self.next_attempt_id = next_attempt_id

    def start(self) -> PopulationStart:
        return PopulationStart(self.next_attempt_id(), None)

    def continue_without_message(self) -> PopulationStart:
        return PopulationStart(self.next_attempt_id(), None)

    def with_instruction(self, text: str) -> PopulationStart:
        if not text.strip():
            raise ValueError("Инструкция не может быть пустой")
        return PopulationStart(self.next_attempt_id(), text)
