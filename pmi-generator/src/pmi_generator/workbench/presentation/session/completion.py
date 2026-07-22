from __future__ import annotations

from collections.abc import Iterable

from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.document import Document


class SlashCommandCompleter(Completer):
    def __init__(
        self,
        commands: list[str],
        descriptions: dict[str, str] | None = None,
    ) -> None:
        self.commands = tuple(commands)
        self.descriptions = descriptions or {}

    def suggestions(self, text: str) -> tuple[str, ...]:
        if not text.startswith("/") or any(character.isspace() for character in text):
            return ()
        return tuple(command for command in self.commands if command.startswith(text))

    def get_completions(
        self,
        document: Document,
        complete_event: object,
    ) -> Iterable[Completion]:
        text = document.text_before_cursor
        for command in self.suggestions(text):
            yield Completion(
                command,
                start_position=-len(text),
                display_meta=self.descriptions.get(command, ""),
            )
