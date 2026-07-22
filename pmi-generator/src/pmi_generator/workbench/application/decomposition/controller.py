from __future__ import annotations


class SkeletonDecisionController:
    def __init__(self, skeleton_ids: list[str]) -> None:
        if not skeleton_ids:
            raise ValueError("Список каркасов пуст")
        self.skeleton_ids = tuple(skeleton_ids)
        self.decisions: dict[str, str] = {}
        self.cursor = 0
        self.screen = "list"

    @property
    def current_id(self) -> str:
        return self.skeleton_ids[self.cursor]

    def open_current(self) -> None:
        self.screen = "detail"

    def record_decision(self, decision: str) -> None:
        if decision not in {"selected", "excluded"}:
            raise ValueError(f"Неизвестное решение {decision}")
        self.decisions[self.current_id] = decision
        unresolved = [
            index for index, item in enumerate(self.skeleton_ids) if item not in self.decisions
        ]
        if unresolved:
            self.cursor = unresolved[0]
        self.screen = "list"
