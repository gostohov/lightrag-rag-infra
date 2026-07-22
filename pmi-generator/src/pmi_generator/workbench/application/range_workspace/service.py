from __future__ import annotations

from typing import Callable

from ..repositories import UnitOfWork
from .derivation import derive_workspace
from .models import RangeWorkspaceState


class RangeWorkspaceService:
    def __init__(self, *, uow_factory: Callable[[], UnitOfWork]) -> None:
        self.uow_factory = uow_factory

    def load(
        self,
        selection_id: str,
        *,
        uow: UnitOfWork | None = None,
    ) -> RangeWorkspaceState:
        if uow is not None:
            return self._load(selection_id, uow)
        with self.uow_factory() as uow:
            return self._load(selection_id, uow)

    @staticmethod
    def _load(selection_id: str, uow: UnitOfWork) -> RangeWorkspaceState:
        skeletons = tuple(uow.records.list_kind("card_skeleton"))
        cards = tuple(uow.cards.list_all())
        review = uow.records.get("selection_review", selection_id)
        decomposition = uow.records.get("decomposition", selection_id)
        sessions = {
            item.card_id: item.session_id
            for item in uow.sessions.list_all()
            if item.selection_id == selection_id
        }
        return derive_workspace(
            selection_id,
            skeletons,
            cards,
            review,
            sessions,
            decomposition,
        )

    def session_for_card(self, selection_id: str, card_id: str) -> str:
        with self.uow_factory() as uow:
            matches = [
                item
                for item in uow.sessions.list_all()
                if item.selection_id == selection_id and item.card_id == card_id
            ]
        if not matches:
            raise ValueError("Для карточки ещё нет сохранённой session")
        return max(matches, key=lambda item: item.updated_at).session_id
