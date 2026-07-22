from __future__ import annotations

from collections.abc import Iterable

from ..domain import TestCard
from .state import StoredRecord


def selection_review_is_current(
    review: StoredRecord | None,
    cards: Iterable[TestCard],
    selection_id: str,
) -> bool:
    selected = tuple(card for card in cards if card.selection_id == selection_id)
    revisions = {card.card_id: card.revision for card in selected}
    return bool(
        review
        and review.payload.get("selection_id", selection_id) == selection_id
        and review.payload.get("card_revisions") == revisions
        and all(card.selection_review_current for card in selected)
    )


__all__ = ["selection_review_is_current"]
