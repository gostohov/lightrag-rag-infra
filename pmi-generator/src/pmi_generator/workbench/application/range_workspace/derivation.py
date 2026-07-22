from __future__ import annotations

from ...domain import TestCard
from ..review_status import selection_review_is_current
from ..state import StoredRecord
from .models import RangeWorkspaceState, WorkspaceItem


def derive_workspace(
    selection_id: str,
    skeletons: tuple[StoredRecord, ...],
    cards: tuple[TestCard, ...],
    review: StoredRecord | None,
    sessions: dict[str, str] | None = None,
    decomposition: StoredRecord | None = None,
) -> RangeWorkspaceState:
    by_card = {card.card_id: card for card in cards if card.selection_id == selection_id}
    sessions = sessions or {}
    items: list[WorkspaceItem] = []
    all_decided = True
    included = 0
    included_incomplete = 0
    excluded = 0

    for skeleton in sorted(skeletons, key=lambda item: item.record_id):
        payload = skeleton.payload
        if payload.get("selection_id") != selection_id:
            continue
        decision = payload.get("decision")
        card_id = str(payload.get("card_id") or "") or None
        title = str(payload.get("title") or skeleton.record_id)
        if decision is None:
            all_decided = False
            items.append(WorkspaceItem(skeleton.record_id, title, "требует решения", "warning"))
            continue
        if decision == "excluded":
            excluded += 1
            items.append(WorkspaceItem(skeleton.record_id, title, "каркас исключён", "muted"))
            continue
        card = by_card.get(card_id or "")
        if card is None:
            all_decided = False
            items.append(WorkspaceItem(skeleton.record_id, title, "карточка не найдена", "error", card_id))
            continue
        card_decision = card.decision
        if card_decision is None or card_decision.revision != card.revision:
            all_decided = False
            readiness = "готова" if card.is_ready else "неполная"
            status = f"{readiness}, требуется решение"
            style = "ready" if card.is_ready else "warning"
        elif card_decision.kind.value == "включить":
            included += 1
            status, style = "включена", "success"
        elif card_decision.kind.value == "включить неполной":
            included_incomplete += 1
            status, style = "включена неполной", "warning"
        else:
            excluded += 1
            status, style = "исключена", "muted"
        items.append(
            WorkspaceItem(
                skeleton.record_id,
                card.title,
                status,
                style,
                card.card_id,
                card.revision,
                sessions.get(card.card_id),
            )
        )

    review_current = selection_review_is_current(
        review,
        by_card.values(),
        selection_id,
    )
    review_stale = review is not None and not review_current
    terminal_status = {
        "no_testable_behavior": "нет тестируемого поведения",
        "insufficient_selection": "недостаточно выбранного текста",
    }.get(str(decomposition.payload.get("outcome"))) if decomposition else None
    terminal_explanation = (
        str(decomposition.payload.get("explanation") or "")
        if decomposition
        else ""
    )
    return RangeWorkspaceState(
        selection_id=selection_id,
        items=tuple(items),
        can_review=bool(items) and all_decided and len(items) == len(tuple(item for item in skeletons if item.payload.get("selection_id") == selection_id)),
        review_current=review_current,
        review_stale=review_stale,
        included=included,
        included_incomplete=included_incomplete,
        excluded=excluded,
        terminal_status=terminal_status,
        terminal_explanation=terminal_explanation,
    )
