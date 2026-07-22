from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable

from .commands import CommandKind, WorkflowCommand
from .errors import WorkflowError
from .models import (
    AttemptState,
    CardWorkflowState,
    RangeReviewState,
    SkeletonState,
    WorkflowStage,
    WorkflowState,
)


Handler = Callable[[WorkflowState, dict[str, Any]], WorkflowState]


def _required(payload: dict[str, Any], name: str) -> Any:
    value = payload.get(name)
    if value is None or value == "":
        raise WorkflowError(f"В команде отсутствует обязательное поле {name}")
    return value


def _card(state: WorkflowState, card_id: str) -> CardWorkflowState:
    try:
        return state.cards[card_id]
    except KeyError as error:
        raise WorkflowError(f"Карточка {card_id} не взята в работу") from error


def _replace_card(
    state: WorkflowState,
    card_id: str,
    card: CardWorkflowState,
    *,
    invalidate_review: bool = False,
) -> WorkflowState:
    cards = dict(state.cards)
    cards[card_id] = card
    return replace(
        state,
        cards=cards,
        range_review=None if invalidate_review else state.range_review,
        export_allowed=False if invalidate_review else state.export_allowed,
    )


def _confirm_selection(state: WorkflowState, payload: dict[str, Any]) -> WorkflowState:
    if state.stage is not WorkflowStage.EMPTY:
        raise WorkflowError("Выбранный диапазон уже подтвержден")
    return replace(
        state,
        stage=WorkflowStage.SELECTION_CONFIRMED,
        selection_id=str(_required(payload, "selection_id")),
    )


def _apply_decomposition(state: WorkflowState, payload: dict[str, Any]) -> WorkflowState:
    attempt = state.active_attempt
    if (
        state.stage is not WorkflowStage.DECOMPOSING
        or attempt is None
        or attempt.kind != "prompt_1"
    ):
        raise WorkflowError("Декомпозицию можно применить только из активного Промпта 1")
    outcome = str(payload.get("outcome", "skeletons_created"))
    skeleton_ids = [str(item) for item in payload.get("skeleton_ids", [])]
    if len(skeleton_ids) != len(set(skeleton_ids)):
        raise WorkflowError("Декомпозиция должна содержать уникальные каркасы")
    if outcome == "skeletons_created" and not skeleton_ids:
        raise WorkflowError("Декомпозиция должна содержать каркасы")
    if outcome not in {
        "skeletons_created",
        "no_testable_behavior",
        "insufficient_selection",
    }:
        raise WorkflowError(f"Неизвестный исход декомпозиции: {outcome}")
    if outcome != "skeletons_created" and skeleton_ids:
        raise WorkflowError("Терминальный исход не может содержать каркасы")
    return replace(
        state,
        stage=WorkflowStage.DECOMPOSITION_REVIEW,
        skeletons={item: SkeletonState() for item in skeleton_ids},
        active_attempt=None,
        decomposition_outcome=outcome,
    )


def _take_skeleton(state: WorkflowState, payload: dict[str, Any]) -> WorkflowState:
    skeleton_id = str(_required(payload, "skeleton_id"))
    card_id = str(_required(payload, "card_id"))
    skeleton = state.skeletons.get(skeleton_id)
    if skeleton is None or skeleton.status != "pending":
        raise WorkflowError(f"Каркас {skeleton_id} уже обработан или не существует")
    if card_id in state.cards:
        raise WorkflowError(f"Карточка {card_id} уже существует")
    skeletons = dict(state.skeletons)
    skeletons[skeleton_id] = SkeletonState(status="selected", card_id=card_id)
    cards = dict(state.cards)
    cards[card_id] = CardWorkflowState()
    return replace(state, stage=WorkflowStage.CARD_WORK, skeletons=skeletons, cards=cards)


def _exclude_skeleton(state: WorkflowState, payload: dict[str, Any]) -> WorkflowState:
    skeleton_id = str(_required(payload, "skeleton_id"))
    skeleton = state.skeletons.get(skeleton_id)
    if skeleton is None or skeleton.status != "pending":
        raise WorkflowError(f"Каркас {skeleton_id} уже обработан или не существует")
    skeletons = dict(state.skeletons)
    skeletons[skeleton_id] = SkeletonState(status="excluded")
    return replace(state, skeletons=skeletons)


def _begin_attempt(state: WorkflowState, payload: dict[str, Any]) -> WorkflowState:
    if state.active_attempt:
        raise WorkflowError(f"Попытка {state.active_attempt.attempt_id} еще выполняется")
    attempt_id = str(_required(payload, "attempt_id"))
    kind = str(_required(payload, "attempt_kind"))
    card_id = str(payload["card_id"]) if payload.get("card_id") else None
    gap_id = payload.get("gap_id")
    if kind == "prompt_1":
        if state.selection_id is None or state.decomposition_outcome is not None:
            raise WorkflowError("Промпт 1 доступен только для нового подтвержденного диапазона")
        stage = WorkflowStage.DECOMPOSING
    elif kind == "prompt_2":
        if card_id is None:
            raise WorkflowError("Промпт 2 требует card_id")
        if card_id not in state.cards:
            raise WorkflowError("Промпт 2 доступен только после принятия каркаса")
        card = _card(state, card_id)
        if card.populated:
            raise WorkflowError("Промпт 2 нельзя повторно применить к заполненной карточке")
        stage = WorkflowStage.POPULATING_CARD
    elif kind == "prompt_3":
        if card_id is None:
            raise WorkflowError("Промпт 3 требует card_id")
        card = _card(state, card_id)
        if not card.populated or not gap_id or card.gap_status(str(gap_id)) != "open":
            raise WorkflowError("Промпт 3 требует открытый связанный пробел")
        stage = WorkflowStage.INVESTIGATING_GAP
    elif kind == "refinement":
        if card_id is None:
            raise WorkflowError("Доработка требует card_id")
        _card(state, card_id)
        stage = WorkflowStage.REFINING_CARD
    elif kind == "prompt_4":
        if not _all_skeletons_decided(state):
            raise WorkflowError("Промпт 4 доступен только после решений по всем каркасам")
        stage = WorkflowStage.RANGE_REVIEW
    else:
        raise WorkflowError(f"Неизвестный вид попытки: {kind}")
    return replace(
        state,
        stage=stage,
        active_attempt=AttemptState(
            attempt_id=attempt_id,
            kind=kind,
            card_id=card_id,
            gap_id=str(gap_id) if gap_id else None,
        ),
    )


def _cancel_attempt(state: WorkflowState, payload: dict[str, Any]) -> WorkflowState:
    attempt_id = str(_required(payload, "attempt_id"))
    if state.active_attempt is None or state.active_attempt.attempt_id != attempt_id:
        raise WorkflowError(f"Попытка {attempt_id} не является активной")
    return replace(
        state,
        stage=WorkflowStage.ANALYST_DECISION,
        active_attempt=None,
        cancelled_attempt_ids=state.cancelled_attempt_ids + (attempt_id,),
    )


def _fail_attempt(state: WorkflowState, payload: dict[str, Any]) -> WorkflowState:
    attempt_id = str(_required(payload, "attempt_id"))
    if state.active_attempt is None or state.active_attempt.attempt_id != attempt_id:
        raise WorkflowError(f"Попытка {attempt_id} не является активной")
    return replace(
        state,
        stage=WorkflowStage.ANALYST_DECISION,
        active_attempt=None,
        failed_attempt_ids=state.failed_attempt_ids + (attempt_id,),
    )


def _apply_attempt_result(state: WorkflowState, payload: dict[str, Any]) -> WorkflowState:
    attempt_id = str(_required(payload, "attempt_id"))
    attempt = state.active_attempt
    if attempt is None or attempt.attempt_id != attempt_id:
        raise WorkflowError(f"Попытка {attempt_id} не является активной")
    if attempt.card_id is None:
        raise WorkflowError("Результат попытки не относится к карточке")
    card = _card(state, attempt.card_id)
    revision = int(_required(payload, "revision"))
    raw_gaps = payload.get("gap_statuses", {})
    if not isinstance(raw_gaps, dict):
        raise WorkflowError("Статусы пробелов должны быть объектом")
    gaps = tuple(sorted((str(key), str(value)) for key, value in raw_gaps.items()))
    if attempt.kind == "prompt_2":
        if revision <= card.revision:
            raise WorkflowError("Промпт 2 должен увеличить ревизию карточки")
        updated = replace(card, populated=True, revision=revision, gaps=gaps)
        stage = WorkflowStage.CARD_WORK
    else:
        outcome = str(_required(payload, "outcome"))
        if attempt.gap_id not in dict(gaps):
            raise WorkflowError("Результат потерял активный пробел")
        if revision < card.revision:
            raise WorkflowError("Ревизия результата меньше текущей")
        if outcome == "resolved" and dict(gaps)[attempt.gap_id] != "resolved":
            raise WorkflowError("Закрытый результат должен закрыть активный пробел")
        updated = replace(card, revision=revision, gaps=gaps)
        stage = (
            WorkflowStage.CARD_WORK
            if outcome == "resolved"
            else WorkflowStage.ANALYST_DECISION
        )
    state = _replace_card(state, attempt.card_id, updated, invalidate_review=True)
    return replace(state, stage=stage, active_attempt=None)


def _request_analyst(state: WorkflowState, payload: dict[str, Any]) -> WorkflowState:
    if state.active_attempt is not None:
        raise WorkflowError("Сначала отмените активную попытку")
    return replace(state, stage=WorkflowStage.ANALYST_DECISION)


def _refine_card(state: WorkflowState, payload: dict[str, Any]) -> WorkflowState:
    card_id = str(_required(payload, "card_id"))
    revision = int(_required(payload, "revision"))
    card = _card(state, card_id)
    attempt = state.active_attempt
    if attempt is None or attempt.kind != "refinement" or attempt.card_id != card_id:
        raise WorkflowError("Доработка не является активной попыткой")
    outcome = str(_required(payload, "outcome"))
    if revision < card.revision or (revision == card.revision and outcome != "no_change"):
        raise WorkflowError("Ревизия доработки не соответствует исходу")
    if outcome == "no_change":
        return replace(state, stage=WorkflowStage.CARD_WORK, active_attempt=None)
    raw_gaps = payload.get("gap_statuses", {})
    if not isinstance(raw_gaps, dict):
        raise WorkflowError("Статусы пробелов должны быть объектом")
    updated = replace(
        card,
        revision=revision,
        populated=True,
        gaps=tuple(sorted((str(key), str(value)) for key, value in raw_gaps.items())),
        decision=None,
        decision_revision=None,
    )
    state = _replace_card(state, card_id, updated, invalidate_review=True)
    return replace(state, stage=WorkflowStage.CARD_WORK, active_attempt=None)


def _decide_card(state: WorkflowState, payload: dict[str, Any]) -> WorkflowState:
    card_id = str(_required(payload, "card_id"))
    decision = str(_required(payload, "decision"))
    revision = int(_required(payload, "revision"))
    card = _card(state, card_id)
    if revision != card.revision:
        raise WorkflowError("Решение относится не к текущей ревизии карточки")
    if decision not in {"include", "include_incomplete", "exclude"}:
        raise WorkflowError(f"Неизвестное решение по карточке: {decision}")
    has_open_gaps = any(status == "open" for _, status in card.gaps)
    if decision == "include" and (not card.populated or has_open_gaps):
        raise WorkflowError("Неполную карточку можно включить только явным решением")
    updated = replace(card, decision=decision, decision_revision=revision)
    state = _replace_card(state, card_id, updated, invalidate_review=True)
    return replace(state, stage=WorkflowStage.CARD_WORK)


def _all_skeletons_decided(state: WorkflowState) -> bool:
    if not state.skeletons or any(item.status == "pending" for item in state.skeletons.values()):
        return False
    return all(
        card.decision is not None and card.decision_revision == card.revision
        for card in state.cards.values()
    )


def _save_range_review(state: WorkflowState, payload: dict[str, Any]) -> WorkflowState:
    if state.active_attempt is None or state.active_attempt.kind != "prompt_4":
        raise WorkflowError("Проверку можно сохранить только из активного Промпта 4")
    review = RangeReviewState(
        revisions=tuple(sorted((card_id, card.revision) for card_id, card in state.cards.items())),
        warnings=tuple(str(item) for item in payload.get("warnings", [])),
    )
    return replace(
        state,
        stage=WorkflowStage.RANGE_REVIEWED,
        range_review=review,
        export_allowed=False,
        active_attempt=None,
    )


def _continue_with_issues(state: WorkflowState, payload: dict[str, Any]) -> WorkflowState:
    review = state.range_review
    if review is None or not review.warnings:
        raise WorkflowError("Нет замечаний, требующих решения аналитика")
    return replace(state, range_review=replace(review, accepted_with_issues=True))


def _request_export(state: WorkflowState, payload: dict[str, Any]) -> WorkflowState:
    review = state.range_review
    current_revisions = tuple(
        sorted((card_id, card.revision) for card_id, card in state.cards.items())
    )
    if review is None or review.revisions != current_revisions:
        raise WorkflowError("Для экспорта нужна актуальная проверка диапазона")
    if review.warnings and not review.accepted_with_issues:
        raise WorkflowError("Проверка диапазона содержит замечания без решения аналитика")
    return replace(state, stage=WorkflowStage.EXPORT_ALLOWED, export_allowed=True)


_HANDLERS: dict[CommandKind, Handler] = {
    CommandKind.CONFIRM_SELECTION: _confirm_selection,
    CommandKind.APPLY_DECOMPOSITION: _apply_decomposition,
    CommandKind.TAKE_SKELETON: _take_skeleton,
    CommandKind.EXCLUDE_SKELETON: _exclude_skeleton,
    CommandKind.BEGIN_ATTEMPT: _begin_attempt,
    CommandKind.CANCEL_ATTEMPT: _cancel_attempt,
    CommandKind.FAIL_ATTEMPT: _fail_attempt,
    CommandKind.APPLY_ATTEMPT_RESULT: _apply_attempt_result,
    CommandKind.REQUEST_ANALYST: _request_analyst,
    CommandKind.REFINE_CARD: _refine_card,
    CommandKind.DECIDE_CARD: _decide_card,
    CommandKind.SAVE_RANGE_REVIEW: _save_range_review,
    CommandKind.CONTINUE_WITH_ISSUES: _continue_with_issues,
    CommandKind.REQUEST_EXPORT: _request_export,
}


def apply_command(state: WorkflowState, command: WorkflowCommand) -> WorkflowState:
    """Применяет одну типизированную команду без побочных эффектов."""

    return _HANDLERS[command.kind](state, command.payload)
