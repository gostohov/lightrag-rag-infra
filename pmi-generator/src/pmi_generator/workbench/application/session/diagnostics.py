from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from ...domain import CardDecisionKind, GapStatus
from ..card_history import card_snapshot_payload
from ..exporting import MarkdownCardRenderer
from ..repositories import UnitOfWork


def export_session_diagnostics(
    run_dir: Path,
    session_id: str,
    card_id: str,
    uow_factory: Callable[[], UnitOfWork],
) -> Path:
    with uow_factory() as uow:
        session = uow.sessions.get(session_id)
        if session is None:
            raise ValueError(f"Session {session_id} не найдена")
        selection_id = session.selection_id
        selection = uow.records.get("source_selection", selection_id)
        decomposition = uow.records.get("decomposition", selection_id)
        skeletons = sorted(
            (
                record
                for record in uow.records.list_kind("card_skeleton")
                if record.payload.get("selection_id") == selection_id
            ),
            key=lambda item: item.record_id,
        )
        card = uow.cards.get(card_id)
        revisions = sorted(
            (
                record
                for record in uow.records.list_kind("card_revision")
                if record.payload.get("card_id") == card_id
            ),
            key=lambda item: int(item.payload.get("revision", 0)),
        )
        review = uow.records.get("selection_review", selection_id)
        expert_card = uow.records.get("expert_card_evaluation", card_id)
        expert_range = uow.records.get("expert_range_evaluation", selection_id)
        attempts = sorted(
            (
                attempt
                for attempt in uow.attempts.list_all()
                if attempt.session_id in {session_id, selection_id}
                or attempt.session_id.startswith(f"{session_id}:")
            ),
            key=lambda item: (item.updated_at, item.attempt_id),
        )
        attempt_diagnostics = {
            attempt.attempt_id: {
                kind: record.payload
                for kind in (
                    "llm_diagnostic",
                    "session_diagnostic",
                    "recovery_diagnostic",
                )
                if (record := uow.records.get(kind, attempt.attempt_id)) is not None
            }
            for attempt in attempts
        }
        attempt_ids = {attempt.attempt_id for attempt in attempts}
        retrieval = sorted(
            (
                record
                for record in uow.records.list_kind("retrieval_observation")
                if record.record_id.partition(":")[0] in attempt_ids
            ),
            key=lambda item: item.record_id,
        )
        conflicts = sorted(
            (
                record
                for record in uow.records.list_kind("refinement_conflict")
                if record.payload.get("card_id") == card_id
            ),
            key=lambda item: item.record_id,
        )
        analyst_proposals = sorted(
            (
                record
                for record in uow.records.list_kind("analyst_answer_proposal")
                if record.payload.get("session_id") == session_id
                and record.payload.get("card_id") == card_id
            ),
            key=lambda item: item.record_id,
        )
        session_events = [
            event
            for event in uow.events.list_for(session_id)
            if event.event_type == "session_event"
        ]
        aggregate_events = {
            aggregate_id: uow.events.list_for(aggregate_id)
            for aggregate_id in (
                selection_id,
                *(item.record_id for item in skeletons),
                card_id,
                *(attempt.attempt_id for attempt in attempts),
            )
        }

    path = run_dir / "review" / "diagnostics" / f"{card_id}-session.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Диагностика эксперимента {session_id}",
        "",
        f"- Карточка: `{card_id}`",
        f"- Диапазон: `{selection_id}`",
        f"- Стадия сессии: {session.current_stage}",
        f"- Доступное продолжение: {session.payload.get('continuation', 'не определено')}",
        "",
        "## Исходный диапазон",
        "",
    ]
    if selection is None:
        lines.append("Каноническая запись диапазона отсутствует.")
    else:
        lines.extend(
            [
                f"- Раздел: `{selection.payload.get('section_id')}`",
                f"- Начало: `{_position(selection.payload.get('start'))}`",
                f"- Конец: `{_position(selection.payload.get('end'))}`",
                "",
                "```text",
                str(selection.payload.get("text", "")),
                "```",
            ]
        )

    lines.extend(["", "## Декомпозиция и решения по каркасам", ""])
    _json_block(
        lines,
        {
            "decomposition": decomposition.payload if decomposition else None,
            "skeletons": [
                {"skeleton_id": item.record_id, **item.payload}
                for item in skeletons
            ],
        },
    )

    lines.extend(["", "## Хронология сессии", ""])
    if not session_events:
        lines.append("События сессии отсутствуют.")
    for event in session_events:
        metadata = dict(event.payload.get("metadata") or {})
        lines.extend(
            [
                f"### {event.payload.get('kind')} · событие {event.sequence}",
                "",
                str(event.payload.get("text", "")),
                "",
            ]
        )
        if metadata:
            lines.append("Структурные metadata:")
            lines.append("")
            _json_block(lines, metadata)
            lines.append("")

    lines.extend(["## Интерпретации ответов аналитика", ""])
    if not analyst_proposals:
        lines.append("Предложения интерпретации отсутствуют.")
    for proposal in analyst_proposals:
        lines.extend([f"### {proposal.record_id}", ""])
        _json_block(lines, proposal.payload)

    lines.extend(["## Версии карточки", ""])
    if not revisions and card is not None:
        lines.append(
            "Исторические snapshots отсутствуют; показано только текущее состояние legacy run."
        )
        _json_block(lines, card_snapshot_payload(card, reason="текущее состояние"))
    for revision in revisions:
        lines.extend(
            [
                f"### Ревизия {revision.payload.get('revision')}",
                "",
                f"Причина snapshot: {revision.payload.get('reason')}",
                "",
            ]
        )
        _json_block(lines, revision.payload)

    lines.extend(["", "## Prompt policy, попытки и tool calls", ""])
    for attempt in attempts:
        lines.extend(
            [
                f"### {attempt.attempt_id}",
                "",
                f"- Стадия: {attempt.stage}",
                f"- Статус: {attempt.status.value}",
                f"- Обновлена: {attempt.updated_at.isoformat()}",
                f"- Policy version: {attempt.payload.get('policy_version', 'не сохранена')}",
                f"- Prompt version: {attempt.payload.get('prompt_version', 'не сохранена')}",
                f"- Fingerprint: {attempt.payload.get('fingerprint', 'не сохранён')}",
                "",
            ]
        )
        diagnostics = attempt_diagnostics.get(attempt.attempt_id, {})
        if diagnostics:
            _json_block(lines, diagnostics)

    lines.extend(["", "## Запросы LightRAG и evidence", ""])
    if not retrieval:
        lines.append("Запросы LightRAG для этой сессии отсутствуют.")
    for record in retrieval:
        lines.extend([f"### {record.record_id}", ""])
        _json_block(lines, record.payload)

    lines.extend(["", "## Детерминированные проверки", ""])
    current_revisions = (
        {card_id: card.revision}
        if card is not None
        else {}
    )
    review_current = bool(
        review
        and card is not None
        and review.payload.get("card_revisions", {}).get(card_id) == card.revision
        and card.selection_review_current
    )
    checks = {
        "all_skeletons_decided": bool(skeletons)
        and all(item.payload.get("decision") is not None for item in skeletons),
        "card_decision_current": bool(
            card
            and card.decision
            and card.decision.revision == card.revision
        ),
        "open_gap_ids": (
            [
                gap_id
                for gap_id, gap in card.gaps.items()
                if gap.status is GapStatus.OPEN
            ]
            if card
            else []
        ),
        "selection_review_current_for_card": review_current,
        "selection_review": review.payload if review else None,
        "card_revision": current_revisions,
        "non_terminal_attempt_ids": [
            attempt.attempt_id
            for attempt in attempts
            if attempt.status.value
            in {
                "создана",
                "выполняется",
                "результат готов к применению",
                "результат применяется",
            }
        ],
    }
    _json_block(lines, checks)

    lines.extend(["", "## История предметных изменений", ""])
    for aggregate_id, events in aggregate_events.items():
        if not events:
            continue
        lines.extend([f"### {aggregate_id}", ""])
        _json_block(
            lines,
            [
                {
                    "sequence": event.sequence,
                    "event_type": event.event_type,
                    "payload": event.payload,
                }
                for event in events
            ],
        )

    if conflicts:
        lines.extend(["", "## Отклонённые изменения и конфликты", ""])
        for record in conflicts:
            lines.extend([f"### {record.record_id}", ""])
            _json_block(lines, record.payload)

    lines.extend(["", "## Принятая карточка и итоговый Markdown", ""])
    if card is None:
        lines.append("Карточка отсутствует.")
    else:
        _json_block(lines, card_snapshot_payload(card, reason="итоговое состояние"))
        if (
            card.decision is not None
            and card.decision.revision == card.revision
            and card.decision.kind is not CardDecisionKind.EXCLUDE
        ):
            lines.extend(["", "```markdown", MarkdownCardRenderer().render(card).rstrip(), "```"])
        else:
            lines.append("Актуальная принятая карточка для Markdown отсутствует.")

    lines.extend(["", "## Оценка полноты диапазона", ""])
    _json_block(
        lines,
        {
            "deterministic_skeleton_coverage": checks["all_skeletons_decided"],
            "prompt_4": review.payload if review else "не выполнен",
            "expert_range_evaluation": (
                expert_range.payload if expert_range else "не оценено"
            ),
        },
    )
    lines.extend(["", "## Экспертная оценка карточки", ""])
    _json_block(
        lines,
        expert_card.payload
        if expert_card
        else {
            "result": "не оценено",
            "reasons": [],
            "requires_redesign": [],
            "comment": None,
        },
    )
    lines.extend(
        [
            "",
            "## Временная статистика",
            "",
            (
                "Полные времена стадий недоступны в текущей схеме событий. "
                "Сохранены timestamps attempts и duration каждого retrieval observation; "
                "отсутствующие значения не интерпретируются как нулевые."
            ),
        ]
    )
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return path


def export_selection_diagnostics(
    run_dir: Path,
    selection_id: str,
    uow_factory: Callable[[], UnitOfWork],
) -> Path:
    with uow_factory() as uow:
        selection = uow.records.get("source_selection", selection_id)
        decomposition = uow.records.get("decomposition", selection_id)
        review = uow.records.get("selection_review", selection_id)
        skeletons = sorted(
            (
                record
                for record in uow.records.list_kind("card_skeleton")
                if record.payload.get("selection_id") == selection_id
            ),
            key=lambda item: item.record_id,
        )
        cards = sorted(
            (
                card
                for card in uow.cards.list_all()
                if card.selection_id == selection_id
            ),
            key=lambda item: item.card_id,
        )
        sessions = {
            session.card_id: session.session_id
            for session in uow.sessions.list_all()
            if session.selection_id == selection_id
        }

    card_diagnostics: list[tuple[str, Path]] = []
    for card in cards:
        session_id = sessions.get(card.card_id)
        if session_id is None:
            continue
        card_diagnostics.append(
            (
                card.card_id,
                export_session_diagnostics(
                    run_dir,
                    session_id,
                    card.card_id,
                    uow_factory,
                ),
            )
        )

    slug = selection_id.lower().replace("_", "-")
    path = (
        run_dir
        / "review"
        / "diagnostics"
        / f"pmi-{slug}-session.md"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Диагностика диапазона {selection_id}",
        "",
        "## Состояние диапазона",
        "",
    ]
    _json_block(
        lines,
        {
            "selection": selection.payload if selection else None,
            "decomposition": decomposition.payload if decomposition else None,
            "skeletons": [
                {"skeleton_id": item.record_id, **item.payload}
                for item in skeletons
            ],
            "card_revisions": {
                card.card_id: card.revision
                for card in cards
            },
            "selection_review": review.payload if review else None,
        },
    )
    if not card_diagnostics:
        lines.extend(
            [
                "",
                "## Сессии карточек",
                "",
                "Сессии карточек отсутствуют.",
            ]
        )
    for card_id, diagnostic_path in card_diagnostics:
        lines.extend(
            [
                "",
                "---",
                "",
                f"## Сессия карточки {card_id}",
                "",
                f"Исходный файл: `{diagnostic_path}`",
                "",
                diagnostic_path.read_text(encoding="utf-8").rstrip(),
            ]
        )
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return path


def _position(value: object) -> str:
    payload = dict(value or {})  # type: ignore[arg-type]
    return f"{payload.get('page_index')}:{payload.get('line_number')}"


def _json_block(lines: list[str], payload: Any) -> None:
    lines.extend(
        [
            "```json",
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
            "```",
            "",
        ]
    )
