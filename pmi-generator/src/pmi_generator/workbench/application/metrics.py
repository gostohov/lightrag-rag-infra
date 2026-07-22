from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from .repositories import UnitOfWork


def collect_metrics(uow_factory: Callable[[], UnitOfWork]) -> dict[str, object]:
    with uow_factory() as uow:
        llm = uow.records.list_kind("llm_diagnostic")
        retrieval = uow.records.list_kind("retrieval_observation")
        cards = uow.cards.list_all()
        revisions = uow.records.list_kind("card_revision")
        sessions = uow.sessions.list_all()
        selections = uow.records.list_kind("source_selection")
        superseded = uow.records.list_kind("selection_supersession")
        reviews = uow.records.list_kind("selection_review")
        expert_cards = uow.records.list_kind("expert_card_evaluation")
        expert_ranges = uow.records.list_kind("expert_range_evaluation")
        analyst_messages = sum(
            1
            for session in sessions
            for event in uow.events.list_for(session.session_id)
            if event.event_type == "session_event"
            and event.payload.get("kind") == "Аналитик"
        )
    profiles: dict[str, int] = {}
    for item in retrieval:
        name = str(item.payload.get("profile", "неизвестно"))
        profiles[name] = profiles.get(name, 0) + 1
    llm_by_stage: dict[str, int] = {}
    for item in llm:
        name = str(item.payload.get("prompt_id", "неизвестно"))
        invocation_count = len(
            item.payload.get("invocations", []) or [item.payload]
        )
        llm_by_stage[name] = (
            llm_by_stage.get(name, 0) + invocation_count
        )
    questions = [
        " ".join(str(item.payload.get("question", "")).split()).casefold()
        for item in retrieval
    ]
    unique_questions = set(questions)
    decisions: dict[str, int] = {}
    for card in cards:
        name = card.decision.kind.value if card.decision else "нет решения"
        decisions[name] = decisions.get(name, 0) + 1
    open_gaps = sum(
        gap.status.value == "открыт"
        for card in cards
        for gap in card.gaps.values()
    )
    invocations = [
        invocation
        for item in llm
        for invocation in (
            item.payload.get("invocations", [])
            or [item.payload]
        )
    ]
    usage = [item.get("usage", {}) for item in invocations]
    return {
        "selections_total": len(selections),
        "active_selections_total": len(selections) - len(superseded),
        "superseded_selections_total": len(superseded),
        "cards_total": len(cards),
        "card_revisions_total": len(revisions),
        "card_decisions": decisions,
        "open_gaps": open_gaps,
        "selection_reviews_total": len(reviews),
        "llm_calls": len(invocations),
        "llm_calls_by_stage": llm_by_stage,
        "llm_prompt_tokens": sum(int(item.get("prompt_tokens", 0)) for item in usage),
        "llm_completion_tokens": sum(
            int(item.get("completion_tokens", 0)) for item in usage
        ),
        "llm_finish_reason_length": sum(
            item.get("finish_reason") == "length" for item in invocations
        ),
        "llm_retries": sum(int(item.payload.get("retry", 0)) for item in llm),
        "rejected_tool_calls": sum(
            len(item.payload.get("rejected_tool_calls", [])) for item in llm
        ),
        "retrieval_calls": len(retrieval),
        "retrieval_unique_questions": len(unique_questions),
        "retrieval_repeated_questions": len(questions) - len(unique_questions),
        "retrieval_profiles": profiles,
        "retrieval_question_characters": sum(
            len(str(item.payload.get("question", ""))) for item in retrieval
        ),
        "retrieval_answer_characters": sum(
            len(str(item.payload.get("answer", ""))) for item in retrieval
        ),
        "retrieval_duration_seconds": round(
            sum(float(item.payload.get("duration_seconds", 0)) for item in retrieval), 3
        ),
        "analyst_messages": analyst_messages,
        "expert_quality": "не оценено",
        "expert_card_evaluations": len(expert_cards),
        "expert_range_evaluations": len(expert_ranges),
        "time_to_first_usable_card_seconds": None,
        "active_analyst_time_seconds": None,
        "baseline_comparison": "не выполнено",
    }


def export_metrics(run_dir: Path, uow_factory: Callable[[], UnitOfWork]) -> Path:
    path = run_dir / "review" / "diagnostics" / "workbench-metrics.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(collect_metrics(uow_factory), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return path
