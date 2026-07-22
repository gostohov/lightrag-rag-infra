from __future__ import annotations

from contextlib import nullcontext
from datetime import UTC, datetime
from typing import Callable

from ...domain import (
    AnalystResolution,
    CardDecision,
    CardMutation,
    ContentField,
    Evidence,
    GapResolutionMode,
    RelatedGap,
)
from ...domain.errors import DomainError
from ...domain.schema import CARD_FIELD_PATHS
from ..card_history import save_card_revision
from ..card_population import AnalystMessage
from ..gap_specs import RELATED_GAP_KEYS, expand_related_gap_spec
from ..repositories import UnitOfWork
from ..state import StoredRecord
from .models import RefinementArguments, RefinementError, RefinementResult


class CardRefinementService:
    def __init__(
        self,
        *,
        uow_factory: Callable[[], UnitOfWork],
        next_id: Callable[[str], str],
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.uow_factory = uow_factory
        self.next_id = next_id
        self.clock = clock or (lambda: datetime.now(UTC))

    def validate_proposal(
        self,
        card_id: str,
        arguments: RefinementArguments,
        *,
        analyst_messages: tuple[AnalystMessage, ...],
        uow: UnitOfWork,
    ) -> None:
        card = uow.cards.get(card_id)
        if card is None:
            raise RefinementError(f"Карточка {card_id} не найдена")
        if arguments.outcome == "no_change":
            if arguments.updates or arguments.gaps:
                raise RefinementError(
                    "no_change не может содержать изменения"
                )
            return
        if arguments.outcome == "updated":
            if not arguments.updates or arguments.gaps:
                raise RefinementError(
                    "updated требует только явные обновления"
                )
            messages = {
                item.message_id: item
                for item in analyst_messages
            }
            paths: set[str] = set()
            for item in arguments.updates:
                if (
                    not isinstance(item, dict)
                    or set(item)
                    != {
                        "path",
                        "value",
                        "evidence_id",
                        "analyst_message_id",
                    }
                ):
                    raise RefinementError(
                        "Обновление имеет неверную структуру"
                    )
                path = str(item["path"])
                if path not in CARD_FIELD_PATHS or path in paths:
                    raise RefinementError(
                        f"Путь {path} неизвестен или повторён"
                    )
                paths.add(path)
                evidence_id = item.get("evidence_id")
                message_id = item.get("analyst_message_id")
                if bool(evidence_id) == bool(message_id):
                    raise RefinementError(
                        "Обновление требует ровно одно основание"
                    )
                if message_id:
                    message = messages.get(str(message_id))
                    if message is None or message.card_id != card.card_id:
                        raise RefinementError(
                            "Сообщение аналитика недоступно карточке"
                        )
                elif str(evidence_id) not in card.evidence:
                    raise RefinementError(
                        f"Evidence {evidence_id} не найдено в карточке"
                    )
            return
        if arguments.outcome == "gaps_created":
            if arguments.updates or not arguments.gaps:
                raise RefinementError(
                    "gaps_created требует только новые пробелы"
                )
            for item in arguments.gaps:
                if not isinstance(item, dict) or frozenset(item) not in {
                    RELATED_GAP_KEYS,
                    RELATED_GAP_KEYS | {"resolution_targets"},
                }:
                    raise RefinementError(
                        "Новый пробел имеет неверную структуру"
                    )
                allowed_paths = tuple(
                    str(value)
                    for value in item["allowed_paths"]
                )
                if (
                    not allowed_paths
                    or len(allowed_paths) != len(set(allowed_paths))
                    or any(
                        path not in CARD_FIELD_PATHS
                        for path in allowed_paths
                    )
                ):
                    raise RefinementError(
                        "Новый пробел содержит неверные пути"
                    )
                for dependency in item["dependencies"]:
                    if str(dependency) not in CARD_FIELD_PATHS:
                        raise RefinementError(
                            "Новый пробел содержит неверную зависимость"
                        )
                GapResolutionMode(str(item["resolution_mode"]))
                expand_related_gap_spec(
                    item,
                    card_id=card_id,
                    next_id=lambda prefix: f"{prefix}_VALIDATION",
                )
                if not all(
                    str(item[key]).strip()
                    for key in (
                        "question",
                        "blocking_reason",
                        "closure_criterion",
                    )
                ):
                    raise RefinementError(
                        "Новый пробел требует содержательное описание"
                    )
            return
        raise RefinementError("Неизвестный исход доработки")

    def apply(
        self,
        card_id: str,
        arguments: RefinementArguments,
        *,
        analyst_messages: tuple[AnalystMessage, ...],
        confirmation_message_id: str | None = None,
        proposal_id: str | None = None,
        expected_revision: int | None = None,
        uow: UnitOfWork | None = None,
    ) -> RefinementResult:
        if arguments.outcome not in {"updated", "gaps_created", "no_change"}:
            raise RefinementError("Неизвестный исход доработки")
        context = nullcontext(uow) if uow is not None else self.uow_factory()
        with context as active_uow:
            card = active_uow.cards.get(card_id)
            if card is None:
                raise RefinementError(f"Карточка {card_id} не найдена")
            if arguments.outcome != "no_change":
                if (
                    not proposal_id
                    or not confirmation_message_id
                    or expected_revision is None
                ):
                    raise RefinementError(
                        "Изменение карточки требует подтверждённый proposal"
                    )
                if confirmation_message_id in {
                    message.message_id for message in analyst_messages
                }:
                    raise RefinementError(
                        "Подтверждение требует отдельное сообщение аналитика"
                    )
            if (
                expected_revision is not None
                and card.revision != expected_revision
            ):
                raise RefinementError(
                    "Ревизия карточки изменилась до доработки"
                )
            try:
                if arguments.outcome == "no_change":
                    if arguments.updates or arguments.gaps:
                        raise RefinementError("no_change не может содержать изменения")
                    return RefinementResult(card_id, card.revision, "no_change", False)
                if arguments.outcome == "updated":
                    if not arguments.updates or arguments.gaps:
                        raise RefinementError("updated требует только явные обновления")
                    mutation, conflict_records = self._updated_mutation(
                        card,
                        arguments,
                        analyst_messages,
                        confirmation_message_id=confirmation_message_id,
                        proposal_id=proposal_id,
                        expected_revision=expected_revision,
                    )
                    changed = card.apply(mutation)
                    if not changed:
                        return RefinementResult(
                            card_id,
                            card.revision,
                            "no_change",
                            False,
                        )
                    active_uow.cards.save(card)
                    save_card_revision(
                        active_uow,
                        card,
                        reason="доработка по сообщению аналитика",
                    )
                    for record in conflict_records:
                        active_uow.records.save(record)
                    gap_ids: tuple[str, ...] = ()
                else:
                    if arguments.updates or not arguments.gaps:
                        raise RefinementError("gaps_created требует только новые пробелы")
                    gaps = self._build_gaps(card_id, arguments.gaps)
                    card.apply(CardMutation(gaps=gaps))
                    active_uow.cards.save(card)
                    save_card_revision(
                        active_uow,
                        card,
                        reason="созданы связанные пробелы",
                    )
                    gap_ids = tuple(item.gap_id for item in gaps)
                active_uow.events.append(
                    card_id,
                    "карточка доработана",
                    {"outcome": arguments.outcome, "revision": card.revision, "gap_ids": list(gap_ids)},
                )
            except (DomainError, KeyError, TypeError, ValueError) as error:
                if isinstance(error, RefinementError):
                    raise
                raise RefinementError(str(error)) from error
        return RefinementResult(card_id, card.revision, arguments.outcome, True, gap_ids)

    def _updated_mutation(
        self,
        card: object,
        arguments: RefinementArguments,
        messages: tuple[AnalystMessage, ...],
        *,
        confirmation_message_id: str | None = None,
        proposal_id: str | None = None,
        expected_revision: int | None = None,
    ):
        available_messages = {item.message_id: item for item in messages}
        fields: dict[str, ContentField] = {}
        evidence: dict[str, Evidence] = {}
        human_by_message: dict[str, Evidence] = {}
        resolutions: list[AnalystResolution] = []
        diagnostics: list[StoredRecord] = []
        for item in arguments.updates:
            if not isinstance(item, dict) or set(item) != {"path", "value", "evidence_id", "analyst_message_id"}:
                raise RefinementError("Обновление имеет неверную структуру")
            path = str(item["path"])
            if path not in CARD_FIELD_PATHS or path in fields:
                raise RefinementError(f"Путь {path} неизвестен или повторён")
            evidence_id = item.get("evidence_id")
            message_id = item.get("analyst_message_id")
            if bool(evidence_id) == bool(message_id):
                raise RefinementError("Обновление требует ровно одно основание")
            if message_id:
                message = available_messages.get(str(message_id))
                if message is None or message.card_id != card.card_id:
                    raise RefinementError("Сообщение аналитика недоступно карточке")
                human = human_by_message.get(message.message_id)
                if human is None:
                    human = Evidence.human_knowledge(
                        evidence_id=self.next_id("EVIDENCE"),
                        card_id=card.card_id,
                        selection_id=card.selection_id,
                        quote=message.text,
                        author=message.author,
                        message_id=message.message_id,
                        collected_at=self.clock(),
                    )
                    human_by_message[message.message_id] = human
                    evidence[human.evidence_id] = human
                evidence_id = human.evidence_id
                resolutions.append(
                    AnalystResolution(
                        resolution_id=self.next_id("RESOLUTION"),
                        card_id=card.card_id,
                        author=message.author,
                        created_at=self.clock(),
                        reason=arguments.reason or message.text,
                        target_paths=(path,),
                        evidence_ids=(human.evidence_id,),
                        source_message_id=message.message_id,
                        confirmation_message_id=str(
                            confirmation_message_id
                        ),
                        proposal_id=proposal_id,
                        expected_revision=expected_revision,
                        values=(
                            {"path": path, "value": item["value"]},
                        ),
                    )
                )
                previous = card.field(path)
                if previous.is_known and previous.value != item["value"]:
                    diagnostics.append(
                        StoredRecord(
                            "refinement_conflict",
                            self.next_id("CONFLICT"),
                            {
                                "card_id": card.card_id,
                                "path": path,
                                "previous_value": previous.value,
                                "previous_evidence_ids": list(previous.evidence_ids),
                                "expert_value": item["value"],
                                "expert_evidence_id": human.evidence_id,
                            },
                        )
                    )
            else:
                existing = card.evidence.get(str(evidence_id))
                if existing is None:
                    raise RefinementError(f"Evidence {evidence_id} не найдено в карточке")
            fields[path] = (
                ContentField.analyst_confirmed(
                    item["value"],
                    (str(evidence_id),),
                )
                if message_id
                else ContentField.confirmed(
                    item["value"],
                    (str(evidence_id),),
                )
            )
        return CardMutation(
            evidence=tuple(evidence.values()),
            fields=fields,
            resolutions=tuple(resolutions),
        ), diagnostics

    def _build_gaps(self, card_id: str, items: list[dict[str, object]]) -> tuple[RelatedGap, ...]:
        result: list[RelatedGap] = []
        for item in items:
            result.extend(
                expand_related_gap_spec(
                    item,
                    card_id=card_id,
                    next_id=self.next_id,
                )
            )
        return tuple(result)


class CardDecisionService:
    def __init__(self, *, uow_factory: Callable[[], UnitOfWork], clock: Callable[[], datetime] | None = None) -> None:
        self.uow_factory = uow_factory
        self.clock = clock or (lambda: datetime.now(UTC))

    def include(self, card_id: str, *, author: str) -> CardDecision:
        with self.uow_factory() as uow:
            card = uow.cards.get(card_id)
            if card is None:
                raise RefinementError(f"Карточка {card_id} не найдена")
            if card.is_ready:
                card.include(author=author, at=self.clock())
            else:
                open_gaps = sum(1 for gap in card.gaps.values() if gap.status.value != "закрыт")
                if open_gaps == 0:
                    raise RefinementError(
                        "Неполная карточка не содержит явного блокирующего пробела"
                    )
                card.include_incomplete(
                    author=author,
                    reason=f"Аналитик включил неполную карточку; открытых пробелов: {open_gaps}",
                    at=self.clock(),
                )
            uow.cards.save(card)
            save_card_revision(uow, card, reason="решение по карточке")
            uow.events.append(card_id, "решение по карточке", {"kind": card.decision.kind.value, "revision": card.revision})
            return card.decision

    def exclude(self, card_id: str, *, author: str, reason: str = "Исключено аналитиком") -> CardDecision:
        with self.uow_factory() as uow:
            card = uow.cards.get(card_id)
            if card is None:
                raise RefinementError(f"Карточка {card_id} не найдена")
            card.exclude(author=author, reason=reason, at=self.clock())
            uow.cards.save(card)
            save_card_revision(uow, card, reason="решение по карточке")
            uow.events.append(card_id, "решение по карточке", {"kind": card.decision.kind.value, "revision": card.revision})
            return card.decision

    def is_current(self, card_id: str, decision: CardDecision) -> bool:
        with self.uow_factory() as uow:
            card = uow.cards.get(card_id)
        return bool(card and card.decision == decision and decision.revision == card.revision)
