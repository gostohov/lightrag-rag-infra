from __future__ import annotations

from contextlib import nullcontext
from datetime import UTC, datetime
from typing import Callable

from ...domain import (
    AnalystResolution,
    CardMutation,
    ContentField,
    EpistemicStatus,
    Evidence,
    GapClosureEvaluation,
    GapClosureOutcome,
    RelatedGap,
    TestCard,
)
from ...domain.errors import DomainError
from ..card_history import save_card_revision
from ..card_population import AnalystMessage
from ..repositories import UnitOfWork
from ..state import StoredRecord
from .models import (
    AnalystConfirmation,
    GapArguments,
    GapInvestigationError,
    GapInvestigationResult,
)


class GapInvestigationService:
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

    def validate_submission(
        self,
        card_id: str,
        gap_id: str,
        arguments: GapArguments,
        *,
        available_evidence: tuple[Evidence, ...],
        analyst_messages: tuple[AnalystMessage, ...],
        analyst_confirmation: AnalystConfirmation | None = None,
    ) -> None:
        with self.uow_factory() as uow:
            card = uow.cards.get(card_id)
        if card is None or gap_id not in card.gaps:
            raise GapInvestigationError("Карточка или пробел не найдены")
        self._validate_context(
            card,
            card.gaps[gap_id].allowed_paths,
            arguments,
            available_evidence,
            analyst_messages,
            gap_id,
            analyst_confirmation,
        )

    def apply(
        self,
        card_id: str,
        gap_id: str,
        arguments: GapArguments,
        *,
        available_evidence: tuple[Evidence, ...],
        analyst_messages: tuple[AnalystMessage, ...],
        analyst_confirmation: AnalystConfirmation | None = None,
        uow: UnitOfWork | None = None,
    ) -> GapInvestigationResult:
        if arguments.outcome not in {
            "resolved",
            "partially_resolved",
            "not_found",
            "contradiction",
        }:
            raise GapInvestigationError("Неизвестный исход исследования")
        context = nullcontext(uow) if uow is not None else self.uow_factory()
        with context as active_uow:
            card = active_uow.cards.get(card_id)
            if card is None or gap_id not in card.gaps:
                raise GapInvestigationError("Карточка или пробел не найдены")
            gap = card.gaps[gap_id]
            try:
                self._validate_context(
                    card,
                    gap.allowed_paths,
                    arguments,
                    available_evidence,
                    analyst_messages,
                    gap_id,
                    analyst_confirmation,
                )
                closure_evaluation: GapClosureEvaluation | None = None
                if arguments.outcome in {
                    "resolved",
                    "partially_resolved",
                }:
                    closure_evaluation = self._closure_evaluation(
                        gap,
                        arguments,
                    )
                    if (
                        arguments.outcome == "resolved"
                        and closure_evaluation.outcome
                        is not GapClosureOutcome.SATISFIED
                    ):
                        raise GapInvestigationError(
                            "Подтверждённые значения не выполняют "
                            "closure contract"
                        )
                    if (
                        arguments.outcome == "partially_resolved"
                        and closure_evaluation.outcome
                        is GapClosureOutcome.SATISFIED
                    ):
                        raise GapInvestigationError(
                            "Выполненный closure contract требует outcome resolved"
                        )
                    if arguments.outcome == "partially_resolved":
                        if set(arguments.unknown_fields) != set(
                            closure_evaluation.remaining_paths
                        ):
                            raise GapInvestigationError(
                                "Остаточные поля не совпадают с closure contract"
                            )
                        if str(arguments.missing_fact).strip() != " ".join(
                            closure_evaluation.remaining_questions
                        ):
                            raise GapInvestigationError(
                                "Остаточный вопрос не совпадает с closure contract"
                            )
                    mutation = self._resolved_mutation(
                        card,
                        gap,
                        arguments,
                        available_evidence,
                        analyst_messages,
                        analyst_confirmation,
                    )
                    card.apply_gap_progress(
                        gap_id,
                        mutation,
                        closure_satisfied_paths=(
                            closure_evaluation.satisfied_paths
                        ),
                        resolve=(
                            closure_evaluation.outcome
                            is GapClosureOutcome.SATISFIED
                        ),
                    )
                    active_uow.cards.save(card)
                    save_card_revision(
                        active_uow,
                        card,
                        reason=(
                            f"разрешён пробел {gap_id}"
                            if arguments.outcome == "resolved"
                            else f"частично заполнен пробел {gap_id}"
                        ),
                    )
                elif arguments.outcome == "not_found":
                    self._validate_not_found(gap.allowed_paths, arguments)
                else:
                    self._validate_contradiction(arguments, available_evidence)
                active_uow.records.save(
                    StoredRecord(
                        "gap_result",
                        f"{card_id}:{gap_id}",
                        {
                            "outcome": arguments.outcome,
                            "summary": arguments.summary,
                            "unknown_fields": list(arguments.unknown_fields),
                            "missing_fact": arguments.missing_fact,
                            "contradictions": list(arguments.contradictions),
                            "closure_evaluation": (
                                {
                                    "outcome": (
                                        closure_evaluation.outcome.value
                                    ),
                                    "satisfied_paths": list(
                                        closure_evaluation.satisfied_paths
                                    ),
                                    "remaining_paths": list(
                                        closure_evaluation.remaining_paths
                                    ),
                                    "remaining_questions": list(
                                        closure_evaluation.remaining_questions
                                    ),
                                }
                                if closure_evaluation is not None
                                else None
                            ),
                            "revision": card.revision,
                        },
                    )
                )
                active_uow.events.append(card_id, "исследование пробела завершено", {"gap_id": gap_id, "outcome": arguments.outcome})
            except (DomainError, KeyError, TypeError, ValueError) as error:
                if isinstance(error, GapInvestigationError):
                    raise
                raise GapInvestigationError(str(error)) from error
        return GapInvestigationResult(
            card_id,
            gap_id,
            arguments.outcome,
            card.revision,
            remaining_questions=(
                closure_evaluation.remaining_questions
                if closure_evaluation is not None
                else ()
            ),
        )

    def _validate_context(
        self,
        card: TestCard,
        allowed_paths: tuple[str, ...],
        arguments: GapArguments,
        available_evidence: tuple[Evidence, ...],
        analyst_messages: tuple[AnalystMessage, ...],
        gap_id: str,
        analyst_confirmation: AnalystConfirmation | None,
    ) -> None:
        if arguments.outcome == "not_found":
            self._validate_not_found(allowed_paths, arguments)
            return
        if arguments.outcome == "contradiction":
            self._validate_contradiction(arguments, available_evidence)
            return

        available = {item.evidence_id: item for item in available_evidence}
        messages = {item.message_id: item for item in analyst_messages}
        updated_paths: set[str] = set()
        for item in arguments.updates:
            expected = {"path", "value", "evidence_id", "analyst_message_id"}
            if not isinstance(item, dict) or set(item) != expected:
                raise GapInvestigationError("Обновление пробела имеет неверную структуру")
            path = str(item["path"])
            if path not in allowed_paths or path in updated_paths:
                raise GapInvestigationError(f"Путь {path} не разрешён или повторён")
            updated_paths.add(path)
            evidence_id = item.get("evidence_id")
            message_id = item.get("analyst_message_id")
            if bool(evidence_id) == bool(message_id):
                raise GapInvestigationError("Обновление требует ровно одно основание")
            if evidence_id:
                evidence = available.get(str(evidence_id))
                if evidence is None or not evidence.supports(
                    card.card_id,
                    card.selection_id,
                ):
                    raise GapInvestigationError(
                        f"Неизвестное или недоступное evidence {evidence_id}"
                    )
            else:
                message = messages.get(str(message_id))
                if message is None or message.card_id != card.card_id:
                    raise GapInvestigationError(
                        f"Сообщение аналитика {message_id} недоступно карточке"
                    )

        if arguments.outcome == "resolved":
            missing_paths = {
                path
                for path in allowed_paths
                if card.field(path).status is EpistemicStatus.UNKNOWN
            } - updated_paths
            if missing_paths:
                raise GapInvestigationError(
                    "resolved не заполняет неизвестные пути gap: "
                    + ", ".join(sorted(missing_paths))
                )
        analyst_updates = [
            {
                "path": str(item["path"]),
                "value": item["value"],
            }
            for item in arguments.updates
            if item.get("analyst_message_id")
        ]
        if not analyst_updates:
            if analyst_confirmation is not None:
                raise GapInvestigationError(
                    "Source update не принимает подтверждение аналитика"
                )
            return
        if analyst_confirmation is None:
            raise GapInvestigationError(
                "Обновление из сообщения аналитика требует подтверждение"
            )
        if (
            analyst_confirmation.gap_id != gap_id
            or analyst_confirmation.expected_revision != card.revision
            or tuple(analyst_updates) != analyst_confirmation.values
        ):
            raise GapInvestigationError(
                "Подтверждение не соответствует gap, revision или значениям"
            )
        message_ids = {
            str(item["analyst_message_id"])
            for item in arguments.updates
            if item.get("analyst_message_id")
        }
        if message_ids != {analyst_confirmation.source_message_id}:
            raise GapInvestigationError(
                "Подтверждение относится к другому сообщению аналитика"
            )
        if analyst_confirmation.source_message_id not in messages:
            raise GapInvestigationError(
                "Исходное сообщение подтверждения недоступно карточке"
            )

    def _resolved_mutation(
        self,
        card: TestCard,
        gap: RelatedGap,
        arguments: GapArguments,
        available_evidence: tuple[Evidence, ...],
        analyst_messages: tuple[AnalystMessage, ...],
        analyst_confirmation: AnalystConfirmation | None,
    ) -> CardMutation:
        if not arguments.updates or arguments.contradictions:
            raise GapInvestigationError(
                "Подтверждённый исход требует обновления без противоречий"
            )
        available = {item.evidence_id: item for item in available_evidence}
        messages = {item.message_id: item for item in analyst_messages}
        evidence_to_add: dict[str, Evidence] = {}
        human_by_message: dict[str, Evidence] = {}
        fields: dict[str, ContentField] = {}
        analyst_paths: list[str] = []
        source_confirmed = all(
            bool(item.get("evidence_id"))
            for item in arguments.updates
        )
        normalized = gap.closure_contract.normalize_values(
            {
                str(item["path"]): item["value"]
                for item in arguments.updates
            },
            source_confirmed=source_confirmed,
        )
        for item in arguments.updates:
            expected = {"path", "value", "evidence_id", "analyst_message_id"}
            if not isinstance(item, dict) or set(item) != expected:
                raise GapInvestigationError("Обновление пробела имеет неверную структуру")
            path = str(item["path"])
            if path not in gap.allowed_paths or path in fields:
                raise GapInvestigationError(f"Путь {path} не разрешён или повторён")
            evidence_id = item.get("evidence_id")
            message_id = item.get("analyst_message_id")
            if bool(evidence_id) == bool(message_id):
                raise GapInvestigationError("Обновление требует ровно одно основание")
            if message_id:
                message = messages.get(str(message_id))
                if message is None or message.card_id != card.card_id:
                    raise GapInvestigationError("Сообщение аналитика недоступно карточке")
                evidence = human_by_message.get(message.message_id)
                if evidence is None:
                    evidence = Evidence.human_knowledge(
                        evidence_id=self.next_id("EVIDENCE"),
                        card_id=card.card_id,
                        selection_id=card.selection_id,
                        quote=message.text,
                        author=message.author,
                        message_id=message.message_id,
                        collected_at=self.clock(),
                    )
                    human_by_message[message.message_id] = evidence
                    evidence_to_add[evidence.evidence_id] = evidence
                evidence_id = evidence.evidence_id
                analyst_paths.append(path)
            else:
                evidence = available.get(str(evidence_id))
                if evidence is None or not evidence.supports(card.card_id, card.selection_id):
                    raise GapInvestigationError(f"Неизвестное или недоступное evidence {evidence_id}")
                if evidence.evidence_id not in card.evidence:
                    evidence_to_add[evidence.evidence_id] = evidence
            fields[path] = (
                ContentField.analyst_confirmed(
                    normalized[path],
                    (str(evidence_id),),
                )
                if message_id
                else ContentField.confirmed(
                    normalized[path],
                    (str(evidence_id),),
                )
            )
        resolutions: tuple[AnalystResolution, ...] = ()
        if analyst_paths:
            if analyst_confirmation is None:
                raise GapInvestigationError(
                    "Обновление из сообщения аналитика требует подтверждение"
                )
            message = messages[analyst_confirmation.source_message_id]
            evidence_ids = tuple(
                dict.fromkeys(
                    evidence_id
                    for path in analyst_paths
                    for evidence_id in fields[path].evidence_ids
                )
            )
            resolutions = (
                AnalystResolution(
                    resolution_id=self.next_id("RESOLUTION"),
                    card_id=card.card_id,
                    author=message.author,
                    created_at=self.clock(),
                    reason=(
                        "Интерпретация ответа аналитика явно подтверждена"
                    ),
                    target_paths=tuple(analyst_paths),
                    evidence_ids=evidence_ids,
                    source_message_id=analyst_confirmation.source_message_id,
                    confirmation_message_id=(
                        analyst_confirmation.confirmation_message_id
                    ),
                    proposal_id=analyst_confirmation.proposal_id,
                    gap_id=analyst_confirmation.gap_id,
                    expected_revision=analyst_confirmation.expected_revision,
                    values=tuple(
                        {
                            "path": path,
                            "value": fields[path].value,
                        }
                        for path in analyst_paths
                    ),
                ),
            )
        return CardMutation(
            evidence=tuple(evidence_to_add.values()),
            fields=fields,
            resolutions=resolutions,
        )

    @staticmethod
    def _closure_evaluation(
        gap: RelatedGap,
        arguments: GapArguments,
    ) -> GapClosureEvaluation:
        origins = {
            "source" if item.get("evidence_id") else "analyst"
            for item in arguments.updates
        }
        if len(origins) != 1:
            raise GapInvestigationError(
                "Один gap result не смешивает source и analyst updates"
            )
        return gap.closure_contract.evaluate(
            {
                str(item["path"]): item["value"]
                for item in arguments.updates
            },
            source_confirmed=origins == {"source"},
            previously_satisfied=gap.closure_satisfied_paths,
        )

    @staticmethod
    def _validate_not_found(allowed_paths: tuple[str, ...], arguments: GapArguments) -> None:
        if arguments.updates or arguments.contradictions:
            raise GapInvestigationError("not_found не может изменять карточку")
        if not arguments.missing_fact or not arguments.unknown_fields:
            raise GapInvestigationError("not_found требует неизвестный факт и поля")
        if set(arguments.unknown_fields) - set(allowed_paths):
            raise GapInvestigationError("not_found ссылается на запрещённые поля")

    @staticmethod
    def _validate_contradiction(
        arguments: GapArguments,
        available: tuple[Evidence, ...],
    ) -> None:
        if arguments.updates or len(arguments.contradictions) < 2:
            raise GapInvestigationError("Противоречие требует минимум два утверждения без обновлений")
        evidence = {item.evidence_id: item for item in available}
        for item in arguments.contradictions:
            if not isinstance(item, dict) or set(item) != {"statement", "evidence_id"}:
                raise GapInvestigationError("Противоречие имеет неверную структуру")
            if not str(item["statement"]).strip() or str(item["evidence_id"]) not in evidence:
                raise GapInvestigationError("Противоречие ссылается на неизвестное evidence")
