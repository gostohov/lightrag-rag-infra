from __future__ import annotations

from contextlib import nullcontext
from datetime import UTC, datetime
from typing import Callable

from ...domain import (
    CardMutation,
    ContentField,
    Derivation,
    EpistemicStatus,
    Evidence,
    GapResolutionMode,
    GapStatus,
    RelatedGap,
)
from ...domain.errors import DomainError
from ...domain.schema import (
    CARD_FIELD_PATHS,
    EXPECTED_RESULT_PATHS,
    REQUIRED_FIELD_PATHS,
)
from ..card_history import save_card_revision
from ..gap_specs import RELATED_GAP_KEYS, expand_related_gap_spec
from ..repositories import UnitOfWork
from .models import (
    PopulationArguments,
    PopulationError,
    PopulationResult,
)


class PopulationService:
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

    def apply(
        self,
        card_id: str,
        arguments: PopulationArguments,
        *,
        available_evidence: tuple[Evidence, ...],
        uow: UnitOfWork | None = None,
    ) -> PopulationResult:
        context = nullcontext(uow) if uow is not None else self.uow_factory()
        with context as active_uow:
            card = active_uow.cards.get(card_id)
            if card is None:
                raise PopulationError(f"Карточка {card_id} не найдена")
            if card.revision != 0:
                raise PopulationError("Промпт 2 применяется только к незаполненной карточке")
            if arguments.analyst_values:
                raise PopulationError(
                    "Prompt 2 не принимает analyst_values; "
                    "используйте подтверждаемый conversation proposal"
                )
            try:
                mutation = self._build_mutation(
                    card_id=card_id,
                    selection_id=card.selection_id,
                    arguments=arguments,
                    existing_evidence={**card.evidence},
                    available_evidence=available_evidence,
                )
                card.apply(mutation)
                active_uow.cards.save(card)
                save_card_revision(
                    active_uow,
                    card,
                    reason="первоначальное заполнение",
                )
                active_uow.events.append(
                    card_id,
                    "первоначальная карточка заполнена",
                    {"revision": card.revision, "gaps": list(card.gaps)},
                )
            except (DomainError, KeyError, TypeError, ValueError) as error:
                if isinstance(error, PopulationError):
                    raise
                raise PopulationError(str(error)) from error
            return PopulationResult(
                card_id=card_id,
                revision=card.revision,
                open_gap_ids=tuple(card.gaps),
            )

    def repair_coverage(
        self,
        card_id: str,
        *,
        uow: UnitOfWork | None = None,
    ) -> PopulationResult:
        context = nullcontext(uow) if uow is not None else self.uow_factory()
        with context as active_uow:
            card = active_uow.cards.get(card_id)
            if card is None:
                raise PopulationError(f"Карточка {card_id} не найдена")
            if card.revision == 0:
                raise PopulationError(
                    "Восстановление покрытия доступно только заполненной карточке"
                )

            acceptable = {
                EpistemicStatus.SOURCE_CONFIRMED,
                EpistemicStatus.ANALYST_CONFIRMED,
                EpistemicStatus.DERIVED,
            }
            unresolved_gap_paths = {
                path
                for gap in card.gaps.values()
                if gap.status is not GapStatus.RESOLVED
                for path in gap.allowed_paths
            }
            missing = sorted(
                path
                for path in REQUIRED_FIELD_PATHS
                if card.field(path).status not in acceptable
                and path not in unresolved_gap_paths
            )
            expected_known = any(
                card.field(path).status in acceptable
                for path in EXPECTED_RESULT_PATHS
            )
            expected_covered = bool(EXPECTED_RESULT_PATHS & unresolved_gap_paths)

            if not expected_known and not expected_covered:
                raise PopulationError(
                    "В сохранённой карточке отсутствует ожидаемый результат; "
                    "автоматическое восстановление не выбирает тип результата"
                )
            gap_specs = [self._coverage_gap(card_id, path) for path in missing]
            if not gap_specs:
                raise PopulationError(
                    "Карточка не требует восстановления блокирующих пробелов"
                )

            card.apply(CardMutation(gaps=tuple(gap_specs)))
            active_uow.cards.save(card)
            save_card_revision(
                active_uow,
                card,
                reason="восстановлены блокирующие пробелы",
            )
            active_uow.events.append(
                card_id,
                "восстановлены блокирующие пробелы",
                {
                    "revision": card.revision,
                    "gap_ids": [gap.gap_id for gap in gap_specs],
                },
            )
            return PopulationResult(
                card_id=card_id,
                revision=card.revision,
                open_gap_ids=tuple(gap.gap_id for gap in gap_specs),
            )

    def _coverage_gap(self, card_id: str, path: str) -> RelatedGap:
        questions = {
            "requirement.condition": "Каково точное условие проверяемого требования?",
            "requirement.behavior": "Какое обязательное поведение задаёт выбранный фрагмент?",
            "test.action": "Какое исполнимое воздействие проверяет это требование?",
            "test.changed_factor": "Какой единственный фактор изменяется в этой проверке?",
            "test.observation.method": "Как наблюдать ожидаемый результат этой проверки?",
        }
        return RelatedGap(
            gap_id=self.next_id("GAP"),
            card_id=card_id,
            question=questions[path],
            blocking_reason=f"Обязательное поле {path} не заполнено и не связано с пробелом",
            allowed_paths=(path,),
            dependencies=(),
            closure_criterion=f"Поле {path} получило подтверждённое значение",
            resolution_mode=GapResolutionMode.SOURCE_FACT,
        )

    def _build_mutation(
        self,
        *,
        card_id: str,
        selection_id: str,
        arguments: PopulationArguments,
        existing_evidence: dict[str, Evidence],
        available_evidence: tuple[Evidence, ...],
    ) -> CardMutation:
        available = {item.evidence_id: item for item in available_evidence}
        evidence_to_add: dict[str, Evidence] = {}
        fields: dict[str, ContentField] = {}
        occupied: set[str] = set()

        def claim(path: str) -> None:
            if path not in CARD_FIELD_PATHS:
                raise PopulationError(f"Неизвестный путь карточки: {path}")
            if path in occupied:
                raise PopulationError(f"Путь {path} присутствует в нескольких разделах")
            occupied.add(path)

        for item in arguments.source_values:
            self._exact_keys(
                item,
                {"path", "value", "evidence_id"},
                "значение из источника",
            )
            path = str(item["path"])
            claim(path)
            evidence_id = str(item["evidence_id"])
            evidence = existing_evidence.get(evidence_id) or available.get(evidence_id)
            if evidence is None or not evidence.supports(card_id, selection_id):
                raise PopulationError(f"Неизвестное или недоступное evidence {evidence_id}")
            if evidence.evidence_id not in existing_evidence:
                evidence_to_add[evidence.evidence_id] = evidence
            fields[path] = ContentField.confirmed(item["value"], (str(evidence_id),))

        derivations: list[Derivation] = []
        known_evidence = {**existing_evidence, **available, **evidence_to_add}
        for item in arguments.derivations:
            self._exact_keys(
                item,
                {"path", "value", "source_evidence_ids", "rule", "scope"},
                "вывод",
            )
            path = str(item["path"])
            claim(path)
            sources = tuple(str(value) for value in item["source_evidence_ids"])
            if not sources or any(source not in known_evidence for source in sources):
                raise PopulationError("Вывод ссылается на неизвестные evidence")
            if not str(item["rule"]).strip() or not str(item["scope"]).strip():
                raise PopulationError("Вывод требует явное правило и область")
            for source in sources:
                if source not in existing_evidence:
                    evidence_to_add[source] = known_evidence[source]
            derivation = Derivation(
                derivation_id=self.next_id("DERIVATION"),
                card_id=card_id,
                source_evidence_ids=sources,
                rule=str(item["rule"]),
                scope=str(item["scope"]),
            )
            derivations.append(derivation)
            fields[path] = ContentField.derived(item["value"], derivation.derivation_id)

        for item in arguments.not_applicable:
            self._exact_keys(item, {"path", "reason"}, "неприменимое поле")
            path = str(item["path"])
            claim(path)
            fields[path] = ContentField.not_applicable(str(item["reason"]))

        gaps: list[RelatedGap] = []
        for item in arguments.gaps:
            if not isinstance(item, dict) or frozenset(item) not in {
                RELATED_GAP_KEYS,
                RELATED_GAP_KEYS | {"resolution_targets"},
            }:
                raise PopulationError(
                    "Раздел пробел имеет неверную структуру"
                )
            allowed_paths = tuple(str(value) for value in item["allowed_paths"])
            if not allowed_paths or len(allowed_paths) != len(set(allowed_paths)):
                raise PopulationError("Пробел требует уникальные разрешённые пути")
            for path in allowed_paths:
                claim(path)
            gaps.extend(
                expand_related_gap_spec(
                    item,
                    card_id=card_id,
                    next_id=self.next_id,
                )
            )

        return CardMutation(
            evidence=tuple(evidence_to_add.values()),
            derivations=tuple(derivations),
            fields=fields,
            gaps=tuple(gaps),
        )

    @staticmethod
    def _exact_keys(item: object, expected: set[str], label: str) -> None:
        if not isinstance(item, dict) or set(item) != expected:
            raise PopulationError(f"Раздел {label} имеет неверную структуру")
