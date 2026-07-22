from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from types import MappingProxyType
from typing import Mapping

from .decisions import AnalystResolution, CardDecision
from .enums import (
    CardDecisionKind,
    EpistemicStatus,
    EvidenceKind,
    GapResolutionMode,
    GapStatus,
)
from .errors import DomainValidationError, EvidenceScopeError
from .evidence import Evidence
from .fields import ContentField, Derivation
from .gaps import RelatedGap
from .schema import CARD_FIELD_PATHS, EXPECTED_RESULT_PATHS, REQUIRED_FIELD_PATHS


@dataclass(frozen=True, slots=True)
class CardMutation:
    evidence: tuple[Evidence, ...] = ()
    derivations: tuple[Derivation, ...] = ()
    fields: Mapping[str, ContentField] = field(default_factory=dict)
    gaps: tuple[RelatedGap, ...] = ()
    gap_progress: Mapping[str, tuple[str, ...]] = field(
        default_factory=dict
    )
    resolved_gap_ids: tuple[str, ...] = ()
    resolutions: tuple[AnalystResolution, ...] = ()

    @property
    def is_empty(self) -> bool:
        return not any(
            (
                self.evidence,
                self.derivations,
                self.fields,
                self.gaps,
                self.gap_progress,
                self.resolved_gap_ids,
                self.resolutions,
            )
        )


@dataclass(frozen=True, slots=True)
class CardState:
    card_id: str
    selection_id: str
    title: str
    section_number: str
    changed_factor: str
    consequences: tuple[str, ...]
    revision: int
    fields: Mapping[str, ContentField]
    evidence: tuple[Evidence, ...]
    derivations: tuple[Derivation, ...]
    gaps: tuple[RelatedGap, ...]
    resolutions: tuple[AnalystResolution, ...]
    decision: CardDecision | None
    selection_review_current: bool


class TestCard:
    def __init__(
        self,
        *,
        card_id: str,
        selection_id: str,
        title: str,
        section_number: str,
        changed_factor: str,
        consequences: tuple[str, ...],
    ) -> None:
        self.card_id = card_id
        self.selection_id = selection_id
        self.title = title
        self.section_number = section_number
        self.changed_factor = changed_factor
        self.consequences = consequences
        self.revision = 0
        self._fields = {path: ContentField.unknown() for path in CARD_FIELD_PATHS}
        self._evidence: dict[str, Evidence] = {}
        self._derivations: dict[str, Derivation] = {}
        self._gaps: dict[str, RelatedGap] = {}
        self._resolutions: dict[str, AnalystResolution] = {}
        self._decision: CardDecision | None = None
        self._selection_review_current = False

    @classmethod
    def create(
        cls,
        *,
        card_id: str,
        selection_id: str,
        title: str,
        section_number: str,
        changed_factors: tuple[str, ...],
        consequences: tuple[str, ...],
    ) -> TestCard:
        if not card_id.strip() or not selection_id.strip() or not title.strip():
            raise DomainValidationError("Карточка должна иметь ID, selection и название")
        if len(changed_factors) != 1 or not changed_factors[0].strip():
            raise DomainValidationError("Карточка должна иметь один изменяемый фактор")
        if not consequences or any(not item.strip() for item in consequences):
            raise DomainValidationError("Карточка должна иметь обязательные последствия")
        return cls(
            card_id=card_id,
            selection_id=selection_id,
            title=title,
            section_number=section_number,
            changed_factor=changed_factors[0],
            consequences=tuple(consequences),
        )

    @property
    def fields(self) -> Mapping[str, ContentField]:
        return MappingProxyType(self._fields)

    @property
    def evidence(self) -> Mapping[str, Evidence]:
        return MappingProxyType(self._evidence)

    @property
    def gaps(self) -> Mapping[str, RelatedGap]:
        return MappingProxyType(self._gaps)

    @property
    def resolutions(self) -> Mapping[str, AnalystResolution]:
        return MappingProxyType(self._resolutions)

    @property
    def derivations(self) -> Mapping[str, Derivation]:
        return MappingProxyType(self._derivations)

    @property
    def decision(self) -> CardDecision | None:
        return self._decision

    @property
    def selection_review_current(self) -> bool:
        return self._selection_review_current

    @property
    def is_ready(self) -> bool:
        if any(gap.status is not GapStatus.RESOLVED for gap in self._gaps.values()):
            return False
        acceptable = {
            EpistemicStatus.SOURCE_CONFIRMED,
            EpistemicStatus.ANALYST_CONFIRMED,
            EpistemicStatus.DERIVED,
        }
        if any(self._fields[path].status not in acceptable for path in REQUIRED_FIELD_PATHS):
            return False
        return any(self._fields[path].status in acceptable for path in EXPECTED_RESULT_PATHS)

    def field(self, path: str) -> ContentField:
        self._assert_path(path)
        return self._fields[path]

    def apply(self, mutation: CardMutation) -> bool:
        if mutation.is_empty:
            return False

        evidence = dict(self._evidence)
        derivations = dict(self._derivations)
        fields = dict(self._fields)
        gaps = dict(self._gaps)
        resolutions = dict(self._resolutions)

        self._apply_evidence(evidence, mutation.evidence)
        self._apply_derivations(derivations, evidence, mutation.derivations)
        self._apply_resolutions(resolutions, evidence, mutation.resolutions)
        self._apply_fields(
            fields,
            evidence,
            derivations,
            resolutions,
            mutation.fields,
        )
        self._apply_gaps(
            gaps,
            mutation.gaps,
            mutation.gap_progress,
            mutation.resolved_gap_ids,
        )

        current = (self._evidence, self._derivations, self._fields, self._gaps, self._resolutions)
        proposed = (evidence, derivations, fields, gaps, resolutions)
        if proposed == current:
            return False

        self._evidence = evidence
        self._derivations = derivations
        self._fields = fields
        self._gaps = gaps
        self._resolutions = resolutions
        self._touch()
        return True

    def resolve_gap(
        self,
        gap_id: str,
        mutation: CardMutation,
        *,
        closure_satisfied_paths: tuple[str, ...],
    ) -> bool:
        return self.apply_gap_progress(
            gap_id,
            mutation,
            closure_satisfied_paths=closure_satisfied_paths,
            resolve=True,
        )

    def apply_gap_progress(
        self,
        gap_id: str,
        mutation: CardMutation,
        *,
        closure_satisfied_paths: tuple[str, ...],
        resolve: bool,
    ) -> bool:
        gap = self._gaps.get(gap_id)
        if gap is None or gap.status is not GapStatus.OPEN:
            raise DomainValidationError(f"Нет открытого пробела {gap_id}")
        gap.assert_allows(tuple(mutation.fields))
        combined = CardMutation(
            evidence=mutation.evidence,
            derivations=mutation.derivations,
            fields=mutation.fields,
            gaps=mutation.gaps,
            gap_progress={
                **mutation.gap_progress,
                gap_id: closure_satisfied_paths,
            },
            resolved_gap_ids=(
                (*mutation.resolved_gap_ids, gap_id)
                if resolve
                else mutation.resolved_gap_ids
            ),
            resolutions=mutation.resolutions,
        )
        return self.apply(combined)

    def change_gap_resolution_mode(
        self,
        gap_id: str,
        mode: GapResolutionMode,
    ) -> bool:
        gap = self._gaps.get(gap_id)
        if gap is None or gap.status is not GapStatus.OPEN:
            raise DomainValidationError(f"Нет открытого пробела {gap_id}")
        if gap.resolution_mode is mode:
            return False
        self._gaps = {
            **self._gaps,
            gap_id: gap.with_resolution_mode(mode),
        }
        self._touch()
        return True

    def leave_gap_open(self, gap_id: str) -> bool:
        gap = self._gaps.get(gap_id)
        if gap is None or gap.status is not GapStatus.OPEN:
            raise DomainValidationError(f"Нет открытого пробела {gap_id}")
        self._gaps = {
            **self._gaps,
            gap_id: gap.leave_open(),
        }
        self._touch()
        return True

    def apply_analyst_resolution(
        self,
        resolution: AnalystResolution,
        mutation: CardMutation,
    ) -> bool:
        if resolution.card_id != self.card_id:
            raise EvidenceScopeError("Решение аналитика относится к другой карточке")
        if set(resolution.target_paths) != set(mutation.fields):
            raise DomainValidationError("Решение должно явно перечислять изменяемые поля")
        combined = CardMutation(
            evidence=mutation.evidence,
            derivations=mutation.derivations,
            fields=mutation.fields,
            gaps=mutation.gaps,
            resolved_gap_ids=mutation.resolved_gap_ids,
            resolutions=(*mutation.resolutions, resolution),
        )
        return self.apply(combined)

    def include(self, *, author: str, at: datetime) -> None:
        if not self.is_ready:
            raise DomainValidationError("Карточка не готова к обычному включению")
        self._selection_review_current = False
        self._decision = CardDecision(
            kind=CardDecisionKind.INCLUDE,
            card_id=self.card_id,
            revision=self.revision,
            author=author,
            created_at=at,
        )

    def include_incomplete(self, *, author: str, reason: str, at: datetime) -> None:
        self._selection_review_current = False
        self._decision = CardDecision(
            kind=CardDecisionKind.INCLUDE_INCOMPLETE,
            card_id=self.card_id,
            revision=self.revision,
            author=author,
            created_at=at,
            reason=reason,
        )

    def exclude(self, *, author: str, reason: str, at: datetime) -> None:
        self._selection_review_current = False
        self._decision = CardDecision(
            kind=CardDecisionKind.EXCLUDE,
            card_id=self.card_id,
            revision=self.revision,
            author=author,
            created_at=at,
            reason=reason,
        )

    def mark_selection_review_current(self) -> None:
        self._selection_review_current = True

    def snapshot(self) -> CardState:
        return CardState(
            card_id=self.card_id,
            selection_id=self.selection_id,
            title=self.title,
            section_number=self.section_number,
            changed_factor=self.changed_factor,
            consequences=self.consequences,
            revision=self.revision,
            fields=dict(self._fields),
            evidence=tuple(self._evidence.values()),
            derivations=tuple(self._derivations.values()),
            gaps=tuple(self._gaps.values()),
            resolutions=tuple(self._resolutions.values()),
            decision=self._decision,
            selection_review_current=self._selection_review_current,
        )

    @classmethod
    def restore(cls, state: CardState) -> TestCard:
        if state.revision < 0:
            raise DomainValidationError("Ревизия карточки не может быть отрицательной")
        restored = cls.create(
            card_id=state.card_id,
            selection_id=state.selection_id,
            title=state.title,
            section_number=state.section_number,
            changed_factors=(state.changed_factor,),
            consequences=state.consequences,
        )
        restored.apply(
            CardMutation(
                evidence=state.evidence,
                derivations=state.derivations,
                fields=state.fields,
                gaps=state.gaps,
                resolutions=state.resolutions,
            )
        )
        if state.decision is not None:
            if state.decision.card_id != state.card_id or state.decision.revision != state.revision:
                raise DomainValidationError("Сохранённое решение не относится к текущей ревизии")
        restored.revision = state.revision
        restored._decision = state.decision
        restored._selection_review_current = state.selection_review_current
        return restored

    def _touch(self) -> None:
        self.revision += 1
        self._decision = None
        self._selection_review_current = False

    def _apply_evidence(self, target: dict[str, Evidence], items: tuple[Evidence, ...]) -> None:
        for item in items:
            if not item.supports(self.card_id, self.selection_id):
                raise EvidenceScopeError(
                    f"Evidence {item.evidence_id} не относится к карточке {self.card_id}"
                )
            existing = target.get(item.evidence_id)
            if existing is not None and existing != item:
                raise DomainValidationError(f"Evidence ID {item.evidence_id} уже занят")
            target[item.evidence_id] = item

    def _apply_derivations(
        self,
        target: dict[str, Derivation],
        evidence: dict[str, Evidence],
        items: tuple[Derivation, ...],
    ) -> None:
        for item in items:
            if item.card_id != self.card_id:
                raise EvidenceScopeError("Вывод относится к другой карточке")
            for evidence_id in item.source_evidence_ids:
                self._assert_evidence(evidence, evidence_id)
            existing = target.get(item.derivation_id)
            if existing is not None and existing != item:
                raise DomainValidationError(f"Derivation ID {item.derivation_id} уже занят")
            target[item.derivation_id] = item

    def _apply_fields(
        self,
        target: dict[str, ContentField],
        evidence: dict[str, Evidence],
        derivations: dict[str, Derivation],
        resolutions: dict[str, AnalystResolution],
        items: Mapping[str, ContentField],
    ) -> None:
        for path, value in items.items():
            self._assert_path(path)
            if value.status is EpistemicStatus.SOURCE_CONFIRMED:
                if not value.evidence_ids:
                    raise DomainValidationError(
                        f"Поле {path}, подтверждённое источником, "
                        "требует evidence"
                    )
                for evidence_id in value.evidence_ids:
                    self._assert_evidence(evidence, evidence_id)
                    if (
                        evidence[evidence_id].kind
                        is not EvidenceKind.SOURCE_FRAGMENT
                    ):
                        raise DomainValidationError(
                            f"Поле {path}, подтверждённое источником, "
                            "требует только source evidence"
                        )
            if value.status is EpistemicStatus.ANALYST_CONFIRMED:
                if not value.evidence_ids:
                    raise DomainValidationError(
                        f"Поле {path}, подтверждённое аналитиком, "
                        "требует evidence"
                    )
                for evidence_id in value.evidence_ids:
                    self._assert_evidence(evidence, evidence_id)
                    if (
                        evidence[evidence_id].kind
                        is not EvidenceKind.HUMAN_KNOWLEDGE
                    ):
                        raise DomainValidationError(
                            f"Поле {path}, подтверждённое аналитиком, "
                            "требует только экспертное знание"
                        )
                if not any(
                    path in resolution.target_paths
                    and set(value.evidence_ids).issubset(
                        resolution.evidence_ids
                    )
                    and (
                        not resolution.values
                        or any(
                            item["path"] == path
                            and item["value"] == value.value
                            for item in resolution.values
                        )
                    )
                    for resolution in resolutions.values()
                ):
                    raise DomainValidationError(
                        f"Поле {path} требует решение аналитика"
                    )
            if value.status is EpistemicStatus.DERIVED:
                if value.derivation_id not in derivations:
                    raise DomainValidationError(
                        f"Поле {path} ссылается на неизвестный вывод {value.derivation_id}"
                    )
            target[path] = value

    def _apply_gaps(
        self,
        target: dict[str, RelatedGap],
        items: tuple[RelatedGap, ...],
        progress: Mapping[str, tuple[str, ...]],
        resolved_ids: tuple[str, ...],
    ) -> None:
        for item in items:
            if item.card_id != self.card_id:
                raise EvidenceScopeError("Пробел относится к другой карточке")
            existing = target.get(item.gap_id)
            if existing is not None and existing != item:
                raise DomainValidationError(f"Gap ID {item.gap_id} уже занят")
            target[item.gap_id] = item
        for gap_id, satisfied_paths in progress.items():
            gap = target.get(gap_id)
            if gap is None or gap.status is not GapStatus.OPEN:
                raise DomainValidationError(
                    f"Нет открытого пробела {gap_id} для closure progress"
                )
            target[gap_id] = gap.with_closure_progress(
                tuple(satisfied_paths)
            )
        for gap_id in resolved_ids:
            gap = target.get(gap_id)
            if gap is None:
                raise DomainValidationError(f"Неизвестный пробел {gap_id}")
            target[gap_id] = gap.resolve()

    def _apply_resolutions(
        self,
        target: dict[str, AnalystResolution],
        evidence: dict[str, Evidence],
        items: tuple[AnalystResolution, ...],
    ) -> None:
        for item in items:
            if item.card_id != self.card_id:
                raise EvidenceScopeError("Решение аналитика относится к другой карточке")
            for path in item.target_paths:
                self._assert_path(path)
            for evidence_id in item.evidence_ids:
                self._assert_evidence(evidence, evidence_id)
            if any(
                evidence[evidence_id].kind
                is not EvidenceKind.HUMAN_KNOWLEDGE
                for evidence_id in item.evidence_ids
            ):
                raise DomainValidationError(
                    "Решение аналитика требует только экспертное знание"
                )
            existing = target.get(item.resolution_id)
            if existing is not None and existing != item:
                raise DomainValidationError(f"Resolution ID {item.resolution_id} уже занят")
            target[item.resolution_id] = item

    def _assert_evidence(self, evidence: dict[str, Evidence], evidence_id: str) -> None:
        item = evidence.get(evidence_id)
        if item is None:
            raise DomainValidationError(f"Неизвестное evidence {evidence_id}")
        if not item.supports(self.card_id, self.selection_id):
            raise EvidenceScopeError(f"Evidence {evidence_id} недоступно этой карточке")

    @staticmethod
    def _assert_path(path: str) -> None:
        if path not in CARD_FIELD_PATHS:
            raise DomainValidationError(f"Неизвестный путь карточки: {path}")
