from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class WorkflowStage(StrEnum):
    EMPTY = "нет выбранного диапазона"
    SELECTION_CONFIRMED = "диапазон подтвержден"
    DECOMPOSING = "декомпозиция диапазона"
    DECOMPOSITION_REVIEW = "выбор каркасов"
    CARD_WORK = "подготовка карточки"
    POPULATING_CARD = "первичное заполнение карточки"
    INVESTIGATING_GAP = "исследование пробела"
    REFINING_CARD = "доработка карточки"
    ANALYST_DECISION = "нужно решение аналитика"
    RANGE_REVIEW = "проверка диапазона"
    RANGE_REVIEWED = "диапазон проверен"
    EXPORT_ALLOWED = "экспорт разрешен"


@dataclass(frozen=True, slots=True)
class SkeletonState:
    status: str = "pending"
    card_id: str | None = None


@dataclass(frozen=True, slots=True)
class CardWorkflowState:
    revision: int = 0
    populated: bool = False
    gaps: tuple[tuple[str, str], ...] = ()
    decision: str | None = None
    decision_revision: int | None = None

    def gap_status(self, gap_id: str) -> str | None:
        return dict(self.gaps).get(gap_id)


@dataclass(frozen=True, slots=True)
class AttemptState:
    attempt_id: str
    kind: str
    card_id: str | None = None
    gap_id: str | None = None


@dataclass(frozen=True, slots=True)
class RangeReviewState:
    revisions: tuple[tuple[str, int], ...]
    warnings: tuple[str, ...]
    accepted_with_issues: bool = False


@dataclass(frozen=True, slots=True)
class WorkflowState:
    stage: WorkflowStage
    selection_id: str | None
    skeletons: dict[str, SkeletonState]
    cards: dict[str, CardWorkflowState]
    active_attempt: AttemptState | None
    cancelled_attempt_ids: tuple[str, ...]
    failed_attempt_ids: tuple[str, ...]
    decomposition_outcome: str | None
    range_review: RangeReviewState | None
    export_allowed: bool

    @classmethod
    def empty(cls) -> WorkflowState:
        return cls(
            stage=WorkflowStage.EMPTY,
            selection_id=None,
            skeletons={},
            cards={},
            active_attempt=None,
            cancelled_attempt_ids=(),
            failed_attempt_ids=(),
            decomposition_outcome=None,
            range_review=None,
            export_allowed=False,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage.value,
            "selection_id": self.selection_id,
            "skeletons": {
                key: {"status": item.status, "card_id": item.card_id}
                for key, item in self.skeletons.items()
            },
            "cards": {
                key: {
                    "revision": item.revision,
                    "populated": item.populated,
                    "gaps": [list(gap) for gap in item.gaps],
                    "decision": item.decision,
                    "decision_revision": item.decision_revision,
                }
                for key, item in self.cards.items()
            },
            "active_attempt": (
                {
                    "attempt_id": self.active_attempt.attempt_id,
                    "kind": self.active_attempt.kind,
                    "card_id": self.active_attempt.card_id,
                    "gap_id": self.active_attempt.gap_id,
                }
                if self.active_attempt
                else None
            ),
            "cancelled_attempt_ids": list(self.cancelled_attempt_ids),
            "failed_attempt_ids": list(self.failed_attempt_ids),
            "decomposition_outcome": self.decomposition_outcome,
            "range_review": (
                {
                    "revisions": [list(item) for item in self.range_review.revisions],
                    "warnings": list(self.range_review.warnings),
                    "accepted_with_issues": self.range_review.accepted_with_issues,
                }
                if self.range_review
                else None
            ),
            "export_allowed": self.export_allowed,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> WorkflowState:
        if not value:
            return cls.empty()
        active = value.get("active_attempt")
        review = value.get("range_review")
        return cls(
            stage=WorkflowStage(value["stage"]),
            selection_id=value.get("selection_id"),
            skeletons={
                key: SkeletonState(status=item["status"], card_id=item.get("card_id"))
                for key, item in value.get("skeletons", {}).items()
            },
            cards={
                key: CardWorkflowState(
                    revision=int(item.get("revision", 0)),
                    populated=bool(item.get("populated", False)),
                    gaps=tuple((str(gap[0]), str(gap[1])) for gap in item.get("gaps", [])),
                    decision=item.get("decision"),
                    decision_revision=item.get("decision_revision"),
                )
                for key, item in value.get("cards", {}).items()
            },
            active_attempt=AttemptState(**active) if active else None,
            cancelled_attempt_ids=tuple(value.get("cancelled_attempt_ids", [])),
            failed_attempt_ids=tuple(value.get("failed_attempt_ids", [])),
            decomposition_outcome=value.get("decomposition_outcome"),
            range_review=(
                RangeReviewState(
                    revisions=tuple(
                        (str(item[0]), int(item[1])) for item in review.get("revisions", [])
                    ),
                    warnings=tuple(review.get("warnings", [])),
                    accepted_with_issues=bool(review.get("accepted_with_issues", False)),
                )
                if review
                else None
            ),
            export_allowed=bool(value.get("export_allowed", False)),
        )
