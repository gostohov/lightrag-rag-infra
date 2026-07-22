from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from collections.abc import Awaitable, Callable
from typing import Any, Protocol


class ConversationTurnKind(StrEnum):
    ANSWER = "answer"
    CLARIFICATION = "clarification"
    TOOL_CALL = "tool_call"


class ConversationEffect(StrEnum):
    READ_ONLY = "read_only"
    EXPENSIVE = "expensive"
    MUTATING = "mutating"


class ConversationAction(StrEnum):
    RESUME = "resume"
    RESEARCH_GAP = "research_gap"
    SUBMIT_ANALYST_ANSWER = "submit_analyst_answer"
    CONFIRM_ANALYST_ANSWER = "confirm_analyst_answer"
    REJECT_ANALYST_ANSWER = "reject_analyst_answer"
    PROPOSE_DESIGN_DECISION = "propose_design_decision"
    CHANGE_GAP_MODE = "change_gap_mode"
    LEAVE_GAP = "leave_gap"
    REFINE_CARD = "refine_card"
    INCLUDE_CARD = "include_card"
    EXCLUDE_CARD = "exclude_card"
    EXPORT_DIAGNOSTICS = "export_diagnostics"
    EXPORT_PMI = "export_pmi"


_ACTION_EFFECTS = {
    ConversationAction.RESUME: ConversationEffect.EXPENSIVE,
    ConversationAction.RESEARCH_GAP: ConversationEffect.EXPENSIVE,
    ConversationAction.SUBMIT_ANALYST_ANSWER: ConversationEffect.MUTATING,
    ConversationAction.CONFIRM_ANALYST_ANSWER: ConversationEffect.MUTATING,
    ConversationAction.REJECT_ANALYST_ANSWER: ConversationEffect.MUTATING,
    ConversationAction.PROPOSE_DESIGN_DECISION: ConversationEffect.READ_ONLY,
    ConversationAction.CHANGE_GAP_MODE: ConversationEffect.MUTATING,
    ConversationAction.LEAVE_GAP: ConversationEffect.MUTATING,
    ConversationAction.REFINE_CARD: ConversationEffect.MUTATING,
    ConversationAction.INCLUDE_CARD: ConversationEffect.MUTATING,
    ConversationAction.EXCLUDE_CARD: ConversationEffect.MUTATING,
    ConversationAction.EXPORT_DIAGNOSTICS: ConversationEffect.READ_ONLY,
    ConversationAction.EXPORT_PMI: ConversationEffect.READ_ONLY,
}

_CONFIRMATION_REQUIRED = frozenset(
    {
        ConversationAction.LEAVE_GAP,
        ConversationAction.CONFIRM_ANALYST_ANSWER,
        ConversationAction.INCLUDE_CARD,
        ConversationAction.EXCLUDE_CARD,
    }
)

_ACTION_USER_LABELS = {
    ConversationAction.RESUME: "продолжить работу",
    ConversationAction.RESEARCH_GAP: "исследовать пробел по новому вопросу",
    ConversationAction.SUBMIT_ANALYST_ANSWER: "использовать ответ аналитика",
    ConversationAction.CONFIRM_ANALYST_ANSWER: (
        "подтвердить предложенную интерпретацию"
    ),
    ConversationAction.REJECT_ANALYST_ANSWER: (
        "отклонить предложенную интерпретацию"
    ),
    ConversationAction.PROPOSE_DESIGN_DECISION: "обсудить проектное решение",
    ConversationAction.CHANGE_GAP_MODE: "изменить способ разрешения пробела",
    ConversationAction.LEAVE_GAP: "оставить пробел открытым",
    ConversationAction.REFINE_CARD: "продолжить доработку карточки",
    ConversationAction.INCLUDE_CARD: "включить карточку в итоговый ПМИ",
    ConversationAction.EXCLUDE_CARD: "исключить карточку из итогового ПМИ",
    ConversationAction.EXPORT_DIAGNOSTICS: "экспортировать диагностику сессии",
    ConversationAction.EXPORT_PMI: "экспортировать итоговый ПМИ",
}
_TERMINAL_TOOL_USER_LABELS = {
    "respond_to_analyst": "ответить аналитику",
    "request_clarification": "уточнить намерение аналитика",
}


def action_effect(action: ConversationAction) -> ConversationEffect:
    return _ACTION_EFFECTS[action]


def requires_confirmation(action: ConversationAction) -> bool:
    return action in _CONFIRMATION_REQUIRED


def action_user_label(action: ConversationAction) -> str:
    return _ACTION_USER_LABELS[action]


def user_facing_conversation_text(text: str) -> str:
    result = text
    replacements = {
        **{
            action.value: label
            for action, label in _ACTION_USER_LABELS.items()
        },
        **_TERMINAL_TOOL_USER_LABELS,
    }
    for internal_name in sorted(replacements, key=len, reverse=True):
        result = result.replace(internal_name, replacements[internal_name])
    return result


@dataclass(frozen=True, slots=True)
class ConversationToolCall:
    action: ConversationAction
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ConversationToolResult:
    action: ConversationAction
    effect: ConversationEffect
    text: str
    awaitable: Awaitable[Any] | None = None
    cancel: Callable[[], object] | None = None

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("Результат conversation tool должен содержать текст")
        if (self.awaitable is None) != (self.cancel is None):
            raise ValueError("Длительная операция требует awaitable и cancel")


@dataclass(frozen=True, slots=True)
class ConversationTurnResult:
    decision: ConversationTurnDecision
    tool_result: ConversationToolResult | None = None
    operation_result: Any = None


@dataclass(frozen=True, slots=True)
class ConversationTurnDecision:
    kind: ConversationTurnKind
    text: str
    tool_call: ConversationToolCall | None = None

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("Решение conversation turn должно содержать отображаемый текст")
        has_tool = self.tool_call is not None
        if self.kind is ConversationTurnKind.TOOL_CALL and not has_tool:
            raise ValueError("Решение tool_call должно содержать ровно один tool")
        if self.kind is not ConversationTurnKind.TOOL_CALL and has_tool:
            raise ValueError("Текстовый ответ или уточнение не может содержать tool")


@dataclass(frozen=True, slots=True)
class ConversationGapClosureContext:
    path: str
    accepted_forms: tuple[str, ...]
    residual_question: str

    def __post_init__(self) -> None:
        if not self.path.strip() or not self.accepted_forms:
            raise ValueError(
                "Conversation closure requirement требует path и формы"
            )
        if not self.residual_question.strip():
            raise ValueError(
                "Conversation closure requirement требует остаточный вопрос"
            )

    def as_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "accepted_forms": list(self.accepted_forms),
            "residual_question": self.residual_question,
        }


@dataclass(frozen=True, slots=True)
class ConversationGapContext:
    gap_id: str
    question: str
    blocking_reason: str
    allowed_paths: tuple[str, ...]
    resolution_mode: str
    closure_schema_version: int = 1
    closure_requirements: tuple[ConversationGapClosureContext, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "gap_id": self.gap_id,
            "question": self.question,
            "blocking_reason": self.blocking_reason,
            "allowed_paths": list(self.allowed_paths),
            "resolution_mode": self.resolution_mode,
            "closure_schema_version": self.closure_schema_version,
            "closure_requirements": [
                item.as_dict() for item in self.closure_requirements
            ],
        }


@dataclass(frozen=True, slots=True)
class ConversationProposalContext:
    proposal_id: str
    gap_id: str | None
    source_message_id: str
    expected_revision: int
    values: tuple[dict[str, Any], ...]
    proposal_kind: str = "gap_answer"
    refinement_arguments: dict[str, Any] | None = None
    closure_evaluation: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "proposal_id": self.proposal_id,
            "proposal_kind": self.proposal_kind,
            "gap_id": self.gap_id,
            "source_message_id": self.source_message_id,
            "expected_revision": self.expected_revision,
            "values": [dict(item) for item in self.values],
            "refinement_arguments": self.refinement_arguments,
            "closure_evaluation": self.closure_evaluation,
        }


@dataclass(frozen=True, slots=True)
class ConversationContext:
    session_id: str
    card_id: str
    card_revision: int
    stage: str
    continuation: str
    fields: dict[str, dict[str, Any]]
    open_gap: ConversationGapContext | None
    available_actions: tuple[ConversationAction, ...]
    pending_proposal: ConversationProposalContext | None = None
    recent_events: tuple[dict[str, object], ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "session_id": self.session_id,
            "card": {
                "card_id": self.card_id,
                "revision": self.card_revision,
                "fields": self.fields,
            },
            "stage": self.stage,
            "continuation": self.continuation,
            "open_gap": self.open_gap.as_dict() if self.open_gap else None,
            "pending_analyst_answer": (
                self.pending_proposal.as_dict()
                if self.pending_proposal
                else None
            ),
            "available_actions": [item.value for item in self.available_actions],
            "available_action_labels": {
                item.value: action_user_label(item)
                for item in self.available_actions
            },
            "recent_events": list(self.recent_events),
        }


class ConversationAgent(Protocol):
    async def decide(
        self,
        *,
        context: ConversationContext,
        message_id: str,
        user_text: str,
    ) -> ConversationTurnDecision: ...


class ConversationAgentError(RuntimeError):
    pass
