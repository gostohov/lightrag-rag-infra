from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class PromptId(StrEnum):
    DECOMPOSITION = "prompt_1"
    DECOMPOSITION_WINDOW = "prompt_1_window"
    DECOMPOSITION_WINDOW_SEMANTIC = "prompt_1_window_semantic"
    DECOMPOSITION_SEMANTIC_SYNTHESIS = "prompt_1_semantic_synthesis"
    DECOMPOSITION_RECONCILIATION = "prompt_1_reconcile"
    POPULATION = "prompt_2"
    GAP_RESEARCH = "prompt_3"
    REFINEMENT = "card_refinement"
    SELECTION_REVIEW = "prompt_4"
    CONVERSATION = "conversation"


@dataclass(frozen=True, slots=True)
class PolicyRule:
    rule_id: str
    title: str
    instruction: str


@dataclass(frozen=True, slots=True)
class PromptInputBudget:
    max_lines: int
    max_estimated_tokens: int
    estimator: str

    def __post_init__(self) -> None:
        if self.max_lines < 1 or self.max_estimated_tokens < 1:
            raise ValueError("Input budget должен быть положительным")
        if not self.estimator.strip():
            raise ValueError("Input budget должен указывать estimator")


@dataclass(frozen=True, slots=True)
class PromptSpec:
    prompt_id: PromptId
    version: str
    instruction: str
    rule_ids: tuple[str, ...]
    allowed_context: tuple[str, ...]
    allowed_tools: tuple[str, ...]
    generation_parameters: dict[str, Any]
    input_budget: PromptInputBudget | None = None
    length_retry_max_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class PromptCall:
    prompt_id: PromptId
    policy_version: str
    prompt_version: str
    fingerprint: str
    rule_ids: tuple[str, ...]
    context: dict[str, Any]
    system_prompt: str
    allowed_tools: tuple[str, ...]
    generation_parameters: dict[str, Any]
    length_retry_max_tokens: int | None = None
