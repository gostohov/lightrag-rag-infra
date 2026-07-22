from __future__ import annotations

import hashlib
import json
from math import ceil

from ....domain.source import TextSelection
from ...prompting import PromptId, PromptPolicy
from ..budget import DecompositionBudgetPolicy
from .models import (
    WINDOW_CANDIDATE_SCHEMA_VERSION,
    DecompositionRoute,
    WindowingDecision,
)
from .semantic_versions import (
    SEMANTIC_CANONICAL_MAPPING_VERSION,
    SEMANTIC_SYNTHESIS_SCHEMA_VERSION,
    SEMANTIC_WINDOW_SCHEMA_VERSION,
)


class WindowingPolicy:
    VERSION = "windowing-policy-14"
    HARD_MAX_LINES = 1920
    HARD_MAX_ESTIMATED_TOKENS = 96_000
    DEFAULT_OVERLAP_LINES = 60
    CHILD_OUTPUT_USAGE_NUMERATOR = 3
    CHILD_OUTPUT_USAGE_DENOMINATOR = 4
    CHILD_PLANNING_OUTPUT_TOKENS = 4096
    CHILD_OUTPUT_FIXED_RESERVE_TOKENS = 2048
    CHILD_OUTPUT_PRIMARY_COMPLEXITY_TOKENS = 40
    SEMANTIC_SPLIT_MAX_DEPTH = 2
    SEMANTIC_SPLIT_MIN_PRIMARY_LINES = 6
    SEMANTIC_SPLIT_MAX_GENERATION_REQUESTS = 14
    RECONCILIATION_MAX_CASES = 128

    def __init__(
        self,
        prompt_policy: PromptPolicy,
        *,
        route_max_lines: int | None = None,
        primary_max_lines: int | None = None,
        overlap_lines: int | None = None,
    ) -> None:
        spec = prompt_policy.prompts[PromptId.DECOMPOSITION]
        child_spec = prompt_policy.prompts[PromptId.DECOMPOSITION_WINDOW]
        semantic_child_spec = prompt_policy.prompts[
            PromptId.DECOMPOSITION_WINDOW_SEMANTIC
        ]
        synthesis_spec = prompt_policy.prompts[
            PromptId.DECOMPOSITION_SEMANTIC_SYNTHESIS
        ]
        reconciliation_spec = prompt_policy.prompts[
            PromptId.DECOMPOSITION_RECONCILIATION
        ]
        if spec.input_budget is None:
            raise ValueError("Prompt 1 не содержит input budget")
        self.prompt_policy_version = prompt_policy.version
        self.single_call_prompt_version = spec.version
        self.legacy_prompt_version = child_spec.version
        self.legacy_candidate_schema_version = WINDOW_CANDIDATE_SCHEMA_VERSION
        self.semantic_prompt_version = semantic_child_spec.version
        self.semantic_schema_version = SEMANTIC_WINDOW_SCHEMA_VERSION
        self.semantic_mapping_version = SEMANTIC_CANONICAL_MAPPING_VERSION
        self.synthesis_prompt_version = synthesis_spec.version
        self.synthesis_schema_version = SEMANTIC_SYNTHESIS_SCHEMA_VERSION
        self.prompt_version = self.semantic_prompt_version
        self.candidate_schema_version = self.semantic_schema_version
        self.reconciliation_prompt_version = reconciliation_spec.version
        self.single_call_max_lines = spec.input_budget.max_lines
        self.single_call_max_estimated_tokens = (
            spec.input_budget.max_estimated_tokens
        )
        self.estimator = spec.input_budget.estimator
        self.child_output_tokens = int(
            semantic_child_spec.generation_parameters["max_tokens"]
        )
        self.child_length_retry_max_tokens = (
            semantic_child_spec.length_retry_max_tokens
        )
        if self.child_length_retry_max_tokens is None:
            raise ValueError(
                "Semantic child prompt не содержит расширенный "
                "generation profile"
            )
        self.child_output_budget_tokens = (
            self.CHILD_PLANNING_OUTPUT_TOKENS
            * self.CHILD_OUTPUT_USAGE_NUMERATOR
            // self.CHILD_OUTPUT_USAGE_DENOMINATOR
        )
        self.semantic_split_max_depth = self.SEMANTIC_SPLIT_MAX_DEPTH
        self.semantic_split_min_primary_lines = (
            self.SEMANTIC_SPLIT_MIN_PRIMARY_LINES
        )
        self.semantic_split_max_generation_requests = (
            self.SEMANTIC_SPLIT_MAX_GENERATION_REQUESTS
        )
        available_primary_tokens = (
            self.child_output_budget_tokens
            - self.CHILD_OUTPUT_FIXED_RESERVE_TOKENS
        )
        if (
            available_primary_tokens
            < self.CHILD_OUTPUT_PRIMARY_COMPLEXITY_TOKENS
        ):
            raise ValueError("Prompt 1 window output budget слишком мал")
        self.output_primary_max_lines = (
            available_primary_tokens
            // self.CHILD_OUTPUT_PRIMARY_COMPLEXITY_TOKENS
        )

        self.route_max_lines = (
            route_max_lines
            if route_max_lines is not None
            else min(
                self.single_call_max_lines,
                self.output_primary_max_lines,
            )
        )
        requested_primary_max_lines = (
            primary_max_lines or self.single_call_max_lines // 2
        )
        if requested_primary_max_lines > self.output_primary_max_lines:
            if primary_max_lines is not None:
                raise ValueError(
                    "primary_max_lines превышает Prompt 1 window output budget"
                )
            requested_primary_max_lines = self.output_primary_max_lines
        self.primary_max_lines = requested_primary_max_lines
        self.overlap_lines = (
            overlap_lines
            if overlap_lines is not None
            else min(
                self.DEFAULT_OVERLAP_LINES,
                (self.single_call_max_lines - self.primary_max_lines) // 2,
            )
        )
        if (
            self.route_max_lines < 1
            or self.primary_max_lines < 1
            or self.overlap_lines < 0
            or self.primary_max_lines + (2 * self.overlap_lines)
            > self.single_call_max_lines
        ):
            raise ValueError("Некорректные evaluation window limits")
        self.hard_max_lines = self.HARD_MAX_LINES
        self.hard_max_estimated_tokens = self.HARD_MAX_ESTIMATED_TOKENS
        self.max_windows = ceil(
            self.hard_max_lines / self.primary_max_lines
        )
        self.reconciliation_max_candidates = 64
        self.reconciliation_max_groups = 64
        self.reconciliation_max_cases = self.RECONCILIATION_MAX_CASES
        self.reconciliation_max_source_lines = self.single_call_max_lines
        self.reconciliation_max_estimated_tokens = (
            self.single_call_max_estimated_tokens
        )
        self.reconciliation_output_tokens = int(
            reconciliation_spec.generation_parameters["max_tokens"]
        )
        self.synthesis_output_tokens = int(
            synthesis_spec.generation_parameters["max_tokens"]
        )
        self.max_repair_attempts = 1
        self.fingerprint = self._fingerprint()
        self._selection_budget = DecompositionBudgetPolicy(
            max_lines=self.single_call_max_lines,
            max_estimated_tokens=self.single_call_max_estimated_tokens,
            estimator=self.estimator,
        )
        self._route_budget = DecompositionBudgetPolicy(
            max_lines=self.route_max_lines,
            max_estimated_tokens=self.single_call_max_estimated_tokens,
            estimator=self.estimator,
        )

    def assess(
        self,
        selection: TextSelection,
        *,
        selection_id: str = "<selection-id>",
    ) -> WindowingDecision:
        budget = self._route_budget.assess(
            selection,
            selection_id=selection_id,
        )
        if budget.within_single_call:
            route = DecompositionRoute.SINGLE_CALL
        elif (
            budget.line_count <= self.hard_max_lines
            and budget.estimated_tokens <= self.hard_max_estimated_tokens
        ):
            route = DecompositionRoute.WINDOWED
        else:
            route = DecompositionRoute.HARD_LIMIT
        return WindowingDecision(
            route=route,
            budget=budget,
            hard_max_lines=self.hard_max_lines,
            hard_max_estimated_tokens=self.hard_max_estimated_tokens,
        )

    def assess_window(
        self,
        selection: TextSelection,
        *,
        window_id: str,
    ) -> WindowingDecision:
        budget = self._selection_budget.assess(
            selection,
            selection_id=window_id,
        )
        return WindowingDecision(
            route=(
                DecompositionRoute.SINGLE_CALL
                if budget.within_single_call
                else DecompositionRoute.HARD_LIMIT
            ),
            budget=budget,
            hard_max_lines=self.hard_max_lines,
            hard_max_estimated_tokens=self.hard_max_estimated_tokens,
        )

    def estimate_child_output(self, primary_line_count: int) -> int:
        if primary_line_count < 1:
            raise ValueError("Prompt 1 window должен содержать primary строки")
        return (
            self.CHILD_OUTPUT_FIXED_RESERVE_TOKENS
            + primary_line_count * self.CHILD_OUTPUT_PRIMARY_COMPLEXITY_TOKENS
        )

    def _fingerprint(self) -> str:
        payload = {
            "version": self.VERSION,
            "prompt_policy_version": self.prompt_policy_version,
            "prompt_version": self.prompt_version,
            "single_call_prompt_version": self.single_call_prompt_version,
            "legacy_prompt_version": self.legacy_prompt_version,
            "legacy_candidate_schema_version": (
                self.legacy_candidate_schema_version
            ),
            "candidate_schema_version": self.candidate_schema_version,
            "semantic_prompt_version": self.semantic_prompt_version,
            "semantic_schema_version": self.semantic_schema_version,
            "semantic_mapping_version": self.semantic_mapping_version,
            "synthesis_prompt_version": self.synthesis_prompt_version,
            "synthesis_schema_version": self.synthesis_schema_version,
            "reconciliation_prompt_version": (
                self.reconciliation_prompt_version
            ),
            "single_call_max_lines": self.single_call_max_lines,
            "single_call_max_estimated_tokens": (
                self.single_call_max_estimated_tokens
            ),
            "route_max_lines": self.route_max_lines,
            "estimator": self.estimator,
            "primary_max_lines": self.primary_max_lines,
            "overlap_lines": self.overlap_lines,
            "max_windows": self.max_windows,
            "hard_max_lines": self.hard_max_lines,
            "hard_max_estimated_tokens": self.hard_max_estimated_tokens,
            "child_output_tokens": self.child_output_tokens,
            "child_planning_output_tokens": (
                self.CHILD_PLANNING_OUTPUT_TOKENS
            ),
            "child_length_retry_max_tokens": (
                self.child_length_retry_max_tokens
            ),
            "child_output_budget_tokens": self.child_output_budget_tokens,
            "child_output_fixed_reserve_tokens": (
                self.CHILD_OUTPUT_FIXED_RESERVE_TOKENS
            ),
            "child_output_primary_complexity_tokens": (
                self.CHILD_OUTPUT_PRIMARY_COMPLEXITY_TOKENS
            ),
            "output_primary_max_lines": self.output_primary_max_lines,
            "semantic_split_max_depth": self.semantic_split_max_depth,
            "semantic_split_min_primary_lines": (
                self.semantic_split_min_primary_lines
            ),
            "semantic_split_max_generation_requests": (
                self.semantic_split_max_generation_requests
            ),
            "reconciliation_max_candidates": self.reconciliation_max_candidates,
            "reconciliation_max_groups": self.reconciliation_max_groups,
            "reconciliation_max_cases": self.reconciliation_max_cases,
            "reconciliation_max_source_lines": (
                self.reconciliation_max_source_lines
            ),
            "reconciliation_max_estimated_tokens": (
                self.reconciliation_max_estimated_tokens
            ),
            "reconciliation_output_tokens": self.reconciliation_output_tokens,
            "synthesis_output_tokens": self.synthesis_output_tokens,
            "max_repair_attempts": self.max_repair_attempts,
        }
        serialized = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def default_windowing_policy(prompt_policy: PromptPolicy) -> WindowingPolicy:
    return WindowingPolicy(prompt_policy)


def evaluation_windowing_policy(
    prompt_policy: PromptPolicy,
    *,
    route_max_lines: int,
    primary_max_lines: int,
    overlap_lines: int,
) -> WindowingPolicy:
    return WindowingPolicy(
        prompt_policy,
        route_max_lines=route_max_lines,
        primary_max_lines=primary_max_lines,
        overlap_lines=overlap_lines,
    )
