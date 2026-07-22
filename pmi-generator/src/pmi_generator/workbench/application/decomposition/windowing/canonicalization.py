from __future__ import annotations

import json
from dataclasses import dataclass

from ....domain.source import SourceDocument, SourcePosition
from ..models import DecompositionError
from ..validation import DecompositionValidator
from .candidates import (
    PrimaryLineAssessment,
    ValidatedWindowCandidate,
    WindowCandidateResult,
    validate_window_binding,
)
from .models import WindowedAttemptState, WindowedAttemptStatus
from .plan import DecompositionWindow, WindowPlan, WindowSourceLine
from .semantic import (
    SEMANTIC_CANONICAL_MAPPING_VERSION,
    SEMANTIC_WINDOW_SCHEMA_VERSION,
    SemanticBehaviorFragment,
    SemanticFact,
    SemanticWindowArguments,
    SemanticWindowResult,
)
from .synthesis import (
    SYNTHESIS_REQUIRED_FIELDS,
    SYNTHESIS_SINGLETON_FIELDS,
    SemanticFactScope,
    SemanticSynthesisArguments,
    semantic_fact_scopes,
)


class SemanticWindowError(ValueError):
    pass


_MISSING_PART_POLICY = {
    "input_value": (
        "test_design.input_value",
        "Какое конкретное входное значение использовать для теста «{title}»?",
    ),
    "action": (
        "test_design.action",
        "Какое конкретное воздействие выполнить для теста «{title}»?",
    ),
}
@dataclass(frozen=True, slots=True)
class _Fact:
    text: str
    positions: tuple[SourcePosition, ...]


@dataclass(frozen=True, slots=True)
class _RawFragment:
    title: str
    summary: str
    facts: tuple[_Fact, ...]


@dataclass(frozen=True, slots=True)
class _SynthesisPart:
    role: str
    text: str
    fact_ids: tuple[str, ...]
    positions: tuple[SourcePosition, ...]


@dataclass(frozen=True, slots=True)
class _SynthesisCandidate:
    title: str
    parts: tuple[_SynthesisPart, ...]


class SemanticWindowCanonicalizer:
    """Validate fact-only child output without creating card candidates."""

    MAX_BEHAVIORS = 64
    MAX_FACTS_PER_BEHAVIOR = 32
    MAX_FACTS_PER_WINDOW = 256

    def __init__(self, document: SourceDocument) -> None:
        self.document = document

    def canonicalize(
        self,
        *,
        parent: WindowedAttemptState,
        plan: WindowPlan,
        window_id: str,
        child_attempt_id: str,
        arguments: SemanticWindowArguments,
    ) -> SemanticWindowResult:
        window = validate_window_binding(
            parent,
            plan,
            window_id,
            child_attempt_id,
            expected_schema_version=SEMANTIC_WINDOW_SCHEMA_VERSION,
        )
        return self._canonicalize(
            parent=parent,
            plan=plan,
            window=window,
            result_window_id=window.window_id,
            result_child_attempt_id=child_attempt_id,
            arguments=arguments,
        )

    def canonicalize_subwindow(
        self,
        *,
        parent: WindowedAttemptState,
        plan: WindowPlan,
        logical_window_id: str,
        logical_child_attempt_id: str,
        subwindow: DecompositionWindow,
        generation_attempt_id: str,
        arguments: SemanticWindowArguments,
    ) -> SemanticWindowResult:
        logical_window = validate_window_binding(
            parent,
            plan,
            logical_window_id,
            logical_child_attempt_id,
            expected_schema_version=SEMANTIC_WINDOW_SCHEMA_VERSION,
        )
        if tuple(
            (line.position, line.text) for line in subwindow.lines
        ) != tuple(
            (line.position, line.text) for line in logical_window.lines
        ):
            raise SemanticWindowError(
                "Semantic subwindow изменяет immutable source context"
            )
        logical_primary = set(logical_window.primary_positions)
        subwindow_primary = set(subwindow.primary_positions)
        if (
            not subwindow_primary
            or not subwindow_primary.issubset(logical_primary)
        ):
            raise SemanticWindowError(
                "Semantic subwindow primary range не входит в logical window"
            )
        return self._canonicalize(
            parent=parent,
            plan=plan,
            window=subwindow,
            result_window_id=subwindow.window_id,
            result_child_attempt_id=generation_attempt_id,
            arguments=arguments,
        )

    def _canonicalize(
        self,
        *,
        parent: WindowedAttemptState,
        plan: WindowPlan,
        window: DecompositionWindow,
        result_window_id: str,
        result_child_attempt_id: str,
        arguments: SemanticWindowArguments,
    ) -> SemanticWindowResult:
        errors: list[str] = []
        fragments = self._fragments(window, arguments.behaviors, errors)
        if errors:
            raise SemanticWindowError(self._error_report(errors))
        ordered = sorted(
            fragments,
            key=lambda item: (
                self.document.position_index(item.facts[0].positions[0]),
                item.title,
                item.summary,
                json.dumps(
                    [
                        {
                            "text": fact.text,
                            "positions": [
                                [
                                    position.page_index,
                                    position.line_number,
                                ]
                                for position in fact.positions
                            ],
                        }
                        for fact in item.facts
                    ],
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            ),
        )
        canonical: list[SemanticBehaviorFragment] = []
        for fragment_index, fragment in enumerate(ordered, start=1):
            fragment_id = (
                f"{result_window_id}:FRAGMENT:{fragment_index:04d}"
            )
            facts = tuple(
                SemanticFact(
                    fact_id=f"{fragment_id}:FACT:{fact_index:04d}",
                    text=fact.text,
                    positions=fact.positions,
                )
                for fact_index, fact in enumerate(fragment.facts, start=1)
            )
            canonical.append(
                SemanticBehaviorFragment(
                    fragment_id=fragment_id,
                    window_id=result_window_id,
                    title=fragment.title,
                    summary=fragment.summary,
                    facts=facts,
                )
            )
        return SemanticWindowResult(
            parent_attempt_id=parent.parent_attempt_id,
            child_attempt_id=result_child_attempt_id,
            window_id=result_window_id,
            plan_hash=plan.plan_hash,
            fragments=tuple(canonical),
        )

    def owned_arguments(
        self,
        window: DecompositionWindow,
        arguments: SemanticWindowArguments,
    ) -> SemanticWindowArguments:
        line_map = self._line_map(window)
        primary = set(window.primary_positions)
        owned: list[dict[str, object]] = []
        for behavior in arguments.behaviors:
            line_ids = {
                str(line_id)
                for fact in behavior["facts"]  # type: ignore[union-attr]
                for line_id in fact["line_ids"]
            }
            positions = {
                line_map[line_id].position
                for line_id in line_ids
                if line_id in line_map
            }
            if positions.intersection(primary):
                owned.append(behavior)
        return SemanticWindowArguments(
            behaviors=json.loads(
                json.dumps(owned, ensure_ascii=False)
            )
        )

    def _fragments(
        self,
        window: DecompositionWindow,
        raw_behaviors: object,
        errors: list[str],
    ) -> tuple[_RawFragment, ...]:
        if not isinstance(raw_behaviors, list):
            errors.append("behaviors должен быть списком")
            return ()
        if len(raw_behaviors) > self.MAX_BEHAVIORS:
            errors.append(
                f"behaviors превышает лимит {self.MAX_BEHAVIORS}"
            )
        line_map = self._line_map(window)
        result: list[_RawFragment] = []
        primary = set(window.primary_positions)
        fact_count = 0
        for index, raw in enumerate(raw_behaviors, start=1):
            prefix = f"behavior {index}"
            if not isinstance(raw, dict) or set(raw) != {
                "title",
                "summary",
                "facts",
            }:
                errors.append(f"{prefix}: неверная структура")
                continue
            title = str(raw["title"]).strip()
            summary = str(raw["summary"]).strip()
            if not title:
                errors.append(f"{prefix}: title обязателен")
            if not summary:
                errors.append(f"{prefix}: summary обязателен")
            facts = self._facts(prefix, raw["facts"], line_map, errors)
            fact_count += len(facts)
            positions = {
                position
                for fact in facts
                for position in fact.positions
            }
            if (
                title
                and summary
                and facts
                and positions.intersection(primary)
            ):
                result.append(_RawFragment(title, summary, facts))
        if fact_count > self.MAX_FACTS_PER_WINDOW:
            errors.append(
                "facts окна превышают лимит "
                f"{self.MAX_FACTS_PER_WINDOW}"
            )
        return tuple(result)

    def _facts(
        self,
        prefix: str,
        raw_facts: object,
        line_map: dict[str, WindowSourceLine],
        errors: list[str],
    ) -> tuple[_Fact, ...]:
        if not isinstance(raw_facts, list) or not raw_facts:
            errors.append(f"{prefix}: facts должен быть непустым списком")
            return ()
        if len(raw_facts) > self.MAX_FACTS_PER_BEHAVIOR:
            errors.append(
                f"{prefix}: facts превышает лимит "
                f"{self.MAX_FACTS_PER_BEHAVIOR}"
            )
        result: list[_Fact] = []
        for index, raw in enumerate(raw_facts, start=1):
            item_prefix = f"{prefix} fact {index}"
            if not isinstance(raw, dict) or set(raw) != {
                "text",
                "line_ids",
            }:
                errors.append(f"{item_prefix}: неверная структура")
                continue
            text = str(raw["text"]).strip()
            if not text:
                errors.append(f"{item_prefix}: text обязателен")
            positions = self._positions(
                item_prefix,
                raw["line_ids"],
                line_map,
                errors,
            )
            if text and positions:
                result.append(_Fact(text, positions))
        return tuple(
            sorted(
                result,
                key=lambda item: (
                    self.document.position_index(item.positions[0]),
                    item.text,
                ),
            )
        )

    def _positions(
        self,
        prefix: str,
        raw_line_ids: object,
        line_map: dict[str, WindowSourceLine],
        errors: list[str],
    ) -> tuple[SourcePosition, ...]:
        if not isinstance(raw_line_ids, list) or not raw_line_ids:
            errors.append(f"{prefix}: line_ids должен быть непустым списком")
            return ()
        line_ids = tuple(str(item) for item in raw_line_ids)
        unknown = sorted(set(line_ids) - set(line_map))
        if unknown:
            errors.append(
                f"{prefix}: неизвестные line_ids {', '.join(unknown)}"
            )
        return tuple(
            sorted(
                {
                    line_map[line_id].position
                    for line_id in line_ids
                    if line_id in line_map
                },
                key=self.document.position_index,
            )
        )

    @staticmethod
    def _line_map(
        window: DecompositionWindow,
    ) -> dict[str, WindowSourceLine]:
        return {
            f"L{index:04d}": line
            for index, line in enumerate(window.lines, start=1)
        }

    @staticmethod
    def _error_report(errors: list[str]) -> str:
        return "Semantic window facts отклонены:\n- " + "\n- ".join(
            sorted(set(errors))
        )


class SemanticSynthesisCanonicalizer:
    """Build canonical window candidates from synthesis and validated facts."""

    MAX_CANDIDATES = 64
    MAPPING_VERSION = SEMANTIC_CANONICAL_MAPPING_VERSION

    def __init__(self, document: SourceDocument) -> None:
        self.document = document
        self.validator = DecompositionValidator(document)

    def canonicalize(
        self,
        *,
        parent: WindowedAttemptState,
        plan: WindowPlan,
        target_window_id: str,
        attempt_id: str,
        fact_results: tuple[SemanticWindowResult, ...],
        arguments: SemanticSynthesisArguments,
    ) -> WindowCandidateResult:
        window = self._validate_binding(
            parent=parent,
            plan=plan,
            target_window_id=target_window_id,
            attempt_id=attempt_id,
            fact_results=fact_results,
        )
        fact_scopes = semantic_fact_scopes(
            plan=plan,
            target_window_id=target_window_id,
            results=fact_results,
        )
        fact_map = self._fact_map(set(fact_scopes), fact_results)
        primary_fact_ids = {
            fact_id
            for fact_id, scope in fact_scopes.items()
            if scope is SemanticFactScope.PRIMARY
        }
        errors: list[str] = []
        candidates, context_only_count = self._candidates(
            arguments.candidates,
            fact_map,
            primary_fact_ids,
            errors,
        )
        if primary_fact_ids and not candidates:
            errors.append(
                "Semantic synthesis не вернул candidate для primary facts "
                "target window"
            )
        if errors:
            raise SemanticWindowError(self._error_report(errors))
        prepared: list[
            tuple[_SynthesisCandidate, dict[str, object], tuple[SourcePosition, ...]]
        ] = []
        for candidate in candidates:
            try:
                payload, positions = self._payload(plan, window, candidate)
            except (DecompositionError, ValueError) as error:
                errors.append(f"«{candidate.title}»: {error}")
                continue
            prepared.append((candidate, payload, positions))
        if errors:
            raise SemanticWindowError(self._error_report(errors))
        prepared.sort(
            key=lambda item: (
                self.document.position_index(item[2][0]),
                item[0].title,
                json.dumps(
                    item[1],
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            )
        )
        canonical_candidates = tuple(
            ValidatedWindowCandidate(
                candidate_id=(
                    f"{target_window_id}:CANDIDATE:{index:04d}"
                ),
                local_candidate_id=f"SYNTHESIS:{index:04d}",
                window_id=target_window_id,
                payload=payload,
                evidence_positions=positions,
            )
            for index, (_candidate, payload, positions) in enumerate(
                prepared,
                start=1,
            )
        )
        primary_evidence = {
            position
            for candidate in canonical_candidates
            for position in candidate.evidence_positions
            if position in set(window.primary_positions)
        }
        assessments = tuple(
            PrimaryLineAssessment(
                position,
                "evidence" if position in primary_evidence else "context",
                (
                    "Строка выбрана semantic synthesis"
                    if position in primary_evidence
                    else "Строка не выбрана semantic synthesis"
                ),
            )
            for position in window.primary_positions
        )
        explanation = (
            f"Application собрал {len(canonical_candidates)} candidates "
            "из validated semantic facts"
            if canonical_candidates
            else "Semantic synthesis не вернул локальных candidates"
        )
        if context_only_count:
            explanation += (
                f"; application исключил {context_only_count} "
                "context-only candidates"
            )
        return WindowCandidateResult(
            parent_attempt_id=parent.parent_attempt_id,
            child_attempt_id=next(
                child.attempt_id
                for child in parent.children
                if child.window_id == target_window_id
            )
            or "",
            window_id=target_window_id,
            plan_hash=plan.plan_hash,
            outcome=(
                "candidates"
                if canonical_candidates
                else "no_local_testable_behavior"
            ),
            explanation=explanation,
            candidates=canonical_candidates,
            boundary_dependencies=(),
            primary_line_assessments=assessments,
        )

    def empty_result(
        self,
        *,
        parent: WindowedAttemptState,
        plan: WindowPlan,
        target_window_id: str,
    ) -> WindowCandidateResult:
        return self.canonicalize(
            parent=parent,
            plan=plan,
            target_window_id=target_window_id,
            attempt_id=f"{target_window_id}:NO_SYNTHESIS",
            fact_results=tuple(
                SemanticWindowResult(
                    parent_attempt_id=parent.parent_attempt_id,
                    child_attempt_id=child.attempt_id or "",
                    window_id=child.window_id,
                    plan_hash=plan.plan_hash,
                    fragments=(),
                )
                for child in parent.children
            ),
            arguments=SemanticSynthesisArguments(candidates=[]),
        )

    def _validate_binding(
        self,
        *,
        parent: WindowedAttemptState,
        plan: WindowPlan,
        target_window_id: str,
        attempt_id: str,
        fact_results: tuple[SemanticWindowResult, ...],
    ) -> DecompositionWindow:
        if parent.status is not WindowedAttemptStatus.RECONCILING:
            raise SemanticWindowError(
                "Semantic synthesis требует parent в стадии reconciliation"
            )
        if not attempt_id.strip():
            raise SemanticWindowError(
                "Semantic synthesis требует attempt ID"
            )
        if (
            parent.window_plan_hash != plan.plan_hash
            or plan.recompute_hash() != plan.plan_hash
            or parent.selection_id != plan.selection_id
            or parent.document_version != plan.document_version
            or parent.policy_version != plan.policy_version
        ):
            raise SemanticWindowError(
                "Semantic synthesis имеет stale parent/plan binding"
            )
        by_window = {result.window_id: result for result in fact_results}
        expected = tuple(window.window_id for window in plan.windows)
        if len(by_window) != len(fact_results) or set(by_window) != set(expected):
            raise SemanticWindowError(
                "Semantic facts не покрывают window plan ровно один раз"
            )
        for child in parent.children:
            result = by_window[child.window_id]
            if (
                result.parent_attempt_id != parent.parent_attempt_id
                or result.child_attempt_id != child.attempt_id
                or result.plan_hash != plan.plan_hash
            ):
                raise SemanticWindowError(
                    "Semantic facts имеют stale parent/child/plan binding"
                )
        try:
            return next(
                item
                for item in plan.windows
                if item.window_id == target_window_id
            )
        except StopIteration as error:
            raise SemanticWindowError(
                f"Неизвестное synthesis window {target_window_id}"
            ) from error

    def _fact_map(
        self,
        allowed_fact_ids: set[str],
        results: tuple[SemanticWindowResult, ...],
    ) -> dict[str, SemanticFact]:
        return {
            fact.fact_id: fact
            for result in results
            for fragment in result.fragments
            for fact in fragment.facts
            if fact.fact_id in allowed_fact_ids
        }

    def _candidates(
        self,
        raw_candidates: object,
        fact_map: dict[str, SemanticFact],
        primary_fact_ids: set[str],
        errors: list[str],
    ) -> tuple[tuple[_SynthesisCandidate, ...], int]:
        if not isinstance(raw_candidates, list):
            errors.append("candidates должен быть списком")
            return (), 0
        if len(raw_candidates) > self.MAX_CANDIDATES:
            errors.append(
                f"candidates превышает лимит {self.MAX_CANDIDATES}"
            )
        result: list[_SynthesisCandidate] = []
        context_only_count = 0
        for index, raw in enumerate(raw_candidates, start=1):
            prefix = f"candidate {index}"
            if not isinstance(raw, dict):
                errors.append(f"{prefix}: неверная структура")
                continue
            required = {"title", *SYNTHESIS_REQUIRED_FIELDS}
            allowed = {*required, "input_value", "action"}
            missing = sorted(required - set(raw))
            unknown = sorted(set(raw) - allowed)
            if missing:
                errors.append(
                    f"{prefix}: отсутствуют поля {', '.join(missing)}"
                )
            if unknown:
                errors.append(
                    f"{prefix}: неизвестные поля {', '.join(unknown)}"
                )
            title = str(raw.get("title", "")).strip()
            if not title:
                errors.append(f"{prefix}: title обязателен")
            parts = self._named_parts(prefix, raw, fact_map, errors)
            referenced = {
                fact_id for part in parts for fact_id in part.fact_ids
            }
            if parts and not referenced.intersection(primary_fact_ids):
                context_only_count += 1
                continue
            if title and parts:
                result.append(_SynthesisCandidate(title, parts))
        return tuple(result), context_only_count

    def _named_parts(
        self,
        prefix: str,
        raw_candidate: dict[str, object],
        fact_map: dict[str, SemanticFact],
        errors: list[str],
    ) -> tuple[_SynthesisPart, ...]:
        result: list[_SynthesisPart] = []
        for role in SYNTHESIS_SINGLETON_FIELDS:
            if role not in raw_candidate:
                continue
            part = self._named_part(
                f"{prefix} {role}",
                role,
                raw_candidate[role],
                fact_map,
                errors,
            )
            if part is not None:
                result.append(part)
        raw_consequences = raw_candidate.get("consequences")
        if not isinstance(raw_consequences, list) or not raw_consequences:
            errors.append(
                f"{prefix}: consequences должен быть непустым списком"
            )
            return tuple(result)
        for index, raw in enumerate(raw_consequences, start=1):
            part = self._named_part(
                f"{prefix} consequence {index}",
                "consequence",
                raw,
                fact_map,
                errors,
            )
            if part is not None:
                result.append(part)
        return tuple(result)

    def _named_part(
        self,
        prefix: str,
        role: str,
        raw: object,
        fact_map: dict[str, SemanticFact],
        errors: list[str],
    ) -> _SynthesisPart | None:
        if not isinstance(raw, dict) or set(raw) != {"text", "fact_ids"}:
            errors.append(f"{prefix}: неверная структура")
            return None
        text = str(raw["text"]).strip()
        if not text:
            errors.append(f"{prefix}: text обязателен")
        raw_ids = raw["fact_ids"]
        if not isinstance(raw_ids, list) or not raw_ids:
            errors.append(
                f"{prefix}: fact_ids должен быть непустым списком"
            )
            return None
        fact_ids = tuple(sorted(set(str(item) for item in raw_ids)))
        unknown = sorted(set(fact_ids) - set(fact_map))
        if unknown:
            errors.append(
                f"{prefix}: неизвестные fact_ids {', '.join(unknown)}"
            )
        positions = tuple(
            sorted(
                {
                    position
                    for fact_id in fact_ids
                    if fact_id in fact_map
                    for position in fact_map[fact_id].positions
                },
                key=self.document.position_index,
            )
        )
        if not text or not positions:
            return None
        return _SynthesisPart(role, text, fact_ids, positions)

    def _payload(
        self,
        plan: WindowPlan,
        window: DecompositionWindow,
        candidate: _SynthesisCandidate,
    ) -> tuple[dict[str, object], tuple[SourcePosition, ...]]:
        by_role: dict[str, list[_SynthesisPart]] = {}
        for part in candidate.parts:
            by_role.setdefault(part.role, []).append(part)
        condition = by_role["condition"][0]
        changed_factor = by_role["changed_factor"][0]
        input_part = by_role.get("input_value", [None])[0]
        action_part = by_role.get("action", [None])[0]
        consequences = sorted(
            by_role["consequence"],
            key=lambda item: (
                self.document.position_index(item.positions[0]),
                item.text,
            ),
        )
        gaps = [
            {
                "kind": role,
                "question": policy[1].format(title=candidate.title),
                "target_paths": [policy[0]],
            }
            for role, policy in _MISSING_PART_POLICY.items()
            if not by_role.get(role)
        ]
        raw = {
            "title": candidate.title,
            "condition": condition.text,
            "changed_factor": changed_factor.text,
            "input_value": input_part.text if input_part is not None else None,
            "action": action_part.text if action_part is not None else None,
            "condition_ranges": self._ranges(condition.positions),
            "changed_factor_ranges": self._ranges(
                changed_factor.positions
            ),
            "input_value_ranges": (
                self._ranges(input_part.positions)
                if input_part is not None
                else []
            ),
            "action_ranges": (
                self._ranges(action_part.positions)
                if action_part is not None
                else []
            ),
            "consequences": [
                {
                    "text": item.text,
                    "evidence_ranges": self._ranges(item.positions),
                }
                for item in consequences
            ],
            "gaps": gaps,
        }
        payload, evidence_positions = self.validator.validate_skeleton(
            self.document.select(
                plan.selection_start,
                plan.selection_end,
            ),
            raw,
        )
        if not evidence_positions.intersection(window.primary_positions):
            raise SemanticWindowError(
                "Application fact ownership invariant нарушен: canonical "
                "candidate не содержит primary evidence"
            )
        return (
            payload,
            tuple(
                sorted(
                    evidence_positions,
                    key=self.document.position_index,
                )
            ),
        )

    def _ranges(
        self,
        positions: tuple[SourcePosition, ...],
    ) -> list[dict[str, int]]:
        ordered = tuple(
            sorted(set(positions), key=self.document.position_index)
        )
        result: list[dict[str, int]] = []
        if not ordered:
            return result
        start = previous = ordered[0]
        for position in ordered[1:]:
            if (
                position.page_index == previous.page_index
                and position.line_number == previous.line_number + 1
            ):
                previous = position
                continue
            result.append(self._range(start, previous))
            start = previous = position
        result.append(self._range(start, previous))
        return result

    @staticmethod
    def _range(
        start: SourcePosition,
        end: SourcePosition,
    ) -> dict[str, int]:
        return {
            "page": start.page_index,
            "line_start": start.line_number,
            "line_end": end.line_number,
        }

    @staticmethod
    def _error_report(errors: list[str]) -> str:
        return "Semantic synthesis отклонён:\n- " + "\n- ".join(
            sorted(set(errors))
        )
