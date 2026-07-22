from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace

from ....domain.source import SourceDocument, SourcePosition, TextSelection
from ...source import SavedSelection
from .models import DecompositionRoute
from .policy import WindowingPolicy


class WindowPlanError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class WindowSourceLine:
    position: SourcePosition
    text: str
    primary: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "page": self.position.page_index,
            "line": self.position.line_number,
            "text": self.text,
            "primary": self.primary,
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> WindowSourceLine:
        return cls(
            position=SourcePosition(int(value["page"]), int(value["line"])),
            text=str(value["text"]),
            primary=bool(value["primary"]),
        )


@dataclass(frozen=True, slots=True)
class DecompositionWindow:
    window_id: str
    index: int
    lines: tuple[WindowSourceLine, ...]
    global_start: SourcePosition
    global_end: SourcePosition
    outline_node_id: str
    outline_label: str
    outline_path: tuple[str, ...]
    input_max_lines: int
    input_max_estimated_tokens: int
    estimated_tokens: int
    output_max_tokens: int
    output_budget_tokens: int
    estimated_output_tokens: int
    policy_version: str

    @property
    def primary_positions(self) -> tuple[SourcePosition, ...]:
        return tuple(line.position for line in self.lines if line.primary)

    def as_selection(self) -> TextSelection:
        positions = tuple(line.position for line in self.lines)
        return TextSelection(
            start=positions[0],
            end=positions[-1],
            positions=positions,
            text="\n".join(line.text for line in self.lines),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "window_id": self.window_id,
            "index": self.index,
            "lines": [line.to_dict() for line in self.lines],
            "global_start": _position_dict(self.global_start),
            "global_end": _position_dict(self.global_end),
            "outline_node_id": self.outline_node_id,
            "outline_label": self.outline_label,
            "outline_path": list(self.outline_path),
            "input_max_lines": self.input_max_lines,
            "input_max_estimated_tokens": self.input_max_estimated_tokens,
            "estimated_tokens": self.estimated_tokens,
            "output_max_tokens": self.output_max_tokens,
            "output_budget_tokens": self.output_budget_tokens,
            "estimated_output_tokens": self.estimated_output_tokens,
            "policy_version": self.policy_version,
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> DecompositionWindow:
        return cls(
            window_id=str(value["window_id"]),
            index=int(value["index"]),
            lines=tuple(
                WindowSourceLine.from_dict(dict(item))
                for item in value["lines"]  # type: ignore[union-attr]
            ),
            global_start=_position_from_dict(dict(value["global_start"])),
            global_end=_position_from_dict(dict(value["global_end"])),
            outline_node_id=str(value["outline_node_id"]),
            outline_label=str(value["outline_label"]),
            outline_path=tuple(str(item) for item in value["outline_path"]),  # type: ignore[union-attr]
            input_max_lines=int(value["input_max_lines"]),
            input_max_estimated_tokens=int(value["input_max_estimated_tokens"]),
            estimated_tokens=int(value["estimated_tokens"]),
            output_max_tokens=int(value["output_max_tokens"]),
            output_budget_tokens=int(value["output_budget_tokens"]),
            estimated_output_tokens=int(value["estimated_output_tokens"]),
            policy_version=str(value["policy_version"]),
        )


@dataclass(frozen=True, slots=True)
class WindowPlan:
    schema_version: str
    selection_id: str
    document_version: str
    selection_start: SourcePosition
    selection_end: SourcePosition
    policy_version: str
    windows: tuple[DecompositionWindow, ...]
    plan_hash: str

    def to_dict(self) -> dict[str, object]:
        return {
            **self._hash_payload(),
            "plan_hash": self.plan_hash,
        }

    def _hash_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "selection_id": self.selection_id,
            "document_version": self.document_version,
            "selection_start": _position_dict(self.selection_start),
            "selection_end": _position_dict(self.selection_end),
            "policy_version": self.policy_version,
            "windows": [window.to_dict() for window in self.windows],
        }

    def recompute_hash(self) -> str:
        serialized = json.dumps(
            self._hash_payload(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def validate(self, selection: SavedSelection) -> None:
        if self.schema_version != WindowPlanner.SCHEMA_VERSION:
            raise WindowPlanError(
                f"Неизвестная window plan schema {self.schema_version}"
            )
        if self.recompute_hash() != self.plan_hash:
            raise WindowPlanError("Window plan hash не соответствует содержимому")
        if self.selection_id != selection.selection_id:
            raise WindowPlanError("Window plan относится к другому selection")
        if self.document_version != selection.document_version:
            raise WindowPlanError("Window plan document_version устарел")
        primary_lines = tuple(
            line
            for window in self.windows
            for line in window.lines
            if line.primary
        )
        if tuple(line.position for line in primary_lines) != selection.selection.positions:
            raise WindowPlanError("Primary ranges не покрывают selection")
        if "\n".join(line.text for line in primary_lines) != selection.selection.text:
            raise WindowPlanError("Window plan source text не совпадает с selection")

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> WindowPlan:
        plan = cls(
            schema_version=str(value["schema_version"]),
            selection_id=str(value["selection_id"]),
            document_version=str(value["document_version"]),
            selection_start=_position_from_dict(dict(value["selection_start"])),
            selection_end=_position_from_dict(dict(value["selection_end"])),
            policy_version=str(value["policy_version"]),
            windows=tuple(
                DecompositionWindow.from_dict(dict(item))
                for item in value["windows"]  # type: ignore[union-attr]
            ),
            plan_hash=str(value["plan_hash"]),
        )
        if plan.recompute_hash() != plan.plan_hash:
            raise WindowPlanError("Window plan hash не соответствует содержимому")
        return plan


class WindowPlanner:
    SCHEMA_VERSION = "window-plan-2"

    def __init__(self, document: SourceDocument, policy: WindowingPolicy) -> None:
        self.document = document
        self.policy = policy

    def build(self, selection: SavedSelection) -> WindowPlan:
        decision = self.policy.assess(
            selection.selection,
            selection_id=selection.selection_id,
        )
        if decision.route is DecompositionRoute.HARD_LIMIT:
            raise WindowPlanError("Selection превышает hard limit windowed Prompt 1")
        if selection.document_version != self.document.metadata.document_version:
            raise WindowPlanError("Selection document_version не совпадает с source")
        self._validate_source(selection)

        positions = selection.selection.positions
        windows: list[DecompositionWindow] = []
        primary_start = 0
        while primary_start < len(positions):
            if len(windows) >= self.policy.max_windows:
                raise WindowPlanError("Window plan превышает max_windows")
            window_id = (
                f"{selection.selection_id}:WINDOW:{len(windows) + 1:04d}"
            )
            primary_count = self._largest_primary(
                positions,
                primary_start,
                window_id,
            )
            primary_end = primary_start + primary_count
            before, after = self._bounded_overlap(
                positions,
                primary_start,
                primary_end,
                window_id,
            )
            context_start = primary_start - before
            context_end = primary_end + after
            window_positions = positions[context_start:context_end]
            window_selection = self._selection(window_positions)
            assessment = self.policy.assess_window(
                window_selection,
                window_id=window_id,
            )
            if assessment.route is not DecompositionRoute.SINGLE_CALL:
                raise WindowPlanError("Window payload не помещается в single-call budget")
            outline = self.document.outline_at(
                positions[primary_start],
                preferred_section_id=selection.anchor_outline_node_id,
            )
            primary_set = set(positions[primary_start:primary_end])
            estimated_output_tokens = (
                self.policy.estimate_child_output(primary_count)
            )
            if (
                estimated_output_tokens
                > self.policy.child_output_budget_tokens
            ):
                raise WindowPlanError(
                    "Window result не помещается в child output budget"
                )
            windows.append(
                DecompositionWindow(
                    window_id=window_id,
                    index=len(windows),
                    lines=tuple(
                        WindowSourceLine(
                            position,
                            self.document.line(position),
                            position in primary_set,
                        )
                        for position in window_positions
                    ),
                    global_start=selection.selection.start,
                    global_end=selection.selection.end,
                    outline_node_id=outline.section_id,
                    outline_label=outline.label,
                    outline_path=outline.path,
                    input_max_lines=self.policy.single_call_max_lines,
                    input_max_estimated_tokens=(
                        self.policy.single_call_max_estimated_tokens
                    ),
                    estimated_tokens=assessment.budget.estimated_tokens,
                    output_max_tokens=self.policy.child_output_tokens,
                    output_budget_tokens=(
                        self.policy.child_output_budget_tokens
                    ),
                    estimated_output_tokens=estimated_output_tokens,
                    policy_version=self.policy.fingerprint,
                )
            )
            primary_start = primary_end

        provisional = WindowPlan(
            schema_version=self.SCHEMA_VERSION,
            selection_id=selection.selection_id,
            document_version=selection.document_version,
            selection_start=selection.selection.start,
            selection_end=selection.selection.end,
            policy_version=self.policy.fingerprint,
            windows=tuple(windows),
            plan_hash="",
        )
        plan = replace(provisional, plan_hash=provisional.recompute_hash())
        plan.validate(selection)
        return plan

    def _largest_primary(
        self,
        positions: tuple[SourcePosition, ...],
        start: int,
        window_id: str,
    ) -> int:
        high = min(self.policy.primary_max_lines, len(positions) - start)
        low = 1
        if not self._fits(positions[start : start + 1], window_id):
            raise WindowPlanError("Даже одна строка не помещается в window budget")
        while low < high:
            middle = (low + high + 1) // 2
            if self._fits(positions[start : start + middle], window_id):
                low = middle
            else:
                high = middle - 1
        return low

    def _bounded_overlap(
        self,
        positions: tuple[SourcePosition, ...],
        primary_start: int,
        primary_end: int,
        window_id: str,
    ) -> tuple[int, int]:
        before = 0
        after = 0
        before_blocked = primary_start == 0
        after_blocked = primary_end == len(positions)
        while not (before_blocked and after_blocked):
            changed = False
            if not before_blocked:
                candidate = min(
                    self.policy.overlap_lines,
                    before + 1,
                    primary_start,
                )
                if candidate == before:
                    before_blocked = True
                elif self._fits(
                    positions[primary_start - candidate : primary_end + after],
                    window_id,
                ):
                    before = candidate
                    changed = True
                else:
                    before_blocked = True
            if not after_blocked:
                candidate = min(
                    self.policy.overlap_lines,
                    after + 1,
                    len(positions) - primary_end,
                )
                if candidate == after:
                    after_blocked = True
                elif self._fits(
                    positions[primary_start - before : primary_end + candidate],
                    window_id,
                ):
                    after = candidate
                    changed = True
                else:
                    after_blocked = True
            if not changed and before_blocked and after_blocked:
                break
        return before, after

    def _fits(
        self,
        positions: tuple[SourcePosition, ...],
        window_id: str,
    ) -> bool:
        return (
            self.policy.assess_window(
                self._selection(positions),
                window_id=window_id,
            ).route
            is DecompositionRoute.SINGLE_CALL
        )

    def _selection(
        self,
        positions: tuple[SourcePosition, ...],
    ) -> TextSelection:
        return TextSelection(
            start=positions[0],
            end=positions[-1],
            positions=positions,
            text="\n".join(self.document.line(position) for position in positions),
        )

    def _validate_source(self, selection: SavedSelection) -> None:
        for position, text in zip(
            selection.selection.positions,
            selection.selection.text.split("\n"),
            strict=True,
        ):
            if self.document.line(position) != text:
                raise WindowPlanError(
                    f"Selection source text не совпадает в {position}"
                )


def _position_dict(position: SourcePosition) -> dict[str, int]:
    return {
        "page": position.page_index,
        "line": position.line_number,
    }


def _position_from_dict(value: dict[str, object]) -> SourcePosition:
    return SourcePosition(int(value["page"]), int(value["line"]))
