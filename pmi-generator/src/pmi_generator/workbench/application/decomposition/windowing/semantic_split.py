from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import StrEnum

from ....domain.source import SourcePosition
from ...repositories import UnitOfWork
from ...state import StoredRecord
from .plan import DecompositionWindow, WindowPlan, WindowSourceLine


class SemanticSubwindowError(ValueError):
    pass


class SemanticSubwindowStatus(StrEnum):
    PLANNED = "planned"
    RUNNING = "running"
    SPLIT = "split"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SemanticCoordinatorStatus(StrEnum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class SemanticSubwindowNode:
    node_id: str
    parent_node_id: str | None
    depth: int
    primary_positions: tuple[SourcePosition, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "node_id": self.node_id,
            "parent_node_id": self.parent_node_id,
            "depth": self.depth,
            "primary_positions": [
                _position_dict(item) for item in self.primary_positions
            ],
        }

    @classmethod
    def from_dict(
        cls,
        value: dict[str, object],
    ) -> SemanticSubwindowNode:
        return cls(
            node_id=str(value["node_id"]),
            parent_node_id=(
                str(value["parent_node_id"])
                if value.get("parent_node_id") is not None
                else None
            ),
            depth=int(value["depth"]),
            primary_positions=tuple(
                _position_from_dict(dict(item))
                for item in value["primary_positions"]  # type: ignore[union-attr]
            ),
        )


@dataclass(frozen=True, slots=True)
class SemanticSubwindowPlan:
    schema_version: str
    parent_attempt_id: str
    logical_child_attempt_id: str
    logical_window_id: str
    parent_plan_hash: str
    context_fingerprint: str
    policy_version: str
    prompt_version: str
    contract_version: str
    max_depth: int
    min_primary_lines: int
    max_generation_requests: int
    nodes: tuple[SemanticSubwindowNode, ...]
    plan_hash: str

    @property
    def root(self) -> SemanticSubwindowNode:
        roots = tuple(
            item for item in self.nodes if item.parent_node_id is None
        )
        if len(roots) != 1:
            raise SemanticSubwindowError(
                "Semantic subwindow plan должен содержать один root"
            )
        return roots[0]

    def node(self, node_id: str) -> SemanticSubwindowNode:
        try:
            return next(item for item in self.nodes if item.node_id == node_id)
        except StopIteration as error:
            raise SemanticSubwindowError(
                f"Неизвестный semantic subwindow {node_id}"
            ) from error

    def children(
        self,
        node_id: str,
    ) -> tuple[SemanticSubwindowNode, ...]:
        self.node(node_id)
        return tuple(
            item for item in self.nodes if item.parent_node_id == node_id
        )

    def masked_window(
        self,
        window: DecompositionWindow,
        node_id: str,
    ) -> DecompositionWindow:
        self._validate_context(window)
        node = self.node(node_id)
        primary = set(node.primary_positions)
        return replace(
            window,
            window_id=node.node_id,
            lines=tuple(
                WindowSourceLine(
                    position=line.position,
                    text=line.text,
                    primary=line.position in primary,
                )
                for line in window.lines
            ),
        )

    def validate_binding(
        self,
        *,
        parent_attempt_id: str,
        logical_child_attempt_id: str,
        parent_plan: WindowPlan,
        window: DecompositionWindow,
    ) -> None:
        if (
            self.parent_attempt_id != parent_attempt_id
            or self.logical_child_attempt_id != logical_child_attempt_id
        ):
            raise SemanticSubwindowError(
                "Semantic subwindow plan имеет stale parent binding"
            )
        if (
            self.logical_window_id != window.window_id
            or self.parent_plan_hash != parent_plan.plan_hash
            or parent_plan.recompute_hash() != parent_plan.plan_hash
        ):
            raise SemanticSubwindowError(
                "Semantic subwindow plan имеет stale window binding"
            )
        self._validate_context(window)
        self._validate_structure()

    def to_dict(self) -> dict[str, object]:
        return {**self._hash_payload(), "plan_hash": self.plan_hash}

    @classmethod
    def from_dict(
        cls,
        value: dict[str, object],
    ) -> SemanticSubwindowPlan:
        plan = cls(
            schema_version=str(value["schema_version"]),
            parent_attempt_id=str(value["parent_attempt_id"]),
            logical_child_attempt_id=str(
                value["logical_child_attempt_id"]
            ),
            logical_window_id=str(value["logical_window_id"]),
            parent_plan_hash=str(value["parent_plan_hash"]),
            context_fingerprint=str(value["context_fingerprint"]),
            policy_version=str(value["policy_version"]),
            prompt_version=str(value["prompt_version"]),
            contract_version=str(value["contract_version"]),
            max_depth=int(value["max_depth"]),
            min_primary_lines=int(value["min_primary_lines"]),
            max_generation_requests=int(
                value["max_generation_requests"]
            ),
            nodes=tuple(
                SemanticSubwindowNode.from_dict(dict(item))
                for item in value["nodes"]  # type: ignore[union-attr]
            ),
            plan_hash=str(value["plan_hash"]),
        )
        if plan.recompute_hash() != plan.plan_hash:
            raise SemanticSubwindowError(
                "Semantic subwindow plan hash не соответствует содержимому"
            )
        plan._validate_structure()
        return plan

    def recompute_hash(self) -> str:
        return _fingerprint(self._hash_payload())

    def _hash_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "parent_attempt_id": self.parent_attempt_id,
            "logical_child_attempt_id": self.logical_child_attempt_id,
            "logical_window_id": self.logical_window_id,
            "parent_plan_hash": self.parent_plan_hash,
            "context_fingerprint": self.context_fingerprint,
            "policy_version": self.policy_version,
            "prompt_version": self.prompt_version,
            "contract_version": self.contract_version,
            "max_depth": self.max_depth,
            "min_primary_lines": self.min_primary_lines,
            "max_generation_requests": self.max_generation_requests,
            "nodes": [item.to_dict() for item in self.nodes],
        }

    def _validate_context(self, window: DecompositionWindow) -> None:
        if _fingerprint(window.to_dict()) != self.context_fingerprint:
            raise SemanticSubwindowError(
                "Semantic subwindow source context изменился"
            )

    def _validate_structure(self) -> None:
        if (
            self.schema_version != SemanticSubwindowPlanner.SCHEMA_VERSION
            or self.max_depth < 1
            or self.min_primary_lines < 1
            or self.max_generation_requests < 4
            or not self.nodes
        ):
            raise SemanticSubwindowError(
                "Некорректная semantic subwindow policy"
            )
        ids = tuple(item.node_id for item in self.nodes)
        if len(ids) != len(set(ids)):
            raise SemanticSubwindowError(
                "Semantic subwindow IDs должны быть уникальны"
            )
        root = self.root
        if root.depth != 0:
            raise SemanticSubwindowError(
                "Semantic subwindow root имеет неверную depth"
            )
        for node in self.nodes:
            if not node.primary_positions:
                raise SemanticSubwindowError(
                    "Semantic subwindow не содержит primary positions"
                )
            children = self.children(node.node_id)
            if children:
                if (
                    len(children) != 2
                    or node.depth >= self.max_depth
                    or any(
                        child.parent_node_id != node.node_id
                        or child.depth != node.depth + 1
                        for child in children
                    )
                    or tuple(
                        position
                        for child in children
                        for position in child.primary_positions
                    )
                    != node.primary_positions
                ):
                    raise SemanticSubwindowError(
                        "Semantic subwindow children не образуют partition"
                    )
            elif (
                node.depth < self.max_depth
                and len(node.primary_positions)
                >= self.min_primary_lines * 2
            ):
                raise SemanticSubwindowError(
                    "Делимый semantic subwindow не содержит children"
                )


class SemanticSubwindowPlanner:
    SCHEMA_VERSION = "semantic-subwindow-plan-1"

    def __init__(
        self,
        *,
        max_depth: int,
        min_primary_lines: int,
        max_generation_requests: int,
        policy_version: str,
        prompt_version: str,
        contract_version: str,
    ) -> None:
        if (
            max_depth < 1
            or min_primary_lines < 1
            or max_generation_requests < 4
        ):
            raise SemanticSubwindowError(
                "Некорректные semantic split limits"
            )
        self.max_depth = max_depth
        self.min_primary_lines = min_primary_lines
        self.max_generation_requests = max_generation_requests
        self.policy_version = policy_version
        self.prompt_version = prompt_version
        self.contract_version = contract_version

    def build(
        self,
        *,
        parent_attempt_id: str,
        logical_child_attempt_id: str,
        parent_plan: WindowPlan,
        window: DecompositionWindow,
    ) -> SemanticSubwindowPlan:
        nodes: list[SemanticSubwindowNode] = []

        def add(
            positions: tuple[SourcePosition, ...],
            *,
            parent_node_id: str | None,
            depth: int,
            path: str,
        ) -> None:
            node_id = (
                f"{window.window_id}:SEMANTIC_SUB:{path or 'ROOT'}"
            )
            nodes.append(
                SemanticSubwindowNode(
                    node_id=node_id,
                    parent_node_id=parent_node_id,
                    depth=depth,
                    primary_positions=positions,
                )
            )
            if (
                depth >= self.max_depth
                or len(positions) < self.min_primary_lines * 2
            ):
                return
            midpoint = len(positions) // 2
            add(
                positions[:midpoint],
                parent_node_id=node_id,
                depth=depth + 1,
                path=f"{path}L",
            )
            add(
                positions[midpoint:],
                parent_node_id=node_id,
                depth=depth + 1,
                path=f"{path}R",
            )

        add(
            window.primary_positions,
            parent_node_id=None,
            depth=0,
            path="",
        )
        provisional = SemanticSubwindowPlan(
            schema_version=self.SCHEMA_VERSION,
            parent_attempt_id=parent_attempt_id,
            logical_child_attempt_id=logical_child_attempt_id,
            logical_window_id=window.window_id,
            parent_plan_hash=parent_plan.plan_hash,
            context_fingerprint=_fingerprint(window.to_dict()),
            policy_version=self.policy_version,
            prompt_version=self.prompt_version,
            contract_version=self.contract_version,
            max_depth=self.max_depth,
            min_primary_lines=self.min_primary_lines,
            max_generation_requests=self.max_generation_requests,
            nodes=tuple(nodes),
            plan_hash="pending",
        )
        plan = replace(
            provisional,
            plan_hash=provisional.recompute_hash(),
        )
        plan.validate_binding(
            parent_attempt_id=parent_attempt_id,
            logical_child_attempt_id=logical_child_attempt_id,
            parent_plan=parent_plan,
            window=window,
        )
        return plan


@dataclass(frozen=True, slots=True)
class SemanticSubwindowNodeState:
    node_id: str
    status: SemanticSubwindowStatus
    generation_attempt_id: str | None = None
    split_reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "node_id": self.node_id,
            "status": self.status.value,
            "generation_attempt_id": self.generation_attempt_id,
            "split_reason": self.split_reason,
        }

    @classmethod
    def from_dict(
        cls,
        value: dict[str, object],
    ) -> SemanticSubwindowNodeState:
        return cls(
            node_id=str(value["node_id"]),
            status=SemanticSubwindowStatus(str(value["status"])),
            generation_attempt_id=(
                str(value["generation_attempt_id"])
                if value.get("generation_attempt_id") is not None
                else None
            ),
            split_reason=(
                str(value["split_reason"])
                if value.get("split_reason") is not None
                else None
            ),
        )


@dataclass(frozen=True, slots=True)
class SemanticSubwindowState:
    parent_attempt_id: str
    logical_child_attempt_id: str
    logical_window_id: str
    plan_hash: str
    status: SemanticCoordinatorStatus
    reserved_generation_requests: int
    nodes: tuple[SemanticSubwindowNodeState, ...]
    stop_reason: str | None = None

    @classmethod
    def started(
        cls,
        plan: SemanticSubwindowPlan,
        *,
        root_generation_attempt_id: str,
        consumed_generation_requests: int,
    ) -> SemanticSubwindowState:
        if (
            consumed_generation_requests != 2
            or not root_generation_attempt_id.strip()
        ):
            raise SemanticSubwindowError(
                "Root semantic split должен следовать после двух генераций"
            )
        return cls(
            parent_attempt_id=plan.parent_attempt_id,
            logical_child_attempt_id=plan.logical_child_attempt_id,
            logical_window_id=plan.logical_window_id,
            plan_hash=plan.plan_hash,
            status=SemanticCoordinatorStatus.RUNNING,
            reserved_generation_requests=consumed_generation_requests,
            nodes=tuple(
                SemanticSubwindowNodeState(
                    node_id=node.node_id,
                    status=(
                        SemanticSubwindowStatus.SPLIT
                        if node.node_id == plan.root.node_id
                        else SemanticSubwindowStatus.PLANNED
                    ),
                    generation_attempt_id=(
                        root_generation_attempt_id
                        if node.node_id == plan.root.node_id
                        else None
                    ),
                    split_reason=(
                        "finish_reason=length после max_tokens=12288"
                        if node.node_id == plan.root.node_id
                        else None
                    ),
                )
                for node in plan.nodes
            ),
        )

    def node(self, node_id: str) -> SemanticSubwindowNodeState:
        try:
            return next(item for item in self.nodes if item.node_id == node_id)
        except StopIteration as error:
            raise SemanticSubwindowError(
                f"Неизвестное semantic subwindow state {node_id}"
            ) from error

    def start_node(
        self,
        node_id: str,
        generation_attempt_id: str,
        *,
        plan: SemanticSubwindowPlan,
    ) -> SemanticSubwindowState:
        self._require_running()
        if not generation_attempt_id.strip():
            raise SemanticSubwindowError(
                "Generation attempt ID не задан"
            )
        node = self.node(node_id)
        if node.status is not SemanticSubwindowStatus.PLANNED:
            raise SemanticSubwindowError(
                f"Semantic subwindow {node_id} уже запускалось"
            )
        if (
            self.reserved_generation_requests + 2
            > plan.max_generation_requests
        ):
            raise SemanticSubwindowError(
                "Semantic generation request limit исчерпан"
            )
        return self._replace_node(
            replace(
                node,
                status=SemanticSubwindowStatus.RUNNING,
                generation_attempt_id=generation_attempt_id,
            ),
            reserved_generation_requests=(
                self.reserved_generation_requests + 2
            ),
        )

    def complete_node(
        self,
        node_id: str,
        generation_attempt_id: str,
    ) -> SemanticSubwindowState:
        node = self._running_node(node_id, generation_attempt_id)
        return self._replace_node(
            replace(node, status=SemanticSubwindowStatus.COMPLETED)
        )

    def recover_node(
        self,
        node_id: str,
        *,
        previous_generation_attempt_id: str,
        recovery_generation_attempt_id: str,
        plan: SemanticSubwindowPlan,
    ) -> SemanticSubwindowState:
        node = self._running_node(
            node_id,
            previous_generation_attempt_id,
        )
        if (
            not recovery_generation_attempt_id.strip()
            or recovery_generation_attempt_id
            == previous_generation_attempt_id
        ):
            raise SemanticSubwindowError(
                "Recovery generation attempt ID должен быть новым"
            )
        if (
            self.reserved_generation_requests + 2
            > plan.max_generation_requests
        ):
            raise SemanticSubwindowError(
                "Semantic generation request limit исчерпан"
            )
        return self._replace_node(
            replace(
                node,
                generation_attempt_id=recovery_generation_attempt_id,
            ),
            reserved_generation_requests=(
                self.reserved_generation_requests + 2
            ),
        )

    def split_node(
        self,
        node_id: str,
        generation_attempt_id: str,
        *,
        plan: SemanticSubwindowPlan,
        reason: str = "finish_reason=length после max_tokens=12288",
    ) -> SemanticSubwindowState:
        node = self._running_node(node_id, generation_attempt_id)
        children = plan.children(node_id)
        if not children:
            raise SemanticSubwindowError(
                f"Semantic subwindow {node_id} больше нельзя делить"
            )
        return self._replace_node(
            replace(
                node,
                status=SemanticSubwindowStatus.SPLIT,
                split_reason=reason,
            )
        )

    def cancel(self, reason: str) -> SemanticSubwindowState:
        self._require_running()
        if not reason.strip():
            raise SemanticSubwindowError("Причина отмены не задана")
        return replace(
            self,
            status=SemanticCoordinatorStatus.CANCELLED,
            stop_reason=reason,
            nodes=tuple(
                replace(item, status=SemanticSubwindowStatus.CANCELLED)
                if item.status
                in {
                    SemanticSubwindowStatus.PLANNED,
                    SemanticSubwindowStatus.RUNNING,
                }
                else item
                for item in self.nodes
            ),
        )

    def complete(
        self,
        *,
        plan: SemanticSubwindowPlan,
    ) -> SemanticSubwindowState:
        self._require_running()
        unfinished = tuple(
            node.node_id
            for node in plan.nodes
            if self._is_active(node.node_id, plan=plan)
            and self.node(node.node_id).status
            is not SemanticSubwindowStatus.COMPLETED
        )
        if unfinished:
            raise SemanticSubwindowError(
                f"Semantic subwindows не завершены: {unfinished}"
            )
        return replace(
            self,
            status=SemanticCoordinatorStatus.COMPLETED,
        )

    def fail(self, reason: str) -> SemanticSubwindowState:
        self._require_running()
        if not reason.strip():
            raise SemanticSubwindowError("Причина ошибки не задана")
        return replace(
            self,
            status=SemanticCoordinatorStatus.FAILED,
            stop_reason=reason,
            nodes=tuple(
                replace(item, status=SemanticSubwindowStatus.FAILED)
                if item.status
                in {
                    SemanticSubwindowStatus.PLANNED,
                    SemanticSubwindowStatus.RUNNING,
                }
                else item
                for item in self.nodes
            ),
        )

    def active_nodes(
        self,
        *,
        plan: SemanticSubwindowPlan,
    ) -> tuple[SemanticSubwindowNodeState, ...]:
        return tuple(
            self.node(node.node_id)
            for node in plan.nodes
            if self._is_active(node.node_id, plan=plan)
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "parent_attempt_id": self.parent_attempt_id,
            "logical_child_attempt_id": self.logical_child_attempt_id,
            "logical_window_id": self.logical_window_id,
            "plan_hash": self.plan_hash,
            "status": self.status.value,
            "reserved_generation_requests": (
                self.reserved_generation_requests
            ),
            "stop_reason": self.stop_reason,
            "nodes": [item.to_dict() for item in self.nodes],
        }

    @classmethod
    def from_dict(
        cls,
        value: dict[str, object],
    ) -> SemanticSubwindowState:
        return cls(
            parent_attempt_id=str(value["parent_attempt_id"]),
            logical_child_attempt_id=str(
                value["logical_child_attempt_id"]
            ),
            logical_window_id=str(value["logical_window_id"]),
            plan_hash=str(value["plan_hash"]),
            status=SemanticCoordinatorStatus(str(value["status"])),
            reserved_generation_requests=int(
                value["reserved_generation_requests"]
            ),
            nodes=tuple(
                SemanticSubwindowNodeState.from_dict(dict(item))
                for item in value["nodes"]  # type: ignore[union-attr]
            ),
            stop_reason=(
                str(value["stop_reason"])
                if value.get("stop_reason") is not None
                else None
            ),
        )

    def _running_node(
        self,
        node_id: str,
        generation_attempt_id: str,
    ) -> SemanticSubwindowNodeState:
        self._require_running()
        node = self.node(node_id)
        if (
            node.status is not SemanticSubwindowStatus.RUNNING
            or node.generation_attempt_id != generation_attempt_id
        ):
            raise SemanticSubwindowError(
                f"Semantic subwindow {node_id} не выполняется"
            )
        return node

    def _replace_node(
        self,
        updated: SemanticSubwindowNodeState,
        *,
        reserved_generation_requests: int | None = None,
    ) -> SemanticSubwindowState:
        return replace(
            self,
            reserved_generation_requests=(
                self.reserved_generation_requests
                if reserved_generation_requests is None
                else reserved_generation_requests
            ),
            nodes=tuple(
                updated if item.node_id == updated.node_id else item
                for item in self.nodes
            ),
        )

    def _is_active(
        self,
        node_id: str,
        *,
        plan: SemanticSubwindowPlan,
    ) -> bool:
        node = plan.node(node_id)
        parent_id = node.parent_node_id
        while parent_id is not None:
            if (
                self.node(parent_id).status
                is not SemanticSubwindowStatus.SPLIT
            ):
                return False
            parent_id = plan.node(parent_id).parent_node_id
        return (
            self.node(node_id).status
            is not SemanticSubwindowStatus.SPLIT
        )

    def _require_running(self) -> None:
        if self.status is not SemanticCoordinatorStatus.RUNNING:
            raise SemanticSubwindowError(
                f"Semantic coordinator имеет статус {self.status.value}"
            )


class SemanticSubwindowStore:
    RECORD_KIND = "decomposition_semantic_subwindow_attempt"
    STORAGE_SCHEMA_VERSION = 2

    def __init__(self, uow_factory: Callable[[], UnitOfWork]) -> None:
        self.uow_factory = uow_factory

    def save(
        self,
        state: SemanticSubwindowState,
        plan: SemanticSubwindowPlan,
        *,
        expected_state: SemanticSubwindowState | None = None,
        uow: UnitOfWork | None = None,
    ) -> None:
        self._validate(state, plan)
        record_id = self._record_id(
            state.parent_attempt_id,
            state.logical_window_id,
        )
        payload: dict[str, object] = {
            "storage_schema_version": self.STORAGE_SCHEMA_VERSION,
            "state": state.to_dict(),
            "plan": plan.to_dict(),
        }
        payload["fingerprint"] = _fingerprint(payload)

        def persist(target: UnitOfWork) -> None:
            existing = target.records.get(self.RECORD_KIND, record_id)
            if expected_state is not None:
                if (
                    existing is None
                    or existing.payload.get("state")
                    != expected_state.to_dict()
                    or existing.payload.get("plan") != plan.to_dict()
                    or existing.payload.get("fingerprint")
                    != _fingerprint(
                        {
                            "storage_schema_version": (
                                self.STORAGE_SCHEMA_VERSION
                            ),
                            "state": expected_state.to_dict(),
                            "plan": plan.to_dict(),
                        }
                    )
                ):
                    raise SemanticSubwindowError(
                        "Semantic subwindow coordinator изменился"
                    )
            elif existing is not None:
                raise SemanticSubwindowError(
                    "Semantic subwindow coordinator уже существует"
                )
            target.records.save(
                StoredRecord(self.RECORD_KIND, record_id, payload)
            )
        if uow is not None:
            persist(uow)
            return
        with self.uow_factory() as owned_uow:
            persist(owned_uow)

    def load_optional(
        self,
        *,
        parent_attempt_id: str,
        logical_window_id: str,
        parent_plan: WindowPlan,
        window: DecompositionWindow,
    ) -> tuple[SemanticSubwindowState, SemanticSubwindowPlan] | None:
        with self.uow_factory() as uow:
            record = uow.records.get(
                self.RECORD_KIND,
                self._record_id(parent_attempt_id, logical_window_id),
            )
        if record is None:
            return None
        return self.load(
            parent_attempt_id=parent_attempt_id,
            logical_window_id=logical_window_id,
            parent_plan=parent_plan,
            window=window,
        )

    def load(
        self,
        *,
        parent_attempt_id: str,
        logical_window_id: str,
        parent_plan: WindowPlan,
        window: DecompositionWindow,
    ) -> tuple[SemanticSubwindowState, SemanticSubwindowPlan]:
        record_id = self._record_id(
            parent_attempt_id,
            logical_window_id,
        )
        with self.uow_factory() as uow:
            record = uow.records.get(self.RECORD_KIND, record_id)
        if record is None:
            raise SemanticSubwindowError(
                "Semantic subwindow coordinator не найден"
            )
        if (
            int(record.payload.get("storage_schema_version", -1))
            != self.STORAGE_SCHEMA_VERSION
        ):
            raise SemanticSubwindowError(
                "Неизвестная semantic subwindow storage schema"
            )
        if record.payload.get("fingerprint") != _fingerprint(
            {
                key: value
                for key, value in record.payload.items()
                if key != "fingerprint"
            }
        ):
            raise SemanticSubwindowError(
                "Semantic subwindow coordinator fingerprint повреждён"
            )
        plan = SemanticSubwindowPlan.from_dict(
            dict(record.payload["plan"])
        )
        state = SemanticSubwindowState.from_dict(
            dict(record.payload["state"])
        )
        plan.validate_binding(
            parent_attempt_id=parent_attempt_id,
            logical_child_attempt_id=state.logical_child_attempt_id,
            parent_plan=parent_plan,
            window=window,
        )
        self._validate(state, plan)
        return state, plan

    @staticmethod
    def _record_id(
        parent_attempt_id: str,
        logical_window_id: str,
    ) -> str:
        return f"{parent_attempt_id}:{logical_window_id}"

    @staticmethod
    def _validate(
        state: SemanticSubwindowState,
        plan: SemanticSubwindowPlan,
    ) -> None:
        if (
            state.parent_attempt_id != plan.parent_attempt_id
            or state.logical_child_attempt_id
            != plan.logical_child_attempt_id
            or state.logical_window_id != plan.logical_window_id
            or state.plan_hash != plan.plan_hash
            or tuple(item.node_id for item in state.nodes)
            != tuple(item.node_id for item in plan.nodes)
            or state.reserved_generation_requests
            > plan.max_generation_requests
            or state.reserved_generation_requests < 2
            or state.reserved_generation_requests % 2 != 0
        ):
            raise SemanticSubwindowError(
                "Semantic subwindow state не соответствует plan"
            )
        for item in state.nodes:
            children = plan.children(item.node_id)
            if (
                item.status is SemanticSubwindowStatus.PLANNED
                and (
                    item.generation_attempt_id is not None
                    or item.split_reason is not None
                )
            ):
                raise SemanticSubwindowError(
                    "Planned semantic subwindow содержит runtime state"
                )
            if (
                item.status is SemanticSubwindowStatus.SPLIT
                and (
                    not children
                    or item.generation_attempt_id is None
                    or item.split_reason is None
                )
            ):
                raise SemanticSubwindowError(
                    "Split semantic subwindow имеет неверное состояние"
                )
            if item.status in {
                SemanticSubwindowStatus.RUNNING,
                SemanticSubwindowStatus.COMPLETED,
            } and item.generation_attempt_id is None:
                raise SemanticSubwindowError(
                    "Semantic subwindow status не связан с generation attempt"
                )


def _position_dict(position: SourcePosition) -> dict[str, int]:
    return {
        "page": position.page_index,
        "line": position.line_number,
    }


def _position_from_dict(value: dict[str, object]) -> SourcePosition:
    return SourcePosition(int(value["page"]), int(value["line"]))


def _fingerprint(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
