from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict

from ...llm import AttemptDiscardedError, GenerationLengthError
from ...repositories import UnitOfWork
from ...state import StoredRecord
from .models import WindowedAttemptState
from .plan import DecompositionWindow, WindowPlan
from .policy import WindowingPolicy
from .semantic import (
    SEMANTIC_WINDOW_SCHEMA_VERSION,
    SemanticWindowArguments,
    SemanticWindowResult,
)
from .semantic_flow import SemanticWindowFlow
from .semantic_service import SemanticWindowService
from .semantic_split import (
    SemanticCoordinatorStatus,
    SemanticSubwindowError,
    SemanticSubwindowPlan,
    SemanticSubwindowPlanner,
    SemanticSubwindowState,
    SemanticSubwindowStatus,
    SemanticSubwindowStore,
)


class SemanticSubwindowResultStore:
    RECORD_KIND = "decomposition_semantic_subwindow_result"
    STORAGE_SCHEMA_VERSION = 2

    def __init__(
        self,
        *,
        uow_factory: Callable[[], UnitOfWork],
        service: SemanticWindowService,
    ) -> None:
        self.uow_factory = uow_factory
        self.service = service

    def save(
        self,
        *,
        parent: WindowedAttemptState,
        parent_plan: WindowPlan,
        split_plan: SemanticSubwindowPlan,
        node_id: str,
        generation_attempt_id: str,
        arguments: SemanticWindowArguments,
        raw_arguments: dict[str, object],
        validated: SemanticWindowResult,
        uow: UnitOfWork,
    ) -> None:
        if raw_arguments != asdict(arguments):
            raise SemanticSubwindowError(
                "Raw semantic subwindow arguments не соответствуют typed"
            )
        expected = self._validate(
            parent=parent,
            parent_plan=parent_plan,
            split_plan=split_plan,
            node_id=node_id,
            generation_attempt_id=generation_attempt_id,
            arguments=arguments,
        )
        if validated != expected:
            raise SemanticSubwindowError(
                "Validated semantic subwindow result не воспроизводится"
            )
        subwindow = self._subwindow(
            parent_plan=parent_plan,
            split_plan=split_plan,
            node_id=node_id,
        )
        owned_arguments = self.service.canonicalizer.owned_arguments(
            subwindow,
            arguments,
        )
        payload: dict[str, object] = {
            "storage_schema_version": self.STORAGE_SCHEMA_VERSION,
            "contract_version": SEMANTIC_WINDOW_SCHEMA_VERSION,
            "split_plan_hash": split_plan.plan_hash,
            "generation_attempt_id": generation_attempt_id,
            "raw_arguments": raw_arguments,
            "owned_arguments": asdict(owned_arguments),
            "context_only_behaviors": (
                len(arguments.behaviors)
                - len(owned_arguments.behaviors)
            ),
            "validated": validated.to_dict(),
        }
        payload = json.loads(json.dumps(payload, ensure_ascii=False))
        payload["fingerprint"] = _fingerprint(payload)
        record_id = self._record_id(
            parent.parent_attempt_id,
            split_plan.logical_window_id,
            node_id,
        )
        existing = uow.records.get(self.RECORD_KIND, record_id)
        if existing is not None:
            if existing.payload != payload:
                raise SemanticSubwindowError(
                    "Для semantic subwindow уже сохранён другой result"
                )
            return
        uow.records.save(StoredRecord(self.RECORD_KIND, record_id, payload))
        uow.events.append(
            parent.parent_attempt_id,
            "semantic subwindow result сохранён",
            {
                "logical_window_id": split_plan.logical_window_id,
                "node_id": node_id,
                "generation_attempt_id": generation_attempt_id,
                "fragments": len(validated.fragments),
            },
        )

    def load(
        self,
        *,
        parent: WindowedAttemptState,
        parent_plan: WindowPlan,
        split_plan: SemanticSubwindowPlan,
        node_id: str,
    ) -> tuple[SemanticWindowArguments, SemanticWindowResult] | None:
        record_id = self._record_id(
            parent.parent_attempt_id,
            split_plan.logical_window_id,
            node_id,
        )
        with self.uow_factory() as uow:
            record = uow.records.get(self.RECORD_KIND, record_id)
        if record is None:
            return None
        payload = record.payload
        if set(payload) != {
            "storage_schema_version",
            "contract_version",
            "split_plan_hash",
            "generation_attempt_id",
            "raw_arguments",
            "owned_arguments",
            "context_only_behaviors",
            "validated",
            "fingerprint",
        }:
            raise SemanticSubwindowError(
                "Semantic subwindow result содержит неизвестные поля"
            )
        expected_fingerprint = payload.get("fingerprint")
        actual_fingerprint = _fingerprint(
            {key: value for key, value in payload.items() if key != "fingerprint"}
        )
        if (
            int(payload.get("storage_schema_version", -1))
            != self.STORAGE_SCHEMA_VERSION
            or payload.get("contract_version")
            != SEMANTIC_WINDOW_SCHEMA_VERSION
            or payload.get("split_plan_hash") != split_plan.plan_hash
            or expected_fingerprint != actual_fingerprint
        ):
            raise SemanticSubwindowError(
                "Semantic subwindow result имеет stale или повреждённый binding"
            )
        raw = payload.get("raw_arguments")
        if (
            not isinstance(raw, dict)
            or set(raw) != {"behaviors"}
            or not isinstance(raw["behaviors"], list)
        ):
            raise SemanticSubwindowError(
                "Raw semantic subwindow result имеет неверную структуру"
            )
        arguments = SemanticWindowArguments(behaviors=raw["behaviors"])
        generation_attempt_id = str(payload["generation_attempt_id"])
        recomputed = self._validate(
            parent=parent,
            parent_plan=parent_plan,
            split_plan=split_plan,
            node_id=node_id,
            generation_attempt_id=generation_attempt_id,
            arguments=arguments,
        )
        stored = SemanticWindowResult.from_dict(dict(payload["validated"]))
        if recomputed != stored:
            raise SemanticSubwindowError(
                "Semantic subwindow validated result изменился"
            )
        subwindow = self._subwindow(
            parent_plan=parent_plan,
            split_plan=split_plan,
            node_id=node_id,
        )
        recomputed_owned = self.service.canonicalizer.owned_arguments(
            subwindow,
            arguments,
        )
        owned = payload["owned_arguments"]
        if (
            not isinstance(owned, dict)
            or set(owned) != {"behaviors"}
            or owned != asdict(recomputed_owned)
            or payload["context_only_behaviors"]
            != len(arguments.behaviors)
            - len(recomputed_owned.behaviors)
        ):
            raise SemanticSubwindowError(
                "Owned semantic subwindow result не воспроизводится"
            )
        return recomputed_owned, stored

    def _validate(
        self,
        *,
        parent: WindowedAttemptState,
        parent_plan: WindowPlan,
        split_plan: SemanticSubwindowPlan,
        node_id: str,
        generation_attempt_id: str,
        arguments: SemanticWindowArguments,
    ) -> SemanticWindowResult:
        subwindow = self._subwindow(
            parent_plan=parent_plan,
            split_plan=split_plan,
            node_id=node_id,
        )
        logical_window = next(
            item
            for item in parent_plan.windows
            if item.window_id == split_plan.logical_window_id
        )
        split_plan.validate_binding(
            parent_attempt_id=parent.parent_attempt_id,
            logical_child_attempt_id=split_plan.logical_child_attempt_id,
            parent_plan=parent_plan,
            window=logical_window,
        )
        return self.service.canonicalizer.canonicalize_subwindow(
            parent=parent,
            plan=parent_plan,
            logical_window_id=split_plan.logical_window_id,
            logical_child_attempt_id=split_plan.logical_child_attempt_id,
            subwindow=subwindow,
            generation_attempt_id=generation_attempt_id,
            arguments=arguments,
        )

    @staticmethod
    def _subwindow(
        *,
        parent_plan: WindowPlan,
        split_plan: SemanticSubwindowPlan,
        node_id: str,
    ) -> DecompositionWindow:
        try:
            logical_window = next(
                item
                for item in parent_plan.windows
                if item.window_id == split_plan.logical_window_id
            )
        except StopIteration as error:
            raise SemanticSubwindowError(
                "Semantic subwindow result относится к неизвестному окну"
            ) from error
        return split_plan.masked_window(logical_window, node_id)

    @staticmethod
    def _record_id(
        parent_attempt_id: str,
        logical_window_id: str,
        node_id: str,
    ) -> str:
        return f"{parent_attempt_id}:{logical_window_id}:{node_id}"


class SemanticSubwindowCoordinator:
    def __init__(
        self,
        *,
        policy: WindowingPolicy,
        semantic_flow: SemanticWindowFlow,
        service: SemanticWindowService,
        uow_factory: Callable[[], UnitOfWork],
        next_id: Callable[[str], str],
        set_active_attempt: Callable[[str | None], None],
        raise_if_cancelled: Callable[[], None],
    ) -> None:
        self.policy = policy
        self.semantic_flow = semantic_flow
        self.service = service
        self.next_id = next_id
        self.set_active_attempt = set_active_attempt
        self.raise_if_cancelled = raise_if_cancelled
        self.state_store = SemanticSubwindowStore(uow_factory)
        self.result_store = SemanticSubwindowResultStore(
            uow_factory=uow_factory,
            service=service,
        )

    async def run(
        self,
        *,
        parent: WindowedAttemptState,
        parent_plan: WindowPlan,
        logical_window: DecompositionWindow,
        logical_child_attempt_id: str,
        root_generation_attempt_id: str | None,
        session_id: str,
    ) -> SemanticWindowResult:
        restored = self.state_store.load_optional(
            parent_attempt_id=parent.parent_attempt_id,
            logical_window_id=logical_window.window_id,
            parent_plan=parent_plan,
            window=logical_window,
        )
        if restored is None:
            if root_generation_attempt_id is None:
                raise SemanticSubwindowError(
                    "Root generation attempt ID отсутствует"
                )
            split_plan = SemanticSubwindowPlanner(
                max_depth=self.policy.semantic_split_max_depth,
                min_primary_lines=self.policy.semantic_split_min_primary_lines,
                max_generation_requests=(
                    self.policy.semantic_split_max_generation_requests
                ),
                policy_version=self.policy.fingerprint,
                prompt_version=self.policy.semantic_prompt_version,
                contract_version=self.policy.semantic_schema_version,
            ).build(
                parent_attempt_id=parent.parent_attempt_id,
                logical_child_attempt_id=logical_child_attempt_id,
                parent_plan=parent_plan,
                window=logical_window,
            )
            state = SemanticSubwindowState.started(
                split_plan,
                root_generation_attempt_id=root_generation_attempt_id,
                consumed_generation_requests=2,
            )
            self.state_store.save(state, split_plan)
        else:
            state, split_plan = restored
            if (
                split_plan.logical_child_attempt_id
                != logical_child_attempt_id
            ):
                raise SemanticSubwindowError(
                    "Semantic split относится к другому logical child"
                )

        if state.status is SemanticCoordinatorStatus.COMPLETED:
            existing = self.service.load(
                logical_window.window_id,
                parent=parent,
                plan=parent_plan,
            )
            if existing is None:
                raise SemanticSubwindowError(
                    "Завершённый semantic coordinator не содержит result"
                )
            return existing
        if state.status is not SemanticCoordinatorStatus.RUNNING:
            raise SemanticSubwindowError(
                f"Semantic coordinator имеет статус {state.status.value}"
            )

        while True:
            self.raise_if_cancelled()
            active = state.active_nodes(plan=split_plan)
            pending = tuple(
                item
                for item in active
                if item.status
                in {
                    SemanticSubwindowStatus.PLANNED,
                    SemanticSubwindowStatus.RUNNING,
                }
            )
            if not pending:
                break
            node_state = pending[0]
            saved = self.result_store.load(
                parent=parent,
                parent_plan=parent_plan,
                split_plan=split_plan,
                node_id=node_state.node_id,
            )
            if saved is not None:
                if node_state.status is not SemanticSubwindowStatus.RUNNING:
                    raise SemanticSubwindowError(
                        "Validated subresult не соответствует node state"
                    )
                assert node_state.generation_attempt_id is not None
                updated = state.complete_node(
                    node_state.node_id,
                    node_state.generation_attempt_id,
                )
                self.state_store.save(
                    updated,
                    split_plan,
                    expected_state=state,
                )
                state = updated
                continue

            generation_attempt_id = self.next_id(
                "ATTEMPT_WINDOW_GENERATION"
            )
            try:
                if node_state.status is SemanticSubwindowStatus.PLANNED:
                    updated = state.start_node(
                        node_state.node_id,
                        generation_attempt_id,
                        plan=split_plan,
                    )
                else:
                    assert node_state.generation_attempt_id is not None
                    updated = state.recover_node(
                        node_state.node_id,
                        previous_generation_attempt_id=(
                            node_state.generation_attempt_id
                        ),
                        recovery_generation_attempt_id=(
                            generation_attempt_id
                        ),
                        plan=split_plan,
                    )
            except SemanticSubwindowError as error:
                failed = state.fail(str(error))
                self.state_store.save(
                    failed,
                    split_plan,
                    expected_state=state,
                )
                raise
            self.state_store.save(
                updated,
                split_plan,
                expected_state=state,
            )
            state = updated
            masked = split_plan.masked_window(
                logical_window,
                node_state.node_id,
            )
            completed_state: SemanticSubwindowState | None = None

            def accept(
                arguments: SemanticWindowArguments,
                raw_arguments: dict[str, object],
                validated: SemanticWindowResult,
                uow: UnitOfWork,
            ) -> SemanticWindowResult:
                nonlocal completed_state
                self.result_store.save(
                    parent=parent,
                    parent_plan=parent_plan,
                    split_plan=split_plan,
                    node_id=node_state.node_id,
                    generation_attempt_id=generation_attempt_id,
                    arguments=arguments,
                    raw_arguments=raw_arguments,
                    validated=validated,
                    uow=uow,
                )
                completed_state = state.complete_node(
                    node_state.node_id,
                    generation_attempt_id,
                )
                self.state_store.save(
                    completed_state,
                    split_plan,
                    expected_state=state,
                    uow=uow,
                )
                return validated

            self.set_active_attempt(generation_attempt_id)
            try:
                await self.semantic_flow.run_subwindow(
                    parent=parent,
                    plan=parent_plan,
                    logical_window_id=logical_window.window_id,
                    logical_child_attempt_id=logical_child_attempt_id,
                    source_window=masked,
                    generation_attempt_id=generation_attempt_id,
                    session_id=session_id,
                    accept=accept,
                )
            except GenerationLengthError as error:
                try:
                    updated = state.split_node(
                        node_state.node_id,
                        generation_attempt_id,
                        plan=split_plan,
                    )
                except SemanticSubwindowError:
                    failed = state.fail(str(error))
                    self.state_store.save(
                        failed,
                        split_plan,
                        expected_state=state,
                    )
                    raise SemanticSubwindowError(
                        "Semantic generation исчерпала bounded split limits"
                    ) from error
                self.state_store.save(
                    updated,
                    split_plan,
                    expected_state=state,
                )
                state = updated
            except (asyncio.CancelledError, AttemptDiscardedError):
                raise
            except Exception as error:
                failed = state.fail(str(error))
                self.state_store.save(
                    failed,
                    split_plan,
                    expected_state=state,
                )
                raise
            finally:
                self.set_active_attempt(None)
            if completed_state is not None:
                state = completed_state

        combined: list[dict[str, object]] = []
        for node_state in state.active_nodes(plan=split_plan):
            if node_state.status is not SemanticSubwindowStatus.COMPLETED:
                raise SemanticSubwindowError(
                    "Semantic split завершён не полностью"
                )
            saved = self.result_store.load(
                parent=parent,
                parent_plan=parent_plan,
                split_plan=split_plan,
                node_id=node_state.node_id,
            )
            if saved is None:
                raise SemanticSubwindowError(
                    "Validated semantic subresult отсутствует"
                )
            arguments, _validated = saved
            combined.extend(arguments.behaviors)

        raw_arguments = {"behaviors": combined}
        arguments = SemanticWindowArguments(behaviors=combined)
        completed = state.complete(plan=split_plan)
        with self.service.uow_factory() as uow:
            result = self.service.accept(
                parent=parent,
                plan=parent_plan,
                window_id=logical_window.window_id,
                child_attempt_id=logical_child_attempt_id,
                arguments=arguments,
                raw_arguments=raw_arguments,
                uow=uow,
            )
            self.state_store.save(
                completed,
                split_plan,
                expected_state=state,
                uow=uow,
            )
        return result


def _fingerprint(payload: dict[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
