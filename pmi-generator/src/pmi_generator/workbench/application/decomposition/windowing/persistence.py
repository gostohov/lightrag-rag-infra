from __future__ import annotations

from collections.abc import Callable

from ...repositories import UnitOfWork
from ...state import StoredRecord
from ...source import SavedSelection
from .models import (
    WindowChildState,
    WindowChildStatus,
    WindowedAttemptState,
    WindowedAttemptStatus,
)
from .plan import WindowPlan, WindowPlanError


class WindowedDecompositionStore:
    RECORD_KIND = "decomposition_windowed_attempt"
    STORAGE_SCHEMA_VERSION = 1

    def __init__(self, uow_factory: Callable[[], UnitOfWork]) -> None:
        self.uow_factory = uow_factory

    def save(
        self,
        state: WindowedAttemptState,
        plan: WindowPlan,
        *,
        uow: UnitOfWork | None = None,
        expected_state: WindowedAttemptState | None = None,
    ) -> None:
        self._validate(state, plan)
        if expected_state is not None:
            self._validate(expected_state, plan)
        payload = {
            "storage_schema_version": self.STORAGE_SCHEMA_VERSION,
            "parent": _state_to_dict(state),
            "plan": plan.to_dict(),
        }
        record = StoredRecord(
            self.RECORD_KIND,
            state.parent_attempt_id,
            payload,
        )
        if uow is not None:
            self._assert_expected(
                uow,
                state.parent_attempt_id,
                plan,
                expected_state,
            )
            uow.records.save(record)
            return
        with self.uow_factory() as owned_uow:
            self._assert_expected(
                owned_uow,
                state.parent_attempt_id,
                plan,
                expected_state,
            )
            owned_uow.records.save(record)

    @staticmethod
    def _validate(
        state: WindowedAttemptState,
        plan: WindowPlan,
    ) -> None:
        if state.parent_attempt_id.strip() == "":
            raise WindowPlanError("Parent attempt ID не задан")
        if state.window_plan_hash != plan.plan_hash:
            raise WindowPlanError("Parent state ссылается на другой plan hash")
        if state.selection_id != plan.selection_id:
            raise WindowPlanError("Parent state относится к другому selection")
        if plan.recompute_hash() != plan.plan_hash:
            raise WindowPlanError("Window plan hash не соответствует содержимому")
        if state.document_version != plan.document_version:
            raise WindowPlanError("Parent state содержит другой document_version")
        if state.policy_version != plan.policy_version:
            raise WindowPlanError("Parent state содержит другую policy version")
        if tuple(child.window_id for child in state.children) != tuple(
            window.window_id for window in plan.windows
        ):
            raise WindowPlanError("Parent children не совпадают с window plan")

    @classmethod
    def _assert_expected(
        cls,
        uow: UnitOfWork,
        parent_attempt_id: str,
        plan: WindowPlan,
        expected_state: WindowedAttemptState | None,
    ) -> None:
        if expected_state is None:
            return
        current = uow.records.get(cls.RECORD_KIND, parent_attempt_id)
        if current is None:
            return
        if (
            int(current.payload.get("storage_schema_version", -1))
            != cls.STORAGE_SCHEMA_VERSION
            or current.payload.get("parent") != _state_to_dict(expected_state)
            or current.payload.get("plan") != plan.to_dict()
        ):
            raise WindowPlanError(
                "Windowed attempt изменился после начала assembly"
            )

    def load(
        self,
        parent_attempt_id: str,
        selection: SavedSelection,
    ) -> tuple[WindowedAttemptState, WindowPlan]:
        with self.uow_factory() as uow:
            record = uow.records.get(self.RECORD_KIND, parent_attempt_id)
        if record is None:
            raise WindowPlanError(
                f"Windowed attempt {parent_attempt_id} не найден"
            )
        version = int(record.payload.get("storage_schema_version", -1))
        if version != self.STORAGE_SCHEMA_VERSION:
            raise WindowPlanError(
                f"Неизвестная windowed storage schema {version}"
            )
        state = _state_from_dict(dict(record.payload["parent"]))
        plan = WindowPlan.from_dict(dict(record.payload["plan"]))
        plan.validate(selection)
        if state.parent_attempt_id != parent_attempt_id:
            raise WindowPlanError("Parent attempt ID не совпадает с record ID")
        if state.window_plan_hash != plan.plan_hash:
            raise WindowPlanError("Parent state содержит другой plan hash")
        if state.document_version != selection.document_version:
            raise WindowPlanError("Parent state document_version устарел")
        if tuple(child.window_id for child in state.children) != tuple(
            window.window_id for window in plan.windows
        ):
            raise WindowPlanError("Parent children не совпадают с window plan")
        return state, plan

    def load_optional(
        self,
        parent_attempt_id: str,
        selection: SavedSelection,
    ) -> tuple[WindowedAttemptState, WindowPlan] | None:
        with self.uow_factory() as uow:
            record = uow.records.get(self.RECORD_KIND, parent_attempt_id)
        if record is None:
            return None
        return self.load(parent_attempt_id, selection)


def _state_to_dict(state: WindowedAttemptState) -> dict[str, object]:
    return {
        "parent_attempt_id": state.parent_attempt_id,
        "selection_id": state.selection_id,
        "document_version": state.document_version,
        "expected_workflow_revision": state.expected_workflow_revision,
        "policy_version": state.policy_version,
        "prompt_version": state.prompt_version,
        "schema_version": state.schema_version,
        "window_plan_hash": state.window_plan_hash,
        "status": state.status.value,
        "stop_reason": state.stop_reason,
        "children": [
            {
                "window_id": child.window_id,
                "status": child.status.value,
                "attempt_id": child.attempt_id,
            }
            for child in state.children
        ],
    }


def _state_from_dict(value: dict[str, object]) -> WindowedAttemptState:
    return WindowedAttemptState(
        parent_attempt_id=str(value["parent_attempt_id"]),
        selection_id=str(value["selection_id"]),
        document_version=str(value["document_version"]),
        expected_workflow_revision=str(value["expected_workflow_revision"]),
        policy_version=str(value["policy_version"]),
        prompt_version=str(value["prompt_version"]),
        schema_version=str(value["schema_version"]),
        window_plan_hash=str(value["window_plan_hash"]),
        children=tuple(
            WindowChildState(
                window_id=str(item["window_id"]),
                status=WindowChildStatus(str(item["status"])),
                attempt_id=(
                    str(item["attempt_id"])
                    if item.get("attempt_id") is not None
                    else None
                ),
            )
            for item in value["children"]  # type: ignore[union-attr]
        ),
        status=WindowedAttemptStatus(str(value["status"])),
        stop_reason=(
            str(value["stop_reason"])
            if value.get("stop_reason") is not None
            else None
        ),
    )
