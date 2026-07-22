from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum

from ..budget import DecompositionBudget


WINDOW_CANDIDATE_SCHEMA_VERSION = "window-candidates-5"


class DecompositionRoute(StrEnum):
    SINGLE_CALL = "single_call"
    WINDOWED = "windowed"
    HARD_LIMIT = "hard_limit"


@dataclass(frozen=True, slots=True)
class WindowingDecision:
    route: DecompositionRoute
    budget: DecompositionBudget
    hard_max_lines: int
    hard_max_estimated_tokens: int


class WindowedAttemptStatus(StrEnum):
    PLANNED = "planned"
    RUNNING = "running"
    RECONCILING = "reconciling"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class WindowChildStatus(StrEnum):
    PLANNED = "planned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class WindowedAttemptError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class WindowChildState:
    window_id: str
    status: WindowChildStatus = WindowChildStatus.PLANNED
    attempt_id: str | None = None


@dataclass(frozen=True, slots=True)
class WindowedAttemptState:
    parent_attempt_id: str
    selection_id: str
    document_version: str
    expected_workflow_revision: str
    policy_version: str
    prompt_version: str
    schema_version: str
    window_plan_hash: str
    children: tuple[WindowChildState, ...]
    status: WindowedAttemptStatus = WindowedAttemptStatus.PLANNED
    stop_reason: str | None = None

    @classmethod
    def planned(
        cls,
        *,
        parent_attempt_id: str,
        selection_id: str,
        document_version: str,
        expected_workflow_revision: str,
        policy_version: str,
        prompt_version: str,
        schema_version: str,
        window_plan_hash: str,
        window_ids: tuple[str, ...],
    ) -> WindowedAttemptState:
        required = {
            "parent_attempt_id": parent_attempt_id,
            "selection_id": selection_id,
            "document_version": document_version,
            "expected_workflow_revision": expected_workflow_revision,
            "policy_version": policy_version,
            "prompt_version": prompt_version,
            "schema_version": schema_version,
            "window_plan_hash": window_plan_hash,
        }
        missing = [name for name, value in required.items() if not value.strip()]
        if missing:
            raise WindowedAttemptError(
                f"Parent attempt не содержит обязательные поля: {missing}"
            )
        if not window_ids or len(window_ids) != len(set(window_ids)):
            raise WindowedAttemptError("Window IDs должны быть непустыми и уникальными")
        return cls(
            parent_attempt_id=parent_attempt_id,
            selection_id=selection_id,
            document_version=document_version,
            expected_workflow_revision=expected_workflow_revision,
            policy_version=policy_version,
            prompt_version=prompt_version,
            schema_version=schema_version,
            window_plan_hash=window_plan_hash,
            children=tuple(WindowChildState(window_id) for window_id in window_ids),
        )

    def start(self) -> WindowedAttemptState:
        self._reject_terminal()
        if self.status is not WindowedAttemptStatus.PLANNED:
            raise WindowedAttemptError("Parent attempt уже запущен")
        return replace(self, status=WindowedAttemptStatus.RUNNING)

    def start_child(
        self,
        window_id: str,
        attempt_id: str,
    ) -> WindowedAttemptState:
        self._require_status(WindowedAttemptStatus.RUNNING)
        if not attempt_id.strip():
            raise WindowedAttemptError("Child attempt ID не задан")
        child = self._child(window_id)
        if child.status is not WindowChildStatus.PLANNED:
            raise WindowedAttemptError(f"Окно {window_id} уже запускалось")
        return self._replace_child(
            replace(
                child,
                status=WindowChildStatus.RUNNING,
                attempt_id=attempt_id,
            )
        )

    def complete_child(
        self,
        window_id: str,
        attempt_id: str,
    ) -> WindowedAttemptState:
        self._require_status(WindowedAttemptStatus.RUNNING)
        child = self._child(window_id)
        if child.attempt_id != attempt_id:
            raise WindowedAttemptError(
                f"Child attempt {attempt_id} не совпадает с активным для {window_id}"
            )
        if child.status is not WindowChildStatus.RUNNING:
            raise WindowedAttemptError(f"Окно {window_id} не выполняется")
        return self._replace_child(
            replace(child, status=WindowChildStatus.COMPLETED)
        )

    def recover_child(
        self,
        window_id: str,
        *,
        previous_attempt_id: str,
        recovery_attempt_id: str,
    ) -> WindowedAttemptState:
        self._require_status(WindowedAttemptStatus.RUNNING)
        child = self._child(window_id)
        if child.status is not WindowChildStatus.RUNNING:
            raise WindowedAttemptError(
                f"Техническое восстановление недоступно для окна {window_id}"
            )
        if child.attempt_id != previous_attempt_id:
            raise WindowedAttemptError(
                f"Child attempt {previous_attempt_id} не совпадает "
                f"с сохранённым для {window_id}"
            )
        if (
            not recovery_attempt_id.strip()
            or recovery_attempt_id == previous_attempt_id
        ):
            raise WindowedAttemptError(
                "Recovery attempt ID должен быть новым и непустым"
            )
        return self._replace_child(
            replace(child, attempt_id=recovery_attempt_id)
        )

    def fail_child(
        self,
        window_id: str,
        attempt_id: str,
        reason: str,
    ) -> WindowedAttemptState:
        self._require_status(WindowedAttemptStatus.RUNNING)
        child = self._child(window_id)
        if child.attempt_id != attempt_id:
            raise WindowedAttemptError(
                f"Child attempt {attempt_id} не совпадает с активным для {window_id}"
            )
        if child.status is not WindowChildStatus.RUNNING:
            raise WindowedAttemptError(f"Окно {window_id} не выполняется")
        if not reason.strip():
            raise WindowedAttemptError("Причина ошибки child attempt не задана")
        state = self._replace_child(
            replace(child, status=WindowChildStatus.FAILED)
        )
        return replace(
            state,
            status=WindowedAttemptStatus.FAILED,
            stop_reason=f"{window_id}: {reason}",
        )

    def begin_reconciliation(self) -> WindowedAttemptState:
        self._require_status(WindowedAttemptStatus.RUNNING)
        unfinished = tuple(
            child.window_id
            for child in self.children
            if child.status is not WindowChildStatus.COMPLETED
        )
        if unfinished:
            raise WindowedAttemptError(
                f"Перед reconciliation не завершены окна: {unfinished}"
            )
        return replace(self, status=WindowedAttemptStatus.RECONCILING)

    def complete(self) -> WindowedAttemptState:
        if self.status is not WindowedAttemptStatus.RECONCILING:
            raise WindowedAttemptError(
                "Завершение parent attempt возможно только после reconciliation"
            )
        return replace(self, status=WindowedAttemptStatus.COMPLETED)

    def fail(self, reason: str) -> WindowedAttemptState:
        self._reject_terminal()
        if not reason.strip():
            raise WindowedAttemptError("Причина ошибки parent attempt не задана")
        return replace(
            self,
            status=WindowedAttemptStatus.FAILED,
            stop_reason=reason,
        )

    def cancel(self, reason: str) -> WindowedAttemptState:
        self._reject_terminal()
        if not reason.strip():
            raise WindowedAttemptError("Причина отмены parent attempt не задана")
        return replace(
            self,
            status=WindowedAttemptStatus.CANCELLED,
            stop_reason=reason,
            children=tuple(
                replace(child, status=WindowChildStatus.CANCELLED)
                for child in self.children
            ),
        )

    def _child(self, window_id: str) -> WindowChildState:
        try:
            return next(
                child for child in self.children if child.window_id == window_id
            )
        except StopIteration as error:
            raise WindowedAttemptError(f"Неизвестное окно {window_id}") from error

    def _replace_child(self, updated: WindowChildState) -> WindowedAttemptState:
        return replace(
            self,
            children=tuple(
                updated if child.window_id == updated.window_id else child
                for child in self.children
            ),
        )

    def _require_status(self, expected: WindowedAttemptStatus) -> None:
        self._reject_terminal()
        if self.status is not expected:
            raise WindowedAttemptError(
                f"Parent attempt имеет статус {self.status}, ожидался {expected}"
            )

    def _reject_terminal(self) -> None:
        if self.status in {
            WindowedAttemptStatus.COMPLETED,
            WindowedAttemptStatus.FAILED,
            WindowedAttemptStatus.CANCELLED,
        }:
            raise WindowedAttemptError(
                f"Операция недоступна в терминальном статусе {self.status}"
            )
