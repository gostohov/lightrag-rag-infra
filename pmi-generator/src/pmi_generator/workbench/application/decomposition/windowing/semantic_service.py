from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict

from ....domain.source import SourceDocument
from ...repositories import UnitOfWork
from ...state import StoredRecord
from .canonicalization import SemanticWindowCanonicalizer, SemanticWindowError
from .models import (
    WindowChildStatus,
    WindowedAttemptState,
    WindowedAttemptStatus,
)
from .plan import WindowPlan
from .semantic import (
    SEMANTIC_WINDOW_SCHEMA_VERSION,
    SemanticWindowArguments,
    SemanticWindowResult,
)


class SemanticWindowService:
    RECORD_KIND = "decomposition_window_semantic_facts"
    STORAGE_SCHEMA_VERSION = 2

    def __init__(
        self,
        *,
        document: SourceDocument,
        uow_factory: Callable[[], UnitOfWork],
    ) -> None:
        self.uow_factory = uow_factory
        self.canonicalizer = SemanticWindowCanonicalizer(document)

    def validate(
        self,
        *,
        parent: WindowedAttemptState,
        plan: WindowPlan,
        window_id: str,
        child_attempt_id: str,
        arguments: SemanticWindowArguments,
    ) -> SemanticWindowResult:
        return self.canonicalizer.canonicalize(
            parent=parent,
            plan=plan,
            window_id=window_id,
            child_attempt_id=child_attempt_id,
            arguments=arguments,
        )

    def accept(
        self,
        *,
        parent: WindowedAttemptState,
        plan: WindowPlan,
        window_id: str,
        child_attempt_id: str,
        arguments: SemanticWindowArguments,
        raw_arguments: dict[str, object],
        uow: UnitOfWork | None = None,
    ) -> SemanticWindowResult:
        if raw_arguments != asdict(arguments):
            raise SemanticWindowError(
                "Raw semantic arguments не соответствуют typed arguments"
            )
        result = self.validate(
            parent=parent,
            plan=plan,
            window_id=window_id,
            child_attempt_id=child_attempt_id,
            arguments=arguments,
        )
        if uow is None:
            with self.uow_factory() as owned_uow:
                self._save(result, raw_arguments, owned_uow)
        else:
            self._save(result, raw_arguments, uow)
        return result

    def load(
        self,
        window_id: str,
        *,
        parent: WindowedAttemptState,
        plan: WindowPlan,
    ) -> SemanticWindowResult | None:
        record_id = f"{parent.parent_attempt_id}:{window_id}"
        with self.uow_factory() as uow:
            record = uow.records.get(self.RECORD_KIND, record_id)
        if record is None:
            return None
        payload = record.payload
        if set(payload) != {
            "schema_version",
            "contract_version",
            "raw_arguments",
            "context_only_behaviors",
            "validated",
            "fingerprint",
        }:
            raise SemanticWindowError(
                "Semantic fact storage содержит неизвестные поля"
            )
        if int(payload["schema_version"]) != self.STORAGE_SCHEMA_VERSION:
            raise SemanticWindowError(
                "Неизвестная semantic fact storage schema"
            )
        if payload["contract_version"] != SEMANTIC_WINDOW_SCHEMA_VERSION:
            raise SemanticWindowError(
                "Неизвестная semantic fact contract version"
            )
        self._verify_fingerprint(payload, "semantic fact storage")
        self._validate_recovery_binding(
            parent=parent,
            plan=plan,
            window_id=window_id,
        )
        raw = self._mapping(payload["raw_arguments"], "raw arguments")
        if set(raw) != {"behaviors"} or not isinstance(
            raw["behaviors"],
            list,
        ):
            raise SemanticWindowError(
                "Raw semantic facts имеют неверную структуру"
            )
        child = next(
            item for item in parent.children if item.window_id == window_id
        )
        assert child.attempt_id is not None
        active_parent = self._active_parent_for_child(
            parent,
            window_id,
        )
        recomputed = self.validate(
            parent=active_parent,
            plan=plan,
            window_id=window_id,
            child_attempt_id=child.attempt_id,
            arguments=SemanticWindowArguments(behaviors=raw["behaviors"]),
        )
        expected_context_only = (
            len(raw["behaviors"]) - len(recomputed.fragments)
        )
        if payload["context_only_behaviors"] != expected_context_only:
            raise SemanticWindowError(
                "Счётчик context-only behaviors не воспроизводится"
            )
        stored = SemanticWindowResult.from_dict(
            self._mapping(payload["validated"], "validated facts")
        )
        if stored != recomputed:
            raise SemanticWindowError(
                "Validated semantic facts не соответствуют raw arguments"
            )
        return stored

    @staticmethod
    def _active_parent_for_child(
        parent: WindowedAttemptState,
        window_id: str,
    ) -> WindowedAttemptState:
        from dataclasses import replace

        return replace(
            parent,
            status=WindowedAttemptStatus.RUNNING,
            children=tuple(
                replace(item, status=WindowChildStatus.RUNNING)
                if item.window_id == window_id
                else item
                for item in parent.children
            ),
        )

    @staticmethod
    def _validate_recovery_binding(
        *,
        parent: WindowedAttemptState,
        plan: WindowPlan,
        window_id: str,
    ) -> None:
        if parent.status not in {
            WindowedAttemptStatus.RUNNING,
            WindowedAttemptStatus.RECONCILING,
        }:
            raise SemanticWindowError("Parent attempt не выполняется")
        if (
            parent.window_plan_hash != plan.plan_hash
            or plan.recompute_hash() != plan.plan_hash
            or parent.selection_id != plan.selection_id
            or parent.document_version != plan.document_version
            or parent.policy_version != plan.policy_version
            or parent.schema_version != SEMANTIC_WINDOW_SCHEMA_VERSION
        ):
            raise SemanticWindowError(
                "Semantic fact recovery имеет stale parent/plan binding"
            )
        try:
            child = next(
                item
                for item in parent.children
                if item.window_id == window_id
            )
            next(
                item for item in plan.windows if item.window_id == window_id
            )
        except StopIteration as error:
            raise SemanticWindowError(
                f"Неизвестное окно {window_id}"
            ) from error
        if (
            child.attempt_id is None
            or child.status
            not in {WindowChildStatus.RUNNING, WindowChildStatus.COMPLETED}
        ):
            raise SemanticWindowError(
                "Сохранённые semantic facts несовместимы с child state"
            )

    def _save(
        self,
        result: SemanticWindowResult,
        raw_arguments: dict[str, object],
        uow: UnitOfWork,
    ) -> None:
        payload: dict[str, object] = {
            "schema_version": self.STORAGE_SCHEMA_VERSION,
            "contract_version": SEMANTIC_WINDOW_SCHEMA_VERSION,
            "raw_arguments": raw_arguments,
            "context_only_behaviors": (
                len(raw_arguments["behaviors"]) - len(result.fragments)
            ),
            "validated": result.to_dict(),
        }
        payload = json.loads(
            json.dumps(payload, ensure_ascii=False)
        )
        payload["fingerprint"] = self._fingerprint(payload)
        record_id = f"{result.parent_attempt_id}:{result.window_id}"
        existing = uow.records.get(self.RECORD_KIND, record_id)
        if existing is not None:
            if existing.payload.get("fingerprint") != payload["fingerprint"]:
                raise SemanticWindowError(
                    "Для окна уже сохранены другие semantic facts"
                )
            return
        uow.records.save(StoredRecord(self.RECORD_KIND, record_id, payload))
        uow.events.append(
            result.parent_attempt_id,
            "semantic facts окна сохранены",
            {
                "window_id": result.window_id,
                "child_attempt_id": result.child_attempt_id,
                "fragments": len(result.fragments),
                "context_only_behaviors": (
                    len(raw_arguments["behaviors"])
                    - len(result.fragments)
                ),
                "contract_version": SEMANTIC_WINDOW_SCHEMA_VERSION,
            },
        )

    @classmethod
    def _verify_fingerprint(
        cls,
        payload: dict[str, object],
        label: str,
    ) -> None:
        expected = payload.get("fingerprint")
        actual = cls._fingerprint(
            {
                key: value
                for key, value in payload.items()
                if key != "fingerprint"
            }
        )
        if expected != actual:
            raise SemanticWindowError(
                f"{label} fingerprint не соответствует содержимому"
            )

    @staticmethod
    def _mapping(value: object, label: str) -> dict[str, object]:
        if not isinstance(value, dict):
            raise SemanticWindowError(f"{label} должен быть объектом")
        return value

    @staticmethod
    def _fingerprint(payload: dict[str, object]) -> str:
        return hashlib.sha256(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
