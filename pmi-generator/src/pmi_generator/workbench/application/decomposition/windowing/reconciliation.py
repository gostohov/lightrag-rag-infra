from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass

from ....domain.source import SourceDocument, SourcePosition
from ...repositories import UnitOfWork
from ...state import StoredRecord
from .candidates import ValidatedWindowCandidate
from .conflicts import ConflictGroup
from .models import WindowedAttemptState, WindowedAttemptStatus


class ReconciliationError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ReconciliationArguments:
    outcome: str
    accepted_candidate_ids: list[str]
    rejected_candidate_ids: list[str]
    resolved_dependency_ids: list[str]
    relations: list[dict[str, object]]
    explanation: str


@dataclass(frozen=True, slots=True)
class ReconciliationDecision:
    parent_attempt_id: str
    attempt_id: str
    group_id: str
    plan_hash: str
    outcome: str
    accepted_candidate_ids: tuple[str, ...]
    rejected_candidate_ids: tuple[str, ...]
    resolved_dependency_ids: tuple[str, ...]
    relations: tuple[dict[str, object], ...]
    explanation: str

    def to_dict(self) -> dict[str, object]:
        return {
            "parent_attempt_id": self.parent_attempt_id,
            "attempt_id": self.attempt_id,
            "group_id": self.group_id,
            "plan_hash": self.plan_hash,
            "outcome": self.outcome,
            "accepted_candidate_ids": list(self.accepted_candidate_ids),
            "rejected_candidate_ids": list(self.rejected_candidate_ids),
            "resolved_dependency_ids": list(self.resolved_dependency_ids),
            "relations": list(self.relations),
            "explanation": self.explanation,
        }

    @classmethod
    def from_dict(
        cls,
        value: dict[str, object],
    ) -> ReconciliationDecision:
        return cls(
            parent_attempt_id=str(value["parent_attempt_id"]),
            attempt_id=str(value["attempt_id"]),
            group_id=str(value["group_id"]),
            plan_hash=str(value["plan_hash"]),
            outcome=str(value["outcome"]),
            accepted_candidate_ids=tuple(
                str(item)
                for item in value["accepted_candidate_ids"]  # type: ignore[union-attr]
            ),
            rejected_candidate_ids=tuple(
                str(item)
                for item in value["rejected_candidate_ids"]  # type: ignore[union-attr]
            ),
            resolved_dependency_ids=tuple(
                str(item)
                for item in value["resolved_dependency_ids"]  # type: ignore[union-attr]
            ),
            relations=tuple(
                dict(item)
                for item in value["relations"]  # type: ignore[union-attr]
            ),
            explanation=str(value["explanation"]),
        )


class ReconciliationService:
    RECORD_KIND = "decomposition_reconciliation"
    SCHEMA_VERSION = 1

    def __init__(
        self,
        *,
        document: SourceDocument,
        uow_factory: Callable[[], UnitOfWork],
    ) -> None:
        self.document = document
        self.uow_factory = uow_factory

    def validate(
        self,
        *,
        parent: WindowedAttemptState,
        plan_hash: str,
        group: ConflictGroup,
        attempt_id: str,
        arguments: ReconciliationArguments,
    ) -> ReconciliationDecision:
        if not attempt_id.strip():
            raise ReconciliationError(
                "Reconciliation требует attempt ID"
            )
        if parent.status is not WindowedAttemptStatus.RECONCILING:
            raise ReconciliationError(
                "Parent attempt не находится в стадии reconciliation"
            )
        if parent.window_plan_hash != plan_hash:
            raise ReconciliationError(
                "Parent attempt относится к другому window plan"
            )
        if group.plan_hash != plan_hash:
            raise ReconciliationError(
                "Conflict group относится к другому window plan"
            )
        if group.parent_attempt_id != parent.parent_attempt_id:
            raise ReconciliationError(
                "Conflict group относится к другому parent attempt"
            )
        if arguments.outcome not in {"resolved", "unresolved"}:
            raise ReconciliationError(
                f"Неизвестный reconciliation outcome {arguments.outcome}"
            )
        if not arguments.explanation.strip():
            raise ReconciliationError(
                "Reconciliation требует объяснение"
            )

        accepted = self._unique_ids(
            arguments.accepted_candidate_ids,
            "accepted candidates",
        )
        rejected = self._unique_ids(
            arguments.rejected_candidate_ids,
            "rejected candidates",
        )
        resolved_dependencies = self._unique_ids(
            arguments.resolved_dependency_ids,
            "resolved dependencies",
        )
        relations = self._validate_relations(group, arguments.relations)
        if arguments.outcome == "unresolved":
            if accepted or rejected or resolved_dependencies or relations:
                raise ReconciliationError(
                    "Unresolved не содержит частичных решений"
                )
        else:
            if set(accepted) & set(rejected):
                raise ReconciliationError(
                    "Candidate одновременно принят и отклонён"
                )
            if set(accepted) | set(rejected) != set(group.candidate_ids):
                raise ReconciliationError(
                    "Resolved decision должна классифицировать все candidates"
                )
            if set(resolved_dependencies) != set(group.dependency_ids):
                raise ReconciliationError(
                    "Resolved decision должна закрыть все dependencies"
                )
            accepted_by_relations = {
                candidate_id
                for relation in relations
                for candidate_id in relation["candidate_ids"]
                if candidate_id in accepted
            }
            if len(accepted) > 1 and accepted_by_relations != set(accepted):
                raise ReconciliationError(
                    "Несколько accepted candidates требуют явную "
                    "merge/split relation"
                )

        return ReconciliationDecision(
            parent_attempt_id=parent.parent_attempt_id,
            attempt_id=attempt_id,
            group_id=group.group_id,
            plan_hash=plan_hash,
            outcome=arguments.outcome,
            accepted_candidate_ids=accepted,
            rejected_candidate_ids=rejected,
            resolved_dependency_ids=resolved_dependencies,
            relations=relations,
            explanation=arguments.explanation,
        )

    def accept(
        self,
        *,
        parent: WindowedAttemptState,
        plan_hash: str,
        group: ConflictGroup,
        attempt_id: str,
        arguments: ReconciliationArguments,
        raw_arguments: dict[str, object],
        uow: UnitOfWork | None = None,
    ) -> ReconciliationDecision:
        if raw_arguments != asdict(arguments):
            raise ReconciliationError(
                "Raw reconciliation arguments не соответствуют typed arguments"
            )
        decision = self.validate(
            parent=parent,
            plan_hash=plan_hash,
            group=group,
            attempt_id=attempt_id,
            arguments=arguments,
        )
        if uow is not None:
            self._save(decision, raw_arguments, uow)
        else:
            with self.uow_factory() as owned_uow:
                self._save(decision, raw_arguments, owned_uow)
        return decision

    def load(
        self,
        *,
        parent_attempt_id: str,
        group_id: str,
        plan_hash: str,
    ) -> ReconciliationDecision | None:
        record_id = self._record_id(parent_attempt_id, group_id)
        with self.uow_factory() as uow:
            record = uow.records.get(self.RECORD_KIND, record_id)
        if record is None:
            return None
        if int(record.payload.get("schema_version", -1)) != self.SCHEMA_VERSION:
            raise ReconciliationError(
                "Неизвестная reconciliation storage schema"
            )
        fingerprint_payload = {
            key: value
            for key, value in record.payload.items()
            if key != "fingerprint"
        }
        if (
            self._fingerprint(fingerprint_payload)
            != record.payload.get("fingerprint")
        ):
            raise ReconciliationError(
                "Reconciliation fingerprint не соответствует содержимому"
            )
        decision = ReconciliationDecision.from_dict(
            dict(record.payload["decision"])
        )
        if (
            decision.parent_attempt_id != parent_attempt_id
            or decision.group_id != group_id
            or decision.plan_hash != plan_hash
        ):
            raise ReconciliationError(
                "Reconciliation decision имеет stale binding"
            )
        return decision

    def _validate_relations(
        self,
        group: ConflictGroup,
        raw_relations: object,
    ) -> tuple[dict[str, object], ...]:
        if not isinstance(raw_relations, list):
            raise ReconciliationError("relations должен быть списком")
        if len(raw_relations) > 64:
            raise ReconciliationError(
                "Reconciliation превышает relation budget"
            )
        allowed_positions = {
            line.position for line in group.source_lines
        }
        candidates = {
            candidate.candidate_id: candidate
            for candidate in group.candidates
        }
        if set(candidates) != set(group.candidate_ids):
            raise ReconciliationError(
                "Conflict group не содержит payload всех candidates"
            )
        result: list[dict[str, object]] = []
        for raw in raw_relations:
            if not isinstance(raw, dict) or set(raw) != {
                "kind",
                "candidate_ids",
                "reason",
            }:
                raise ReconciliationError(
                    "Reconciliation relation имеет неверную структуру"
                )
            kind = str(raw["kind"])
            if kind not in {"merge", "split"}:
                raise ReconciliationError(
                    f"Неизвестный relation kind {kind}"
                )
            candidate_ids = self._unique_ids(
                raw["candidate_ids"],
                "relation candidates",
            )
            if (
                len(candidate_ids) < 2
                or not set(candidate_ids) <= set(group.candidate_ids)
            ):
                raise ReconciliationError(
                    "Relation требует минимум два известных candidate ID"
                )
            reason = str(raw["reason"])
            if not reason.strip():
                raise ReconciliationError(
                    "Relation требует причину"
                )
            source = self._candidate_source(
                candidate_ids,
                candidates,
                allowed_positions,
            )
            result.append(
                {
                    "kind": kind,
                    "candidate_ids": list(candidate_ids),
                    "source": source,
                    "reason": reason,
                }
            )
        return tuple(result)

    def _candidate_source(
        self,
        candidate_ids: tuple[str, ...],
        candidates: dict[str, ValidatedWindowCandidate],
        allowed_positions: set[SourcePosition],
    ) -> list[dict[str, object]]:
        positions = {
            position
            for candidate_id in candidate_ids
            for position in candidates[candidate_id].evidence_positions
        }
        if not positions:
            raise ReconciliationError(
                "Relation candidates не содержат evidence"
            )
        if not positions <= allowed_positions:
            raise ReconciliationError(
                "Relation candidate evidence выходит за bounded "
                "conflict context"
            )
        ordered = sorted(
            positions,
            key=lambda item: (item.page_index, item.line_number),
        )
        ranges: list[tuple[SourcePosition, SourcePosition]] = []
        start = ordered[0]
        end = ordered[0]
        for position in ordered[1:]:
            if (
                position.page_index == end.page_index
                and position.line_number == end.line_number + 1
            ):
                end = position
                continue
            ranges.append((start, end))
            start = position
            end = position
        ranges.append((start, end))

        result: list[dict[str, object]] = []
        for start, end in ranges:
            excerpt = self.document.select(start, end)
            result.append(
                {
                    "page": start.page_index,
                    "line_start": start.line_number,
                    "line_end": end.line_number,
                    "quote": excerpt.text,
                }
            )
        return result

    def _save(
        self,
        decision: ReconciliationDecision,
        raw_arguments: dict[str, object],
        uow: UnitOfWork,
    ) -> None:
        payload = json.loads(
            json.dumps(
                {
                    "schema_version": self.SCHEMA_VERSION,
                    "raw_arguments": raw_arguments,
                    "decision": decision.to_dict(),
                },
                ensure_ascii=False,
            )
        )
        payload["fingerprint"] = self._fingerprint(payload)
        record_id = self._record_id(
            decision.parent_attempt_id,
            decision.group_id,
        )
        existing = uow.records.get(self.RECORD_KIND, record_id)
        if existing is not None:
            if existing.payload.get("fingerprint") != payload["fingerprint"]:
                raise ReconciliationError(
                    "Для conflict group уже сохранено другое решение"
                )
            return
        uow.records.save(
            StoredRecord(self.RECORD_KIND, record_id, payload)
        )
        uow.events.append(
            decision.parent_attempt_id,
            "reconciliation decision сохранено",
            {
                "group_id": decision.group_id,
                "outcome": decision.outcome,
            },
        )

    @staticmethod
    def _unique_ids(raw: object, label: str) -> tuple[str, ...]:
        if not isinstance(raw, list):
            raise ReconciliationError(f"{label} должен быть списком")
        values = tuple(str(item) for item in raw)
        if (
            any(not item.strip() for item in values)
            or len(values) != len(set(values))
        ):
            raise ReconciliationError(
                f"{label} должны быть непустыми и уникальными"
            )
        return tuple(sorted(values))

    @staticmethod
    def _record_id(parent_attempt_id: str, group_id: str) -> str:
        return f"{parent_attempt_id}:{group_id}"

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


def reconciliation_context(group: ConflictGroup) -> dict[str, object]:
    return group.to_context()
