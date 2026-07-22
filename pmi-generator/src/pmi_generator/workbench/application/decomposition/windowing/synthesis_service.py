from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict

from ....domain.source import SourceDocument
from ...repositories import UnitOfWork
from ...state import StoredRecord
from .canonicalization import (
    SemanticSynthesisCanonicalizer,
    SemanticWindowError,
)
from .candidates import WindowCandidateResult
from .models import WindowedAttemptState
from .plan import WindowPlan
from .semantic import (
    SEMANTIC_CANONICAL_MAPPING_VERSION,
    SEMANTIC_SYNTHESIS_SCHEMA_VERSION,
    SemanticWindowResult,
)
from .synthesis import SemanticSynthesisArguments


class SemanticSynthesisService:
    RECORD_KIND = "decomposition_window_result"
    STORAGE_SCHEMA_VERSION = 5

    def __init__(
        self,
        *,
        document: SourceDocument,
        uow_factory: Callable[[], UnitOfWork],
    ) -> None:
        self.uow_factory = uow_factory
        self.canonicalizer = SemanticSynthesisCanonicalizer(document)

    def validate(
        self,
        *,
        parent: WindowedAttemptState,
        plan: WindowPlan,
        target_window_id: str,
        attempt_id: str,
        fact_results: tuple[SemanticWindowResult, ...],
        arguments: SemanticSynthesisArguments,
    ) -> WindowCandidateResult:
        return self.canonicalizer.canonicalize(
            parent=parent,
            plan=plan,
            target_window_id=target_window_id,
            attempt_id=attempt_id,
            fact_results=fact_results,
            arguments=arguments,
        )

    def accept(
        self,
        *,
        parent: WindowedAttemptState,
        plan: WindowPlan,
        target_window_id: str,
        attempt_id: str,
        fact_results: tuple[SemanticWindowResult, ...],
        arguments: SemanticSynthesisArguments,
        raw_arguments: dict[str, object],
        uow: UnitOfWork | None = None,
    ) -> WindowCandidateResult:
        if raw_arguments != asdict(arguments):
            raise SemanticWindowError(
                "Raw synthesis arguments не соответствуют typed arguments"
            )
        result = self.validate(
            parent=parent,
            plan=plan,
            target_window_id=target_window_id,
            attempt_id=attempt_id,
            fact_results=fact_results,
            arguments=arguments,
        )
        if uow is None:
            with self.uow_factory() as owned_uow:
                self._save(
                    result=result,
                    attempt_id=attempt_id,
                    raw_arguments=raw_arguments,
                    fact_results=fact_results,
                    uow=owned_uow,
                )
        else:
            self._save(
                result=result,
                attempt_id=attempt_id,
                raw_arguments=raw_arguments,
                fact_results=fact_results,
                uow=uow,
            )
        return result

    def load(
        self,
        *,
        parent: WindowedAttemptState,
        plan: WindowPlan,
        target_window_id: str,
        fact_results: tuple[SemanticWindowResult, ...],
    ) -> WindowCandidateResult | None:
        record_id = f"{parent.parent_attempt_id}:{target_window_id}"
        with self.uow_factory() as uow:
            record = uow.records.get(self.RECORD_KIND, record_id)
        if record is None:
            return None
        payload = record.payload
        if set(payload) != {
            "schema_version",
            "contract_version",
            "mapping_version",
            "attempt_id",
            "facts_fingerprint",
            "raw_synthesis",
            "canonical",
            "fingerprint",
        }:
            raise SemanticWindowError(
                "Semantic synthesis storage содержит неизвестные поля"
            )
        if int(payload["schema_version"]) != self.STORAGE_SCHEMA_VERSION:
            raise SemanticWindowError(
                "Неизвестная semantic synthesis storage schema"
            )
        if payload["contract_version"] != SEMANTIC_SYNTHESIS_SCHEMA_VERSION:
            raise SemanticWindowError(
                "Неизвестная semantic synthesis contract version"
            )
        if payload["mapping_version"] != SEMANTIC_CANONICAL_MAPPING_VERSION:
            raise SemanticWindowError(
                "Неизвестная semantic synthesis mapping version"
            )
        self._verify_fingerprint(payload, "semantic synthesis storage")
        if payload["facts_fingerprint"] != self._facts_fingerprint(
            fact_results
        ):
            raise SemanticWindowError(
                "Semantic synthesis относится к другим facts"
            )
        raw = self._mapping(payload["raw_synthesis"], "raw synthesis")
        if set(raw) != {"candidates"} or not isinstance(
            raw["candidates"],
            list,
        ):
            raise SemanticWindowError(
                "Raw semantic synthesis имеет неверную структуру"
            )
        recomputed = self.validate(
            parent=parent,
            plan=plan,
            target_window_id=target_window_id,
            attempt_id=str(payload["attempt_id"]),
            fact_results=fact_results,
            arguments=SemanticSynthesisArguments(
                candidates=raw["candidates"]
            ),
        )
        stored = WindowCandidateResult.from_dict(
            self._mapping(payload["canonical"], "canonical synthesis")
        )
        if stored != recomputed:
            raise SemanticWindowError(
                "Canonical result не соответствует raw synthesis"
            )
        return stored

    def _save(
        self,
        *,
        result: WindowCandidateResult,
        attempt_id: str,
        raw_arguments: dict[str, object],
        fact_results: tuple[SemanticWindowResult, ...],
        uow: UnitOfWork,
    ) -> None:
        payload: dict[str, object] = {
            "schema_version": self.STORAGE_SCHEMA_VERSION,
            "contract_version": SEMANTIC_SYNTHESIS_SCHEMA_VERSION,
            "mapping_version": SEMANTIC_CANONICAL_MAPPING_VERSION,
            "attempt_id": attempt_id,
            "facts_fingerprint": self._facts_fingerprint(fact_results),
            "raw_synthesis": raw_arguments,
            "canonical": result.to_dict(),
        }
        payload = json.loads(json.dumps(payload, ensure_ascii=False))
        payload["fingerprint"] = self._fingerprint(payload)
        record_id = f"{result.parent_attempt_id}:{result.window_id}"
        existing = uow.records.get(self.RECORD_KIND, record_id)
        if existing is not None:
            if existing.payload.get("fingerprint") != payload["fingerprint"]:
                raise SemanticWindowError(
                    "Для окна уже сохранён другой synthesis result"
                )
            return
        uow.records.save(StoredRecord(self.RECORD_KIND, record_id, payload))
        uow.events.append(
            result.parent_attempt_id,
            "semantic synthesis окна сохранён",
            {
                "window_id": result.window_id,
                "attempt_id": attempt_id,
                "candidates": len(result.candidates),
                "contract_version": SEMANTIC_SYNTHESIS_SCHEMA_VERSION,
                "mapping_version": SEMANTIC_CANONICAL_MAPPING_VERSION,
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

    @classmethod
    def _facts_fingerprint(
        cls,
        results: tuple[SemanticWindowResult, ...],
    ) -> str:
        return cls._fingerprint(
            {
                "results": [
                    result.to_dict()
                    for result in sorted(
                        results,
                        key=lambda item: item.window_id,
                    )
                ]
            }
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
