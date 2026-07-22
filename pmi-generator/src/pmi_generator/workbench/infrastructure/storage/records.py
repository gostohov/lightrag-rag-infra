from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from ...application.state import AttemptRecord, AttemptStatus, SessionRecord, StoredRecord
from .errors import StorageError


def encode_session(value: SessionRecord) -> str:
    return _dump(
        {
            "session_id": value.session_id,
            "selection_id": value.selection_id,
            "card_id": value.card_id,
            "current_stage": value.current_stage,
            "payload": value.payload,
            "updated_at": value.updated_at.isoformat(),
        }
    )


def decode_session(raw: str) -> SessionRecord:
    value = _load(raw)
    return SessionRecord(
        session_id=value["session_id"],
        selection_id=value["selection_id"],
        card_id=value.get("card_id"),
        current_stage=value["current_stage"],
        payload=value["payload"],
        updated_at=datetime.fromisoformat(value["updated_at"]),
    )


def encode_attempt(value: AttemptRecord) -> str:
    return _dump(
        {
            "attempt_id": value.attempt_id,
            "session_id": value.session_id,
            "stage": value.stage,
            "status": value.status.value,
            "payload": value.payload,
            "updated_at": value.updated_at.isoformat(),
        }
    )


def decode_attempt(raw: str) -> AttemptRecord:
    value = _load(raw)
    return AttemptRecord(
        attempt_id=value["attempt_id"],
        session_id=value["session_id"],
        stage=value["stage"],
        status=AttemptStatus(value["status"]),
        payload=value["payload"],
        updated_at=datetime.fromisoformat(value["updated_at"]),
    )


def encode_stored_record(value: StoredRecord) -> str:
    return _dump(value.payload)


def decode_stored_record(kind: str, record_id: str, raw: str) -> StoredRecord:
    return StoredRecord(kind=kind, record_id=record_id, payload=_load(raw))


def encode_payload(value: dict[str, Any]) -> str:
    return _dump(value)


def decode_payload(raw: str) -> dict[str, Any]:
    return _load(raw)


def _dump(value: dict[str, Any]) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise StorageError(f"Запись содержит несериализуемое значение: {exc}") from exc


def _load(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise StorageError(f"Повреждённая JSON-запись в базе: {exc}") from exc
    if not isinstance(value, dict):
        raise StorageError("Запись в базе должна быть JSON-объектом")
    return value
