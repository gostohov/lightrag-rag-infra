from __future__ import annotations

import sqlite3
from pathlib import Path

from ...application.state import AttemptRecord, AttemptStatus, EventRecord, SessionRecord, StoredRecord
from ...domain import TestCard
from .card_codec import decode_card, encode_card
from .errors import StorageConflictError, StorageSchemaError
from .records import (
    decode_attempt,
    decode_payload,
    decode_session,
    decode_stored_record,
    encode_attempt,
    encode_payload,
    encode_session,
    encode_stored_record,
)


SCHEMA_VERSION = 1


def workbench_database_path(run_dir: Path) -> Path:
    return run_dir / "review" / "workbench.sqlite3"


def _prepare(connection: sqlite3.Connection) -> None:
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute(
        "CREATE TABLE IF NOT EXISTS schema_meta (version INTEGER NOT NULL)"
    )
    row = connection.execute("SELECT version FROM schema_meta LIMIT 1").fetchone()
    if row is None:
        connection.execute("INSERT INTO schema_meta(version) VALUES (?)", (SCHEMA_VERSION,))
    elif int(row["version"]) != SCHEMA_VERSION:
        raise StorageSchemaError(
            f"Версия базы {row['version']} несовместима с ожидаемой {SCHEMA_VERSION}"
        )
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS cards (
            card_id TEXT PRIMARY KEY,
            revision INTEGER NOT NULL,
            payload TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            payload TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS attempts (
            attempt_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            status TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            payload TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS attempts_session_status
            ON attempts(session_id, status, updated_at);
        CREATE TABLE IF NOT EXISTS records (
            kind TEXT NOT NULL,
            record_id TEXT NOT NULL,
            payload TEXT NOT NULL,
            PRIMARY KEY(kind, record_id)
        );
        CREATE TABLE IF NOT EXISTS events (
            sequence INTEGER PRIMARY KEY AUTOINCREMENT,
            aggregate_id TEXT NOT NULL,
            event_type TEXT NOT NULL,
            payload TEXT NOT NULL
        );
        """
    )


class SqliteCardRepository:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection

    def get(self, card_id: str) -> TestCard | None:
        row = self._connection.execute(
            "SELECT payload FROM cards WHERE card_id = ?",
            (card_id,),
        ).fetchone()
        return decode_card(row["payload"]) if row else None

    def list_all(self) -> list[TestCard]:
        rows = self._connection.execute("SELECT payload FROM cards ORDER BY card_id").fetchall()
        return [decode_card(row["payload"]) for row in rows]

    def save(self, card: TestCard) -> None:
        payload = encode_card(card)
        row = self._connection.execute(
            "SELECT revision, payload FROM cards WHERE card_id = ?",
            (card.card_id,),
        ).fetchone()
        if row is not None:
            revision = int(row["revision"])
            if revision > card.revision:
                raise StorageConflictError(
                    f"Карточка {card.card_id} уже сохранена в ревизии {revision}"
                )
            if revision == card.revision and row["payload"] == payload:
                return
        self._connection.execute(
            """
            INSERT INTO cards(card_id, revision, payload) VALUES (?, ?, ?)
            ON CONFLICT(card_id) DO UPDATE SET revision = excluded.revision, payload = excluded.payload
            """,
            (card.card_id, card.revision, payload),
        )


class SqliteSessionRepository:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection

    def get(self, session_id: str) -> SessionRecord | None:
        row = self._connection.execute(
            "SELECT payload FROM sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        return decode_session(row["payload"]) if row else None

    def list_all(self) -> list[SessionRecord]:
        rows = self._connection.execute(
            "SELECT payload FROM sessions ORDER BY session_id"
        ).fetchall()
        return [decode_session(row["payload"]) for row in rows]

    def save(self, session: SessionRecord) -> None:
        self._connection.execute(
            """
            INSERT INTO sessions(session_id, payload) VALUES (?, ?)
            ON CONFLICT(session_id) DO UPDATE SET payload = excluded.payload
            """,
            (session.session_id, encode_session(session)),
        )


class SqliteAttemptRepository:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection

    def get(self, attempt_id: str) -> AttemptRecord | None:
        row = self._connection.execute(
            "SELECT payload FROM attempts WHERE attempt_id = ?",
            (attempt_id,),
        ).fetchone()
        return decode_attempt(row["payload"]) if row else None

    def active_for_session(self, session_id: str) -> AttemptRecord | None:
        row = self._connection.execute(
            """
            SELECT payload FROM attempts
            WHERE session_id = ? AND status = ?
            ORDER BY updated_at DESC LIMIT 1
            """,
            (session_id, AttemptStatus.ACTIVE.value),
        ).fetchone()
        return decode_attempt(row["payload"]) if row else None

    def list_for_session(self, session_id: str) -> list[AttemptRecord]:
        rows = self._connection.execute(
            """
            SELECT payload FROM attempts
            WHERE session_id = ? ORDER BY updated_at, attempt_id
            """,
            (session_id,),
        ).fetchall()
        return [decode_attempt(row["payload"]) for row in rows]

    def list_all(self) -> list[AttemptRecord]:
        rows = self._connection.execute(
            "SELECT payload FROM attempts ORDER BY updated_at, attempt_id"
        ).fetchall()
        return [decode_attempt(row["payload"]) for row in rows]

    def save(self, attempt: AttemptRecord) -> None:
        self._connection.execute(
            """
            INSERT INTO attempts(attempt_id, session_id, status, updated_at, payload)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(attempt_id) DO UPDATE SET
                session_id = excluded.session_id,
                status = excluded.status,
                updated_at = excluded.updated_at,
                payload = excluded.payload
            """,
            (
                attempt.attempt_id,
                attempt.session_id,
                attempt.status.value,
                attempt.updated_at.isoformat(),
                encode_attempt(attempt),
            ),
        )


class SqliteRecordRepository:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection

    def get(self, kind: str, record_id: str) -> StoredRecord | None:
        row = self._connection.execute(
            "SELECT payload FROM records WHERE kind = ? AND record_id = ?",
            (kind, record_id),
        ).fetchone()
        return decode_stored_record(kind, record_id, row["payload"]) if row else None

    def list_kind(self, kind: str) -> list[StoredRecord]:
        rows = self._connection.execute(
            "SELECT record_id, payload FROM records WHERE kind = ? ORDER BY record_id",
            (kind,),
        ).fetchall()
        return [decode_stored_record(kind, row["record_id"], row["payload"]) for row in rows]

    def save(self, record: StoredRecord) -> None:
        self._connection.execute(
            """
            INSERT INTO records(kind, record_id, payload) VALUES (?, ?, ?)
            ON CONFLICT(kind, record_id) DO UPDATE SET payload = excluded.payload
            """,
            (record.kind, record.record_id, encode_stored_record(record)),
        )


class SqliteEventRepository:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection

    def append(self, aggregate_id: str, event_type: str, payload: dict[str, object]) -> int:
        cursor = self._connection.execute(
            "INSERT INTO events(aggregate_id, event_type, payload) VALUES (?, ?, ?)",
            (aggregate_id, event_type, encode_payload(payload)),
        )
        return int(cursor.lastrowid)

    def list_for(self, aggregate_id: str) -> list[EventRecord]:
        rows = self._connection.execute(
            """
            SELECT sequence, event_type, payload FROM events
            WHERE aggregate_id = ? ORDER BY sequence
            """,
            (aggregate_id,),
        ).fetchall()
        return [
            EventRecord(
                sequence=int(row["sequence"]),
                aggregate_id=aggregate_id,
                event_type=row["event_type"],
                payload=decode_payload(row["payload"]),
            )
            for row in rows
        ]


class SqliteUnitOfWork:
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self._connection: sqlite3.Connection | None = None

    def __enter__(self) -> SqliteUnitOfWork:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.database_path, isolation_level=None)
        _prepare(connection)
        connection.execute("BEGIN IMMEDIATE")
        self._connection = connection
        self.cards = SqliteCardRepository(connection)
        self.sessions = SqliteSessionRepository(connection)
        self.attempts = SqliteAttemptRepository(connection)
        self.records = SqliteRecordRepository(connection)
        self.events = SqliteEventRepository(connection)
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self._connection is None:
            return
        try:
            self._connection.execute("ROLLBACK" if exc_type else "COMMIT")
        finally:
            self._connection.close()
            self._connection = None
