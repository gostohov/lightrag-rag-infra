from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from types import TracebackType
from typing import Any

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command as LangGraphCommand

from ...application.workflow import WorkflowCommand, WorkflowError, WorkflowState
from .graph import build_workflow_graph


class SqliteWorkflowRuntime:
    """Выполняет команды графа и хранит checkpoint в базе текущего прогона."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self._connection: sqlite3.Connection | None = None
        self._graph: Any = None

    def __enter__(self) -> SqliteWorkflowRuntime:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.database_path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS workflow_journal (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_id TEXT NOT NULL,
                command TEXT NOT NULL,
                result TEXT NOT NULL,
                error TEXT
            )
            """
        )
        self._connection.commit()
        checkpointer = SqliteSaver(self._connection)
        checkpointer.setup()
        self._graph = build_workflow_graph(checkpointer)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._connection is not None:
            self._connection.close()
        self._connection = None
        self._graph = None

    def execute(self, thread_id: str, command: WorkflowCommand) -> WorkflowState:
        graph = self._require_graph()
        config = self._config(thread_id)
        try:
            snapshot = graph.get_state(config)
            graph_input: object = (
                LangGraphCommand(resume=command.to_dict())
                if snapshot.next
                else {"command": command.to_dict()}
            )
            result = graph.invoke(graph_input, config=config)
        except WorkflowError as error:
            self._append_journal(thread_id, command, "rejected", str(error))
            raise
        self._append_journal(thread_id, command, "accepted", None)
        return WorkflowState.from_dict(result.get("workflow"))

    def current_state(self, thread_id: str) -> WorkflowState:
        snapshot = self._require_graph().get_state(self._config(thread_id))
        return WorkflowState.from_dict(snapshot.values.get("workflow") if snapshot.values else None)

    def waiting_for_input(self, thread_id: str) -> bool:
        return bool(self._require_graph().get_state(self._config(thread_id)).next)

    def journal(self, thread_id: str) -> list[dict[str, Any]]:
        connection = self._require_connection()
        rows = connection.execute(
            """
            SELECT sequence, command, result, error
            FROM workflow_journal WHERE thread_id = ? ORDER BY sequence
            """,
            (thread_id,),
        ).fetchall()
        return [
            {
                "sequence": int(row["sequence"]),
                "command": json.loads(row["command"]),
                "result": row["result"],
                "error": row["error"],
            }
            for row in rows
        ]

    def _append_journal(
        self,
        thread_id: str,
        command: WorkflowCommand,
        result: str,
        error: str | None,
    ) -> None:
        connection = self._require_connection()
        connection.execute(
            """
            INSERT INTO workflow_journal(thread_id, command, result, error)
            VALUES (?, ?, ?, ?)
            """,
            (
                thread_id,
                json.dumps(command.to_dict(), ensure_ascii=False, sort_keys=True),
                result,
                error,
            ),
        )
        connection.commit()

    def _require_graph(self) -> Any:
        if self._graph is None:
            raise RuntimeError("SqliteWorkflowRuntime нужно открыть через with")
        return self._graph

    def _require_connection(self) -> sqlite3.Connection:
        if self._connection is None:
            raise RuntimeError("SqliteWorkflowRuntime нужно открыть через with")
        return self._connection

    @staticmethod
    def _config(thread_id: str) -> dict[str, dict[str, str]]:
        return {"configurable": {"thread_id": thread_id}}
