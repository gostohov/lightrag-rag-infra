from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from ...application.workflow import (
    CommandKind,
    WorkflowCommand,
    WorkflowStage,
    WorkflowState,
    apply_command,
)


class GraphState(TypedDict, total=False):
    workflow: dict[str, Any]
    command: dict[str, Any] | None


COMMAND_NODES: dict[CommandKind, str] = {
    CommandKind.CONFIRM_SELECTION: "диапазон",
    CommandKind.APPLY_DECOMPOSITION: "декомпозиция",
    CommandKind.TAKE_SKELETON: "решение по каркасу",
    CommandKind.EXCLUDE_SKELETON: "решение по каркасу",
    CommandKind.BEGIN_ATTEMPT: "попытка",
    CommandKind.CANCEL_ATTEMPT: "попытка",
    CommandKind.FAIL_ATTEMPT: "попытка",
    CommandKind.APPLY_ATTEMPT_RESULT: "попытка",
    CommandKind.REQUEST_ANALYST: "решение аналитика",
    CommandKind.REFINE_CARD: "доработка",
    CommandKind.DECIDE_CARD: "решение по карточке",
    CommandKind.SAVE_RANGE_REVIEW: "проверка диапазона",
    CommandKind.CONTINUE_WITH_ISSUES: "проверка диапазона",
    CommandKind.REQUEST_EXPORT: "экспорт",
}


def route_command(state: GraphState) -> str:
    raw_command = state.get("command")
    if raw_command is None:
        return "без команды"
    return COMMAND_NODES[WorkflowCommand.from_dict(raw_command).kind]


def command_node(*allowed: CommandKind):
    expected = set(allowed)

    def execute(state: GraphState) -> GraphState:
        raw_command = state.get("command")
        if raw_command is None:
            return {}
        command = WorkflowCommand.from_dict(raw_command)
        if command.kind not in expected:
            raise ValueError(f"Команда {command.kind.value} попала в неверный узел")
        current = WorkflowState.from_dict(state.get("workflow"))
        updated = apply_command(current, command)
        return {"workflow": updated.to_dict(), "command": None}

    return execute


def no_command(state: GraphState) -> GraphState:
    return {}


def wait_for_analyst(state: GraphState) -> GraphState:
    workflow = WorkflowState.from_dict(state.get("workflow"))
    resumed_command = interrupt(
        {
            "stage": workflow.stage.value,
            "selection_id": workflow.selection_id,
        }
    )
    return {"command": resumed_command}


def after_command(state: GraphState) -> str:
    workflow = WorkflowState.from_dict(state.get("workflow"))
    if workflow.stage in {
        WorkflowStage.DECOMPOSITION_REVIEW,
        WorkflowStage.CARD_WORK,
        WorkflowStage.ANALYST_DECISION,
        WorkflowStage.RANGE_REVIEWED,
    }:
        return "ожидание аналитика"
    return "завершить"


def build_workflow_graph(checkpointer: object) -> object:
    builder = StateGraph(GraphState)
    nodes = {
        "диапазон": command_node(CommandKind.CONFIRM_SELECTION),
        "декомпозиция": command_node(CommandKind.APPLY_DECOMPOSITION),
        "решение по каркасу": command_node(
            CommandKind.TAKE_SKELETON,
            CommandKind.EXCLUDE_SKELETON,
        ),
        "попытка": command_node(
            CommandKind.BEGIN_ATTEMPT,
            CommandKind.CANCEL_ATTEMPT,
            CommandKind.FAIL_ATTEMPT,
            CommandKind.APPLY_ATTEMPT_RESULT,
        ),
        "решение аналитика": command_node(CommandKind.REQUEST_ANALYST),
        "доработка": command_node(CommandKind.REFINE_CARD),
        "решение по карточке": command_node(CommandKind.DECIDE_CARD),
        "проверка диапазона": command_node(
            CommandKind.SAVE_RANGE_REVIEW,
            CommandKind.CONTINUE_WITH_ISSUES,
        ),
        "экспорт": command_node(CommandKind.REQUEST_EXPORT),
        "без команды": no_command,
    }
    for name, node in nodes.items():
        builder.add_node(name, node)
        if name == "без команды":
            builder.add_edge(name, END)
        else:
            builder.add_conditional_edges(
                name,
                after_command,
                {
                    "ожидание аналитика": "ожидание аналитика",
                    "завершить": END,
                },
            )
    builder.add_node("ожидание аналитика", wait_for_analyst)
    builder.add_conditional_edges(
        "ожидание аналитика",
        route_command,
        {name: name for name in nodes},
    )
    builder.add_conditional_edges(START, route_command, list(nodes))
    return builder.compile(checkpointer=checkpointer)
