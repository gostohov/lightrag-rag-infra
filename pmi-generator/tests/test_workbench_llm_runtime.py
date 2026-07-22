from __future__ import annotations

import asyncio
import json
import tempfile
import threading
import unittest
from time import sleep
from dataclasses import dataclass, replace
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

from pmi_generator.workbench.application.llm import (
    AttemptDiscardedError,
    LlmToolRuntime,
    RawCompletion,
    TechnicalLlmError,
    ToolContractError,
    ToolSpec,
    TypedToolRegistry,
)
from pmi_generator.workbench.application.metrics import collect_metrics
from pmi_generator.workbench.application.prompting import PromptId, default_policy
from pmi_generator.workbench.application.repositories import UnitOfWork
from pmi_generator.workbench.application.state import AttemptStatus, StoredRecord
from pmi_generator.workbench.infrastructure.llm import ScriptedLlmTransport
from pmi_generator.workbench.infrastructure.llm import (
    OpenAICompatibleTransport,
    OpenAITransportSettings,
)
from pmi_generator.workbench.infrastructure.storage import (
    InMemoryDatabase,
    InMemoryUnitOfWork,
    SqliteUnitOfWork,
)


@dataclass(frozen=True, slots=True)
class SubmittedResult:
    outcome: str
    count: int


@dataclass(frozen=True, slots=True)
class NestedSubmittedResult:
    outcome: str
    items: list[dict[str, str]]


def registry() -> TypedToolRegistry:
    tools = TypedToolRegistry()
    tools.register(
        ToolSpec(
            name="submit_result",
            description="Сохранить атомарный результат",
            arguments_type=SubmittedResult,
            json_schema={
                "type": "object",
                "properties": {
                    "outcome": {"type": "string"},
                    "count": {"type": "integer"},
                },
                "required": ["outcome", "count"],
                "additionalProperties": False,
            },
        )
    )
    return tools


def completion(
    *,
    tool: str = "submit_result",
    arguments: object = None,
    finish_reason: str = "tool_calls",
) -> RawCompletion:
    return RawCompletion(
        finish_reason=finish_reason,
        tool_calls=(
            {
                "id": "call-1",
                "name": tool,
                "arguments": arguments or {"outcome": "ok", "count": 1},
            },
        ),
        usage={"prompt_tokens": 10, "completion_tokens": 5},
        model="test-model",
    )


class LlmToolRuntimeTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.database = InMemoryDatabase()
        self.call = replace(
            default_policy().build_call(
                PromptId.DECOMPOSITION,
                {"selection": "Требование"},
            ),
            allowed_tools=("submit_result",),
        )

    def runtime(self, transport: ScriptedLlmTransport, *, retries: int = 0) -> LlmToolRuntime:
        return LlmToolRuntime(
            transport=transport,
            tools=registry(),
            uow_factory=lambda: InMemoryUnitOfWork(self.database),
            max_retries=retries,
        )

    async def test_valid_tool_call_decodes_to_typed_result(self) -> None:
        runtime = self.runtime(ScriptedLlmTransport([completion()]))

        result = await runtime.invoke("ATTEMPT_0001", "SESSION_0001", self.call)

        self.assertEqual(result.name, "submit_result")
        self.assertEqual(result.arguments, SubmittedResult(outcome="ok", count=1))
        with InMemoryUnitOfWork(self.database) as uow:
            self.assertEqual(
                uow.attempts.get("ATTEMPT_0001").status,
                AttemptStatus.RESULT_READY,
            )

        runtime.begin_apply("ATTEMPT_0001")
        runtime.complete_apply("ATTEMPT_0001")

        with InMemoryUnitOfWork(self.database) as uow:
            self.assertEqual(
                uow.attempts.get("ATTEMPT_0001").status,
                AttemptStatus.COMPLETED,
            )

    async def test_invalid_contextual_schema_does_not_create_attempt(self) -> None:
        def invalid_schema(_context: dict[str, object]) -> dict[str, object]:
            raise ValueError("invalid context")

        tools = TypedToolRegistry()
        tools.register(
            ToolSpec(
                name="submit_result",
                description="Сохранить атомарный результат",
                arguments_type=SubmittedResult,
                json_schema={
                    "type": "object",
                    "properties": {
                        "outcome": {"type": "string"},
                        "count": {"type": "integer"},
                    },
                    "required": ["outcome", "count"],
                    "additionalProperties": False,
                },
                contextual_schema=invalid_schema,
            )
        )
        transport = ScriptedLlmTransport([completion()])
        runtime = LlmToolRuntime(
            transport=transport,
            tools=tools,
            uow_factory=lambda: InMemoryUnitOfWork(self.database),
        )

        with self.assertRaisesRegex(
            ToolContractError,
            "не смог построить контекстную JSON Schema",
        ):
            await runtime.invoke(
                "ATTEMPT_BAD_SCHEMA",
                "SESSION_0001",
                self.call,
            )

        self.assertFalse(transport.calls)
        with InMemoryUnitOfWork(self.database) as uow:
            self.assertIsNone(uow.attempts.get("ATTEMPT_BAD_SCHEMA"))

    async def test_unknown_tool_and_invalid_arguments_are_rejected_atomically(self) -> None:
        for attempt_id, response in (
            ("ATTEMPT_UNKNOWN", completion(tool="unknown")),
            ("ATTEMPT_INVALID", completion(arguments={"outcome": "ok", "count": "one"})),
        ):
            with self.subTest(attempt=attempt_id):
                runtime = self.runtime(ScriptedLlmTransport([response]))
                with self.assertRaises(ToolContractError):
                    await runtime.invoke(attempt_id, "SESSION_0001", self.call)
                with InMemoryUnitOfWork(self.database) as uow:
                    self.assertEqual(uow.attempts.get(attempt_id).status, AttemptStatus.FAILED)

    def test_registry_enforces_nested_json_schema_and_enum(self) -> None:
        tools = TypedToolRegistry()
        tools.register(
            ToolSpec(
                name="submit_nested",
                description="Проверить рекурсивную JSON Schema",
                arguments_type=NestedSubmittedResult,
                json_schema={
                    "type": "object",
                    "properties": {
                        "outcome": {"type": "string", "enum": ["accepted"]},
                        "items": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {"path": {"type": "string", "enum": ["known.path"]}},
                                "required": ["path"],
                                "additionalProperties": False,
                            },
                        },
                    },
                    "required": ["outcome", "items"],
                    "additionalProperties": False,
                },
            )
        )

        invalid_calls = (
            {"outcome": "unknown", "items": [{"path": "known.path"}]},
            {"outcome": "accepted", "items": [{"path": "unknown.path"}]},
            {
                "outcome": "accepted",
                "items": [{"path": "known.path", "unexpected": "value"}],
            },
        )
        for index, arguments in enumerate(invalid_calls):
            with self.subTest(index=index), self.assertRaises(ToolContractError):
                tools.decode(
                    {
                        "id": f"call-{index}",
                        "name": "submit_nested",
                        "arguments": arguments,
                    },
                    ("submit_nested",),
                )

    def test_wire_schema_omits_vllm_grammar_limits_but_local_validation_keeps_them(
        self,
    ) -> None:
        spec = ToolSpec(
            name="submit_grammar_compatible",
            description="Разделить wire schema и локальную проверку",
            arguments_type=NestedSubmittedResult,
            json_schema={
                "type": "object",
                "properties": {
                    "outcome": {"type": "string", "enum": ["accepted"]},
                    "items": {
                        "type": "array",
                        "minItems": 1,
                        "uniqueItems": True,
                        "items": {
                            "type": "object",
                            "properties": {
                                "path": {
                                    "type": "string",
                                    "minLength": 1,
                                    "enum": ["known.path"],
                                }
                            },
                            "required": ["path"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["outcome", "items"],
                "additionalProperties": False,
            },
        )
        tools = TypedToolRegistry()
        tools.register(spec)

        wire_schema = spec.openai_schema()["function"]["parameters"]
        wire_text = json.dumps(wire_schema, sort_keys=True)
        self.assertNotIn("uniqueItems", wire_text)
        self.assertNotIn("minItems", wire_text)
        self.assertNotIn("minLength", wire_text)
        self.assertNotIn("oneOf", wire_text)
        self.assertTrue(spec.json_schema["properties"]["items"]["uniqueItems"])

        with self.assertRaises(ToolContractError):
            tools.decode(
                {
                    "id": "duplicate-call",
                    "name": spec.name,
                    "arguments": {
                        "outcome": "accepted",
                        "items": [{"path": "known.path"}, {"path": "known.path"}],
                    },
                },
                (spec.name,),
            )

    async def test_finish_reason_length_is_technical_incomplete_result(self) -> None:
        runtime = self.runtime(
            ScriptedLlmTransport(
                [
                    replace(
                        completion(finish_reason="length"),
                        response_preview={
                            "content": "partial tool output",
                            "content_truncated": False,
                            "reasoning_content_present": True,
                            "reasoning_content_chars": 3200,
                        },
                    )
                ]
            ),
        )

        with self.assertRaisesRegex(TechnicalLlmError, "оборван"):
            await runtime.invoke("ATTEMPT_LENGTH", "SESSION_0001", self.call)

        diagnostic = self.database.records[("llm_diagnostic", "ATTEMPT_LENGTH")]
        self.assertEqual(diagnostic.payload["finish_reason"], "length")
        self.assertEqual(
            diagnostic.payload["response_preview"]["content"],
            "partial tool output",
        )
        self.assertNotIn("reasoning_content", diagnostic.payload["response_preview"])
        self.assertEqual(
            diagnostic.payload["rejected_tool_calls"],
            list(completion(finish_reason="length").tool_calls),
        )

    async def test_finish_reason_length_is_not_retried_without_prompt_change(self) -> None:
        transport = ScriptedLlmTransport([completion(finish_reason="length")])
        runtime = self.runtime(transport, retries=1)

        with self.assertRaisesRegex(TechnicalLlmError, "оборван"):
            await runtime.invoke("ATTEMPT_LENGTH_ONCE", "SESSION_0001", self.call)

        self.assertEqual(len(transport.calls), 1)
        diagnostic = self.database.records[
            ("llm_diagnostic", "ATTEMPT_LENGTH_ONCE")
        ]
        self.assertEqual(diagnostic.payload["retry"], 0)

    async def test_finish_reason_length_uses_one_declared_extended_profile(
        self,
    ) -> None:
        transport = ScriptedLlmTransport(
            [completion(finish_reason="length"), completion()]
        )
        runtime = self.runtime(transport, retries=1)
        call = replace(self.call, length_retry_max_tokens=8192)

        result = await runtime.invoke(
            "ATTEMPT_LENGTH_EXTENDED",
            "SESSION_0001",
            call,
        )

        self.assertEqual(result.arguments.count, 1)
        self.assertEqual(len(transport.calls), 2)
        first_call = transport.calls[0]["call"]
        second_call = transport.calls[1]["call"]
        self.assertEqual(first_call.generation_parameters["max_tokens"], 4096)
        self.assertEqual(second_call.generation_parameters["max_tokens"], 8192)
        self.assertEqual(second_call.system_prompt, first_call.system_prompt)
        self.assertEqual(second_call.context, first_call.context)
        self.assertEqual(second_call.allowed_tools, first_call.allowed_tools)
        diagnostic = self.database.records[
            ("llm_diagnostic", "ATTEMPT_LENGTH_EXTENDED")
        ]
        self.assertEqual(diagnostic.payload["retry"], 1)
        self.assertEqual(
            [item["finish_reason"] for item in diagnostic.payload["invocations"]],
            ["length", "tool_calls"],
        )
        self.assertEqual(
            [
                item["generation_parameters"]["max_tokens"]
                for item in diagnostic.payload["invocations"]
            ],
            [4096, 8192],
        )
        metrics = collect_metrics(
            lambda: InMemoryUnitOfWork(self.database)
        )
        self.assertEqual(metrics["llm_calls"], 2)
        self.assertEqual(metrics["llm_retries"], 1)
        self.assertEqual(metrics["llm_finish_reason_length"], 1)
        self.assertEqual(metrics["llm_prompt_tokens"], 20)
        self.assertEqual(metrics["llm_completion_tokens"], 10)

    async def test_extended_profile_is_bounded_after_second_length(self) -> None:
        transport = ScriptedLlmTransport(
            [
                completion(finish_reason="length"),
                completion(finish_reason="length"),
                completion(),
            ]
        )
        runtime = self.runtime(transport, retries=2)
        call = replace(self.call, length_retry_max_tokens=8192)

        with self.assertRaisesRegex(TechnicalLlmError, "оборван"):
            await runtime.invoke(
                "ATTEMPT_LENGTH_EXTENDED_FAILED",
                "SESSION_0001",
                call,
            )

        self.assertEqual(len(transport.calls), 2)
        diagnostic = self.database.records[
            ("llm_diagnostic", "ATTEMPT_LENGTH_EXTENDED_FAILED")
        ]
        self.assertEqual(
            [item["finish_reason"] for item in diagnostic.payload["invocations"]],
            ["length", "length"],
        )

    async def test_cancel_during_extended_profile_discards_late_response(
        self,
    ) -> None:
        transport = ScriptedLlmTransport(
            [
                completion(finish_reason="length"),
                (0.05, completion()),
            ]
        )
        runtime = self.runtime(transport, retries=1)
        call = replace(self.call, length_retry_max_tokens=8192)
        pending = asyncio.create_task(
            runtime.invoke(
                "ATTEMPT_LENGTH_EXTENDED_CANCEL",
                "SESSION_0001",
                call,
            )
        )
        while len(transport.calls) < 2:
            await asyncio.sleep(0)
        runtime.cancel("ATTEMPT_LENGTH_EXTENDED_CANCEL")

        with self.assertRaises(AttemptDiscardedError):
            await pending

        self.assertEqual(len(transport.calls), 2)
        with InMemoryUnitOfWork(self.database) as uow:
            attempt = uow.attempts.get("ATTEMPT_LENGTH_EXTENDED_CANCEL")
        self.assertEqual(attempt.status, AttemptStatus.DISCARDED)

    async def test_sqlite_preserves_both_extended_profile_invocations(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            database_path = Path(temporary) / "workbench.sqlite3"
            transport = ScriptedLlmTransport(
                [completion(finish_reason="length"), completion()]
            )
            runtime = LlmToolRuntime(
                transport=transport,
                tools=registry(),
                uow_factory=lambda: SqliteUnitOfWork(database_path),
                max_retries=1,
            )
            call = replace(self.call, length_retry_max_tokens=8192)

            await runtime.invoke(
                "ATTEMPT_LENGTH_EXTENDED_SQLITE",
                "SESSION_0001",
                call,
            )

            with SqliteUnitOfWork(database_path) as uow:
                diagnostic = uow.records.get(
                    "llm_diagnostic",
                    "ATTEMPT_LENGTH_EXTENDED_SQLITE",
                )
            self.assertIsNotNone(diagnostic)
            self.assertEqual(
                [
                    item["generation_parameters"]["max_tokens"]
                    for item in diagnostic.payload["invocations"]
                ],
                [4096, 8192],
            )
            self.assertEqual(
                [
                    item["finish_reason"]
                    for item in diagnostic.payload["invocations"]
                ],
                ["length", "tool_calls"],
            )

    async def test_http_500_and_timeout_follow_bounded_retry_policy(self) -> None:
        transport = ScriptedLlmTransport(
            [
                TechnicalLlmError("HTTP 500", retryable=True),
                TechnicalLlmError("timeout", retryable=True),
                completion(),
            ]
        )
        runtime = self.runtime(transport, retries=2)

        result = await runtime.invoke("ATTEMPT_RETRY", "SESSION_0001", self.call)

        self.assertEqual(result.arguments.count, 1)
        self.assertEqual(len(transport.calls), 3)

    async def test_contract_error_gets_one_bounded_repair_with_feedback(self) -> None:
        rejected = completion(arguments={"outcome": "ok", "count": "one"})
        transport = ScriptedLlmTransport([rejected, completion()])
        runtime = self.runtime(transport, retries=1)

        result = await runtime.invoke("ATTEMPT_CONTRACT", "SESSION_0001", self.call)

        self.assertEqual(result.arguments.count, 1)
        self.assertEqual(len(transport.calls), 2)
        repair_call = transport.calls[1]["call"]
        self.assertIn("отклонён локальной проверкой", repair_call.system_prompt)
        self.assertIn("count", repair_call.system_prompt)
        diagnostic = self.database.records[("llm_diagnostic", "ATTEMPT_CONTRACT")]
        self.assertEqual(diagnostic.payload["retry"], 1)
        self.assertEqual(
            diagnostic.payload["rejected_tool_calls"],
            list(rejected.tool_calls),
        )

    async def test_repeated_contract_error_exhausts_bounded_repair(self) -> None:
        rejected = completion(arguments={"outcome": "ok", "count": "one"})
        transport = ScriptedLlmTransport([rejected, rejected, completion()])
        runtime = self.runtime(transport, retries=1)

        with self.assertRaises(ToolContractError):
            await runtime.invoke("ATTEMPT_CONTRACT_FAILED", "SESSION_0001", self.call)

        self.assertEqual(len(transport.calls), 2)
        diagnostic = self.database.records[
            ("llm_diagnostic", "ATTEMPT_CONTRACT_FAILED")
        ]
        self.assertEqual(diagnostic.payload["retry"], 1)
        self.assertEqual(len(diagnostic.payload["rejected_tool_calls"]), 2)
        with InMemoryUnitOfWork(self.database) as uow:
            self.assertEqual(
                uow.attempts.get("ATTEMPT_CONTRACT_FAILED").status,
                AttemptStatus.FAILED,
            )

    async def test_cancelled_attempt_discards_late_response(self) -> None:
        transport = ScriptedLlmTransport([(0.05, completion())])
        runtime = self.runtime(transport)
        pending = asyncio.create_task(
            runtime.invoke("ATTEMPT_CANCEL", "SESSION_0001", self.call)
        )
        await asyncio.sleep(0.01)
        runtime.cancel("ATTEMPT_CANCEL")

        with self.assertRaises(AttemptDiscardedError):
            await pending

        with InMemoryUnitOfWork(self.database) as uow:
            attempt = uow.attempts.get("ATTEMPT_CANCEL")
            self.assertEqual(attempt.status, AttemptStatus.DISCARDED)

    async def test_cancel_during_context_validation_discards_ready_result(self) -> None:
        runtime = self.runtime(ScriptedLlmTransport([completion()]))

        def cancel_before_ready(_result: object) -> None:
            runtime.cancel("ATTEMPT_CANCEL_DURING_VALIDATION")

        with self.assertRaises(AttemptDiscardedError):
            await runtime.invoke(
                "ATTEMPT_CANCEL_DURING_VALIDATION",
                "SESSION_0001",
                self.call,
                validate_result=cancel_before_ready,
            )

        with InMemoryUnitOfWork(self.database) as uow:
            attempt = uow.attempts.get("ATTEMPT_CANCEL_DURING_VALIDATION")
            events = uow.events.list_for("ATTEMPT_CANCEL_DURING_VALIDATION")
        self.assertEqual(attempt.status, AttemptStatus.DISCARDED)
        self.assertTrue(
            any(item.event_type == "поздний результат отброшен" for item in events)
        )
        self.assertFalse(
            any(
                item.event_type == "результат попытки готов к применению"
                for item in events
            )
        )

    async def test_cancelled_attempt_does_not_start_retry_after_late_error(self) -> None:
        transport = ScriptedLlmTransport(
            [
                (0.05, TechnicalLlmError("HTTP 500", retryable=True)),
                completion(),
            ]
        )
        runtime = self.runtime(transport, retries=1)
        pending = asyncio.create_task(
            runtime.invoke("ATTEMPT_CANCEL_ERROR", "SESSION_0001", self.call)
        )
        await asyncio.sleep(0.01)
        runtime.cancel("ATTEMPT_CANCEL_ERROR")

        with self.assertRaises(AttemptDiscardedError):
            await pending

        self.assertEqual(len(transport.calls), 1)

    async def test_second_response_cannot_complete_same_attempt(self) -> None:
        runtime = self.runtime(ScriptedLlmTransport([]))
        runtime.create_attempt("ATTEMPT_RACE", "SESSION_0001", self.call)

        first = runtime.accept_response("ATTEMPT_RACE", completion())
        with self.assertRaises(AttemptDiscardedError):
            runtime.accept_response("ATTEMPT_RACE", completion())

        self.assertEqual(first.arguments.count, 1)
        with InMemoryUnitOfWork(self.database) as uow:
            self.assertEqual(
                uow.attempts.get("ATTEMPT_RACE").status,
                AttemptStatus.RESULT_READY,
            )

    async def test_cancel_ready_result_prevents_domain_apply(self) -> None:
        runtime = self.runtime(ScriptedLlmTransport([completion()]))
        await runtime.invoke("ATTEMPT_READY", "SESSION_0001", self.call)

        runtime.cancel("ATTEMPT_READY")

        with self.assertRaises(AttemptDiscardedError):
            runtime.begin_apply("ATTEMPT_READY")
        with InMemoryUnitOfWork(self.database) as uow:
            self.assertEqual(
                uow.attempts.get("ATTEMPT_READY").status,
                AttemptStatus.CANCELLED,
            )

    async def test_apply_claim_rejects_late_cancellation(self) -> None:
        runtime = self.runtime(ScriptedLlmTransport([completion()]))
        await runtime.invoke("ATTEMPT_APPLYING", "SESSION_0001", self.call)

        runtime.begin_apply("ATTEMPT_APPLYING")

        with self.assertRaises(AttemptDiscardedError):
            runtime.cancel("ATTEMPT_APPLYING")
        runtime.complete_apply("ATTEMPT_APPLYING")
        with InMemoryUnitOfWork(self.database) as uow:
            self.assertEqual(
                uow.attempts.get("ATTEMPT_APPLYING").status,
                AttemptStatus.COMPLETED,
            )

    async def test_failed_domain_apply_marks_attempt_failed(self) -> None:
        runtime = self.runtime(ScriptedLlmTransport([completion()]))
        await runtime.invoke("ATTEMPT_APPLY_FAILED", "SESSION_0001", self.call)
        runtime.begin_apply("ATTEMPT_APPLY_FAILED")

        runtime.fail_apply("ATTEMPT_APPLY_FAILED", "domain validation failed")

        with InMemoryUnitOfWork(self.database) as uow:
            self.assertEqual(
                uow.attempts.get("ATTEMPT_APPLY_FAILED").status,
                AttemptStatus.FAILED,
            )

    async def test_atomic_apply_rolls_back_domain_writes_before_marking_failed(self) -> None:
        runtime = self.runtime(ScriptedLlmTransport([completion()]))
        await runtime.invoke("ATTEMPT_ATOMIC_FAILED", "SESSION_0001", self.call)

        def fail_after_write(uow: UnitOfWork) -> None:
            uow.records.save(
                StoredRecord("test_result", "RESULT_0001", {"applied": True})
            )
            raise RuntimeError("domain write failed")

        with self.assertRaisesRegex(RuntimeError, "domain write failed"):
            runtime.apply_result("ATTEMPT_ATOMIC_FAILED", fail_after_write)

        with InMemoryUnitOfWork(self.database) as uow:
            self.assertIsNone(uow.records.get("test_result", "RESULT_0001"))
            self.assertEqual(
                uow.attempts.get("ATTEMPT_ATOMIC_FAILED").status,
                AttemptStatus.FAILED,
            )

    async def test_atomic_apply_commits_domain_write_and_completion_together(self) -> None:
        runtime = self.runtime(ScriptedLlmTransport([completion()]))
        await runtime.invoke("ATTEMPT_ATOMIC_OK", "SESSION_0001", self.call)

        def save_result(uow: UnitOfWork) -> str:
            uow.records.save(
                StoredRecord("test_result", "RESULT_0002", {"applied": True})
            )
            return "saved"

        result = runtime.apply_result("ATTEMPT_ATOMIC_OK", save_result)

        self.assertEqual(result, "saved")
        with InMemoryUnitOfWork(self.database) as uow:
            self.assertIsNotNone(uow.records.get("test_result", "RESULT_0002"))
            self.assertEqual(
                uow.attempts.get("ATTEMPT_ATOMIC_OK").status,
                AttemptStatus.COMPLETED,
            )

    async def test_sqlite_cancel_after_atomic_claim_cannot_override_apply(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            database_path = Path(temporary) / "workbench.sqlite3"
            runtime = LlmToolRuntime(
                transport=ScriptedLlmTransport([completion()]),
                tools=registry(),
                uow_factory=lambda: SqliteUnitOfWork(database_path),
            )
            await runtime.invoke("ATTEMPT_SQLITE_RACE", "SESSION_0001", self.call)
            applying = threading.Event()
            release = threading.Event()
            apply_errors: list[BaseException] = []
            cancel_errors: list[BaseException] = []

            def operation(uow: UnitOfWork) -> str:
                uow.records.save(
                    StoredRecord("test_result", "RESULT_SQLITE", {"applied": True})
                )
                applying.set()
                if not release.wait(1):
                    raise TimeoutError("Тест не освободил domain apply")
                return "saved"

            def apply() -> None:
                try:
                    runtime.apply_result("ATTEMPT_SQLITE_RACE", operation)
                except BaseException as error:
                    apply_errors.append(error)

            def cancel() -> None:
                try:
                    runtime.cancel("ATTEMPT_SQLITE_RACE")
                except BaseException as error:
                    cancel_errors.append(error)

            apply_thread = threading.Thread(target=apply)
            apply_thread.start()
            self.assertTrue(applying.wait(1))
            cancel_thread = threading.Thread(target=cancel)
            cancel_thread.start()
            sleep(0.05)
            release.set()
            apply_thread.join(1)
            cancel_thread.join(1)

            self.assertFalse(apply_thread.is_alive())
            self.assertFalse(cancel_thread.is_alive())
            self.assertFalse(apply_errors)
            self.assertEqual(len(cancel_errors), 1)
            self.assertIsInstance(cancel_errors[0], AttemptDiscardedError)
            with SqliteUnitOfWork(database_path) as uow:
                self.assertIsNotNone(uow.records.get("test_result", "RESULT_SQLITE"))
                self.assertEqual(
                    uow.attempts.get("ATTEMPT_SQLITE_RACE").status,
                    AttemptStatus.COMPLETED,
                )

    async def test_diagnostics_do_not_store_api_key_or_extra_headers(self) -> None:
        secret = "do-not-store-this-key"
        transport = ScriptedLlmTransport(
            [completion()],
            metadata={
                "model": "test-model",
                "base_url": "https://vllm.example/v1",
                "api_key": secret,
                "Authorization": f"Bearer {secret}",
                "X-Debug": secret,
            },
        )
        runtime = self.runtime(transport)

        await runtime.invoke("ATTEMPT_SECRET", "SESSION_0001", self.call)

        payload = self.database.records[("llm_diagnostic", "ATTEMPT_SECRET")].payload
        self.assertNotIn(secret, str(payload))
        self.assertEqual(payload["transport"], {
            "model": "test-model",
            "base_url": "https://vllm.example/v1",
        })

    async def test_fake_transport_reproduces_delay_error_and_late_response(self) -> None:
        transport = ScriptedLlmTransport(
            [
                (0.001, TechnicalLlmError("temporary", retryable=True)),
                (0.001, completion()),
            ]
        )

        result = await self.runtime(transport, retries=1).invoke(
            "ATTEMPT_FAKE",
            "SESSION_0001",
            self.call,
        )

        self.assertEqual(result.arguments.outcome, "ok")
        self.assertEqual(len(transport.calls), 2)


class OpenAITransportTests(unittest.TestCase):
    @patch("pmi_generator.workbench.infrastructure.llm.openai.build_url_opener")
    def test_generation_parameters_reach_vllm_request(
        self,
        build_opener: Mock,
    ) -> None:
        response = Mock()
        response.read.return_value = json.dumps(
            {
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "tool_calls": [
                                {
                                    "id": "call-1",
                                    "function": {
                                        "name": "ask_lightrag",
                                        "arguments": {"question": "Какое значение?"},
                                    },
                                }
                            ]
                        },
                    }
                ],
                "usage": {},
                "model": "test-model",
            }
        ).encode()
        opener = MagicMock()
        opener.open.return_value.__enter__.return_value = response
        build_opener.return_value = opener
        transport = OpenAICompatibleTransport(
            OpenAITransportSettings(
                base_url="https://vllm.example/v1",
                model="test-model",
            )
        )
        policy = default_policy()
        spec = policy.prompts[PromptId.GAP_RESEARCH]
        call = policy.build_call(
            PromptId.GAP_RESEARCH,
            {key: {} for key in spec.allowed_context},
        )

        completion_result = transport._complete_sync(
            call,
            [
                {
                    "type": "function",
                    "function": {
                        "name": "ask_lightrag",
                        "parameters": {"type": "object"},
                    },
                }
            ],
        )

        request = opener.open.call_args.args[0]
        payload = json.loads(request.data.decode())
        self.assertEqual(payload["max_tokens"], 1024)
        self.assertIs(payload["parallel_tool_calls"], False)
        self.assertEqual(payload["chat_template_kwargs"], {"enable_thinking": False})
        self.assertEqual(completion_result.response_preview, {})

    def test_decode_bounds_content_and_does_not_store_reasoning_text(self) -> None:
        completion_result = OpenAICompatibleTransport._decode(
            {
                "choices": [
                    {
                        "finish_reason": "length",
                        "message": {
                            "content": "x" * 3000,
                            "reasoning_content": "private chain",
                        },
                    }
                ],
                "usage": {"completion_tokens": 1024},
                "model": "test-model",
            }
        )

        preview = completion_result.response_preview
        self.assertEqual(len(preview["content"]), 2000)
        self.assertIs(preview["content_truncated"], True)
        self.assertIs(preview["reasoning_content_present"], True)
        self.assertEqual(preview["reasoning_content_chars"], len("private chain"))
        self.assertNotIn("private chain", str(preview))

    def test_decode_bounds_partial_tool_arguments_for_length_diagnostic(
        self,
    ) -> None:
        completion_result = OpenAICompatibleTransport._decode(
            {
                "choices": [
                    {
                        "finish_reason": "length",
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call-partial",
                                    "function": {
                                        "name": (
                                            "submit_semantic_window_result"
                                        ),
                                        "arguments": "x" * 3000,
                                    },
                                }
                            ],
                        },
                    }
                ],
                "usage": {"completion_tokens": 4096},
                "model": "test-model",
            }
        )

        preview = completion_result.response_preview
        self.assertEqual(preview["tool_calls_count"], 1)
        self.assertEqual(
            preview["tool_calls"][0]["name"],
            "submit_semantic_window_result",
        )
        self.assertEqual(
            len(preview["tool_calls"][0]["arguments"]),
            2000,
        )
        self.assertIs(
            preview["tool_calls"][0]["arguments_truncated"],
            True,
        )

    @patch("pmi_generator.workbench.infrastructure.llm.openai.build_url_opener")
    def test_no_proxy_uses_shared_http_transport_factory(self, build_opener: Mock) -> None:
        opener = Mock()
        build_opener.return_value = opener

        transport = OpenAICompatibleTransport(
            OpenAITransportSettings(
                base_url="https://vllm.example/v1",
                model="test-model",
                verify_ssl=True,
                no_proxy=True,
            )
        )

        self.assertIs(transport.opener, opener)
        build_opener.assert_called_once_with(
            verify_ssl=True,
            ca_file=None,
            no_proxy=True,
        )

    @patch("pmi_generator.workbench.infrastructure.llm.openai.build_url_opener")
    def test_proxy_environment_is_preserved_when_bypass_is_disabled(
        self,
        build_opener: Mock,
    ) -> None:
        OpenAICompatibleTransport(
            OpenAITransportSettings(
                base_url="https://vllm.example/v1",
                model="test-model",
                no_proxy=False,
            )
        )

        build_opener.assert_called_once_with(
            verify_ssl=True,
            ca_file=None,
            no_proxy=False,
        )


if __name__ == "__main__":
    unittest.main()
