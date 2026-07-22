from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import UTC, datetime
from typing import Callable, TypeAlias, TypeVar

from ..prompting import PromptCall
from ..repositories import UnitOfWork
from ..state import AttemptRecord, AttemptStatus, StoredRecord
from .errors import (
    AttemptDiscardedError,
    GenerationLengthError,
    TechnicalLlmError,
    ToolContractError,
)
from .models import DecodedToolCall, RawCompletion
from .ports import LlmTransport
from .tools import TypedToolRegistry


ResultT = TypeVar("ResultT")
RepairPromptBuilder: TypeAlias = Callable[
    [PromptCall, str, tuple[dict[str, object], ...]],
    PromptCall,
]


class LlmToolRuntime:
    def __init__(
        self,
        *,
        transport: LlmTransport,
        tools: TypedToolRegistry,
        uow_factory: Callable[[], UnitOfWork],
        max_retries: int = 1,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.transport = transport
        self.tools = tools
        self.uow_factory = uow_factory
        self.max_retries = max(0, max_retries)
        self.clock = clock or (lambda: datetime.now(UTC))

    async def invoke(
        self,
        attempt_id: str,
        session_id: str,
        call: PromptCall,
        *,
        validate_result: Callable[[DecodedToolCall], None] | None = None,
        repair_prompt_builder: RepairPromptBuilder | None = None,
    ) -> DecodedToolCall:
        schemas = self.tools.openai_schemas(
            call.allowed_tools,
            context=call.context,
        )
        self.create_attempt(attempt_id, session_id, call)
        failures: list[str] = []
        rejected_tool_calls: list[dict[str, object]] = []
        request_call = call
        length_retry_used = False
        for retry in range(self.max_retries + 1):
            response: RawCompletion | None = None
            try:
                response = await self.transport.complete(request_call, schemas)
                result = self.accept_response(
                    attempt_id,
                    response,
                    validate_result=validate_result,
                    schema_context=call.context,
                    request_call=request_call,
                    retry=retry,
                )
                return result
            except TechnicalLlmError as error:
                failures.append(str(error))
                if response is not None:
                    rejected_tool_calls.extend(response.tool_calls)
                self._save_diagnostic(
                    attempt_id,
                    request_call,
                    finish_reason=getattr(error, "finish_reason", None),
                    errors=failures,
                    retry=retry,
                    raw_tool_calls=tuple(rejected_tool_calls),
                    response=response,
                )
                if not self._is_active(attempt_id):
                    raise AttemptDiscardedError(
                        f"Результат attempt {attempt_id} больше не является актуальным"
                    ) from error
                if (
                    getattr(error, "finish_reason", None) == "length"
                    and not length_retry_used
                    and retry < self.max_retries
                    and self._can_use_length_retry(request_call)
                ):
                    request_call = self._length_retry_call(request_call)
                    length_retry_used = True
                    self._record_length_retry(attempt_id, request_call)
                    continue
                if error.retryable and retry < self.max_retries:
                    continue
                self._fail_attempt(attempt_id, str(error))
                raise
            except ToolContractError as error:
                failures.append(str(error))
                if response is not None:
                    rejected_tool_calls.extend(response.tool_calls)
                self._save_diagnostic(
                    attempt_id,
                    request_call,
                    errors=failures,
                    retry=retry,
                    raw_tool_calls=tuple(rejected_tool_calls),
                    response=response,
                )
                if not self._is_active(attempt_id):
                    raise AttemptDiscardedError(
                        f"Результат attempt {attempt_id} больше не является актуальным"
                    ) from error
                if retry < self.max_retries:
                    request_call = (
                        repair_prompt_builder(
                            request_call,
                            str(error),
                            tuple(rejected_tool_calls),
                        )
                        if repair_prompt_builder is not None
                        else replace(
                            request_call,
                            system_prompt=_contract_repair_prompt(
                                request_call.system_prompt,
                                str(error),
                            ),
                        )
                    )
                    continue
                self._fail_attempt(attempt_id, str(error))
                raise
            except asyncio.CancelledError:
                self._cancel_if_active(attempt_id)
                raise
        raise AssertionError("Недостижимое состояние retry loop")

    def create_attempt(self, attempt_id: str, session_id: str, call: PromptCall) -> None:
        now = self.clock()
        created = AttemptRecord(
            attempt_id=attempt_id,
            session_id=session_id,
            stage=call.prompt_id.value,
            status=AttemptStatus.CREATED,
            payload={
                "policy_version": call.policy_version,
                "prompt_version": call.prompt_version,
                "fingerprint": call.fingerprint,
                "allowed_tools": list(call.allowed_tools),
                "length_retry_max_tokens": call.length_retry_max_tokens,
            },
            updated_at=now,
        )
        with self.uow_factory() as uow:
            if uow.attempts.get(attempt_id) is not None:
                raise ToolContractError(f"Attempt {attempt_id} уже существует")
            uow.attempts.save(created)
            uow.events.append(attempt_id, "попытка создана", {"stage": call.prompt_id.value})
            uow.attempts.save(created.with_status(AttemptStatus.ACTIVE, now))
            uow.events.append(attempt_id, "попытка выполняется", {})
        self._save_diagnostic(attempt_id, call, errors=[], retry=0)

    def cancel(self, attempt_id: str) -> None:
        with self.uow_factory() as uow:
            attempt = uow.attempts.get(attempt_id)
            if attempt is None or attempt.status not in {
                AttemptStatus.ACTIVE,
                AttemptStatus.RESULT_READY,
            }:
                raise AttemptDiscardedError(f"Attempt {attempt_id} не является активным")
            uow.attempts.save(attempt.with_status(AttemptStatus.CANCELLED, self.clock()))
            uow.events.append(attempt_id, "попытка отменена", {})

    def begin_apply(self, attempt_id: str) -> None:
        with self.uow_factory() as uow:
            attempt = uow.attempts.get(attempt_id)
            if attempt is None or attempt.status is not AttemptStatus.RESULT_READY:
                raise AttemptDiscardedError(
                    f"Результат attempt {attempt_id} больше нельзя применить"
                )
            uow.attempts.save(
                attempt.with_status(AttemptStatus.APPLYING, self.clock())
            )
            uow.events.append(attempt_id, "применение результата начато", {})

    def complete_apply(self, attempt_id: str) -> None:
        with self.uow_factory() as uow:
            attempt = uow.attempts.get(attempt_id)
            if attempt is None or attempt.status is not AttemptStatus.APPLYING:
                raise AttemptDiscardedError(
                    f"Attempt {attempt_id} не находится в стадии применения"
                )
            uow.attempts.save(
                attempt.with_status(AttemptStatus.COMPLETED, self.clock())
            )
            uow.events.append(attempt_id, "попытка завершена", {})

    def fail_apply(self, attempt_id: str, message: str) -> None:
        with self.uow_factory() as uow:
            attempt = uow.attempts.get(attempt_id)
            if attempt is None or attempt.status is not AttemptStatus.APPLYING:
                raise AttemptDiscardedError(
                    f"Attempt {attempt_id} не находится в стадии применения"
                )
            uow.attempts.save(attempt.with_status(AttemptStatus.FAILED, self.clock()))
            uow.events.append(
                attempt_id,
                "применение результата завершилось ошибкой",
                {"error": message},
            )

    def apply_result(
        self,
        attempt_id: str,
        operation: Callable[[UnitOfWork], ResultT],
    ) -> ResultT:
        try:
            with self.uow_factory() as uow:
                attempt = uow.attempts.get(attempt_id)
                if attempt is None or attempt.status is not AttemptStatus.RESULT_READY:
                    raise AttemptDiscardedError(
                        f"Результат attempt {attempt_id} больше нельзя применить"
                    )
                applying = attempt.with_status(AttemptStatus.APPLYING, self.clock())
                uow.attempts.save(applying)
                uow.events.append(attempt_id, "применение результата начато", {})
                result = operation(uow)
                uow.attempts.save(
                    applying.with_status(AttemptStatus.COMPLETED, self.clock())
                )
                uow.events.append(attempt_id, "попытка завершена", {})
                return result
        except AttemptDiscardedError:
            raise
        except Exception as error:
            self._fail_atomic_apply(attempt_id, str(error))
            raise

    def _fail_atomic_apply(self, attempt_id: str, message: str) -> None:
        with self.uow_factory() as uow:
            attempt = uow.attempts.get(attempt_id)
            if attempt is None or attempt.status not in {
                AttemptStatus.RESULT_READY,
                AttemptStatus.APPLYING,
            }:
                return
            uow.attempts.save(attempt.with_status(AttemptStatus.FAILED, self.clock()))
            uow.events.append(
                attempt_id,
                "применение результата завершилось ошибкой",
                {"error": message},
            )

    def accept_response(
        self,
        attempt_id: str,
        response: RawCompletion,
        *,
        validate_result: Callable[[DecodedToolCall], None] | None = None,
        schema_context: dict[str, object] | None = None,
        request_call: PromptCall | None = None,
        retry: int = 0,
    ) -> DecodedToolCall:
        allowed_tools: tuple[str, ...] = ()
        discarded_message: str | None = None
        with self.uow_factory() as uow:
            attempt = uow.attempts.get(attempt_id)
            if attempt is None:
                raise AttemptDiscardedError(f"Attempt {attempt_id} не найден")
            if attempt.status is not AttemptStatus.ACTIVE:
                if attempt.status is AttemptStatus.CANCELLED:
                    uow.attempts.save(
                        attempt.with_status(AttemptStatus.DISCARDED, self.clock())
                    )
                uow.events.append(
                    attempt_id,
                    "поздний результат отброшен",
                    {"previous_status": attempt.status.value},
                )
                discarded_message = (
                    f"Результат attempt {attempt_id} больше не является актуальным"
                )
            elif response.finish_reason == "length":
                raise GenerationLengthError()
            elif response.finish_reason != "tool_calls" or len(response.tool_calls) != 1:
                raise ToolContractError("LLM-вызов должен вернуть ровно один завершённый tool call")
            else:
                allowed_tools = tuple(attempt.payload.get("allowed_tools", ()))
                if not allowed_tools:
                    diagnostic = uow.records.get("llm_diagnostic", attempt_id)
                    allowed_tools = tuple(diagnostic.payload.get("allowed_tools", ()))

        if discarded_message is not None:
            raise AttemptDiscardedError(discarded_message)

        decoded = self.tools.decode(
            response.tool_calls[0],
            allowed_tools,
            context=schema_context,
        )
        if validate_result is not None:
            validate_result(decoded)

        with self.uow_factory() as uow:
            attempt = uow.attempts.get(attempt_id)
            if attempt is None or attempt.status is not AttemptStatus.ACTIVE:
                if attempt is not None:
                    if attempt.status is AttemptStatus.CANCELLED:
                        uow.attempts.save(
                            attempt.with_status(AttemptStatus.DISCARDED, self.clock())
                        )
                    uow.events.append(
                        attempt_id,
                        "поздний результат отброшен",
                        {"previous_status": attempt.status.value},
                    )
                discarded_message = (
                    f"Результат attempt {attempt_id} больше не является актуальным"
                )
            else:
                uow.attempts.save(
                    attempt.with_status(AttemptStatus.RESULT_READY, self.clock())
                )
                uow.events.append(
                    attempt_id,
                    "результат попытки готов к применению",
                    {"tool": decoded.name, "finish_reason": response.finish_reason},
                )
                diagnostic = uow.records.get("llm_diagnostic", attempt_id)
                payload = dict(diagnostic.payload if diagnostic else {})
                generation_parameters = (
                    dict(request_call.generation_parameters)
                    if request_call is not None
                    else dict(payload.get("generation_parameters", {}))
                )
                invocations = list(payload.get("invocations", []))
                invocations.append(
                    self._invocation_diagnostic(
                        generation_parameters=generation_parameters,
                        finish_reason=response.finish_reason,
                        retry=retry,
                        response=response,
                    )
                )
                payload.update(
                    {
                        "finish_reason": response.finish_reason,
                        "tool_calls": [
                            {
                                "id": decoded.call_id,
                                "name": decoded.name,
                                "arguments": decoded.raw_arguments,
                            }
                        ],
                        "usage": response.usage,
                        "response_model": response.model,
                        "response_preview": response.response_preview,
                        "generation_parameters": generation_parameters,
                        "retry": retry,
                        "invocations": invocations,
                    }
                )
                uow.records.save(StoredRecord("llm_diagnostic", attempt_id, payload))
        if discarded_message is not None:
            raise AttemptDiscardedError(discarded_message)
        return decoded

    def _fail_attempt(self, attempt_id: str, message: str) -> None:
        with self.uow_factory() as uow:
            attempt = uow.attempts.get(attempt_id)
            if attempt is not None and attempt.status is AttemptStatus.ACTIVE:
                uow.attempts.save(attempt.with_status(AttemptStatus.FAILED, self.clock()))
                uow.events.append(attempt_id, "попытка завершилась ошибкой", {"error": message})

    def _cancel_if_active(self, attempt_id: str) -> None:
        with self.uow_factory() as uow:
            attempt = uow.attempts.get(attempt_id)
            if attempt is not None and attempt.status is AttemptStatus.ACTIVE:
                uow.attempts.save(attempt.with_status(AttemptStatus.CANCELLED, self.clock()))
                uow.events.append(attempt_id, "попытка отменена", {})

    def _is_active(self, attempt_id: str) -> bool:
        with self.uow_factory() as uow:
            attempt = uow.attempts.get(attempt_id)
        return attempt is not None and attempt.status is AttemptStatus.ACTIVE

    def _record_length_retry(
        self,
        attempt_id: str,
        request_call: PromptCall,
    ) -> None:
        with self.uow_factory() as uow:
            uow.events.append(
                attempt_id,
                "расширенный generation profile применён",
                {
                    "max_tokens": request_call.generation_parameters[
                        "max_tokens"
                    ],
                },
            )

    @staticmethod
    def _can_use_length_retry(call: PromptCall) -> bool:
        retry_max_tokens = call.length_retry_max_tokens
        current_max_tokens = call.generation_parameters.get("max_tokens")
        return (
            isinstance(retry_max_tokens, int)
            and not isinstance(retry_max_tokens, bool)
            and isinstance(current_max_tokens, int)
            and not isinstance(current_max_tokens, bool)
            and retry_max_tokens > current_max_tokens
        )

    @staticmethod
    def _length_retry_call(call: PromptCall) -> PromptCall:
        retry_max_tokens = call.length_retry_max_tokens
        assert retry_max_tokens is not None
        return replace(
            call,
            generation_parameters={
                **call.generation_parameters,
                "max_tokens": retry_max_tokens,
            },
        )

    def _save_diagnostic(
        self,
        attempt_id: str,
        call: PromptCall,
        *,
        finish_reason: str | None = None,
        errors: list[str],
        retry: int,
        raw_tool_calls: tuple[dict[str, object], ...] = (),
        response: RawCompletion | None = None,
    ) -> None:
        metadata = self.transport.public_metadata()
        public_transport = {
            key: metadata[key]
            for key in ("model", "base_url")
            if key in metadata
        }
        payload = {
            "attempt_id": attempt_id,
            "prompt_id": call.prompt_id.value,
            "policy_version": call.policy_version,
            "prompt_version": call.prompt_version,
            "fingerprint": call.fingerprint,
            "rule_ids": list(call.rule_ids),
            "allowed_tools": list(call.allowed_tools),
            "generation_parameters": call.generation_parameters,
            "length_retry_max_tokens": call.length_retry_max_tokens,
            "context": call.context,
            "transport": public_transport,
            "finish_reason": finish_reason,
            "errors": list(errors),
            "retry": retry,
            "rejected_tool_calls": [dict(item) for item in raw_tool_calls],
        }
        with self.uow_factory() as uow:
            previous = uow.records.get("llm_diagnostic", attempt_id)
        invocations = list(
            previous.payload.get("invocations", [])
            if previous is not None
            else []
        )
        if response is not None or errors:
            effective_finish_reason = (
                finish_reason
                if finish_reason is not None
                else response.finish_reason
                if response is not None
                else None
            )
            invocations.append(
                self._invocation_diagnostic(
                    generation_parameters=dict(call.generation_parameters),
                    finish_reason=effective_finish_reason,
                    retry=retry,
                    response=response,
                    error=errors[-1] if errors else None,
                )
            )
        payload["invocations"] = invocations
        if response is not None:
            payload["usage"] = response.usage
            payload["response_model"] = response.model
            payload["response_preview"] = response.response_preview
        with self.uow_factory() as uow:
            uow.records.save(StoredRecord("llm_diagnostic", attempt_id, payload))

    @staticmethod
    def _invocation_diagnostic(
        *,
        generation_parameters: dict[str, object],
        finish_reason: str | None,
        retry: int,
        response: RawCompletion | None,
        error: str | None = None,
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            "retry": retry,
            "generation_parameters": generation_parameters,
            "finish_reason": finish_reason,
        }
        if error is not None:
            payload["error"] = error
        if response is not None:
            payload.update(
                {
                    "usage": response.usage,
                    "response_model": response.model,
                    "response_preview": response.response_preview,
                }
            )
        return payload


def _contract_repair_prompt(system_prompt: str, error: str) -> str:
    error_detail = error[:1200]
    return (
        f"{system_prompt}\n\n"
        "Предыдущий tool call отклонён локальной проверкой:\n"
        f"{error_detail}\n"
        "Верни заново ровно один полный tool call. Заполни все required-поля, "
        "не повторяй отклонённые аргументы и не возвращай Markdown."
    )
