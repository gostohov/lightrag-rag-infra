from __future__ import annotations

import re
from datetime import UTC, datetime
from time import monotonic
from typing import Callable

from ...domain import Evidence, GapResolutionMode, TestCard
from ..llm import AttemptDiscardedError, LlmToolRuntime, ToolContractError
from ..prompting import PromptId, PromptPolicy
from ..repositories import UnitOfWork
from ..session import SessionEventKind, SessionService
from ..state import AttemptRecord, AttemptStatus, StoredRecord
from .budgets import RetrievalBudgetPolicy
from .models import (
    GapArguments,
    GapInvestigationError,
    GapInvestigationResult,
    RetrievalCall,
    RetrievalObservation,
)
from .ports import GapAgentPort, GapAgentStepLimitError, RetrievalPort
from .service import GapInvestigationService


class GapInvestigationFlow:
    def __init__(
        self,
        *,
        policy: PromptPolicy,
        runtime: LlmToolRuntime,
        agent: GapAgentPort,
        retrieval: RetrievalPort,
        budgets: RetrievalBudgetPolicy,
        service: GapInvestigationService,
        sessions: SessionService,
        uow_factory: Callable[[], UnitOfWork],
        next_id: Callable[[str], str],
        clock: Callable[[], datetime] | None = None,
        max_steps: int = 8,
    ) -> None:
        self.policy = policy
        self.runtime = runtime
        self.agent = agent
        self.retrieval = retrieval
        self.budgets = budgets
        self.service = service
        self.sessions = sessions
        self.uow_factory = uow_factory
        self.next_id = next_id
        self.clock = clock or (lambda: datetime.now(UTC))
        self.max_steps = max(1, max_steps)
        self._active_children: dict[str, str | None] = {}

    async def run(
        self,
        *,
        attempt_id: str,
        session_id: str,
        card_id: str,
        gap_id: str,
        selection: dict[str, object],
        research_question: str | None = None,
        research_message_id: str | None = None,
    ) -> GapInvestigationResult:
        card = self._load_card(card_id)
        if gap_id not in card.gaps:
            raise GapInvestigationError(f"Пробел {gap_id} не найден")
        if card.gaps[gap_id].resolution_mode is not GapResolutionMode.SOURCE_FACT:
            raise GapInvestigationError(
                "LightRAG доступен только для пробела типа source_fact"
            )
        normalized_research_question = (
            research_question.strip() if research_question else ""
        )
        initial_question = (
            normalized_research_question or card.gaps[gap_id].question.strip()
        )
        self._start_attempt(attempt_id, session_id, card_id, gap_id)
        self._active_children[attempt_id] = None
        observations: list[RetrievalObservation] = []
        evidence: dict[str, Evidence] = dict(card.evidence)
        retrieved_evidence: dict[str, Evidence] = {}
        calls: dict[str, RetrievalCall] = {}
        seen: set[tuple[str, str]] = set()
        submitted_arguments: GapArguments | None = None

        def call_factory():
            self._ensure_active(attempt_id)
            current_card = self._load_card(card_id)
            gap = current_card.gaps[gap_id]
            if observations:
                latest = observations[-1]
                if latest.profile_name == self.budgets.broad.name:
                    allowed_tools = (
                        ("ask_lightrag", "submit_gap_result")
                        if latest.evidence_ids
                        else ("submit_gap_result",)
                    )
                elif (
                    latest.profile_name == self.budgets.narrow.name
                    and latest.evidence_ids
                ):
                    allowed_tools = (
                        "ask_lightrag",
                        "expand_lightrag",
                        "submit_gap_result",
                    )
                elif latest.profile_name == self.budgets.narrow.name:
                    allowed_tools = (
                        "expand_lightrag",
                        "submit_gap_result",
                    )
                else:
                    allowed_tools = ("submit_gap_result",)
            else:
                allowed_tools = ("ask_lightrag",)
            return self.policy.build_call(
                PromptId.GAP_RESEARCH,
                {
                    "selection": selection,
                    "card": self._card_context(current_card),
                    "gap": self._gap_context(gap),
                    "evidence": [
                        self._evidence_context(item) for item in evidence.values()
                    ],
                    "research_question": (
                        {"text": normalized_research_question}
                        if normalized_research_question
                        else None
                    ),
                    "observations": [item.as_context() for item in observations],
                },
                allowed_tools=allowed_tools,
            )

        async def ask(call_id: str, question: str) -> object:
            outbound_question = initial_question if not calls else question.strip()
            retrieval_call = RetrievalCall(
                call_id,
                outbound_question,
                self.budgets.narrow,
            )
            self._check_duplicate(retrieval_call, seen)
            calls[call_id] = retrieval_call
            observation, found = await self._retrieve(
                attempt_id,
                session_id,
                self._load_card(card_id),
                retrieval_call,
            )
            observations.append(observation)
            found_by_id = {item.evidence_id: item for item in found}
            evidence.update(found_by_id)
            retrieved_evidence.update(found_by_id)
            return observation.as_context()

        async def expand(call_id: str, source_call_id: str, reason: str) -> object:
            original = calls.get(source_call_id)
            if original is None:
                raise GapInvestigationError("Расширение ссылается на неизвестный вызов")
            retrieval_call = RetrievalCall(
                call_id,
                original.question,
                self.budgets.broad,
                reason,
            )
            self._check_duplicate(retrieval_call, seen)
            calls[call_id] = retrieval_call
            observation, found = await self._retrieve(
                attempt_id,
                session_id,
                self._load_card(card_id),
                retrieval_call,
            )
            observations.append(observation)
            found_by_id = {item.evidence_id: item for item in found}
            evidence.update(found_by_id)
            retrieved_evidence.update(found_by_id)
            return observation.as_context()

        def submit(arguments: GapArguments) -> object:
            nonlocal submitted_arguments
            result = self._apply_result(
                attempt_id,
                lambda uow: self.service.apply(
                    card_id,
                    gap_id,
                    arguments,
                    available_evidence=tuple(retrieved_evidence.values()),
                    analyst_messages=(),
                    uow=uow,
                ),
            )
            submitted_arguments = arguments
            return result

        def validate(arguments: GapArguments) -> None:
            try:
                self.service.validate_submission(
                    card_id,
                    gap_id,
                    arguments,
                    available_evidence=tuple(retrieved_evidence.values()),
                    analyst_messages=(),
                )
            except GapInvestigationError as error:
                raise ToolContractError(str(error)) from error

        try:
            result = await self.agent.run(
                attempt_id=attempt_id,
                session_id=session_id,
                call_factory=call_factory,
                ask_lightrag=ask,
                expand_lightrag=expand,
                validate_result=validate,
                submit_result=submit,
                child_started=lambda child_id: self._active_children.__setitem__(
                    attempt_id, child_id
                ),
                child_finished=lambda: self._active_children.__setitem__(
                    attempt_id, None
                ),
                max_steps=self.max_steps,
            )
            if not isinstance(result, GapInvestigationResult):
                raise GapInvestigationError("LangChain worker не вернул результат Промпта 3")
            kind, text, metadata = self._completion_event(
                card,
                gap_id,
                result,
                submitted_arguments,
            )
            self.sessions.append(
                session_id,
                kind,
                text,
                {
                    "attempt_id": attempt_id,
                    "gap_id": gap_id,
                    **metadata,
                },
            )
            return GapInvestigationResult(
                result.card_id,
                result.gap_id,
                result.outcome,
                result.revision,
                len(observations),
            )
        except GapAgentStepLimitError as error:
            current_card = self._load_card(card_id)
            revision = current_card.revision
            result = self._apply_result(
                attempt_id,
                lambda _uow: GapInvestigationResult(
                    card_id,
                    gap_id,
                    "technical_limit",
                    revision,
                    len(observations),
                ),
            )
            self.sessions.append(
                session_id,
                SessionEventKind.WORKBENCH,
                (
                    "Исследование остановлено: достигнут технический предел.\n\n"
                    f"Причина: {error}\n"
                    f"Вопрос: {current_card.gaps[gap_id].question}\n"
                    "Используйте /continue для новой попытки."
                ),
                {
                    "attempt_id": attempt_id,
                    "gap_id": gap_id,
                    "outcome": "technical_limit",
                    "technical_reason": str(error),
                },
            )
            return result
        except AttemptDiscardedError:
            raise
        except Exception as error:
            self._fail_attempt(attempt_id, str(error))
            self.sessions.append(session_id, SessionEventKind.ERROR, f"Исследование не завершено: {error}", {"attempt_id": attempt_id})
            raise
        finally:
            self._active_children.pop(attempt_id, None)

    def cancel(self, attempt_id: str) -> None:
        child = self._active_children.get(attempt_id)
        if child:
            try:
                self.runtime.cancel(child)
            except AttemptDiscardedError:
                pass
        with self.uow_factory() as uow:
            attempt = uow.attempts.get(attempt_id)
            if attempt is None or attempt.status is not AttemptStatus.ACTIVE:
                raise AttemptDiscardedError("Исследование уже не активно")
            uow.attempts.save(attempt.with_status(AttemptStatus.CANCELLED, self.clock()))
            uow.events.append(attempt_id, "исследование отменено", {})

    async def _retrieve(self, attempt_id: str, session_id: str, card: TestCard, call: RetrievalCall):
        self.sessions.append(
            session_id,
            SessionEventKind.OPERATION,
            f"LightRAG: {call.question}\nПрофиль: {call.profile.name}\nСтатус: выполняется",
            {"attempt_id": attempt_id, "call_id": call.call_id},
        )
        started = monotonic()
        response = await self.retrieval.query(call.question, call.profile)
        duration = monotonic() - started
        self._ensure_active(attempt_id)
        found: list[Evidence] = []
        for fragment in response.fragments:
            if fragment.is_exact:
                found.append(
                    fragment.to_evidence(
                        self.next_id("EVIDENCE"), card.card_id, card.selection_id, self.clock()
                    )
                )
        observation = RetrievalObservation(
            call.call_id,
            call.question,
            call.profile.name,
            response.answer,
            tuple(item.evidence_id for item in found),
            duration,
        )
        with self.uow_factory() as uow:
            uow.records.save(
                StoredRecord(
                    "retrieval_observation",
                    f"{attempt_id}:{call.call_id}",
                    {
                        **observation.as_context(),
                        "duration_seconds": duration,
                        "parameters": {
                            "kg_top_k": call.profile.kg_top_k,
                            "chunk_top_k": call.profile.chunk_top_k,
                            "max_entity_tokens": call.profile.max_entity_tokens,
                            "max_relation_tokens": call.profile.max_relation_tokens,
                            "max_total_tokens": call.profile.max_total_tokens,
                        },
                        "reason": call.reason,
                    },
                )
            )
        self.sessions.append(
            session_id,
            SessionEventKind.OPERATION,
            (
                f"LightRAG: {call.question}\n"
                f"Профиль: {call.profile.name}\n"
                "Статус: завершено\n"
                f"Время: {duration:.1f} с\n"
                f"Точных фрагментов: {len(found)}\n"
                f"{response.answer}"
            ),
            {
                "attempt_id": attempt_id,
                "call_id": call.call_id,
                "evidence_ids": list(observation.evidence_ids),
                "profile": call.profile.name,
                "lightrag_result": True,
                "markdown_body_start_line": 5,
            },
        )
        return observation, tuple(found)

    @staticmethod
    def _check_duplicate(call: RetrievalCall, seen: set[tuple[str, str]]) -> None:
        key = (re.sub(r"\s+", " ", call.question).strip().casefold(), call.profile.name)
        if key in seen:
            raise GapAgentStepLimitError(
                "Заблокирован повтор одинакового вопроса и профиля"
            )
        seen.add(key)

    def _start_attempt(self, attempt_id: str, session_id: str, card_id: str, gap_id: str) -> None:
        with self.uow_factory() as uow:
            if uow.attempts.get(attempt_id):
                raise GapInvestigationError(f"Attempt {attempt_id} уже существует")
            uow.attempts.save(
                AttemptRecord(
                    attempt_id, session_id, "исследование пробела", AttemptStatus.ACTIVE,
                    {"card_id": card_id, "gap_id": gap_id}, self.clock(),
                )
            )
        self.sessions.append(session_id, SessionEventKind.OPERATION, "Исследую связанный пробел.\nСтатус: выполняется", {"attempt_id": attempt_id, "gap_id": gap_id})

    def _ensure_active(self, attempt_id: str) -> None:
        with self.uow_factory() as uow:
            attempt = uow.attempts.get(attempt_id)
        if attempt is None or attempt.status is not AttemptStatus.ACTIVE:
            raise AttemptDiscardedError(f"Результат attempt {attempt_id} больше не актуален")

    def _apply_result(
        self,
        attempt_id: str,
        operation: Callable[[UnitOfWork], GapInvestigationResult],
    ) -> GapInvestigationResult:
        with self.uow_factory() as uow:
            attempt = uow.attempts.get(attempt_id)
            if attempt is None or attempt.status is not AttemptStatus.ACTIVE:
                raise AttemptDiscardedError(
                    f"Результат attempt {attempt_id} больше нельзя применить"
                )
            applying = attempt.with_status(AttemptStatus.APPLYING, self.clock())
            uow.attempts.save(applying)
            uow.events.append(attempt_id, "применение исследования начато", {})
            result = operation(uow)
            uow.attempts.save(
                applying.with_status(AttemptStatus.COMPLETED, self.clock())
            )
            uow.events.append(
                attempt_id,
                "исследование завершено",
                {"outcome": result.outcome},
            )
            return result

    def _fail_attempt(self, attempt_id: str, error: str) -> None:
        with self.uow_factory() as uow:
            attempt = uow.attempts.get(attempt_id)
            if attempt and attempt.status in {
                AttemptStatus.ACTIVE,
                AttemptStatus.APPLYING,
            }:
                uow.attempts.save(attempt.with_status(AttemptStatus.FAILED, self.clock()))
                uow.records.save(StoredRecord("gap_diagnostic", attempt_id, {"error": error}))

    @staticmethod
    def _completion_event(
        card: TestCard,
        gap_id: str,
        result: GapInvestigationResult,
        arguments: GapArguments | None,
    ) -> tuple[SessionEventKind, str, dict[str, object]]:
        if result.outcome == "resolved":
            return (
                SessionEventKind.OPERATION,
                "Исследование пробела завершено.\nИсход: resolved",
                {"outcome": result.outcome},
            )
        if arguments is None:
            raise GapInvestigationError("Результат остановки не содержит аргументы Prompt 3")

        gap = card.gaps[gap_id]
        lines = [
            "Исследование остановлено: требуется решение аналитика.",
            "",
            f"Вопрос: {gap.question}",
        ]
        if result.outcome == "not_found":
            lines.extend(
                [
                    f"Не удалось определить: {arguments.missing_fact}",
                    f"Что найдено: {arguments.summary}",
                ]
            )
        else:
            lines.extend(
                [
                    "Причина: обнаружены противоречащие источники.",
                    f"Что найдено: {arguments.summary}",
                ]
            )
            lines.extend(
                f"- {item['statement']}"
                for item in arguments.contradictions
            )
        affected = arguments.unknown_fields or list(gap.allowed_paths)
        lines.append("Затронутые поля: " + ", ".join(affected))
        return (
            SessionEventKind.ASSISTANT,
            "\n".join(lines),
            {
                "outcome": result.outcome,
                "missing_fact": arguments.missing_fact,
                "unknown_fields": list(arguments.unknown_fields),
            },
        )

    def _load_card(self, card_id: str) -> TestCard:
        with self.uow_factory() as uow:
            card = uow.cards.get(card_id)
        if card is None:
            raise GapInvestigationError(f"Карточка {card_id} не найдена")
        return card

    @staticmethod
    def _card_context(card: TestCard) -> dict[str, object]:
        return {
            "card_id": card.card_id,
            "selection_id": card.selection_id,
            "revision": card.revision,
            "title": card.title,
            "fields": {
                path: {"status": field.status.value, "value": field.value}
                for path, field in card.fields.items()
            },
        }

    @staticmethod
    def _gap_context(gap: object) -> dict[str, object]:
        return {
            "gap_id": gap.gap_id,
            "question": gap.question,
            "blocking_reason": gap.blocking_reason,
            "allowed_paths": list(gap.allowed_paths),
            "dependencies": list(gap.dependencies),
            "closure_criterion": gap.closure_criterion,
            "resolution_mode": gap.resolution_mode.value,
        }

    @staticmethod
    def _evidence_context(item: Evidence) -> dict[str, object]:
        return {
            "evidence_id": item.evidence_id,
            "quote": item.quote,
            "address": (
                {
                    "document_id": item.address.document_id,
                    "document_version": item.address.document_version,
                    "page": item.address.page,
                    "line_start": item.address.line_start,
                    "line_end": item.address.line_end,
                    "chunk_id": item.address.chunk_id,
                }
                if item.address else None
            ),
        }
