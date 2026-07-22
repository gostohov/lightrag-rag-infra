from __future__ import annotations

from typing import Any

from ...domain import Evidence, TestCard
from ..llm import AttemptDiscardedError, LlmToolRuntime
from ..prompting import PromptId, PromptPolicy
from ..session import SessionEventKind, SessionService
from .models import PopulationArguments, PopulationError, PopulationResult
from .service import PopulationService


class CardPopulationFlow:
    def __init__(
        self,
        *,
        policy: PromptPolicy,
        runtime: LlmToolRuntime,
        service: PopulationService,
        sessions: SessionService,
    ) -> None:
        self.policy = policy
        self.runtime = runtime
        self.service = service
        self.sessions = sessions
        self._attempt_sessions: dict[str, str] = {}

    async def run(
        self,
        *,
        attempt_id: str,
        session_id: str,
        card_id: str,
        selection: dict[str, object],
        skeleton: dict[str, object],
        available_evidence: tuple[Evidence, ...],
    ) -> PopulationResult:
        card = self._load_card(card_id)
        call = self.policy.build_call(
            PromptId.POPULATION,
            {
                "selection": selection,
                "skeleton": skeleton,
                "card": self._card_context(card),
                "evidence": [self._evidence_context(item) for item in available_evidence],
            },
        )
        self._attempt_sessions[attempt_id] = session_id
        self.sessions.append(
            session_id,
            SessionEventKind.OPERATION,
            "Заполняю первоначальную карточку.\nСтатус: выполняется",
            {"attempt_id": attempt_id},
        )
        try:
            decoded = await self.runtime.invoke(attempt_id, session_id, call)
            if decoded.name != "submit_card_population" or not isinstance(
                decoded.arguments, PopulationArguments
            ):
                raise PopulationError("Runtime вернул неожиданный результат Промпта 2")
            result = self.runtime.apply_result(
                attempt_id,
                lambda uow: self.service.apply(
                    card_id,
                    decoded.arguments,
                    available_evidence=available_evidence,
                    uow=uow,
                ),
            )
        except AttemptDiscardedError:
            raise
        except Exception as error:
            self.sessions.append(
                session_id,
                SessionEventKind.ERROR,
                f"Первоначальное заполнение не применено: {error}",
                {"attempt_id": attempt_id},
            )
            raise
        finally:
            self._attempt_sessions.pop(attempt_id, None)
        self.sessions.append(
            session_id,
            SessionEventKind.OPERATION,
            (
                "Первоначальная карточка заполнена.\n"
                f"Статус: завершено\nБлокирующих пробелов: {len(result.open_gap_ids)}"
            ),
            {"attempt_id": attempt_id, "revision": result.revision},
        )
        return result

    def cancel(self, attempt_id: str) -> None:
        session_id = self._attempt_sessions.get(attempt_id)
        if session_id is None:
            self.runtime.cancel(attempt_id)
            return
        self.sessions.cancel_operation(session_id, attempt_id)

    def _load_card(self, card_id: str) -> TestCard:
        with self.service.uow_factory() as uow:
            card = uow.cards.get(card_id)
        if card is None:
            raise PopulationError(f"Карточка {card_id} не найдена")
        return card

    @staticmethod
    def _card_context(card: TestCard) -> dict[str, Any]:
        return {
            "card_id": card.card_id,
            "selection_id": card.selection_id,
            "title": card.title,
            "section_number": card.section_number,
            "changed_factor": card.changed_factor,
            "consequences": list(card.consequences),
            "revision": card.revision,
        }

    @staticmethod
    def _evidence_context(item: Evidence) -> dict[str, Any]:
        return {
            "evidence_id": item.evidence_id,
            "kind": item.kind.value,
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
                if item.address
                else None
            ),
        }
