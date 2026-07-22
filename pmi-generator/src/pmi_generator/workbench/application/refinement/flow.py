from __future__ import annotations

from ..card_population import AnalystMessage
from ..conversation import AnalystProposalService
from ..llm import LlmToolRuntime
from ..prompting import PromptId, PromptPolicy
from ..repositories import UnitOfWork
from ..session import SessionEventKind, SessionService
from .models import (
    RefinementArguments,
    RefinementError,
    RefinementProposalResult,
)
from .service import CardRefinementService


class CardRefinementFlow:
    def __init__(
        self,
        *,
        policy: PromptPolicy,
        runtime: LlmToolRuntime,
        service: CardRefinementService,
        sessions: SessionService,
        proposals: AnalystProposalService | None = None,
    ) -> None:
        self.policy = policy
        self.runtime = runtime
        self.service = service
        self.sessions = sessions
        self.proposals = proposals

    async def plan(
        self,
        *,
        attempt_id: str,
        session_id: str,
        card_id: str,
        message_id: str,
        expected_revision: int,
    ) -> RefinementProposalResult:
        if self.proposals is None:
            raise RefinementError(
                "Refinement proposal service не настроен"
            )
        with self.service.uow_factory() as uow:
            card = uow.cards.get(card_id)
        if card is None or card.revision != expected_revision:
            raise RefinementError(
                "Карточка изменилась до построения refinement proposal"
            )
        messages = self._messages(session_id, card_id)
        message = next(
            (
                item
                for item in messages
                if item.message_id == message_id
            ),
            None,
        )
        if message is None:
            raise RefinementError(f"Сообщение {message_id} не найдено")
        call = self.policy.build_call(
            PromptId.REFINEMENT,
            {
                "card": {
                    "card_id": card.card_id,
                    "revision": card.revision,
                    "fields": {
                        path: {
                            "status": field.status.value,
                            "value": field.value,
                        }
                        for path, field in card.fields.items()
                    },
                },
                "message": {
                    "message_id": message.message_id,
                    "text": message.text,
                    "author": message.author,
                },
                "evidence": [
                    {
                        "evidence_id": item.evidence_id,
                        "quote": item.quote,
                    }
                    for item in card.evidence.values()
                ],
            },
        )
        decoded = await self.runtime.invoke(
            attempt_id,
            session_id,
            call,
        )
        if (
            decoded.name != "submit_card_refinement"
            or not isinstance(decoded.arguments, RefinementArguments)
        ):
            raise RefinementError(
                "Runtime вернул неожиданный результат доработки"
            )

        def persist(uow: UnitOfWork) -> RefinementProposalResult:
            current = uow.cards.get(card_id)
            if current is None or current.revision != expected_revision:
                raise RefinementError(
                    "Карточка изменилась до сохранения refinement proposal"
                )
            self.service.validate_proposal(
                card_id,
                decoded.arguments,
                analyst_messages=(message,),
                uow=uow,
            )
            if decoded.arguments.outcome == "no_change":
                return RefinementProposalResult(
                    card_id,
                    expected_revision,
                    "no_change",
                    None,
                )
            arguments = {
                "outcome": decoded.arguments.outcome,
                "updates": decoded.arguments.updates,
                "gaps": decoded.arguments.gaps,
                "reason": decoded.arguments.reason,
            }
            values = [
                {
                    "path": str(item["path"]),
                    "value": item["value"],
                }
                for item in decoded.arguments.updates
            ]
            proposal = self.proposals.create_refinement(
                session_id=session_id,
                card_id=card_id,
                source_message_id=message_id,
                expected_revision=expected_revision,
                arguments=arguments,
                values=values,
                uow=uow,
            )
            return RefinementProposalResult(
                card_id,
                expected_revision,
                decoded.arguments.outcome,
                proposal.record_id,
            )

        return self.runtime.apply_result(attempt_id, persist)

    def _messages(self, session_id: str, card_id: str) -> tuple[AnalystMessage, ...]:
        return tuple(
            AnalystMessage(
                str(event.metadata.get("message_id") or f"MSG_{event.sequence:06d}"),
                card_id,
                str(event.metadata.get("author") or "Аналитик"),
                event.text,
                event.created_at,
            )
            for event in self.sessions.history(session_id)
            if event.kind is SessionEventKind.ANALYST
        )
