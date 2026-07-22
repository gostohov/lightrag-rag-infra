from __future__ import annotations

from ..llm import LlmToolRuntime
from ..prompting import PromptId, PromptPolicy
from ..source import SavedSelection
from .models import SelectionReviewArguments, SelectionReviewError, SelectionReviewResult
from .service import SelectionReviewService


class SelectionReviewFlow:
    def __init__(self, *, policy: PromptPolicy, runtime: LlmToolRuntime, service: SelectionReviewService) -> None:
        self.policy = policy
        self.runtime = runtime
        self.service = service

    async def run(self, attempt_id: str, session_id: str, selection: SavedSelection) -> SelectionReviewResult:
        if not self.service.workspace.load(selection.selection_id).can_review:
            raise SelectionReviewError("Диапазон пока не готов к Prompt 4")
        with self.service.uow_factory() as uow:
            cards = [card for card in uow.cards.list_all() if card.selection_id == selection.selection_id]
            skeletons = [record for record in uow.records.list_kind("card_skeleton") if record.payload.get("selection_id") == selection.selection_id]
        call = self.policy.build_call(
            PromptId.SELECTION_REVIEW,
            {
                "selection": {
                    "selection_id": selection.selection_id,
                    "text": selection.selection.text,
                    "positions": [{"page": item.page_index, "line": item.line_number} for item in selection.selection.positions],
                },
                "cards": [self._card_context(card) for card in cards],
                "skeleton_decisions": [dict(record.payload, skeleton_id=record.record_id) for record in skeletons],
            },
        )
        decoded = await self.runtime.invoke(attempt_id, session_id, call)
        if decoded.name != "submit_selection_review" or not isinstance(decoded.arguments, SelectionReviewArguments):
            raise SelectionReviewError("Runtime вернул неожиданный результат Prompt 4")
        return self.runtime.apply_result(
            attempt_id,
            lambda uow: self.service.apply(selection, decoded.arguments, uow=uow),
        )

    @staticmethod
    def _card_context(card: object) -> dict[str, object]:
        return {
            "card_id": card.card_id,
            "revision": card.revision,
            "title": card.title,
            "fields": {path: {"status": field.status.value, "value": field.value} for path, field in card.fields.items()},
            "evidence": [{"evidence_id": item.evidence_id, "kind": item.kind.value, "quote": item.quote} for item in card.evidence.values()],
            "analyst_resolutions": [
                {"resolution_id": item.resolution_id, "target_paths": list(item.target_paths), "reason": item.reason}
                for item in card.resolutions.values()
            ],
            "decision": card.decision.kind.value if card.decision else None,
        }
