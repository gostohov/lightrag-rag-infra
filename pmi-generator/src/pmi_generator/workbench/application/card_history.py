from __future__ import annotations

from typing import Any

from ..domain import TestCard
from .repositories import UnitOfWork
from .state import StoredRecord


def save_card_revision(
    uow: UnitOfWork,
    card: TestCard,
    *,
    reason: str,
) -> None:
    payload = card_snapshot_payload(card, reason=reason)
    uow.records.save(
        StoredRecord(
            "card_revision",
            f"{card.card_id}:r{card.revision:06d}",
            payload,
        )
    )


def card_snapshot_payload(
    card: TestCard,
    *,
    reason: str,
) -> dict[str, Any]:
    decision = card.decision
    payload: dict[str, Any] = {
        "card_id": card.card_id,
        "selection_id": card.selection_id,
        "revision": card.revision,
        "reason": reason,
        "title": card.title,
        "section_number": card.section_number,
        "changed_factor": card.changed_factor,
        "consequences": list(card.consequences),
        "fields": {
            path: {
                "status": field.status.value,
                "value": field.value,
                "evidence_ids": list(field.evidence_ids),
                "derivation_id": field.derivation_id,
                "reason": field.reason,
            }
            for path, field in card.fields.items()
        },
        "evidence": {
            item.evidence_id: {
                "kind": item.kind.value,
                "scope": item.scope.value,
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
                "author": item.author,
                "message_id": item.message_id,
            }
            for item in card.evidence.values()
        },
        "derivations": {
            item.derivation_id: {
                "source_evidence_ids": list(item.source_evidence_ids),
                "rule": item.rule,
                "scope": item.scope,
            }
            for item in card.derivations.values()
        },
        "gaps": {
            gap.gap_id: {
                "question": gap.question,
                "blocking_reason": gap.blocking_reason,
                "allowed_paths": list(gap.allowed_paths),
                "dependencies": list(gap.dependencies),
                "closure_criterion": gap.closure_criterion,
                "closure_contract": {
                    "schema_version": gap.closure_contract.schema_version,
                    "requirements": [
                        {
                            "path": item.path,
                            "accepted_forms": [
                                form.value for form in item.accepted_forms
                            ],
                            "residual_question": item.residual_question,
                        }
                        for item in gap.closure_contract.requirements
                    ],
                },
                "closure_satisfied_paths": list(
                    gap.closure_satisfied_paths
                ),
                "resolution_mode": gap.resolution_mode.value,
                "status": gap.status.value,
            }
            for gap in card.gaps.values()
        },
        "resolutions": {
            item.resolution_id: {
                "author": item.author,
                "created_at": item.created_at.isoformat(),
                "reason": item.reason,
                "target_paths": list(item.target_paths),
                "evidence_ids": list(item.evidence_ids),
            }
            for item in card.resolutions.values()
        },
        "decision": (
            {
                "kind": decision.kind.value,
                "revision": decision.revision,
                "author": decision.author,
                "created_at": decision.created_at.isoformat(),
                "reason": decision.reason,
            }
            if decision
            else None
        ),
    }
    return payload


__all__ = ["card_snapshot_payload", "save_card_revision"]
