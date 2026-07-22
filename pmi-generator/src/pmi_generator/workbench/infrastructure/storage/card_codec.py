from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any

from ...domain import (
    AnalystResolution,
    CardDecision,
    CardDecisionKind,
    CardState,
    ContentField,
    Derivation,
    DomainError,
    EpistemicStatus,
    Evidence,
    EvidenceKind,
    EvidenceScope,
    GapResolutionMode,
    GapClosureContract,
    GapPathClosure,
    GapValueForm,
    GapStatus,
    RelatedGap,
    SourceAddress,
    TestCard,
)
from .errors import StorageError


def encode_card(card: TestCard) -> str:
    state = card.snapshot()
    payload = {
        "card_id": state.card_id,
        "selection_id": state.selection_id,
        "title": state.title,
        "section_number": state.section_number,
        "changed_factor": state.changed_factor,
        "consequences": list(state.consequences),
        "revision": state.revision,
        "fields": {path: _field_to_dict(value) for path, value in state.fields.items()},
        "evidence": [_evidence_to_dict(item) for item in state.evidence],
        "derivations": [_derivation_to_dict(item) for item in state.derivations],
        "gaps": [_gap_to_dict(item) for item in state.gaps],
        "resolutions": [_resolution_to_dict(item) for item in state.resolutions],
        "decision": _decision_to_dict(state.decision),
        "selection_review_current": state.selection_review_current,
    }
    try:
        return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise StorageError(f"Карточка содержит несериализуемое значение: {exc}") from exc


def decode_card(raw: str) -> TestCard:
    try:
        payload = json.loads(raw)
        evidence = tuple(
            _evidence_from_dict(item)
            for item in payload["evidence"]
        )
        resolutions = [
            _resolution_from_dict(item)
            for item in payload["resolutions"]
        ]
        fields, resolutions = _migrate_legacy_analyst_fields(
            card_id=str(payload["card_id"]),
            raw_fields=payload["fields"],
            evidence=evidence,
            resolutions=resolutions,
        )
        state = CardState(
            card_id=str(payload["card_id"]),
            selection_id=str(payload["selection_id"]),
            title=str(payload["title"]),
            section_number=str(payload["section_number"]),
            changed_factor=str(payload["changed_factor"]),
            consequences=tuple(str(item) for item in payload["consequences"]),
            revision=int(payload["revision"]),
            fields=fields,
            evidence=evidence,
            derivations=tuple(_derivation_from_dict(item) for item in payload["derivations"]),
            gaps=tuple(_gap_from_dict(item) for item in payload["gaps"]),
            resolutions=tuple(resolutions),
            decision=_decision_from_dict(payload.get("decision")),
            selection_review_current=bool(payload["selection_review_current"]),
        )
        return TestCard.restore(state)
    except (DomainError, KeyError, TypeError, ValueError) as exc:
        raise StorageError(f"Некорректная запись карточки: {exc}") from exc


def _field_to_dict(value: ContentField) -> dict[str, Any]:
    return {
        "status": value.status.value,
        "value": value.value,
        "evidence_ids": list(value.evidence_ids),
        "derivation_id": value.derivation_id,
        "reason": value.reason,
    }


def _field_from_dict(value: dict[str, Any]) -> ContentField:
    return ContentField(
        status=EpistemicStatus(value["status"]),
        value=value.get("value"),
        evidence_ids=tuple(value.get("evidence_ids") or ()),
        derivation_id=value.get("derivation_id"),
        reason=value.get("reason"),
    )


def _migrate_legacy_analyst_fields(
    *,
    card_id: str,
    raw_fields: dict[str, dict[str, Any]],
    evidence: tuple[Evidence, ...],
    resolutions: list[AnalystResolution],
) -> tuple[dict[str, ContentField], list[AnalystResolution]]:
    evidence_by_id = {item.evidence_id: item for item in evidence}
    fields: dict[str, ContentField] = {}
    legacy_groups: dict[
        tuple[str, ...],
        list[tuple[str, ContentField]],
    ] = {}
    for path, raw_field in raw_fields.items():
        field = _field_from_dict(raw_field)
        if field.status is not EpistemicStatus.SOURCE_CONFIRMED:
            fields[path] = field
            continue
        kinds = {
            evidence_by_id[evidence_id].kind
            for evidence_id in field.evidence_ids
            if evidence_id in evidence_by_id
        }
        if kinds == {EvidenceKind.HUMAN_KNOWLEDGE}:
            migrated = ContentField.analyst_confirmed(
                field.value,
                field.evidence_ids,
            )
            fields[path] = migrated
            legacy_groups.setdefault(
                migrated.evidence_ids,
                [],
            ).append((path, migrated))
            continue
        if EvidenceKind.HUMAN_KNOWLEDGE in kinds:
            raise StorageError(
                f"Legacy-поле {path} содержит смешанное source и analyst evidence"
            )
        fields[path] = field

    for evidence_ids, grouped_fields in sorted(legacy_groups.items()):
        target_paths = tuple(sorted(path for path, _field in grouped_fields))
        if any(
            set(target_paths).issubset(resolution.target_paths)
            and set(evidence_ids).issubset(resolution.evidence_ids)
            for resolution in resolutions
        ):
            continue
        human = [evidence_by_id[evidence_id] for evidence_id in evidence_ids]
        fingerprint = hashlib.sha256(
            json.dumps(
                {
                    "card_id": card_id,
                    "target_paths": target_paths,
                    "evidence_ids": evidence_ids,
                },
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()[:16].upper()
        resolutions.append(
            AnalystResolution(
                resolution_id=f"RESOLUTION_LEGACY_{fingerprint}",
                card_id=card_id,
                author=next(
                    (
                        item.author
                        for item in human
                        if item.author
                    ),
                    "Аналитик",
                ),
                created_at=min(item.collected_at for item in human),
                reason="Миграция legacy human knowledge",
                target_paths=target_paths,
                evidence_ids=evidence_ids,
                source_message_id=next(
                    (
                        item.message_id
                        for item in human
                        if item.message_id
                    ),
                    None,
                ),
                values=tuple(
                    {
                        "path": path,
                        "value": field.value,
                    }
                    for path, field in grouped_fields
                ),
            )
        )
    return fields, resolutions


def _address_to_dict(value: SourceAddress | None) -> dict[str, Any] | None:
    if value is None:
        return None
    return {
        "document_id": value.document_id,
        "document_version": value.document_version,
        "page": value.page,
        "line_start": value.line_start,
        "line_end": value.line_end,
        "chunk_id": value.chunk_id,
    }


def _address_from_dict(value: dict[str, Any] | None) -> SourceAddress | None:
    return SourceAddress(**value) if value is not None else None


def _evidence_to_dict(value: Evidence) -> dict[str, Any]:
    return {
        "evidence_id": value.evidence_id,
        "kind": value.kind.value,
        "scope": value.scope.value,
        "selection_id": value.selection_id,
        "quote": value.quote,
        "collected_at": value.collected_at.isoformat(),
        "card_id": value.card_id,
        "address": _address_to_dict(value.address),
        "author": value.author,
        "message_id": value.message_id,
    }


def _evidence_from_dict(value: dict[str, Any]) -> Evidence:
    return Evidence(
        evidence_id=value["evidence_id"],
        kind=EvidenceKind(value["kind"]),
        scope=EvidenceScope(value["scope"]),
        selection_id=value["selection_id"],
        quote=value["quote"],
        collected_at=datetime.fromisoformat(value["collected_at"]),
        card_id=value.get("card_id"),
        address=_address_from_dict(value.get("address")),
        author=value.get("author"),
        message_id=value.get("message_id"),
    )


def _derivation_to_dict(value: Derivation) -> dict[str, Any]:
    return {
        "derivation_id": value.derivation_id,
        "card_id": value.card_id,
        "source_evidence_ids": list(value.source_evidence_ids),
        "rule": value.rule,
        "scope": value.scope,
    }


def _derivation_from_dict(value: dict[str, Any]) -> Derivation:
    return Derivation(
        derivation_id=value["derivation_id"],
        card_id=value["card_id"],
        source_evidence_ids=tuple(value["source_evidence_ids"]),
        rule=value["rule"],
        scope=value["scope"],
    )


def _gap_to_dict(value: RelatedGap) -> dict[str, Any]:
    return {
        "gap_id": value.gap_id,
        "card_id": value.card_id,
        "question": value.question,
        "blocking_reason": value.blocking_reason,
        "allowed_paths": list(value.allowed_paths),
        "dependencies": list(value.dependencies),
        "closure_criterion": value.closure_criterion,
        "closure_contract": {
            "schema_version": value.closure_contract.schema_version,
            "requirements": [
                {
                    "path": item.path,
                    "accepted_forms": [
                        form.value for form in item.accepted_forms
                    ],
                    "residual_question": item.residual_question,
                }
                for item in value.closure_contract.requirements
            ],
        },
        "closure_satisfied_paths": list(value.closure_satisfied_paths),
        "resolution_mode": value.resolution_mode.value,
        "status": value.status.value,
    }


def _gap_from_dict(value: dict[str, Any]) -> RelatedGap:
    raw_contract = value.get("closure_contract")
    contract = (
        GapClosureContract(
            schema_version=int(raw_contract["schema_version"]),
            requirements=tuple(
                GapPathClosure(
                    path=str(item["path"]),
                    accepted_forms=tuple(
                        GapValueForm(str(form))
                        for form in item["accepted_forms"]
                    ),
                    residual_question=str(item["residual_question"]),
                )
                for item in raw_contract["requirements"]
            ),
        )
        if raw_contract is not None
        else GapClosureContract.legacy(
            tuple(value["allowed_paths"]),
            question=str(value["question"]),
        )
    )
    return RelatedGap(
        gap_id=value["gap_id"],
        card_id=value["card_id"],
        question=value["question"],
        blocking_reason=value["blocking_reason"],
        allowed_paths=tuple(value["allowed_paths"]),
        dependencies=tuple(value["dependencies"]),
        closure_criterion=value["closure_criterion"],
        closure_contract=contract,
        closure_satisfied_paths=tuple(
            value.get("closure_satisfied_paths") or ()
        ),
        resolution_mode=GapResolutionMode(
            value.get("resolution_mode", GapResolutionMode.SOURCE_FACT.value)
        ),
        status=GapStatus(value["status"]),
    )


def _resolution_to_dict(value: AnalystResolution) -> dict[str, Any]:
    return {
        "resolution_id": value.resolution_id,
        "card_id": value.card_id,
        "author": value.author,
        "created_at": value.created_at.isoformat(),
        "reason": value.reason,
        "target_paths": list(value.target_paths),
        "evidence_ids": list(value.evidence_ids),
        "source_message_id": value.source_message_id,
        "confirmation_message_id": value.confirmation_message_id,
        "proposal_id": value.proposal_id,
        "gap_id": value.gap_id,
        "expected_revision": value.expected_revision,
        "values": list(value.values),
    }


def _resolution_from_dict(value: dict[str, Any]) -> AnalystResolution:
    return AnalystResolution(
        resolution_id=value["resolution_id"],
        card_id=value["card_id"],
        author=value["author"],
        created_at=datetime.fromisoformat(value["created_at"]),
        reason=value["reason"],
        target_paths=tuple(value["target_paths"]),
        evidence_ids=tuple(value["evidence_ids"]),
        source_message_id=value.get("source_message_id"),
        confirmation_message_id=value.get("confirmation_message_id"),
        proposal_id=value.get("proposal_id"),
        gap_id=value.get("gap_id"),
        expected_revision=value.get("expected_revision"),
        values=tuple(value.get("values") or ()),
    )


def _decision_to_dict(value: CardDecision | None) -> dict[str, Any] | None:
    if value is None:
        return None
    return {
        "kind": value.kind.value,
        "card_id": value.card_id,
        "revision": value.revision,
        "author": value.author,
        "created_at": value.created_at.isoformat(),
        "reason": value.reason,
    }


def _decision_from_dict(value: dict[str, Any] | None) -> CardDecision | None:
    if value is None:
        return None
    return CardDecision(
        kind=CardDecisionKind(value["kind"]),
        card_id=value["card_id"],
        revision=int(value["revision"]),
        author=value["author"],
        created_at=datetime.fromisoformat(value["created_at"]),
        reason=value.get("reason"),
    )
