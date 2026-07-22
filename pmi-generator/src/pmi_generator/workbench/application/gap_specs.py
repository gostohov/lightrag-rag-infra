from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from ..domain import (
    GapClosureContract,
    GapPathClosure,
    GapResolutionMode,
    GapValueForm,
    RelatedGap,
)


RELATED_GAP_KEYS = frozenset(
    {
        "question",
        "blocking_reason",
        "allowed_paths",
        "dependencies",
        "closure_criterion",
        "resolution_mode",
    }
)
RESOLUTION_TARGET_KEYS = frozenset(
    {
        "path",
        "resolution_mode",
        "accepted_forms",
        "residual_question",
    }
)


def expand_related_gap_spec(
    item: Mapping[str, Any],
    *,
    card_id: str,
    next_id: Callable[[str], str],
) -> tuple[RelatedGap, ...]:
    keys = frozenset(item)
    if keys not in {
        RELATED_GAP_KEYS,
        RELATED_GAP_KEYS | {"resolution_targets"},
    }:
        raise ValueError("Пробел имеет неверную структуру")

    allowed_paths = tuple(str(value) for value in item["allowed_paths"])
    if not allowed_paths or len(allowed_paths) != len(set(allowed_paths)):
        raise ValueError("Пробел требует уникальные разрешённые пути")
    dependencies = tuple(str(value) for value in item["dependencies"])
    question = str(item["question"])
    blocking_reason = str(item["blocking_reason"])
    closure_criterion = str(item["closure_criterion"])
    raw_targets = item.get("resolution_targets")
    if raw_targets is None:
        if len(allowed_paths) > 1:
            raise ValueError(
                "Multi-path gap требует typed resolution targets"
            )
        return (
            RelatedGap(
                gap_id=next_id("GAP"),
                card_id=card_id,
                question=question,
                blocking_reason=blocking_reason,
                allowed_paths=allowed_paths,
                dependencies=dependencies,
                closure_criterion=closure_criterion,
                resolution_mode=GapResolutionMode(
                    str(item["resolution_mode"])
                ),
            ),
        )
    if not isinstance(raw_targets, list) or not raw_targets:
        raise ValueError("resolution targets требуют непустой список")

    targets: dict[str, Mapping[str, Any]] = {}
    for target in raw_targets:
        if (
            not isinstance(target, Mapping)
            or frozenset(target) != RESOLUTION_TARGET_KEYS
        ):
            raise ValueError("resolution target имеет неверную структуру")
        path = str(target["path"])
        if path in targets:
            raise ValueError(f"resolution target {path} повторён")
        targets[path] = target
    if set(targets) != set(allowed_paths):
        raise ValueError(
            "resolution targets должны точно покрывать allowed_paths"
        )

    mode_priority = {
        GapResolutionMode.SOURCE_FACT: 0,
        GapResolutionMode.DESIGN_DECISION: 1,
        GapResolutionMode.EXTERNAL_INPUT: 2,
    }
    ordered_paths = sorted(
        allowed_paths,
        key=lambda path: (
            mode_priority[
                GapResolutionMode(
                    str(targets[path]["resolution_mode"])
                )
            ],
            allowed_paths.index(path),
        ),
    )
    result: list[RelatedGap] = []
    for path in ordered_paths:
        target = targets[path]
        raw_forms = target["accepted_forms"]
        if (
            not isinstance(raw_forms, list)
            or not raw_forms
            or len(raw_forms) != len(set(str(value) for value in raw_forms))
        ):
            raise ValueError(
                f"resolution target {path} требует уникальные accepted_forms"
            )
        forms = tuple(GapValueForm(str(value)) for value in raw_forms)
        residual_question = str(target["residual_question"]).strip()
        if not residual_question:
            raise ValueError(
                f"resolution target {path} требует остаточный вопрос"
            )
        result.append(
            RelatedGap(
                gap_id=next_id("GAP"),
                card_id=card_id,
                question=residual_question,
                blocking_reason=blocking_reason,
                allowed_paths=(path,),
                dependencies=dependencies,
                closure_criterion=closure_criterion,
                closure_contract=GapClosureContract(
                    requirements=(
                        GapPathClosure(
                            path=path,
                            accepted_forms=forms,
                            residual_question=residual_question,
                        ),
                    ),
                ),
                resolution_mode=GapResolutionMode(
                    str(target["resolution_mode"])
                ),
            )
        )
    return tuple(result)
