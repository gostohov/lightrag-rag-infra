from __future__ import annotations

import json

from ...domain.source import SourceDocument, SourcePosition, TextSelection
from .models import DecompositionArguments, DecompositionError


class DecompositionValidator:
    def __init__(self, document: SourceDocument) -> None:
        self.document = document

    def validate(
        self,
        selection: TextSelection,
        arguments: DecompositionArguments,
    ) -> list[dict[str, object]]:
        allowed_outcomes = {
            "skeletons_created",
            "no_testable_behavior",
            "insufficient_selection",
        }
        if arguments.outcome not in allowed_outcomes:
            raise DecompositionError(f"Неизвестный исход {arguments.outcome}")
        if arguments.outcome != "skeletons_created":
            if arguments.skeletons:
                raise DecompositionError(
                    "Каркасы допустимы только для skeletons_created"
                )
            if not arguments.explanation.strip():
                raise DecompositionError(
                    "Неположительный исход требует объяснения"
                )
            self.validate_line_assessments(
                selection,
                arguments.line_assessments,
                evidence_positions=set(),
            )
            return []
        if not arguments.skeletons:
            raise DecompositionError(
                "skeletons_created требует непустой набор каркасов"
            )

        validated_with_positions = [
            self.validate_skeleton(selection, item)
            for item in arguments.skeletons
        ]
        validated = [item for item, _positions in validated_with_positions]
        evidence_positions = {
            position
            for _item, positions in validated_with_positions
            for position in positions
        }
        self.validate_line_assessments(
            selection,
            arguments.line_assessments,
            evidence_positions=evidence_positions,
        )
        canonical = [
            json.dumps(item, ensure_ascii=False, sort_keys=True)
            for item in validated
        ]
        if len(canonical) != len(set(canonical)):
            raise DecompositionError(
                "Результат содержит полностью одинаковые каркасы"
            )
        return validated

    def validate_skeleton(
        self,
        selection: TextSelection,
        item: dict[str, object],
    ) -> tuple[dict[str, object], set[SourcePosition]]:
        required = {
            "title",
            "condition",
            "changed_factor",
            "input_value",
            "action",
            "condition_ranges",
            "changed_factor_ranges",
            "input_value_ranges",
            "action_ranges",
            "consequences",
            "gaps",
        }
        if set(item) != required:
            raise DecompositionError(
                "Каркас содержит неизвестные поля или неполон"
            )
        for name in ("title", "condition", "changed_factor"):
            if not str(item[name]).strip():
                raise DecompositionError(f"Поле {name} обязательно")
        consequences = item["consequences"]
        if not isinstance(consequences, list) or not consequences:
            raise DecompositionError(
                "Каркас должен содержать обязательные последствия"
            )
        gaps = self._validate_gaps(item["gaps"])
        gap_kinds = {str(gap["kind"]) for gap in gaps}
        if item["input_value"] is None and "input_value" not in gap_kinds:
            raise DecompositionError(
                "Отсутствующее входное значение требует блокирующий пробел"
            )
        if item["action"] is None and "action" not in gap_kinds:
            raise DecompositionError(
                "Отсутствующее воздействие требует блокирующий пробел"
            )

        condition_evidence = self.evidence(
            selection,
            item["condition_ranges"],
        )
        if item["changed_factor_ranges"] == []:
            raise DecompositionError(
                "changed_factor требует непустые changed_factor_ranges"
            )
        changed_factor_evidence = self.evidence(
            selection,
            item["changed_factor_ranges"],
        )
        if item["input_value"] is None:
            if item["input_value_ranges"] != []:
                raise DecompositionError(
                    "input_value=null требует пустые input_value_ranges"
                )
            input_value_evidence: list[dict[str, object]] = []
        else:
            if item["input_value_ranges"] == []:
                raise DecompositionError(
                    "input_value требует непустые input_value_ranges"
                )
            input_value_evidence = self.evidence(
                selection,
                item["input_value_ranges"],
            )
        if item["action"] is None:
            if item["action_ranges"] != []:
                raise DecompositionError(
                    "action=null требует пустые action_ranges"
                )
            action_evidence: list[dict[str, object]] = []
        else:
            if item["action_ranges"] == []:
                raise DecompositionError(
                    "action требует непустые action_ranges"
                )
            action_evidence = self.evidence(
                selection,
                item["action_ranges"],
            )
        normalized_consequences: list[dict[str, object]] = []
        for consequence in consequences:
            if not isinstance(consequence, dict) or set(consequence) != {
                "text",
                "evidence_ranges",
            }:
                raise DecompositionError(
                    "Последствие имеет неверную структуру"
                )
            if not str(consequence["text"]).strip():
                raise DecompositionError("Текст последствия обязателен")
            normalized_consequences.append(
                {
                    "text": str(consequence["text"]),
                    "evidence": self.evidence(
                        selection,
                        consequence["evidence_ranges"],
                    ),
                }
            )
        payload = {
            "title": str(item["title"]),
            "condition": str(item["condition"]),
            "changed_factor": str(item["changed_factor"]),
            "input_value": item["input_value"],
            "action": item["action"],
            "condition_evidence": condition_evidence,
            "changed_factor_evidence": changed_factor_evidence,
            "input_value_evidence": input_value_evidence,
            "action_evidence": action_evidence,
            "consequences": normalized_consequences,
            "gaps": gaps,
        }
        evidence_positions = self.evidence_positions(
            condition_evidence
            + changed_factor_evidence
            + input_value_evidence
            + action_evidence
            + [
                evidence
                for consequence in normalized_consequences
                for evidence in consequence["evidence"]
            ]
        )
        return payload, evidence_positions

    def evidence(
        self,
        selection: TextSelection,
        raw_ranges: object,
    ) -> list[dict[str, object]]:
        if not isinstance(raw_ranges, list) or not raw_ranges:
            raise DecompositionError("Доказательная привязка обязательна")
        allowed = set(selection.positions)
        result: list[dict[str, object]] = []
        for raw in raw_ranges:
            if not isinstance(raw, dict) or set(raw) != {
                "page",
                "line_start",
                "line_end",
            }:
                raise DecompositionError(
                    "Диапазон evidence имеет неверную структуру"
                )
            start = SourcePosition(int(raw["page"]), int(raw["line_start"]))
            end = SourcePosition(int(raw["page"]), int(raw["line_end"]))
            if end.line_number < start.line_number:
                raise DecompositionError(
                    "Конец evidence range находится перед началом"
                )
            try:
                excerpt = self.document.select(start, end)
            except ValueError as error:
                raise DecompositionError(str(error)) from error
            if not set(excerpt.positions) <= allowed:
                raise DecompositionError(
                    "Координаты evidence выходят за пределы выбранного диапазона"
                )
            result.append(
                {
                    "page": start.page_index,
                    "line_start": start.line_number,
                    "line_end": end.line_number,
                    "quote": excerpt.text,
                }
            )
        return result

    def evidence_positions(
        self,
        evidence: list[dict[str, object]],
    ) -> set[SourcePosition]:
        result: set[SourcePosition] = set()
        for item in evidence:
            excerpt = self.document.select(
                SourcePosition(
                    int(item["page"]),
                    int(item["line_start"]),
                ),
                SourcePosition(
                    int(item["page"]),
                    int(item["line_end"]),
                ),
            )
            result.update(excerpt.positions)
        return result

    @staticmethod
    def validate_line_assessments(
        selection: TextSelection,
        raw_assessments: object,
        *,
        evidence_positions: set[SourcePosition],
    ) -> None:
        if not isinstance(raw_assessments, list):
            raise DecompositionError(
                "line_assessments должен быть списком"
            )
        assessed: dict[SourcePosition, str] = {}
        for item in raw_assessments:
            if not isinstance(item, dict) or set(item) != {
                "page",
                "line",
                "role",
                "reason",
            }:
                raise DecompositionError(
                    "line_assessments имеет неверную структуру"
                )
            position = SourcePosition(int(item["page"]), int(item["line"]))
            if position in assessed:
                raise DecompositionError(
                    f"Строка {position.page_index}:{position.line_number} "
                    "классифицирована дважды"
                )
            role = str(item["role"])
            if role not in {"evidence", "context"}:
                raise DecompositionError(
                    f"Неизвестная роль строки: {role}"
                )
            if not str(item["reason"]).strip():
                raise DecompositionError(
                    "Классификация строки требует причину"
                )
            assessed[position] = role

        selected_positions = set(selection.positions)
        missing = selected_positions - set(assessed)
        outside = set(assessed) - selected_positions
        if missing:
            formatted = ", ".join(
                f"{item.page_index}:{item.line_number}"
                for item in sorted(missing)
            )
            raise DecompositionError(
                f"Строки selection не классифицированы: {formatted}"
            )
        if outside:
            raise DecompositionError(
                "line_assessments выходит за пределы selection"
            )

        assessed_evidence = {
            position
            for position, role in assessed.items()
            if role == "evidence"
        }
        if evidence_positions - assessed_evidence:
            raise DecompositionError(
                "Роль evidence обязательна для каждой строки, "
                "использованной каркасом"
            )
        if assessed_evidence - evidence_positions:
            raise DecompositionError(
                "Роль evidence допустима только для строки, "
                "использованной каркасом"
            )

    @staticmethod
    def _validate_gaps(raw_gaps: object) -> list[dict[str, object]]:
        if not isinstance(raw_gaps, list):
            raise DecompositionError("Поле gaps должно быть списком")
        result: list[dict[str, object]] = []
        for gap in raw_gaps:
            if not isinstance(gap, dict) or set(gap) != {
                "kind",
                "question",
                "target_paths",
            }:
                raise DecompositionError("Пробел имеет неверную структуру")
            kind = str(gap["kind"])
            question = str(gap["question"])
            target_paths = gap["target_paths"]
            if not kind.strip() or not question.strip():
                raise DecompositionError(
                    "Пробел требует kind и question"
                )
            if (
                not isinstance(target_paths, list)
                or not target_paths
                or any(not str(path).strip() for path in target_paths)
            ):
                raise DecompositionError(
                    "Пробел требует непустые target_paths"
                )
            result.append(
                {
                    "kind": kind,
                    "question": question,
                    "target_paths": [str(path) for path in target_paths],
                }
            )
        return result
