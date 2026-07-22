from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from ...domain.source import SourcePosition
from .models import (
    AnalystDecision,
    CorpusValidationError,
    EvaluationRange,
    EvaluationSelection,
    EvaluationSourceLine,
    ExpectedLineAssessment,
    ExpectedObligation,
    ForbiddenClaim,
    QualityCase,
    QualityCorpus,
)


class QualityCorpusCodec:
    SCHEMA_VERSION = 1
    CASE_KEYS = {
        "case_id",
        "source_hash",
        "reason",
        "comparison_selection",
        "baseline_selection",
        "free_selection",
        "expected_outcome",
        "expected_obligations",
        "forbidden_claims",
        "allowed_groupings",
        "line_expectations",
        "analyst_decision",
    }

    @classmethod
    def load_file(cls, path: Path) -> QualityCorpus:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise CorpusValidationError(
                f"Не удалось прочитать corpus {path}: {error}"
            ) from error
        return cls.load(_mapping(value, "corpus"))

    @classmethod
    def load(cls, value: Mapping[str, object]) -> QualityCorpus:
        root = _mapping(value, "corpus")
        _exact_keys(root, {"schema_version", "cases"}, "corpus")
        schema_version = _integer(root["schema_version"], "schema_version")
        if schema_version != cls.SCHEMA_VERSION:
            raise CorpusValidationError(
                f"Неподдерживаемый schema_version: {schema_version}"
            )
        raw_cases = _sequence(root["cases"], "cases")
        cases = tuple(
            cls._case(_mapping(item, f"cases.{index}"), index)
            for index, item in enumerate(raw_cases)
        )
        case_ids = [case.case_id for case in cases]
        if len(case_ids) != len(set(case_ids)):
            raise CorpusValidationError("case_id должен быть уникальным")
        canonical = cls.dump_payload(schema_version, cases)
        version = "sha256:" + hashlib.sha256(
            json.dumps(
                canonical,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        return QualityCorpus(schema_version, version, cases)

    @classmethod
    def dump_payload(
        cls,
        schema_version: int,
        cases: tuple[QualityCase, ...],
    ) -> dict[str, object]:
        return {
            "schema_version": schema_version,
            "cases": [cls._case_payload(case) for case in cases],
        }

    @classmethod
    def _case(
        cls,
        value: Mapping[str, object],
        index: int,
    ) -> QualityCase:
        path = f"cases.{index}"
        _exact_keys(value, cls.CASE_KEYS, path)
        source_hash = _text(value["source_hash"], f"{path}.source_hash")
        if len(source_hash) != 64:
            raise CorpusValidationError(
                f"{path}.source_hash должен быть SHA-256"
            )
        try:
            int(source_hash, 16)
        except ValueError as error:
            raise CorpusValidationError(
                f"{path}.source_hash должен быть SHA-256"
            ) from error
        comparison = cls._selection(
            _mapping(value["comparison_selection"], f"{path}.comparison_selection"),
            f"{path}.comparison_selection",
        )
        baseline = cls._selection(
            _mapping(value["baseline_selection"], f"{path}.baseline_selection"),
            f"{path}.baseline_selection",
        )
        free = cls._selection(
            _mapping(value["free_selection"], f"{path}.free_selection"),
            f"{path}.free_selection",
        )
        outcome = _text(value["expected_outcome"], f"{path}.expected_outcome")
        if outcome not in {
            "skeletons_created",
            "no_testable_behavior",
            "insufficient_selection",
        }:
            raise CorpusValidationError(f"{path}.expected_outcome неизвестен")
        obligations = tuple(
            cls._obligation(
                _mapping(item, f"{path}.expected_obligations.{item_index}"),
                f"{path}.expected_obligations.{item_index}",
            )
            for item_index, item in enumerate(
                _sequence(
                    value["expected_obligations"],
                    f"{path}.expected_obligations",
                )
            )
        )
        if not obligations and outcome != "no_testable_behavior":
            raise CorpusValidationError(
                f"{path}.expected_obligations не может быть пустым "
                f"для {outcome}"
            )
        obligation_ids = [item.obligation_id for item in obligations]
        if len(obligation_ids) != len(set(obligation_ids)):
            raise CorpusValidationError(
                f"{path}.expected_obligations содержит дубли ID"
            )
        source_positions = {
            (line.page, line.line) for line in free.lines
        }
        for obligation in obligations:
            for evidence in obligation.evidence:
                positions = {
                    (evidence.page, line)
                    for line in range(
                        evidence.line_start,
                        evidence.line_end + 1,
                    )
                }
                if not positions <= source_positions:
                    raise CorpusValidationError(
                        f"{path}.expected_obligations evidence выходит "
                        "за free_selection"
                    )
        forbidden_claims = tuple(
            cls._forbidden_claim(
                _mapping(item, f"{path}.forbidden_claims.{item_index}"),
                f"{path}.forbidden_claims.{item_index}",
            )
            for item_index, item in enumerate(
                _sequence(value["forbidden_claims"], f"{path}.forbidden_claims")
            )
        )
        groupings = tuple(
            tuple(
                _text(item, f"{path}.allowed_groupings.{group_index}")
                for item in _sequence(
                    group,
                    f"{path}.allowed_groupings.{group_index}",
                )
            )
            for group_index, group in enumerate(
                _sequence(value["allowed_groupings"], f"{path}.allowed_groupings")
            )
        )
        grouped_ids = [item for group in groupings for item in group]
        if sorted(grouped_ids) != sorted(obligation_ids):
            raise CorpusValidationError(
                f"{path}.allowed_groupings должен покрывать obligations ровно один раз"
            )
        line_expectations = tuple(
            cls._line_expectation(
                _mapping(item, f"{path}.line_expectations.{item_index}"),
                f"{path}.line_expectations.{item_index}",
            )
            for item_index, item in enumerate(
                _sequence(
                    value["line_expectations"],
                    f"{path}.line_expectations",
                )
            )
        )
        expected_positions = {
            (item.page, item.line) for item in free.lines
        }
        actual_positions = {
            (item.page, item.line) for item in line_expectations
        }
        if (
            expected_positions != actual_positions
            or len(actual_positions) != len(line_expectations)
        ):
            raise CorpusValidationError(
                f"{path}.line_expectations должен классифицировать "
                "каждую строку free_selection ровно один раз"
            )
        decision_value = _mapping(
            value["analyst_decision"],
            f"{path}.analyst_decision",
        )
        _exact_keys(
            decision_value,
            {"version", "status", "comment"},
            f"{path}.analyst_decision",
        )
        decision = AnalystDecision(
            _positive_integer(
                decision_value["version"],
                f"{path}.analyst_decision.version",
            ),
            _text(
                decision_value["status"],
                f"{path}.analyst_decision.status",
            ),
            _text(
                decision_value["comment"],
                f"{path}.analyst_decision.comment",
                allow_empty=True,
            ),
        )
        if decision.status != "approved":
            raise CorpusValidationError(
                f"{path}.analyst_decision.status должен быть approved"
            )
        return QualityCase(
            case_id=_text(value["case_id"], f"{path}.case_id"),
            source_hash=source_hash,
            reason=_text(value["reason"], f"{path}.reason"),
            comparison_selection=comparison,
            baseline_selection=baseline,
            free_selection=free,
            expected_outcome=outcome,
            expected_obligations=obligations,
            forbidden_claims=forbidden_claims,
            allowed_groupings=groupings,
            line_expectations=line_expectations,
            analyst_decision=decision,
        )

    @classmethod
    def _selection(
        cls,
        value: Mapping[str, object],
        path: str,
    ) -> EvaluationSelection:
        _exact_keys(value, {"start", "end", "lines"}, path)
        start = cls._position(_mapping(value["start"], f"{path}.start"), f"{path}.start")
        end = cls._position(_mapping(value["end"], f"{path}.end"), f"{path}.end")
        lines = tuple(
            cls._line(
                _mapping(item, f"{path}.lines.{index}"),
                f"{path}.lines.{index}",
            )
            for index, item in enumerate(
                _sequence(value["lines"], f"{path}.lines")
            )
        )
        if not lines:
            raise CorpusValidationError(f"{path}.lines не может быть пустым")
        positions = tuple(item.position for item in lines)
        if positions[0] != start or positions[-1] != end:
            raise CorpusValidationError(
                f"{path}.start/end не совпадают с lines"
            )
        if tuple(sorted(positions)) != positions or len(set(positions)) != len(
            positions
        ):
            raise CorpusValidationError(
                f"{path}.lines должен иметь уникальный стабильный порядок"
            )
        return EvaluationSelection(start, end, lines)

    @staticmethod
    def _position(value: Mapping[str, object], path: str) -> SourcePosition:
        _exact_keys(value, {"page", "line"}, path)
        return SourcePosition(
            _positive_integer(value["page"], f"{path}.page"),
            _positive_integer(value["line"], f"{path}.line"),
        )

    @staticmethod
    def _line(value: Mapping[str, object], path: str) -> EvaluationSourceLine:
        _exact_keys(value, {"page", "line", "text"}, path)
        return EvaluationSourceLine(
            _positive_integer(value["page"], f"{path}.page"),
            _positive_integer(value["line"], f"{path}.line"),
            _text(value["text"], f"{path}.text", allow_empty=True),
        )

    @staticmethod
    def _obligation(
        value: Mapping[str, object],
        path: str,
    ) -> ExpectedObligation:
        _exact_keys(
            value,
            {"obligation_id", "description", "evidence", "context_loss"},
            path,
        )
        evidence = tuple(
            QualityCorpusCodec._range(
                _mapping(item, f"{path}.evidence.{index}"),
                f"{path}.evidence.{index}",
            )
            for index, item in enumerate(
                _sequence(value["evidence"], f"{path}.evidence")
            )
        )
        if not evidence:
            raise CorpusValidationError(f"{path}.evidence не может быть пустым")
        context_loss = value["context_loss"]
        if not isinstance(context_loss, bool):
            raise CorpusValidationError(f"{path}.context_loss должен быть bool")
        return ExpectedObligation(
            _text(value["obligation_id"], f"{path}.obligation_id"),
            _text(value["description"], f"{path}.description"),
            evidence,
            context_loss,
        )

    @staticmethod
    def _range(value: Mapping[str, object], path: str) -> EvaluationRange:
        _exact_keys(value, {"page", "line_start", "line_end"}, path)
        result = EvaluationRange(
            _positive_integer(value["page"], f"{path}.page"),
            _positive_integer(value["line_start"], f"{path}.line_start"),
            _positive_integer(value["line_end"], f"{path}.line_end"),
        )
        if result.line_end < result.line_start:
            raise CorpusValidationError(f"{path} имеет обратные границы")
        return result

    @staticmethod
    def _forbidden_claim(
        value: Mapping[str, object],
        path: str,
    ) -> ForbiddenClaim:
        _exact_keys(value, {"claim_id", "description"}, path)
        return ForbiddenClaim(
            _text(value["claim_id"], f"{path}.claim_id"),
            _text(value["description"], f"{path}.description"),
        )

    @staticmethod
    def _line_expectation(
        value: Mapping[str, object],
        path: str,
    ) -> ExpectedLineAssessment:
        _exact_keys(value, {"page", "line", "role"}, path)
        role = _text(value["role"], f"{path}.role")
        if role not in {"evidence", "context"}:
            raise CorpusValidationError(f"{path}.role неизвестен")
        return ExpectedLineAssessment(
            _positive_integer(value["page"], f"{path}.page"),
            _positive_integer(value["line"], f"{path}.line"),
            role,
        )

    @classmethod
    def _case_payload(cls, case: QualityCase) -> dict[str, object]:
        def selection(value: EvaluationSelection) -> dict[str, object]:
            return {
                "start": {"page": value.start.page_index, "line": value.start.line_number},
                "end": {"page": value.end.page_index, "line": value.end.line_number},
                "lines": [
                    {"page": line.page, "line": line.line, "text": line.text}
                    for line in value.lines
                ],
            }

        return {
            "case_id": case.case_id,
            "source_hash": case.source_hash,
            "reason": case.reason,
            "comparison_selection": selection(case.comparison_selection),
            "baseline_selection": selection(case.baseline_selection),
            "free_selection": selection(case.free_selection),
            "expected_outcome": case.expected_outcome,
            "expected_obligations": [
                {
                    "obligation_id": item.obligation_id,
                    "description": item.description,
                    "evidence": [
                        {
                            "page": evidence.page,
                            "line_start": evidence.line_start,
                            "line_end": evidence.line_end,
                        }
                        for evidence in item.evidence
                    ],
                    "context_loss": item.context_loss,
                }
                for item in case.expected_obligations
            ],
            "forbidden_claims": [
                {"claim_id": item.claim_id, "description": item.description}
                for item in case.forbidden_claims
            ],
            "allowed_groupings": [list(group) for group in case.allowed_groupings],
            "line_expectations": [
                {"page": item.page, "line": item.line, "role": item.role}
                for item in case.line_expectations
            ],
            "analyst_decision": {
                "version": case.analyst_decision.version,
                "status": case.analyst_decision.status,
                "comment": case.analyst_decision.comment,
            },
        }


def _mapping(value: object, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise CorpusValidationError(f"{path} должен быть object")
    return value


def _sequence(value: object, path: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise CorpusValidationError(f"{path} должен быть array")
    return value


def _exact_keys(
    value: Mapping[str, object],
    expected: set[str],
    path: str,
) -> None:
    missing = expected - set(value)
    extra = set(value) - expected
    if missing or extra:
        details: list[str] = []
        if missing:
            details.append("missing=" + ",".join(sorted(missing)))
        if extra:
            details.append("unexpected=" + ",".join(sorted(extra)))
        raise CorpusValidationError(f"{path}: {'; '.join(details)}")


def _text(value: object, path: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise CorpusValidationError(f"{path} должен быть непустой строкой")
    return value


def _integer(value: object, path: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise CorpusValidationError(f"{path} должен быть integer")
    return value


def _positive_integer(value: object, path: str) -> int:
    result = _integer(value, path)
    if result < 1:
        raise CorpusValidationError(f"{path} должен быть положительным")
    return result


__all__ = ["QualityCorpusCodec"]
