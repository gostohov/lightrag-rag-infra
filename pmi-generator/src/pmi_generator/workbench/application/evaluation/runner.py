from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from ...domain.source import SourcePosition
from .models import (
    EvaluationSelection,
    EvaluationSourceLine,
    QualityCase,
    QualityCorpus,
)


class ExperimentMode(str, Enum):
    WINDOWING = "experiment_a_windowing"
    SELECTION = "experiment_b_selection"


class ExperimentStage(str, Enum):
    PILOT = "pilot"
    FULL = "full"


@dataclass(frozen=True, slots=True)
class PinnedConfiguration:
    model_id: str
    server_configuration_json: str
    prompt_version: str
    policy_version: str
    sampling_json: str
    budgets_json: str
    repeats: int

    @classmethod
    def build(
        cls,
        *,
        model_id: str,
        server_configuration: dict[str, object],
        prompt_version: str,
        policy_version: str,
        sampling: dict[str, object],
        budgets: dict[str, object],
        repeats: int,
    ) -> PinnedConfiguration:
        if not model_id.strip() or not prompt_version.strip() or not policy_version.strip():
            raise PairValidationError("Pinned configuration содержит пустую version")
        if repeats < 1:
            raise PairValidationError("repeats должен быть положительным")
        return cls(
            model_id,
            _canonical_json(server_configuration),
            prompt_version,
            policy_version,
            _canonical_json(sampling),
            _canonical_json(budgets),
            repeats,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "server_configuration": json.loads(self.server_configuration_json),
            "prompt_version": self.prompt_version,
            "policy_version": self.policy_version,
            "sampling": json.loads(self.sampling_json),
            "budgets": json.loads(self.budgets_json),
            "repeats": self.repeats,
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> PinnedConfiguration:
        return cls.build(
            model_id=str(value["model_id"]),
            server_configuration=dict(value["server_configuration"]),  # type: ignore[arg-type]
            prompt_version=str(value["prompt_version"]),
            policy_version=str(value["policy_version"]),
            sampling=dict(value["sampling"]),  # type: ignore[arg-type]
            budgets=dict(value["budgets"]),  # type: ignore[arg-type]
            repeats=int(value["repeats"]),
        )


@dataclass(frozen=True, slots=True)
class ExperimentVariant:
    role: str
    selection: EvaluationSelection
    prompt_1_mode: str
    configuration: PinnedConfiguration
    tool_schema_version: str
    workflow_version: str

    @property
    def selection_fingerprint(self) -> str:
        return _fingerprint(_selection_payload(self.selection))

    def to_dict(self) -> dict[str, object]:
        return {
            "role": self.role,
            "selection": _selection_payload(self.selection),
            "selection_fingerprint": self.selection_fingerprint,
            "prompt_1_mode": self.prompt_1_mode,
            "configuration": self.configuration.to_dict(),
            "tool_schema_version": self.tool_schema_version,
            "workflow_version": self.workflow_version,
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> ExperimentVariant:
        raw_selection = dict(value["selection"])  # type: ignore[arg-type]
        raw_start = dict(raw_selection["start"])  # type: ignore[arg-type]
        raw_end = dict(raw_selection["end"])  # type: ignore[arg-type]
        selection = EvaluationSelection(
            SourcePosition(int(raw_start["page"]), int(raw_start["line"])),
            SourcePosition(int(raw_end["page"]), int(raw_end["line"])),
            tuple(
                EvaluationSourceLine(
                    int(item["page"]),
                    int(item["line"]),
                    str(item["text"]),
                )
                for item in raw_selection["lines"]  # type: ignore[union-attr]
            ),
        )
        result = cls(
            role=str(value["role"]),
            selection=selection,
            prompt_1_mode=str(value["prompt_1_mode"]),
            configuration=PinnedConfiguration.from_dict(
                dict(value["configuration"])  # type: ignore[arg-type]
            ),
            tool_schema_version=str(value["tool_schema_version"]),
            workflow_version=str(value["workflow_version"]),
        )
        if value.get("selection_fingerprint") != result.selection_fingerprint:
            raise PairValidationError("Manifest selection fingerprint повреждён")
        return result


@dataclass(frozen=True, slots=True)
class ExperimentPair:
    pair_id: str
    mode: ExperimentMode
    case_id: str
    source_hash: str
    baseline: ExperimentVariant
    candidate: ExperimentVariant

    def to_dict(self) -> dict[str, object]:
        return {
            "pair_id": self.pair_id,
            "mode": self.mode.value,
            "case_id": self.case_id,
            "source_hash": self.source_hash,
            "baseline": self.baseline.to_dict(),
            "candidate": self.candidate.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> ExperimentPair:
        return cls(
            pair_id=str(value["pair_id"]),
            mode=ExperimentMode(str(value["mode"])),
            case_id=str(value["case_id"]),
            source_hash=str(value["source_hash"]),
            baseline=ExperimentVariant.from_dict(
                dict(value["baseline"])  # type: ignore[arg-type]
            ),
            candidate=ExperimentVariant.from_dict(
                dict(value["candidate"])  # type: ignore[arg-type]
            ),
        )


@dataclass(frozen=True, slots=True)
class ExperimentManifest:
    experiment_id: str
    schema_version: int
    stage: ExperimentStage
    corpus_version: str
    pairs: tuple[ExperimentPair, ...]
    pilot_experiment_id: str | None = None
    thresholds: SemanticThresholds | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "experiment_id": self.experiment_id,
            "schema_version": self.schema_version,
            "stage": self.stage.value,
            "corpus_version": self.corpus_version,
            "pairs": [pair.to_dict() for pair in self.pairs],
            "pilot_experiment_id": self.pilot_experiment_id,
            "thresholds": (
                self.thresholds.to_dict()
                if self.thresholds is not None
                else None
            ),
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> ExperimentManifest:
        thresholds = (
            SemanticThresholds.from_dict(dict(value["thresholds"]))  # type: ignore[arg-type]
            if value.get("thresholds") is not None
            else None
        )
        result = cls(
            experiment_id=str(value["experiment_id"]),
            schema_version=int(value["schema_version"]),
            stage=ExperimentStage(str(value["stage"])),
            corpus_version=str(value["corpus_version"]),
            pairs=tuple(
                ExperimentPair.from_dict(dict(item))
                for item in value["pairs"]  # type: ignore[union-attr]
            ),
            pilot_experiment_id=(
                str(value["pilot_experiment_id"])
                if value.get("pilot_experiment_id") is not None
                else None
            ),
            thresholds=thresholds,
        )
        payload = result.to_dict()
        payload.pop("experiment_id")
        expected_id = "EXPERIMENT_" + _fingerprint(payload)[:20].upper()
        if expected_id != result.experiment_id:
            raise PairValidationError("Manifest experiment ID повреждён")
        return result


@dataclass(frozen=True, slots=True)
class SemanticThresholds:
    minimum_semantic_recall: float
    maximum_variability: float
    maximum_missing: int
    maximum_merges: int
    maximum_splits: int
    maximum_duplicates: int
    maximum_ungrounded: int

    def __post_init__(self) -> None:
        if not 0 <= self.minimum_semantic_recall <= 1:
            raise PairValidationError("minimum_semantic_recall вне [0, 1]")
        if not 0 <= self.maximum_variability <= 1:
            raise PairValidationError("maximum_variability вне [0, 1]")
        counts = (
            self.maximum_missing,
            self.maximum_merges,
            self.maximum_splits,
            self.maximum_duplicates,
            self.maximum_ungrounded,
        )
        if any(value < 0 for value in counts):
            raise PairValidationError("Semantic count threshold не может быть отрицательным")

    @classmethod
    def strict_example(cls) -> SemanticThresholds:
        return cls(1.0, 0.0, 0, 0, 0, 0, 0)

    def to_dict(self) -> dict[str, object]:
        return {
            "minimum_semantic_recall": self.minimum_semantic_recall,
            "maximum_variability": self.maximum_variability,
            "maximum_missing": self.maximum_missing,
            "maximum_merges": self.maximum_merges,
            "maximum_splits": self.maximum_splits,
            "maximum_duplicates": self.maximum_duplicates,
            "maximum_ungrounded": self.maximum_ungrounded,
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> SemanticThresholds:
        return cls(
            minimum_semantic_recall=float(value["minimum_semantic_recall"]),
            maximum_variability=float(value["maximum_variability"]),
            maximum_missing=int(value["maximum_missing"]),
            maximum_merges=int(value["maximum_merges"]),
            maximum_splits=int(value["maximum_splits"]),
            maximum_duplicates=int(value["maximum_duplicates"]),
            maximum_ungrounded=int(value["maximum_ungrounded"]),
        )


@dataclass(frozen=True, slots=True)
class PilotReceipt:
    pilot_experiment_id: str
    run_ids: tuple[str, ...]
    successful: bool


@dataclass(frozen=True, slots=True)
class RunRecord:
    run_id: str
    pair_id: str
    variant: str
    repeat: int
    status: str
    diagnostic_path: str
    error: str | None

    def __post_init__(self) -> None:
        if self.variant not in {"baseline", "candidate"}:
            raise PairValidationError("variant должен быть baseline/candidate")
        if self.status not in {"valid", "invalid"}:
            raise PairValidationError("status должен быть valid/invalid")
        if self.repeat < 1:
            raise PairValidationError("repeat должен быть положительным")
        if self.status == "invalid" and not self.error:
            raise PairValidationError("invalid run обязан сохранить error")

    def to_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "pair_id": self.pair_id,
            "variant": self.variant,
            "repeat": self.repeat,
            "status": self.status,
            "diagnostic_path": self.diagnostic_path,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> RunRecord:
        return cls(
            run_id=str(value["run_id"]),
            pair_id=str(value["pair_id"]),
            variant=str(value["variant"]),
            repeat=int(value["repeat"]),
            status=str(value["status"]),
            diagnostic_path=str(value["diagnostic_path"]),
            error=(
                str(value["error"])
                if value.get("error") is not None
                else None
            ),
        )


class PairValidationError(ValueError):
    pass


class PairedExperimentRunner:
    MANIFEST_SCHEMA_VERSION = 1

    def __init__(self, corpus: QualityCorpus) -> None:
        self.corpus = corpus

    def default_pair(
        self,
        mode: ExperimentMode,
        case: QualityCase,
        configuration: PinnedConfiguration,
    ) -> ExperimentPair:
        if mode is ExperimentMode.WINDOWING:
            return self.build_pair(
                mode=mode,
                case=case,
                baseline_selection=case.comparison_selection,
                candidate_selection=case.comparison_selection,
                baseline_mode="single_call",
                candidate_mode="windowed",
                baseline_configuration=configuration,
                candidate_configuration=configuration,
                baseline_tool_schema="submit_decomposition",
                candidate_tool_schema=(
                    "submit_semantic_window_result+"
                    "submit_reconciliation_case"
                ),
                baseline_workflow="prompt_1_single",
                candidate_workflow="prompt_1_windowed",
            )
        return self.build_pair(
            mode=mode,
            case=case,
            baseline_selection=case.baseline_selection,
            candidate_selection=case.free_selection,
            baseline_mode="single_call",
            candidate_mode="single_call",
            baseline_configuration=configuration,
            candidate_configuration=configuration,
            baseline_tool_schema="submit_decomposition",
            candidate_tool_schema="submit_decomposition",
            baseline_workflow="prompt_1_single",
            candidate_workflow="prompt_1_single",
        )

    def build_pair(
        self,
        *,
        mode: ExperimentMode,
        case: QualityCase,
        baseline_selection: EvaluationSelection,
        candidate_selection: EvaluationSelection,
        baseline_mode: str,
        candidate_mode: str,
        baseline_configuration: PinnedConfiguration,
        candidate_configuration: PinnedConfiguration,
        baseline_tool_schema: str,
        candidate_tool_schema: str,
        baseline_workflow: str,
        candidate_workflow: str,
    ) -> ExperimentPair:
        baseline = ExperimentVariant(
            "baseline",
            baseline_selection,
            baseline_mode,
            baseline_configuration,
            baseline_tool_schema,
            baseline_workflow,
        )
        candidate = ExperimentVariant(
            "candidate",
            candidate_selection,
            candidate_mode,
            candidate_configuration,
            candidate_tool_schema,
            candidate_workflow,
        )
        self._validate_pair(mode, baseline, candidate)
        payload = {
            "mode": mode.value,
            "case_id": case.case_id,
            "source_hash": case.source_hash,
            "baseline": baseline.to_dict(),
            "candidate": candidate.to_dict(),
        }
        return ExperimentPair(
            "PAIR_" + _fingerprint(payload)[:16].upper(),
            mode,
            case.case_id,
            case.source_hash,
            baseline,
            candidate,
        )

    def manifest(
        self,
        stage: ExperimentStage,
        pairs: tuple[ExperimentPair, ...],
    ) -> ExperimentManifest:
        if not pairs:
            raise PairValidationError("Experiment manifest не может быть пустым")
        ordered = tuple(sorted(pairs, key=lambda item: item.pair_id))
        if len({pair.pair_id for pair in ordered}) != len(ordered):
            raise PairValidationError("Experiment manifest содержит duplicate pair")
        payload = {
            "schema_version": self.MANIFEST_SCHEMA_VERSION,
            "stage": stage.value,
            "corpus_version": self.corpus.version,
            "pairs": [pair.to_dict() for pair in ordered],
            "pilot_experiment_id": None,
            "thresholds": None,
        }
        experiment_id = "EXPERIMENT_" + _fingerprint(payload)[:20].upper()
        return ExperimentManifest(
            experiment_id,
            self.MANIFEST_SCHEMA_VERSION,
            stage,
            self.corpus.version,
            ordered,
        )

    def full_manifest(
        self,
        pairs: tuple[ExperimentPair, ...],
        *,
        pilot_receipt: PilotReceipt,
        thresholds: SemanticThresholds,
    ) -> ExperimentManifest:
        if not self.corpus.ready_for_full_run:
            raise PairValidationError(
                "Full experiment требует минимум 12 corpus cases"
            )
        if not pilot_receipt.successful:
            raise PairValidationError(
                "Full experiment требует успешный pilot"
            )
        ordered = tuple(sorted(pairs, key=lambda item: item.pair_id))
        modes = {pair.mode for pair in ordered}
        case_ids = {pair.case_id for pair in ordered}
        if len(modes) != 1:
            raise PairValidationError(
                "Full Experiment A и B запускаются раздельно"
            )
        if len(case_ids) < self.corpus.minimum_full_run_cases:
            raise PairValidationError(
                "Full experiment требует минимум 12 разных cases"
            )
        payload = {
            "schema_version": self.MANIFEST_SCHEMA_VERSION,
            "stage": ExperimentStage.FULL.value,
            "corpus_version": self.corpus.version,
            "pairs": [pair.to_dict() for pair in ordered],
            "pilot_experiment_id": pilot_receipt.pilot_experiment_id,
            "thresholds": thresholds.to_dict(),
        }
        return ExperimentManifest(
            experiment_id="EXPERIMENT_" + _fingerprint(payload)[:20].upper(),
            schema_version=self.MANIFEST_SCHEMA_VERSION,
            stage=ExperimentStage.FULL,
            corpus_version=self.corpus.version,
            pairs=ordered,
            pilot_experiment_id=pilot_receipt.pilot_experiment_id,
            thresholds=thresholds,
        )

    @staticmethod
    def _validate_pair(
        mode: ExperimentMode,
        baseline: ExperimentVariant,
        candidate: ExperimentVariant,
    ) -> None:
        if baseline.configuration != candidate.configuration:
            raise PairValidationError(
                f"{mode.value}: undeclared configuration difference"
            )
        if mode is ExperimentMode.WINDOWING:
            if baseline.selection != candidate.selection:
                raise PairValidationError(
                    "Experiment A требует одинаковый selection"
                )
            if (
                baseline.prompt_1_mode != "single_call"
                or candidate.prompt_1_mode != "windowed"
            ):
                raise PairValidationError(
                    "Experiment A требует single_call/windowed"
                )
            return
        if baseline.prompt_1_mode != candidate.prompt_1_mode:
            raise PairValidationError(
                "Experiment B требует одинаковый Prompt 1 mode"
            )
        if (
            baseline.tool_schema_version != candidate.tool_schema_version
            or baseline.workflow_version != candidate.workflow_version
        ):
            raise PairValidationError(
                "Experiment B содержит undeclared workflow/schema difference"
            )


class ExperimentStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    def create(self, manifest: ExperimentManifest) -> Path:
        directory = self.root / manifest.experiment_id
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "raw").mkdir(exist_ok=True)
        (directory / "runs").mkdir(exist_ok=True)
        path = directory / "manifest.json"
        content = _pretty_json(manifest.to_dict())
        if path.exists():
            if path.read_text(encoding="utf-8") != content:
                raise PairValidationError(
                    "Experiment ID уже связан с другим manifest"
                )
        else:
            path.write_text(content, encoding="utf-8")
        return directory

    def record(
        self,
        manifest: ExperimentManifest,
        record: RunRecord,
        raw_diagnostic: dict[str, object],
    ) -> None:
        if record.pair_id not in {pair.pair_id for pair in manifest.pairs}:
            raise PairValidationError("Run ссылается на неизвестную pair")
        expected_diagnostic = f"raw/{record.run_id}.json"
        if record.diagnostic_path != expected_diagnostic:
            raise PairValidationError(
                f"diagnostic_path должен быть {expected_diagnostic}"
            )
        directory = self.create(manifest)
        raw_path = directory / expected_diagnostic
        record_path = directory / "runs" / f"{record.run_id}.json"
        try:
            with raw_path.open("x", encoding="utf-8") as stream:
                stream.write(_pretty_json(raw_diagnostic))
            with record_path.open("x", encoding="utf-8") as stream:
                stream.write(_pretty_json(record.to_dict()))
        except FileExistsError as error:
            raise PairValidationError(
                f"Run {record.run_id} уже сохранён"
            ) from error

    def records(
        self,
        manifest: ExperimentManifest,
    ) -> tuple[RunRecord, ...]:
        directory = self.root / manifest.experiment_id / "runs"
        if not directory.exists():
            return ()
        return tuple(
            RunRecord.from_dict(json.loads(path.read_text(encoding="utf-8")))
            for path in sorted(directory.glob("*.json"))
        )

    def load_manifest(self, experiment_id: str) -> ExperimentManifest:
        path = self.root / experiment_id / "manifest.json"
        if not path.is_file():
            raise PairValidationError(
                f"Manifest {experiment_id} не найден"
            )
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            return ExperimentManifest.from_dict(dict(value))
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            if isinstance(error, PairValidationError):
                raise
            raise PairValidationError(
                f"Manifest {experiment_id} невалиден: {error}"
            ) from error

    def pilot_receipt(
        self,
        manifest: ExperimentManifest,
        evaluations: dict[str, object],
    ) -> PilotReceipt:
        if manifest.stage is not ExperimentStage.PILOT:
            raise PairValidationError("Pilot receipt требует pilot manifest")
        modes = [pair.mode for pair in manifest.pairs]
        if modes.count(ExperimentMode.WINDOWING) != 1 or modes.count(
            ExperimentMode.SELECTION
        ) != 1:
            return PilotReceipt(manifest.experiment_id, (), False)
        records = self.records(manifest)
        expected = {
            (pair.pair_id, variant, 1)
            for pair in manifest.pairs
            for variant in ("baseline", "candidate")
        }
        actual = {
            (record.pair_id, record.variant, record.repeat)
            for record in records
        }
        successful = (
            len(records) == 4
            and actual == expected
            and all(record.status == "valid" for record in records)
            and set(evaluations)
            == {record.run_id for record in records}
            and all(
                bool(
                    getattr(
                        getattr(evaluation, "hard_gates", None),
                        "passed",
                        False,
                    )
                )
                for evaluation in evaluations.values()
            )
        )
        return PilotReceipt(
            manifest.experiment_id,
            tuple(sorted(record.run_id for record in records)),
            successful,
        )


def _selection_payload(selection: EvaluationSelection) -> dict[str, object]:
    return {
        "start": {
            "page": selection.start.page_index,
            "line": selection.start.line_number,
        },
        "end": {
            "page": selection.end.page_index,
            "line": selection.end.line_number,
        },
        "lines": [
            {"page": item.page, "line": item.line, "text": item.text}
            for item in selection.lines
        ],
    }


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _pretty_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


__all__ = [
    "ExperimentManifest",
    "ExperimentMode",
    "ExperimentPair",
    "ExperimentStage",
    "ExperimentStore",
    "ExperimentVariant",
    "PairValidationError",
    "PairedExperimentRunner",
    "PinnedConfiguration",
    "PilotReceipt",
    "RunRecord",
    "SemanticThresholds",
]
