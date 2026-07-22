from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from .models import PromptCall, PromptId
from .policy import PromptPolicy


@dataclass(frozen=True, slots=True)
class RegressionCase:
    case_id: str
    prompt_id: PromptId
    rule_ids: tuple[str, ...]
    context: dict[str, Any]
    expected_tool: str
    required_arguments: tuple[str, ...]
    assertions: tuple[dict[str, Any], ...]
    scripted_response: dict[str, Any]


@dataclass(frozen=True, slots=True)
class RegressionResult:
    case_id: str
    status: str
    messages: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RegressionReport:
    mode: str
    policy_version: str
    results: tuple[RegressionResult, ...]

    @property
    def success(self) -> bool:
        return all(result.status == "passed" for result in self.results)


class PromptModel(Protocol):
    async def invoke(self, call: PromptCall, case: RegressionCase) -> dict[str, Any]: ...


class ScriptedPromptModel:
    def __init__(self, cases: list[RegressionCase]) -> None:
        self._responses = {case.case_id: case.scripted_response for case in cases}
        self.calls: list[PromptCall] = []

    async def invoke(self, call: PromptCall, case: RegressionCase) -> dict[str, Any]:
        self.calls.append(call)
        return self._responses[case.case_id]


class RegressionHarness:
    def __init__(self, policy: PromptPolicy, model: PromptModel) -> None:
        self.policy = policy
        self.model = model

    async def run(
        self,
        cases: list[RegressionCase],
        *,
        mode: str = "offline",
        output: Path | None = None,
    ) -> RegressionReport:
        results: list[RegressionResult] = []
        for case in cases:
            call = self.policy.build_call(case.prompt_id, case.context)
            response = await self.model.invoke(call, case)
            results.append(self._validate(case, response))
        report = RegressionReport(
            mode=mode,
            policy_version=self.policy.version,
            results=tuple(results),
        )
        if output is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(
                json.dumps(asdict(report), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        return report

    @staticmethod
    def _validate(case: RegressionCase, response: dict[str, Any]) -> RegressionResult:
        structural: list[str] = []
        if response.get("tool_name") != case.expected_tool:
            structural.append(f"Ожидался tool {case.expected_tool}")
        arguments = response.get("arguments")
        if not isinstance(arguments, dict):
            structural.append("arguments должен быть объектом")
            arguments = {}
        missing = [name for name in case.required_arguments if name not in arguments]
        if missing:
            structural.append(f"Не хватает arguments: {missing}")
        if structural:
            return RegressionResult(case.case_id, "structural_failure", tuple(structural))

        semantic: list[str] = []
        for assertion in case.assertions:
            actual = _read_path(response, str(assertion["path"]))
            if "equals" in assertion and actual != assertion["equals"]:
                semantic.append(
                    f"{assertion['path']}: ожидалось {assertion['equals']!r}, получено {actual!r}"
                )
            if "not_contains" in assertion and assertion["not_contains"] in (actual or []):
                semantic.append(f"{assertion['path']} содержит запрещённое значение")
        return RegressionResult(
            case.case_id,
            "semantic_failure" if semantic else "passed",
            tuple(semantic),
        )


def _read_path(value: dict[str, Any], path: str) -> Any:
    current: Any = value
    for part in path.split("."):
        if isinstance(current, dict):
            if part not in current:
                return None
            current = current[part]
            continue
        if isinstance(current, list) and part.isdigit():
            index = int(part)
            if index >= len(current):
                return None
            current = current[index]
            continue
        return None
    return current


def load_regression_cases(path: Path | None = None) -> list[RegressionCase]:
    source = path or Path(__file__).with_name("fixtures") / "cases.json"
    raw = json.loads(source.read_text(encoding="utf-8"))
    return [
        RegressionCase(
            case_id=item["case_id"],
            prompt_id=PromptId(item["prompt_id"]),
            rule_ids=tuple(item["rule_ids"]),
            context=dict(item["context"]),
            expected_tool=item["expected_tool"],
            required_arguments=tuple(item["required_arguments"]),
            assertions=tuple(item.get("assertions", [])),
            scripted_response=dict(item["scripted_response"]),
        )
        for item in raw
    ]


async def run_live_regression(
    policy: PromptPolicy,
    model: PromptModel,
    cases: list[RegressionCase],
    *,
    output: Path,
) -> RegressionReport:
    """Live-прогон возможен только явным вызовом с переданным сетевым клиентом."""

    return await RegressionHarness(policy, model).run(cases, mode="live", output=output)
