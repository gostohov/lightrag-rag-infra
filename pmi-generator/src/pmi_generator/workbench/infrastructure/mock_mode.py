from __future__ import annotations

import asyncio
import hashlib
import json
from typing import Any

from ..application.gap_investigation import (
    RetrievalFragment,
    RetrievalProfile,
    RetrievalResponse,
)
from ..application.llm import RawCompletion
from ..application.prompting import PromptCall, PromptId
from ..application.source import SavedSelection
from ..domain import SourceDocument


class MockLlmTransport:
    def __init__(self, *, delay: float = 1.0) -> None:
        self.delay = max(0.0, delay)
        self.calls: list[PromptCall] = []

    async def complete(
        self,
        call: PromptCall,
        tools: list[dict[str, Any]],
    ) -> RawCompletion:
        self.calls.append(call)
        if self.delay:
            await asyncio.sleep(self.delay)
        if not tools:
            raise RuntimeError("Mock Workbench prompt не содержит terminal tools")
        name, arguments = self._response(call)
        available = {
            str(item.get("function", {}).get("name", ""))
            for item in tools
        }
        if name not in call.allowed_tools or name not in available:
            raise RuntimeError(
                f"Mock responder выбрал недоступный tool {name} для {call.prompt_id.value}"
            )
        fingerprint = hashlib.sha256(
            json.dumps(
                {
                    "prompt_id": call.prompt_id.value,
                    "context": call.context,
                    "tool": name,
                },
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            ).encode("utf-8")
        ).hexdigest()[:16]
        return RawCompletion(
            finish_reason="tool_calls",
            tool_calls=(
                {
                    "id": f"mock-{call.prompt_id.value}-{fingerprint}",
                    "name": name,
                    "arguments": arguments,
                },
            ),
            usage={"mock": True},
            model="pmi-workbench-mock",
        )

    def public_metadata(self) -> dict[str, Any]:
        return {"model": "pmi-workbench-mock"}

    def _response(self, call: PromptCall) -> tuple[str, dict[str, Any]]:
        responders = {
            PromptId.DECOMPOSITION: self._decomposition,
            PromptId.DECOMPOSITION_WINDOW_SEMANTIC: (
                self._window_decomposition
            ),
            PromptId.DECOMPOSITION_SEMANTIC_SYNTHESIS: self._synthesis,
            PromptId.DECOMPOSITION_RECONCILIATION: self._reconciliation,
            PromptId.POPULATION: self._population,
            PromptId.GAP_RESEARCH: self._gap_research,
            PromptId.REFINEMENT: self._refinement,
            PromptId.SELECTION_REVIEW: self._selection_review,
            PromptId.CONVERSATION: self._conversation,
        }
        return responders[call.prompt_id](call.context)

    @staticmethod
    def _window_decomposition(
        context: dict[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        window = context["window"]
        primary_lines = [
            item for item in window["lines"] if item["primary"]
        ]
        meaningful_input = [
            item
            for item in primary_lines
            if str(item.get("text") or "").strip()
        ]
        if not meaningful_input:
            return (
                "submit_semantic_window_result",
                {"behaviors": []},
            )
        meaningful = meaningful_input
        anchors = [meaningful[0]]
        if meaningful[-1]["text"] != meaningful[0]["text"]:
            anchors.append(meaningful[-1])

        def behavior(source_line: dict[str, Any]) -> dict[str, Any]:
            text = _compact(source_line["text"], limit=120)
            line_id = str(source_line["line_id"])
            return {
                "title": f"[mock] Проверка: {text}",
                "summary": f"[mock] Тестируемое поведение: {text}",
                "facts": [
                    {
                        "text": text,
                        "line_ids": [line_id],
                    },
                ],
            }

        return (
            "submit_semantic_window_result",
            {
                "behaviors": [
                    behavior(item)
                    for item in anchors
                ],
            },
        )

    @staticmethod
    def _synthesis(
        context: dict[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        synthesis = context["synthesis"]
        candidates = []
        for fragment in synthesis["target_fragments"]:
            fact_id = str(fragment["target_facts"][0]["fact_id"])
            summary = _compact(fragment["summary"], limit=120)
            candidates.append(
                {
                    "title": str(fragment["title"]),
                    "condition": {
                        "text": summary,
                        "fact_ids": [fact_id],
                    },
                    "changed_factor": {
                        "text": f"[mock] Проверяемый фактор: {summary}",
                        "fact_ids": [fact_id],
                    },
                    "input_value": {
                        "text": f"[mock] Контрольное значение: {summary}",
                        "fact_ids": [fact_id],
                    },
                    "action": {
                        "text": f"[mock] Выполнить проверку: {summary}",
                        "fact_ids": [fact_id],
                    },
                    "consequences": [
                        {
                            "text": f"[mock] Ожидается: {summary}",
                            "fact_ids": [fact_id],
                        },
                    ],
                }
            )
        return "submit_semantic_synthesis", {"candidates": candidates}

    @staticmethod
    def _reconciliation(
        context: dict[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        case = context["case"]
        decision = {
            "candidate_pair": "duplicate_keep_a",
            "candidate_review": "keep",
            "dependency_review": "resolved",
        }[case["case_kind"]]
        return (
            "submit_reconciliation_case",
            {
                "decision": decision,
                "reason": "[mock] Semantic case разрешён",
            },
        )

    @staticmethod
    def _conversation(
        context: dict[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        messages = context.get("messages") or []
        turn: dict[str, Any] = {}
        for item in reversed(messages):
            if not isinstance(item, dict) or item.get("type") != "human":
                continue
            try:
                candidate = json.loads(str(item.get("content") or ""))
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, dict) and "current_context" in candidate:
                turn = candidate
                break
        if not turn:
            raise RuntimeError("Mock conversation agent не получил current turn")
        user_text = str(turn.get("user_text") or "").strip()
        normalized = user_text.casefold()
        normalized_decision = normalized.rstrip(".!")
        current = turn.get("current_context") or {}
        available = set(current.get("available_actions") or [])
        gap = current.get("open_gap")
        pending = current.get("pending_analyst_answer")

        if (
            "confirm_analyst_answer" in available
            and pending
            and normalized_decision
            in {
                "да",
                "да, подтверждаю",
                "да, подтверждаю доработку",
                "да, подтверждаю эту доработку",
                "да, применяй именно это изменение",
                "подтверждаю",
                "применяй",
            }
        ):
            return (
                "confirm_analyst_answer",
                {
                    "announcement": (
                        "[mock] Применяю подтверждённое предложение."
                    )
                },
            )
        if (
            "reject_analyst_answer" in available
            and pending
            and normalized_decision
            in {
                "нет",
                "нет, отклоняю эту доработку",
                "отклоняю",
                "не применяй",
            }
        ):
            return (
                "reject_analyst_answer",
                {
                    "announcement": "[mock] Отклоняю предложение.",
                },
            )

        if (
            "propose_design_decision" in available
            and (
                "lightrag" in normalized
                or "проектн" in normalized
                or "поищи" in normalized
            )
        ):
            return (
                "propose_design_decision",
                {
                    "announcement": (
                        "[mock] Объясняю границу поиска и проектного решения."
                    ),
                },
            )
        if "?" in user_text or normalized.startswith(("почему", "как ")):
            return (
                "respond_to_analyst",
                {
                    "text": (
                        "[mock] Отвечаю по текущему состоянию карточки и trace. "
                        "Этот ответ не меняет карточку и не создаёт evidence."
                    )
                },
            )
        if "research_gap" in available and any(
            token in normalized
            for token in ("поищи", "найди", "проверь в источ")
        ):
            return (
                "research_gap",
                {
                    "question": user_text,
                    "announcement": f"[mock] Исследую: {user_text}",
                },
            )
        if "leave_gap" in available and any(
            token in normalized
            for token in ("оставь пробел", "оставить пробел")
        ):
            return (
                "leave_gap",
                {
                    "decision": "leave_open",
                    "reason": user_text,
                    "announcement": "[mock] Сохраняю явное решение оставить пробел.",
                },
            )
        if "resume" in available and normalized in {
            "продолжай",
            "продолжить",
            "продолжай.",
        }:
            return (
                "resume",
                {
                    "announcement": "[mock] Продолжаю сохранённую стадию.",
                },
            )
        if "submit_analyst_answer" in available and isinstance(gap, dict):
            return (
                "submit_analyst_answer",
                {
                    "values": [
                        {
                            "path": str(path),
                            "value": f"[mock] {user_text}",
                        }
                        for path in gap.get("allowed_paths") or []
                    ],
                    "announcement": "[mock] Применяю ответ аналитика к текущему gap.",
                },
            )
        if "refine_card" in available:
            return (
                "refine_card",
                {
                    "announcement": "[mock] Дорабатываю карточку.",
                },
            )
        return (
            "request_clarification",
            {
                "text": (
                    "[mock] Уточните, нужно объяснение, исследование или изменение "
                    "текущей карточки."
                )
            },
        )

    @staticmethod
    def _decomposition(context: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        selection = context["selection"]
        source_lines = _meaningful_source_lines(selection)

        def skeleton(source_line: dict[str, Any]) -> dict[str, Any]:
            page = int(source_line["page"])
            line = int(source_line["line"])
            text = _compact(source_line["text"], limit=120)
            evidence_range = {
                "page": page,
                "line_start": line,
                "line_end": line,
            }
            return {
                "title": f"[mock] Проверка: {text}",
                "condition": text,
                "changed_factor": f"[mock] Проверяемый фактор: {text}",
                "input_value": f"[mock] Контрольное значение для: {text}",
                "action": f"[mock] Проверить выполнение утверждения: {text}",
                "condition_ranges": [evidence_range],
                "changed_factor_ranges": [evidence_range],
                "input_value_ranges": [evidence_range],
                "action_ranges": [evidence_range],
                "consequences": [
                    {
                        "text": f"[mock] Ожидается соблюдение: {text}",
                        "evidence_ranges": [evidence_range],
                    }
                ],
                "gaps": [],
            }

        anchors = [source_lines[0]]
        if source_lines[-1]["text"] != source_lines[0]["text"]:
            anchors.append(source_lines[-1])
        evidence_positions = {
            (int(item["page"]), int(item["line"]))
            for item in anchors
        }
        return (
            "submit_decomposition",
            {
                "outcome": "skeletons_created",
                "explanation": (
                    "[mock] Каркасы построены из адресованных строк selection"
                ),
                "skeletons": [skeleton(item) for item in anchors],
                "line_assessments": [
                    {
                        "page": int(item["page"]),
                        "line": int(item["line"]),
                        "role": (
                            "evidence"
                            if (int(item["page"]), int(item["line"]))
                            in evidence_positions
                            else "context"
                        ),
                        "reason": (
                            "[mock] Строка использована каркасом"
                            if (int(item["page"]), int(item["line"]))
                            in evidence_positions
                            else "[mock] Промежуточный контекст selection"
                        ),
                    }
                    for item in context["selection"]["lines"]
                ],
            },
        )

    @staticmethod
    def _population(context: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        evidence = context.get("evidence") or []
        if not evidence:
            raise RuntimeError("Mock Prompt 2 не получил source evidence")
        evidence_id = str(evidence[0]["evidence_id"])
        skeleton = context.get("skeleton") or {}
        condition = _compact(
            skeleton.get("condition") or evidence[0].get("quote") or "",
            limit=240,
        )
        action = _compact(
            skeleton.get("action") or f"[mock] Проверить: {condition}",
            limit=240,
        )
        changed_factor = _compact(
            skeleton.get("changed_factor") or f"[mock] Фактор: {condition}",
            limit=240,
        )
        consequences = skeleton.get("consequences") or []
        consequence = (
            _compact(consequences[0].get("text"), limit=240)
            if consequences and isinstance(consequences[0], dict)
            else _compact(evidence[0].get("quote") or condition, limit=240)
        )

        def confirmed(path: str, value: str) -> dict[str, Any]:
            return {
                "path": path,
                "value": value,
                "evidence_id": evidence_id,
            }

        return (
            "submit_card_population",
            {
                "source_values": [
                    confirmed(
                        "requirement.condition",
                        condition,
                    ),
                    confirmed(
                        "requirement.behavior",
                        consequence,
                    ),
                    confirmed(
                        "test.action",
                        action,
                    ),
                    confirmed(
                        "test.changed_factor",
                        changed_factor,
                    ),
                    confirmed(
                        "test.expected.response_data",
                        consequence,
                    ),
                ],
                "derivations": [],
                "not_applicable": [],
                "gaps": [
                    {
                        "question": (
                            "[mock] Каким способом наблюдения подтвердить: "
                            f"{_compact(consequence, limit=180)}?"
                        ),
                        "blocking_reason": (
                            "[mock] Для результата не зафиксирован способ наблюдения"
                        ),
                        "allowed_paths": ["test.observation.method"],
                        "dependencies": [],
                        "closure_criterion": (
                            "[mock] Найден точный source fragment, применимый "
                            "как способ наблюдения"
                        ),
                        "resolution_mode": "source_fact",
                    }
                ],
            },
        )

    @staticmethod
    def _gap_research(context: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        observations = context.get("observations") or []
        gap = context["gap"]
        if not observations:
            research = context.get("research_question")
            research_text = (
                str(research.get("text") or "")
                if isinstance(research, dict)
                else ""
            )
            question = " ".join(
                (research_text or str(gap["question"])).split()
            )
            return (
                "ask_lightrag",
                {"question": f"{question} [mock]"},
            )
        observation = observations[-1]
        evidence_ids = observation.get("evidence_ids") or []
        if not evidence_ids:
            raise RuntimeError("Mock Prompt 3 не получил точный retrieval fragment")
        allowed_paths = gap.get("allowed_paths") or []
        if not allowed_paths:
            raise RuntimeError("Mock Prompt 3 не получил разрешённые пути gap")
        return (
            "submit_gap_result",
            {
                "outcome": "resolved",
                "updates": [
                    {
                        "path": str(allowed_paths[0]),
                        "value": (
                            "[mock] Способ наблюдения по результату retrieval: "
                            f"{_compact(observation.get('answer') or '', limit=240)}"
                        ),
                        "evidence_id": str(evidence_ids[0]),
                        "analyst_message_id": None,
                    }
                ],
                "unknown_fields": [],
                "missing_fact": None,
                "summary": (
                    "[mock] Пробел закрыт ответом retrieval: "
                    f"{_compact(observation.get('answer') or '', limit=240)}"
                ),
                "contradictions": [],
            },
        )

    @staticmethod
    def _refinement(context: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        message = context["message"]
        return (
            "submit_card_refinement",
            {
                "outcome": "updated",
                "updates": [
                    {
                        "path": "test.observation.method",
                        "value": f"[mock] {str(message['text']).strip()}",
                        "evidence_id": None,
                        "analyst_message_id": str(message["message_id"]),
                    }
                ],
                "gaps": [],
                "reason": "[mock] Доработка по сообщению аналитика",
            },
        )

    @staticmethod
    def _selection_review(
        context: dict[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        del context
        return (
            "submit_selection_review",
            {"outcome": "approved", "issues": []},
        )


class MockRetrieval:
    def __init__(
        self,
        document: SourceDocument,
        selection: SavedSelection,
        *,
        delay: float = 1.0,
    ) -> None:
        self.document = document
        self.selection = selection
        self.delay = max(0.0, delay)
        self.calls: list[tuple[str, RetrievalProfile]] = []

    async def query(
        self,
        question: str,
        profile: RetrievalProfile,
    ) -> RetrievalResponse:
        self.calls.append((question, profile))
        if self.delay:
            await asyncio.sleep(self.delay)
        candidates = [
            item
            for item in self.selection.selection.positions
            if self.document.line(item).strip()
        ] or [self.selection.selection.positions[0]]
        question_tokens = _tokens(question)
        position = max(
            candidates,
            key=lambda item: len(
                question_tokens & _tokens(self.document.line(item))
            ),
        )
        quote = self.document.line(position)
        return RetrievalResponse(
            answer=(
                f"[mock] Найден фрагмент стр. "
                f"{position.page_index}:{position.line_number}: {quote}"
            ),
            fragments=(
                RetrievalFragment(
                    document_id=self.document.metadata.document_id,
                    document_version=self.document.metadata.document_version,
                    page=position.page_index,
                    line_start=position.line_number,
                    line_end=position.line_number,
                    chunk_id=self.selection.section_id,
                    quote=quote,
                ),
            ),
        )


def _meaningful_source_lines(selection: dict[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in selection.get("lines") or []:
        if not isinstance(item, dict):
            continue
        text = _compact(item.get("text") or "", limit=500)
        normalized = text.casefold()
        if not text or normalized in seen:
            continue
        seen.add(normalized)
        result.append(
            {
                "page": int(item["page"]),
                "line": int(item["line"]),
                "text": text,
            }
        )
    if result:
        return result
    start = selection["start"]
    text = next(
        (
            _compact(line, limit=500)
            for line in str(selection.get("text") or "").splitlines()
            if line.strip()
        ),
        "[mock] Непустой selection",
    )
    return [
        {
            "page": int(start["page"]),
            "line": int(start["line"]),
            "text": text,
        }
    ]


def _compact(value: object, *, limit: int) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: max(1, limit - 1)].rstrip() + "…"


def _tokens(value: object) -> set[str]:
    tokens: set[str] = set()
    current: list[str] = []
    for character in str(value).casefold():
        if character.isalnum():
            current.append(character)
            continue
        if current:
            tokens.add("".join(current))
            current = []
    if current:
        tokens.add("".join(current))
    return tokens


__all__ = ["MockLlmTransport", "MockRetrieval"]
