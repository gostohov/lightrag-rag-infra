from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from typing import Any

from .models import PolicyRule, PromptCall, PromptId, PromptInputBudget, PromptSpec


class PromptPolicyError(ValueError):
    pass


class PromptPolicy:
    def __init__(
        self,
        *,
        version: str,
        rules: tuple[PolicyRule, ...],
        prompts: tuple[PromptSpec, ...],
    ) -> None:
        self.version = version
        self.rules = {rule.rule_id: rule for rule in rules}
        self.prompts = {prompt.prompt_id: prompt for prompt in prompts}
        if len(self.rules) != len(rules):
            raise PromptPolicyError("ID правил prompt policy должны быть уникальными")
        for prompt in prompts:
            unknown = set(prompt.rule_ids) - set(self.rules)
            if unknown:
                raise PromptPolicyError(f"Промпт ссылается на неизвестные правила: {unknown}")
            if prompt.length_retry_max_tokens is not None:
                base_max_tokens = prompt.generation_parameters.get("max_tokens")
                if (
                    not isinstance(base_max_tokens, int)
                    or isinstance(base_max_tokens, bool)
                    or not isinstance(prompt.length_retry_max_tokens, int)
                    or isinstance(prompt.length_retry_max_tokens, bool)
                    or prompt.length_retry_max_tokens <= base_max_tokens
                ):
                    raise PromptPolicyError(
                        "Расширенный generation profile должен превышать "
                        "базовый max_tokens"
                    )

    def build_call(
        self,
        prompt_id: PromptId,
        context: dict[str, Any],
        *,
        allowed_tools: tuple[str, ...] | None = None,
    ) -> PromptCall:
        spec = self.prompts[prompt_id]
        unexpected = set(context) - set(spec.allowed_context)
        if unexpected:
            raise PromptPolicyError(
                f"Для {prompt_id.value} не разрешены поля контекста: {sorted(unexpected)}"
            )
        effective_tools = spec.allowed_tools if allowed_tools is None else allowed_tools
        if not effective_tools and spec.allowed_tools:
            raise PromptPolicyError(
                f"Набор tools для {prompt_id.value} не может быть пустым"
            )
        if len(set(effective_tools)) != len(effective_tools):
            raise PromptPolicyError(
                f"Tools для {prompt_id.value} должны быть уникальными"
            )
        unexpected_tools = set(effective_tools) - set(spec.allowed_tools)
        if unexpected_tools:
            raise PromptPolicyError(
                f"Для {prompt_id.value} не разрешены tools: {sorted(unexpected_tools)}"
            )
        rules_text = "\n".join(
            f"[{rule_id}] {self.rules[rule_id].instruction}" for rule_id in spec.rule_ids
        )
        system_prompt = f"{spec.instruction}\n\nПредметные правила:\n{rules_text}"
        fingerprint_input = json.dumps(
            {
                "policy_version": self.version,
                "prompt_version": spec.version,
                "prompt_id": prompt_id.value,
                "system_prompt": system_prompt,
                "tools": effective_tools,
                "generation_parameters": spec.generation_parameters,
                "length_retry_max_tokens": spec.length_retry_max_tokens,
                "input_budget": (
                    asdict(spec.input_budget)
                    if spec.input_budget is not None
                    else None
                ),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        return PromptCall(
            prompt_id=prompt_id,
            policy_version=self.version,
            prompt_version=spec.version,
            fingerprint=hashlib.sha256(fingerprint_input.encode("utf-8")).hexdigest(),
            rule_ids=spec.rule_ids,
            context=dict(context),
            system_prompt=system_prompt,
            allowed_tools=effective_tools,
            generation_parameters=dict(spec.generation_parameters),
            length_retry_max_tokens=spec.length_retry_max_tokens,
        )

    def with_prompt_text(self, prompt_id: PromptId, instruction: str) -> PromptPolicy:
        prompts = tuple(
            replace(spec, instruction=instruction) if spec.prompt_id is prompt_id else spec
            for spec in self.prompts.values()
        )
        return PromptPolicy(version=self.version, rules=tuple(self.rules.values()), prompts=prompts)


def default_policy() -> PromptPolicy:
    rules = (
        PolicyRule("PMI-INFO-001", "Информационный тезис", "Не создавай тест из декларативного или терминологического тезиса."),
        PolicyRule(
            "PMI-SPLIT-001",
            "Независимые условия",
            (
                "Разные независимые условия оформляй отдельными каркасами; "
                "части одного составного условия не разделяй."
            ),
        ),
        PolicyRule("PMI-CONS-001", "Несколько последствий", "Последствия одного воздействия сохраняй проверками одной карточки."),
        PolicyRule("PMI-VALUES-001", "Конкретные значения", "Для каждого явно перечисленного конкретного значения предложи отдельный каркас."),
        PolicyRule("PMI-NOGUESS-001", "Класс значений", "Не выбирай произвольное значение из класса или диапазона."),
        PolicyRule("PMI-REVERSE-001", "Обратный вывод", "Не выводи обратное правило из одностороннего требования."),
        PolicyRule("PMI-RANGE-001", "Недостаточный диапазон", "Сообщи о недостаточном диапазоне вместо дополнения знаниями модели."),
        PolicyRule("PMI-SW-001", "Прямой код ошибки", "Явный код ошибки является допустимым наблюдаемым результатом."),
        PolicyRule("PMI-OBS-001", "Внутреннее состояние", "SW 9000 сам по себе не доказывает изменение внутреннего состояния."),
        PolicyRule("PMI-EVIDENCE-001", "Прямое доказательство", "Похожий retrieval-фрагмент не подтверждает искомое значение."),
        PolicyRule("PMI-HUMAN-001", "Экспертное знание", "Утверждение аналитика имеет приоритет только в текущей карточке."),
    )
    common_card_rules = (
        "PMI-NOGUESS-001",
        "PMI-REVERSE-001",
        "PMI-SW-001",
        "PMI-OBS-001",
        "PMI-EVIDENCE-001",
        "PMI-HUMAN-001",
    )
    prompts = (
        PromptSpec(
            prompt_id=PromptId.DECOMPOSITION,
            version="1.3.0",
            instruction=(
                "Построй каркасы атомарных функциональных тестов только из точного "
                "выбранного диапазона. Заполняй input_value только значением, которое "
                "явно приведено в источнике как тестовый вход. Условие вида «не равно X» "
                "не разрешает выбирать произвольное другое значение: верни input_value=null "
                "и gap kind=input_value. Части одного составного условия A И B сохраняй "
                "в одном условии и не создавай отдельные каркасы для A и B, если источник "
                "не задаёт последствия для них независимо. Для каждого каркаса укажи "
                "condition_ranges, changed_factor_ranges, input_value_ranges и "
                "action_ranges; null-значению соответствует пустой список координат. "
                "Классифицируй каждую строку selection ровно один раз в line_assessments "
                "как evidence либо context с непустой причиной. Evidence-строки должны "
                "точно совпасть со строками, использованными координатами каркасов. "
                "Верни один submit_decomposition."
            ),
            rule_ids=(
                "PMI-INFO-001",
                "PMI-SPLIT-001",
                "PMI-CONS-001",
                "PMI-VALUES-001",
                "PMI-NOGUESS-001",
                "PMI-REVERSE-001",
                "PMI-RANGE-001",
            ),
            allowed_context=("selection", "existing_card_summaries"),
            allowed_tools=("submit_decomposition",),
            generation_parameters={
                "temperature": 0.0,
                "max_tokens": 4096,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            input_budget=PromptInputBudget(
                max_lines=240,
                max_estimated_tokens=12_000,
                estimator="decomposition-json-utf8-div4-v1",
            ),
        ),
        PromptSpec(
            prompt_id=PromptId.DECOMPOSITION_WINDOW,
            version="1.7.0",
            instruction=(
                "Обработай ровно одно immutable окно большого selection Prompt 1. "
                "Строки primary=true принадлежат этому окну. Overlap нужен только "
                "для продолжения и контекста "
                "поведения, evidence которого затрагивает primary=true. Каждый "
                "candidate обязан использовать evidence хотя бы одной primary=true "
                "строки. Не создавай самостоятельные candidates для поведения, все "
                "source ranges которого находятся только в primary=false overlap: "
                "его обработает primary-owner другого окна. При этом все переданные "
                "primary=false строки доступны как source context текущего окна: "
                "используй их для завершения primary-owned candidate и не объявляй "
                "boundary dependency, если нужная информация есть в overlap. "
                "Построй только локальные candidates с точными source ranges. "
                "Нумерация строк начинается заново на каждой странице. Source "
                "range не может пересекать границу страниц: если evidence "
                "продолжается на следующей странице, верни для неё отдельный "
                "source range с фактическими page/line из window context. "
                "Для каждого candidate соблюдай полный контракт каркаса Prompt 1. "
                "Заполняй input_value только конкретным значением, которое источник "
                "явно задаёт как тестовый вход; название команды не является входным "
                "значением. Если конкретное значение отсутствует, верни "
                "input_value=null, пустой input_value_ranges и обязательный "
                "gap kind=input_value. Если воздействие отсутствует, верни "
                "action=null, пустой action_ranges и обязательный gap kind=action. "
                "Null без соответствующего blocking gap недопустим. Не возвращай "
                "primary_line_assessments: application детерминированно построит "
                "полное покрытие primary range из validated source ranges candidates. "
                "Соблюдай точную матрицу результата: outcome=candidates требует "
                "непустой candidates и допускает boundary_dependencies; "
                "outcome=boundary_dependency требует пустой candidates и непустой "
                "boundary_dependencies; outcome=no_local_testable_behavior требует "
                "пустые candidates и boundary_dependencies. Если repair добавляет "
                "dependency, одновременно смени outcome на boundary_dependency, если "
                "готовых candidates нет. "
                "Не назначай domain IDs и не считай собственный текст evidence. "
                "Если обязательная часть поведения отсутствует во всех переданных "
                "primary и overlap строках, верни typed boundary dependency с "
                "направлением, missing_field и source range, достигающим указанной "
                "границы окна. "
                "Не объявляй глобальные no_testable_behavior или "
                "insufficient_selection. Не используй candidates других окон. "
                "Верни ровно один submit_window_candidates."
            ),
            rule_ids=(
                "PMI-INFO-001",
                "PMI-SPLIT-001",
                "PMI-CONS-001",
                "PMI-VALUES-001",
                "PMI-NOGUESS-001",
                "PMI-REVERSE-001",
                "PMI-RANGE-001",
            ),
            allowed_context=("window",),
            allowed_tools=("submit_window_candidates",),
            generation_parameters={
                "temperature": 0.0,
                "max_tokens": 4096,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        ),
        PromptSpec(
            prompt_id=PromptId.DECOMPOSITION_RECONCILIATION,
            version="2.0.0",
            instruction=(
                "Разреши ровно один bounded semantic reconciliation case. "
                "Application уже связал case с кандидатами и source evidence. "
                "Для candidate_pair сравни A и B и выбери: duplicate_keep_a, "
                "duplicate_keep_b, keep_separate либо unresolved. Для "
                "candidate_review выбери keep, reject либо unresolved. Для "
                "dependency_review выбери resolved либо unresolved. Не создавай "
                "новые candidates, не переписывай source и не объединяй элементы "
                "только по сходству текста. Верни только semantic decision и "
                "краткую причину, без candidate IDs, accepted/rejected списков, "
                "relations, coordinates и source ranges. Верни ровно один "
                "submit_reconciliation_case."
            ),
            rule_ids=(
                "PMI-SPLIT-001",
                "PMI-CONS-001",
                "PMI-NOGUESS-001",
                "PMI-REVERSE-001",
                "PMI-RANGE-001",
            ),
            allowed_context=("case",),
            allowed_tools=("submit_reconciliation_case",),
            generation_parameters={
                "temperature": 0.0,
                "max_tokens": 2048,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        ),
        PromptSpec(
            prompt_id=PromptId.DECOMPOSITION_WINDOW_SEMANTIC,
            version="2.1.0",
            instruction=(
                "Обработай ровно одно immutable окно большого selection "
                "Prompt 1. Извлеки только локальные смысловые фрагменты. Для "
                "каждого behavior задай короткий title, свободное summary и "
                "атомарные facts с opaque line_ids из window context. Не "
                "раскладывай факты по полям карточки и не определяй, какие поля "
                "отсутствуют. Не возвращай roles, missing_parts, "
                "boundary_needs, coordinates, ranges, quotes, target_paths, "
                "IDs, gaps, outcome, explanation или line assessments. "
                "Каждый behavior обязан ссылаться хотя бы на одну primary=true "
                "строку; primary=false используй только как контекст. "
                "Пустой behaviors означает отсутствие локального тестируемого "
                "поведения. Верни ровно один submit_semantic_window_result."
            ),
            rule_ids=(
                "PMI-INFO-001",
                "PMI-SPLIT-001",
                "PMI-CONS-001",
                "PMI-VALUES-001",
                "PMI-NOGUESS-001",
                "PMI-REVERSE-001",
                "PMI-RANGE-001",
            ),
            allowed_context=("window",),
            allowed_tools=("submit_semantic_window_result",),
            generation_parameters={
                "temperature": 0.0,
                "max_tokens": 8192,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            length_retry_max_tokens=12_288,
        ),
        PromptSpec(
            prompt_id=PromptId.DECOMPOSITION_SEMANTIC_SYNTHESIS,
            version="1.3.0",
            instruction=(
                "Собери содержание каркасов для одного target window из "
                "validated target_fragments. Каждый fragment содержит "
                "target_facts, которые нужно обработать, и supporting_facts "
                "того же пограничного поведения, которые можно использовать "
                "только для его понимания и дополнения. Создавай candidate "
                "только с опорой минимум на один target_fact; не создавай "
                "отдельный candidate из supporting_facts. Для каждого "
                "candidate верни title, одно поле condition, одно поле "
                "changed_factor и непустой "
                "список consequences. Каждый slot содержит text и fact_ids. "
                "Если несколько facts вместе образуют условие, сформулируй "
                "единый condition и укажи все его fact_ids. Если это "
                "независимые проверки, создай разные candidates. "
                "Input_value добавляй только для конкретного воспроизводимого "
                "входа из facts. Action добавляй только для конкретного "
                "выполнимого тестового воздействия; общее описание команды не "
                "является конкретным action. Если input_value или action не "
                "подтверждены facts, просто не возвращай соответствующее поле: "
                "application создаст typed gap. Не возвращай missing_parts, "
                "parts, roles, boundary_needs, line_ids, coordinates, ranges, "
                "quotes, gaps, target_paths, domain IDs или line assessments. "
                "Не объединяй "
                "независимые проверки только по сходству текста. Верни ровно "
                "один submit_semantic_synthesis."
            ),
            rule_ids=(
                "PMI-INFO-001",
                "PMI-SPLIT-001",
                "PMI-CONS-001",
                "PMI-VALUES-001",
                "PMI-NOGUESS-001",
                "PMI-REVERSE-001",
                "PMI-RANGE-001",
            ),
            allowed_context=("synthesis",),
            allowed_tools=("submit_semantic_synthesis",),
            generation_parameters={
                "temperature": 0.0,
                "max_tokens": 4096,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        ),
        PromptSpec(
            prompt_id=PromptId.POPULATION,
            version="1.6.0",
            instruction=(
                "Атомарно заполни одну карточку из диапазона, каркаса и проверенных "
                "evidence. В source_values помещай только прямо подтверждённые "
                "источником значения с evidence_id. Не используй сообщения "
                "аналитика: они проходят отдельный подтверждаемый conversation "
                "flow. Значения, выведенные из каркаса или evidence, помещай в "
                "derivations с "
                "source_evidence_ids, rule и scope. Если основания или конкретного "
                "значения нет, создай gap и явно выбери resolution_mode: "
                "source_fact для факта из доступного корпуса, design_decision для "
                "проектирования теста аналитиком либо external_input для сведений "
                "о стенде, реализации или процессе вне корпуса. Если один "
                "предметный вопрос затрагивает paths с разными режимами или "
                "требованиями к конкретности, обязательно передай "
                "resolution_targets для каждого allowed_path: application "
                "разделит их на независимые gaps. Для воспроизводимых тестовых "
                "данных требуй exact, finite_set или deterministic_rule. "
                "Source facts не объединяй с design decisions. Отсутствие поля "
                "в источнике не означает "
                "«не применимо»: оставь его неизвестным, если оно не блокирует тест. "
                "Не помещай обязательные поля в not_applicable: каждое из них "
                "должно иметь значение, derivation либо gap. "
                "Каждый путь указывай ровно в одном разделе, включая allowed_paths "
                "пробелов. До submit покрой значением, выводом или gap каждое "
                "обязательное поле: requirement.condition, requirement.behavior, "
                "test.action, test.changed_factor, test.observation.method. Также "
                "заполни хотя бы один ожидаемый результат либо создай для него gap. "
                "Не обращайся к retrieval."
            ),
            rule_ids=common_card_rules,
            allowed_context=("selection", "skeleton", "card", "evidence"),
            allowed_tools=("submit_card_population",),
            generation_parameters={
                "temperature": 0.0,
                "max_tokens": 4096,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        ),
        PromptSpec(
            prompt_id=PromptId.GAP_RESEARCH,
            version="1.4.8",
            instruction=(
                "Исследуй один связанный пробел короткими предметными вопросами и "
                "измени только разрешённые поля. Если observations нет, первым "
                "вызови ask_lightrag с одним коротким вопросом, прямо "
                "указанным в gap. Не пытайся завершить исследование до ответа retrieval. "
                "После observation выбери ровно один следующий tool call. Если narrow "
                "observation не содержит evidence, не перефразируй вопрос: расширь тот "
                "же вызов либо заверши not_found. Если broad observation также не "
                "содержит evidence, заверши not_found. Для not_found укажи "
                "missing_fact кратким текстом либо объектом field/description, где "
                "field входит в unknown_fields. Для resolved верни updates для "
                "каждого всё ещё неизвестного пути из "
                "gap.allowed_paths; unknown_fields, missing_fact и contradictions "
                "должны быть пустыми. В evidence_id используй только ID из "
                "observations.evidence_ids текущего контекста. Никогда "
                "не подставляй call_id инструмента ни в evidence_id, ни в "
                "analyst_message_id. analyst_message_id всегда null; экспертные "
                "ответы применяет отдельный application flow. call_id разрешён "
                "только для expand_lightrag. Если передан research_question, используй его "
                "только для формулировки следующего LightRAG-запроса или выбора "
                "стратегии поиска; это typed intent conversation agent, не evidence, "
                "не ответ аналитика и не "
                "основание для submit_gap_result."
            ),
            rule_ids=("PMI-NOGUESS-001", "PMI-OBS-001", "PMI-EVIDENCE-001", "PMI-HUMAN-001"),
            allowed_context=(
                "selection",
                "card",
                "gap",
                "evidence",
                "research_question",
                "observations",
            ),
            allowed_tools=("ask_lightrag", "expand_lightrag", "submit_gap_result"),
            generation_parameters={
                "temperature": 0.0,
                "max_tokens": 1024,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        ),
        PromptSpec(
            prompt_id=PromptId.REFINEMENT,
            version="1.1.0",
            instruction=(
                "Обработай одно сообщение аналитика для текущей ревизии карточки. "
                "Верни полные значения явно изменяемых полей, новые пробелы или no_change."
            ),
            rule_ids=common_card_rules,
            allowed_context=("card", "message", "evidence"),
            allowed_tools=("submit_card_refinement",),
            generation_parameters={
                "temperature": 0.0,
                "max_tokens": 2048,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        ),
        PromptSpec(
            prompt_id=PromptId.SELECTION_REVIEW,
            version="1.1.0",
            instruction=(
                "Сопоставь точный диапазон с карточками и решениями по каркасам. "
                "Не изменяй карточки и не вызывай retrieval."
            ),
            rule_ids=tuple(rule.rule_id for rule in rules),
            allowed_context=("selection", "cards", "skeleton_decisions"),
            allowed_tools=("submit_selection_review",),
            generation_parameters={
                "temperature": 0.0,
                "max_tokens": 4096,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        ),
        PromptSpec(
            prompt_id=PromptId.CONVERSATION,
            version="1.4.0",
            instruction=(
                "Выбери ровно один terminal tool: ответ без side effects, "
                "уточняющий вопрос либо одно действие из available_actions. "
                "Передавай только аргументы schema выбранного tool. Системные "
                "идентификаторы добавляет код. Не считай собственный текст "
                "evidence и не обходи application guards. Ответ аналитика "
                "сначала становится показываемой pending-интерпретацией; "
                "применяй её только отдельным confirm tool после явного "
                "подтверждения пользователя. Специализированная доработка "
                "также сначала строит точное предложение и не изменяет "
                "карточку до подтверждения. Значения analyst answer обязаны "
                "соответствовать open_gap.closure_requirements и schema: "
                "exact, finite_set и deterministic_rule передаются явной "
                "tagged-формой; недостаточно конкретный, но сохраняемый ответ "
                "передаётся как confirmed_value без усиления смысла. Leave "
                "gap означает только оставить вопрос "
                "нерешённым и никогда не закрывает gap; достаточное "
                "подтверждённое значение закрывается application "
                "автоматически. Workbench имеет typed доступ к "
                "LightRAG через research_gap для source_fact. Не утверждай, "
                "что LightRAG технически недоступен. Для design_decision "
                "объясняй только неприменимость поиска к выбору конкретного "
                "проектного значения и используй propose_design_decision."
            ),
            rule_ids=("PMI-EVIDENCE-001", "PMI-HUMAN-001"),
            allowed_context=("messages",),
            allowed_tools=(
                "respond_to_analyst",
                "request_clarification",
                "resume",
                "research_gap",
                "submit_analyst_answer",
                "confirm_analyst_answer",
                "reject_analyst_answer",
                "propose_design_decision",
                "change_gap_mode",
                "leave_gap",
                "refine_card",
                "include_card",
                "exclude_card",
                "export_diagnostics",
                "export_pmi",
            ),
            generation_parameters={
                "temperature": 0.0,
                "max_tokens": 1024,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        ),
    )
    return PromptPolicy(version="2026-07-20.24", rules=rules, prompts=prompts)
