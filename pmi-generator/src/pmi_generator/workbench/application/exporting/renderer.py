from __future__ import annotations

from typing import Any

from ...domain import TestCard
from ...domain.enums import CardDecisionKind, EpistemicStatus, GapStatus


FIELD_LABELS = {
    "requirement.condition": "Условие",
    "requirement.preconditions": "Нормативные предусловия",
    "requirement.behavior": "Требуемое поведение",
    "requirement.consequences": "Обязательные последствия",
    "test.initial_state": "Исходное состояние",
    "test.preconditions": "Предусловия",
    "test.previous_commands": "Предыдущие команды",
    "test.action": "Воздействие",
    "test.changed_factor": "Изменяемый фактор",
    "test.control_values": "Контрольные значения",
    "test.command.cla": "CLA",
    "test.command.ins": "INS",
    "test.command.p1": "P1",
    "test.command.p2": "P2",
    "test.command.lc": "Lc",
    "test.command.data": "Data",
    "test.command.le": "Le",
    "test.expected.status_word": "Статусное слово",
    "test.expected.response_data": "Данные ответа",
    "test.expected.state_change": "Изменение состояния",
    "test.expected.no_state_change": "Отсутствие изменения",
    "test.observation.kind": "Тип наблюдения",
    "test.observation.method": "Способ наблюдения",
    "test.observation.value": "Наблюдаемое значение",
    "test.observation.causal_link": "Причинная связь",
    "test.observation.alternative_causes": "Альтернативные причины",
    "test.observation.exclusions": "Исключение альтернатив",
}


class MarkdownCardRenderer:
    def render_working(self, card: TestCard) -> str:
        lines = [
            f"# {card.title}",
            "",
            f"- Карточка: `{card.card_id}`",
            f"- Раздел: `{card.section_number}`",
            f"- Диапазон: `{card.selection_id}`",
            f"- Ревизия: `{card.revision}`",
            (
                "- Статус: **готова**"
                if card.is_ready
                else "- Статус: **неполная**"
            ),
        ]
        used_evidence: set[str] = set()
        lines.extend(["", "## Нормативное требование", ""])
        self._render_fields(lines, card, "requirement.", used_evidence)
        lines.extend(["", "## Проект проверки", ""])
        self._render_fields(lines, card, "test.", used_evidence)
        open_gaps = [
            gap
            for gap in card.gaps.values()
            if gap.status is not GapStatus.RESOLVED
        ]
        if open_gaps:
            lines.extend(["", "## Блокирующие пробелы", ""])
            for gap in open_gaps:
                lines.extend(
                    [
                        f"- **{gap.gap_id}:** {gap.question}",
                        f"  Тип разрешения: `{gap.resolution_mode.value}`",
                        f"  Почему блокирует: {gap.blocking_reason}",
                    ]
                )
        lines.extend(["", "## Доказательства", ""])
        if not used_evidence:
            lines.append("Подтверждающие доказательства отсутствуют.")
        for evidence_id in sorted(used_evidence):
            evidence = card.evidence[evidence_id]
            if evidence.address:
                address = evidence.address
                location = (
                    f"{address.document_id} v{address.document_version}, "
                    f"стр. {address.page}:{address.line_start}-{address.line_end}"
                )
            else:
                location = (
                    f"Экспертное знание, {evidence.author}, "
                    f"сообщение `{evidence.message_id}`"
                )
            lines.extend(
                [
                    f"- `{evidence_id}` — {location}",
                    f"  > {evidence.quote}",
                ]
            )
        return "\n".join(lines).rstrip() + "\n"

    def render(self, card: TestCard) -> str:
        decision = card.decision
        if decision is None or decision.revision != card.revision:
            raise ValueError("Карточка не имеет актуального решения для renderer")
        if decision.kind is CardDecisionKind.EXCLUDE:
            raise ValueError("Исключённая карточка не рендерится в ПМИ")
        lines = [
            f"# {card.title}",
            "",
            f"- Карточка: `{card.card_id}`",
            f"- Раздел: `{card.section_number}`",
            f"- Диапазон: `{card.selection_id}`",
            f"- Ревизия: `{card.revision}`",
        ]
        if decision.kind is CardDecisionKind.INCLUDE_INCOMPLETE:
            lines.extend(
                [
                    "- Статус: **включена неполной**",
                    f"- Основание: {decision.reason}",
                ]
            )
        else:
            lines.append("- Статус: **включена**")
        used_evidence: set[str] = set()
        lines.extend(["", "## Нормативное требование", ""])
        self._render_fields(lines, card, "requirement.", used_evidence)
        lines.extend(["", "## Проект проверки", ""])
        self._render_fields(lines, card, "test.", used_evidence)

        open_gaps = [gap for gap in card.gaps.values() if gap.status is not GapStatus.RESOLVED]
        if open_gaps:
            lines.extend(["", "## Оставленные пробелы", ""])
            for gap in open_gaps:
                lines.extend(
                    [
                        f"### {gap.gap_id}",
                        "",
                        f"- Вопрос: {gap.question}",
                        f"- Тип разрешения: `{gap.resolution_mode.value}`",
                        f"- Почему блокирует: {gap.blocking_reason}",
                        f"- Критерий закрытия: {gap.closure_criterion}",
                        "",
                    ]
                )

        lines.extend(["", "## Ссылки", ""])
        if not used_evidence:
            lines.append("Подтверждающие ссылки в карточке отсутствуют.")
        for evidence_id in sorted(used_evidence):
            evidence = card.evidence[evidence_id]
            if evidence.address:
                address = evidence.address
                location = f"{address.document_id} v{address.document_version}, стр. {address.page}:{address.line_start}-{address.line_end}"
                if address.chunk_id:
                    location += f", `{address.chunk_id}`"
            else:
                location = f"Экспертное знание, {evidence.author}, сообщение `{evidence.message_id}`"
            lines.extend([f"- `{evidence_id}` — {location}", f"  > {evidence.quote}"])
        return "\n".join(lines).rstrip() + "\n"

    def _render_fields(
        self,
        lines: list[str],
        card: TestCard,
        prefix: str,
        used_evidence: set[str],
    ) -> None:
        rendered = 0
        for path in sorted(card.fields):
            if not path.startswith(prefix):
                continue
            field = card.field(path)
            if field.status is EpistemicStatus.UNKNOWN:
                continue
            label = FIELD_LABELS[path]
            if field.status is EpistemicStatus.NOT_APPLICABLE:
                lines.append(f"- **{label}:** не применимо ({field.reason})")
                rendered += 1
                continue
            provenance = (
                " _(подтверждено аналитиком)_"
                if field.status is EpistemicStatus.ANALYST_CONFIRMED
                else ""
            )
            lines.append(
                f"- **{label}:** {self._format_value(field.value)}"
                f"{provenance}"
            )
            used_evidence.update(field.evidence_ids)
            if field.derivation_id:
                derivation = card.derivations[field.derivation_id]
                used_evidence.update(derivation.source_evidence_ids)
            rendered += 1
        if not rendered:
            lines.append("Заполненные поля отсутствуют.")

    @staticmethod
    def _format_value(value: Any) -> str:
        if isinstance(value, (list, tuple)):
            return "; ".join(str(item) for item in value)
        if isinstance(value, dict):
            return "; ".join(f"{key}={value[key]}" for key in sorted(value))
        return str(value)
