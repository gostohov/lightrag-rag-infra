from __future__ import annotations

from ...domain.source import SourceDocument, SourceSection, TextSelection


def render_structure(
    document: SourceDocument,
    *,
    cursor: int = 0,
    query: str = "",
    sections_with_work: set[str] | None = None,
) -> str:
    sections_with_work = sections_with_work or set()
    normalized = query.casefold().strip()
    sections = [
        section
        for section in document.sections
        if not normalized or normalized in section.label.casefold()
    ]
    lines = [
        "PMI Workbench / Структура спецификации",
        "",
        f"Поиск: {query or 'не задан'}",
        "",
    ]
    for index, section in enumerate(sections):
        prefix = ">" if index == cursor else " "
        suffix = "  [есть работа]" if section.section_id in sections_with_work else ""
        indent = "  " * max(0, len(section.path) - 1)
        lines.append(f"{prefix} {indent}{section.label}{suffix}")
    lines.extend(
        [
            "",
            (
                "[↑/↓] раздел  [PgUp/PgDn] страница  [Enter] открыть  "
                "[E] экспорт ПМИ  [/] поиск  [Esc] выход"
            ),
        ]
    )
    return "\n".join(lines)


def render_confirmation(
    section: SourceSection,
    selection: TextSelection,
    *,
    source_name: str = "specification.pdf",
) -> str:
    start = selection.start
    end = selection.end
    return "\n".join(
        [
            f"PMI Workbench / Структура / {section.label} / Новый диапазон",
            "",
            f"Источник: {source_name}",
            (
                f"Диапазон: стр. {start.page_index}:{start.line_number:03d} — "
                f"стр. {end.page_index}:{end.line_number:03d}"
            ),
            f"Строк: {len(selection.positions)}",
            "",
            selection.text,
            "",
            "Действия:",
            "> Построить каркасы карточек",
            "  Изменить диапазон",
            "  Отмена",
            "",
            "[↑/↓] выбор  [Enter] выполнить  [Esc] назад",
        ]
    )
