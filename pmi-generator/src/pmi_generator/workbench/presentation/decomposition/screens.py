from __future__ import annotations

from ...application.decomposition import SkeletonDecisionController


def render_decomposition_progress(elapsed: str = "00:00") -> str:
    return "\n".join(
        [
            "PMI Workbench / Построение каркасов",
            "",
            "⠹ Построение каркасов карточек",
            f"  Выполняется: {elapsed}",
            "  Стадия: анализ выбранного текста",
            "",
            "Ввод временно заблокирован.  [Esc] прервать",
        ]
    )


def render_decomposition_outcome(outcome: str, explanation: str) -> str:
    labels = {
        "skeletons_created": "Каркасы построены",
        "no_testable_behavior": "Проверяемое поведение не найдено",
        "insufficient_selection": "Недостаточный диапазон",
    }
    return f"{labels[outcome]}\n\n{explanation}".rstrip()


def render_skeletons(controller: SkeletonDecisionController) -> str:
    lines = [
        "PMI Workbench / Каркасы выбранного диапазона",
        "",
        f"Решения: {len(controller.decisions)} из {len(controller.skeleton_ids)}",
        "",
    ]
    labels = {None: "без решения", "selected": "в работу", "excluded": "исключён"}
    for index, skeleton_id in enumerate(controller.skeleton_ids):
        cursor = ">" if index == controller.cursor else " "
        decision = labels[controller.decisions.get(skeleton_id)]
        lines.append(f"{cursor} [{decision}] {skeleton_id}")
    lines.extend(
        [
            "",
            "[↑/↓] каркас  [Enter] открыть  [PgUp/PgDn] страница  [Esc] назад",
        ]
    )
    return "\n".join(lines)
