from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from prompt_toolkit.formatted_text import AnyFormattedText
from prompt_toolkit.formatted_text.utils import fragment_list_to_text

from ..application.facade import WorkbenchFacade
from ..application.conversation import (
    ConversationAction,
    ConversationToolCall,
)
from ..application.session import SessionEventKind
from ..application.source import SavedSelection, SelectionRangeSummary
from ..domain import (
    ExecutionMode,
    GapResolutionMode,
    SourceDocument,
    TextSelection,
)
from .operation import OperationCancelledByUser, TerminalOperationRunner
from .decomposition import SkeletonDetailScreen, SkeletonListScreen
from .range_workspace import RangeWorkspaceScreen
from .result import ResultScreen
from .selection_review import SelectionReviewScreen
from .session import TerminalSessionShell, render_session_context_fragments
from .source import (
    CreateSelection,
    OpenRange,
    SourceNavigationState,
    select_structure_action,
    select_text_range,
)


class TerminalWorkbench:
    mode_label: str | None = None

    def __init__(
        self,
        document: SourceDocument,
        *,
        facade: WorkbenchFacade,
    ) -> None:
        self.document = document
        self.facade = facade
        self._operation_runner = TerminalOperationRunner()
        self.source_name = document.metadata.original_name
        self.mode_label = (
            "Тестовый режим: mock"
            if document.metadata.execution_mode is ExecutionMode.MOCK
            else None
        )
        self._structure_notice: tuple[str, str] | None = None
        self._source_navigation = SourceNavigationState()

    def run(self) -> int:
        while True:
            choice = self._main_menu()
            if choice is None:
                return 0
            if choice == "export":
                self._export_full()
                continue
            kind, value = choice.split(":", 1)
            if kind != "section":
                raise ValueError(f"Неизвестное действие структуры: {choice}")
            section = next(item for item in self.document.sections if item.section_id == value)
            action = select_text_range(
                self.document,
                section,
                ranges=self._canvas_ranges(),
                source_name=self.source_name,
                mode_label=self.mode_label,
                state=self._source_navigation,
                assess_decomposition=self.facade.assess_decomposition,
                assess_windowing=self.facade.assess_decomposition_route,
            )
            if action is None:
                continue
            if isinstance(action, OpenRange):
                self._open_range(action.selection_id)
                continue
            self._create_selection(
                section.section_id,
                action.selection,
                supersede_selection_ids=action.supersede_selection_ids,
            )

    def _main_menu(self) -> str | None:
        return select_structure_action(
            self.document,
            self.facade.selection_ranges(),
            source_name=self.source_name,
            notice=self._structure_notice,
            mode_label=self.mode_label,
            state=self._source_navigation,
        )

    def _create_selection(
        self,
        section_id: str,
        selection: TextSelection,
        *,
        supersede_selection_ids: tuple[str, ...] = (),
    ) -> None:
        saved = self.facade.save_selection(
            section_id,
            selection,
            supersede_selection_ids=supersede_selection_ids,
        )
        if self._decompose_selection(saved) == "range":
            self._open_range(saved.selection_id)

    def _decompose_selection(self, saved: SavedSelection) -> str:
        while True:
            operation = self.facade.decompose(saved)
            try:
                result = self._wait(
                    "Построение каркасов карточек",
                    operation.awaitable,
                    cancel=operation.cancel,
                    context=lambda width, height: self._decomposition_operation_context(
                        saved,
                        width=width,
                        height=height,
                        progress=operation.progress(),
                    ),
                    full_screen=True,
                )
            except OperationCancelledByUser:
                return "range"
            except Exception as error:
                action = ResultScreen(
                    self._selection_breadcrumb(saved, "Ошибка"),
                    (
                        "Не удалось построить каркасы карточек.\n"
                        f"Причина: {error}\n\n"
                        "Диагностика сохранена."
                    ),
                    (("retry", "Повторить"), ("back", "Назад к диапазону")),
                    kind="error",
                    mode_label=self.mode_label,
                ).run()
                if action == "retry":
                    continue
                return "range"
            break
        if result.outcome != "skeletons_created":
            action = ResultScreen(
                self._selection_breadcrumb(saved, "Результат декомпозиции"),
                self._decomposition_outcome_text(result.outcome, result.explanation),
                (("change", "Изменить диапазон"), ("back", "Назад к разделу")),
                kind="warning",
                mode_label=self.mode_label,
            ).run()
            if action == "change":
                self._select_new_range(saved.section_id)
            return "done"
        self._decide_skeletons(saved, result.skeleton_ids)
        return "done"

    def _decide_skeletons(self, selection: SavedSelection, skeleton_ids: tuple[str, ...]) -> None:
        screen = SkeletonListScreen(
            self.facade.skeletons(skeleton_ids),
            selection,
            mode_label=self.mode_label,
        )
        while True:
            records = self.facade.skeletons(skeleton_ids)
            screen.update(records)
            unresolved = [item for item in records if item.payload.get("decision") is None]
            if not unresolved:
                self._open_range(selection.selection_id)
                return
            selected_id = screen.run()
            if selected_id is None:
                return
            if self._decide_skeleton(selection, selected_id):
                screen.update(self.facade.skeletons(skeleton_ids))
                screen.advance_after_decision(selected_id)

    def _decide_skeleton(
        self,
        selection: SavedSelection,
        skeleton_id: str,
    ) -> bool:
        skeleton = self.facade.skeleton(skeleton_id)
        if skeleton is None:
            return False
        result = SkeletonDetailScreen(
            skeleton,
            mode_label=self.mode_label,
        ).run()
        if result.action == "take":
            card_id = self.facade.take_skeleton(selection.selection_id, skeleton_id)
            self._prepare_card(selection, skeleton_id, card_id)
            return True
        if result.action == "exclude":
            self.facade.exclude_skeleton(
                selection.selection_id,
                skeleton_id,
                result.reason,
            )
            return True
        if result.action == "open_session":
            card_id = str(skeleton.payload.get("card_id") or "")
            if not card_id:
                return False
            session_id, created = self.facade.ensure_card_session(
                selection.selection_id,
                card_id,
            )
            if created:
                self.facade.append(
                    session_id,
                    SessionEventKind.WORKBENCH,
                    (
                        "Первоначальное заполнение карточки не завершено.\n"
                        "/continue — повторить заполнение"
                    ),
                )
            self._open_card_session(
                selection,
                skeleton_id,
                session_id,
                card_id,
                start_preparation=False,
            )
        return False

    def _prepare_card(self, selection: SavedSelection, skeleton_id: str, card_id: str) -> None:
        session_id = self.facade.open_card_session(selection.selection_id, card_id)
        if not self.facade.history(session_id):
            card = self.facade.card(card_id)
            title = card.title if card is not None else card_id
            self.facade.append(
                session_id,
                SessionEventKind.WORKBENCH,
                f"Создана рабочая карточка «{title}».",
            )
        self._open_card_session(
            selection,
            skeleton_id,
            session_id,
            card_id,
            start_preparation=True,
        )

    def _populate_card(
        self,
        selection: SavedSelection,
        skeleton_id: str,
        session_id: str,
        card_id: str,
    ) -> bool:
        try:
            operation = self.facade.populate(
                selection,
                skeleton_id,
                session_id,
                card_id,
            )
            self._wait(
                "Первоначальное заполнение карточки",
                operation.awaitable,
                cancel=operation.cancel,
                context=lambda width, height: self._session_operation_context(
                    session_id,
                    width=width,
                    height=height,
                ),
            )
            return True
        except OperationCancelledByUser:
            return False
        except Exception as error:
            self.facade.append(
                session_id,
                SessionEventKind.WORKBENCH,
                (
                    "Первоначальное заполнение карточки не завершено.\n"
                    f"Причина: {error}\n"
                    "/continue — повторить заполнение"
                ),
            )
            return False

    def _investigate_open_gaps(
        self,
        selection: SavedSelection,
        session_id: str,
        card_id: str,
    ) -> bool:
        for gap_id in self.facade.open_gap_ids(card_id):
            card = self.facade.card(card_id)
            if card is None:
                raise ValueError(f"Карточка {card_id} не найдена")
            gap = card.gaps.get(gap_id)
            if (
                gap is not None
                and gap.resolution_mode is not GapResolutionMode.SOURCE_FACT
            ):
                reason = (
                    "нужно проектное решение аналитика"
                    if gap.resolution_mode is GapResolutionMode.DESIGN_DECISION
                    else "нужны внешние сведения аналитика"
                )
                self.facade.append(
                    session_id,
                    SessionEventKind.ASSISTANT,
                    (
                        f"Исследование {gap_id} ожидает участия аналитика.\n\n"
                        f"Вопрос: {gap.question}\n"
                        f"Причина: {reason}"
                    ),
                    {
                        "gap_id": gap_id,
                        "resolution_mode": gap.resolution_mode.value,
                        "outcome": "awaiting_analyst",
                    },
                )
                return False
            try:
                operation = self.facade.investigate_gap(
                    selection,
                    session_id,
                    card_id,
                    gap_id,
                )
                result = self._wait(
                    f"Исследование пробела {gap_id}",
                    operation.awaitable,
                    cancel=operation.cancel,
                    context=lambda width, height: self._session_operation_context(
                        session_id,
                        width=width,
                        height=height,
                    ),
                )
                if result.outcome != "resolved":
                    return False
            except OperationCancelledByUser:
                return False
            except Exception as error:
                self.facade.append(
                    session_id,
                    SessionEventKind.ERROR,
                    f"Исследование {gap_id} остановлено: {error}",
                )
                return False
        return True

    def _open_card_session(
        self,
        selection: SavedSelection,
        skeleton_id: str,
        session_id: str,
        card_id: str,
        *,
        start_preparation: bool = False,
    ) -> None:
        def continue_flow() -> None:
            self._continue_card(selection, skeleton_id, session_id, card_id)

        def shortcut_message(text: str) -> str:
            sequence = self.facade.append(
                session_id,
                SessionEventKind.ANALYST,
                text,
                {"author": "Аналитик", "shortcut": True},
            )
            return f"MSG_{sequence:06d}"

        def run_shortcut(
            tool_call: ConversationToolCall,
            *,
            message_id: str = "",
        ) -> None:
            before = self.facade.card(card_id).revision
            try:
                result = self.facade.dispatch_conversation_tool(
                    selection,
                    skeleton_id,
                    session_id,
                    card_id,
                    message_id,
                    tool_call,
                )
                self.facade.append(
                    session_id,
                    SessionEventKind.WORKBENCH,
                    result.text,
                    {
                        "conversation_action": result.action.value,
                        "effect": result.effect.value,
                        "shortcut": True,
                    },
                )
                if result.awaitable is not None:
                    self._wait(
                        result.text.splitlines()[0],
                        result.awaitable,
                        cancel=result.cancel,
                        context=lambda width, height: self._session_operation_context(
                            session_id,
                            width=width,
                            height=height,
                        ),
                    )
                current = self.facade.card(card_id)
                if current is not None and current.revision != before:
                    self._append_card_snapshot(session_id, card_id)
            except OperationCancelledByUser:
                return
            except Exception as error:
                self.facade.append(
                    session_id,
                    SessionEventKind.ERROR,
                    f"Команда не выполнена: {error}",
                )

        def include() -> None:
            message_id = shortcut_message("/include")
            revision = self.facade.card(card_id).revision
            run_shortcut(
                ConversationToolCall(
                    ConversationAction.INCLUDE_CARD,
                    {
                        "expected_revision": revision,
                        "confirmation_message_id": message_id,
                    },
                ),
                message_id=message_id,
            )

        def exclude() -> None:
            message_id = shortcut_message("/exclude")
            revision = self.facade.card(card_id).revision
            run_shortcut(
                ConversationToolCall(
                    ConversationAction.EXCLUDE_CARD,
                    {
                        "expected_revision": revision,
                        "confirmation_message_id": message_id,
                    },
                ),
                message_id=message_id,
            )

        def continue_shortcut() -> None:
            revision = self.facade.card(card_id).revision
            run_shortcut(
                ConversationToolCall(
                    ConversationAction.RESUME,
                    {"expected_revision": revision},
                )
            )

        def export_diagnostics() -> None:
            run_shortcut(
                ConversationToolCall(
                    ConversationAction.EXPORT_DIAGNOSTICS,
                    {},
                )
            )

        def converse(message_id: str) -> None:
            before = self.facade.card(card_id).revision
            try:
                operation = self.facade.conversation_turn(
                    selection,
                    skeleton_id,
                    session_id,
                    card_id,
                    message_id,
                )
                self._wait(
                    "Conversation agent",
                    operation.awaitable,
                    cancel=operation.cancel,
                    context=lambda width, height: self._session_operation_context(
                        session_id,
                        width=width,
                        height=height,
                    ),
                )
                current = self.facade.card(card_id)
                if current is not None and current.revision != before:
                    self._append_card_snapshot(session_id, card_id)
            except OperationCancelledByUser:
                return
            except Exception:
                # Application operations persist their own failure event and trace.
                return

        shell = TerminalSessionShell(
            self.facade,
            session_id,
            diagnostics_exporter=lambda: self.facade.export_diagnostics(session_id, card_id),
            command_handlers={
                "/include": include,
                "/exclude": exclude,
                "/continue": continue_shortcut,
                "/export-diagnostics": export_diagnostics,
            },
            command_descriptions={
                "/include": self._include_command_description(card_id),
                "/continue": "продолжить сохранённую стадию",
                "/export-diagnostics": "обновить диагностический Markdown",
            },
            message_handler=converse,
            startup_handler=continue_flow if start_preparation else None,
            breadcrumb=self._card_session_breadcrumb(selection, card_id),
            mode_label=self.mode_label,
        )
        shell.run()

    def _continue_card(
        self,
        selection: SavedSelection,
        skeleton_id: str,
        session_id: str,
        card_id: str,
    ) -> None:
        if self.facade.card(card_id) is None:
            self.facade.append(
                session_id,
                SessionEventKind.ERROR,
                f"Карточка {card_id} не найдена.",
            )
            return
        route = self.facade.continuation(session_id)
        if route == "population":
            if (
                self._populate_card(selection, skeleton_id, session_id, card_id)
                and self._investigate_open_gaps(selection, session_id, card_id)
            ):
                self._append_card_snapshot(session_id, card_id)
            return
        if route == "gap_investigation":
            if self._investigate_open_gaps(
                selection,
                session_id,
                card_id,
            ):
                self._append_card_snapshot(session_id, card_id)
            return
        if route == "coverage_repair":
            self.facade.repair_card_coverage(session_id, card_id)
            if self._investigate_open_gaps(selection, session_id, card_id):
                self._append_card_snapshot(session_id, card_id)
            return
        self.facade.append(
            session_id,
            SessionEventKind.WORKBENCH,
            "Исследование завершено. Сохраните решение /include или /exclude.",
        )

    def _open_range(self, selection_id: str) -> None:
        controller = self.facade.range_controller(selection_id)
        saved = self.facade.load_selection(selection_id)
        section_number = next(
            (
                section.label
                for section in self.document.sections
                if section.section_id == saved.section_id
            ),
            "",
        )
        screen = RangeWorkspaceScreen(
            controller,
            saved,
            section_number=section_number,
            mode_label=self.mode_label,
        )
        while True:
            state = controller.state
            if state.terminal_status:
                action = ResultScreen(
                    self._selection_breadcrumb(saved, "Результат декомпозиции"),
                    self._decomposition_outcome_text(
                        (
                            "no_testable_behavior"
                            if state.terminal_status == "нет тестируемого поведения"
                            else "insufficient_selection"
                        ),
                        state.terminal_explanation,
                    ),
                    (("change", "Изменить диапазон"), ("back", "Назад к разделу")),
                    kind="warning",
                    mode_label=self.mode_label,
                ).run()
                if action == "change":
                    self._select_new_range(saved.section_id)
                return
            choice = screen.run()
            if choice is None:
                return
            kind, item_id = choice
            if kind == "decompose":
                saved = self.facade.load_selection(selection_id)
                self._decompose_selection(saved)
                continue
            if kind == "review":
                self._review_selection(selection_id)
                continue
            saved = self.facade.load_selection(selection_id)
            if kind == "skeleton":
                self._decide_skeleton(saved, item_id)
                continue
            if kind == "session":
                session_id = item_id
                item = next(value for value in state.items if value.session_id == session_id)
                card_id = str(item.card_id)
                created = False
            else:
                card_id = item_id
                item = next(value for value in state.items if value.card_id == card_id)
                session_id, created = self.facade.ensure_card_session(selection_id, card_id)
            if created:
                self.facade.append(
                    session_id,
                    SessionEventKind.WORKBENCH,
                    "Первоначальное заполнение карточки не завершено.\n/continue — повторить заполнение",
                )
            self._open_card_session(saved, item.skeleton_id, session_id, card_id)
            controller.return_from_session()

    def _review_selection(self, selection_id: str) -> None:
        controller = self.facade.range_controller(selection_id)
        saved = self.facade.load_selection(selection_id)
        section_number = next(
            (
                section.label
                for section in self.document.sections
                if section.section_id == saved.section_id
            ),
            "",
        )
        operation_screen = RangeWorkspaceScreen(
            controller,
            saved,
            section_number=section_number,
            mode_label=self.mode_label,
        )
        while True:
            operation = self.facade.review_selection(selection_id)
            try:
                self._wait(
                    "Проверка выбранного диапазона",
                    operation.awaitable,
                    cancel=operation.cancel,
                    context=lambda width, height: fragment_list_to_text(
                        operation_screen.render(
                            width=width,
                            height=height,
                        )
                    ),
                    full_screen=True,
                )
            except OperationCancelledByUser:
                return
            except Exception as error:
                action = ResultScreen(
                    self._selection_breadcrumb(saved, "Ошибка"),
                    (
                        "Проверка выбранного диапазона не завершена.\n"
                        f"Причина: {error}\n\n"
                        "Диагностика сохранена."
                    ),
                    (("retry", "Повторить"), ("back", "Назад к карточкам")),
                    kind="error",
                    mode_label=self.mode_label,
                ).run()
                if action == "retry":
                    continue
                return
            break
        record = self.facade.review_record(selection_id)
        if record is None:
            ResultScreen(
                self._selection_breadcrumb(saved, "Ошибка"),
                "Результат проверки диапазона не найден.\n\nДиагностика сохранена.",
                (("back", "Назад к карточкам"),),
                kind="error",
                mode_label=self.mode_label,
            ).run()
            return
        issues = list(record.payload["issues"])
        screen = SelectionReviewScreen(
            issues,
            selection=saved,
            section_number=section_number,
            mode_label=self.mode_label,
        )
        issues_accepted = bool(record.payload.get("analyst_decision"))
        while True:
            action = screen.run()
            if action == "back":
                return
            try:
                if issues and not issues_accepted:
                    self.facade.accept_review_issues(selection_id)
                    issues_accepted = True
                screen.export_paths = self.facade.export_selection(selection_id)
            except Exception as error:
                retry = ResultScreen(
                    self._selection_breadcrumb(saved, "Ошибка экспорта"),
                    f"Не удалось сформировать файлы диапазона.\nПричина: {error}",
                    (("retry", "Повторить"), ("back", "Назад к результату")),
                    kind="error",
                    mode_label=self.mode_label,
                ).run()
                if retry == "retry":
                    continue
                continue

    def _export_full(self) -> None:
        try:
            path = self.facade.export_full()
            cards = sum(
                state.included + state.included_incomplete
                for record in self.facade.selections()
                for state in (self.facade.workspace(record.record_id),)
            )
            self._structure_notice = (
                "class:success",
                (
                    f"{self.mode_label}\n" if self.mode_label else ""
                )
                + f"ПМИ сформирован:\n{path}\n\nКарточек: {cards}",
            )
        except Exception as error:
            self._structure_notice = (
                "class:error",
                (
                    f"{self.mode_label}\n" if self.mode_label else ""
                )
                + f"Экспорт заблокирован: {error}",
            )

    def _wait(
        self,
        label: str,
        awaitable: Awaitable[Any],
        cancel: Callable[[], object] | None = None,
        context: Callable[[int, int], str] | None = None,
        *,
        full_screen: bool = False,
    ) -> Any:
        return self._operation_runner.run(
            label,
            awaitable,
            cancel,
            context,
            full_screen=full_screen,
        )

    def _session_operation_context(
        self,
        session_id: str,
        *,
        width: int,
        height: int,
    ) -> AnyFormattedText:
        events = self.facade.history(session_id)
        return render_session_context_fragments(
            events,
            width=width,
            height=height,
            mode_label=self.mode_label,
        )

    def _append_card_snapshot(self, session_id: str, card_id: str) -> None:
        card = self.facade.card(card_id)
        if card is None:
            self.facade.append(
                session_id,
                SessionEventKind.ERROR,
                f"Карточка {card_id} не найдена.",
            )
            return
        for event in reversed(self.facade.history(session_id)):
            if not event.metadata.get("card_snapshot"):
                continue
            if int(event.metadata.get("revision", -1)) == card.revision:
                return
            break
        self.facade.append(
            session_id,
            SessionEventKind.ASSISTANT,
            (
                "Подготовка рабочей карточки завершена.\n\n"
                + self.facade.working_card_snapshot(card_id)
            ),
            {
                "revision": card.revision,
                "card_snapshot": True,
            },
        )

    def _card_session_breadcrumb(
        self,
        selection: SavedSelection,
        card_id: str,
    ) -> str:
        card = self.facade.card(card_id)
        title = card.title if card is not None else card_id
        section_number = card.section_number if card is not None else selection.section_id
        return (
            f"PMI Workbench / {section_number} / Карточки / "
            f"{title} / Сессия"
        )

    def _include_command_description(self, card_id: str) -> str:
        card = self.facade.card(card_id)
        if card is not None and not card.is_ready:
            return "оставить пробелы и включить карточку неполной"
        return "включить карточку в итоговый ПМИ"

    def _select_new_range(self, section_id: str) -> None:
        section = next(
            item for item in self.document.sections if item.section_id == section_id
        )
        action = select_text_range(
            self.document,
            section,
            ranges=self._canvas_ranges(),
            source_name=self.source_name,
            mode_label=self.mode_label,
            state=self._source_navigation,
            navigate_to_anchor=False,
        )
        if isinstance(action, OpenRange):
            self._open_range(action.selection_id)
        elif isinstance(action, CreateSelection):
            self._create_selection(
                section_id,
                action.selection,
                supersede_selection_ids=action.supersede_selection_ids,
            )

    def _canvas_ranges(self) -> tuple[SelectionRangeSummary, ...]:
        return self.facade.selection_ranges()

    def _selection_breadcrumb(self, saved: SavedSelection, screen: str) -> str:
        section_number = next(
            (
                section.label
                for section in self.document.sections
                if section.section_id == saved.section_id
            ),
            "",
        )
        return f"PMI Workbench / {section_number} / {screen}"

    def _decomposition_operation_context(
        self,
        saved: SavedSelection,
        *,
        width: int,
        height: int,
        progress: object | None = None,
    ) -> str:
        selection = saved.selection
        lines = [
            self._selection_breadcrumb(saved, "Новый диапазон"),
        ]
        if self.mode_label:
            lines.append(self.mode_label)
        lines.extend(
            [
                "",
                f"Источник: {self.source_name}",
                (
                    f"Диапазон: стр. {selection.start.page_index}:"
                    f"{selection.start.line_number:03d} — "
                    f"стр. {selection.end.page_index}:{selection.end.line_number:03d}"
                ),
                "",
            ]
        )
        if progress is not None:
            completed = int(getattr(progress, "completed_windows", 0))
            total = int(getattr(progress, "total_windows", 0))
            stage = str(getattr(progress, "stage", ""))
            if total:
                lines.extend(
                    [
                        "Большой диапазон: обработка может занять больше времени.",
                        f"Обработано фрагментов: {completed} из {total}",
                        f"Стадия: {stage}",
                        "",
                    ]
                )
        return "\n".join(lines[:height])

    @staticmethod
    def _decomposition_outcome_text(outcome: str, explanation: str) -> str:
        messages = {
            "no_testable_behavior": (
                "В выбранном диапазоне не найдено проверяемого функционального поведения."
            ),
            "insufficient_selection": (
                "Выбранного текста недостаточно для определения границ сценария."
            ),
        }
        body = messages.get(outcome, "Декомпозиция завершена без каркасов.")
        if explanation:
            body += f"\n\nОбъяснение:\n  {explanation}"
        return body
