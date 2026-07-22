from __future__ import annotations

import io
import unittest
from unittest.mock import Mock, patch

from pmi_generator.workbench.application.conversation import (
    ConversationAction,
    ConversationEffect,
    ConversationToolCall,
    ConversationTurnDecision,
    ConversationTurnKind,
    action_effect,
    requires_confirmation,
)
from pmi_generator.workbench.application.session import SessionService
from pmi_generator.workbench.infrastructure.storage import (
    InMemoryDatabase,
    InMemoryUnitOfWork,
)
from pmi_generator.workbench.presentation.session.shell import TerminalSessionShell


class ConversationDecisionContractTests(unittest.TestCase):
    def test_turn_is_exactly_answer_clarification_or_one_tool_call(self) -> None:
        answer = ConversationTurnDecision(
            ConversationTurnKind.ANSWER,
            "Текущее поле не заполнено.",
        )
        clarification = ConversationTurnDecision(
            ConversationTurnKind.CLARIFICATION,
            "Вы хотите изменить карточку или только обсудить вариант?",
        )
        tool = ConversationTurnDecision(
            ConversationTurnKind.TOOL_CALL,
            "Исследую текущий пробел.",
            ConversationToolCall(
                ConversationAction.RESEARCH_GAP,
                {"question": "Можно ли прочитать PTH через GET DATA?"},
            ),
        )

        self.assertIsNone(answer.tool_call)
        self.assertIsNone(clarification.tool_call)
        self.assertEqual(tool.tool_call.action, ConversationAction.RESEARCH_GAP)
        with self.assertRaises(ValueError):
            ConversationTurnDecision(
                ConversationTurnKind.TOOL_CALL,
                "Продолжаю.",
            )
        with self.assertRaises(ValueError):
            ConversationTurnDecision(
                ConversationTurnKind.ANSWER,
                "Объяснение.",
                ConversationToolCall(ConversationAction.EXPORT_DIAGNOSTICS),
            )

    def test_side_effect_and_confirmation_matrix_is_explicit(self) -> None:
        self.assertIs(
            action_effect(ConversationAction.PROPOSE_DESIGN_DECISION),
            ConversationEffect.READ_ONLY,
        )
        self.assertIs(
            action_effect(ConversationAction.RESEARCH_GAP),
            ConversationEffect.EXPENSIVE,
        )
        self.assertIs(
            action_effect(ConversationAction.SUBMIT_ANALYST_ANSWER),
            ConversationEffect.MUTATING,
        )
        self.assertFalse(
            requires_confirmation(
                ConversationAction.SUBMIT_ANALYST_ANSWER
            )
        )
        self.assertTrue(
            requires_confirmation(
                ConversationAction.CONFIRM_ANALYST_ANSWER
            )
        )
        self.assertFalse(requires_confirmation(ConversationAction.RESEARCH_GAP))
        self.assertTrue(requires_confirmation(ConversationAction.LEAVE_GAP))
        self.assertTrue(requires_confirmation(ConversationAction.INCLUDE_CARD))
        self.assertTrue(requires_confirmation(ConversationAction.EXCLUDE_CARD))


class LegacyRouteCharacterizationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.database = InMemoryDatabase()
        self.service = SessionService(
            uow_factory=lambda: InMemoryUnitOfWork(self.database)
        )
        self.service.open("SESSION_1", "SELECTION_1", "CARD_1")

    def test_plain_input_is_saved_as_analyst_message_before_handler(self) -> None:
        seen: list[str] = []
        shell = TerminalSessionShell(
            self.service,
            "SESSION_1",
            output=io.StringIO(),
            message_handler=seen.append,
        )

        message_id = shell.controller.submit("Используйте GET DATA.")
        shell.message_handler(message_id)

        self.assertEqual(seen, ["MSG_000001"])
        self.assertEqual(self.service.history("SESSION_1")[-1].text, "Используйте GET DATA.")

    def test_continue_without_text_uses_dedicated_handler(self) -> None:
        handler = Mock()
        shell = TerminalSessionShell(
            self.service,
            "SESSION_1",
            output=io.StringIO(),
            command_handlers={"/continue": handler},
        )

        self.assertTrue(shell.handle_command("/continue"))

        handler.assert_called_once_with()
        self.assertEqual(self.service.history("SESSION_1"), [])


if __name__ == "__main__":
    unittest.main()
