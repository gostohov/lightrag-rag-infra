from __future__ import annotations

import os
import tempfile
import unittest
from dataclasses import dataclass, replace
from pathlib import Path

from pmi_generator.workbench.application.llm import LlmToolRuntime, ToolSpec, TypedToolRegistry
from pmi_generator.workbench.application.prompting import PromptId, default_policy
from pmi_generator.workbench.infrastructure.llm import (
    OpenAICompatibleTransport,
    OpenAITransportSettings,
)
from pmi_generator.workbench.infrastructure.storage import SqliteUnitOfWork


@dataclass(frozen=True, slots=True)
class SmokeResult:
    status: str


@unittest.skipUnless(os.getenv("PMI_VLLM_SMOKE") == "1", "opt-in vLLM smoke test")
class VllmSmokeTest(unittest.IsolatedAsyncioTestCase):
    async def test_native_tool_call(self) -> None:
        registry = TypedToolRegistry()
        registry.register(
            ToolSpec(
                name="submit_smoke_result",
                description="Вернуть status=ok",
                arguments_type=SmokeResult,
                json_schema={
                    "type": "object",
                    "properties": {"status": {"type": "string"}},
                    "required": ["status"],
                    "additionalProperties": False,
                },
            )
        )
        call = replace(
            default_policy().build_call(
                PromptId.DECOMPOSITION,
                {"selection": "Вызови submit_smoke_result со status=ok."},
            ),
            allowed_tools=("submit_smoke_result",),
        )
        with tempfile.TemporaryDirectory() as tmp:
            database = Path(tmp) / "workbench.sqlite3"
            runtime = LlmToolRuntime(
                transport=OpenAICompatibleTransport(
                    OpenAITransportSettings(
                        base_url=os.environ["PMI_LLM_URL"],
                        model=os.environ["PMI_LLM_MODEL"],
                        api_key=os.getenv("PMI_LLM_API_KEY"),
                        verify_ssl=os.getenv("PMI_INSECURE") != "1",
                    )
                ),
                tools=registry,
                uow_factory=lambda: SqliteUnitOfWork(database),
            )
            result = await runtime.invoke("SMOKE_ATTEMPT", "SMOKE_SESSION", call)

        self.assertEqual(result.arguments, SmokeResult(status="ok"))
