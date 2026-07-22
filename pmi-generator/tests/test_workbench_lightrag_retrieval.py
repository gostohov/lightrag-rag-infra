from __future__ import annotations

import unittest
from typing import Any

from pmi_generator.workbench.application.gap_investigation import (
    RetrievalBudgetPolicy,
)
from pmi_generator.workbench.infrastructure.retrieval import LightRAGRetrieval


class RecordingLightRAGClient:
    def __init__(self) -> None:
        self.payloads: list[dict[str, Any]] = []

    def query(self, payload: dict[str, Any]) -> dict[str, str]:
        self.payloads.append(payload)
        return {"response": "Ответ LightRAG"}


class LightRAGRetrievalTests(unittest.IsolatedAsyncioTestCase):
    async def test_query_explicitly_disables_unconfigured_reranker(self) -> None:
        client = RecordingLightRAGClient()
        retrieval = LightRAGRetrieval(client)  # type: ignore[arg-type]
        policy = RetrievalBudgetPolicy.defaults()

        await retrieval.query("Узкий вопрос", policy.narrow)
        await retrieval.query("Расширенный вопрос", policy.broad)

        self.assertEqual(len(client.payloads), 2)
        self.assertTrue(
            all(
                payload["enable_rerank"] is False
                for payload in client.payloads
            )
        )
