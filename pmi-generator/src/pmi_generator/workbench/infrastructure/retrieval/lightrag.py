from __future__ import annotations

import asyncio
from typing import Any

from pmi_generator.clients.lightrag import LightRAGClient, LightRAGClientError

from ...application.gap_investigation import (
    RetrievalFragment,
    RetrievalProfile,
    RetrievalResponse,
    TechnicalRetrievalError,
)


class LightRAGRetrieval:
    def __init__(self, client: LightRAGClient) -> None:
        self.client = client

    async def query(self, question: str, profile: RetrievalProfile) -> RetrievalResponse:
        payload = {
            "query": question,
            "mode": "mix",
            "top_k": profile.kg_top_k,
            "chunk_top_k": profile.chunk_top_k,
            "max_entity_tokens": profile.max_entity_tokens,
            "max_relation_tokens": profile.max_relation_tokens,
            "max_total_tokens": profile.max_total_tokens,
            "include_references": True,
            "include_chunk_content": True,
            "enable_rerank": False,
        }
        try:
            raw = await asyncio.to_thread(self.client.query, payload)
        except LightRAGClientError as error:
            raise TechnicalRetrievalError(str(error)) from error
        return self._decode(raw)

    @classmethod
    def _decode(cls, raw: Any) -> RetrievalResponse:
        if isinstance(raw, str):
            return RetrievalResponse(raw, ())
        if not isinstance(raw, dict):
            raise TechnicalRetrievalError("LightRAG вернул ответ неизвестного формата")
        answer = str(raw.get("response") or raw.get("answer") or raw.get("content") or "")
        fragments: list[RetrievalFragment] = []
        for item in cls._items(raw):
            if not isinstance(item, dict):
                continue
            metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
            fragments.append(
                RetrievalFragment(
                    document_id=cls._text(item, metadata, "document_id", "file_name", "source"),
                    document_version=cls._text(item, metadata, "document_version", "version"),
                    page=cls._integer(item, metadata, "page", "page_number"),
                    line_start=cls._integer(item, metadata, "line_start"),
                    line_end=cls._integer(item, metadata, "line_end"),
                    chunk_id=cls._text(item, metadata, "chunk_id", "id"),
                    quote=cls._text(item, metadata, "quote", "content", "text"),
                )
            )
        return RetrievalResponse(answer, tuple(fragments))

    @staticmethod
    def _items(raw: dict[str, Any]) -> list[Any]:
        result: list[Any] = []
        for key in ("references", "chunks", "sources"):
            value = raw.get(key)
            if isinstance(value, list):
                result.extend(value)
        data = raw.get("data")
        if isinstance(data, dict):
            for key in ("references", "chunks", "sources"):
                value = data.get(key)
                if isinstance(value, list):
                    result.extend(value)
        return result

    @staticmethod
    def _text(item: dict[str, Any], metadata: dict[str, Any], *names: str) -> str | None:
        for name in names:
            value = item.get(name, metadata.get(name))
            if value is not None and str(value).strip():
                return str(value)
        return None

    @staticmethod
    def _integer(item: dict[str, Any], metadata: dict[str, Any], *names: str) -> int | None:
        value = LightRAGRetrieval._text(item, metadata, *names)
        try:
            return int(value) if value is not None else None
        except ValueError:
            return None
