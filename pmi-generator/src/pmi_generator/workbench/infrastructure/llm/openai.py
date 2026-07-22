from __future__ import annotations

import asyncio
import json
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from pmi_generator.clients.http_transport import build_url_opener

from ...application.llm import RawCompletion, TechnicalLlmError
from ...application.prompting import PromptCall


_RESPONSE_PREVIEW_LIMIT = 2000


@dataclass(frozen=True, slots=True)
class OpenAITransportSettings:
    base_url: str
    model: str
    api_key: str | None = None
    timeout: float = 300.0
    verify_ssl: bool = True
    no_proxy: bool = False


class OpenAICompatibleTransport:
    def __init__(self, settings: OpenAITransportSettings) -> None:
        self.settings = settings
        self.opener = build_url_opener(
            verify_ssl=settings.verify_ssl,
            ca_file=None,
            no_proxy=settings.no_proxy,
        )

    async def complete(
        self,
        call: PromptCall,
        tools: list[dict[str, Any]],
    ) -> RawCompletion:
        return await asyncio.to_thread(self._complete_sync, call, tools)

    def public_metadata(self) -> dict[str, Any]:
        return {"base_url": self.settings.base_url, "model": self.settings.model}

    def _complete_sync(
        self,
        call: PromptCall,
        tools: list[dict[str, Any]],
    ) -> RawCompletion:
        payload = {
            "model": self.settings.model,
            "messages": [
                {"role": "system", "content": call.system_prompt},
                {
                    "role": "user",
                    "content": json.dumps(call.context, ensure_ascii=False, sort_keys=True),
                },
            ],
            **call.generation_parameters,
        }
        if tools:
            payload.update(
                {
                    "tools": tools,
                    "tool_choice": "required",
                    "parallel_tool_calls": False,
                }
            )
        request = urllib.request.Request(
            self._endpoint(),
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=self._headers(),
            method="POST",
        )
        try:
            with self.opener.open(
                request,
                timeout=self.settings.timeout,
            ) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise TechnicalLlmError(
                f"vLLM HTTP {error.code}: {detail}",
                retryable=error.code == 429 or error.code >= 500,
            ) from error
        except (urllib.error.URLError, TimeoutError, socket.timeout) as error:
            raise TechnicalLlmError(f"vLLM timeout/network error: {error}", retryable=True) from error
        except json.JSONDecodeError as error:
            raise TechnicalLlmError("vLLM вернул невалидный JSON", retryable=False) from error
        return self._decode(body)

    @staticmethod
    def _decode(body: dict[str, Any]) -> RawCompletion:
        try:
            choice = body["choices"][0]
            message = choice["message"]
        except (KeyError, IndexError, TypeError) as error:
            raise TechnicalLlmError("Ответ vLLM не содержит choices[0].message", retryable=False) from error
        calls = tuple(
            {
                "id": item.get("id", ""),
                "name": item.get("function", {}).get("name", ""),
                "arguments": item.get("function", {}).get("arguments", {}),
            }
            for item in message.get("tool_calls") or []
        )
        content = message.get("content")
        text = content if isinstance(content, str) else ""
        return RawCompletion(
            finish_reason=str(choice.get("finish_reason", "")),
            tool_calls=calls,
            usage=dict(body.get("usage") or {}),
            model=str(body.get("model", "")),
            content=text,
            response_preview=_response_preview(
                message,
                include_tool_calls=(
                    str(choice.get("finish_reason", "")) == "length"
                ),
            ),
        )

    def _endpoint(self) -> str:
        base = self.settings.base_url.rstrip("/")
        return base if base.endswith("/chat/completions") else f"{base}/chat/completions"

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.settings.api_key:
            headers["Authorization"] = f"Bearer {self.settings.api_key}"
        return headers


def _response_preview(
    message: dict[str, Any],
    *,
    include_tool_calls: bool = False,
) -> dict[str, Any]:
    preview: dict[str, Any] = {}
    content = message.get("content")
    if content:
        text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
        preview["content"] = text[:_RESPONSE_PREVIEW_LIMIT]
        preview["content_truncated"] = len(text) > _RESPONSE_PREVIEW_LIMIT
    reasoning = message.get("reasoning_content")
    if reasoning:
        text = reasoning if isinstance(reasoning, str) else json.dumps(reasoning, ensure_ascii=False)
        preview["reasoning_content_present"] = True
        preview["reasoning_content_chars"] = len(text)
    tool_calls = message.get("tool_calls") if include_tool_calls else None
    if isinstance(tool_calls, list) and tool_calls:
        bounded_calls: list[dict[str, Any]] = []
        for item in tool_calls:
            function = item.get("function", {}) if isinstance(item, dict) else {}
            arguments = (
                function.get("arguments", "")
                if isinstance(function, dict)
                else ""
            )
            text = (
                arguments
                if isinstance(arguments, str)
                else json.dumps(arguments, ensure_ascii=False)
            )
            bounded_calls.append(
                {
                    "id": str(item.get("id", "")) if isinstance(item, dict) else "",
                    "name": (
                        str(function.get("name", ""))
                        if isinstance(function, dict)
                        else ""
                    ),
                    "arguments": text[:_RESPONSE_PREVIEW_LIMIT],
                    "arguments_truncated": (
                        len(text) > _RESPONSE_PREVIEW_LIMIT
                    ),
                }
            )
        preview["tool_calls_count"] = len(tool_calls)
        preview["tool_calls"] = bounded_calls
    return preview
