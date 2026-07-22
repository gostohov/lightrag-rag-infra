from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from .http_transport import build_url_opener


class LightRAGClientError(RuntimeError):
    pass


class LightRAGClient:
    def __init__(
        self,
        base_url: str,
        api_key: str | None,
        timeout: float,
        verify_ssl: bool,
        ca_file: Path | None,
        no_proxy: bool,
    ) -> None:
        self.query_data_endpoint = _query_data_endpoint(base_url)
        self.query_endpoint = _query_endpoint(base_url)
        self.api_key = api_key
        self.timeout = timeout
        self.opener = build_url_opener(verify_ssl=verify_ssl, ca_file=ca_file, no_proxy=no_proxy)

    def query_data(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._post_json(self.query_data_endpoint, payload)

    def query(self, payload: dict[str, Any]) -> Any:
        return self._post_json(self.query_endpoint, payload)

    def _post_json(self, endpoint: str, payload: dict[str, Any]) -> Any:
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=self._headers(),
            method="POST",
        )
        try:
            with self.opener.open(request, timeout=self.timeout) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise LightRAGClientError(f"HTTP {exc.code}: {error_body}") from exc
        except urllib.error.URLError as exc:
            raise LightRAGClientError(str(exc.reason)) from exc
        except TimeoutError as exc:
            raise LightRAGClientError("request timed out") from exc

        try:
            return json.loads(body)
        except json.JSONDecodeError as exc:
            raise LightRAGClientError(f"invalid JSON response: {exc}") from exc

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        return headers


def _query_data_endpoint(base_url: str) -> str:
    value = _api_base_url(base_url)
    return f"{value}/query/data"


def _query_endpoint(base_url: str) -> str:
    value = _api_base_url(base_url)
    return f"{value}/query"


def _api_base_url(base_url: str) -> str:
    value = base_url.rstrip("/")
    if value.endswith("/query/data"):
        return value[: -len("/query/data")]
    if value.endswith("/query"):
        return value[: -len("/query")]
    return value
