from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from ..domain import ExecutionMode


@dataclass(frozen=True, slots=True)
class WorkbenchSettings:
    run_dir: Path
    llm_url: str | None
    llm_model: str | None
    llm_api_key: str | None
    lightrag_url: str | None
    lightrag_api_key: str | None
    llm_timeout: float
    retrieval_timeout: float
    verify_ssl: bool
    no_proxy: bool
    execution_mode: ExecutionMode = ExecutionMode.PRODUCTION

    @classmethod
    def from_environment(
        cls,
        run_dir: Path,
        *,
        execution_mode: ExecutionMode = ExecutionMode.PRODUCTION,
    ) -> WorkbenchSettings:
        return cls(
            run_dir=run_dir,
            llm_url=os.getenv("PMI_LLM_URL"),
            llm_model=os.getenv("PMI_LLM_MODEL"),
            llm_api_key=os.getenv("PMI_LLM_API_KEY"),
            lightrag_url=os.getenv("PMI_LIGHTRAG_URL"),
            lightrag_api_key=os.getenv("PMI_LIGHTRAG_API_KEY"),
            llm_timeout=float(os.getenv("PMI_TIMEOUT", "600")),
            retrieval_timeout=float(os.getenv("PMI_LIGHTRAG_QUERY_TIMEOUT", "900")),
            verify_ssl=os.getenv("PMI_INSECURE", "0") not in {"1", "true", "yes"},
            no_proxy=os.getenv("PMI_NO_PROXY", "0") in {"1", "true", "yes"},
            execution_mode=execution_mode,
        )

    def require_llm(self) -> tuple[str, str]:
        if not self.llm_url or not self.llm_model:
            raise ValueError("Для LLM нужны PMI_LLM_URL и PMI_LLM_MODEL")
        return self.llm_url, self.llm_model

    def require_lightrag(self) -> str:
        if not self.lightrag_url:
            raise ValueError("Для retrieval нужен PMI_LIGHTRAG_URL")
        return self.lightrag_url
