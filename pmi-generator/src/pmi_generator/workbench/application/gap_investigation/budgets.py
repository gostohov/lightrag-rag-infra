from __future__ import annotations

from .models import RetrievalProfile


class RetrievalBudgetPolicy:
    def __init__(self, narrow: RetrievalProfile, broad: RetrievalProfile) -> None:
        self.narrow = narrow
        self.broad = broad

    @classmethod
    def defaults(cls) -> RetrievalBudgetPolicy:
        return cls(
            narrow=RetrievalProfile(
                name="узкий поиск",
                kg_top_k=12,
                chunk_top_k=8,
                max_entity_tokens=2500,
                max_relation_tokens=3000,
                max_total_tokens=10000,
            ),
            broad=RetrievalProfile(
                name="расширенный поиск",
                kg_top_k=40,
                chunk_top_k=20,
                max_entity_tokens=6000,
                max_relation_tokens=8000,
                max_total_tokens=30000,
            ),
        )

