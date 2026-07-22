from __future__ import annotations

from collections.abc import Awaitable, Callable

from .models import GapInvestigationResult


class GapQueue:
    async def run(
        self,
        gap_ids: tuple[str, ...],
        worker: Callable[[str], Awaitable[GapInvestigationResult]],
    ) -> tuple[GapInvestigationResult, ...]:
        results: list[GapInvestigationResult] = []
        for gap_id in gap_ids:
            results.append(await worker(gap_id))
        return tuple(results)

