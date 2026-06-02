"""No-op reranker — passes through results without reranking."""

from __future__ import annotations

from semhood.providers.reranker.base import RerankerProvider


class NoneReranker(RerankerProvider):
    """Passthrough reranker — returns indices in original order. For use when reranking is disabled."""

    async def rerank(
        self, query: str, documents: list[str], top_n: int = 8
    ) -> list[int]:
        return list(range(min(top_n, len(documents))))
