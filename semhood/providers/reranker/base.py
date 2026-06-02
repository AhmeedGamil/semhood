"""Abstract interface for reranker providers."""

from __future__ import annotations

from abc import ABC, abstractmethod


class RerankerProvider(ABC):
    """
    Abstract base class for reranking search results.

    Reranking takes a query + list of candidate documents and re-orders them
    by relevance using a cross-encoder model. This is optional — set provider
    to "none" to skip reranking entirely.
    """

    @abstractmethod
    async def rerank(
        self, query: str, documents: list[str], top_n: int = 8
    ) -> list[int]:
        """
        Rerank documents by relevance to query.

        Returns indices of the top_n most relevant documents, sorted by relevance.
        """
        ...
