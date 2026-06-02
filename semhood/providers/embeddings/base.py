"""Abstract interface for embedding providers."""

from __future__ import annotations

from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    """
    Abstract base class for text/code embedding.

    All implementations must support batch embedding (for indexing efficiency)
    and single-query embedding (for search).
    """

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts. Returns list of vectors."""
        ...

    @abstractmethod
    async def embed_query(self, text: str) -> list[float]:
        """Embed a single query string. Returns one vector."""
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Vector dimension size (e.g. 1536 for Voyage Code 2)."""
        ...
