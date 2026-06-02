"""Voyage AI embedding provider (Voyage Code 2)."""

from __future__ import annotations

import voyageai

from semhood.providers.embeddings.base import EmbeddingProvider


class VoyageProvider(EmbeddingProvider):
    """Voyage Code 2 — best-in-class for code + natural language semantic search."""

    # Voyage Code 2 dimensions
    _DIMENSIONS = {"voyage-code-2": 1536, "voyage-code-3": 1024}

    def __init__(self, api_key: str, model: str = "voyage-code-2"):
        self._model = model
        self._client = voyageai.AsyncClient(api_key=api_key)
        self._dim = self._DIMENSIONS.get(model, 1536)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        # Voyage supports up to 128 texts per batch
        all_vectors = []
        for i in range(0, len(texts), 128):
            batch = texts[i : i + 128]
            result = await self._client.embed(batch, model=self._model)
            all_vectors.extend(result.embeddings)

        return all_vectors

    async def embed_query(self, text: str) -> list[float]:
        result = await self._client.embed([text], model=self._model)
        return result.embeddings[0]

    @property
    def dimension(self) -> int:
        return self._dim
