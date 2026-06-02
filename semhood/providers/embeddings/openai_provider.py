"""OpenAI embedding provider."""

from __future__ import annotations

from openai import AsyncOpenAI

from semhood.providers.embeddings.base import EmbeddingProvider


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI text-embedding models (text-embedding-3-small, text-embedding-3-large)."""

    _DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        self._model = model
        self._client = AsyncOpenAI(api_key=api_key)
        self._dim = self._DIMENSIONS.get(model, 1536)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        all_vectors = []
        # OpenAI supports up to 2048 texts per batch
        for i in range(0, len(texts), 2048):
            batch = texts[i : i + 2048]
            response = await self._client.embeddings.create(
                model=self._model,
                input=batch,
            )
            all_vectors.extend([d.embedding for d in response.data])

        return all_vectors

    async def embed_query(self, text: str) -> list[float]:
        response = await self._client.embeddings.create(
            model=self._model,
            input=text,
        )
        return response.data[0].embedding

    @property
    def dimension(self) -> int:
        return self._dim
