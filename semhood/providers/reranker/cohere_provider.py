"""Cohere reranker provider."""

from __future__ import annotations

import cohere

from semhood.providers.reranker.base import RerankerProvider


class CohereRerankerProvider(RerankerProvider):
    """Cohere Rerank — cross-encoder reranking via API."""

    def __init__(self, api_key: str, model: str = "rerank-english-v3.0"):
        self._model = model
        self._client = cohere.AsyncClientV2(api_key=api_key)

    async def rerank(
        self, query: str, documents: list[str], top_n: int = 8
    ) -> list[int]:
        if not documents:
            return []

        response = await self._client.rerank(
            model=self._model,
            query=query,
            documents=documents,
            top_n=min(top_n, len(documents)),
        )

        return [r.index for r in response.results]
