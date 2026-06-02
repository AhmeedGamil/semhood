"""Local embedding provider using sentence-transformers (HuggingFace)."""

from __future__ import annotations

from semhood.providers.embeddings.base import EmbeddingProvider


class LocalEmbeddingProvider(EmbeddingProvider):
    """
    Local embedding using sentence-transformers.

    Requires: pip install sentence-transformers torch
    Runs fully offline — no API calls, no cost.

    ``trust_remote_code`` must be enabled for models that ship custom
    modeling code on HuggingFace (e.g. ``jinaai/jina-embeddings-v2-base-code``).
    Only enable it for models you trust.
    """

    def __init__(
        self,
        model: str = "sentence-transformers/all-mpnet-base-v2",
        device: str = "cpu",
        trust_remote_code: bool = False,
    ):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "Local embeddings require 'sentence-transformers'. "
                "Install with: pip install sentence-transformers torch"
            ) from exc

        self._model_name = model
        self._model = SentenceTransformer(
            model, device=device, trust_remote_code=trust_remote_code
        )
        self._dim = self._model.get_sentence_embedding_dimension()

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        # sentence-transformers is synchronous, but fast on CPU/GPU
        embeddings = self._model.encode(texts, show_progress_bar=False)
        return [vec.tolist() for vec in embeddings]

    async def embed_query(self, text: str) -> list[float]:
        embedding = self._model.encode(text, show_progress_bar=False)
        return embedding.tolist()

    @property
    def dimension(self) -> int:
        return self._dim
