"""Stage 1 structural builder.

Builds the always-present `code` dense vector and the BM25 sparse vector
for a chunk. No LLM, no network calls — just hashing + embedding the raw
source.

Also exposes ``build_description`` (formerly on ``IndexingStrategy``) as
a free function used by the Stage 2 enricher to produce the text that
gets embedded into the ``description`` vector.
"""

from __future__ import annotations

import hashlib

from semhood.core.models import Chunk
from semhood.providers.embeddings.base import EmbeddingProvider

# =============================================================================
# Public helpers
# =============================================================================


def build_description(chunk: Chunk) -> str:
    """Rich semantic description text for embedding into the description vector.

    Combines: identity (name/class/namespace), docblock, logic_summary,
    parameters, return type, and called functions.
    """
    parts: list[str] = []

    # Identity
    if chunk.class_name:
        parts.append(f"Method {chunk.name} in class {chunk.class_name}")
    else:
        parts.append(f"Function {chunk.name}")

    if chunk.namespace:
        parts.append(f"in {chunk.namespace}")

    # Docblock
    if chunk.docblock:
        parts.append(f". {chunk.docblock}")

    # Logic summary (post-enrichment)
    if chunk.logic_summary:
        parts.append(f". {chunk.logic_summary}")

    # Signature
    if chunk.parameters:
        param_str = ", ".join(
            f"{p.name}: {p.type}" if p.type else p.name
            for p in chunk.parameters
        )
        parts.append(f". Parameters: {param_str}")

    if chunk.return_type:
        parts.append(f". Returns: {chunk.return_type}")

    # Calls
    if chunk.calls:
        parts.append(f". Calls: {', '.join(chunk.calls[:10])}")

    return " ".join(parts)


# =============================================================================
# StructuralBuilder
# =============================================================================


class StructuralBuilder:
    """Builds the always-present `code` vector and BM25 sparse vector.

    No LLM. No I/O outside the embedder. Pure deterministic structural
    representation — what the indexer can produce on every run, fast and
    free.
    """

    def source_hash(self, chunk: Chunk) -> str:
        """sha256 of chunk.source — used to detect whether source changed."""
        return hashlib.sha256((chunk.source or "").encode("utf-8")).hexdigest()

    async def build_code_vector(
        self,
        chunk: Chunk,
        embedder: EmbeddingProvider,
        max_chars: int = 8000,
    ) -> list[float]:
        """Embed the raw source code (truncated to fit token limits)."""
        text = (chunk.source or chunk.name)[:max_chars]
        vectors = await embedder.embed_batch([text])
        return vectors[0]

    def build_sparse(self, chunk: Chunk) -> dict:
        """Build BM25 sparse vector from chunk metadata.

        Same logic that lived inside ``Indexer._build_sparse`` previously.
        """
        text_parts = [
            chunk.name,
            chunk.class_name or "",
            chunk.file,
            " ".join(chunk.calls[:20]),
            chunk.namespace or "",
        ]
        text = " ".join(text_parts).lower()

        tokens = text.split()
        term_freq: dict[str, int] = {}
        for token in tokens:
            term_freq[token] = term_freq.get(token, 0) + 1

        if not term_freq:
            return {"indices": [], "values": []}

        indices: list[int] = []
        values: list[float] = []
        for token, freq in term_freq.items():
            idx = int(hashlib.md5(token.encode()).hexdigest()[:8], 16) % 100000
            indices.append(idx)
            values.append(float(freq))

        return {"indices": indices, "values": values}
