"""Shared enrichment-write helper.

Both the direct-API ``Enricher`` (Stage 2 worker) and the MCP-driven
``save_enrichment`` tool need the same final step: take a logic_summary
+ developer_queries pair, embed them, patch the description /
developer_queries vectors, and flip the chunk's state to ``fresh``.

This module centralizes that logic so the two callers stay in sync.
"""

from __future__ import annotations

from datetime import UTC, datetime

from semhood.core.enrichment_state import FRESH
from semhood.core.enrichment_store import EnrichmentStore
from semhood.core.models import Chunk
from semhood.core.structural import build_description
from semhood.providers.embeddings.base import EmbeddingProvider
from semhood.providers.vector_db.base import VectorDBProvider


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def payload_to_chunk(payload: dict) -> Chunk:
    """Coerce a stored payload back into a Chunk for re-rendering.

    Pydantic ignores unknown fields and validates the rest. Any field
    saved by an older version that isn't on the current model is dropped.
    """
    fields = set(Chunk.model_fields.keys())
    safe = {k: v for k, v in (payload or {}).items() if k in fields}
    return Chunk(**safe)


async def write_enrichment(
    *,
    vector_db: VectorDBProvider,
    embedder: EmbeddingProvider,
    point_id: str,
    chunk: Chunk,
    summary: str,
    queries: list[str],
    version: int,
    final_state: str = FRESH,
    store: EnrichmentStore | None = None,
) -> None:
    """Persist an enrichment result for a single chunk.

    Embeds the description text + developer queries, patches both vectors
    and the matching payload fields atomically. When ``store`` is given, the
    model-agnostic text is also mirrored to the portable enrichment file
    (keyed by ``source_hash``) so it survives embedding-model changes and can
    be shared. This is the single choke point both enrichment routes — the
    ``Enricher`` worker and the MCP ``save_enrichment`` tool — pass through.
    """
    chunk.logic_summary = summary
    chunk.developer_queries = queries

    description_text = build_description(chunk)
    queries_text = "\n".join(queries) if queries else chunk.name

    emb = await embedder.embed_batch([description_text, queries_text])

    await vector_db.update_vectors_and_payload(
        point_id=point_id,
        vectors={
            "description": emb[0],
            "developer_queries": emb[1],
        },
        payload={
            "logic_summary": summary,
            "description": description_text,
            "developer_queries": queries,
            "enrichment_state": final_state,
            "enrichment_version": version,
            "enriched_at": now_iso(),
            "enrichment_error": None,
        },
    )

    # Mirror the text to the portable, committed store. Keyed by content, so it
    # outlives the (model-specific) vectors above. Only real LLM enrichment is
    # persisted — trivial/deterministic summaries are free to recompute on
    # read-back, so storing them would only bloat the committed file.
    if store is not None and chunk.source_hash and final_state == FRESH:
        await store.put(
            chunk.source_hash,
            logic_summary=summary,
            developer_queries=queries,
            enrichment_version=version,
        )
