"""Per-project operations, decoupled from any transport.

A :class:`ProjectEngine` is everything semhood can do to *one* project's index:
search, symbol lookup, status, indexing, enrichment. It holds a vector store
for that project plus references to **shared** providers (the embedder, LLM,
reranker, parser) that the daemon loads once and hands to every project.

Why this layer exists: search logic shouldn't know whether it was reached via
the CLI, the MCP server, or HTTP. The daemon owns a registry of these engines
(one per project, all sharing one warm embedder); the CLI and MCP server are
thin clients that call the daemon. Keeping the logic here — not in the MCP
server — means it's testable without any transport and reusable everywhere.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from semhood.config import SemhoodConfig
from semhood.core.enrichment_state import ALL_STATES, FRESH, PENDING
from semhood.core.enrichment_writer import payload_to_chunk, write_enrichment

logger = logging.getLogger("semhood.engine")


# =============================================================================
# Result formatting — keep it terse so AI agents don't blow their context
# =============================================================================


def hit_to_dict(hit: dict, *, include_source: bool = False) -> dict:
    p = hit.get("payload") or {}
    out = {
        "score": round(float(hit.get("score", 0.0)), 4),
        "name": p.get("name"),
        "class_name": p.get("class_name"),
        "qualified_name": (
            f"{p['class_name']}::{p['name']}"
            if p.get("class_name") and p.get("name") else p.get("name")
        ),
        "file": p.get("file"),
        "line_start": p.get("line_start"),
        "line_end": p.get("line_end"),
        "language": p.get("language"),
        "type": p.get("type"),
        "enrichment_state": p.get("enrichment_state"),
        "summary": p.get("logic_summary") or p.get("docblock"),
    }
    if include_source:
        out["source"] = p.get("source")
        out["calls"] = p.get("calls", [])
        out["called_by"] = p.get("called_by", [])
    return out


def rrf_merge(per_vector: dict[str, list[dict]], k: int = 60) -> list[dict]:
    """Reciprocal-rank-fusion across the named vectors into one ranked list."""
    scores: dict[str, float] = {}
    items: dict[str, dict] = {}
    for hits in per_vector.values():
        for rank, hit in enumerate(hits):
            hid = hit["id"]
            scores[hid] = scores.get(hid, 0.0) + 1.0 / (k + rank + 1)
            items[hid] = hit
    sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)
    return [{**items[hid], "rrf_score": round(scores[hid], 4)} for hid in sorted_ids]


def pending_chunk_to_dict(point: dict, max_source_chars: int) -> dict:
    p = point.get("payload") or {}
    src = p.get("source") or ""
    truncated = len(src) > max_source_chars
    if truncated:
        src = src[:max_source_chars]
    return {
        "id": point["id"],
        "name": p.get("name"),
        "class_name": p.get("class_name"),
        "qualified_name": (
            f"{p['class_name']}::{p['name']}"
            if p.get("class_name") and p.get("name") else p.get("name")
        ),
        "file": p.get("file"),
        "language": p.get("language"),
        "line_start": p.get("line_start"),
        "line_end": p.get("line_end"),
        "type": p.get("type"),
        "docblock": p.get("docblock"),
        "parameters": p.get("parameters", []),
        "return_type": p.get("return_type"),
        "calls": p.get("calls", []),
        "called_by": p.get("called_by", []),
        "source": src,
        "source_truncated": truncated,
    }


# =============================================================================
# ProjectEngine
# =============================================================================


class ProjectEngine:
    """All operations on a single project's index.

    Search / symbol / enrichment-tool methods need only ``embedder`` and
    ``vector_db``. The heavier ``index``/``enrich``/``query`` methods lazily
    assemble an Indexer/Enricher/QueryPipeline from the shared providers passed
    in at construction; they raise if those providers weren't supplied.
    """

    def __init__(
        self,
        *,
        config: SemhoodConfig,
        embedder: Any,
        vector_db: Any,
        root: str | None = None,
        llm: Any = None,
        reranker: Any = None,
        parser: Any = None,
        structural: Any = None,
        parsing_policy: Any = None,
    ):
        self.config = config
        self.embedder = embedder
        self.vector_db = vector_db
        self.root = root
        self._llm = llm
        self._reranker = reranker
        self._parser = parser
        self._structural = structural
        self._parsing_policy = parsing_policy

        # Lazily built heavy components (need the shared providers above).
        self._indexer = None
        self._enricher = None
        self._pipeline = None
        self._enrichment_store = None

    # ------------------------------------------------------------------
    # Search & lookup (embedder + vector_db only)
    # ------------------------------------------------------------------

    async def search(
        self, query: str, top_k: int = 8, vectors: list[str] | None = None
    ) -> dict[str, Any]:
        if not query or not isinstance(query, str):
            raise ValueError("'query' must be a non-empty string")
        top_k = int(top_k)
        vectors = vectors or ["code", "description", "developer_queries"]

        qvec = await self.embedder.embed_query(query)
        return {
            "query": query,
            "embedding_dim": self.embedder.dimension,
            "results": await self._search_one(qvec, top_k, vectors),
        }

    async def search_many(
        self, queries: list[str], top_k: int = 8, vectors: list[str] | None = None
    ) -> dict[str, Any]:
        """Run several queries in one call.

        Embeds every query in a single batch, then runs each query's
        multi-vector search concurrently, returning results grouped per query
        (order preserved). Saves an agent N round-trips when it wants to look up
        several things at once.
        """
        if not isinstance(queries, list) or not queries:
            raise ValueError("'queries' must be a non-empty list")
        if not all(isinstance(q, str) and q for q in queries):
            raise ValueError("each query must be a non-empty string")
        top_k = int(top_k)
        vectors = vectors or ["code", "description", "developer_queries"]

        qvecs = await self.embedder.embed_batch(queries)
        per_query = await asyncio.gather(
            *(self._search_one(qv, top_k, vectors) for qv in qvecs)
        )
        return {
            "embedding_dim": self.embedder.dimension,
            "results": [
                {"query": q, "results": r}
                for q, r in zip(queries, per_query, strict=True)
            ],
        }

    async def _search_one(
        self, query_vector: list[float], top_k: int, vectors: list[str]
    ) -> list[dict]:
        """Multi-vector search + RRF fusion for a single embedded query."""
        per_vector: dict[str, list[dict]] = {}
        for vname in vectors:
            try:
                per_vector[vname] = await self.vector_db.search_dense(
                    vname, query_vector, top_k=top_k
                )
            except Exception as exc:
                logger.warning("search on vector %r failed: %s", vname, exc)
                per_vector[vname] = []
        merged = rrf_merge(per_vector)[:top_k]
        return [hit_to_dict(h) for h in merged]

    async def find_symbol(
        self, name: str, class_name: str | None = None
    ) -> dict[str, Any]:
        if not name or not isinstance(name, str):
            raise ValueError("'name' must be a non-empty string")
        points = await self.vector_db.get_points_by_name(name, class_name)
        return {
            "name": name,
            "class_name": class_name,
            "matches": [
                hit_to_dict({"payload": p["payload"], "id": p["id"]}) for p in points
            ],
        }

    async def get_chunk_context(
        self, name: str, class_name: str | None = None
    ) -> dict[str, Any]:
        if not name or not isinstance(name, str):
            raise ValueError("'name' must be a non-empty string")
        points = await self.vector_db.get_points_by_name(name, class_name)
        if not points:
            return {"name": name, "class_name": class_name, "found": False}

        point = points[0]
        return {
            "name": name,
            "class_name": class_name,
            "found": True,
            "chunk": hit_to_dict(
                {"payload": point["payload"], "id": point["id"]}, include_source=True
            ),
            "alternative_matches": len(points) - 1,
        }

    async def index_status(self) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for st in ALL_STATES:
            try:
                pts = await self.vector_db.scroll_by_payload(
                    {"enrichment_state": st}, limit=100000
                )
                counts[st] = len(pts)
            except Exception as exc:
                logger.warning("status scroll for %r failed: %s", st, exc)
                counts[st] = -1
        return {
            "root": self.root,
            "vector_db": self.config.vector_db.provider,
            "embedder": self.config.embeddings.provider,
            "embedding_dim": self.embedder.dimension,
            "enrichment_enabled": self.config.enrichment.enabled,
            "enrichment_version": self.config.enrichment.version,
            "counts": counts,
            "total": sum(c for c in counts.values() if c >= 0),
        }

    # ------------------------------------------------------------------
    # Agent-driven enrichment (embedder + vector_db only)
    # ------------------------------------------------------------------

    async def list_pending_enrichments(
        self, limit: int = 5, max_source_chars: int = 4000
    ) -> dict[str, Any]:
        limit = int(limit)
        if limit < 1 or limit > 50:
            raise ValueError("'limit' must be between 1 and 50")
        max_source_chars = int(max_source_chars)
        if max_source_chars < 200:
            raise ValueError("'max_source_chars' must be >= 200")

        points = await self.vector_db.scroll_by_payload(
            {"enrichment_state": PENDING}, limit=limit
        )
        return {
            "count": len(points),
            "chunks": [pending_chunk_to_dict(p, max_source_chars) for p in points],
            "next_steps": (
                "For each chunk, write a 1-3 sentence logic_summary and 5-10 short "
                "developer_queries describing what a developer would type to find "
                "this code. Then call save_enrichment(chunk_id=...) with both. Loop "
                "until enrichment_progress() reports zero pending."
            ),
        }

    async def save_enrichment(
        self, chunk_id: str, logic_summary: str, developer_queries: list[str] | None
    ) -> dict[str, Any]:
        if not chunk_id or not isinstance(chunk_id, str):
            raise ValueError("'chunk_id' must be a non-empty string")
        if not logic_summary or not isinstance(logic_summary, str):
            raise ValueError("'logic_summary' must be a non-empty string")
        queries = developer_queries or []
        if not isinstance(queries, list) or not all(isinstance(q, str) for q in queries):
            raise ValueError("'developer_queries' must be a list of strings")

        payload_map = await self.vector_db.get_by_ids([chunk_id])
        if chunk_id not in payload_map:
            raise ValueError(f"chunk_id not found: {chunk_id!r}")

        chunk = payload_to_chunk(payload_map[chunk_id])
        await write_enrichment(
            vector_db=self.vector_db,
            embedder=self.embedder,
            point_id=chunk_id,
            chunk=chunk,
            summary=logic_summary,
            queries=queries,
            version=self.config.enrichment.version,
            final_state=FRESH,
            store=self._get_enrichment_store(),
        )
        return {
            "chunk_id": chunk_id,
            "qualified_name": chunk.qualified_name(),
            "state": FRESH,
            "queries_stored": len(queries),
        }

    async def enrichment_progress(self) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for st in ALL_STATES:
            try:
                pts = await self.vector_db.scroll_by_payload(
                    {"enrichment_state": st}, limit=100000
                )
                counts[st] = len(pts)
            except Exception:
                counts[st] = 0
        total = sum(counts.values())
        return {
            "counts": counts,
            "total": total,
            "pending_remaining": counts.get("pending", 0),
            "done": total - counts.get("pending", 0),
        }

    async def compact_enrichment(self) -> dict[str, Any]:
        """Prune portable-store records no longer referenced by any indexed chunk.

        Orphans (from changed/deleted code) are otherwise kept as a cross-version
        cache, so this is opt-in. ``live`` = every ``source_hash`` currently in
        the index.
        """
        store = self._get_enrichment_store()
        if store is None:
            return {"removed": 0, "kept": 0}
        points = await self.vector_db.scroll_by_payload({}, limit=1_000_000)
        live = {
            p["payload"].get("source_hash")
            for p in points
            if p.get("payload", {}).get("source_hash")
        }
        removed = await store.compact(live)
        return {"removed": removed, "kept": await store.count()}

    # ------------------------------------------------------------------
    # Heavy operations — lazily assembled from shared providers
    # ------------------------------------------------------------------

    def _require(self, name: str, value: Any) -> Any:
        if value is None:
            raise RuntimeError(
                f"This operation needs a {name}; the engine was built without one."
            )
        return value

    def _get_enrichment_store(self):
        """Per-project portable enrichment store, or ``None`` if the engine has
        no project root (search-only engines). Keyed to ``<root>/.semhood``."""
        if self._enrichment_store is None and self.root:
            from semhood.core.enrichment_store import EnrichmentStore
            from semhood.paths import enrichment_file_for

            self._enrichment_store = EnrichmentStore(enrichment_file_for(self.root))
        return self._enrichment_store

    def _get_indexer(self):
        if self._indexer is None:
            from semhood.core.indexer import Indexer

            self._indexer = Indexer(
                parser=self._require("parser", self._parser),
                embedder=self.embedder,
                vector_db=self.vector_db,
                structural=self._require("structural builder", self._structural),
                parsing_policy=self._require("parsing policy", self._parsing_policy),
                enrichment_version=self.config.enrichment.version,
                enrichment_store=self._get_enrichment_store(),
            )
        return self._indexer

    def _get_enricher(self):
        if self._enricher is None:
            from semhood.core.enricher import Enricher
            from semhood.core.policy import EnrichmentPolicy

            policy = EnrichmentPolicy(
                skip_patterns=self.config.enrichment.skip_patterns,
                skip_trivial_enabled=self.config.enrichment.skip_trivial.enabled,
                max_trivial_lines=self.config.enrichment.skip_trivial.max_lines,
                require_control_flow=self.config.enrichment.skip_trivial.require_control_flow,
            )
            self._enricher = Enricher(
                llm=self._require("llm", self._llm),
                embedder=self.embedder,
                vector_db=self.vector_db,
                policy=policy,
                version=self.config.enrichment.version,
                max_source_chars=self.config.enrichment.max_source_chars,
                concurrency=self.config.enrichment.llm_concurrency,
                max_attempts=self.config.enrichment.retry.max_attempts,
                backoff_seconds=self.config.enrichment.retry.backoff_seconds,
                jitter=self.config.enrichment.retry.jitter,
                store=self._get_enrichment_store(),
            )
        return self._enricher

    def _get_pipeline(self):
        if self._pipeline is None:
            from semhood.core.query_pipeline import QueryPipeline

            self._pipeline = QueryPipeline(
                vector_db=self.vector_db,
                embedder=self.embedder,
                llm=self._require("llm", self._llm),
                reranker=self._require("reranker", self._reranker),
                top_k=self.config.query.top_k,
                graph_expansion=self.config.query.graph_expansion,
                query_expansion=self.config.query.query_expansion,
            )
        return self._pipeline

    async def index(
        self, path: str, changed_only: bool = False, reset: bool = False
    ) -> dict[str, Any]:
        indexer = self._get_indexer()
        if changed_only:
            result = await indexer.index_changed(path)
        else:
            result = await indexer.index_all(path, reset=reset)
        return {
            "files_indexed": result.files_indexed,
            "chunks_indexed": result.chunks_indexed,
            "chunks_skipped": result.chunks_skipped,
            "stale_callers_cleaned": result.stale_callers_cleaned,
        }

    async def enrich(self, force: bool = False) -> dict[str, Any]:
        if not self.config.enrichment.enabled:
            raise RuntimeError("Enrichment is disabled in config")
        enricher = self._get_enricher()
        result = await enricher.run(force=force)
        return {
            "total": result.get("total", 0),
            "enriched": result.get("enriched", 0),
            "skipped_trivial": result.get("skipped_trivial", 0),
            "failed": result.get("failed", 0),
        }

    async def query(self, question: str) -> dict[str, Any]:
        pipeline = self._get_pipeline()
        result = await pipeline.query(question)
        return {"answer": result.answer, "sources": result.sources}
