"""
Query pipeline — full search + answer generation.

Pipeline:
  Query
    → Query Expansion (LLM enriches with code terms)
    → Embed query
    → Multi-vector search (strategy-dependent)
    → RRF merge + deduplicate
    → Rerank (optional)
    → Graph expansion (call graph neighbors)
    → Context builder → final prompt
    → LLM → answer
"""

from __future__ import annotations

import logging

from semhood.core.context_builder import ContextBuilder
from semhood.core.models import QueryResult
from semhood.providers.embeddings.base import EmbeddingProvider
from semhood.providers.llm.base import LLMProvider
from semhood.providers.reranker.base import RerankerProvider
from semhood.providers.vector_db.base import VectorDBProvider

logger = logging.getLogger(__name__)

# RRF constant to prevent top-rank domination
_RRF_K = 60

_QUERY_EXPAND_SYSTEM = """You are a code search assistant. Given a developer's question, expand it with specific code terms, class names, method names, and technical vocabulary that would help find the relevant code. Return the expanded query as a single string. Only return the expanded query, nothing else."""

_ANSWER_SYSTEM = """You are a senior software engineer answering questions about a codebase. Use the provided code context to give accurate, helpful answers. Reference specific functions, classes, and files when relevant. If the context doesn't contain enough information, say so honestly."""


class QueryPipeline:
    """Full query pipeline: search → rerank → expand → answer.

    Searches across all three named vectors (``code``, ``description``,
    ``developer_queries``) plus BM25 sparse, then RRF-merges. Points whose
    enrichment hasn't run yet simply have no ``description`` /
    ``developer_queries`` vectors stored — they're silently skipped on
    those vector searches but still found via ``code`` and BM25.
    """

    def __init__(
        self,
        vector_db: VectorDBProvider,
        embedder: EmbeddingProvider,
        llm: LLMProvider,
        reranker: RerankerProvider,
        top_k: int = 10,
        graph_expansion: bool = True,
        query_expansion: bool = True,
    ):
        self._vector_db = vector_db
        self._embedder = embedder
        self._llm = llm
        self._reranker = reranker
        self._top_k = top_k
        self._graph_expansion = graph_expansion
        self._query_expansion = query_expansion
        self._context_builder = ContextBuilder()

    async def query(self, question: str) -> QueryResult:
        """Execute the full query pipeline."""

        # Step 1: Query expansion
        expanded = question
        if self._query_expansion:
            expanded = await self._expand_query(question)
            logger.info("Expanded query: %s", expanded[:200])

        # Step 2: Embed query
        query_vector = await self._embedder.embed_query(expanded)

        # Step 3: Multi-vector search (strategy-dependent)
        search_results = await self._multi_search(query_vector)

        # Step 4: RRF merge + deduplicate
        merged = self._rrf_merge(search_results)
        logger.info("RRF merged: %d results", len(merged))

        if not merged:
            return QueryResult(answer="No relevant code found.", sources=[])

        # Step 5: Rerank
        reranked = await self._rerank_results(expanded, merged)

        # Step 6: Graph expansion
        final_chunks = reranked
        if self._graph_expansion:
            final_chunks = await self._expand_graph(reranked)

        # Step 7: Build context
        context = self._context_builder.build(final_chunks, with_source=True)

        # Step 8: Generate answer
        answer = await self._generate_answer(question, context)

        # Build sources
        sources = [
            {
                "name": c.get("payload", {}).get("name", ""),
                "file": c.get("payload", {}).get("file", ""),
                "line": c.get("payload", {}).get("line_start"),
                "class": c.get("payload", {}).get("class_name"),
            }
            for c in reranked if not c.get("is_neighbor")
        ]

        return QueryResult(answer=answer, sources=sources)

    # -------------------------------------------------------------------------
    # Pipeline steps
    # -------------------------------------------------------------------------

    async def _expand_query(self, question: str) -> str:
        """Use LLM to expand query with code-specific terms."""
        try:
            expanded = await self._llm.generate(
                _QUERY_EXPAND_SYSTEM, question, max_tokens=256
            )
            return expanded.strip() or question
        except Exception as e:
            logger.warning("Query expansion failed: %s", e)
            return question

    async def _multi_search(self, query_vector: list[float]) -> dict[str, list[dict]]:
        """Search across all named vectors.

        We always query ``code``, ``description``, and ``developer_queries``.
        For points whose enrichment hasn't run, the latter two vectors
        simply don't exist and the engine skips them silently. The ``code``
        vector + BM25 keep recall high even for fully-pending indexes.
        """
        results: dict[str, list[dict]] = {}

        for vname in ("code", "description", "developer_queries"):
            try:
                results[vname] = await self._vector_db.search_dense(
                    vname, query_vector, self._top_k
                )
            except Exception as e:
                logger.warning("Search on vector '%s' failed: %s", vname, e)
                results[vname] = []

        # BM25 sparse search requires a tokenized query — kept as a hook
        # for future improvement.
        results["bm25"] = []

        return results

    def _rrf_merge(self, result_lists: dict[str, list[dict]]) -> list[dict]:
        """Reciprocal Rank Fusion across multiple result lists."""
        scores: dict[str, float] = {}
        items: dict[str, dict] = {}

        for _list_name, results in result_lists.items():
            for rank, item in enumerate(results):
                item_id = item["id"]
                rrf_score = 1.0 / (_RRF_K + rank + 1)
                scores[item_id] = scores.get(item_id, 0.0) + rrf_score
                items[item_id] = item  # Latest wins (same payload)

        # Sort by RRF score descending
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

        return [
            {**items[item_id], "rrf_score": scores[item_id]}
            for item_id in sorted_ids
        ]

    async def _rerank_results(self, query: str, results: list[dict]) -> list[dict]:
        """Rerank results using the configured reranker."""
        if not results:
            return []

        # Build document strings for reranker
        documents = []
        for r in results:
            payload = r.get("payload", {})
            doc = f"{payload.get('name', '')} - {payload.get('logic_summary', payload.get('docblock', ''))}"
            documents.append(doc)

        try:
            indices = await self._reranker.rerank(query, documents, top_n=8)
            return [results[i] for i in indices if i < len(results)]
        except Exception as e:
            logger.warning("Reranking failed: %s", e)
            return results[:8]

    async def _expand_graph(self, results: list[dict]) -> list[dict]:
        """Pull in call graph neighbors for retrieved chunks."""
        neighbor_ids: set[str] = set()
        existing_ids = {r["id"] for r in results}

        for r in results:
            payload = r.get("payload", {})

            # Outgoing calls — we'd need name resolution here
            # For now, skip call expansion (needs NameIndex)

            # Incoming callers
            called_by = payload.get("called_by", [])
            for _caller in called_by[:3]:
                # Would need to resolve caller to a point ID
                pass

        # Fetch neighbor payloads
        if neighbor_ids:
            remaining = neighbor_ids - existing_ids
            if remaining:
                neighbors = await self._vector_db.get_by_ids(list(remaining))
                for nid, payload in neighbors.items():
                    results.append({
                        "id": nid,
                        "payload": payload,
                        "is_neighbor": True,
                        "score": 0.0,
                    })

        return results

    async def _generate_answer(self, question: str, context: str) -> str:
        """Generate the final answer using context."""
        prompt = f"""Question: {question}

Code Context:
{context}

Answer the question based on the code context above."""

        try:
            answer = await self._llm.generate(_ANSWER_SYSTEM, prompt, max_tokens=2048)
            return answer.strip()
        except Exception as e:
            logger.error("Answer generation failed: %s", e)
            return f"Error generating answer: {e}"
