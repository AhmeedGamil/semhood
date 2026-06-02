"""
Enricher — Stage 2 of the semhood pipeline.

Drains the ``enrichment_state == pending`` queue. For each chunk:

  1. ``policy.should_skip_by_pattern`` → mark ``skipped_trivial`` and move on.
  2. ``policy.is_trivial``             → write a deterministic template
     summary + queries into the description / developer_queries vectors and
     mark ``skipped_trivial``.
  3. Otherwise → call the LLM (with concurrency + retry/backoff/jitter),
     embed the result, patch the point's vectors + payload, and mark
     ``fresh``. After ``max_attempts`` failures we mark ``failed``.

The Enricher reads only the vector DB and the LLM. It never touches the
filesystem or the parser.
"""

from __future__ import annotations

import asyncio
import logging
import random

from semhood.core.enrichment_state import (
    FAILED,
    FRESH,
    PENDING,
    SKIPPED_TRIVIAL,
)
from semhood.core.enrichment_store import EnrichmentStore
from semhood.core.enrichment_writer import (
    now_iso as _now,
)
from semhood.core.enrichment_writer import (
    payload_to_chunk as _payload_to_chunk,
)
from semhood.core.enrichment_writer import (
    write_enrichment,
)
from semhood.core.models import Chunk
from semhood.core.policy import EnrichmentPolicy
from semhood.providers.embeddings.base import EmbeddingProvider
from semhood.providers.llm.base import LLMProvider
from semhood.providers.vector_db.base import VectorDBProvider

logger = logging.getLogger(__name__)


ENRICH_SYSTEM_PROMPT = """You are a senior code-reading assistant. Given a code chunk, produce:
1. A `logic_summary` — 1-3 sentences describing what this code does and why it exists.
2. A list of `developer_queries` — 5-10 short natural-language queries a developer
   might type when looking for this code.

Respond ONLY in JSON: {"logic_summary": "...", "developer_queries": ["...", "..."]}"""


def _build_user_prompt(chunk: Chunk, max_chars: int) -> str:
    src = chunk.source or ""
    truncated = ""
    if len(src) > max_chars:
        truncated = f"\n... [source truncated, {len(src) - max_chars} chars omitted]"
        src = src[:max_chars]
    qn = chunk.qualified_name()
    return (
        f"File: {chunk.file}\n"
        f"Chunk: {qn}\n"
        f"Language: {chunk.language}\n"
        f"\n```{chunk.language}\n{src}{truncated}\n```"
    )


class Enricher:
    """Stage 2 worker — drains the pending queue."""

    def __init__(
        self,
        llm: LLMProvider,
        embedder: EmbeddingProvider,
        vector_db: VectorDBProvider,
        policy: EnrichmentPolicy,
        version: int,
        max_source_chars: int,
        concurrency: int,
        max_attempts: int,
        backoff_seconds: float,
        jitter: bool,
        store: EnrichmentStore | None = None,
    ):
        self._llm = llm
        self._embedder = embedder
        self._vdb = vector_db
        self._policy = policy
        self._version = version
        self._max_chars = max_source_chars
        self._sem = asyncio.Semaphore(max(1, concurrency))
        self._max_attempts = max(1, max_attempts)
        self._backoff = max(0.0, float(backoff_seconds))
        self._jitter = bool(jitter)
        self._store = store

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    async def run(self, force: bool = False) -> dict:
        """Drain the pending queue.

        If ``force`` is set, every chunk (regardless of current state) is
        first reset to ``pending`` and then enriched.
        """
        if force:
            await self._mark_all_pending()

        points = await self._vdb.scroll_by_payload(
            filter={"enrichment_state": PENDING},
            limit=100000,
        )

        if not points:
            return {"enriched": 0, "skipped_trivial": 0, "failed": 0, "total": 0}

        results = await asyncio.gather(
            *(self._enrich_one(p) for p in points),
            return_exceptions=False,
        )

        ok = sum(1 for r in results if r == "ok")
        skipped = sum(1 for r in results if r == "skipped_trivial")
        failed = sum(1 for r in results if r == "failed")
        return {
            "enriched": ok,
            "skipped_trivial": skipped,
            "failed": failed,
            "total": len(results),
        }

    # ------------------------------------------------------------------
    # Per-point work
    # ------------------------------------------------------------------

    async def _enrich_one(self, point: dict) -> str:
        async with self._sem:
            payload = point.get("payload") or {}
            point_id = point["id"]

            try:
                chunk = _payload_to_chunk(payload)
            except Exception as e:
                logger.warning("Skipping malformed point %s: %s", point_id, e)
                return "failed"

            # 1. Pattern-based skip.
            if self._policy.should_skip_by_pattern(chunk):
                await self._vdb.set_payload_by_id(point_id, {
                    "enrichment_state": SKIPPED_TRIVIAL,
                    "enrichment_version": self._version,
                    "enriched_at": _now(),
                    "enrichment_error": None,
                })
                return "skipped_trivial"

            # 2. Trivial chunk → deterministic template.
            if self._policy.is_trivial(chunk):
                summary, queries = self._policy.deterministic_summary(chunk)
                await self._patch_enrichment(
                    point_id, chunk, summary, queries, SKIPPED_TRIVIAL
                )
                return "skipped_trivial"

            # 3. LLM call with retries.
            for attempt in range(1, self._max_attempts + 1):
                try:
                    user = _build_user_prompt(chunk, self._max_chars)
                    raw = await self._llm.generate_json(
                        ENRICH_SYSTEM_PROMPT, user, max_tokens=1024
                    )
                    summary = (raw or {}).get("logic_summary") or ""
                    queries = (raw or {}).get("developer_queries") or []
                    if not summary:
                        raise ValueError("empty logic_summary")
                    await self._patch_enrichment(
                        point_id, chunk, summary, queries, FRESH
                    )
                    return "ok"
                except Exception as e:
                    logger.warning(
                        "Enrichment attempt %d/%d failed for %s: %s",
                        attempt, self._max_attempts, chunk.qualified_name(), e,
                    )
                    if attempt == self._max_attempts:
                        await self._vdb.set_payload_by_id(point_id, {
                            "enrichment_state": FAILED,
                            "enrichment_attempts": attempt,
                            "enrichment_error": str(e)[:500],
                            "enrichment_version": self._version,
                            "enriched_at": _now(),
                        })
                        return "failed"
                    delay = self._backoff * (2 ** (attempt - 1))
                    if self._jitter and delay > 0:
                        delay += random.uniform(0, delay * 0.25)
                    if delay > 0:
                        await asyncio.sleep(delay)
            return "failed"

    async def _patch_enrichment(
        self,
        point_id: str,
        chunk: Chunk,
        summary: str,
        queries: list[str],
        final_state: str,
    ) -> None:
        await write_enrichment(
            vector_db=self._vdb,
            embedder=self._embedder,
            point_id=point_id,
            chunk=chunk,
            summary=summary,
            queries=queries,
            version=self._version,
            final_state=final_state,
            store=self._store,
        )

    async def _mark_all_pending(self) -> None:
        """Reset every non-pending point back to pending (for ``--force``)."""
        for state in (FRESH, FAILED, SKIPPED_TRIVIAL):
            points = await self._vdb.scroll_by_payload(
                filter={"enrichment_state": state}, limit=100000
            )
            for p in points:
                await self._vdb.set_payload_by_id(p["id"], {
                    "enrichment_state": PENDING,
                })
