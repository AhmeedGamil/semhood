"""Portable, content-addressed enrichment store.

Enrichment text (``logic_summary`` + ``developer_queries``) is the expensive,
model-agnostic product of Stage 2. The vector DB holds a copy *embedded* for the
current model, but those vectors die whenever the embedding model (and thus the
vector dimension) changes. This store keeps the **text** in a separate file,
keyed by ``source_hash`` (the chunk's content), so it survives model changes,
travels with the repo in git, and can be shared across a team.

Layout: ``<repo>/.semhood/enrichment.jsonl`` — one JSON record per line, sorted
by ``source_hash`` for stable, line-based git diffs. The in-memory map is the
source of truth; the file is its durable mirror.

Concurrency: a single ``asyncio.Lock`` serializes every writer (the ``semhood
enrich`` worker's concurrent gather, the MCP ``save_enrichment`` calls, and any
subagent fan-out). This is sufficient because the daemon is single-process /
single-event-loop. Writes are atomic (write a temp file, then ``os.replace``),
so a crash mid-write never corrupts the file. Durability is per-write: the
record is on disk before ``put`` returns, because the MCP path has no
"end of run" at which a buffered flush could happen.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# Fields persisted per chunk. Kept deliberately small: text + provenance, never
# secrets, never the embedding vectors (those are model-specific and live in the
# vector DB).
_RECORD_FIELDS = ("source_hash", "logic_summary", "developer_queries", "enrichment_version")


class EnrichmentStore:
    """One per project. ``source_hash`` -> enrichment record."""

    def __init__(self, path: str | Path):
        self._path = Path(path)
        self._map: dict[str, dict] = {}
        self._lock = asyncio.Lock()
        self._loaded = False

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    async def _ensure_loaded(self) -> None:
        """Read the JSONL file into ``_map`` once. Tolerates a missing file and
        skips any corrupt/partial lines rather than failing the whole load."""
        if self._loaded:
            return
        if self._path.exists():
            try:
                text = self._path.read_text(encoding="utf-8")
            except OSError as e:
                logger.warning("Could not read enrichment store %s: %s", self._path, e)
                text = ""
            for lineno, line in enumerate(text.splitlines(), 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    sh = rec.get("source_hash")
                    if sh:
                        self._map[sh] = rec
                except json.JSONDecodeError:
                    logger.warning(
                        "Skipping malformed line %d in %s", lineno, self._path
                    )
        self._loaded = True

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    async def get(self, source_hash: str) -> dict | None:
        """Return the record for ``source_hash``, or ``None`` if absent."""
        await self._ensure_loaded()
        return self._map.get(source_hash)

    async def count(self) -> int:
        await self._ensure_loaded()
        return len(self._map)

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    async def put(
        self,
        source_hash: str,
        *,
        logic_summary: str,
        developer_queries: list[str],
        enrichment_version: int,
    ) -> None:
        """Upsert one chunk's enrichment text and flush atomically.

        Keyed by ``source_hash``, so re-enriching the same content replaces the
        record instead of duplicating it. Serialized by the lock; safe under the
        worker's concurrent gather and concurrent MCP/subagent calls.
        """
        if not source_hash:
            return
        record = {
            "source_hash": source_hash,
            "logic_summary": logic_summary,
            "developer_queries": list(developer_queries or []),
            "enrichment_version": enrichment_version,
        }
        async with self._lock:
            await self._ensure_loaded()
            self._map[source_hash] = record
            await self._flush()

    async def compact(self, live_hashes: set[str]) -> int:
        """Drop records whose ``source_hash`` is not in ``live_hashes``.

        Orphaned records (from code that changed or was deleted) are otherwise
        kept as a cross-version cache — useful on branch switch / revert — so
        this is opt-in, never automatic. Returns the number of records removed.
        """
        async with self._lock:
            await self._ensure_loaded()
            stale = [h for h in self._map if h not in live_hashes]
            for h in stale:
                del self._map[h]
            if stale:
                await self._flush()
            return len(stale)

    # ------------------------------------------------------------------
    # Atomic flush
    # ------------------------------------------------------------------

    async def _flush(self) -> None:
        """Write the whole map to disk atomically. Caller must hold the lock.

        Records are sorted by ``source_hash`` so the file is deterministic and
        diffs cleanly in git. Write to a temp file in the same directory, then
        ``os.replace`` — atomic on both Windows and POSIX. The ``.semhood``
        directory (and its ``.gitignore``) is scaffolded here, lazily, so it
        only appears once there is enrichment to persist.
        """
        from semhood.paths import scaffold_dotdir

        scaffold_dotdir(self._path.parent)
        lines = [
            json.dumps(
                {k: self._map[h].get(k) for k in _RECORD_FIELDS},
                separators=(",", ":"),
                ensure_ascii=False,
            )
            for h in sorted(self._map)
        ]
        body = ("\n".join(lines) + "\n") if lines else ""
        tmp = self._path.with_name(self._path.name + ".tmp")
        tmp.write_text(body, encoding="utf-8")
        os.replace(tmp, self._path)
