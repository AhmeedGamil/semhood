"""
Indexer — Stage 1 of the semhood pipeline.

Pipeline (fast, free, deterministic):
  Source files (any language)
    → TreeSitterParser      → Chunk objects per file
    → CallGraphBuilder      → attaches called_by to every chunk
    → StructuralBuilder     → source_hash + code vector + BM25 sparse
    → VectorDBProvider      → upsert (state = pending unless source unchanged)

The indexer never calls an LLM. Enrichment is Stage 2 (``semhood enrich``).

Incremental re-indexing:
  - For each chunk, compute ``source_hash``.
  - If a point exists with the same id, same hash, and the same
    ``enrichment_version`` and a terminal state (fresh / skipped_trivial),
    we preserve its existing description / developer_queries vectors and
    enrichment payload.
  - Otherwise we reset the chunk's state to ``pending`` and let the
    enricher pick it up.
"""

from __future__ import annotations

import logging
import subprocess
import uuid
from pathlib import Path

from semhood.core.call_graph import CallGraphBuilder
from semhood.core.enrichment_state import FRESH, PENDING, SKIPPED_TRIVIAL
from semhood.core.enrichment_store import EnrichmentStore
from semhood.core.enrichment_writer import now_iso
from semhood.core.models import Chunk, IndexResult
from semhood.core.policy import ParsingPolicy
from semhood.core.structural import StructuralBuilder, build_description
from semhood.parsing.parser import TreeSitterParser
from semhood.providers.embeddings.base import EmbeddingProvider
from semhood.providers.vector_db.base import VectorDBProvider

logger = logging.getLogger(__name__)


class Indexer:
    """Orchestrates the structural indexing pipeline."""

    def __init__(
        self,
        parser: TreeSitterParser,
        embedder: EmbeddingProvider,
        vector_db: VectorDBProvider,
        structural: StructuralBuilder,
        parsing_policy: ParsingPolicy,
        enrichment_version: int = 1,
        call_graph: CallGraphBuilder | None = None,
        enrichment_store: EnrichmentStore | None = None,
    ):
        self._parser = parser
        self._embedder = embedder
        self._vector_db = vector_db
        self._structural = structural
        self._parsing_policy = parsing_policy
        self._enrichment_version = enrichment_version
        self._call_graph = call_graph or CallGraphBuilder()
        self._enrichment_store = enrichment_store

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    async def index_all(self, base_path: str, reset: bool = False) -> IndexResult:
        """Index all parseable files under ``base_path``."""
        files = self._discover_files(base_path)
        logger.info("Discovered %d files to index", len(files))
        return await self.index_files(files, base_path, reset=reset)

    async def index_changed(self, base_path: str) -> IndexResult:
        """Index only files changed since the last commit (git diff HEAD~1)."""
        files = self._get_changed_files(base_path)
        if not files:
            logger.info("No changed files to index")
            return IndexResult()
        logger.info("Indexing %d changed files", len(files))
        return await self.index_files(files, base_path)

    async def index_files(
        self, file_paths: list[str], base_path: str, reset: bool = False
    ) -> IndexResult:
        """Index a specific list of file paths.

        ``reset=True`` drops and rebuilds the collection — used after an
        intentional embedding-model change. Enrichment text is restored from the
        portable store during the per-chunk pass, so no LLM tokens are spent.
        """
        await self._vector_db.ensure_collection(
            vector_dim=self._embedder.dimension, recreate=reset
        )

        # Phase 0: snapshot old call data (for stale called_by cleanup).
        rel_file_paths = [
            str(Path(fp).relative_to(base_path)) if base_path else fp
            for fp in file_paths
        ]
        old_calls_by_file = await self._snapshot_old_calls(rel_file_paths)

        # Phase 1: parse files into chunks
        all_chunks: list[Chunk] = []
        file_chunks: dict[str, list[int]] = {}

        for file_path in file_paths:
            if not Path(file_path).exists():
                continue

            chunks = self._parser.parse(file_path, base_path)
            if not chunks:
                continue

            start = len(all_chunks)
            all_chunks.extend(chunks)
            file_chunks[file_path] = list(range(start, start + len(chunks)))

        if not all_chunks:
            return IndexResult()

        # Phase 2: cross-file call graph
        self._call_graph.attach_called_by(all_chunks)

        # Phase 2.5: clean up called_by entries pointing at removed call sites
        stale_cleaned = await self._cleanup_stale_called_by(
            old_calls_by_file, all_chunks, file_chunks
        )

        # Phase 3: per-file structural upsert with selective hash diffing
        total_chunks = 0
        skipped = 0

        for file_path, indices in file_chunks.items():
            rel = (
                str(Path(file_path).relative_to(base_path))
                if base_path else file_path
            )

            # Cap chunks per file BEFORE we read existing points so that
            # truncated chunks don't poison the diff.
            batch_chunks = [all_chunks[i] for i in indices]
            cap = self._parsing_policy.max_chunks_per_file
            if cap and len(batch_chunks) > cap:
                logger.warning(
                    "File %s produced %d chunks (cap %d), truncating",
                    rel, len(batch_chunks), cap,
                )
                batch_chunks = batch_chunks[:cap]

            # Existing points for this file, keyed by id
            existing = await self._vector_db.get_points_by_file(rel)
            existing_by_id = {p["id"]: p["payload"] for p in existing}

            points: list[dict] = []
            for chunk in batch_chunks:
                new_hash = self._structural.source_hash(chunk)
                chunk.source_hash = new_hash

                pid = self._point_id(chunk)
                old = existing_by_id.get(pid)

                preserve_enrichment = bool(
                    old
                    and old.get("source_hash") == new_hash
                    and old.get("enrichment_version") == self._enrichment_version
                    and old.get("enrichment_state") in (FRESH, SKIPPED_TRIVIAL)
                )

                restored_from_store = False
                if preserve_enrichment:
                    chunk.logic_summary = old.get("logic_summary")
                    chunk.developer_queries = old.get("developer_queries", []) or []
                    chunk.description = old.get("description")
                    chunk.enrichment_state = old.get("enrichment_state", PENDING)
                    chunk.enrichment_version = old.get("enrichment_version", 0)
                    chunk.enriched_at = old.get("enriched_at")
                    chunk.enrichment_attempts = old.get("enrichment_attempts", 0) or 0
                    chunk.enrichment_error = old.get("enrichment_error")
                else:
                    # Vector-DB preserve missed (new/changed chunk, or the
                    # collection was rebuilt for a new embedding model). Fall
                    # back to the portable store: if enrichment text exists for
                    # this exact content, restore it and re-embed below — no LLM.
                    rec = (
                        await self._enrichment_store.get(new_hash)
                        if self._enrichment_store is not None
                        else None
                    )
                    if rec:
                        chunk.logic_summary = rec.get("logic_summary")
                        chunk.developer_queries = rec.get("developer_queries", []) or []
                        chunk.description = build_description(chunk)
                        chunk.enrichment_state = FRESH
                        chunk.enrichment_version = rec.get(
                            "enrichment_version", self._enrichment_version
                        )
                        chunk.enriched_at = now_iso()
                        chunk.enrichment_attempts = 0
                        chunk.enrichment_error = None
                        restored_from_store = True
                    else:
                        chunk.enrichment_state = PENDING
                        chunk.enrichment_version = 0
                        chunk.enriched_at = None
                        chunk.enrichment_attempts = 0
                        chunk.enrichment_error = None

                try:
                    code_vec = await self._structural.build_code_vector(
                        chunk, self._embedder
                    )
                    vectors: dict[str, list[float]] = {"code": code_vec}

                    # Carry forward enrichment vectors for unchanged chunks.
                    if preserve_enrichment:
                        existing_vecs = await self._vector_db.get_vectors_by_id(pid)
                        for vname in ("description", "developer_queries"):
                            if existing_vecs.get(vname):
                                vectors[vname] = existing_vecs[vname]
                    elif restored_from_store:
                        # Re-embed the restored text with the CURRENT model. This
                        # is what makes an embedding-model change cost no tokens.
                        queries_text = (
                            "\n".join(chunk.developer_queries)
                            if chunk.developer_queries
                            else chunk.name
                        )
                        emb = await self._embedder.embed_batch(
                            [chunk.description or "", queries_text]
                        )
                        vectors["description"] = emb[0]
                        vectors["developer_queries"] = emb[1]

                    sparse = self._structural.build_sparse(chunk)
                    points.append({
                        "id": pid,
                        "vectors": vectors,
                        "sparse": sparse,
                        "payload": chunk.to_payload(),
                    })
                except Exception as e:
                    logger.warning(
                        "Failed to embed chunk %s: %s",
                        chunk.qualified_name(), e,
                    )
                    skipped += 1

            # Targeted delete: any old point with no corresponding new chunk
            # is stale (the chunk disappeared from the file).
            new_ids = {p["id"] for p in points}
            stale_ids = [pid for pid in existing_by_id if pid not in new_ids]
            if stale_ids:
                await self._vector_db.delete_by_ids(stale_ids)

            if points:
                await self._vector_db.upsert(points)
                total_chunks += len(points)

            logger.info("Indexed %s: %d chunks", rel, len(points))

        return IndexResult(
            files_indexed=len(file_chunks),
            chunks_indexed=total_chunks,
            chunks_skipped=skipped,
            stale_callers_cleaned=stale_cleaned,
        )

    # -------------------------------------------------------------------------
    # Stale called_by cleanup
    # -------------------------------------------------------------------------

    async def _snapshot_old_calls(self, file_paths: list[str]) -> dict[str, dict]:
        """Snapshot old call data from vector DB before re-indexing."""
        snapshot: dict[str, dict] = {}

        for file_path in file_paths:
            old_points = await self._vector_db.get_points_by_file(file_path)
            for point in old_points:
                payload = point.get("payload", {})
                key = f"{payload.get('class_name', '')}::{payload.get('name', '')}"
                if file_path not in snapshot:
                    snapshot[file_path] = {}
                snapshot[file_path][key] = {
                    "calls": payload.get("calls", []),
                    "file": payload.get("file", file_path),
                }

        return snapshot

    async def _cleanup_stale_called_by(
        self,
        old_calls_by_file: dict[str, dict],
        all_chunks: list[Chunk],
        file_chunks: dict[str, list[int]],
    ) -> int:
        """Find removed call targets and clean their called_by in vector DB."""
        if not old_calls_by_file:
            return 0

        new_calls_by_file: dict[str, dict[str, list[str]]] = {}
        for _, indices in file_chunks.items():
            for i in indices:
                chunk = all_chunks[i]
                rel = chunk.file
                if rel not in new_calls_by_file:
                    new_calls_by_file[rel] = {}
                key = f"{chunk.class_name or ''}::{chunk.name}"
                new_calls_by_file[rel][key] = chunk.calls

        removals: list[tuple[str, str, str]] = []

        for rel_file, old_chunks in old_calls_by_file.items():
            new_chunks = new_calls_by_file.get(rel_file, {})

            for chunk_key, old_data in old_chunks.items():
                old_calls = set(old_data.get("calls", []) or [])
                if chunk_key in new_chunks:
                    new_calls = set(new_chunks[chunk_key])
                else:
                    new_calls = set()

                for call_target in (old_calls - new_calls):
                    removals.append((rel_file, chunk_key, call_target))

        if not removals:
            return 0

        callee_cache: dict[str, list[dict]] = {}
        patched_state: dict[str, list[dict]] = {}
        cleaned = 0

        for caller_file, caller_key, call_target in removals:
            normalized = self._normalize_call(call_target)

            if "::" in normalized:
                callee_class, callee_name = normalized.split("::", 1)
            else:
                callee_class, callee_name = None, normalized

            if not callee_name:
                continue

            cache_key = f"{callee_class or ''}::{callee_name}"
            if cache_key not in callee_cache:
                points = await self._vector_db.get_points_by_name(
                    callee_name, callee_class
                )
                if not points and callee_class:
                    points = await self._vector_db.get_points_by_name(callee_name)
                callee_cache[cache_key] = points

            points = callee_cache[cache_key]
            if not points:
                continue

            caller_class, _, caller_name = caller_key.partition("::")
            caller_class = caller_class or None

            for point in points:
                pid = point["id"]
                current = patched_state.get(
                    pid, list(point["payload"].get("called_by") or [])
                )

                filtered = [
                    cb for cb in current
                    if not (
                        cb.get("file") == caller_file
                        and cb.get("function") == caller_name
                        and (cb.get("class_name") or None) == caller_class
                    )
                ]

                if len(filtered) != len(current):
                    patched_state[pid] = filtered
                    cleaned += 1

        for pid, called_by in patched_state.items():
            await self._vector_db.set_payload_by_id(pid, {"called_by": called_by})

        return cleaned

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _point_id(self, chunk: Chunk) -> str:
        """Deterministic UUID v5 for a chunk."""
        seed = f"{chunk.file}::{chunk.name}::{chunk.line_start}"
        return str(uuid.uuid5(uuid.NAMESPACE_URL, seed))

    def _normalize_call(self, call: str) -> str:
        """Normalize a call target for comparison."""
        for prefix in ("self.", "cls.", "this.", "this->", "super.", "$this->"):
            if call.startswith(prefix):
                return call[len(prefix):]
        if "::" in call:
            return call
        if "." in call:
            parts = call.split(".", 1)
            if parts[0] and parts[0][0].isupper():
                return f"{parts[0]}::{parts[1]}"
            return parts[1]
        return call

    def _discover_files(self, base_path: str) -> list[str]:
        """Discover all parseable source files under base_path."""
        files: list[str] = []
        base = Path(base_path)

        for path in base.rglob("*"):
            if not path.is_file():
                continue

            if self._parsing_policy.should_skip_path(path):
                continue

            if self._parser.detect_language(str(path)):
                files.append(str(path))

        return sorted(files)

    def _get_changed_files(self, base_path: str) -> list[str]:
        """Get files changed since last commit via git diff."""
        try:
            result = subprocess.run(
                ["git", "-C", base_path, "diff", "HEAD~1", "--name-only"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode != 0:
                return []

            files: list[str] = []
            for rel in result.stdout.strip().split("\n"):
                if not rel:
                    continue
                abs_path = str(Path(base_path) / rel)
                if Path(abs_path).exists() and self._parser.detect_language(abs_path):
                    files.append(abs_path)
            return files

        except Exception as e:
            logger.warning("git diff failed: %s", e)
            return []
