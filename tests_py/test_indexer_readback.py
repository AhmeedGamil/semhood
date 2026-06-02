"""Indexer read-back — restore enrichment text from the portable store on
reindex (the path that makes an embedding-model change cost no LLM tokens)."""

from __future__ import annotations

import hashlib
from pathlib import Path

from semhood.core.enrichment_state import FRESH, PENDING
from semhood.core.enrichment_store import EnrichmentStore
from semhood.core.indexer import Indexer
from semhood.core.models import Chunk, ChunkType
from semhood.core.policy import ParsingPolicy
from semhood.core.structural import StructuralBuilder


class FakeEmbedder:
    dimension = 8

    async def embed_batch(self, texts: list[str]):
        return [[float(len(t) % 7)] * self.dimension for t in texts]


class FakeParser:
    def __init__(self, chunks: list[Chunk]):
        self._chunks = chunks

    def parse(self, file_path: str, base_path: str):
        return self._chunks

    def detect_language(self, path: str):
        return "python"


class CapturingVectorDB:
    """Empty collection that records what gets upserted."""

    def __init__(self):
        self.upserted: list[dict] = []

    async def ensure_collection(self, vector_dim: int, recreate: bool = False):
        pass

    async def get_points_by_file(self, file_path: str):
        return []

    async def get_points_by_name(self, name: str, class_name: str | None = None):
        return []

    async def get_vectors_by_id(self, point_id: str):
        return {}

    async def delete_by_ids(self, ids: list[str]):
        pass

    async def upsert(self, points: list[dict]):
        self.upserted.extend(points)


def _chunk(source: str) -> Chunk:
    return Chunk(
        name="foo", type=ChunkType.FUNCTION, file="mod.py", language="python",
        line_start=1, line_end=2, source=source,
    )


def _parsing_policy() -> ParsingPolicy:
    return ParsingPolicy(
        exclude_dirs=[], exclude_files=[], max_file_size_mb=1.0, max_chunks_per_file=500
    )


def _make_indexer(store, vdb, chunk) -> Indexer:
    return Indexer(
        parser=FakeParser([chunk]),
        embedder=FakeEmbedder(),
        vector_db=vdb,
        structural=StructuralBuilder(),
        parsing_policy=_parsing_policy(),
        enrichment_version=1,
        enrichment_store=store,
    )


async def test_readback_hit_restores_fresh_and_reembeds(tmp_path: Path):
    """Same content as a stored record → restore text, re-embed, mark fresh,
    no LLM (the indexer never calls one)."""
    src = "def foo():\n    return 1"
    (tmp_path / "mod.py").write_text(src, encoding="utf-8")
    source_hash = hashlib.sha256(src.encode("utf-8")).hexdigest()

    store = EnrichmentStore(tmp_path / ".semhood" / "enrichment.jsonl")
    await store.put(
        source_hash, logic_summary="does foo", developer_queries=["find foo"],
        enrichment_version=1,
    )

    vdb = CapturingVectorDB()
    await _make_indexer(store, vdb, _chunk(src)).index_files(
        [str(tmp_path / "mod.py")], str(tmp_path)
    )

    assert len(vdb.upserted) == 1
    pt = vdb.upserted[0]
    assert pt["payload"]["enrichment_state"] == FRESH
    assert pt["payload"]["logic_summary"] == "does foo"
    # Re-embedded from the restored text: both enrichment vectors present.
    assert "description" in pt["vectors"]
    assert "developer_queries" in pt["vectors"]


async def test_readback_miss_marks_pending(tmp_path: Path):
    """No stored record for this content → pending, no enrichment vectors."""
    src = "def bar():\n    return 2"
    (tmp_path / "mod.py").write_text(src, encoding="utf-8")

    store = EnrichmentStore(tmp_path / ".semhood" / "enrichment.jsonl")  # empty
    vdb = CapturingVectorDB()
    await _make_indexer(store, vdb, _chunk(src)).index_files(
        [str(tmp_path / "mod.py")], str(tmp_path)
    )

    pt = vdb.upserted[0]
    assert pt["payload"]["enrichment_state"] == PENDING
    assert "description" not in pt["vectors"]
    assert "developer_queries" not in pt["vectors"]


async def test_changed_content_never_gets_stale_enrichment(tmp_path: Path):
    """The store has enrichment for the OLD source; the file now has NEW source.
    Content-addressing must MISS → pending, never attach the stale summary."""
    old_src = "def foo():\n    return 1"
    new_src = "def foo():\n    return 999"
    (tmp_path / "mod.py").write_text(new_src, encoding="utf-8")
    old_hash = hashlib.sha256(old_src.encode("utf-8")).hexdigest()

    store = EnrichmentStore(tmp_path / ".semhood" / "enrichment.jsonl")
    await store.put(
        old_hash, logic_summary="OLD summary", developer_queries=["x"], enrichment_version=1
    )

    vdb = CapturingVectorDB()
    await _make_indexer(store, vdb, _chunk(new_src)).index_files(
        [str(tmp_path / "mod.py")], str(tmp_path)
    )

    pt = vdb.upserted[0]
    assert pt["payload"]["enrichment_state"] == PENDING
    assert not pt["payload"].get("logic_summary")


async def test_no_store_falls_back_to_pending(tmp_path: Path):
    """An engine without a project root has no store; indexing still works and
    leaves chunks pending (the pre-feature behavior)."""
    src = "def baz():\n    return 3"
    (tmp_path / "mod.py").write_text(src, encoding="utf-8")

    vdb = CapturingVectorDB()
    await _make_indexer(None, vdb, _chunk(src)).index_files(
        [str(tmp_path / "mod.py")], str(tmp_path)
    )

    pt = vdb.upserted[0]
    assert pt["payload"]["enrichment_state"] == PENDING
