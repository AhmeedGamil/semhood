"""Engine + MCP-manifest tests.

The per-project logic now lives in :class:`semhood.engine.ProjectEngine` (the
daemon and the MCP proxy both call it). These tests exercise that logic directly
with in-memory fakes — no transport, no model, no daemon — plus a sanity check
that the MCP server still advertises the expected tool manifest.
"""

from __future__ import annotations

import pytest

from semhood import mcp_server
from semhood.config import SemhoodConfig
from semhood.core.enrichment_state import FRESH, PENDING
from semhood.engine import ProjectEngine


class FakeEmbedder:
    dimension = 8

    async def embed_query(self, text: str):
        # Deterministic, length-only embedding for tests.
        return [float(len(text) % 10)] * self.dimension

    async def embed_batch(self, texts: list[str]):
        return [await self.embed_query(t) for t in texts]


class FakeVectorDB:
    """In-memory stand-in. Stores per-vector hits returned from search_dense."""

    def __init__(self):
        self.points_by_name: dict[tuple[str, str | None], list[dict]] = {}
        self.scroll_results: dict[str, list[dict]] = {}
        self.dense_hits: dict[str, list[dict]] = {}
        self.points_by_id: dict[str, dict] = {}
        self.updated_payloads: dict[str, dict] = {}
        self.updated_vectors: dict[str, dict] = {}

    async def search_dense(self, vector_name: str, vector, top_k: int = 8):
        return list(self.dense_hits.get(vector_name, []))[:top_k]

    async def get_points_by_name(self, name: str, class_name: str | None = None):
        return list(self.points_by_name.get((name, class_name), []))

    async def scroll_by_payload(self, filter: dict, limit: int = 100000):
        state = filter.get("enrichment_state")
        return list(self.scroll_results.get(state, []))

    async def get_by_ids(self, ids: list[str]):
        return {pid: self.points_by_id[pid] for pid in ids if pid in self.points_by_id}

    async def update_vectors_and_payload(self, point_id: str, vectors: dict, payload: dict) -> None:
        self.updated_vectors[point_id] = vectors
        self.updated_payloads[point_id] = payload


@pytest.fixture
def engine(tmp_path) -> ProjectEngine:
    """A ProjectEngine wired to in-memory fakes, rooted at a real temp dir so
    enrichment writes (the portable store) land somewhere disposable."""
    return ProjectEngine(
        config=SemhoodConfig(),
        embedder=FakeEmbedder(),
        vector_db=FakeVectorDB(),
        root=str(tmp_path),
    )


# ------------------------------------------------------------------------
# Search & lookup
# ------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_returns_merged_results(engine):
    engine.vector_db.dense_hits["code"] = [
        {"id": "a", "score": 0.9, "payload": {"name": "charge", "class_name": "PP",
                                              "file": "p.py", "line_start": 10,
                                              "enrichment_state": PENDING}},
        {"id": "b", "score": 0.7, "payload": {"name": "refund", "class_name": "PP",
                                              "file": "p.py", "line_start": 90,
                                              "enrichment_state": PENDING}},
    ]
    out = await engine.search("how to charge", top_k=5)
    assert out["query"] == "how to charge"
    assert out["embedding_dim"] == FakeEmbedder.dimension
    assert len(out["results"]) == 2
    assert out["results"][0]["qualified_name"] == "PP::charge"


@pytest.mark.asyncio
async def test_search_rejects_empty_query(engine):
    with pytest.raises(ValueError):
        await engine.search("")


@pytest.mark.asyncio
async def test_search_many_returns_results_per_query(engine):
    engine.vector_db.dense_hits["code"] = [
        {"id": "a", "score": 0.9, "payload": {"name": "charge", "class_name": "PP",
                                              "file": "p.py", "line_start": 10,
                                              "enrichment_state": PENDING}},
    ]
    out = await engine.search_many(["how to charge", "how to refund"], top_k=5)
    # One result set per query, in order.
    assert [r["query"] for r in out["results"]] == ["how to charge", "how to refund"]
    assert all(len(r["results"]) == 1 for r in out["results"])
    assert out["results"][0]["results"][0]["qualified_name"] == "PP::charge"
    assert out["embedding_dim"] == FakeEmbedder.dimension


@pytest.mark.asyncio
async def test_search_many_rejects_empty_list(engine):
    with pytest.raises(ValueError):
        await engine.search_many([])


@pytest.mark.asyncio
async def test_search_many_rejects_blank_query(engine):
    with pytest.raises(ValueError):
        await engine.search_many(["ok", ""])


@pytest.mark.asyncio
async def test_find_symbol_returns_matches(engine):
    engine.vector_db.points_by_name[("charge", None)] = [
        {"id": "a", "payload": {"name": "charge", "class_name": "PP",
                                "file": "p.py", "line_start": 10}},
    ]
    out = await engine.find_symbol("charge")
    assert out["matches"][0]["name"] == "charge"


@pytest.mark.asyncio
async def test_get_chunk_context_includes_source(engine):
    engine.vector_db.points_by_name[("charge", "PP")] = [
        {"id": "a", "payload": {"name": "charge", "class_name": "PP",
                                "file": "p.py", "line_start": 10,
                                "source": "def charge(): ...",
                                "calls": ["gateway.authorize"],
                                "called_by": []}},
    ]
    out = await engine.get_chunk_context("charge", "PP")
    assert out["found"] is True
    assert out["chunk"]["source"] == "def charge(): ..."
    assert out["chunk"]["calls"] == ["gateway.authorize"]


@pytest.mark.asyncio
async def test_get_chunk_context_missing_returns_found_false(engine):
    out = await engine.get_chunk_context("nope")
    assert out["found"] is False


@pytest.mark.asyncio
async def test_index_status_aggregates_counts(engine):
    engine.vector_db.scroll_results[PENDING] = [{"id": str(i)} for i in range(5)]
    engine.vector_db.scroll_results[FRESH] = [{"id": str(i)} for i in range(3)]
    out = await engine.index_status()
    assert out["counts"][PENDING] == 5
    assert out["counts"][FRESH] == 3
    assert out["total"] == 8


def test_build_server_registers_all_tools():
    """Sanity: MCP server creation works and the tool manifest is unchanged."""
    server = mcp_server.build_server()
    expected = {
        "search", "search_many", "find_symbol", "get_chunk_context", "index_status",
        "list_pending_enrichments", "save_enrichment", "enrichment_progress",
    }
    actual = {t.name for t in mcp_server.TOOLS}
    assert actual == expected
    assert server.name == "semhood"


def test_proxy_payload_injects_root():
    """Each tool maps to a daemon endpoint with the project root attached."""
    mcp_server._PROJECT_ROOT = "/fake/project"
    endpoint, payload = mcp_server._payload_for("search", {"query": "x", "top_k": 3})
    assert endpoint == "search"
    assert payload["root"] == "/fake/project"
    assert payload["query"] == "x" and payload["top_k"] == 3


# ------------------------------------------------------------------------
# Agent-driven enrichment
# ------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_pending_enrichments_returns_chunks(engine):
    engine.vector_db.scroll_results[PENDING] = [
        {"id": "c1", "payload": {
            "name": "charge", "class_name": "PP", "type": "method",
            "file": "p.py", "language": "python",
            "line_start": 10, "line_end": 40,
            "source": "def charge(self): pass",
            "calls": ["gateway.authorize"], "called_by": [],
        }},
        {"id": "c2", "payload": {
            "name": "refund", "class_name": "PP", "type": "method",
            "file": "p.py", "language": "python",
            "line_start": 50, "line_end": 60,
            "source": "def refund(self): pass",
        }},
    ]
    out = await engine.list_pending_enrichments(limit=5)
    assert out["count"] == 2
    assert out["chunks"][0]["qualified_name"] == "PP::charge"
    assert "next_steps" in out


@pytest.mark.asyncio
async def test_list_pending_truncates_long_source(engine):
    big_src = "x" * 10000
    engine.vector_db.scroll_results[PENDING] = [
        {"id": "c1", "payload": {
            "name": "big", "type": "function",
            "file": "p.py", "language": "python",
            "line_start": 1, "line_end": 100,
            "source": big_src,
        }},
    ]
    out = await engine.list_pending_enrichments(limit=1, max_source_chars=500)
    assert out["chunks"][0]["source_truncated"] is True
    assert len(out["chunks"][0]["source"]) == 500


@pytest.mark.asyncio
async def test_list_pending_rejects_bad_limit(engine):
    with pytest.raises(ValueError):
        await engine.list_pending_enrichments(limit=0)
    with pytest.raises(ValueError):
        await engine.list_pending_enrichments(limit=100)


@pytest.mark.asyncio
async def test_save_enrichment_writes_through(engine):
    engine.vector_db.points_by_id["c1"] = {
        "name": "charge", "class_name": "PP", "type": "method",
        "file": "p.py", "language": "python",
        "line_start": 10, "line_end": 40,
        "source": "def charge(): ...",
    }
    out = await engine.save_enrichment(
        "c1",
        "Authorizes a payment with retry on transient failures.",
        ["how to charge", "retry transient failure"],
    )
    assert out["state"] == FRESH
    assert out["queries_stored"] == 2
    assert "c1" in engine.vector_db.updated_vectors
    assert "description" in engine.vector_db.updated_vectors["c1"]
    assert engine.vector_db.updated_payloads["c1"]["enrichment_state"] == FRESH


@pytest.mark.asyncio
async def test_save_enrichment_rejects_unknown_chunk(engine):
    with pytest.raises(ValueError, match="not found"):
        await engine.save_enrichment("missing", "x", [])


@pytest.mark.asyncio
async def test_save_enrichment_rejects_empty_summary(engine):
    engine.vector_db.points_by_id["c1"] = {"name": "x", "type": "function",
                                           "file": "p.py", "language": "python",
                                           "line_start": 1, "line_end": 2}
    with pytest.raises(ValueError):
        await engine.save_enrichment("c1", "", ["q"])


@pytest.mark.asyncio
async def test_save_enrichment_persists_to_portable_store(engine):
    """The MCP route mirrors text to the portable store, keyed by source_hash."""
    engine.vector_db.points_by_id["c1"] = {
        "name": "charge", "type": "function", "file": "p.py", "language": "python",
        "line_start": 1, "line_end": 3, "source": "def charge(): ...",
        "source_hash": "abc123",
    }
    await engine.save_enrichment("c1", "Charges a card.", ["how to charge"])

    rec = await engine._get_enrichment_store().get("abc123")
    assert rec is not None
    assert rec["logic_summary"] == "Charges a card."
    assert rec["developer_queries"] == ["how to charge"]


@pytest.mark.asyncio
async def test_compact_enrichment_removes_orphans(engine):
    store = engine._get_enrichment_store()
    await store.put("live", logic_summary="l", developer_queries=[], enrichment_version=1)
    await store.put("orphan", logic_summary="o", developer_queries=[], enrichment_version=1)

    # Only "live" is still referenced by an indexed chunk (empty-filter scroll).
    engine.vector_db.scroll_results[None] = [{"id": "c1", "payload": {"source_hash": "live"}}]

    out = await engine.compact_enrichment()
    assert out["removed"] == 1
    assert out["kept"] == 1
    assert await store.get("orphan") is None
    assert await store.get("live") is not None


@pytest.mark.asyncio
async def test_enrichment_progress_aggregates(engine):
    engine.vector_db.scroll_results[PENDING] = [{"id": str(i)} for i in range(7)]
    engine.vector_db.scroll_results[FRESH] = [{"id": str(i)} for i in range(3)]
    out = await engine.enrichment_progress()
    assert out["pending_remaining"] == 7
    assert out["done"] == 3
    assert out["total"] == 10
