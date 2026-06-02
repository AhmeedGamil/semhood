"""EnrichmentStore — persistence, atomic upsert, concurrency, compaction."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from semhood.core.enrichment_store import EnrichmentStore


def _read_lines(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


async def test_put_then_get_roundtrips(tmp_path: Path):
    store = EnrichmentStore(tmp_path / ".semhood" / "enrichment.jsonl")
    await store.put(
        "hash-a", logic_summary="does a thing", developer_queries=["q1", "q2"], enrichment_version=1
    )
    rec = await store.get("hash-a")
    assert rec is not None
    assert rec["logic_summary"] == "does a thing"
    assert rec["developer_queries"] == ["q1", "q2"]
    assert rec["enrichment_version"] == 1


async def test_get_missing_returns_none(tmp_path: Path):
    store = EnrichmentStore(tmp_path / "enrichment.jsonl")
    assert await store.get("nope") is None


async def test_persists_across_instances(tmp_path: Path):
    path = tmp_path / ".semhood" / "enrichment.jsonl"
    s1 = EnrichmentStore(path)
    await s1.put("h1", logic_summary="s1", developer_queries=[], enrichment_version=1)

    # A fresh instance must load what the first one wrote.
    s2 = EnrichmentStore(path)
    rec = await s2.get("h1")
    assert rec is not None and rec["logic_summary"] == "s1"


async def test_put_is_upsert_not_append(tmp_path: Path):
    path = tmp_path / "enrichment.jsonl"
    store = EnrichmentStore(path)
    await store.put("h1", logic_summary="first", developer_queries=[], enrichment_version=1)
    await store.put("h1", logic_summary="second", developer_queries=["x"], enrichment_version=2)

    lines = _read_lines(path)
    assert len(lines) == 1  # replaced, not duplicated
    assert lines[0]["logic_summary"] == "second"
    assert lines[0]["enrichment_version"] == 2


async def test_file_is_sorted_by_hash(tmp_path: Path):
    path = tmp_path / "enrichment.jsonl"
    store = EnrichmentStore(path)
    for h in ("ccc", "aaa", "bbb"):
        await store.put(h, logic_summary=h, developer_queries=[], enrichment_version=1)

    hashes = [rec["source_hash"] for rec in _read_lines(path)]
    assert hashes == ["aaa", "bbb", "ccc"]  # deterministic order → clean git diffs


async def test_concurrent_puts_all_survive(tmp_path: Path):
    """Many simultaneous writers (worker gather / MCP fan-out) must not clobber
    each other — the lock serializes the rewrite, distinct keys all persist."""
    path = tmp_path / "enrichment.jsonl"
    store = EnrichmentStore(path)

    await asyncio.gather(
        *(
            store.put(f"h{i:03d}", logic_summary=f"s{i}", developer_queries=[], enrichment_version=1)
            for i in range(100)
        )
    )

    lines = _read_lines(path)
    assert len(lines) == 100
    assert {rec["source_hash"] for rec in lines} == {f"h{i:03d}" for i in range(100)}


async def test_tolerates_missing_file(tmp_path: Path):
    store = EnrichmentStore(tmp_path / "does" / "not" / "exist.jsonl")
    assert await store.count() == 0
    assert await store.get("x") is None


async def test_skips_corrupt_lines(tmp_path: Path):
    path = tmp_path / "enrichment.jsonl"
    path.write_text(
        '{"source_hash":"good","logic_summary":"ok","developer_queries":[],"enrichment_version":1}\n'
        "this is not json\n"
        '{"source_hash":"good2","logic_summary":"ok2","developer_queries":[],"enrichment_version":1}\n',
        encoding="utf-8",
    )
    store = EnrichmentStore(path)
    assert await store.get("good") is not None
    assert await store.get("good2") is not None
    assert await store.count() == 2  # the junk line is skipped, not fatal


async def test_compact_prunes_orphans_only(tmp_path: Path):
    path = tmp_path / "enrichment.jsonl"
    store = EnrichmentStore(path)
    for h in ("live1", "orphan", "live2"):
        await store.put(h, logic_summary=h, developer_queries=[], enrichment_version=1)

    removed = await store.compact(live_hashes={"live1", "live2"})
    assert removed == 1
    assert await store.get("orphan") is None
    assert await store.get("live1") is not None
    assert await store.get("live2") is not None


async def test_empty_put_ignored(tmp_path: Path):
    store = EnrichmentStore(tmp_path / "enrichment.jsonl")
    await store.put("", logic_summary="x", developer_queries=[], enrichment_version=1)
    assert await store.count() == 0
