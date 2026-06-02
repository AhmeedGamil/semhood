# Design B — Portable, Shareable Enrichment Store (+ config-change guardrails)

Status: **proposed** — spec for review before implementation. Target release: **v3.1.0**.

---

## 1. Problem

Enrichment (the LLM-written `logic_summary` + `developer_queries`) is the **expensive** part of
semhood — it costs API tokens or agent time. Today it is stored in **only one place**: the Qdrant
payload inside `~/.semhood/indexes/<id>/` ([enrichment_writer.py](../../semhood/core/enrichment_writer.py)).

That couples the expensive *text* to the disposable *vectors*. Consequences:

- **Change the embedding model → lose all enrichment.** A new model means a new vector dimension,
  which means the collection must be recreated. `get_points_by_file` then returns nothing, so the
  indexer's `preserve_enrichment` check ([indexer.py:155](../../semhood/core/indexer.py#L155)) always
  fails and every chunk is reset to `pending`. **All enrichment text is gone → re-pay every token.**
- **No team sharing.** Each developer re-enriches identical code from scratch.
- **Silent failure on model swap.** `ensure_collection` returns early if the collection exists
  *without checking the dimension* ([qdrant_provider.py:52](../../semhood/providers/vector_db/qdrant_provider.py#L52)),
  so switching the model just errors mid-upsert with a cryptic Qdrant message.

**Key insight:** the enrichment *text* is keyed by `source_hash` (content) and is **model-agnostic**.
Only the *vectors* are model-specific. So the text can — and should — live in a separate,
content-addressed, **committed** file. Changing the embedder then becomes *re-embed* (cheap/free),
not *re-enrich* (expensive).

---

## 2. The storage split

| Artifact | Location | In git? | Rebuild cost |
|---|---|---|---|
| Vectors / index | `~/.semhood/indexes/<id>/` (outside repo) | ❌ never | cheap (re-embed) |
| Global config | `~/.semhood/config.yaml` | ❌ never | n/a |
| **Enrichment text (this feature)** | **`<repo>/.semhood/enrichment.jsonl`** | ✅ **committed** | expensive (LLM) |

`.semhood` is already a recognized project-root marker
([paths.py:38](../../semhood/paths.py#L38)), so the in-repo `.semhood/` directory fits the existing model.

---

## 3. File format

`<repo>/.semhood/enrichment.jsonl` — one JSON object per line, **sorted by `source_hash`** on every
write (stable, line-based git diffs):

```jsonl
{"source_hash":"7b21...","logic_summary":"Paginates the results list.","developer_queries":["pagination","page size"],"enrichment_version":1}
{"source_hash":"a3f9...","logic_summary":"Validates a JWT and returns the user.","developer_queries":["how is auth verified","jwt validation"],"enrichment_version":1}
```

- **Key:** `source_hash` (content-addressed → portable across model/line/file moves).
- **No secrets** ever (keys stay in `~/.semhood/.env`).
- Records for old hashes are kept as a **cross-version cache** (branch switch / revert reuse);
  optional `--compact` prunes truly-orphaned ones.

---

## 4. `EnrichmentStore` (new — `semhood/core/enrichment_store.py`)

One instance **per project**, owned by the `Engine`. In-memory map is the source of truth; the file
is its durable mirror.

```python
class EnrichmentStore:
    def __init__(self, path: Path):
        self._path = path                 # <repo>/.semhood/enrichment.jsonl
        self._map: dict[str, dict] = {}   # source_hash -> record
        self._lock = asyncio.Lock()
        self._loaded = False

    async def _ensure_loaded(self) -> None:
        # lazy: read the JSONL into _map once
        ...

    async def get(self, source_hash: str) -> dict | None:
        await self._ensure_loaded()
        return self._map.get(source_hash)

    async def put(self, source_hash: str, record: dict) -> None:
        async with self._lock:            # serializes ALL writers (worker + MCP + subagents)
            await self._ensure_loaded()
            self._map[source_hash] = record
            await self._flush()           # atomic: write .tmp then os.replace()

    async def _flush(self) -> None:
        tmp = self._path.with_suffix(".jsonl.tmp")
        lines = [json.dumps(self._map[h], separators=(",", ":"))
                 for h in sorted(self._map)]
        tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
        os.replace(tmp, self._path)       # atomic on Windows + POSIX
```

**Concurrency:** single `asyncio.Lock`; sufficient because the daemon is single-process /
single-event-loop. (Multi-worker would need an OS file lock — out of scope, documented as a constraint.)
Generation + embedding happen *outside* the lock (parallel); only the millisecond file write is serialized.

**Durability:** per-write (`put` flushes immediately) — the MCP path has no "end of run", so
buffer-to-end is not an option. The `semhood enrich` worker may add one final flush as a backstop only.

---

## 5. Write side — hook into the single choke point

Both enrichment routes funnel through `write_enrichment`
([enrichment_writer.py:37](../../semhood/core/enrichment_writer.py#L37)):
the `Enricher` worker ([enricher.py:211](../../semhood/core/enricher.py#L211)) and the MCP
`save_enrichment` ([engine.py:272](../../semhood/engine.py#L272)). Add one optional param and one step:

```python
async def write_enrichment(*, vector_db, embedder, point_id, chunk, summary, queries,
                           version, final_state=FRESH, store: EnrichmentStore | None = None):
    ...                                   # existing: embed + update_vectors_and_payload
    if store is not None and chunk.source_hash:
        await store.put(chunk.source_hash, {
            "source_hash": chunk.source_hash,
            "logic_summary": summary,
            "developer_queries": queries,
            "enrichment_version": version,
        })
```

One change covers **all three routes** (API LLM, Ollama, MCP agent + subagents).

---

## 6. Read side — import / read-back in the indexer

In the per-chunk loop ([indexer.py:148](../../semhood/core/indexer.py#L148)), when the vector-DB
preserve check fails (e.g. the collection was wiped for a model change), fall back to the store:

```
new_hash = source_hash(chunk)
if vector-DB preserve hit:                 # unchanged code, same dim collection still present
    reuse text + reuse stored vectors      # (existing behavior)
elif store.get(new_hash):                  # NEW: text exists for this exact content
    chunk.logic_summary / developer_queries / description = stored
    chunk.enrichment_state = FRESH
    re-embed the text → description + developer_queries vectors   # NO LLM
else:
    chunk.enrichment_state = PENDING        # changed or never enriched → re-enrich
```

Hit/miss table:

| `store.get(current source_hash)` | meaning | action |
|---|---|---|
| HIT | code unchanged since enrichment | restore text, **re-embed only**, mark `fresh` |
| MISS | code changed / never enriched | mark `pending` → re-enrich |

**Safety guarantee:** matching is by `source_hash`, so stale enrichment can never attach to changed
code — a changed chunk simply MISSes and falls to `pending`. Importing the file is always safe.

---

## 7. Path helper (new — `semhood/paths.py`)

```python
def enrichment_file_for(root: str | Path, *, create: bool = True) -> Path:
    """<repo>/.semhood/enrichment.jsonl for the given project root."""
    d = find_project_root(root) / ".semhood"
    if create:
        d.mkdir(parents=True, exist_ok=True)
    return d / "enrichment.jsonl"
```

The directory lives in the **repo**, not in `~/.semhood`. Drop a `.semhood/.gitignore` that keeps
`enrichment.jsonl` tracked but ignores any future local scratch files. Do **not** auto-edit the
user's root `.gitignore`; document "commit `.semhood/enrichment.jsonl` to share enrichment."

---

## 8. Config-change guardrails (ships with B)

The config lets developers swap embedder/vector_db/llm. Changing the **embedding model** changes the
vector dimension. Guardrails:

1. **Dimension-change warning.** Before re-indexing, compare `embedder.dimension` to the existing
   collection's dimension. If different: warn —
   *"Embedding model changed (768→1536). Re-indexing will rebuild vectors. Your enrichment is preserved
   and will be re-embedded (no LLM tokens). Continue?"* (With B, this is reassuring, not destructive.)
2. **Fix `ensure_collection`** ([qdrant_provider.py:52](../../semhood/providers/vector_db/qdrant_provider.py#L52))
   to detect an existing-collection dimension mismatch and raise a clear, actionable error instead of
   the cryptic mid-upsert Qdrant failure.
3. **Warn before wiping enrichment** (e.g. `--force` / destructive reindex): show how many enriched
   chunks / rough token value at stake.
4. **Daemon config caching:** editing `~/.semhood/config.yaml` requires `semhood stop` to take effect
   (config loaded once at [daemon.py:194](../../semhood/daemon.py#L194)). Document; optionally detect
   config mtime change and warn.

---

## 9. Wiring

`Engine` ([engine.py](../../semhood/engine.py)) holds `self.root`, `self.embedder`, `self.vector_db`
and lazily builds `Indexer`/`Enricher`. Add:

- `self._enrichment_store = EnrichmentStore(enrichment_file_for(self.root))` (lazy).
- Pass `store=self._enrichment_store` into the `Enricher` and into `save_enrichment`'s
  `write_enrichment` call.
- Pass the store into the `Indexer` for read-back.

Per-project store ⇒ per-project lock ⇒ different projects never contend; same-project writers
(worker, MCP, subagents) fully serialized.

---

## 10. Subagent fan-out (enables free parallel enrichment)

The MCP queue has **no claim/lease** — `list_pending_enrichments` hands every caller the same pending
set ([engine.py:242](../../semhood/engine.py#L242)). For v3.1 use **orchestrator-partitioning**: the
top agent fetches all pending `chunk_id`s once and gives each subagent a **disjoint slice**. A proper
`in_progress` lease state is a later (v3.2) robustness upgrade. The store's lock already makes the
resulting concurrent `save_enrichment` writes safe.

---

## 11. Implementation order

1. `EnrichmentStore` + unit tests (concurrency, atomic write, sorted output, upsert).
2. `enrichment_file_for` path helper + `.semhood/.gitignore` scaffolding.
3. Wire `store` through `write_enrichment` (+ `Enricher`, `save_enrichment`).
4. Indexer read-back (hit → re-embed, miss → pending) + tests.
5. Config guardrails: `ensure_collection` dimension check + dimension-change warning + reindex/force warnings.
6. `--compact` command (prune orphaned records).
7. Docs (README section: what to commit, model-change story) + bump `pyproject.toml` → `3.1.0`.
8. Cut GitHub Release `v3.1.0` → triggers PyPI publish.

---

## 12. Out of scope (later)

- `in_progress` lease state for the pending queue (v3.2).
- Coalesced/debounced flush (only if heavy fan-out shows the per-write rewrite as a bottleneck).
- OS-level file lock (only if the daemon ever runs multi-process).
