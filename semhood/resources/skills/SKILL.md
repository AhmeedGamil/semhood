---
name: semhood
description: Use semantic code search before reading files. Find relevant chunks by intent (e.g. "where is auth handled?") via the semhood MCP server, then read only what's needed.
keywords:
  - code search
  - semantic search
  - find function
  - find class
  - codebase
  - where is
  - how does
  - implementation
  - semhood
---

# semhood — semantic code search for AI agents

## When to use this skill

**Use semhood instead of grep/read_file when** the user asks anything that requires understanding what code *does*, not just where a string appears:

- "where is X handled / implemented / validated?"
- "how does X work?"
- "find the function that does Y"
- "show me how we retry / cache / authenticate / parse Z"
- any "explain", "why", "how", "where" question about the codebase
- before reading files when you don't already know exactly which file to open

**Use ripgrep/read_file directly when** you already know the exact file path or you want a literal-string match (e.g. find all callers of a specific function name, find a specific log message, find a TODO comment).

## Available MCP tools

Four for retrieval, three for agent-driven enrichment.

### Retrieval

- **`search`** — semantic search. Args: `query` (string, required), `top_k` (int, default 8), `vectors` (array, default all). Returns ranked chunks with file, line, qualified_name, score, and a short summary. Always start here.
- **`find_symbol`** — exact-name lookup. Args: `name` (required), `class_name` (optional disambiguator). Use when you already know the symbol name.
- **`get_chunk_context`** — full source + calls + called_by for a known symbol. Args: `name`, optional `class_name`. Use after `search` or `find_symbol` when you need the body without a separate `read_file` call.
- **`index_status`** — counts per enrichment state. Use to check if the index exists and whether stage 2 has run.

### Agent-driven enrichment (alternative to `semhood enrich` CLI)

When the user asks to "enrich the index" or "improve search quality", you can act as the enricher yourself:

- **`list_pending_enrichments`** — get a batch of unenriched chunks with full source + signature + calls + called_by. Args: `limit` (1-50, default 5), `max_source_chars` (default 4000).
- **`save_enrichment`** — persist your written summary + queries for one chunk. Args: `chunk_id`, `logic_summary` (1-3 sentences), `developer_queries` (5-10 short search phrases).
- **`enrichment_progress`** — loop sentinel. Returns counts per state and `pending_remaining`. Stop looping when it hits 0.

## Standard workflows

### Workflow 1: answering "where is X?" or "how does X work?"

```
1. Call search(query="natural-language description of intent", top_k=5)
2. Inspect the top 2-3 results. Each has file, line_start, qualified_name, summary.
3. If the summary already answers the question, cite it and stop.
4. Otherwise call get_chunk_context(name=..., class_name=...) on the most likely match
   to fetch its source without a separate read_file.
5. Read 1-2 related files only if needed.
```

Compare to grep-first: ~50K tokens reading scattered files vs. ~2K tokens for a focused answer.

### Workflow 2: navigating call graph

```
1. search() to find the relevant chunk.
2. get_chunk_context() — its `calls` field lists what it calls; `called_by` lists callers.
3. Follow either direction with find_symbol() to fetch each related chunk.
```

### Workflow 3: agent-driven enrichment

User asks: *"enrich the codebase index"* or *"improve the search quality"*.

```
1. enrichment_progress()  → see how many chunks are pending
2. Loop:
   a. list_pending_enrichments(limit=5)  → get a batch with source + signature
   b. For each chunk:
      - Read the source.
      - Write a 1-3 sentence logic_summary in plain language. Describe behavior,
        key operations, returns, side effects. Do NOT restate the function name.
      - Write 5-10 specific developer_queries. Each should be a short phrase a
        developer would type to find this code. Include class names, error types,
        domain vocabulary. Avoid generic queries like "what does this do".
      - Call save_enrichment(chunk_id, logic_summary, developer_queries)
   c. Repeat from (a) until enrichment_progress shows pending_remaining == 0.
3. Confirm to the user with the final progress count.
```

Be deliberate — the summaries you write get embedded into the `description` and `developer_queries` vectors and are how future searches find this code. Quality matters more than speed.

## Output handling

All tool results come back as JSON in a `TextContent` block. Parse it. Each `search` hit looks like:

```json
{
  "score": 0.42,
  "name": "charge",
  "class_name": "PaymentProcessor",
  "qualified_name": "PaymentProcessor::charge",
  "file": "payments.py",
  "line_start": 45,
  "line_end": 86,
  "language": "python",
  "type": "method",
  "enrichment_state": "fresh",
  "summary": "Authorizes and captures a payment with retry on transient gateway errors..."
}
```

When citing a result to the user, prefer `qualified_name` + `file:line_start` so they can click through.

## Pitfalls and gotchas

- **`enrichment_state == "pending"`** on every result means stage 2 hasn't run. The `description` / `developer_queries` vectors will be empty and recall on natural-language queries is weaker. Flag this to the user; the fix is `semhood enrich` or the agent-driven workflow above.
- **Empty results** can mean an index hasn't been built yet (`semhood index .`), or the dimension changed (delete the project's dir under `~/.semhood/indexes/` and re-index), or the query is too short. Try a longer phrase before assuming nothing exists.
- **The `code` vector matches on raw source**, so query terms that appear in the code (function names, error strings) work even pre-enrichment. The `description` and `developer_queries` vectors handle paraphrasing.
- **Don't call `search` and immediately `read_file` on every result.** Read the `summary` field first; often it's enough to answer the question without opening any file.
- **Tool results are advisory rankings, not ground truth.** Trust the top-1/top-3, verify before quoting line ranges in code edits.

## Quick command reference

The user-side CLI commands the user might mention:

| Command | Purpose |
|---|---|
| `semhood index .` | Build/update the structural index. Run when the codebase changes. |
| `semhood enrich` | Stage 2 — uses semhood's configured LLM key. |
| `semhood enrich --force` | Re-enrich every chunk (after prompt/version bumps). |
| `semhood status` | Counts per state + provider summary. Same info as `index_status`. |
| `semhood-mcp` | MCP server — a thin proxy to the warm daemon (what you're talking to right now). |
