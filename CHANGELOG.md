# Changelog

All notable changes to semhood are documented here. This project follows
[Semantic Versioning](https://semver.org/).

## [0.1.0] — 2026-06-02

Initial public release of **semhood** — AST-based, local-first semantic code
search for humans and AI agents. Every result ships with its call graph (what it
calls + what calls it).

### Highlights
- **AST-based chunking** via tree-sitter — every function, method, and class
  becomes a chunk carrying its signature, docstring, and call graph. Languages:
  Python, JavaScript, TypeScript, Java, Go, PHP, C#, Ruby, Rust, C++.
- **Local-first** — runs fully offline with a local embedder and zero API keys.
  Remote providers (Voyage, OpenAI) and rerankers (Cohere) are a one-line opt-in.
- **One warm daemon, many thin clients** — the embedding model loads once and
  stays warm machine-wide. The CLI and MCP server are thin clients that talk to
  the daemon over localhost; repeated searches are ~80–130 ms instead of cold
  model loads. Auto-starts on first use (`semhood serve` / `semhood stop`).
- **Central, per-project index store** at `~/.semhood/indexes/<id>/`, keyed by
  git root — no per-project config file. One global config at
  `~/.semhood/config.yaml`, created with working defaults on first run.
- **Portable LLM enrichment** — optional logic summary + developer queries per
  chunk, committed once to `.semhood/enrichment.jsonl` and shared with your team.
- **MCP server** (`semhood-mcp`) — register once, globally; it exposes search and
  enrichment tools to Claude Desktop, Cursor, Cline, Continue, Kiro, Zed, and any
  MCP-aware client, and forwards each call to the daemon tagged with the editor's
  project root.
- **`semhood install-skill`** — installs bundled usage guidance into your AI
  editors at the user level, each in its own convention (Claude Code skill,
  Cursor/Windsurf rules, Kiro steering, Codex `AGENTS.md`). Idempotent.
