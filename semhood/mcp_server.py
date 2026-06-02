"""MCP server for semhood — a thin proxy to the warm daemon.

Exposes semantic-search and structural-context tools to any MCP-aware host:
Claude Desktop, Cursor, Cline, Continue, Kiro, Zed, Windsurf.

Architecture: this process holds **no** model and **no** index. Each tool call
is forwarded over localhost to the semhood daemon (auto-started on first use),
tagged with the current project's root. That means:

  * register this server **once, globally**, in your editor — it works in every
    workspace you open, with no per-project config; and
  * no matter how many editors connect, there is one warm embedder and one
    writer per on-disk index (required, since local Qdrant locks its dir).

The project root is resolved once at startup, in priority order:
    --root CLI arg  >  $SEMHOOD_PROJECT_ROOT  >  nearest VCS root of the cwd.
Editors that launch the server with the workspace as cwd (most do) need no
extra configuration; others can pass ``--root ${workspaceFolder}``.

Run with:
    semhood-mcp                         # root = cwd's project
    semhood-mcp --root /path/to/repo    # explicit project root
"""

from __future__ import annotations

import argparse
import asyncio
import json as _json
import logging
import os
import sys
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from semhood import client
from semhood.paths import find_project_root

logger = logging.getLogger("semhood.mcp")

# Resolved once at startup; every tool call is keyed to this project.
_PROJECT_ROOT: str = ""


def _resolve_root(arg_root: str | None) -> str:
    if arg_root:
        return str(find_project_root(arg_root))
    env_root = os.environ.get("SEMHOOD_PROJECT_ROOT")
    if env_root:
        return str(find_project_root(env_root))
    return str(find_project_root())


# =============================================================================
# Tool manifest — unchanged contract; handlers just proxy to the daemon
# =============================================================================


TOOLS: list[Tool] = [
    Tool(
        name="search",
        description=(
            "Semantic search over the indexed codebase. Returns the top-k "
            "code chunks (functions, methods, classes) ranked by relevance. "
            "Use this BEFORE reading files when looking for code by intent or "
            "behavior — saves tokens vs. grep-then-read."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural-language description of what you're looking for.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Maximum number of results (default 8).",
                    "default": 8, "minimum": 1, "maximum": 50,
                },
                "vectors": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Which named vectors to search. Default is all three: "
                        "code (raw source embedding), description (LLM summary), "
                        "developer_queries (LLM-generated NL queries). Pre-enrichment "
                        "only 'code' is populated."
                    ),
                    "default": ["code", "description", "developer_queries"],
                },
            },
            "required": ["query"],
        },
    ),
    Tool(
        name="search_many",
        description=(
            "Run several semantic searches in ONE call. Pass a list of queries "
            "and get results grouped per query. Prefer this over calling 'search' "
            "repeatedly when you're looking up multiple things at once (e.g. "
            "several methods or concepts) — one round-trip, queries embedded in a "
            "single batch."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "queries": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Natural-language queries to search for, one result set each.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Maximum results per query (default 8).",
                    "default": 8, "minimum": 1, "maximum": 50,
                },
            },
            "required": ["queries"],
        },
    ),
    Tool(
        name="find_symbol",
        description=(
            "Find code chunks by exact name. Useful when you know a function or "
            "class name and want its location + metadata (no embedding required, "
            "exact match)."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Exact function, method, or class name."},
                "class_name": {
                    "type": "string",
                    "description": "Optional class name to disambiguate methods that share a name.",
                },
            },
            "required": ["name"],
        },
    ),
    Tool(
        name="get_chunk_context",
        description=(
            "Get the full context for a chunk: source code, calls, called_by, and "
            "summary. Use after `search` or `find_symbol` to read the chunk's body "
            "without a separate read_file call."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Function/method/class name."},
                "class_name": {
                    "type": "string",
                    "description": "Optional class name for disambiguation.",
                },
            },
            "required": ["name"],
        },
    ),
    Tool(
        name="index_status",
        description=(
            "Report the indexed-chunk counts per enrichment state and basic config. "
            "Use to check whether the index exists and whether enrichment has run."
        ),
        inputSchema={"type": "object", "properties": {}},
    ),
    Tool(
        name="list_pending_enrichments",
        description=(
            "Get a batch of code chunks that need LLM enrichment. Use this when "
            "the user asks you to enrich the codebase index and you want to act "
            "as the enricher yourself (vs. running `semhood enrich` which uses "
            "its own configured LLM key). Returns each chunk's source, signature, "
            "calls, and called_by — everything you need to write a useful summary. "
            "After each chunk, call save_enrichment(chunk_id=...) with your "
            "logic_summary and developer_queries."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "How many chunks to fetch in this batch.",
                    "default": 5, "minimum": 1, "maximum": 50,
                },
                "max_source_chars": {
                    "type": "integer",
                    "description": (
                        "Truncate each chunk's source to this many characters to "
                        "keep tool-result payloads small. Sufficient for the summary task."
                    ),
                    "default": 4000, "minimum": 200,
                },
            },
        },
    ),
    Tool(
        name="save_enrichment",
        description=(
            "Persist an enrichment you wrote for a single chunk. Embeds the "
            "summary + queries into the description and developer_queries "
            "vectors and flips the chunk's state from 'pending' to 'fresh'. "
            "Call this once per chunk returned from list_pending_enrichments."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "chunk_id": {
                    "type": "string",
                    "description": "The chunk's id from list_pending_enrichments.",
                },
                "logic_summary": {
                    "type": "string",
                    "description": (
                        "1-3 sentences in plain language: what the chunk does and "
                        "why it exists. Don't restate the function name — describe "
                        "behavior, key operations, returns, side effects."
                    ),
                },
                "developer_queries": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "5-10 short search queries a developer might type to find "
                        "this code. Be specific — include class names, error types, "
                        "domain vocabulary. Avoid generic queries."
                    ),
                },
            },
            "required": ["chunk_id", "logic_summary", "developer_queries"],
        },
    ),
    Tool(
        name="enrichment_progress",
        description=(
            "Quick stats for the enrichment loop: counts per state, total chunks, "
            "and how many are still pending. Use to decide when to stop calling "
            "list_pending_enrichments."
        ),
        inputSchema={"type": "object", "properties": {}},
    ),
]


# =============================================================================
# Proxy: map each tool to a daemon endpoint + payload
# =============================================================================


def _payload_for(name: str, args: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Return (daemon_endpoint, payload) for a tool call, injecting the root."""
    base = {"root": _PROJECT_ROOT}
    if name == "search":
        return "search", {**base, "query": args.get("query"),
                          "top_k": args.get("top_k", 8),
                          "vectors": args.get("vectors")}
    if name == "search_many":
        return "search_many", {**base, "queries": args.get("queries") or [],
                              "top_k": args.get("top_k", 8),
                              "vectors": args.get("vectors")}
    if name == "find_symbol":
        return "find_symbol", {**base, "name": args.get("name"),
                              "class_name": args.get("class_name")}
    if name == "get_chunk_context":
        return "get_chunk_context", {**base, "name": args.get("name"),
                                    "class_name": args.get("class_name")}
    if name == "index_status":
        return "index_status", base
    if name == "list_pending_enrichments":
        return "list_pending_enrichments", {**base, "limit": args.get("limit", 5),
                                           "max_source_chars": args.get("max_source_chars", 4000)}
    if name == "save_enrichment":
        return "save_enrichment", {**base, "chunk_id": args.get("chunk_id"),
                                  "logic_summary": args.get("logic_summary"),
                                  "developer_queries": args.get("developer_queries") or []}
    if name == "enrichment_progress":
        return "enrichment_progress", base
    raise ValueError(f"unknown tool: {name}")


async def _call(name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Forward one tool call to the daemon (off the event loop)."""
    endpoint, payload = _payload_for(name, args)
    # urllib is synchronous; keep the asyncio loop responsive.
    return await asyncio.to_thread(client.request, endpoint, payload)


# =============================================================================
# MCP server wiring
# =============================================================================


def build_server() -> Server:
    """Construct the MCP server and register all tools."""
    server = Server("semhood")

    @server.list_tools()
    async def _list_tools() -> list[Tool]:
        return TOOLS

    @server.call_tool()
    async def _call_tool(name: str, arguments: dict | None) -> list[TextContent]:
        try:
            result = await _call(name, arguments or {})
        except Exception as exc:
            logger.exception("tool %s failed", name)
            return [TextContent(type="text",
                                text=_json.dumps({"error": str(exc), "tool": name}, default=str))]
        return [TextContent(type="text", text=_json.dumps(result, indent=2, default=str))]

    return server


# =============================================================================
# Entry point
# =============================================================================


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="semhood-mcp",
        description="MCP server for semhood semantic code search (proxies to the daemon).",
    )
    parser.add_argument(
        "--root", default=None,
        help="Project root to search (default: $SEMHOOD_PROJECT_ROOT or the cwd's repo).",
    )
    # Accepted for backwards compatibility with older editor configs; the daemon
    # now owns config via ~/.semhood/config.yaml, so this is informational only.
    parser.add_argument("-c", "--config", default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--log-level", default=os.environ.get("SEMHOOD_LOG_LEVEL", "INFO"),
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args(argv)


async def _serve(root: str) -> None:
    server = build_server()
    init_options = server.create_initialization_options()
    async with stdio_server() as (read_stream, write_stream):
        logger.info("semhood MCP proxy ready — project root=%s", root)
        await server.run(read_stream, write_stream, init_options)


def main(argv: list[str] | None = None) -> int:
    global _PROJECT_ROOT
    args = _parse_args(argv)
    # MCP uses stdout for JSON-RPC frames; logging MUST go to stderr.
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        stream=sys.stderr,
    )
    _PROJECT_ROOT = _resolve_root(args.root)
    try:
        asyncio.run(_serve(_PROJECT_ROOT))
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        logger.exception("MCP server crashed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
