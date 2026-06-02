"""The semhood daemon: one warm process serving every project.

This is the heart of the "install once, works everywhere" design. The daemon:

  * loads the embedding model **once** (the ~15s / ~500 MB cost) and keeps it
    warm for the life of the process, and
  * holds a registry of lightweight :class:`ProjectEngine` objects — one per
    project root — each pointing at that project's own central index under
    ``~/.semhood/indexes/<id>/``, all sharing the one embedder.

Every other entry point (the CLI, the editor's MCP server) is a thin HTTP
client to this process. So no matter how many terminals and editors you have
open, there is exactly one model in memory and exactly one writer per on-disk
index — which is required anyway, since local Qdrant locks its directory.

Binds to 127.0.0.1 only: this is a local developer tool, never a network
service. Auto-started on demand by the client; see :mod:`semhood.client`.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from semhood.config import (
    SemhoodConfig,
    create_embedder,
    create_llm,
    create_reranker,
    create_vector_db,
    load_global_config,
)
from semhood.core.policy import ParsingPolicy
from semhood.core.structural import StructuralBuilder
from semhood.engine import ProjectEngine
from semhood.parsing.parser import TreeSitterParser
from semhood.paths import (
    daemon_info_path,
    index_dir_for,
    list_indexed_projects,
    project_id,
)

logger = logging.getLogger("semhood.daemon")

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = int(os.environ.get("SEMHOOD_DAEMON_PORT", "7711"))


# =============================================================================
# Engine registry — shared providers + one engine per project
# =============================================================================


class EngineRegistry:
    """Owns the shared (warm) providers and a cache of per-project engines."""

    def __init__(self, config: SemhoodConfig):
        self.config = config

        # The expensive, warm provider — loaded once, shared by every project.
        self.embedder = create_embedder(config)

        # Cheap shared providers. The LLM is optional: search works with zero
        # keys, so we don't let a missing key stop the daemon from starting.
        self.reranker = create_reranker(config)
        self.parser = TreeSitterParser(allowed_languages=config.parsing.languages or None)
        self.structural = StructuralBuilder()
        self.parsing_policy = ParsingPolicy(
            exclude_dirs=config.parsing.exclude_dirs,
            exclude_files=config.parsing.exclude_files,
            max_file_size_mb=config.parsing.max_file_size_mb,
            max_chunks_per_file=config.parsing.max_chunks_per_file,
        )
        try:
            self.llm = create_llm(config)
        except Exception as exc:
            logger.warning("LLM unavailable (%s); enrich/query will be disabled", exc)
            self.llm = None

        self._engines: dict[str, ProjectEngine] = {}
        self._lock = asyncio.Lock()

        logger.info(
            "semhood daemon warm — embedder=%s dim=%d, llm=%s",
            config.embeddings.provider,
            self.embedder.dimension,
            config.llm.provider if self.llm else "disabled",
        )

    def _build_vector_db(self, root: str):
        """Create a vector store isolated to this project's central index."""
        provider = self.config.vector_db.provider
        pid = project_id(root)

        # Remote backends share one server → isolate by collection. On-disk
        # backends → isolate by directory under ~/.semhood/indexes/<id>/.
        if provider == "qdrant" and self.config.vector_db.qdrant.url:
            return create_vector_db(
                self.config, collection_override=f"semhood_{pid}"
            )
        subdir = "qdrant_data" if provider == "qdrant" else "chroma_data"
        path = str(index_dir_for(root) / subdir)
        return create_vector_db(self.config, path_override=path)

    async def get_engine(self, root: str) -> ProjectEngine:
        if not root:
            raise ValueError("'root' (project directory) is required")
        pid = project_id(root)

        existing = self._engines.get(pid)
        if existing is not None:
            return existing

        async with self._lock:
            existing = self._engines.get(pid)  # re-check under lock
            if existing is not None:
                return existing

            vector_db = self._build_vector_db(root)
            await vector_db.ensure_collection(vector_dim=self.embedder.dimension)
            engine = ProjectEngine(
                config=self.config,
                embedder=self.embedder,
                vector_db=vector_db,
                root=root,
                llm=self.llm,
                reranker=self.reranker,
                parser=self.parser,
                structural=self.structural,
                parsing_policy=self.parsing_policy,
            )
            self._engines[pid] = engine
            logger.info("opened index for project root=%s (id=%s)", root, pid)
            return engine


# =============================================================================
# Request models
# =============================================================================


class RootReq(BaseModel):
    root: str


class SearchReq(RootReq):
    query: str
    top_k: int = 8
    vectors: list[str] | None = None


class SearchManyReq(RootReq):
    queries: list[str]
    top_k: int = 8
    vectors: list[str] | None = None


class SymbolReq(RootReq):
    name: str
    class_name: str | None = None


class IndexReq(RootReq):
    path: str | None = None
    changed_only: bool = False
    reset: bool = False


class EnrichReq(RootReq):
    force: bool = False


class QueryReq(RootReq):
    question: str


class ListPendingReq(RootReq):
    limit: int = 5
    max_source_chars: int = 4000


class SaveEnrichmentReq(RootReq):
    chunk_id: str
    logic_summary: str
    developer_queries: list[str] = []


# =============================================================================
# App
# =============================================================================


def create_app(config: SemhoodConfig | None = None) -> FastAPI:
    app = FastAPI(title="semhood daemon", version="3.1.0")
    registry = EngineRegistry(config or load_global_config())
    app.state.registry = registry

    async def _run(root: str, coro_name: str, *args, **kwargs):
        """Resolve the project engine and invoke one of its methods, mapping
        validation errors to 400 and missing-capability errors to 409."""
        try:
            engine = await registry.get_engine(root)
            method = getattr(engine, coro_name)
            return await method(*args, **kwargs)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.get("/health")
    async def health():
        return {
            "ok": True,
            "embedder": registry.config.embeddings.provider,
            "embedding_dim": registry.embedder.dimension,
            "llm": registry.config.llm.provider if registry.llm else None,
            "projects_loaded": len(registry._engines),
        }

    @app.get("/projects")
    async def projects():
        return {"projects": list_indexed_projects()}

    @app.post("/search")
    async def search(req: SearchReq):
        return await _run(req.root, "search", req.query, req.top_k, req.vectors)

    @app.post("/search_many")
    async def search_many(req: SearchManyReq):
        return await _run(req.root, "search_many", req.queries, req.top_k, req.vectors)

    @app.post("/find_symbol")
    async def find_symbol(req: SymbolReq):
        return await _run(req.root, "find_symbol", req.name, req.class_name)

    @app.post("/get_chunk_context")
    async def get_chunk_context(req: SymbolReq):
        return await _run(req.root, "get_chunk_context", req.name, req.class_name)

    @app.post("/index_status")
    async def index_status(req: RootReq):
        return await _run(req.root, "index_status")

    @app.post("/index")
    async def index(req: IndexReq):
        return await _run(req.root, "index", req.path or req.root, req.changed_only, req.reset)

    @app.post("/enrich")
    async def enrich(req: EnrichReq):
        return await _run(req.root, "enrich", req.force)

    @app.post("/query")
    async def query(req: QueryReq):
        return await _run(req.root, "query", req.question)

    @app.post("/list_pending_enrichments")
    async def list_pending(req: ListPendingReq):
        return await _run(
            req.root, "list_pending_enrichments", req.limit, req.max_source_chars
        )

    @app.post("/save_enrichment")
    async def save_enrichment(req: SaveEnrichmentReq):
        return await _run(
            req.root, "save_enrichment", req.chunk_id, req.logic_summary,
            req.developer_queries,
        )

    @app.post("/enrichment_progress")
    async def enrichment_progress(req: RootReq):
        return await _run(req.root, "enrichment_progress")

    @app.post("/compact_enrichment")
    async def compact_enrichment(req: RootReq):
        return await _run(req.root, "compact_enrichment")

    @app.post("/shutdown")
    async def shutdown():
        # Graceful self-terminate, used by `semhood stop`.
        asyncio.get_event_loop().call_later(0.1, lambda: os._exit(0))
        return {"ok": True}

    return app


# =============================================================================
# Runner
# =============================================================================


def _write_info(host: str, port: int) -> None:
    daemon_info_path().write_text(
        json.dumps({"host": host, "port": port, "pid": os.getpid()}, indent=2),
        encoding="utf-8",
    )


def _clear_info() -> None:
    try:
        daemon_info_path().unlink()
    except OSError:
        pass


def run(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> None:
    """Start the daemon (blocking). Writes daemon.json so clients can find it."""
    import uvicorn

    logging.basicConfig(
        level=getattr(logging, os.environ.get("SEMHOOD_LOG_LEVEL", "INFO")),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    app = create_app()
    _write_info(host, port)
    try:
        uvicorn.run(app, host=host, port=port, log_level="warning")
    finally:
        _clear_info()


if __name__ == "__main__":
    run()
