"""
Pydantic models used across the semhood pipeline.

These models define the language-agnostic data structures that flow through
every stage: parsing → enrichment → embedding → storage → retrieval → answer.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

# =============================================================================
# Enums
# =============================================================================


class ChunkType(str, Enum):
    """Type of AST node this chunk represents."""

    FUNCTION = "function"
    METHOD = "method"
    CLASS = "class"
    CONSTRUCTOR = "constructor"
    PROPERTY = "property"
    INTERFACE = "interface"
    MODULE = "module"


# =============================================================================
# Sub-models
# =============================================================================


class Parameter(BaseModel):
    """A function/method parameter."""

    name: str
    type: str | None = None
    default: str | None = None


class CallerRef(BaseModel):
    """A reference to a calling function (used in called_by)."""

    function: str
    class_name: str | None = None
    file: str
    line: int


# =============================================================================
# Chunk — the core data unit
# =============================================================================


class Chunk(BaseModel):
    """
    Language-agnostic chunk extracted from source code.

    This is the universal data structure that flows through the entire pipeline.
    Every language extractor produces Chunk objects, and every downstream
    component (call graph, enricher, embedder, storage) consumes them.
    """

    # Identity
    name: str = Field(description="Function/method/class name")
    type: ChunkType = Field(description="AST node type")
    class_name: str | None = Field(default=None, description="Parent class name")
    namespace: str | None = Field(default=None, description="Module/package/namespace")
    file: str = Field(description="Relative file path")
    language: str = Field(description="Language identifier (python, javascript, etc.)")
    line_start: int
    line_end: int

    # Signature
    visibility: str = "public"
    is_static: bool = False
    parameters: list[Parameter] = Field(default_factory=list)
    return_type: str | None = None
    throws: list[str] = Field(default_factory=list)

    # Content
    docblock: str | None = None
    source: str = ""
    calls: list[str] = Field(default_factory=list)

    # Class members (populated by extractors for CLASS chunks)
    methods: list[str] = Field(default_factory=list, description="Method names in this class")

    # Call graph (populated by CallGraphBuilder)
    called_by: list[CallerRef] = Field(default_factory=list)

    # Smart strategy fields (populated by LLM enrichment)
    logic_summary: str | None = None
    description: str | None = None
    developer_queries: list[str] = Field(default_factory=list)

    # Enrichment lifecycle (Stage 2 worker uses these)
    source_hash: str | None = Field(
        default=None, description="sha256 hex of chunk.source — used for incremental re-enrichment"
    )
    enrichment_state: str = Field(
        default="pending",
        description="One of: pending | fresh | failed | skipped_trivial",
    )
    enrichment_version: int = Field(
        default=0, description="Bumps when prompt or model changes"
    )
    enriched_at: str | None = Field(
        default=None, description="ISO 8601 timestamp of last enrichment"
    )
    enrichment_attempts: int = 0
    enrichment_error: str | None = None

    # Route/endpoint metadata (populated by route parsers)
    is_endpoint_handler: bool = False
    route: str | None = None
    http_method: str | None = None
    middleware: list[str] = Field(default_factory=list)

    # Git metadata
    last_modified: str | None = None
    git_blame: str | None = None
    commit_message: str | None = None

    # Vectors (populated by embedder, not stored in payload)
    vectors: dict[str, list[float]] = Field(default_factory=dict, exclude=True)
    sparse: dict = Field(default_factory=dict, exclude=True)

    def qualified_name(self) -> str:
        """Return fully-qualified name like 'ClassName::methodName'."""
        if self.class_name:
            return f"{self.class_name}::{self.name}"
        return self.name

    def to_payload(self) -> dict:
        """Convert to Qdrant payload dict (excludes vectors)."""
        return self.model_dump(exclude={"vectors", "sparse"})


# =============================================================================
# Pipeline results
# =============================================================================


class IndexResult(BaseModel):
    """Result of an indexing operation."""

    files_indexed: int = 0
    chunks_indexed: int = 0
    chunks_skipped: int = 0
    stale_callers_cleaned: int = 0


class SearchResult(BaseModel):
    """A single search result from the vector DB."""

    id: str
    score: float
    payload: dict = Field(default_factory=dict)


class QueryResult(BaseModel):
    """Final result of a query."""

    answer: str
    sources: list[dict] = Field(default_factory=list)


class QueryTraceResult(QueryResult):
    """Query result with intermediate pipeline data for evaluation."""

    expanded_query: str = ""
    merged_count: int = 0
    reranked: list[dict] = Field(default_factory=list)
    with_neighbors: list[dict] = Field(default_factory=list)
    context: str = ""
