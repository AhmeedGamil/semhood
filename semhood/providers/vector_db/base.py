"""Abstract interface for vector database providers."""

from __future__ import annotations

from abc import ABC, abstractmethod


class VectorDBProvider(ABC):
    """Abstract base class for vector database operations.

    The collection always carries three named dense vectors:
      - ``code``               — populated at index time (StructuralBuilder)
      - ``description``        — populated at enrichment time
      - ``developer_queries``  — populated at enrichment time

    Plus one sparse vector ``bm25``.

    Stage 1 (``semhood index``) writes only ``code`` + ``bm25`` and sets
    ``enrichment_state = pending`` on the payload. Stage 2 (``semhood
    enrich``) drains the pending queue and patches in the description /
    developer_queries vectors.
    """

    # ------------------------------------------------------------------
    # Collection lifecycle
    # ------------------------------------------------------------------

    @abstractmethod
    async def ensure_collection(self, vector_dim: int, recreate: bool = False) -> None:
        """Create the collection if it doesn't exist.

        The collection always has three named dense vectors plus the
        ``bm25`` sparse vector. There is no longer a ``strategy`` knob.

        If the collection exists with a different vector dimension (the
        embedding model changed), implementations must raise a clear error
        rather than fail later. ``recreate=True`` drops and rebuilds it.
        """
        ...

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    @abstractmethod
    async def upsert(self, points: list[dict]) -> None:
        """Upsert a batch of points.

        Each point dict contains:
          - ``id``: str (deterministic UUID)
          - ``vectors``: dict[str, list[float]] — named vectors (subset of
            {code, description, developer_queries})
          - ``sparse``: dict — BM25 indices + values
          - ``payload``: dict — chunk metadata
        """
        ...

    @abstractmethod
    async def delete_by_file(self, file_path: str) -> None:
        """Delete all points belonging to a file path."""
        ...

    @abstractmethod
    async def delete_by_ids(self, ids: list[str]) -> None:
        """Delete a batch of points by ID."""
        ...

    @abstractmethod
    async def set_payload_by_id(self, point_id: str, payload: dict) -> None:
        """Update specific payload fields on an existing point."""
        ...

    @abstractmethod
    async def update_vectors_and_payload(
        self,
        point_id: str,
        vectors: dict[str, list[float]],
        payload: dict,
    ) -> None:
        """Atomically patch named vectors + payload fields on an existing point."""
        ...

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    @abstractmethod
    async def search_dense(
        self, vector_name: str, vector: list[float], top_k: int = 10
    ) -> list[dict]:
        """Search a named dense vector. Returns list of {id, score, payload}.

        If the named vector doesn't exist on a point yet (e.g. a chunk
        whose enrichment hasn't run), that point is silently skipped by
        the underlying engine.
        """
        ...

    @abstractmethod
    async def search_sparse(self, sparse_vector: dict, top_k: int = 10) -> list[dict]:
        """BM25 sparse search. Returns list of {id, score, payload}."""
        ...

    @abstractmethod
    async def get_by_ids(self, ids: list[str]) -> dict[str, dict]:
        """Fetch payloads by point IDs. Returns {id: payload}."""
        ...

    @abstractmethod
    async def get_points_by_file(self, file_path: str) -> list[dict]:
        """Get all points for a file (used for stale called_by snapshot
        and for incremental hash-diffing during reindex)."""
        ...

    @abstractmethod
    async def get_points_by_name(
        self, name: str, class_name: str | None = None
    ) -> list[dict]:
        """Get all points whose chunk name matches `name` (and optional class).

        Used to resolve a callee key (``Class::method`` or bare ``method``)
        to its stored point so its ``called_by`` can be patched after a
        caller removes a call.
        """
        ...

    @abstractmethod
    async def get_vectors_by_id(self, point_id: str) -> dict[str, list[float]]:
        """Return the named dense vectors for a point. Empty dict if missing.

        Used during reindex to carry forward `description` and
        `developer_queries` vectors when a chunk's source hasn't changed.
        """
        ...

    @abstractmethod
    async def scroll_by_payload(
        self, filter: dict, limit: int = 10000
    ) -> list[dict]:
        """Paginated scroll of points matching exact-match payload filter.

        Returns list of {id, payload}. Used by the enricher to drain
        ``enrichment_state == pending`` and by ``semhood status`` to
        count chunks per state.
        """
        ...
