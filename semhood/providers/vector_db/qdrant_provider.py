"""Qdrant vector database provider."""

from __future__ import annotations

import logging
from typing import Any

from qdrant_client import AsyncQdrantClient, models

from semhood.providers.vector_db.base import VectorDBProvider

logger = logging.getLogger(__name__)


# All collections always carry these three named dense vectors plus
# the bm25 sparse vector. ``code`` is populated at index time;
# ``description`` and ``developer_queries`` are populated at enrichment time.
_DENSE_VECTOR_NAMES = ("code", "description", "developer_queries")


class QdrantProvider(VectorDBProvider):
    """Qdrant implementation — supports remote server, local on-disk, or in-memory."""

    def __init__(
        self,
        url: str | None = None,
        path: str | None = None,
        collection: str = "semhood",
    ):
        """
        Args:
            url: Remote Qdrant server URL (e.g. http://localhost:6333).
            path: Local on-disk storage path. Use ":memory:" for in-memory.
            collection: Collection name.

        Priority: if url is set, connect to remote. Otherwise use path.
        If neither is set, defaults to in-memory.
        """
        self._collection = collection

        if url:
            self._client = AsyncQdrantClient(url=url)
        elif path and path != ":memory:":
            self._client = AsyncQdrantClient(path=path)
        else:
            self._client = AsyncQdrantClient(location=":memory:")

    # ------------------------------------------------------------------
    # Collection lifecycle
    # ------------------------------------------------------------------

    @staticmethod
    def _existing_dim(info: Any) -> int | None:
        """Best-effort read of an existing collection's vector dimension."""
        try:
            vectors = info.config.params.vectors
        except AttributeError:
            return None
        if isinstance(vectors, dict):  # named vectors
            for vp in vectors.values():
                size = getattr(vp, "size", None)
                if size:
                    return int(size)
            return None
        return getattr(vectors, "size", None)

    async def ensure_collection(self, vector_dim: int, recreate: bool = False) -> None:
        """Create collection with three named dense vectors + bm25 sparse.

        If the collection already exists, verify its vector dimension matches the
        current embedding model. A mismatch (the embedding model was changed in
        config) raises a clear, actionable error instead of failing later with a
        cryptic upsert error. Pass ``recreate=True`` to drop and rebuild — used by
        ``index --reset`` after an intentional model change (enrichment is
        restored from ``.semhood/enrichment.jsonl``, no LLM tokens).
        """
        try:
            info = await self._client.get_collection(self._collection)
        except Exception:
            info = None

        if info is not None:
            if not recreate:
                existing = self._existing_dim(info)
                if existing is not None and existing != vector_dim:
                    raise ValueError(
                        f"Embedding dimension mismatch for collection "
                        f"'{self._collection}': the index was built at {existing}-dim "
                        f"but the configured embedding model produces {vector_dim}-dim "
                        f"vectors. The embedding model was likely changed. Rebuild with "
                        f"`semhood index --reset` — your enrichment is preserved in "
                        f".semhood/enrichment.jsonl and will be re-embedded (no LLM tokens)."
                    )
                return  # exists and compatible
            await self._client.delete_collection(self._collection)

        vectors_config = {
            name: models.VectorParams(
                size=vector_dim, distance=models.Distance.COSINE
            )
            for name in _DENSE_VECTOR_NAMES
        }

        await self._client.create_collection(
            collection_name=self._collection,
            vectors_config=vectors_config,
            sparse_vectors_config={
                "bm25": models.SparseVectorParams(
                    modifier=models.Modifier.IDF,
                ),
            },
        )

        # Payload indexes for fast filtering during indexing + enrichment.
        for field in ("file", "name", "class_name", "enrichment_state"):
            await self._client.create_payload_index(
                collection_name=self._collection,
                field_name=field,
                field_schema=models.PayloadSchemaType.KEYWORD,
            )

        logger.info(
            "Created Qdrant collection '%s' (dim=%d, vectors=%s)",
            self._collection, vector_dim, list(_DENSE_VECTOR_NAMES),
        )

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    async def upsert(self, points: list[dict]) -> None:
        if not points:
            return

        qdrant_points = []
        for p in points:
            vector_data: dict[str, Any] = {}

            # Dense vectors (only the names actually provided)
            for name, vec in p.get("vectors", {}).items():
                vector_data[name] = vec

            # Sparse vector (BM25)
            sparse = p.get("sparse", {})
            if sparse.get("indices"):
                vector_data["bm25"] = models.SparseVector(
                    indices=sparse["indices"],
                    values=sparse["values"],
                )

            qdrant_points.append(
                models.PointStruct(
                    id=p["id"],
                    vector=vector_data,
                    payload=p["payload"],
                )
            )

        await self._client.upsert(
            collection_name=self._collection,
            points=qdrant_points,
        )

    async def delete_by_file(self, file_path: str) -> None:
        await self._client.delete(
            collection_name=self._collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="file",
                            match=models.MatchValue(value=file_path),
                        )
                    ]
                )
            ),
        )

    async def delete_by_ids(self, ids: list[str]) -> None:
        if not ids:
            return
        await self._client.delete(
            collection_name=self._collection,
            points_selector=models.PointIdsList(points=ids),
        )

    async def set_payload_by_id(self, point_id: str, payload: dict) -> None:
        await self._client.set_payload(
            collection_name=self._collection,
            payload=payload,
            points=[point_id],
        )

    async def update_vectors_and_payload(
        self,
        point_id: str,
        vectors: dict[str, list[float]],
        payload: dict,
    ) -> None:
        if vectors:
            await self._client.update_vectors(
                collection_name=self._collection,
                points=[models.PointVectors(id=point_id, vector=vectors)],
            )
        if payload:
            await self._client.set_payload(
                collection_name=self._collection,
                payload=payload,
                points=[point_id],
            )

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    async def search_dense(
        self, vector_name: str, vector: list[float], top_k: int = 10
    ) -> list[dict]:
        results = await self._client.query_points(
            collection_name=self._collection,
            query=vector,
            using=vector_name,
            limit=top_k,
            with_payload=True,
        )

        return [
            {"id": str(p.id), "score": p.score, "payload": p.payload or {}}
            for p in results.points
        ]

    async def search_sparse(self, sparse_vector: dict, top_k: int = 10) -> list[dict]:
        if not sparse_vector.get("indices"):
            return []

        results = await self._client.query_points(
            collection_name=self._collection,
            query=models.SparseVector(
                indices=sparse_vector["indices"],
                values=sparse_vector["values"],
            ),
            using="bm25",
            limit=top_k,
            with_payload=True,
        )

        return [
            {"id": str(p.id), "score": p.score, "payload": p.payload or {}}
            for p in results.points
        ]

    async def get_by_ids(self, ids: list[str]) -> dict[str, dict]:
        if not ids:
            return {}

        results = await self._client.retrieve(
            collection_name=self._collection,
            ids=ids,
            with_payload=True,
            with_vectors=False,
        )

        return {str(p.id): p.payload or {} for p in results}

    async def get_points_by_file(self, file_path: str) -> list[dict]:
        results = await self._client.scroll(
            collection_name=self._collection,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="file",
                        match=models.MatchValue(value=file_path),
                    )
                ]
            ),
            limit=500,
            with_payload=True,
            with_vectors=False,
        )

        points, _ = results
        return [
            {"id": str(p.id), "payload": p.payload or {}}
            for p in points
        ]

    async def get_points_by_name(
        self, name: str, class_name: str | None = None
    ) -> list[dict]:
        must: list[Any] = [
            models.FieldCondition(key="name", match=models.MatchValue(value=name))
        ]
        if class_name:
            must.append(
                models.FieldCondition(
                    key="class_name", match=models.MatchValue(value=class_name)
                )
            )

        results = await self._client.scroll(
            collection_name=self._collection,
            scroll_filter=models.Filter(must=must),
            limit=200,
            with_payload=True,
            with_vectors=False,
        )

        points, _ = results
        return [
            {"id": str(p.id), "payload": p.payload or {}}
            for p in points
        ]

    async def get_vectors_by_id(self, point_id: str) -> dict[str, list[float]]:
        res = await self._client.retrieve(
            collection_name=self._collection,
            ids=[point_id],
            with_vectors=True,
        )
        if not res:
            return {}
        vec = res[0].vector or {}
        # When a collection has named vectors, qdrant returns dict[str, list[float]].
        return vec if isinstance(vec, dict) else {}

    async def scroll_by_payload(
        self, filter: dict, limit: int = 10000
    ) -> list[dict]:
        must = [
            models.FieldCondition(key=k, match=models.MatchValue(value=v))
            for k, v in (filter or {}).items()
        ]

        out: list[dict] = []
        next_offset = None
        while True:
            page_size = max(1, min(500, limit - len(out)))
            points, next_offset = await self._client.scroll(
                collection_name=self._collection,
                scroll_filter=models.Filter(must=must) if must else None,
                limit=page_size,
                offset=next_offset,
                with_payload=True,
                with_vectors=False,
            )
            out.extend(
                {"id": str(p.id), "payload": p.payload or {}}
                for p in points
            )
            if not next_offset or len(out) >= limit:
                break

        return out
