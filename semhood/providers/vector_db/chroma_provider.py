"""ChromaDB vector database provider — local, no server needed."""

from __future__ import annotations

import json
import logging

from semhood.providers.vector_db.base import VectorDBProvider

logger = logging.getLogger(__name__)


# Always present, regardless of indexing stage. ``code`` is filled at index
# time; ``description`` and ``developer_queries`` are filled at enrichment time.
_DENSE_VECTOR_NAMES = ("code", "description", "developer_queries")


class ChromaProvider(VectorDBProvider):
    """ChromaDB implementation — runs locally as a file or in-memory.

    No server, no Docker, no setup. Just ``pip install chromadb``.

    ChromaDB has no native named-vector support, so each named vector
    lives in its own sub-collection. The main collection holds payloads.
    """

    def __init__(self, path: str = "./chroma_data", collection: str = "semhood"):
        try:
            import chromadb
        except ImportError as exc:
            raise ImportError(
                "ChromaDB provider requires 'chromadb'. "
                "Install with: pip install chromadb"
            ) from exc

        self._collection_name = collection

        if path == ":memory:":
            self._client = chromadb.Client()
        else:
            self._client = chromadb.PersistentClient(path=path)

        self._collection = None
        self._collections: dict = {}

    # ------------------------------------------------------------------
    # Collection lifecycle
    # ------------------------------------------------------------------

    async def ensure_collection(self, vector_dim: int, recreate: bool = False) -> None:
        # Chroma infers dimension from the first add (no fixed dim at creation),
        # so a model change surfaces as an add-time error rather than here. On an
        # explicit reset, drop the collections so they rebind to the new model.
        if recreate:
            for vname in _DENSE_VECTOR_NAMES:
                try:
                    self._client.delete_collection(f"{self._collection_name}_{vname}")
                except Exception:
                    pass
            try:
                self._client.delete_collection(self._collection_name)
            except Exception:
                pass

        # One sub-collection per named dense vector.
        self._collections = {}
        for vname in _DENSE_VECTOR_NAMES:
            col_name = f"{self._collection_name}_{vname}"
            self._collections[vname] = self._client.get_or_create_collection(
                name=col_name,
                metadata={"hnsw:space": "cosine"},
            )

        # Main collection holds payloads.
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        logger.info(
            "ChromaDB ready — collections=%s",
            list(self._collections.keys()),
        )

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    async def upsert(self, points: list[dict]) -> None:
        if not points:
            return

        for p in points:
            point_id = p["id"]
            payload = p["payload"]

            self._collection.upsert(
                ids=[point_id],
                metadatas=[self._flatten_payload(payload)],
                documents=[payload.get("source", "")],
            )

            for vname, vec in p.get("vectors", {}).items():
                col = self._collections.get(vname)
                if col is None:
                    continue
                col.upsert(
                    ids=[point_id],
                    embeddings=[vec],
                    metadatas=[{"file": payload.get("file", "")}],
                )

    async def delete_by_file(self, file_path: str) -> None:
        for col in self._collections.values():
            try:
                results = col.get(where={"file": file_path})
                if results["ids"]:
                    col.delete(ids=results["ids"])
            except Exception:
                pass

        try:
            results = self._collection.get(where={"file": file_path})
            if results["ids"]:
                self._collection.delete(ids=results["ids"])
        except Exception:
            pass

    async def delete_by_ids(self, ids: list[str]) -> None:
        if not ids:
            return
        for col in self._collections.values():
            try:
                col.delete(ids=ids)
            except Exception:
                pass
        try:
            self._collection.delete(ids=ids)
        except Exception:
            pass

    async def set_payload_by_id(self, point_id: str, payload: dict) -> None:
        # Merge into existing metadata so we don't blow away unrelated fields.
        try:
            existing = self._collection.get(ids=[point_id], include=["metadatas"])
            current = self._unflatten_payload(
                existing["metadatas"][0] if existing.get("metadatas") else {}
            )
        except Exception:
            current = {}
        current.update(payload)
        flat = self._flatten_payload(current)
        try:
            self._collection.update(ids=[point_id], metadatas=[flat])
        except Exception as e:
            logger.warning("Failed to update payload for %s: %s", point_id, e)

    async def update_vectors_and_payload(
        self,
        point_id: str,
        vectors: dict[str, list[float]],
        payload: dict,
    ) -> None:
        for vname, vec in (vectors or {}).items():
            col = self._collections.get(vname)
            if col is None:
                continue
            try:
                col.upsert(ids=[point_id], embeddings=[vec])
            except Exception as e:
                logger.warning("Failed to update vector %s for %s: %s", vname, point_id, e)
        if payload:
            await self.set_payload_by_id(point_id, payload)

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    async def search_dense(
        self, vector_name: str, vector: list[float], top_k: int = 10
    ) -> list[dict]:
        col = self._collections.get(vector_name)
        if not col:
            return []

        count = col.count() or 0
        if count == 0:
            return []

        results = col.query(
            query_embeddings=[vector],
            n_results=min(top_k, count),
        )

        if not results["ids"] or not results["ids"][0]:
            return []

        ids = results["ids"][0]
        distances = (
            results["distances"][0] if results.get("distances") else [0.0] * len(ids)
        )

        payloads = await self.get_by_ids(ids)

        return [
            {
                "id": pid,
                "score": 1.0 - dist,
                "payload": payloads.get(pid, {}),
            }
            for pid, dist in zip(ids, distances, strict=False)
        ]

    async def search_sparse(self, sparse_vector: dict, top_k: int = 10) -> list[dict]:
        # ChromaDB doesn't support sparse vectors natively.
        return []

    async def get_by_ids(self, ids: list[str]) -> dict[str, dict]:
        if not ids:
            return {}

        try:
            results = self._collection.get(ids=ids, include=["metadatas", "documents"])
        except Exception:
            return {}

        out: dict[str, dict] = {}
        for i, pid in enumerate(results["ids"]):
            meta = results["metadatas"][i] if results["metadatas"] else {}
            payload = self._unflatten_payload(meta)
            if results.get("documents") and results["documents"][i]:
                payload["source"] = results["documents"][i]
            out[pid] = payload

        return out

    async def get_points_by_file(self, file_path: str) -> list[dict]:
        try:
            results = self._collection.get(
                where={"file": file_path},
                include=["metadatas", "documents"],
            )
        except Exception:
            return []

        points = []
        for i, pid in enumerate(results["ids"]):
            meta = results["metadatas"][i] if results["metadatas"] else {}
            payload = self._unflatten_payload(meta)
            if results.get("documents") and results["documents"][i]:
                payload["source"] = results["documents"][i]
            points.append({"id": pid, "payload": payload})

        return points

    async def get_points_by_name(
        self, name: str, class_name: str | None = None
    ) -> list[dict]:
        where: dict
        if class_name:
            where = {"$and": [{"name": name}, {"class_name": class_name}]}
        else:
            where = {"name": name}

        try:
            results = self._collection.get(
                where=where,
                include=["metadatas", "documents"],
            )
        except Exception:
            return []

        points = []
        for i, pid in enumerate(results["ids"]):
            meta = results["metadatas"][i] if results["metadatas"] else {}
            payload = self._unflatten_payload(meta)
            if results.get("documents") and results["documents"][i]:
                payload["source"] = results["documents"][i]
            points.append({"id": pid, "payload": payload})

        return points

    async def get_vectors_by_id(self, point_id: str) -> dict[str, list[float]]:
        out: dict[str, list[float]] = {}
        for vname, col in self._collections.items():
            try:
                res = col.get(ids=[point_id], include=["embeddings"])
            except Exception:
                continue
            embs = res.get("embeddings") or []
            if embs and embs[0] is not None:
                out[vname] = list(embs[0])
        return out

    async def scroll_by_payload(
        self, filter: dict, limit: int = 10000
    ) -> list[dict]:
        # ChromaDB's `where` accepts a dict directly for single-field eq, or
        # `$and` for multiple fields.
        if not filter:
            where = None
        elif len(filter) == 1:
            where = dict(filter)
        else:
            where = {"$and": [{k: v} for k, v in filter.items()]}

        try:
            results = self._collection.get(
                where=where,
                limit=limit,
                include=["metadatas", "documents"],
            )
        except Exception:
            return []

        out: list[dict] = []
        for i, pid in enumerate(results.get("ids", [])):
            meta = results["metadatas"][i] if results.get("metadatas") else {}
            payload = self._unflatten_payload(meta)
            if results.get("documents") and results["documents"][i]:
                payload["source"] = results["documents"][i]
            out.append({"id": pid, "payload": payload})
        return out

    # ------------------------------------------------------------------
    # Helpers — ChromaDB metadata only supports str/int/float/bool
    # ------------------------------------------------------------------

    def _flatten_payload(self, payload: dict) -> dict:
        flat: dict = {}
        for key, value in payload.items():
            if isinstance(value, (str, int, float, bool)):
                flat[key] = value
            elif isinstance(value, list):
                flat[key] = json.dumps(value)
            elif value is None:
                flat[key] = ""
            else:
                flat[key] = json.dumps(value)
        return flat

    def _unflatten_payload(self, flat: dict) -> dict:
        payload: dict = {}
        for key, value in flat.items():
            if isinstance(value, str) and value.startswith(("[", "{")):
                try:
                    payload[key] = json.loads(value)
                except json.JSONDecodeError:
                    payload[key] = value
            elif value == "":
                payload[key] = None
            else:
                payload[key] = value
        return payload
