"""ensure_collection guardrail — a changed embedding dimension must fail clearly,
and `--reset` (recreate=True) must rebuild instead of erroring."""

from __future__ import annotations

import pytest

from semhood.providers.vector_db.qdrant_provider import QdrantProvider


def _provider() -> QdrantProvider:
    # In-memory Qdrant: no url, no path -> ":memory:".
    return QdrantProvider(collection="guardrail_test")


async def test_same_dimension_is_idempotent():
    p = _provider()
    await p.ensure_collection(vector_dim=8)
    await p.ensure_collection(vector_dim=8)  # must not raise


async def test_dimension_change_raises_clear_error():
    p = _provider()
    await p.ensure_collection(vector_dim=8)

    with pytest.raises(ValueError) as exc:
        await p.ensure_collection(vector_dim=16)

    msg = str(exc.value)
    assert "dimension mismatch" in msg.lower()
    assert "--reset" in msg  # tells the user exactly how to recover


async def test_recreate_rebuilds_at_new_dimension():
    p = _provider()
    await p.ensure_collection(vector_dim=8)

    # Reset rebuilds rather than erroring...
    await p.ensure_collection(vector_dim=16, recreate=True)
    # ...and the collection is now compatible at the new dimension.
    await p.ensure_collection(vector_dim=16)
