"""StructuralBuilder + build_description tests."""

from __future__ import annotations

import hashlib

from semhood.core.models import Chunk, ChunkType, Parameter
from semhood.core.structural import StructuralBuilder, build_description


def _chunk(**kw) -> Chunk:
    defaults = dict(
        name="foo", type=ChunkType.FUNCTION, file="x.py", language="python",
        line_start=1, line_end=5, source="def foo():\n    return 1",
    )
    defaults.update(kw)
    return Chunk(**defaults)


def test_source_hash_is_sha256_hex():
    sb = StructuralBuilder()
    c = _chunk(source="hello world")
    assert sb.source_hash(c) == hashlib.sha256(b"hello world").hexdigest()


def test_source_hash_changes_with_source():
    sb = StructuralBuilder()
    h1 = sb.source_hash(_chunk(source="a"))
    h2 = sb.source_hash(_chunk(source="b"))
    assert h1 != h2


def test_build_sparse_returns_indices_and_values():
    sb = StructuralBuilder()
    sparse = sb.build_sparse(_chunk(name="charge", class_name="PaymentProcessor"))
    assert "indices" in sparse and "values" in sparse
    assert len(sparse["indices"]) == len(sparse["values"]) > 0


def test_build_sparse_empty_for_blank_chunk():
    sb = StructuralBuilder()
    # Note: file= still injects tokens, so we rig a chunk with no tokens at all.
    c = _chunk(name="", file="", source="")
    sparse = sb.build_sparse(c)
    assert sparse["indices"] == []


def test_build_description_for_method():
    c = _chunk(
        name="charge", class_name="PaymentProcessor", type=ChunkType.METHOD,
        docblock="Authorize and capture.",
        parameters=[Parameter(name="amount", type="int")],
        return_type="PaymentResult",
        calls=["self._gateway.authorize"],
    )
    desc = build_description(c)
    assert "Method charge in class PaymentProcessor" in desc
    assert "Authorize and capture." in desc
    assert "amount: int" in desc
    assert "Returns: PaymentResult" in desc
    assert "self._gateway.authorize" in desc


def test_build_description_for_function_no_docblock():
    c = _chunk(name="parse", docblock=None)
    desc = build_description(c)
    assert "Function parse" in desc
    assert "Method " not in desc
