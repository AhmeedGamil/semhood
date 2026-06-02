"""Chunk model + payload roundtrip tests."""

from __future__ import annotations

from semhood.core.enrichment_state import (
    ALL_STATES,
    FAILED,
    FRESH,
    PENDING,
    SKIPPED_TRIVIAL,
)
from semhood.core.models import Chunk, ChunkType


def test_chunk_defaults_have_pending_enrichment_state():
    c = Chunk(name="foo", type=ChunkType.FUNCTION, file="x.py", language="python",
              line_start=1, line_end=2)
    assert c.enrichment_state == PENDING
    assert c.enrichment_version == 0
    assert c.source_hash is None
    assert c.enrichment_attempts == 0
    assert c.enrichment_error is None


def test_to_payload_excludes_vectors():
    c = Chunk(name="foo", type=ChunkType.FUNCTION, file="x.py", language="python",
              line_start=1, line_end=2, source="def foo(): pass")
    c.vectors = {"code": [0.1] * 4}
    c.sparse = {"indices": [1], "values": [1.0]}
    payload = c.to_payload()
    assert "vectors" not in payload
    assert "sparse" not in payload
    assert payload["name"] == "foo"
    assert payload["enrichment_state"] == PENDING


def test_qualified_name_with_class():
    c = Chunk(name="charge", type=ChunkType.METHOD, file="p.py", language="python",
              line_start=1, line_end=2, class_name="PaymentProcessor")
    assert c.qualified_name() == "PaymentProcessor::charge"


def test_qualified_name_without_class():
    c = Chunk(name="helper", type=ChunkType.FUNCTION, file="u.py", language="python",
              line_start=1, line_end=2)
    assert c.qualified_name() == "helper"


def test_all_states_constant_covers_every_state():
    # Defensive — ensure no one adds a new state without updating ALL_STATES.
    assert set(ALL_STATES) == {PENDING, FRESH, FAILED, SKIPPED_TRIVIAL}


def test_strategy_enum_was_removed():
    # Refactor invariant — Strategy enum must not creep back in.
    import semhood.core.models as m
    assert not hasattr(m, "Strategy")
