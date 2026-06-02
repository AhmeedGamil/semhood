"""EnrichmentPolicy + ParsingPolicy tests."""

from __future__ import annotations

from pathlib import Path

from semhood.core.models import Chunk, ChunkType
from semhood.core.policy import EnrichmentPolicy, ParsingPolicy


def make_chunk(file="x.py", source="return 1", class_name=None, name="foo", docblock=None):
    return Chunk(
        name=name,
        type=ChunkType.METHOD if class_name else ChunkType.FUNCTION,
        file=file,
        language="python",
        line_start=1,
        line_end=2,
        source=source,
        class_name=class_name,
        docblock=docblock,
    )


# ----- EnrichmentPolicy --------------------------------------------------


def test_skip_by_pattern_matches_tests_dir():
    p = EnrichmentPolicy(
        skip_patterns=["**/tests/**"], skip_trivial_enabled=False,
        max_trivial_lines=3, require_control_flow=True,
    )
    assert p.should_skip_by_pattern(make_chunk(file="src/tests/test_x.py"))
    assert not p.should_skip_by_pattern(make_chunk(file="src/main.py"))


def test_skip_by_pattern_matches_test_prefix():
    p = EnrichmentPolicy(
        skip_patterns=["**/test_*.py"], skip_trivial_enabled=False,
        max_trivial_lines=3, require_control_flow=True,
    )
    assert p.should_skip_by_pattern(make_chunk(file="src/test_orders.py"))
    assert not p.should_skip_by_pattern(make_chunk(file="src/orders.py"))


def test_is_trivial_short_chunk_no_control_flow():
    p = EnrichmentPolicy(
        skip_patterns=[], skip_trivial_enabled=True,
        max_trivial_lines=3, require_control_flow=True,
    )
    assert p.is_trivial(make_chunk(source="return 1"))


def test_is_trivial_rejects_chunk_with_if():
    p = EnrichmentPolicy(
        skip_patterns=[], skip_trivial_enabled=True,
        max_trivial_lines=3, require_control_flow=True,
    )
    assert not p.is_trivial(make_chunk(source="if x: return 1"))


def test_is_trivial_rejects_long_chunk():
    p = EnrichmentPolicy(
        skip_patterns=[], skip_trivial_enabled=True,
        max_trivial_lines=3, require_control_flow=True,
    )
    long_src = "\n".join(f"x{i} = {i}" for i in range(10))
    assert not p.is_trivial(make_chunk(source=long_src))


def test_is_trivial_disabled_when_flag_off():
    p = EnrichmentPolicy(
        skip_patterns=[], skip_trivial_enabled=False,
        max_trivial_lines=3, require_control_flow=True,
    )
    assert not p.is_trivial(make_chunk(source="return 1"))


def test_deterministic_summary_uses_class_name_when_present():
    p = EnrichmentPolicy(
        skip_patterns=[], skip_trivial_enabled=True,
        max_trivial_lines=3, require_control_flow=True,
    )
    s, queries = p.deterministic_summary(make_chunk(class_name="Order", name="total"))
    assert "Method total in Order" in s
    assert "Order total" in queries


# ----- ParsingPolicy -----------------------------------------------------


def test_parsing_policy_skips_excluded_dir(tmp_path: Path):
    pol = ParsingPolicy(
        exclude_dirs=["node_modules"], exclude_files=[],
        max_file_size_mb=1.0, max_chunks_per_file=500,
    )
    nm_dir = tmp_path / "node_modules"
    nm_dir.mkdir()
    f = nm_dir / "x.js"
    f.write_text("// hi")
    assert pol.should_skip_path(f)


def test_parsing_policy_skips_glob_match(tmp_path: Path):
    pol = ParsingPolicy(
        exclude_dirs=[], exclude_files=["*.min.js"],
        max_file_size_mb=1.0, max_chunks_per_file=500,
    )
    f = tmp_path / "bundle.min.js"
    f.write_text("// hi")
    assert pol.should_skip_path(f)


def test_parsing_policy_skips_oversized_file(tmp_path: Path):
    pol = ParsingPolicy(
        exclude_dirs=[], exclude_files=[],
        max_file_size_mb=0.001,  # ~1 KB
        max_chunks_per_file=500,
    )
    f = tmp_path / "big.py"
    f.write_text("x" * 5000)
    assert pol.should_skip_path(f)


def test_parsing_policy_keeps_normal_file(tmp_path: Path):
    pol = ParsingPolicy(
        exclude_dirs=["node_modules"], exclude_files=["*.min.js"],
        max_file_size_mb=1.0, max_chunks_per_file=500,
    )
    f = tmp_path / "main.py"
    f.write_text("def foo(): pass")
    assert not pol.should_skip_path(f)
