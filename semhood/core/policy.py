"""Stateless enrichment + parsing policies.

Decides:
  - whether a file should be parsed at all (ParsingPolicy)
  - whether a chunk should be enriched, skipped as trivial, or skipped by
    pattern (EnrichmentPolicy)

Both are pure data + simple checks — no I/O, no async. Easy to unit test.
"""

from __future__ import annotations

import re
from fnmatch import fnmatch
from pathlib import Path

from semhood.core.models import Chunk

_CONTROL_FLOW = re.compile(r"\b(if|for|while|try|switch|case|catch|match)\b")


class EnrichmentPolicy:
    """Decides per-chunk whether to enrich, template, or skip outright."""

    def __init__(
        self,
        skip_patterns: list[str],
        skip_trivial_enabled: bool,
        max_trivial_lines: int,
        require_control_flow: bool,
    ):
        self._skip_patterns = list(skip_patterns or [])
        self._skip_trivial = skip_trivial_enabled
        self._max_lines = max_trivial_lines
        self._require_control_flow = require_control_flow

    def should_skip_by_pattern(self, chunk: Chunk) -> bool:
        """True if the chunk's file matches a skip pattern (e.g. tests)."""
        if not self._skip_patterns:
            return False
        # fnmatch's '**' is not as expressive as glob, but matching against the
        # full path string covers the common '**/tests/**' case if we also try
        # the bare filename.
        path = chunk.file or ""
        name = Path(path).name if path else ""
        for pat in self._skip_patterns:
            if fnmatch(path, pat) or fnmatch(name, pat):
                return True
            # Path-segment fallback: treat '**/tests/**' as "tests segment in path".
            if "**" in pat:
                middle = pat.strip("*/").strip("*")
                if middle and middle in path.replace("\\", "/"):
                    return True
        return False

    def is_trivial(self, chunk: Chunk) -> bool:
        """True if the chunk is short enough + simple enough to skip the LLM."""
        if not self._skip_trivial:
            return False
        src = (chunk.source or "").strip()
        if not src:
            return True
        line_count = len([ln for ln in src.splitlines() if ln.strip()])
        if line_count > self._max_lines:
            return False
        if self._require_control_flow and _CONTROL_FLOW.search(src):
            return False
        return True

    def deterministic_summary(self, chunk: Chunk) -> tuple[str, list[str]]:
        """Cheap template summary for trivial chunks (no LLM call)."""
        kind = "Method" if chunk.class_name else "Function"
        owner = f" in {chunk.class_name}" if chunk.class_name else ""
        summary = f"{kind} {chunk.name}{owner}. {chunk.docblock or 'Trivial implementation.'}"
        queries = [chunk.name, chunk.qualified_name()]
        if chunk.class_name:
            queries.append(f"{chunk.class_name} {chunk.name}")
        return summary, queries


class ParsingPolicy:
    """Decides per-file whether to parse it at all."""

    def __init__(
        self,
        exclude_dirs: list[str],
        exclude_files: list[str],
        max_file_size_mb: float,
        max_chunks_per_file: int,
    ):
        self.exclude_dirs = set(exclude_dirs or [])
        self.exclude_files = list(exclude_files or [])
        self.max_file_size = int(max_file_size_mb * 1024 * 1024)
        self.max_chunks_per_file = max_chunks_per_file

    def should_skip_path(self, path: Path) -> bool:
        if any(part in self.exclude_dirs for part in path.parts):
            return True
        if any(fnmatch(path.name, pat) for pat in self.exclude_files):
            return True
        try:
            if path.stat().st_size > self.max_file_size:
                return True
        except OSError:
            return True
        return False
