"""
Cross-file call graph builder.

Language-agnostic: operates on Chunk objects regardless of source language.
Each language extractor normalizes calls via its normalize_call() method
before chunks reach this stage.
"""

from __future__ import annotations

import logging

from semhood.core.models import CallerRef, Chunk

logger = logging.getLogger(__name__)


class CallGraphBuilder:
    """
    Builds the called_by inverted map across all chunks.

    The AST extractor gives each chunk a `calls` list.
    This class does the cross-file pass: it collects every call seen in every
    chunk, then inverts the map so each function knows who calls it.
    """

    def attach_called_by(self, chunks: list[Chunk]) -> None:
        """
        Given all chunks from all files, attach called_by to each chunk (in-place).

        Algorithm:
          1. Build forward map: callee_key → list of callers
          2. For each chunk, look up its key in the forward map → those are its callers
        """
        # Step 1: Build forward map
        forward_map: dict[str, list[CallerRef]] = {}

        for chunk in chunks:
            caller_ref = CallerRef(
                function=chunk.name,
                class_name=chunk.class_name,
                file=chunk.file,
                line=chunk.line_start,
            )

            for call in chunk.calls:
                normalized = self._normalize(call)
                if normalized not in forward_map:
                    forward_map[normalized] = []
                forward_map[normalized].append(caller_ref)

        # Step 2: Invert — for each chunk, find who calls it
        for chunk in chunks:
            chunk.called_by = []

            # Match fully-qualified key: "Class::method"
            if chunk.class_name:
                fq_key = f"{chunk.class_name}::{chunk.name}"
                if fq_key in forward_map:
                    chunk.called_by.extend(forward_map[fq_key])

            # Also match bare name (handles unresolved instance calls)
            bare_key = chunk.name
            if bare_key in forward_map:
                for caller in forward_map[bare_key]:
                    if not self._caller_present(chunk.called_by, caller):
                        chunk.called_by.append(caller)

            # Deduplicate
            chunk.called_by = self._deduplicate(chunk.called_by)

    def _normalize(self, call: str) -> str:
        """Normalize call target to a lookup key."""
        # Static/module calls: Class::method, Module.func → keep as-is
        if "::" in call:
            return call

        # Strip self/this/cls patterns → bare method name
        for prefix in ("self.", "cls.", "this.", "this->", "super.", "$this->"):
            if call.startswith(prefix):
                return call[len(prefix):]

        # Instance calls: obj.method, var->method → bare method
        if "." in call:
            parts = call.split(".", 1)
            if parts[0] and parts[0][0].isupper():
                return f"{parts[0]}::{parts[1]}"
            return parts[1]

        if "->" in call:
            return call.split("->", 1)[1]

        return call

    def _caller_present(self, callers: list[CallerRef], candidate: CallerRef) -> bool:
        """Check if a caller is already in the list."""
        return any(
            c.function == candidate.function
            and c.file == candidate.file
            and c.line == candidate.line
            for c in callers
        )

    def _deduplicate(self, callers: list[CallerRef]) -> list[CallerRef]:
        """Deduplicate callers by (function, file, line)."""
        seen: set[str] = set()
        result: list[CallerRef] = []
        for c in callers:
            key = f"{c.file}:{c.line}:{c.function}"
            if key not in seen:
                seen.add(key)
                result.append(c)
        return result
