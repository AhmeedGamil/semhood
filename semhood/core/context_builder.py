"""
Context builder — formats retrieved chunks into an LLM prompt.

Produces a structured prompt with primary chunks (retrieved) and
neighbor chunks (from graph expansion), formatted appropriately
for the query being answered.
"""

from __future__ import annotations

_MAX_SOURCE_LINES = 50


class ContextBuilder:
    """Formats retrieved chunks into a structured LLM prompt."""

    def build(self, chunks: list[dict], with_source: bool = True) -> str:
        """
        Build context string from retrieved chunks.

        Args:
            chunks: List of {id, payload, is_neighbor?} dicts.
            with_source: Whether to include source code in the context.

        Returns:
            Formatted context string for the LLM prompt.
        """
        if not chunks:
            return ""

        primary = [c for c in chunks if not c.get("is_neighbor")]
        neighbors = [c for c in chunks if c.get("is_neighbor")]

        parts: list[str] = []

        # Primary chunks
        for i, chunk in enumerate(primary, 1):
            payload = chunk.get("payload", {})
            parts.append(self._format_chunk(payload, i, with_source))

        # Neighbor chunks (supporting context)
        if neighbors:
            parts.append("\n--- Supporting Context (Call Graph Neighbors) ---\n")
            for i, chunk in enumerate(neighbors, 1):
                payload = chunk.get("payload", {})
                parts.append(self._format_chunk(payload, i, with_source=False))

        return "\n".join(parts)

    def _format_chunk(self, payload: dict, index: int, with_source: bool) -> str:
        """Format a single chunk for the LLM prompt."""
        lines: list[str] = []

        # Header
        name = payload.get("name", "Unknown")
        class_name = payload.get("class_name")
        language = payload.get("language", "")

        if class_name:
            lines.append(f"[{index}] {class_name}.{name} ({language})")
        else:
            lines.append(f"[{index}] {name} ({language})")

        # File location
        file = payload.get("file", "")
        line_start = payload.get("line_start")
        line_end = payload.get("line_end")
        if file:
            loc = f"  File: {file}"
            if line_start:
                loc += f" (lines {line_start}-{line_end})"
            lines.append(loc)

        # Logic summary
        summary = payload.get("logic_summary")
        if summary:
            lines.append(f"  Summary: {summary}")

        # Docblock (if no logic summary)
        elif payload.get("docblock"):
            lines.append(f"  Docblock: {payload['docblock']}")

        # Signature
        params = payload.get("parameters", [])
        if params:
            param_str = ", ".join(
                f"{p['name']}: {p['type']}" if p.get("type") else p["name"]
                for p in params
            )
            lines.append(f"  Parameters: ({param_str})")

        ret = payload.get("return_type")
        if ret:
            lines.append(f"  Returns: {ret}")

        # Call relationships
        calls = payload.get("calls", [])
        if calls:
            lines.append(f"  Calls: {', '.join(calls[:8])}")

        called_by = payload.get("called_by", [])
        if called_by:
            callers = [
                f"{c.get('class_name', '')}.{c['function']}" if c.get("class_name")
                else c["function"]
                for c in called_by[:5]
            ]
            lines.append(f"  Called by: {', '.join(callers)}")

        # Route
        route = payload.get("route")
        if route:
            method = payload.get("http_method", "")
            lines.append(f"  Route: {method} {route}")

        # Source code
        if with_source:
            source = payload.get("source", "")
            if source:
                source_lines = source.split("\n")
                if len(source_lines) > _MAX_SOURCE_LINES:
                    source = "\n".join(source_lines[:_MAX_SOURCE_LINES])
                    source += f"\n  ... ({len(source_lines) - _MAX_SOURCE_LINES} more lines)"
                lines.append(f"  Source:\n```{payload.get('language', '')}\n{source}\n```")

        lines.append("")  # Blank line between chunks
        return "\n".join(lines)
