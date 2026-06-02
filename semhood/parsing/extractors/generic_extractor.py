"""Generic fallback extractor for any Tree-sitter supported language."""

from __future__ import annotations

from tree_sitter import Node, Tree

from semhood.core.models import Chunk, ChunkType
from semhood.parsing.extractors.base import LanguageExtractor


class GenericExtractor(LanguageExtractor):
    """
    Fallback extractor for languages without a dedicated extractor.

    Uses common Tree-sitter node types that most grammars share:
      - function_definition / function_declaration
      - method_definition / method_declaration
      - class_definition / class_declaration

    Produces basic chunks with name, source, line numbers, and calls.
    """

    def __init__(self, language: str):
        self._language = language

    @property
    def language_name(self) -> str:
        return self._language

    # Common function/method node types across grammars
    _FUNCTION_TYPES = {
        "function_definition", "function_declaration",  # incl. Go/JS
        "method_definition", "method_declaration",
        "function_item",  # Rust
    }

    _CLASS_TYPES = {
        "class_definition", "class_declaration",
        "class_specifier",  # C++
        "struct_item",  # Rust
        "type_declaration",  # Go
    }

    def extract(self, tree: Tree, file_path: str, source: bytes) -> list[Chunk]:
        chunks: list[Chunk] = []
        self._walk(tree.root_node, file_path, source, chunks, None)
        return chunks

    def normalize_call(self, call: str) -> str:
        # Generic normalization: strip common self/this prefixes
        for prefix in ("self.", "this.", "this->", "$this->"):
            if call.startswith(prefix):
                return call[len(prefix):]
        # Static calls: keep as-is if contains :: or .
        if "::" in call:
            return call
        if "." in call:
            parts = call.split(".", 1)
            if parts[0] and parts[0][0].isupper():
                return f"{parts[0]}::{parts[1]}"
            return parts[1]
        return call

    def _walk(
        self, node: Node, file_path: str, source: bytes,
        chunks: list[Chunk], class_name: str | None
    ) -> None:
        """Recursively walk the tree looking for function/class nodes."""
        if node.type in self._CLASS_TYPES:
            name_node = self._find_child_by_field(node, "name")
            name = self._node_text(name_node, source) if name_node else "Unknown"

            class_chunk = Chunk(
                name=name,
                type=ChunkType.CLASS,
                file=file_path,
                language=self._language,
                line_start=node.start_point[0] + 1,
                line_end=node.end_point[0] + 1,
                source=self._node_text(node, source),
                calls=[],
            )
            chunks.append(class_chunk)

            # Walk children with class context, track methods
            before = len(chunks)
            for child in node.children:
                self._walk(child, file_path, source, chunks, name)

            # Link class → methods
            class_chunk.methods = [
                c.name for c in chunks[before:] if c.class_name == name
            ]
            return

        if node.type in self._FUNCTION_TYPES:
            name_node = self._find_child_by_field(node, "name")
            name = self._node_text(name_node, source) if name_node else "anonymous"

            calls = self._extract_calls(node, source)

            chunks.append(Chunk(
                name=name,
                type=ChunkType.METHOD if class_name else ChunkType.FUNCTION,
                class_name=class_name,
                file=file_path,
                language=self._language,
                line_start=node.start_point[0] + 1,
                line_end=node.end_point[0] + 1,
                source=self._node_text(node, source),
                calls=list(set(calls)),
            ))
            return  # Don't recurse into nested functions

        # Keep walking
        for child in node.children:
            self._walk(child, file_path, source, chunks, class_name)

    def _extract_calls(self, node: Node, source: bytes) -> list[str]:
        """Generic call extraction — looks for call_expression nodes."""
        calls: list[str] = []
        self._walk_calls(node, source, calls)
        return calls

    def _walk_calls(self, node: Node, source: bytes, calls: list[str]) -> None:
        if node.type in ("call_expression", "call"):
            func = self._find_child_by_field(node, "function")
            if func:
                text = self._node_text(func, source)
                if text and not text.startswith("("):
                    calls.append(text)

        for child in node.children:
            self._walk_calls(child, source, calls)
