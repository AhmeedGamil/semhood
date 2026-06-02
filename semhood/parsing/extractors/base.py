"""Abstract base class for language-specific chunk extractors."""

from __future__ import annotations

from abc import ABC, abstractmethod

from tree_sitter import Tree

from semhood.core.models import Chunk


class LanguageExtractor(ABC):
    """
    Extracts structured chunks from a Tree-sitter syntax tree.

    Each language implements this interface to extract functions, methods,
    and classes from its AST into Chunk objects.
    """

    @property
    @abstractmethod
    def language_name(self) -> str:
        """Language identifier (e.g. 'python', 'javascript')."""
        ...

    @abstractmethod
    def extract(self, tree: Tree, file_path: str, source: bytes) -> list[Chunk]:
        """
        Walk the syntax tree and extract function/class/method chunks.

        Args:
            tree: Parsed Tree-sitter syntax tree.
            file_path: Relative path of the source file.
            source: Raw source code as bytes.

        Returns:
            List of Chunk objects — one per function/method/class.
        """
        ...

    @abstractmethod
    def normalize_call(self, call: str) -> str:
        """
        Normalize a call target for call graph building.

        Each language has different call syntax:
          Python:  self.method() → "method", Module.func() → "Module::func"
          JS:      this.method() → "method", Class.func() → "Class::func"
          Java:    this.method() → "method", Class.func() → "Class::func"
          PHP:     $this->method() → "method", Class::method() → "Class::method"

        Returns a normalized key for the CallGraphBuilder.
        """
        ...

    # -------------------------------------------------------------------------
    # Shared helpers
    # -------------------------------------------------------------------------

    def _node_text(self, node, source: bytes) -> str:
        """Extract text content from a tree-sitter node."""
        return source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")

    def _find_children_by_type(self, node, type_name: str) -> list:
        """Find all direct children of a specific node type."""
        return [child for child in node.children if child.type == type_name]

    def _find_child_by_type(self, node, type_name: str):
        """Find the first direct child of a specific node type."""
        for child in node.children:
            if child.type == type_name:
                return child
        return None

    def _find_child_by_field(self, node, field_name: str):
        """Find a child node by its field name."""
        return node.child_by_field_name(field_name)
