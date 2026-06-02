"""JavaScript-specific chunk extractor using Tree-sitter."""

from __future__ import annotations

from tree_sitter import Node, Tree

from semhood.core.models import Chunk, ChunkType, Parameter
from semhood.parsing.extractors.base import LanguageExtractor


class JavaScriptExtractor(LanguageExtractor):
    """
    Extracts functions, methods, and classes from JavaScript source files.

    Node types handled:
      - function_declaration
      - class_declaration → class + individual method_definition nodes
      - arrow_function (when assigned to named variable)
      - export_statement (unwraps to inner declaration)
    """

    @property
    def language_name(self) -> str:
        return "javascript"

    def extract(self, tree: Tree, file_path: str, source: bytes) -> list[Chunk]:
        chunks: list[Chunk] = []
        self._walk_program(tree.root_node, file_path, source, chunks)
        return chunks

    def normalize_call(self, call: str) -> str:
        # this.method() → strip this
        if call.startswith("this."):
            return call[5:]
        # super.method() → strip super
        if call.startswith("super."):
            return call[6:]
        # Module.func() → Module::func
        if "." in call:
            parts = call.split(".", 1)
            if parts[0] and parts[0][0].isupper():
                return f"{parts[0]}::{parts[1]}"
            return parts[1]
        return call

    # -------------------------------------------------------------------------
    # Tree walking
    # -------------------------------------------------------------------------

    def _walk_program(
        self, root: Node, file_path: str, source: bytes, chunks: list[Chunk]
    ) -> None:
        for child in root.children:
            if child.type == "function_declaration":
                chunks.append(self._extract_function(child, file_path, source, None))
            elif child.type == "class_declaration":
                self._extract_class(child, file_path, source, chunks)
            elif child.type == "export_statement":
                # export default/named — unwrap inner declaration
                self._walk_export(child, file_path, source, chunks)
            elif child.type == "lexical_declaration":
                # const myFunc = (...) => { ... }
                self._try_extract_arrow(child, file_path, source, chunks)

    def _walk_export(
        self, node: Node, file_path: str, source: bytes, chunks: list[Chunk]
    ) -> None:
        """Unwrap export statements to find the actual declaration."""
        for child in node.children:
            if child.type == "function_declaration":
                chunks.append(self._extract_function(child, file_path, source, None))
            elif child.type == "class_declaration":
                self._extract_class(child, file_path, source, chunks)
            elif child.type == "lexical_declaration":
                self._try_extract_arrow(child, file_path, source, chunks)

    def _extract_class(
        self, node: Node, file_path: str, source: bytes, chunks: list[Chunk]
    ) -> None:
        """Extract a class and its methods."""
        name_node = self._find_child_by_field(node, "name")
        class_name = self._node_text(name_node, source) if name_node else "Anonymous"

        # Class-level chunk
        class_chunk = Chunk(
            name=class_name,
            type=ChunkType.CLASS,
            file=file_path,
            language="javascript",
            line_start=node.start_point[0] + 1,
            line_end=node.end_point[0] + 1,
            source=self._node_text(node, source),
            calls=[],
        )
        chunks.append(class_chunk)

        # Methods inside class body
        method_names: list[str] = []
        body = self._find_child_by_field(node, "body")
        if body:
            for child in body.children:
                if child.type == "method_definition":
                    method_chunk = self._extract_method(child, file_path, source, class_name)
                    chunks.append(method_chunk)
                    method_names.append(method_chunk.name)

        # Link class → methods
        class_chunk.methods = method_names

    def _extract_function(
        self, node: Node, file_path: str, source: bytes, class_name: str | None
    ) -> Chunk:
        """Extract a standalone function declaration."""
        name_node = self._find_child_by_field(node, "name")
        name = self._node_text(name_node, source) if name_node else "anonymous"

        params = self._extract_parameters(node, source)
        calls = self._extract_calls(node, source)
        jsdoc = self._extract_jsdoc(node, source)

        return Chunk(
            name=name,
            type=ChunkType.FUNCTION if not class_name else ChunkType.METHOD,
            class_name=class_name,
            file=file_path,
            language="javascript",
            line_start=node.start_point[0] + 1,
            line_end=node.end_point[0] + 1,
            parameters=params,
            docblock=jsdoc,
            source=self._node_text(node, source),
            calls=list(set(calls)),
        )

    def _extract_method(
        self, node: Node, file_path: str, source: bytes, class_name: str
    ) -> Chunk:
        """Extract a class method."""
        name_node = self._find_child_by_field(node, "name")
        name = self._node_text(name_node, source) if name_node else "anonymous"

        chunk_type = ChunkType.CONSTRUCTOR if name == "constructor" else ChunkType.METHOD

        # Check static keyword
        is_static = any(child.type == "static" for child in node.children)

        # Visibility from naming convention (JS has no keywords for this, # prefix = private)
        visibility = "private" if name.startswith("#") or name.startswith("_") else "public"

        params = self._extract_parameters(node, source)
        calls = self._extract_calls(node, source)
        jsdoc = self._extract_jsdoc(node, source)

        return Chunk(
            name=name,
            type=chunk_type,
            class_name=class_name,
            file=file_path,
            language="javascript",
            line_start=node.start_point[0] + 1,
            line_end=node.end_point[0] + 1,
            visibility=visibility,
            is_static=is_static,
            parameters=params,
            docblock=jsdoc,
            source=self._node_text(node, source),
            calls=list(set(calls)),
        )

    def _try_extract_arrow(
        self, node: Node, file_path: str, source: bytes, chunks: list[Chunk]
    ) -> None:
        """Try to extract named arrow functions (const foo = () => {})."""
        for child in node.children:
            if child.type == "variable_declarator":
                name_node = self._find_child_by_field(child, "name")
                value_node = self._find_child_by_field(child, "value")
                if name_node and value_node and value_node.type == "arrow_function":
                    name = self._node_text(name_node, source)
                    calls = self._extract_calls(value_node, source)
                    params = self._extract_parameters(value_node, source)

                    chunks.append(Chunk(
                        name=name,
                        type=ChunkType.FUNCTION,
                        file=file_path,
                        language="javascript",
                        line_start=node.start_point[0] + 1,
                        line_end=node.end_point[0] + 1,
                        parameters=params,
                        source=self._node_text(node, source),
                        calls=list(set(calls)),
                    ))

    # -------------------------------------------------------------------------
    # Detail extractors
    # -------------------------------------------------------------------------

    def _extract_parameters(self, node: Node, source: bytes) -> list[Parameter]:
        """Extract function parameters."""
        params_node = self._find_child_by_field(node, "parameters") or \
                      self._find_child_by_type(node, "formal_parameters")
        if not params_node:
            return []

        params = []
        for child in params_node.children:
            if child.type == "identifier":
                params.append(Parameter(name=self._node_text(child, source)))
            elif child.type == "assignment_pattern":
                # Default parameter: name = value
                left = self._find_child_by_field(child, "left")
                right = self._find_child_by_field(child, "right")
                if left:
                    params.append(Parameter(
                        name=self._node_text(left, source),
                        default=self._node_text(right, source) if right else None,
                    ))
            elif child.type in ("rest_pattern", "rest_element"):
                for sub in child.children:
                    if sub.type == "identifier":
                        params.append(Parameter(name=f"...{self._node_text(sub, source)}"))

        return params

    def _extract_calls(self, node: Node, source: bytes) -> list[str]:
        """Extract all function/method calls."""
        calls: list[str] = []
        self._walk_calls(node, source, calls)
        return calls

    def _walk_calls(self, node: Node, source: bytes, calls: list[str]) -> None:
        if node.type == "call_expression":
            func = self._find_child_by_field(node, "function")
            if func:
                text = self._node_text(func, source)
                if text and not text.startswith("("):
                    calls.append(text)

        for child in node.children:
            self._walk_calls(child, source, calls)

    def _extract_jsdoc(self, node: Node, source: bytes) -> str | None:
        """Extract JSDoc comment preceding a function/method."""
        # Check previous sibling for a comment node
        prev = node.prev_named_sibling
        if prev and prev.type == "comment":
            text = self._node_text(prev, source)
            if text.startswith("/**"):
                # Strip /** and */
                text = text[3:]
                if text.endswith("*/"):
                    text = text[:-2]
                # Clean up * prefixes on lines
                lines = [
                    line.strip().lstrip("* ").strip()
                    for line in text.split("\n")
                ]
                return "\n".join(line for line in lines if line).strip()
        return None
