"""Python-specific chunk extractor using Tree-sitter."""

from __future__ import annotations

from tree_sitter import Node, Tree

from semhood.core.models import Chunk, ChunkType, Parameter
from semhood.parsing.extractors.base import LanguageExtractor


class PythonExtractor(LanguageExtractor):
    """
    Extracts functions, methods, and classes from Python source files.

    Node types handled:
      - function_definition → standalone function or method
      - class_definition → class-level chunk + individual methods
      - decorated_definition → unwraps to the inner definition
    """

    @property
    def language_name(self) -> str:
        return "python"

    def extract(self, tree: Tree, file_path: str, source: bytes) -> list[Chunk]:
        chunks: list[Chunk] = []
        self._walk_module(tree.root_node, file_path, source, chunks)
        return chunks

    def normalize_call(self, call: str) -> str:
        # self.method() or cls.method() → strip self/cls
        if call.startswith("self.") or call.startswith("cls."):
            return call.split(".", 1)[1]
        # Module.func() → Module::func (our normalized format)
        if "." in call:
            parts = call.split(".", 1)
            # Only if first part looks like a class/module (capitalized)
            if parts[0] and parts[0][0].isupper():
                return f"{parts[0]}::{parts[1]}"
            return parts[1]  # variable.method() → bare method name
        return call

    # -------------------------------------------------------------------------
    # Tree walking
    # -------------------------------------------------------------------------

    def _walk_module(
        self, root: Node, file_path: str, source: bytes, chunks: list[Chunk]
    ) -> None:
        """Walk the top-level module node."""
        for child in root.children:
            if child.type == "function_definition":
                chunks.append(self._extract_function(child, file_path, source, None))
            elif child.type == "class_definition":
                self._extract_class(child, file_path, source, chunks)
            elif child.type == "decorated_definition":
                inner = self._unwrap_decorated(child)
                if inner and inner.type == "function_definition":
                    chunks.append(self._extract_function(inner, file_path, source, None))
                elif inner and inner.type == "class_definition":
                    self._extract_class(inner, file_path, source, chunks)

    def _extract_class(
        self, node: Node, file_path: str, source: bytes, chunks: list[Chunk]
    ) -> None:
        """Extract a class and its methods."""
        name_node = self._find_child_by_field(node, "name")
        class_name = self._node_text(name_node, source) if name_node else "Unknown"

        # Class-level chunk (signature + docstring)
        docstring = self._extract_docstring(node, source)
        class_chunk = Chunk(
            name=class_name,
            type=ChunkType.CLASS,
            class_name=None,
            namespace=self._infer_module(file_path),
            file=file_path,
            language="python",
            line_start=node.start_point[0] + 1,
            line_end=node.end_point[0] + 1,
            docblock=docstring,
            source=self._node_text(node, source),
            calls=[],
        )
        chunks.append(class_chunk)

        # Individual methods
        method_names: list[str] = []
        body = self._find_child_by_field(node, "body")
        if body:
            for child in body.children:
                if child.type == "function_definition":
                    method_chunk = self._extract_function(child, file_path, source, class_name)
                    chunks.append(method_chunk)
                    method_names.append(method_chunk.name)
                elif child.type == "decorated_definition":
                    inner = self._unwrap_decorated(child)
                    if inner and inner.type == "function_definition":
                        method_chunk = self._extract_function(inner, file_path, source, class_name)
                        chunks.append(method_chunk)
                        method_names.append(method_chunk.name)

        # Link class → methods
        class_chunk.methods = method_names

    def _extract_function(
        self, node: Node, file_path: str, source: bytes, class_name: str | None
    ) -> Chunk:
        """Extract a single function or method."""
        name_node = self._find_child_by_field(node, "name")
        name = self._node_text(name_node, source) if name_node else "anonymous"

        # Determine type
        if class_name:
            if name == "__init__":
                chunk_type = ChunkType.CONSTRUCTOR
            else:
                chunk_type = ChunkType.METHOD
        else:
            chunk_type = ChunkType.FUNCTION

        # Visibility
        visibility = "private" if name.startswith("_") and not name.startswith("__") else "public"

        # Is static?
        is_static = self._has_decorator(node, "staticmethod", source)

        # Parameters
        params = self._extract_parameters(node, source, is_method=class_name is not None)

        # Return type
        return_type = self._extract_return_type(node, source)

        # Docstring
        docstring = self._extract_docstring(node, source)

        # Function body source
        func_source = self._node_text(node, source)

        # Calls
        calls = self._extract_calls(node, source)

        # Throws (raise statements)
        throws = self._extract_raises(node, source)

        return Chunk(
            name=name,
            type=chunk_type,
            class_name=class_name,
            namespace=self._infer_module(file_path),
            file=file_path,
            language="python",
            line_start=node.start_point[0] + 1,
            line_end=node.end_point[0] + 1,
            visibility=visibility,
            is_static=is_static,
            parameters=params,
            return_type=return_type,
            throws=throws,
            docblock=docstring,
            source=func_source,
            calls=list(set(calls)),  # Deduplicate
        )

    # -------------------------------------------------------------------------
    # Detail extractors
    # -------------------------------------------------------------------------

    def _extract_parameters(
        self, func_node: Node, source: bytes, is_method: bool
    ) -> list[Parameter]:
        """Extract function parameters."""
        params_node = self._find_child_by_field(func_node, "parameters")
        if not params_node:
            return []

        params = []
        skip_first = is_method  # Skip 'self' or 'cls'

        for child in params_node.children:
            if child.type in ("identifier", "typed_parameter", "default_parameter",
                              "typed_default_parameter"):
                if skip_first:
                    skip_first = False
                    continue

                name = ""
                ptype = None
                default = None

                if child.type == "identifier":
                    name = self._node_text(child, source)
                elif child.type == "typed_parameter":
                    name_node = self._find_child_by_type(child, "identifier")
                    type_node = child.child_by_field_name("type")
                    name = self._node_text(name_node, source) if name_node else ""
                    ptype = self._node_text(type_node, source) if type_node else None
                elif child.type == "default_parameter":
                    name_node = child.child_by_field_name("name")
                    value_node = child.child_by_field_name("value")
                    name = self._node_text(name_node, source) if name_node else ""
                    default = self._node_text(value_node, source) if value_node else None
                elif child.type == "typed_default_parameter":
                    name_node = child.child_by_field_name("name")
                    type_node = child.child_by_field_name("type")
                    value_node = child.child_by_field_name("value")
                    name = self._node_text(name_node, source) if name_node else ""
                    ptype = self._node_text(type_node, source) if type_node else None
                    default = self._node_text(value_node, source) if value_node else None

                if name and name not in ("self", "cls", "*", "**", "/"):
                    params.append(Parameter(name=name, type=ptype, default=default))

        return params

    def _extract_return_type(self, func_node: Node, source: bytes) -> str | None:
        """Extract return type annotation."""
        ret_type = func_node.child_by_field_name("return_type")
        if ret_type:
            return self._node_text(ret_type, source)
        return None

    def _extract_docstring(self, node: Node, source: bytes) -> str | None:
        """Extract docstring from function or class body."""
        body = self._find_child_by_field(node, "body")
        if not body or not body.children:
            return None

        first_stmt = body.children[0]
        if first_stmt.type == "expression_statement":
            expr = first_stmt.children[0] if first_stmt.children else None
            if expr and expr.type == "string":
                raw = self._node_text(expr, source)
                # Strip triple quotes
                for quote in ('"""', "'''"):
                    if raw.startswith(quote) and raw.endswith(quote):
                        return raw[3:-3].strip()
                return raw.strip("\"'").strip()
        return None

    def _extract_calls(self, node: Node, source: bytes) -> list[str]:
        """Extract all function/method calls within a node."""
        calls = []
        self._walk_calls(node, source, calls)
        return calls

    def _walk_calls(self, node: Node, source: bytes, calls: list[str]) -> None:
        """Recursively find call expressions."""
        if node.type == "call":
            func = node.child_by_field_name("function")
            if func:
                call_text = self._node_text(func, source)
                # Clean up common patterns
                if call_text and not call_text.startswith("("):
                    calls.append(call_text)

        for child in node.children:
            self._walk_calls(child, source, calls)

    def _extract_raises(self, node: Node, source: bytes) -> list[str]:
        """Extract exception types from raise statements."""
        raises = []
        self._walk_raises(node, source, raises)
        return list(set(raises))

    def _walk_raises(self, node: Node, source: bytes, raises: list[str]) -> None:
        """Recursively find raise statements."""
        if node.type == "raise_statement":
            # raise ExceptionType(...) or raise ExceptionType
            for child in node.children:
                if child.type == "call":
                    func = child.child_by_field_name("function")
                    if func:
                        raises.append(self._node_text(func, source))
                elif child.type == "identifier":
                    raises.append(self._node_text(child, source))

        for child in node.children:
            self._walk_raises(child, source, raises)

    def _has_decorator(self, node: Node, decorator_name: str, source: bytes) -> bool:
        """Check if a function has a specific decorator."""
        # Walk up to the decorated_definition parent
        parent = node.parent
        if parent and parent.type == "decorated_definition":
            for child in parent.children:
                if child.type == "decorator":
                    text = self._node_text(child, source)
                    if decorator_name in text:
                        return True
        return False

    def _unwrap_decorated(self, node: Node) -> Node | None:
        """Unwrap a decorated_definition to get the inner function/class."""
        for child in node.children:
            if child.type in ("function_definition", "class_definition"):
                return child
        return None

    def _infer_module(self, file_path: str) -> str:
        """Infer Python module path from file path."""
        # Convert file path to module notation: app/services/auth.py → app.services.auth
        path = file_path.replace("\\", "/")
        if path.endswith(".py"):
            path = path[:-3]
        if path.endswith("/__init__"):
            path = path[:-9]
        return path.replace("/", ".")
