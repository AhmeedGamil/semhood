"""TypeScript chunk extractor — extends JavaScript extractor with type annotations."""

from __future__ import annotations

from tree_sitter import Node

from semhood.core.models import Parameter
from semhood.parsing.extractors.javascript_extractor import JavaScriptExtractor


class TypeScriptExtractor(JavaScriptExtractor):
    """
    TypeScript extractor — inherits all JavaScript extraction logic
    and adds TypeScript-specific handling for:
      - Type annotations on parameters and return types
      - Interfaces
      - Access modifiers (public/private/protected)
    """

    @property
    def language_name(self) -> str:
        return "typescript"

    def _extract_parameters(self, node: Node, source: bytes) -> list[Parameter]:
        """Extract parameters with TypeScript type annotations."""
        params_node = self._find_child_by_field(node, "parameters") or \
                      self._find_child_by_type(node, "formal_parameters")
        if not params_node:
            return []

        params = []
        for child in params_node.children:
            if child.type == "required_parameter" or child.type == "optional_parameter":
                name_node = child.child_by_field_name("pattern") or \
                            self._find_child_by_type(child, "identifier")
                type_node = child.child_by_field_name("type")

                name = self._node_text(name_node, source) if name_node else ""
                ptype = self._node_text(type_node, source) if type_node else None

                if name:
                    # Strip type annotation prefix ':'
                    if ptype and ptype.startswith(":"):
                        ptype = ptype[1:].strip()
                    params.append(Parameter(name=name, type=ptype))

            elif child.type == "identifier":
                params.append(Parameter(name=self._node_text(child, source)))

        return params

    def _extract_method(self, node, file_path, source, class_name):
        """Override to capture TypeScript access modifiers."""
        chunk = super()._extract_method(node, file_path, source, class_name)
        chunk.language = "typescript"

        # Check for access modifiers
        for child in node.children:
            if child.type == "accessibility_modifier":
                chunk.visibility = self._node_text(child, source)
                break

        # Check return type annotation
        ret = node.child_by_field_name("return_type")
        if ret:
            rt = self._node_text(ret, source)
            if rt.startswith(":"):
                rt = rt[1:].strip()
            chunk.return_type = rt

        return chunk

    def _extract_function(self, node, file_path, source, class_name):
        """Override to set language to typescript."""
        chunk = super()._extract_function(node, file_path, source, class_name)
        chunk.language = "typescript"

        # Check return type annotation
        ret = node.child_by_field_name("return_type")
        if ret:
            rt = self._node_text(ret, source)
            if rt.startswith(":"):
                rt = rt[1:].strip()
            chunk.return_type = rt

        return chunk
