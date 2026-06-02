"""
Universal Tree-sitter parser.

Handles language detection, grammar loading, and dispatching to
the correct language-specific extractor.
"""

from __future__ import annotations

import logging
from pathlib import Path

from tree_sitter import Language, Parser

from semhood.core.models import Chunk
from semhood.parsing.extractors.base import LanguageExtractor
from semhood.parsing.extractors.generic_extractor import GenericExtractor
from semhood.parsing.extractors.javascript_extractor import JavaScriptExtractor
from semhood.parsing.extractors.python_extractor import PythonExtractor
from semhood.parsing.extractors.typescript_extractor import TypeScriptExtractor

logger = logging.getLogger(__name__)

# =============================================================================
# Extension → language mapping
# =============================================================================

EXTENSION_MAP: dict[str, str] = {
    ".py": "python",
    ".pyw": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".php": "php",
    ".cs": "csharp",
    ".rb": "ruby",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".c": "c",
    ".h": "c",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".swift": "swift",
    ".dart": "dart",
    ".lua": "lua",
    ".ex": "elixir",
    ".exs": "elixir",
    ".scala": "scala",
}

# =============================================================================
# Grammar loader map: language name → module import
# =============================================================================

_GRAMMAR_LOADERS: dict[str, str] = {
    "python": "tree_sitter_python",
    "javascript": "tree_sitter_javascript",
    "typescript": "tree_sitter_typescript",
    "java": "tree_sitter_java",
    "go": "tree_sitter_go",
    "rust": "tree_sitter_rust",
    "php": "tree_sitter_php",
    "csharp": "tree_sitter_c_sharp",
    "ruby": "tree_sitter_ruby",
    "cpp": "tree_sitter_cpp",
}


def _load_language(lang_name: str) -> Language | None:
    """Dynamically load a tree-sitter language grammar."""
    module_name = _GRAMMAR_LOADERS.get(lang_name)
    if not module_name:
        return None

    try:
        import importlib
        module = importlib.import_module(module_name)

        # tree-sitter-typescript exposes two languages: typescript and tsx
        if lang_name == "typescript":
            if hasattr(module, "language_typescript"):
                return Language(module.language_typescript())
            return Language(module.language())
        else:
            return Language(module.language())

    except (ImportError, AttributeError) as e:
        logger.warning("Could not load grammar for '%s': %s", lang_name, e)
        return None


# =============================================================================
# Extractor registry
# =============================================================================

_EXTRACTORS: dict[str, type[LanguageExtractor]] = {
    "python": PythonExtractor,
    "javascript": JavaScriptExtractor,
    "typescript": TypeScriptExtractor,
}


def _get_extractor(lang_name: str) -> LanguageExtractor:
    """Get the appropriate extractor for a language."""
    extractor_cls = _EXTRACTORS.get(lang_name)
    if extractor_cls:
        return extractor_cls()
    return GenericExtractor(lang_name)


# =============================================================================
# TreeSitterParser
# =============================================================================


class TreeSitterParser:
    """
    Universal parser that handles any Tree-sitter supported language.

    Usage:
        parser = TreeSitterParser()
        chunks = parser.parse("app/services/auth.py")
    """

    def __init__(self, allowed_languages: list[str] | None = None):
        """
        Args:
            allowed_languages: If set, only parse files in these languages.
                               If None, parse all languages with available grammars.
        """
        self._allowed = set(allowed_languages) if allowed_languages else None
        self._parsers: dict[str, Parser] = {}
        self._extractors: dict[str, LanguageExtractor] = {}

    def parse(self, file_path: str, base_path: str = "") -> list[Chunk]:
        """
        Parse a source file and return extracted chunks.

        Args:
            file_path: Absolute path to the source file.
            base_path: Base path to compute relative file paths.

        Returns:
            List of Chunk objects extracted from the file.
        """
        lang = self.detect_language(file_path)
        if not lang:
            return []

        if self._allowed and lang not in self._allowed:
            return []

        parser = self._get_parser(lang)
        if not parser:
            return []

        extractor = self._get_extractor(lang)

        try:
            source = Path(file_path).read_bytes()
        except OSError as e:
            logger.warning("Cannot read file '%s': %s", file_path, e)
            return []

        try:
            tree = parser.parse(source)
        except Exception as e:
            logger.warning("Parse error in '%s': %s", file_path, e)
            return []

        # Compute relative path
        rel_path = file_path
        if base_path:
            try:
                rel_path = str(Path(file_path).relative_to(base_path))
            except ValueError:
                pass

        try:
            chunks = extractor.extract(tree, rel_path, source)
            return chunks
        except Exception as e:
            logger.warning("Extraction error in '%s': %s", file_path, e)
            return []

    def detect_language(self, file_path: str) -> str | None:
        """Detect language from file extension."""
        ext = Path(file_path).suffix.lower()
        return EXTENSION_MAP.get(ext)

    def get_extractor_for(self, language: str) -> LanguageExtractor:
        """Get the extractor instance for a language (for call normalization)."""
        return self._get_extractor(language)

    def _get_parser(self, lang_name: str) -> Parser | None:
        """Get or create a parser for a language."""
        if lang_name in self._parsers:
            return self._parsers[lang_name]

        language = _load_language(lang_name)
        if not language:
            return None

        parser = Parser(language)
        self._parsers[lang_name] = parser
        return parser

    def _get_extractor(self, lang_name: str) -> LanguageExtractor:
        """Get or create an extractor for a language."""
        if lang_name not in self._extractors:
            self._extractors[lang_name] = _get_extractor(lang_name)
        return self._extractors[lang_name]
