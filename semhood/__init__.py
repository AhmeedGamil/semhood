"""semhood — AST-based semantic code search that knows the neighborhood.

Public surface intentionally tiny. Most callers use the CLI (``semhood``) or
the MCP server (``semhood-mcp``). For programmatic use, import from the
submodules directly (e.g. ``from semhood.core.indexer import Indexer``).
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("semhood")
except PackageNotFoundError:  # running from source without an install
    __version__ = "0.0.0+unknown"
