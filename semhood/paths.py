"""Filesystem layout for semhood's central, per-project index store.

semhood is meant to behave like ripgrep: install once, register once, and it
"just works" in any project you open. To get there we keep **one** global
config and store **each project's index separately**, keyed by the project's
root path — so opening any folder routes to the right index automatically with
no per-project config file.

Layout (override the base with ``$SEMHOOD_HOME``)::

    ~/.semhood/
      config.yaml                 # one global config (shared embedder, keys, ...)
      indexes/
        <project-id>/             # one directory per project root
          qdrant_data/            # that project's vector store
          meta.json               # {"root": "<absolute path>"} for `semhood list`

``<project-id>`` is a short, stable hash of the project's absolute root path,
so the same checkout always maps to the same index dir across sessions.

Why a central store (vs. a ``.semhood/`` folder inside each repo): the local
Qdrant backend locks its on-disk directory, so exactly one process may hold an
index open at a time. A single daemon owning ``~/.semhood/indexes/*`` is the
clean way to satisfy that — the CLI and the editor's MCP server both talk to
that one warm process instead of each trying to open the directory.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

# Markers that identify a project root, in priority order. We treat the
# nearest ancestor containing any of these as the root, falling back to the
# starting directory itself when none is found.
_ROOT_MARKERS = (".git", ".hg", ".svn", ".semhood")


def semhood_home() -> Path:
    """Base directory for all semhood state (``$SEMHOOD_HOME`` or ``~/.semhood``)."""
    override = os.environ.get("SEMHOOD_HOME")
    base = Path(override).expanduser() if override else Path.home() / ".semhood"
    base.mkdir(parents=True, exist_ok=True)
    return base


def global_config_path() -> Path:
    """Path to the single shared config file (may not exist yet)."""
    return semhood_home() / "config.yaml"


def daemon_info_path() -> Path:
    """File where the running daemon records its host/port/pid for clients."""
    return semhood_home() / "daemon.json"


def indexes_root() -> Path:
    """Directory holding one subdirectory per indexed project."""
    root = semhood_home() / "indexes"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _normalize_root(path: str | Path) -> Path:
    """Resolve to an absolute path, case-folded on case-insensitive platforms.

    On Windows the same repo can be addressed as ``D:\\Code`` or ``d:\\code``;
    folding the case keeps both pointing at one index dir.
    """
    resolved = Path(path).expanduser().resolve()
    if os.name == "nt":
        return Path(str(resolved).lower())
    return resolved


def find_project_root(start: str | Path | None = None) -> Path:
    """Walk upward from ``start`` (default: cwd) to the nearest project root.

    Returns the first ancestor that contains a VCS/`.semhood` marker, or the
    starting directory itself if no marker is found before the filesystem root.
    """
    current = Path(start).expanduser().resolve() if start else Path.cwd().resolve()
    if current.is_file():
        current = current.parent

    for directory in (current, *current.parents):
        if any((directory / marker).exists() for marker in _ROOT_MARKERS):
            return directory
    return current


def project_id(root: str | Path) -> str:
    """Stable short id for a project root — used as its index directory name."""
    normalized = _normalize_root(root)
    digest = hashlib.sha256(str(normalized).encode("utf-8")).hexdigest()
    return digest[:16]


def index_dir_for(root: str | Path, *, create: bool = True) -> Path:
    """Return (and optionally create) the central index directory for ``root``.

    Also writes/refreshes a ``meta.json`` recording the real project path so
    ``semhood list`` can map opaque ids back to human-readable directories.
    """
    normalized = _normalize_root(root)
    directory = indexes_root() / project_id(normalized)
    if create:
        directory.mkdir(parents=True, exist_ok=True)
        meta = directory / "meta.json"
        if not meta.exists():
            meta.write_text(
                json.dumps({"root": str(normalized)}, indent=2),
                encoding="utf-8",
            )
    return directory


def qdrant_path_for(root: str | Path) -> str:
    """On-disk Qdrant path for a project's central index."""
    return str(index_dir_for(root) / "qdrant_data")


# Written into a new ``.semhood/`` so the portable enrichment file is tracked by
# git while transient scratch (the atomic-write temp) is not.
_DOTDIR_GITIGNORE = """\
# semhood keeps the portable enrichment cache here.
#
# enrichment.jsonl IS meant to be committed: it's content-addressed enrichment
# text that survives embedding-model changes (re-embed, no re-enrich) and lets a
# team share enrichment instead of each developer re-paying for it. Commit it.
#
# Only transient scratch is ignored:
*.tmp
"""


def scaffold_dotdir(directory: str | Path) -> Path:
    """Create a ``.semhood`` directory and write its ``.gitignore`` if absent.

    Idempotent. Called lazily — at the moment something is first written — so a
    project that is only searched (never enriched) never grows a ``.semhood``.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    gitignore = directory / ".gitignore"
    if not gitignore.exists():
        gitignore.write_text(_DOTDIR_GITIGNORE, encoding="utf-8")
    return directory


def project_dotdir(root: str | Path, *, create: bool = True) -> Path:
    """Return (and optionally create) the in-repo ``<root>/.semhood`` directory.

    Unlike the index (which lives centrally under ``~/.semhood/indexes``), this
    directory lives **inside the project** so its contents travel with the repo
    in git. ``root`` is taken as authoritative (callers pass the already-resolved
    project root), so we place ``.semhood`` directly under it.
    """
    directory = Path(root).expanduser().resolve() / ".semhood"
    if create:
        scaffold_dotdir(directory)
    return directory


def enrichment_file_for(root: str | Path, *, create: bool = False) -> Path:
    """Path to a project's portable enrichment store (``.semhood/enrichment.jsonl``).

    Defaults to ``create=False``: constructing the store must not touch the
    filesystem. The directory is scaffolded lazily on the first write.
    """
    return project_dotdir(root, create=create) / "enrichment.jsonl"


def list_indexed_projects() -> list[dict[str, str]]:
    """Enumerate known project indexes as ``{"id", "root", "path"}`` records."""
    out: list[dict[str, str]] = []
    root_dir = indexes_root()
    for entry in sorted(root_dir.iterdir()):
        if not entry.is_dir():
            continue
        meta_file = entry / "meta.json"
        root_path = ""
        if meta_file.exists():
            try:
                root_path = json.loads(meta_file.read_text(encoding="utf-8")).get("root", "")
            except (ValueError, OSError):
                root_path = ""
        out.append({"id": entry.name, "root": root_path, "path": str(entry)})
    return out
