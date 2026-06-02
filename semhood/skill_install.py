"""Install the bundled semhood guidance into AI editors — once, user-level.

The package ships one skill file (``resources/skills/SKILL.md``) that teaches an
agent *when* to reach for semhood's tools. Different editors call this concept
different things and keep it in different places — a "skill" in Claude Code, a
"rule" in Cursor/Windsurf, "steering" in Kiro, ``AGENTS.md`` in Codex. This
module writes the same guidance into each one's convention.

Everything here installs at the **user level**, so it applies to every project —
matching the rest of semhood's "set it up once" design. Editors that use a
single shared rules file (Windsurf, Codex) get a delimited, idempotent block so
re-running updates in place without clobbering the user's own rules.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from importlib.resources import files
from pathlib import Path

import yaml

_MARK_BEGIN = "<!-- BEGIN semhood (managed by `semhood install-skill`) -->"
_MARK_END = "<!-- END semhood -->"


# ---------------------------------------------------------------------------
# Bundled skill content
# ---------------------------------------------------------------------------


def _load_skill() -> tuple[str, dict, str]:
    """Return (raw_text, frontmatter, body) of the packaged SKILL.md."""
    raw = files("semhood").joinpath("resources/skills/SKILL.md").read_text(encoding="utf-8")
    if raw.startswith("---"):
        _, fm_raw, body = raw.split("---", 2)
        fm = yaml.safe_load(fm_raw) or {}
        return raw, fm, body.lstrip("\n")
    return raw, {}, raw


# ---------------------------------------------------------------------------
# Write helpers
# ---------------------------------------------------------------------------


def _write_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _upsert_block(path: Path, body: str) -> None:
    """Insert or replace a delimited semhood block in a shared rules file."""
    block = f"{_MARK_BEGIN}\n\n{body.strip()}\n\n{_MARK_END}\n"
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    if _MARK_BEGIN in existing and _MARK_END in existing:
        new = re.sub(
            re.escape(_MARK_BEGIN) + r".*?" + re.escape(_MARK_END),
            block.rstrip("\n"),
            existing,
            flags=re.DOTALL,
        )
    elif existing.strip():
        new = existing.rstrip("\n") + "\n\n" + block
    else:
        new = block
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(new, encoding="utf-8")


# ---------------------------------------------------------------------------
# Per-editor installers — each returns the path it wrote
# ---------------------------------------------------------------------------


def _install_claude(home: Path, raw: str, fm: dict, body: str) -> Path:
    # Claude Code reads ~/.claude/skills/<name>/SKILL.md verbatim (frontmatter and all).
    path = home / ".claude" / "skills" / "semhood" / "SKILL.md"
    _write_file(path, raw)
    return path


def _install_cursor(home: Path, raw: str, fm: dict, body: str) -> Path:
    # Cursor rules are .mdc with their own frontmatter (description + alwaysApply).
    path = home / ".cursor" / "rules" / "semhood.mdc"
    front = (
        "---\n"
        f"description: {fm.get('description', 'semhood semantic code search')}\n"
        "alwaysApply: false\n"
        "---\n\n"
    )
    _write_file(path, front + body)
    return path


def _install_kiro(home: Path, raw: str, fm: dict, body: str) -> Path:
    # Kiro steering files are plain markdown, always included by default.
    path = home / ".kiro" / "steering" / "semhood.md"
    _write_file(path, body)
    return path


def _install_windsurf(home: Path, raw: str, fm: dict, body: str) -> Path:
    # Windsurf loads one global rules file; append a managed block to it.
    path = home / ".codeium" / "windsurf" / "memories" / "global_rules.md"
    _upsert_block(path, body)
    return path


def _install_codex(home: Path, raw: str, fm: dict, body: str) -> Path:
    # Codex reads ~/.codex/AGENTS.md for global instructions; append a block.
    path = home / ".codex" / "AGENTS.md"
    _upsert_block(path, body)
    return path


# key -> (label, detect-dir relative to home, installer)
_Installer = Callable[[Path, str, dict, str], Path]
TARGETS: dict[str, tuple[str, str, _Installer]] = {
    "claude": ("Claude Code", ".claude", _install_claude),
    "cursor": ("Cursor", ".cursor", _install_cursor),
    "windsurf": ("Windsurf", ".codeium", _install_windsurf),
    "kiro": ("Kiro", ".kiro", _install_kiro),
    "codex": ("Codex", ".codex", _install_codex),
}
TARGET_KEYS = list(TARGETS)


def detected_targets(home: Path | None = None) -> list[str]:
    """Keys whose editor config directory exists on this machine."""
    home = home or Path.home()
    return [k for k, (_, probe, _) in TARGETS.items() if (home / probe).exists()]


def install_skill(
    targets: list[str] | None = None, *, home: Path | None = None
) -> list[dict]:
    """Install the guidance to ``targets`` (or all detected if None).

    Returns one status record per known target: ``status`` is "installed",
    "skipped" (not detected, in auto mode), or "error".
    """
    home = home or Path.home()
    raw, fm, body = _load_skill()
    detected = set(detected_targets(home))
    chosen = targets if targets is not None else sorted(detected)

    results: list[dict] = []
    for key in TARGET_KEYS:
        label = TARGETS[key][0]
        if key not in chosen:
            if targets is None:  # auto mode — report what we skipped and why
                results.append({"target": key, "label": label, "status": "skipped",
                                "reason": "not detected"})
            continue
        try:
            path = TARGETS[key][2](home, raw, fm, body)
            results.append({"target": key, "label": label, "status": "installed",
                            "path": str(path), "detected": key in detected})
        except Exception as exc:
            results.append({"target": key, "label": label, "status": "error",
                            "error": str(exc)})
    return results
