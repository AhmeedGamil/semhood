"""Skill installer — target detection, per-editor formats, idempotent blocks."""

from __future__ import annotations

from pathlib import Path

from semhood.skill_install import (
    TARGET_KEYS,
    detected_targets,
    install_skill,
)


def _make_editor_dirs(home: Path, *keys_to_dirs: str) -> None:
    for d in keys_to_dirs:
        (home / d).mkdir(parents=True, exist_ok=True)


def test_detects_only_present_editors(tmp_path: Path):
    _make_editor_dirs(tmp_path, ".claude", ".cursor")
    assert set(detected_targets(tmp_path)) == {"claude", "cursor"}


def test_auto_installs_to_detected_and_skips_rest(tmp_path: Path):
    _make_editor_dirs(tmp_path, ".claude", ".codex")
    results = install_skill(home=tmp_path)

    by_target = {r["target"]: r for r in results}
    assert by_target["claude"]["status"] == "installed"
    assert by_target["codex"]["status"] == "installed"
    assert by_target["cursor"]["status"] == "skipped"

    # Claude gets the verbatim SKILL.md (frontmatter preserved).
    claude_file = Path(by_target["claude"]["path"])
    assert claude_file == tmp_path / ".claude" / "skills" / "semhood" / "SKILL.md"
    assert claude_file.read_text(encoding="utf-8").startswith("---")

    # Codex gets a managed block appended to AGENTS.md.
    codex_file = Path(by_target["codex"]["path"])
    assert codex_file == tmp_path / ".codex" / "AGENTS.md"
    assert "BEGIN semhood" in codex_file.read_text(encoding="utf-8")


def test_explicit_target_installs_even_if_undetected(tmp_path: Path):
    # No editor dirs exist, but the user asked for cursor explicitly.
    results = install_skill(["cursor"], home=tmp_path)
    cursor = next(r for r in results if r["target"] == "cursor")
    assert cursor["status"] == "installed"
    mdc = Path(cursor["path"])
    assert mdc.suffix == ".mdc"
    text = mdc.read_text(encoding="utf-8")
    assert "alwaysApply: false" in text and "description:" in text


def test_all_targets(tmp_path: Path):
    results = install_skill(list(TARGET_KEYS), home=tmp_path)
    assert {r["target"] for r in results if r["status"] == "installed"} == set(TARGET_KEYS)


def test_block_upsert_is_idempotent(tmp_path: Path):
    # Pre-seed Codex AGENTS.md with the user's own content.
    codex = tmp_path / ".codex"
    codex.mkdir()
    (codex / "AGENTS.md").write_text("# My rules\nkeep me\n", encoding="utf-8")

    install_skill(["codex"], home=tmp_path)
    install_skill(["codex"], home=tmp_path)  # twice

    text = (codex / "AGENTS.md").read_text(encoding="utf-8")
    assert text.count("BEGIN semhood") == 1  # not duplicated
    assert "keep me" in text  # user content preserved
