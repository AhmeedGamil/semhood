"""config.yaml loading + .env substitution + SemhoodConfig defaults."""

from __future__ import annotations

from pathlib import Path

from semhood.config import SemhoodConfig, load_config


def test_load_config_returns_defaults_when_missing(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SEMHOOD_CONFIG", raising=False)
    cfg = load_config()
    assert isinstance(cfg, SemhoodConfig)
    assert cfg.embeddings.provider == "local"  # zero-key default for public release
    assert cfg.enrichment.enabled is True
    assert cfg.enrichment.version == 1


def test_load_config_substitutes_env_vars(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("MY_API_KEY", "test-key-123")
    p = tmp_path / "config.yaml"
    p.write_text(
        "embeddings:\n"
        "  provider: openai\n"
        "  openai:\n"
        "    api_key: ${MY_API_KEY}\n"
        "    model: text-embedding-3-small\n"
    )
    cfg = load_config(str(p))
    assert cfg.embeddings.openai.api_key == "test-key-123"


def test_load_config_supports_legacy_strategy_field(tmp_path: Path):
    """The pre-refactor 'strategy: smart' field must still parse cleanly."""
    p = tmp_path / "config.yaml"
    p.write_text("strategy: smart\n")
    cfg = load_config(str(p))
    # Accepted but ignored — value is stored on the field for diagnostics.
    assert cfg.strategy == "smart"


def test_enrichment_skip_trivial_defaults():
    cfg = SemhoodConfig()
    assert cfg.enrichment.skip_trivial.enabled is True
    assert cfg.enrichment.skip_trivial.max_lines == 3
    assert cfg.enrichment.skip_trivial.require_control_flow is True


def test_parsing_policy_size_default():
    cfg = SemhoodConfig()
    assert cfg.parsing.max_file_size_mb == 1.0
    assert cfg.parsing.max_chunks_per_file == 500



def test_openrouter_provider_in_factory(tmp_path):
    """OpenRouter must be selectable as an LLM provider."""
    p = tmp_path / "config.yaml"
    p.write_text(
        "llm:\n"
        "  provider: openrouter\n"
        "  openrouter:\n"
        "    api_key: sk-or-test\n"
        "    model: anthropic/claude-sonnet-4\n"
    )
    cfg = load_config(str(p))
    assert cfg.llm.provider == "openrouter"
    assert cfg.llm.openrouter.api_key == "sk-or-test"
    assert cfg.llm.openrouter.model == "anthropic/claude-sonnet-4"

    # The factory should construct it without crashing
    from semhood.config import create_llm
    llm = create_llm(cfg)
    assert llm._model == "anthropic/claude-sonnet-4"
