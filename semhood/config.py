"""
Configuration loader for semhood.

Loads config.yaml, substitutes ${ENV_VAR} references, and provides
factory methods to instantiate the correct provider implementations.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

# =============================================================================
# Config models
# =============================================================================


class QdrantConfig(BaseModel):
    url: str | None = None
    path: str | None = "./qdrant_data"
    collection: str = "semhood"


class ChromaConfig(BaseModel):
    path: str = "./chroma_data"
    collection: str = "semhood"


class PineconeConfig(BaseModel):
    api_key: str = ""
    index: str = "semhood"
    environment: str = "us-east-1"


class WeaviateConfig(BaseModel):
    url: str = "http://localhost:8080"
    collection: str = "Semhood"


class MilvusConfig(BaseModel):
    url: str = "http://localhost:19530"
    collection: str = "semhood"


class VectorDBConfig(BaseModel):
    provider: str = "qdrant"
    qdrant: QdrantConfig = Field(default_factory=QdrantConfig)
    chroma: ChromaConfig = Field(default_factory=ChromaConfig)
    pinecone: PineconeConfig = Field(default_factory=PineconeConfig)
    weaviate: WeaviateConfig = Field(default_factory=WeaviateConfig)
    milvus: MilvusConfig = Field(default_factory=MilvusConfig)


class VoyageEmbeddingConfig(BaseModel):
    api_key: str = ""
    model: str = "voyage-code-2"


class OpenAIEmbeddingConfig(BaseModel):
    api_key: str = ""
    model: str = "text-embedding-3-small"


class CohereEmbeddingConfig(BaseModel):
    api_key: str = ""
    model: str = "embed-english-v3.0"


class LocalEmbeddingConfig(BaseModel):
    model: str = "sentence-transformers/all-mpnet-base-v2"
    device: str = "cpu"
    trust_remote_code: bool = False


class EmbeddingsConfig(BaseModel):
    # Local-by-default: semhood works offline with zero API keys out of the box.
    # Switch to "voyage"/"openai" in config for higher-quality, no-load embeddings.
    provider: str = "local"
    voyage: VoyageEmbeddingConfig = Field(default_factory=VoyageEmbeddingConfig)
    openai: OpenAIEmbeddingConfig = Field(default_factory=OpenAIEmbeddingConfig)
    cohere: CohereEmbeddingConfig = Field(default_factory=CohereEmbeddingConfig)
    local: LocalEmbeddingConfig = Field(default_factory=LocalEmbeddingConfig)


class AnthropicLLMConfig(BaseModel):
    api_key: str = ""
    model: str = "claude-sonnet-4-20250514"


class OpenAILLMConfig(BaseModel):
    api_key: str = ""
    model: str = "gpt-4o"


class GoogleLLMConfig(BaseModel):
    api_key: str = ""
    model: str = "gemini-1.5-pro"


class OllamaLLMConfig(BaseModel):
    url: str = "http://localhost:11434"
    model: str = "deepseek-coder-v2"


class OpenRouterLLMConfig(BaseModel):
    api_key: str = ""
    model: str = "anthropic/claude-sonnet-4"
    base_url: str | None = None
    http_referer: str | None = None
    x_title: str | None = None


class LLMConfig(BaseModel):
    provider: str = "anthropic"
    anthropic: AnthropicLLMConfig = Field(default_factory=AnthropicLLMConfig)
    openai: OpenAILLMConfig = Field(default_factory=OpenAILLMConfig)
    google: GoogleLLMConfig = Field(default_factory=GoogleLLMConfig)
    ollama: OllamaLLMConfig = Field(default_factory=OllamaLLMConfig)
    openrouter: OpenRouterLLMConfig = Field(default_factory=OpenRouterLLMConfig)


class CohereRerankerConfig(BaseModel):
    api_key: str = ""
    model: str = "rerank-english-v3.0"


class LocalRerankerConfig(BaseModel):
    model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    device: str = "cpu"


class RerankerConfig(BaseModel):
    provider: str = "none"
    cohere: CohereRerankerConfig = Field(default_factory=CohereRerankerConfig)
    local: LocalRerankerConfig = Field(default_factory=LocalRerankerConfig)


class ParsingConfig(BaseModel):
    exclude_dirs: list[str] = Field(default_factory=lambda: [
        "vendor", "node_modules", ".git", "__pycache__", "venv", ".venv",
        "dist", "build", ".tox", ".mypy_cache", ".pytest_cache",
        "target", "bin", "obj",
    ])
    exclude_files: list[str] = Field(default_factory=lambda: [
        "*.min.js", "*.bundle.js", "*.map", "*.lock",
        "*_pb2.py", "*.pb.go", "*_gen.go", "*.generated.*",
    ])
    languages: list[str] = Field(default_factory=lambda: [
        "python", "javascript", "typescript", "java", "go",
        "php", "csharp", "ruby", "rust", "cpp",
    ])
    max_file_size_mb: float = 1.0
    max_chunks_per_file: int = 500


# -----------------------------------------------------------------------------
# Enrichment
# -----------------------------------------------------------------------------


class EnrichmentRetryConfig(BaseModel):
    max_attempts: int = 3
    backoff_seconds: float = 2.0
    jitter: bool = True


class SkipTrivialConfig(BaseModel):
    enabled: bool = True
    max_lines: int = 3
    require_control_flow: bool = True


class EnrichmentConfig(BaseModel):
    enabled: bool = True
    version: int = 1
    max_source_chars: int = 30000
    llm_concurrency: int = 4
    retry: EnrichmentRetryConfig = Field(default_factory=EnrichmentRetryConfig)
    skip_patterns: list[str] = Field(default_factory=lambda: [
        "**/test_*.py",
        "**/*_test.go",
        "**/tests/**",
        "**/__tests__/**",
    ])
    skip_trivial: SkipTrivialConfig = Field(default_factory=SkipTrivialConfig)


class QueryConfig(BaseModel):
    top_k: int = 10
    rerank_candidates: int = 20
    rerank_top_n: int = 8
    graph_expansion: bool = True
    query_expansion: bool = True


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000


class SemhoodConfig(BaseModel):
    """Root configuration model."""

    vector_db: VectorDBConfig = Field(default_factory=VectorDBConfig)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    parsing: ParsingConfig = Field(default_factory=ParsingConfig)
    enrichment: EnrichmentConfig = Field(default_factory=EnrichmentConfig)
    query: QueryConfig = Field(default_factory=QueryConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)

    # Legacy field — accepted but ignored. The structural/enrichment split
    # replaced strategy selection with always-three-vector indexing.
    strategy: str | None = None


# =============================================================================
# Loader
# =============================================================================

_ENV_VAR_PATTERN = re.compile(r"\$\{([^}]+)\}")


def _load_dotenv_if_present() -> None:
    """Load .env.semhood (preferred) or .env from cwd if python-dotenv is available."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    # cwd-local files first, then the global ~/.semhood/.env (for the daemon,
    # whose working directory is wherever it happened to be auto-started from).
    candidates = [Path(".env.semhood"), Path(".env")]
    try:
        from semhood.paths import semhood_home
        candidates.append(semhood_home() / ".env")
    except Exception:
        pass

    for p in candidates:
        if p.exists():
            # override=False: real environment variables win over the file.
            load_dotenv(p, override=False)
            return


def _substitute_env_vars(value: Any) -> Any:
    """Recursively substitute ${ENV_VAR} references in config values."""
    if isinstance(value, str):
        def _replace(match: re.Match) -> str:
            var_name = match.group(1)
            return os.environ.get(var_name, "")
        return _ENV_VAR_PATTERN.sub(_replace, value)
    elif isinstance(value, dict):
        return {k: _substitute_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_substitute_env_vars(item) for item in value]
    return value


def load_config(config_path: str | Path | None = None) -> SemhoodConfig:
    """
    Load configuration from YAML file with environment variable substitution.

    Search order:
      1. Explicit path argument
      2. SEMHOOD_CONFIG environment variable
      3. ./config.yaml in current directory
      4. Default config (all defaults)

    Before resolving ``${VAR}`` references, we try to load a ``.env.semhood``
    or ``.env`` file from the current directory so users don't have to
    export their keys by hand.
    """
    _load_dotenv_if_present()

    if config_path is None:
        config_path = os.environ.get("SEMHOOD_CONFIG", "config.yaml")

    path = Path(config_path)
    if path.exists():
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        raw = _substitute_env_vars(raw)
        return SemhoodConfig(**raw)

    # No config file found — use defaults
    return SemhoodConfig()


# Minimal, zero-key default written to ~/.semhood/config.yaml on first run.
# Ships working offline (local embedder); remote providers are one edit away.
_DEFAULT_GLOBAL_CONFIG = """\
# semhood global config. One file for every project you index.
# Per-project indexes live under ~/.semhood/indexes/<id>/ automatically.
# Reference env vars as ${VAR_NAME}; a ~/.semhood/.env is loaded if present.

embeddings:
  provider: "local"            # local | voyage | openai | cohere
  local:
    model: "sentence-transformers/all-mpnet-base-v2"
    device: "cpu"              # cpu | cuda | mps
  # voyage:                    # code-tuned, no local load — needs an API key
  #   api_key: "${VOYAGE_API_KEY}"
  #   model: "voyage-code-2"

vector_db:
  provider: "qdrant"
  qdrant:
    collection: "semhood"      # path is set per-project by the daemon

# LLM is only needed for `semhood enrich` / `semhood query`. Search works without
# it, and editor agents can enrich via MCP using their own model (no key here).
llm:
  provider: "anthropic"
  anthropic:
    api_key: "${ANTHROPIC_API_KEY}"
    model: "claude-sonnet-4-20250514"

reranker:
  provider: "none"
"""


def load_global_config() -> SemhoodConfig:
    """Load the single shared config from ~/.semhood/config.yaml.

    Writes a zero-key default on first run so a fresh install works offline.
    This is what the daemon loads; per-project index paths are layered on top.
    """
    from semhood.paths import global_config_path

    path = global_config_path()
    if not path.exists():
        path.write_text(_DEFAULT_GLOBAL_CONFIG, encoding="utf-8")
    return load_config(path)


# =============================================================================
# Provider factories
# =============================================================================


def create_vector_db(
    config: SemhoodConfig,
    *,
    path_override: str | None = None,
    collection_override: str | None = None,
):
    """Create the configured VectorDB provider instance.

    The daemon serves many projects from one process, so it passes overrides to
    isolate each project's index:
      * ``path_override`` — for on-disk backends (local qdrant / chroma), point
        at this project's central index dir. Forces local mode (url ignored).
      * ``collection_override`` — for remote backends (qdrant url, etc.), give
        each project its own collection instead of a separate path.
    """
    provider = config.vector_db.provider

    if provider == "qdrant":
        from semhood.providers.vector_db.qdrant_provider import QdrantProvider
        cfg = config.vector_db.qdrant
        url, path = cfg.url, cfg.path
        collection = collection_override or cfg.collection
        if path_override is not None:
            path, url = path_override, None  # per-project on-disk store
        return QdrantProvider(url=url, path=path, collection=collection)
    elif provider == "chroma":
        from semhood.providers.vector_db.chroma_provider import ChromaProvider
        cfg = config.vector_db.chroma
        return ChromaProvider(
            path=path_override or cfg.path,
            collection=collection_override or cfg.collection,
        )
    else:
        raise ValueError(f"Unknown vector_db provider: {provider}")


def create_embedder(config: SemhoodConfig):
    """Create the configured Embedding provider instance."""
    provider = config.embeddings.provider

    if provider == "voyage":
        from semhood.providers.embeddings.voyage_provider import VoyageProvider
        cfg = config.embeddings.voyage
        return VoyageProvider(api_key=cfg.api_key, model=cfg.model)
    elif provider == "openai":
        from semhood.providers.embeddings.openai_provider import OpenAIEmbeddingProvider
        cfg = config.embeddings.openai
        return OpenAIEmbeddingProvider(api_key=cfg.api_key, model=cfg.model)
    elif provider == "local":
        from semhood.providers.embeddings.local_provider import LocalEmbeddingProvider
        cfg = config.embeddings.local
        return LocalEmbeddingProvider(
            model=cfg.model,
            device=cfg.device,
            trust_remote_code=cfg.trust_remote_code,
        )
    else:
        raise ValueError(f"Unknown embeddings provider: {provider}")


def create_llm(config: SemhoodConfig):
    """Create the configured LLM provider instance."""
    provider = config.llm.provider

    if provider == "anthropic":
        from semhood.providers.llm.anthropic_provider import AnthropicProvider
        cfg = config.llm.anthropic
        return AnthropicProvider(api_key=cfg.api_key, model=cfg.model)
    elif provider == "openai":
        from semhood.providers.llm.openai_provider import OpenAILLMProvider
        cfg = config.llm.openai
        return OpenAILLMProvider(api_key=cfg.api_key, model=cfg.model)
    elif provider == "ollama":
        from semhood.providers.llm.ollama_provider import OllamaProvider
        cfg = config.llm.ollama
        return OllamaProvider(url=cfg.url, model=cfg.model)
    elif provider == "openrouter":
        from semhood.providers.llm.openrouter_provider import OpenRouterProvider
        cfg = config.llm.openrouter
        return OpenRouterProvider(
            api_key=cfg.api_key,
            model=cfg.model,
            base_url=cfg.base_url,
            http_referer=cfg.http_referer,
            x_title=cfg.x_title,
        )
    else:
        raise ValueError(f"Unknown llm provider: {provider}")


def create_reranker(config: SemhoodConfig):
    """Create the configured Reranker provider instance."""
    provider = config.reranker.provider

    if provider == "none":
        from semhood.providers.reranker.none_provider import NoneReranker
        return NoneReranker()
    elif provider == "cohere":
        from semhood.providers.reranker.cohere_provider import CohereRerankerProvider
        cfg = config.reranker.cohere
        return CohereRerankerProvider(api_key=cfg.api_key, model=cfg.model)
    else:
        raise ValueError(f"Unknown reranker provider: {provider}")
