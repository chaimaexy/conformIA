"""Configuration centralisée.

Deux sources :
- `.env` / variables d'environnement pour les secrets (Settings)
- `config.yaml` pour les paramètres de l'application (AppConfig)
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[2]
REPO_DIR = BACKEND_DIR.parent
DATA_DIR = REPO_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"


class Settings(BaseSettings):
    """Secrets et paramètres dépendants de l'environnement."""

    model_config = SettingsConfigDict(
        env_file=(REPO_DIR / ".env", BACKEND_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    environment: Literal["local", "staging", "production"] = "local"
    log_level: str = "INFO"
    cors_origins: list[str] = ["http://localhost:3000"]

    mistral_api_key: str | None = None

    supabase_url: str | None = None
    # Nouveau format de clés Supabase (remplace anon / service_role)
    supabase_publishable_key: str | None = None  # côté client, soumise à RLS
    supabase_secret_key: str | None = None  # côté serveur uniquement, contourne RLS
    supabase_jwks_url: str | None = None  # vérification des JWT Supabase Auth
    database_url: str | None = None

    config_path: Path = BACKEND_DIR / "config.yaml"


class LLMConfig(BaseModel):
    provider: Literal["mistral"] = "mistral"
    chat_model: str = "ministral-8b-latest"
    temperature: float = 0.0
    max_tokens: int = 1024


class EmbeddingsConfig(BaseModel):
    provider: Literal["mistral"] = "mistral"
    model: str = "mistral-embed"
    dimension: int = 1024
    compare_with: list[str] = Field(default_factory=list)  # ex. "local:BAAI/bge-m3"


class RerankerConfig(BaseModel):
    model: str = "BAAI/bge-reranker-v2-m3"
    candidates: list[str] = Field(default_factory=list)
    answerability_threshold: float | None = None  # calibré sur le golden set


class ChunkingConfig(BaseModel):
    strategy: Literal["structural", "recursive", "semantic"] = "structural"
    chunk_size: int = 1000  # récursif
    chunk_overlap: int = 150  # récursif
    max_chars: int = 2000  # structurel et sémantique : taille maximale d'un chunk
    min_chars: int = 300  # sémantique : fusion des fragments plus courts
    semantic_percentile: float = 90  # sémantique : seuil de coupure (percentile des distances)


class RetrievalConfig(BaseModel):
    top_k: int = 8
    final_k: int = 4
    hybrid: bool = True
    multi_query: bool = False
    rrf_k: int = 60


class CorpusSource(BaseModel):
    id: str
    title: str
    kind: Literal["eurlex", "cnil"] = "eurlex"
    url: str | None = None  # page d'index (fiches CNIL)
    celex: str | None = None  # acte de base (JO) : source des considérants
    consolidated: str | None = None  # version consolidée : source des articles et annexes
    corrigenda: list[str] = Field(default_factory=list)  # rectificatifs FR à appliquer/vérifier


class CorpusConfig(BaseModel):
    language: str = "fr"
    sources: list[CorpusSource] = Field(default_factory=list)


class AppConfig(BaseModel):
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    corpus: CorpusConfig = Field(default_factory=CorpusConfig)


def load_app_config(path: Path) -> AppConfig:
    if not path.exists():
        return AppConfig()
    with path.open(encoding="utf-8") as f:
        return AppConfig.model_validate(yaml.safe_load(f) or {})


@lru_cache
def get_settings() -> Settings:
    return Settings()


@lru_cache
def get_app_config() -> AppConfig:
    return load_app_config(get_settings().config_path)
