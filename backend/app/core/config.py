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
    cohere_api_key: str | None = None  # reranker Cohere (clé d'essai gratuite)
    jina_api_key: str | None = None  # reranker Jina AI (comparaison, phase 4 ter)
    models_cache_dir: Path | None = None  # cache Hugging Face (défaut : data/cache/hf)

    supabase_url: str | None = None
    # Nouveau format de clés Supabase (remplace anon / service_role)
    supabase_publishable_key: str | None = None  # côté client, soumise à RLS
    supabase_secret_key: str | None = None  # côté serveur uniquement, contourne RLS
    supabase_jwks_url: str | None = None  # vérification des JWT Supabase Auth
    database_url: str | None = None

    config_path: Path = BACKEND_DIR / "config.yaml"


class RateLimit(BaseModel):
    rpm: int = 2  # requêtes par minute
    tpm: int = 40000  # tokens par minute (entrée + sortie)


class LLMConfig(BaseModel):
    provider: Literal["mistral"] = "mistral"
    chat_model: str = "ministral-8b-latest"
    judge_model: str = (
        "mistral-large-latest"  # évaluation : raisonnement complet (effort par défaut)
    )
    temperature: float = 0.0
    max_tokens: int = 1024
    max_tokens_by_model: dict[str, int] = Field(default_factory=dict)  # modèles qui raisonnent
    rate_limits: dict[str, RateLimit] = Field(default_factory=dict)  # par modèle ; défaut : 2/min
    # effort de raisonnement des réponses (pas du juge), par modèle qui raisonne : low/medium/high
    reasoning_effort: dict[str, str] = Field(default_factory=dict)


class EmbeddingsConfig(BaseModel):
    provider: Literal["mistral"] = "mistral"
    model: str = "mistral-embed"
    dimension: int = 1024
    compare_with: list[str] = Field(default_factory=list)  # ex. "local:BAAI/bge-m3"


class RerankerConfig(BaseModel):
    model: str | None = "cohere:rerank-v5.0-pro"  # None : pas de reranking
    candidates: list[str] = Field(default_factory=list)  # comparés en phase 4
    answerability_threshold: float | None = None  # « je ne sais pas » sous ce score
    verification_threshold: float | None = None  # zone grise en dessous : vérification par le LLM


class ChunkingConfig(BaseModel):
    strategy: Literal["structural", "recursive", "semantic"] = "structural"
    chunk_size: int = 1000  # récursif
    chunk_overlap: int = 150  # récursif
    max_chars: int = 2000  # structurel et sémantique : taille maximale d'un chunk
    min_chars: int = 300  # sémantique : fusion des fragments plus courts
    semantic_percentile: float = 90  # sémantique : seuil de coupure (percentile des distances)
    article_references: bool = False  # structurel : Renvois aussi pour les articles cités


class RetrievalConfig(BaseModel):
    dense_models: list[str] = Field(default_factory=lambda: ["mistral-embed"])
    bm25: bool = True
    candidates: int = 50  # résultats de chaque retriever avant fusion
    rrf_k: int = 60
    multi_query: bool = False
    rerank_candidates: int = 20  # candidats fusionnés relus par le reranker
    final_k: int = 4  # chunks envoyés au LLM
    parent_max_chars: int = 0  # section découpée jointe en entier jusqu'à cette taille (0 : non)
    temporal_sections: list[str] = Field(default_factory=list)  # ex. « Article 113 »


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
