"""Interface commune des modèles d'embedding + cache disque.

Deux modèles sont comparés (voir config.yaml) :
- `mistral-embed` : API Mistral (60 requêtes/min sur l'offre gratuite)
- `local:BAAI/bge-m3` : encodeur de la famille BERT (XLM-RoBERTa), exécuté sur CPU
Les deux produisent des vecteurs de 1024 dimensions, normalisés (norme 1) : la
similarité cosinus se réduit alors à un produit scalaire.
"""

import hashlib
import sqlite3
from pathlib import Path
from typing import Protocol

import numpy as np

from app.core.config import DATA_DIR

CACHE_PATH = DATA_DIR / "cache" / "embeddings.sqlite"


class Embedder(Protocol):
    name: str  # identifiant stable, utilisé comme clé de cache et nom de colonne
    dimension: int

    def embed(self, texts: list[str]) -> np.ndarray: ...


def normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.where(norms == 0, 1, norms)


class EmbeddingCache:
    """Cache SQLite : (modèle, texte) -> vecteur. Ré-indexer ne coûte plus rien."""

    def __init__(self, path: Path = CACHE_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute("CREATE TABLE IF NOT EXISTS emb (key TEXT PRIMARY KEY, vec BLOB)")

    @staticmethod
    def key(model: str, text: str) -> str:
        return hashlib.sha256(f"{model}\0{text}".encode()).hexdigest()

    def get_many(self, model: str, texts: list[str]) -> dict[str, np.ndarray]:
        keys = [self.key(model, t) for t in texts]
        found: dict[str, np.ndarray] = {}
        for i in range(0, len(keys), 500):
            batch = keys[i : i + 500]
            marks = ",".join("?" * len(batch))
            rows = self.db.execute(f"SELECT key, vec FROM emb WHERE key IN ({marks})", batch)
            found.update({k: np.frombuffer(v, dtype=np.float32) for k, v in rows})
        return found

    def put_many(self, model: str, texts: list[str], vectors: np.ndarray) -> None:
        rows = [
            (self.key(model, t), v.astype(np.float32).tobytes())
            for t, v in zip(texts, vectors, strict=True)
        ]
        self.db.executemany("INSERT OR REPLACE INTO emb VALUES (?, ?)", rows)
        self.db.commit()


class CachedEmbedder:
    """Enveloppe un Embedder : ne calcule que les textes absents du cache."""

    def __init__(self, embedder: Embedder, cache: EmbeddingCache | None = None):
        self.inner = embedder
        self.cache = cache or EmbeddingCache()
        self.name = embedder.name
        self.dimension = embedder.dimension

    def embed(self, texts: list[str]) -> np.ndarray:
        cached = self.cache.get_many(self.name, texts)
        missing = list(
            dict.fromkeys(t for t in texts if self.cache.key(self.name, t) not in cached)
        )
        if missing:
            vectors = normalize(np.asarray(self.inner.embed(missing), dtype=np.float32))
            self.cache.put_many(self.name, missing, vectors)
            cached.update(
                {self.cache.key(self.name, t): v for t, v in zip(missing, vectors, strict=True)}
            )
        return np.stack([cached[self.cache.key(self.name, t)] for t in texts])
