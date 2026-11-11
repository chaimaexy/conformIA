"""Reranking par cross-encodeur (BERT) local, sur CPU.

Un bi-encodeur (bge-m3, mistral-embed) encode question et chunk SÉPARÉMENT : rapide (les
chunks sont encodés une fois pour toutes), mais la comparaison se réduit à un cosinus.
Un cross-encodeur lit la paire (question, chunk) ENSEMBLE, avec l'attention entre leurs mots :
bien plus précis, mais une passe complète du modèle par paire. On ne l'applique donc qu'aux
~20 meilleurs candidats de la recherche hybride.

Son score (probabilité que le chunk réponde à la question) sert aussi de garde-fou :
si même le meilleur chunk a un score faible, l'assistant répond « je ne sais pas ».

Candidats comparés (config.yaml, reranker.candidates) :
- cohere:<modèle>                 : API Cohere Rerank (référence commerciale)
- Qwen3-Reranker-0.6B (seq-cls)   : LLM Qwen3 affiné pour le reranking, Apache 2.0 (open SOTA)
- BAAI/bge-reranker-v2-m3         : XLM-RoBERTa large, multilingue, Apache 2.0, 568M
- crossencoder-camembert-base-mmarcoFR : CamemBERT base, français, MIT, 110M
"""

import hashlib
import logging
import sqlite3
import time
from pathlib import Path

import httpx

from app.core.config import DATA_DIR, get_settings
from app.embeddings.local import models_cache

logger = logging.getLogger(__name__)
CACHE_PATH = DATA_DIR / "cache" / "rerank.sqlite"


class RerankCache:
    """(modèle, question, chunk) -> score : comparer les configurations ne recalcule rien."""

    def __init__(self, path: Path = CACHE_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute("CREATE TABLE IF NOT EXISTS rr (key TEXT PRIMARY KEY, score REAL)")

    @staticmethod
    def key(model: str, query: str, passage: str) -> str:
        return hashlib.sha256(f"{model}\0{query}\0{passage}".encode()).hexdigest()

    def get_many(self, keys: list[str]) -> dict[str, float]:
        found: dict[str, float] = {}
        for i in range(0, len(keys), 500):
            part = keys[i : i + 500]
            marks = ",".join("?" * len(part))
            sql = f"SELECT key, score FROM rr WHERE key IN ({marks})"  # noqa: S608 (placeholders)
            found.update(self.db.execute(sql, part).fetchall())
        return found

    def put_many(self, items: list[tuple[str, float]]) -> None:
        self.db.executemany("INSERT OR REPLACE INTO rr VALUES (?, ?)", items)
        self.db.commit()


class Reranker:
    """Score de pertinence (0 à 1) de chaque passage, avec cache disque."""

    name: str

    def __init__(self, name: str):
        self.name = name
        self.cache = RerankCache()

    def _predict(self, query: str, passages: list[str]) -> list[float]:
        raise NotImplementedError

    def scores(self, query: str, passages: list[str]) -> list[float]:
        keys = [RerankCache.key(self.name, query, p) for p in passages]
        known = self.cache.get_many(keys)
        todo = [i for i, k in enumerate(keys) if k not in known]
        if todo:
            raw = self._predict(query, [passages[i] for i in todo])
            new = [(keys[i], float(s)) for i, s in zip(todo, raw, strict=True)]
            self.cache.put_many(new)
            known.update(new)
        return [known[k] for k in keys]


class CrossEncoderReranker(Reranker):
    """Cross-encodeur local (sentence-transformers), sur CPU."""

    def __init__(self, model_id: str, max_length: int = 512, batch_size: int = 8):
        super().__init__(model_id)
        self.model_id = model_id
        self.max_length = max_length
        self.batch_size = batch_size
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder  # import lourd : à la demande

            logger.info("Chargement du reranker %s (CPU)...", self.model_id)
            self._model = CrossEncoder(
                self.model_id,
                device="cpu",
                max_length=self.max_length,
                cache_folder=models_cache(self.model_id),
            )
        return self._model

    def _pairs(self, query: str, passages: list[str]) -> list[tuple[str, str]]:
        return [(query, p) for p in passages]

    def _predict(self, query: str, passages: list[str]) -> list[float]:
        import torch

        return list(
            self.model.predict(
                self._pairs(query, passages),
                batch_size=self.batch_size,
                activation_fn=torch.nn.Sigmoid(),
                show_progress_bar=False,
            )
        )


class Qwen3Reranker(CrossEncoderReranker):
    """Qwen3-Reranker : un LLM qui répond « yes »/« no » à « ce document répond-il ? ».

    La version « seq-cls » (conversion par le mainteneur de sentence-transformers) expose
    directement le logit yes - no comme un cross-encodeur. Le format de prompt est imposé ;
    l'instruction décrit la tâche (en anglais, langue d'entraînement des instructions).
    """

    PREFIX = (
        "<|im_start|>system\nJudge whether the Document meets the requirements based on the "
        'Query and the Instruct provided. Note that the answer can only be "yes" or "no".'
        "<|im_end|>\n<|im_start|>user\n"
    )
    SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    INSTRUCTION = (
        "Given a question about EU AI regulation or data protection law, retrieve the legal "
        "passages that answer it"
    )

    def __init__(self, model_id: str = "tomaarsen/Qwen3-Reranker-0.6B-seq-cls"):
        super().__init__(model_id, max_length=640, batch_size=4)  # prompt + chunk de 2 000 car.

    def _pairs(self, query: str, passages: list[str]) -> list[tuple[str, str]]:
        q = f"{self.PREFIX}<Instruct>: {self.INSTRUCTION}\n<Query>: {query}\n"
        return [(q, f"<Document>: {p}{self.SUFFIX}") for p in passages]


class CohereReranker(Reranker):
    """API Cohere Rerank. Clé d'essai : gratuite, limitée (~10 requêtes/min), non commerciale."""

    URL = "https://api.cohere.com/v2/rerank"
    MIN_INTERVAL = 6.5  # secondes entre deux appels (10 requêtes/min)

    def __init__(self, model: str):
        super().__init__(f"cohere:{model}")
        self.model = model
        self._last_call = 0.0

    def _predict(self, query: str, passages: list[str]) -> list[float]:
        key = get_settings().cohere_api_key
        if not key:
            raise RuntimeError("COHERE_API_KEY manquante : ajoute-la dans le fichier .env")
        body = {"model": self.model, "query": query, "documents": passages}
        for _ in range(5):
            time.sleep(max(0.0, self._last_call + self.MIN_INTERVAL - time.monotonic()))
            self._last_call = time.monotonic()
            headers = {"Authorization": f"Bearer {key}"}
            r = httpx.post(self.URL, headers=headers, json=body, timeout=60)
            if r.status_code == 200:
                scores = [0.0] * len(passages)
                for result in r.json()["results"]:
                    scores[result["index"]] = result["relevance_score"]
                return scores
            if r.status_code == 429 or r.status_code >= 500:
                logger.warning("Cohere %s, nouvel essai dans 60 s", r.status_code)
                time.sleep(60)
                continue
            r.raise_for_status()
        raise RuntimeError("Cohere : échec après 5 essais")


def get_reranker(spec: str) -> Reranker:
    """« cohere:<modèle> », un id Qwen3-Reranker, ou tout cross-encodeur Hugging Face."""
    if spec.startswith("cohere:"):
        return CohereReranker(spec.removeprefix("cohere:"))
    if "Qwen3-Reranker" in spec:
        return Qwen3Reranker(spec)
    return CrossEncoderReranker(spec)
