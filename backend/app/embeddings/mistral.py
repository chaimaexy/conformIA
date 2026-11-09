"""Embeddings Mistral (`mistral-embed`) via l'API REST, avec lots et gestion du débit.

Offre gratuite : 60 requêtes/minute. On envoie des lots (64 textes max, ~12 000 tokens
estimés max) espacés d'au moins 1,1 s, et on réessaie avec attente croissante en cas
de 429 (débit dépassé) ou d'erreur serveur.
"""

import logging
import time

import httpx
import numpy as np

from app.llm.client import _api_key

logger = logging.getLogger(__name__)

URL = "https://api.mistral.ai/v1/embeddings"
MAX_BATCH = 64
MAX_BATCH_TOKENS = 12_000
MIN_INTERVAL = 1.1  # secondes entre deux requêtes (60/min)


def estimate_tokens(text: str) -> int:
    return len(text) // 3 + 1  # estimation prudente pour le français


def batches(texts: list[str]) -> list[list[str]]:
    out: list[list[str]] = []
    current: list[str] = []
    tokens = 0
    for text in texts:
        t = estimate_tokens(text)
        if current and (len(current) >= MAX_BATCH or tokens + t > MAX_BATCH_TOKENS):
            out.append(current)
            current, tokens = [], 0
        current.append(text)
        tokens += t
    return [*out, current] if current else out


class MistralEmbedder:
    name = "mistral-embed"
    dimension = 1024

    def __init__(self, max_retries: int = 6):
        self.max_retries = max_retries
        self._last_call = 0.0
        self._client = httpx.Client(timeout=120)

    def _post(self, batch: list[str]) -> list[list[float]]:
        for attempt in range(self.max_retries):
            wait = MIN_INTERVAL - (time.monotonic() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.monotonic()
            response = self._client.post(
                URL,
                headers={"Authorization": f"Bearer {_api_key()}"},
                json={"model": self.name, "input": batch},
            )
            if response.status_code == 200:
                data = sorted(response.json()["data"], key=lambda d: d["index"])
                return [d["embedding"] for d in data]
            if response.status_code in {429, 500, 502, 503, 504}:
                delay = 2**attempt * 2
                logger.warning("Mistral %s, nouvel essai dans %ss", response.status_code, delay)
                time.sleep(delay)
                continue
            response.raise_for_status()
        raise RuntimeError(f"Mistral embeddings : échec après {self.max_retries} essais")

    def embed(self, texts: list[str]) -> np.ndarray:
        groups = batches(texts)
        vectors: list[list[float]] = []
        for i, batch in enumerate(groups, start=1):
            vectors += self._post(batch)
            if len(groups) > 5 and i % 20 == 0:
                logger.info("mistral-embed : %d/%d lots", i, len(groups))
        return np.asarray(vectors, dtype=np.float32)
