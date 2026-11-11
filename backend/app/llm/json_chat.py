"""Appel Mistral en mode JSON, avec cache disque, limitation de débit et nouvelles tentatives.

Offre gratuite : 2 requêtes/min pour ministral-8b. Le cache SQLite (modèle, température,
prompt) -> réponse rend les évaluations reproductibles et gratuites une fois calculées.
"""

import hashlib
import json
import logging
import sqlite3
import threading
import time
from pathlib import Path

import httpx

from app.core.config import DATA_DIR, get_app_config
from app.llm.client import _api_key

logger = logging.getLogger(__name__)
URL = "https://api.mistral.ai/v1/chat/completions"
CACHE_PATH = DATA_DIR / "cache" / "llm.sqlite"
MIN_INTERVAL = 31.0  # secondes entre deux appels réels (2 requêtes/min)


class LLMCache:
    def __init__(self, path: Path = CACHE_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.execute("CREATE TABLE IF NOT EXISTS llm (key TEXT PRIMARY KEY, answer TEXT)")

    @staticmethod
    def key(model: str, temperature: float, prompt: str) -> str:
        return hashlib.sha256(f"{model}\0{temperature}\0{prompt}".encode()).hexdigest()

    def get(self, key: str) -> str | None:
        row = self.db.execute("SELECT answer FROM llm WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def put(self, key: str, answer: str) -> None:
        self.db.execute("INSERT OR REPLACE INTO llm VALUES (?, ?)", (key, answer))
        self.db.commit()


_cache: LLMCache | None = None
_lock = threading.Lock()
_last_call = 0.0


def chat_json(prompt: str, temperature: float = 0.0, model: str | None = None) -> dict | list:
    """Réponse JSON du modèle (en cache si déjà demandée)."""
    global _cache, _last_call
    model = model or get_app_config().llm.chat_model
    with _lock:
        _cache = _cache or LLMCache()
        key = LLMCache.key(model, temperature, prompt)
        if (cached := _cache.get(key)) is not None:
            return json.loads(cached)
        body = {
            "model": model,
            "temperature": temperature,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "user", "content": prompt}],
        }
        for _ in range(5):
            time.sleep(max(0.0, _last_call + MIN_INTERVAL - time.monotonic()))
            _last_call = time.monotonic()
            r = httpx.post(
                URL, headers={"Authorization": f"Bearer {_api_key()}"}, json=body, timeout=120
            )
            if r.status_code == 200:
                answer = r.json()["choices"][0]["message"]["content"]
                parsed = json.loads(answer)  # lève une erreur si le JSON est invalide
                _cache.put(key, answer)
                return parsed
            if r.status_code == 429 or r.status_code >= 500:
                logger.warning("Mistral %s, nouvel essai dans 60 s", r.status_code)
                time.sleep(60)
                continue
            r.raise_for_status()
        raise RuntimeError("Mistral : échec après 5 essais")
