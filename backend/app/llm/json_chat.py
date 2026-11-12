"""Appels Mistral : JSON, texte, ou texte en streaming, avec rythme imposé et nouvelles tentatives.

Offre gratuite : 2 requêtes/min pour ministral-8b. Le cache SQLite (modèle, température,
messages) -> réponse rend les évaluations reproductibles et gratuites une fois calculées.
Le streaming (réponse affichée au fil de l'eau dans l'interface) n'est pas mis en cache.
"""

import hashlib
import json
import logging
import sqlite3
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import httpx

from app.core.config import DATA_DIR, get_app_config
from app.llm.client import _api_key

logger = logging.getLogger(__name__)
URL = "https://api.mistral.ai/v1/chat/completions"
CACHE_PATH = DATA_DIR / "cache" / "llm.sqlite"
MIN_INTERVAL = 31.0  # secondes entre deux appels réels (2 requêtes/min)
Messages = list[dict[str, str]]


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


def _wait_turn() -> None:
    """Respecte l'intervalle minimal entre deux appels réels (à appeler sous _lock)."""
    global _last_call
    time.sleep(max(0.0, _last_call + MIN_INTERVAL - time.monotonic()))
    _last_call = time.monotonic()


def _body(messages: Messages, temperature: float, model: str, **extra) -> dict:
    cfg = get_app_config().llm
    return {
        "model": model,
        "temperature": temperature,
        "max_tokens": cfg.max_tokens,
        "messages": messages,
        **extra,
    }


def _complete(body: dict) -> str:
    for _ in range(5):
        _wait_turn()
        r = httpx.post(
            URL, headers={"Authorization": f"Bearer {_api_key()}"}, json=body, timeout=120
        )
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"]
        if r.status_code == 429 or r.status_code >= 500:
            logger.warning("Mistral %s, nouvel essai dans 60 s", r.status_code)
            time.sleep(60)
            continue
        r.raise_for_status()
    raise RuntimeError("Mistral : échec après 5 essais")


def _cached(messages: Messages, temperature: float, model: str, json_mode: bool) -> str:
    global _cache
    with _lock:
        _cache = _cache or LLMCache()
        prompt = messages[0]["content"] if len(messages) == 1 else json.dumps(messages)
        # JSON : même clé qu'avant (réponses du multi-query déjà en cache) ; texte : clé distincte
        key = LLMCache.key(model if json_mode else f"{model}:texte", temperature, prompt)
        if (cached := _cache.get(key)) is not None:
            return cached
        extra = {"response_format": {"type": "json_object"}} if json_mode else {}
        answer = _complete(_body(messages, temperature, model, **extra))
        if json_mode:
            json.loads(answer)  # lève une erreur si le JSON est invalide : rien en cache
        _cache.put(key, answer)
        return answer


def chat_json(prompt: str, temperature: float = 0.0, model: str | None = None) -> dict | list:
    """Réponse JSON du modèle (en cache si déjà demandée)."""
    model = model or get_app_config().llm.chat_model
    messages = [{"role": "user", "content": prompt}]
    return json.loads(_cached(messages, temperature, model, json_mode=True))


def chat_text(messages: Messages, temperature: float = 0.0, model: str | None = None) -> str:
    """Réponse texte du modèle (en cache si déjà demandée)."""
    model = model or get_app_config().llm.chat_model
    return _cached(messages, temperature, model, json_mode=False)


def stream_text(
    messages: Messages, temperature: float = 0.0, model: str | None = None
) -> Iterator[str]:
    """Réponse texte morceau par morceau (Server-Sent Events de l'API Mistral), sans cache."""
    model = model or get_app_config().llm.chat_model
    body = _body(messages, temperature, model, stream=True)
    with _lock:
        _wait_turn()
    headers = {"Authorization": f"Bearer {_api_key()}"}
    with httpx.stream("POST", URL, headers=headers, json=body, timeout=120) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            delta = json.loads(line[6:])["choices"][0]["delta"].get("content")
            if delta:
                yield delta
