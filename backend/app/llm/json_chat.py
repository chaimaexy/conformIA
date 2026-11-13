"""Appels Mistral : JSON, texte, ou texte en streaming, avec débit limité et nouvelles tentatives.

Chaque modèle a ses limites (config.yaml, llm.rate_limits) : requêtes ET tokens par minute.
Offre gratuite : 2 requêtes/min pour ministral-8b ; 20 000 tokens/min pour mistral-large.
Le cache SQLite (modèle, température, messages) -> réponse rend les évaluations
reproductibles et gratuites une fois calculées. Le streaming n'est pas mis en cache.
"""

import hashlib
import json
import logging
import sqlite3
import threading
import time
from collections import deque
from collections.abc import Iterator
from pathlib import Path

import httpx

from app.core.config import DATA_DIR, get_app_config
from app.llm.client import _api_key

logger = logging.getLogger(__name__)
URL = "https://api.mistral.ai/v1/chat/completions"
CACHE_PATH = DATA_DIR / "cache" / "llm.sqlite"
CHARS_PER_TOKEN = 3.0  # estimation prudente pour du français
EXPECTED_COMPLETION = 6000  # tokens de sortie comptés d'avance (raisonnement compris), au plus
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


class RateLimiter:
    """Fenêtre glissante d'une minute : nombre de requêtes et tokens consommés, par modèle.

    Les appels sont aussi espacés d'au moins 60/rpm secondes (avec 2 requêtes/min, deux
    appels collés déclenchent une erreur 429 chez Mistral).
    """

    def __init__(self, rpm: int, tpm: int, clock=time.monotonic, sleep=time.sleep):
        self.rpm, self.tpm = rpm, tpm
        self.clock, self.sleep = clock, sleep
        self.calls: deque[tuple[float, int]] = deque()

    def wait(self, tokens: int) -> None:
        tokens = min(tokens, self.tpm)  # une requête plus grosse que la limite passe seule
        while True:
            now = self.clock()
            while self.calls and now - self.calls[0][0] >= 60:
                self.calls.popleft()
            used = sum(t for _, t in self.calls)
            spaced = not self.calls or now - self.calls[-1][0] >= 60 / self.rpm
            if spaced and len(self.calls) < self.rpm and used + tokens <= self.tpm:
                self.calls.append((now, tokens))
                return
            next_free = self.calls[0][0] + 60 if self.calls else now
            if not spaced:
                next_free = min(next_free, self.calls[-1][0] + 60 / self.rpm)
            self.sleep(max(0.05, next_free - now))


_cache: LLMCache | None = None
_lock = threading.Lock()
_limiters: dict[str, RateLimiter] = {}


def estimate_tokens(body: dict) -> int:
    chars = sum(len(m["content"]) for m in body["messages"])
    return int(chars / CHARS_PER_TOKEN) + min(body.get("max_tokens", 1024), EXPECTED_COMPLETION)


def _wait_turn(body: dict) -> None:
    """Attend que le modèle accepte un appel de cette taille (à appeler sous _lock)."""
    model = body["model"]
    if model not in _limiters:
        limit = get_app_config().llm.rate_limits.get(model)
        _limiters[model] = RateLimiter(limit.rpm, limit.tpm) if limit else RateLimiter(2, 40000)
    _limiters[model].wait(estimate_tokens(body))


def _body(
    messages: Messages, temperature: float, model: str, effort: str | None = None, **extra
) -> dict:
    cfg = get_app_config().llm
    if effort:  # modèle qui raisonne : low / medium / high (défaut de l'API sinon)
        extra["reasoning_effort"] = effort
    return {
        "model": model,
        "temperature": temperature,
        "top_p": 1,  # exigé par mistral-large avec une température de 0 (échantillonnage glouton)
        # un modèle qui raisonne (mistral-large) consomme des tokens avant de répondre
        "max_tokens": cfg.max_tokens_by_model.get(model, cfg.max_tokens),
        "messages": messages,
        **extra,
    }


def text_of(content: str | list | None) -> str:
    """Texte d'une réponse Mistral. Un modèle qui raisonne (mistral-large) renvoie une liste de
    blocs : {"type": "thinking", ...} (son raisonnement, ignoré) puis {"type": "text", ...}."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return "".join(c.get("text", "") for c in content if c.get("type") == "text")


def _complete(body: dict) -> str:
    for _ in range(5):
        _wait_turn(body)
        r = httpx.post(
            URL, headers={"Authorization": f"Bearer {_api_key()}"}, json=body, timeout=300
        )
        if r.status_code == 200:
            return text_of(r.json()["choices"][0]["message"]["content"])
        if r.status_code == 429 or r.status_code >= 500:
            logger.warning("Mistral %s, nouvel essai dans 60 s", r.status_code)
            time.sleep(60)
            continue
        r.raise_for_status()
    raise RuntimeError("Mistral : échec après 5 essais")


def _cached(
    messages: Messages, temperature: float, model: str, json_mode: bool, effort: str | None = None
) -> str:
    global _cache
    with _lock:
        _cache = _cache or LLMCache()
        prompt = messages[0]["content"] if len(messages) == 1 else json.dumps(messages)
        # JSON : même clé qu'avant (réponses du multi-query déjà en cache) ; texte : clé distincte
        name = model if json_mode else f"{model}:texte"
        name += f":{effort}" if effort else ""  # sans effort : clés déjà en cache inchangées
        key = LLMCache.key(name, temperature, prompt)
        if (cached := _cache.get(key)) is not None:
            return cached
        extra = {"response_format": {"type": "json_object"}} if json_mode else {}
        answer = _complete(_body(messages, temperature, model, effort, **extra))
        if json_mode:
            json.loads(answer)  # lève une erreur si le JSON est invalide : rien en cache
        _cache.put(key, answer)
        return answer


def chat_json(
    prompt: str, temperature: float = 0.0, model: str | None = None, effort: str | None = None
) -> dict | list:
    """Réponse JSON du modèle (en cache si déjà demandée)."""
    model = model or get_app_config().llm.chat_model
    messages = [{"role": "user", "content": prompt}]
    return json.loads(_cached(messages, temperature, model, json_mode=True, effort=effort))


def chat_text(
    messages: Messages,
    temperature: float = 0.0,
    model: str | None = None,
    effort: str | None = None,
) -> str:
    """Réponse texte du modèle (en cache si déjà demandée)."""
    model = model or get_app_config().llm.chat_model
    return _cached(messages, temperature, model, json_mode=False, effort=effort)


def stream_text(
    messages: Messages,
    temperature: float = 0.0,
    model: str | None = None,
    effort: str | None = None,
) -> Iterator[str]:
    """Réponse texte morceau par morceau (Server-Sent Events de l'API Mistral), sans cache."""
    model = model or get_app_config().llm.chat_model
    body = _body(messages, temperature, model, effort, stream=True)
    with _lock:
        _wait_turn(body)
    headers = {"Authorization": f"Bearer {_api_key()}"}
    with httpx.stream("POST", URL, headers=headers, json=body, timeout=300) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            delta = text_of(json.loads(line[6:])["choices"][0]["delta"].get("content"))
            if delta:
                yield delta
