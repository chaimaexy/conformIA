"""File des analyses : une seule à la fois.

L'offre gratuite de Mistral permet 2 requêtes par minute pour tout le compte. Deux analyses
lancées en même temps se gêneraient (erreurs 429, nouveaux essais) ; en file, chacune passe à son
tour et l'utilisateur voit sa position. Une analyse fait environ 7 appels, soit 2 à 4 minutes.

- Avec REDIS_URL (docker compose) : l'API dépose la tâche dans Celery ; un worker qui ne traite
  qu'une tâche à la fois l'exécute. Les événements (étapes, questions, rapport) passent par une
  liste Redis, que l'API relit pour le flux SSE du client.
- Sans Redis (poste de développement, tests) : la même file dans un fil d'exécution de l'API,
  événements en mémoire.
"""

import json
import logging
import threading
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)
TERMINAL = ("questions", "rapport", "erreur")  # le flux s'arrête : l'analyse attend ou est finie
EVENTS_TTL_S = 24 * 3600  # les événements d'une analyse sont gardés un jour
PENDING_KEY = "conformia:analyses:en_attente"
FAILURE = "L'analyse a échoué. Réessayez dans quelques minutes."


class MemoryEvents:
    """Événements et file d'attente en mémoire (un seul processus)."""

    def __init__(self):
        self._events: dict[str, list[dict]] = {}
        self._pending: list[str] = []
        self._lock = threading.Lock()

    def push(self, analysis_id: str, event: dict) -> None:
        with self._lock:
            self._events.setdefault(analysis_id, []).append(event)

    def read(self, analysis_id: str, start: int = 0) -> list[dict]:
        with self._lock:
            return list(self._events.get(analysis_id, [])[start:])

    def count(self, analysis_id: str) -> int:
        with self._lock:
            return len(self._events.get(analysis_id, []))

    def wait(self, analysis_id: str) -> int:
        """Ajoute l'analyse à la file ; renvoie sa position (1 : la prochaine)."""
        with self._lock:
            self._pending.append(analysis_id)
            return len(self._pending)

    def started(self, analysis_id: str) -> None:
        with self._lock:
            if analysis_id in self._pending:
                self._pending.remove(analysis_id)


class RedisEvents:
    """Mêmes opérations sur Redis : partagées entre l'API et le worker Celery."""

    def __init__(self, client):
        self.redis = client

    @staticmethod
    def _key(analysis_id: str) -> str:
        return f"conformia:analyses:{analysis_id}:evenements"

    def push(self, analysis_id: str, event: dict) -> None:
        key = self._key(analysis_id)
        self.redis.rpush(key, json.dumps(event, ensure_ascii=False))
        self.redis.expire(key, EVENTS_TTL_S)

    def read(self, analysis_id: str, start: int = 0) -> list[dict]:
        return [json.loads(e) for e in self.redis.lrange(self._key(analysis_id), start, -1)]

    def count(self, analysis_id: str) -> int:
        return self.redis.llen(self._key(analysis_id))

    def wait(self, analysis_id: str) -> int:
        return self.redis.rpush(PENDING_KEY, analysis_id)

    def started(self, analysis_id: str) -> None:
        self.redis.lrem(PENDING_KEY, 0, analysis_id)


def run_job(service, events, job: dict, records=None) -> None:
    """Exécute une analyse (début ou reprise) et publie ses événements. Exécuté par le worker.
    records : table des analyses (statut, copie du rapport), tenue à jour au fil des événements."""
    analysis_id = job["analyse_id"]
    events.started(analysis_id)

    def publish(event: dict) -> None:
        events.push(analysis_id, event)
        if records is not None:
            records.on_event(analysis_id, event)

    try:
        if job["type"] == "debut":
            run = service.start(job["texte"], job.get("reponses"), analysis_id, job.get("document"))
        else:
            if records is not None:
                records.set_status(analysis_id, "en_cours")
            run = service.resume(analysis_id, job.get("reponses") or {})
        for event in run:
            publish(event)
    except Exception:  # panne réseau, quota : l'utilisateur doit le savoir, pas attendre
        logger.exception("Analyse %s en échec", analysis_id)
        publish({"type": "erreur", "analyse_id": analysis_id, "message": FAILURE})


class AnalysisQueue:
    """Dépose les analyses dans la file ; submit(job) les fait exécuter par run_job."""

    def __init__(self, events, submit: Callable[[dict], None]):
        self.events, self._submit = events, submit

    def start(
        self,
        text: str,
        answers: dict | None = None,
        document: str | None = None,
        analysis_id: str | None = None,
    ) -> dict:
        job = {
            "type": "debut",
            "analyse_id": analysis_id or uuid.uuid4().hex,
            "texte": text,
            "reponses": answers or {},
            "document": document,
        }
        return self._enqueue(job)

    def resume(self, analysis_id: str, answers: dict) -> dict:
        return self._enqueue({"type": "reprise", "analyse_id": analysis_id, "reponses": answers})

    def _enqueue(self, job: dict) -> dict:
        analysis_id = job["analyse_id"]
        since = self.events.count(analysis_id)  # le client lit le flux à partir d'ici
        position = self.events.wait(analysis_id)
        self.events.push(
            analysis_id, {"type": "en_attente", "analyse_id": analysis_id, "position": position}
        )
        self._submit(job)
        return {"analyse_id": analysis_id, "position": position, "depuis": since}


def local_queue(service, records=None) -> AnalysisQueue:
    """File sans Redis : un seul fil d'exécution, donc une analyse à la fois."""
    events = MemoryEvents()
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="analyses")

    def submit(job: dict) -> None:
        pool.submit(run_job, service, events, job, records)

    return AnalysisQueue(events, submit)


def redis_queue(url: str) -> AnalysisQueue:
    import redis

    from app.workers.celery_app import execute

    return AnalysisQueue(RedisEvents(redis.Redis.from_url(url)), execute.delay)
