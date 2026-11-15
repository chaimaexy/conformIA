"""Worker Celery des analyses : une tâche à la fois (2 requêtes/min chez Mistral).

celery -A app.workers.celery_app worker --concurrency=1 --loglevel=info
# sous Windows, sans Docker : ajouter --pool=solo (Celery n'y gère pas les processus fils)
"""

import logging
from functools import lru_cache

from celery import Celery

from app.core.config import get_settings

logger = logging.getLogger(__name__)

celery_app = Celery("conformia", broker=get_settings().redis_url or "memory://")
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    worker_concurrency=1,  # une analyse à la fois : la limite de débit est celle du compte
    worker_prefetch_multiplier=1,  # ne réserve pas la tâche suivante pendant la première
    task_acks_late=True,  # si le worker tombe pendant une analyse, la tâche est rejouée
    broker_connection_retry_on_startup=True,
)


@lru_cache
def _runtime():
    """Service d'analyse et événements Redis, créés une fois par processus worker."""
    import redis

    from app.agents.graph import AnalysisService
    from app.agents.queue import RedisEvents

    return AnalysisService(), RedisEvents(redis.Redis.from_url(get_settings().redis_url))


@celery_app.task(name="analyses.executer")
def execute(job: dict) -> None:
    from app.agents.queue import run_job

    service, events = _runtime()
    run_job(service, events, job)
