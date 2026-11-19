import logging
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.api.routes import analyses, chat, conversations, health
from app.core.config import get_settings
from app.core.logging import setup_logging

logger = logging.getLogger(__name__)


def interrupt_unfinished() -> None:
    """Sans Redis, les analyses tournent dans ce processus : celles du précédent sont perdues."""
    if get_settings().redis_url:
        return  # file Redis : le worker les reprend, l'API ne les exécute pas
    try:
        if n := analyses.get_records().interrupt_unfinished():
            logger.warning("%d analyse(s) interrompue(s) par le redémarrage : mises en erreur", n)
    except Exception:
        logger.exception("Analyses interrompues : vérification impossible")


def warm_up() -> None:
    """Charge les passages (Supabase), l'index BM25 et l'agent avant la première requête.

    Environ 20 secondes ; en arrière-plan, pour que /health réponde tout de suite. Sans cela,
    la première question après un réveil du serveur attendrait ce chargement.
    """
    try:
        from app.rag.service import default_retriever

        retriever = default_retriever()
        _ = retriever.bm25, retriever.section_parts, retriever.documents, retriever.section_heads
        chat.get_service()
        analyses.get_service()  # graphe et tables de LangGraph (pas la file : une seule)
        logger.info("Préchargement terminé : %d passages", len(retriever.chunks))
    except Exception:
        logger.exception("Préchargement en échec : chargement à la première requête")


@asynccontextmanager
async def lifespan(app: FastAPI):
    interrupt_unfinished()
    if get_settings().environment != "local":
        threading.Thread(target=warm_up, name="prechargement", daemon=True).start()
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    setup_logging(settings.log_level)

    app = FastAPI(
        title="ConformIA API",
        description="Copilote RAG de conformité IA : AI Act, RGPD, CNIL.",
        version=__version__,
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(health.router)
    app.include_router(chat.router)
    app.include_router(conversations.router)
    app.include_router(analyses.router)
    return app


app = create_app()
