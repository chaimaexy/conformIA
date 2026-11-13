"""Routeur : réécrit la question avec l'historique et détecte l'intention, en UN appel LLM.

Offre gratuite Mistral : 2 requêtes/min. Le routeur n'est donc appelé que lorsqu'il sert :
- s'il y a un historique (« et pour les PME ? » doit être réécrit) ;
- si le message est très court (« bonjour », « merci ») : probablement de la conversation.
Une première question complète va directement à la recherche ; si elle est hors sujet, le
garde-fou du reranker la bloque (score trop bas).
"""

import logging
from dataclasses import dataclass

from app.core.config import get_app_config
from app.llm.json_chat import chat_json
from app.prompts.rag import ROUTER
from app.rag.memory import format_history

logger = logging.getLogger(__name__)
INTENTIONS = ("juridique", "conversation", "hors_sujet")
SHORT_MESSAGE = 3  # mots


@dataclass(frozen=True)
class Route:
    intention: str
    question: str  # question autonome à chercher (vide si pas juridique)
    routed: bool  # le LLM a-t-il été appelé ?


def needs_router(message: str, history: list[dict[str, str]]) -> bool:
    return bool(history) or len(message.split()) <= SHORT_MESSAGE


def route(message: str, history: list[dict[str, str]]) -> Route:
    if not needs_router(message, history):
        return Route("juridique", message, routed=False)
    prompt = ROUTER.format(history=format_history(history), message=message)
    try:
        cfg = get_app_config().llm
        data = chat_json(prompt, effort=cfg.reasoning_effort.get(cfg.chat_model))
    except Exception:  # le routeur ne doit jamais bloquer une question : on la cherche telle quelle
        logger.exception("Routeur indisponible : question cherchée telle quelle")
        return Route("juridique", message, routed=False)
    intention = data.get("intention") if isinstance(data, dict) else None
    if intention not in INTENTIONS:
        intention = "juridique"
    question = str(data.get("question") or "").strip() if isinstance(data, dict) else ""
    if intention == "juridique" and not question:
        question = message
    return Route(intention, question, routed=True)
