"""Fabrique des modèles Mistral (chat + embeddings) via LangChain.

Les imports sont faits à l'intérieur des fonctions pour que l'application
démarre (et que les tests passent) même sans clé API.
"""

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel

from app.core.config import get_app_config, get_settings


class MissingAPIKeyError(RuntimeError):
    pass


def _api_key() -> str:
    key = get_settings().mistral_api_key
    if not key:
        raise MissingAPIKeyError("MISTRAL_API_KEY manquante : ajoute-la dans le fichier .env")
    return key


def get_chat_model(model: str | None = None, temperature: float | None = None) -> BaseChatModel:
    from langchain_mistralai import ChatMistralAI

    cfg = get_app_config().llm
    return ChatMistralAI(
        model=model or cfg.chat_model,
        temperature=cfg.temperature if temperature is None else temperature,
        max_tokens=cfg.max_tokens,
        api_key=_api_key(),
    )


def get_embeddings() -> Embeddings:
    from langchain_mistralai import MistralAIEmbeddings

    cfg = get_app_config().embeddings
    return MistralAIEmbeddings(model=cfg.model, api_key=_api_key())
