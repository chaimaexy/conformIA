"""Questions-réponses sourcées.

- POST /chat         : réponse complète en JSON ;
- POST /chat/stream  : même réponse en Server-Sent Events (`data: {...}` par événement) :
    statut (recherche, redaction) -> sources -> texte (morceaux) -> fin.
  L'événement « fin » contient la réponse définitive et le résultat de la vérification des
  citations ; s'il indique `refused: true`, l'interface remplace le texte reçu par `answer`.

Chaque question appartient à une conversation de l'utilisateur connecté (historique dans la
base, voir /conversations). Sans `session_id`, une conversation est créée ; renvoyer le
`session_id` reçu pour la poursuivre (« et pour les PME ? »). Quota : questions par heure et
par utilisateur (429).
"""

import json
from datetime import timedelta
from functools import lru_cache

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.api.sse import SSE_HEADERS
from app.core.auth import CurrentUser, User
from app.core.config import get_app_config
from app.rag.service import ChatService
from app.storage.db import get_database
from app.storage.records import Conversations, QuotaExceeded, check_quota

router = APIRouter(prefix="/chat", tags=["chat"])


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    session_id: str | None = Field(default=None, max_length=64)


@lru_cache
def get_conversations() -> Conversations:
    return Conversations(get_database())


@lru_cache
def get_service() -> ChatService:
    return ChatService(memory=get_conversations())


def conversation_for(request: ChatRequest, user: User) -> str:
    """Conversation de l'utilisateur (créée au premier message), après contrôle du quota."""
    conversations = get_conversations()
    if request.session_id and not conversations.owned(request.session_id, user.id):
        raise HTTPException(404, "Conversation inconnue")
    limit = get_app_config().quotas.questions_par_heure
    try:
        check_quota(
            conversations.questions_since, user.id, limit, timedelta(hours=1), "questions par heure"
        )
    except QuotaExceeded as exc:
        raise HTTPException(
            429, exc.detail(), headers={"Retry-After": str(exc.retry_after_s())}
        ) from exc
    return request.session_id or conversations.create(user.id, request.message)


@router.post("")
def chat(request: ChatRequest, user: CurrentUser) -> dict:
    session_id = conversation_for(request, user)
    return get_service().answer(request.message, session_id).to_dict()


@router.post("/stream")
def chat_stream(request: ChatRequest, user: CurrentUser) -> StreamingResponse:
    session_id = conversation_for(request, user)  # avant le flux : une erreur reste une erreur HTTP

    def events():
        for event in get_service().stream(request.message, session_id):
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream", headers=SSE_HEADERS)
