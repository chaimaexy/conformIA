"""Questions-réponses sourcées.

- POST /chat         : réponse complète en JSON ;
- POST /chat/stream  : même réponse en Server-Sent Events (`data: {...}` par événement) :
    statut (recherche, redaction) -> sources -> texte (morceaux) -> fin.
  L'événement « fin » contient la réponse définitive et le résultat de la vérification des
  citations ; s'il indique `refused: true`, l'interface remplace le texte reçu par `answer`.
Envoyer le `session_id` reçu pour poursuivre la conversation (« et pour les PME ? »).
"""

import json
from functools import lru_cache

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.rag.service import ChatService

router = APIRouter(prefix="/chat", tags=["chat"])


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    session_id: str | None = Field(default=None, max_length=64)


@lru_cache
def get_service() -> ChatService:
    return ChatService()


@router.post("")
def chat(request: ChatRequest) -> dict:
    return get_service().answer(request.message, request.session_id).to_dict()


@router.post("/stream")
def chat_stream(request: ChatRequest) -> StreamingResponse:
    def events():
        for event in get_service().stream(request.message, request.session_id):
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")
