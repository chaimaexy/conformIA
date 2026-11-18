"""Historique du chat de l'utilisateur connecté.

- GET    /conversations       : ses conversations, la plus récente d'abord ;
- GET    /conversations/{id}  : les messages d'une conversation (avec les sources des réponses) ;
- DELETE /conversations/{id}  : efface la conversation et ses messages (droit à l'effacement).

Une conversation n'existe que pour son auteur : pour les autres, 404.
"""

from fastapi import APIRouter, HTTPException

from app.api.routes import chat
from app.core.auth import CurrentUser

router = APIRouter(prefix="/conversations", tags=["conversations"])


@router.get("")
def list_conversations(user: CurrentUser) -> list[dict]:
    return chat.get_conversations().of_owner(user.id)


@router.get("/{conversation_id}")
def read(conversation_id: str, user: CurrentUser) -> dict:
    conversations = chat.get_conversations()
    if not conversations.owned(conversation_id, user.id):
        raise HTTPException(404, "Conversation inconnue")
    return {"id": conversation_id, "messages": conversations.messages(conversation_id)}


@router.delete("/{conversation_id}", status_code=204)
def delete(conversation_id: str, user: CurrentUser) -> None:
    if not chat.get_conversations().delete(conversation_id, user.id):
        raise HTTPException(404, "Conversation inconnue")
