"""Analyses de conformité d'un projet (agent LangGraph), en Server-Sent Events.

- POST /analyses                    : lance une analyse à partir d'une description libre
  (et, si on les connaît déjà, des réponses aux questions fermées) ;
- POST /analyses/{id}/reponses      : répond aux questions posées, l'analyse reprend ;
- GET  /analyses/{id}               : état de l'analyse (questions en attente ou rapport).

Événements (`data: {...}`) : debut -> etape (une par étape terminée) -> questions (l'analyse
attend : renvoyer les réponses) ou rapport (fin, avec le rapport JSON et sa version Markdown).
Avec 2 requêtes par minute (offre gratuite de Mistral), une analyse prend 2 à 4 minutes.
"""

import json
from functools import lru_cache

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.agents.graph import AnalysisService

router = APIRouter(prefix="/analyses", tags=["analyses"])


class AnalysisRequest(BaseModel):
    description: str = Field(min_length=20, max_length=5000)
    reponses: dict[str, str | list[str]] = Field(default_factory=dict)


class AnswersRequest(BaseModel):
    reponses: dict[str, str | list[str]]


@lru_cache
def get_service() -> AnalysisService:
    return AnalysisService()


def _sse(events) -> StreamingResponse:
    def body():
        for event in events:
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(body(), media_type="text/event-stream")


@router.post("")
def start(request: AnalysisRequest) -> StreamingResponse:
    return _sse(get_service().start(request.description, request.reponses))


@router.post("/{analysis_id}/reponses")
def answer(analysis_id: str, request: AnswersRequest) -> StreamingResponse:
    service = get_service()
    if service.get(analysis_id) is None:
        raise HTTPException(404, "Analyse inconnue")
    if not service.waiting(analysis_id):
        raise HTTPException(409, "Cette analyse n'attend pas de réponse")
    return _sse(service.resume(analysis_id, request.reponses))


@router.get("/{analysis_id}")
def status(analysis_id: str) -> dict:
    result = get_service().get(analysis_id)
    if result is None:
        raise HTTPException(404, "Analyse inconnue")
    return result
