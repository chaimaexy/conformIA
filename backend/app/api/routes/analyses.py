"""Analyses de conformité d'un projet (agent LangGraph), passées par une file d'attente.

- POST /analyses                    : lance une analyse à partir d'une description libre ;
- POST /analyses/document           : lance une analyse à partir d'un PDF du projet (cahier des
  charges, documentation, analyse d'impact). Le PDF est lu tout de suite, en mémoire, et n'est
  pas conservé ; un fichier refusé (pas un PDF, protégé, scanné, trop long) l'est aussitôt ;
- GET  /analyses/{id}/flux          : événements de l'analyse, en Server-Sent Events ;
- POST /analyses/{id}/reponses      : répond aux questions posées, l'analyse reprend ;
- GET  /analyses/{id}               : état de l'analyse (en attente, questions ou rapport).

Les trois POST répondent tout de suite (202) avec l'identifiant, la position dans la file et
l'adresse du flux. Événements du flux (`data: {...}`) : en_attente (position) -> debut -> etape
(une par étape terminée) -> questions (l'analyse attend : envoyer les réponses) ou rapport (fin,
JSON et Markdown) ou erreur. Une analyse prend 2 à 4 minutes une fois son tour venu.
"""

import json
import time
from functools import lru_cache
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.agents.document import MAX_BYTES, DocumentError, read_pdf
from app.agents.graph import AnalysisService
from app.agents.queue import TERMINAL, AnalysisQueue, local_queue, redis_queue
from app.core.config import get_settings

router = APIRouter(prefix="/analyses", tags=["analyses"])
POLL_S = 1.0
STREAM_TIMEOUT_S = 30 * 60  # file + analyse : au-delà, le client relance le flux


class AnalysisRequest(BaseModel):
    description: str = Field(min_length=20, max_length=5000)
    reponses: dict[str, str | list[str]] = Field(default_factory=dict)


class AnswersRequest(BaseModel):
    reponses: dict[str, str | list[str]]


@lru_cache
def get_service() -> AnalysisService:
    return AnalysisService()


@lru_cache
def get_queue() -> AnalysisQueue:
    url = get_settings().redis_url
    return redis_queue(url) if url else local_queue(get_service())


def _accepted(ticket: dict, **extra) -> dict:
    flux = f"/analyses/{ticket['analyse_id']}/flux?depuis={ticket['depuis']}"
    return {**ticket, "flux": flux, **extra}


@router.post("", status_code=202)
def start(request: AnalysisRequest) -> dict:
    return _accepted(get_queue().start(request.description, request.reponses))


@router.post("/document", status_code=202)
def start_from_document(
    fichier: Annotated[UploadFile, File(description="PDF du projet, 30 pages et 10 Mo au plus")],
    reponses: Annotated[str, Form(description="réponses déjà connues, en JSON")] = "{}",
) -> dict:
    data = fichier.file.read(MAX_BYTES + 1)  # au-delà de la limite, inutile de tout lire
    try:
        document = read_pdf(data, fichier.filename or "document.pdf")
        answers = json.loads(reponses or "{}")
    except DocumentError as exc:
        raise HTTPException(422, str(exc)) from exc
    except json.JSONDecodeError as exc:
        raise HTTPException(422, "Le champ « reponses » doit être du JSON.") from exc
    if not isinstance(answers, dict):
        raise HTTPException(422, "Le champ « reponses » doit être un objet JSON.")
    ticket = get_queue().start(document.text(), answers, document=document.name)
    info = {
        "nom": document.name,
        "pages": len(document.pages),
        "pages_sans_texte": document.empty_pages,  # images ou scans : non lues
    }
    return _accepted(ticket, document=info)


@router.get("/{analysis_id}/flux")
def stream(analysis_id: str, depuis: int = 0) -> StreamingResponse:
    queue = get_queue()
    if not queue.events.count(analysis_id):
        raise HTTPException(404, "Analyse inconnue")

    def body():
        position, deadline = max(depuis, 0), time.monotonic() + STREAM_TIMEOUT_S
        while time.monotonic() < deadline:
            events = queue.events.read(analysis_id, position)
            for event in events:
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                if event["type"] in TERMINAL:
                    return
            position += len(events)
            time.sleep(POLL_S)
        timeout = {"type": "erreur", "message": "Délai dépassé : relancez le flux."}
        yield f"data: {json.dumps(timeout, ensure_ascii=False)}\n\n"

    return StreamingResponse(body(), media_type="text/event-stream")


@router.post("/{analysis_id}/reponses", status_code=202)
def answer(analysis_id: str, request: AnswersRequest) -> dict:
    service = get_service()
    if service.get(analysis_id) is None:
        raise HTTPException(404, "Analyse inconnue")
    queue = get_queue()
    last = (queue.events.read(analysis_id, max(queue.events.count(analysis_id) - 1, 0)) or [{}])[0]
    # réponses déjà envoyées : la reprise est dans la file, l'état ne l'a pas encore vue
    if not service.waiting(analysis_id) or (last and last["type"] not in TERMINAL):
        raise HTTPException(409, "Cette analyse n'attend pas de réponse")
    return _accepted(queue.resume(analysis_id, request.reponses))


@router.get("/{analysis_id}")
def status(analysis_id: str) -> dict:
    result = get_service().get(analysis_id)
    if result is not None:
        return result
    if get_queue().events.count(analysis_id):  # déposée, pas encore commencée
        return {"analyse_id": analysis_id, "statut": "en_attente", "questions": [], "rapport": None}
    raise HTTPException(404, "Analyse inconnue")
