"""Analyses de conformité d'un projet (agent LangGraph), passées par une file d'attente.

- POST /analyses                    : lance une analyse à partir d'une description libre ;
- POST /analyses/document           : lance une analyse à partir d'un PDF du projet (cahier des
  charges, documentation, analyse d'impact). Le PDF est lu tout de suite, en mémoire, et n'est
  pas conservé ; un fichier refusé (pas un PDF, protégé, scanné, trop long) l'est aussitôt ;
- GET  /analyses/{id}/flux          : événements de l'analyse, en Server-Sent Events ;
- POST /analyses/{id}/reponses      : répond aux questions posées, l'analyse reprend ;
- GET  /analyses/{id}               : état de l'analyse (en attente, questions ou rapport) ;
- GET  /analyses                    : analyses de l'utilisateur, les plus récentes d'abord ;
- DELETE /analyses/{id}             : efface l'analyse et son état (droit à l'effacement).

Chaque route demande une session (jeton Supabase, voir app/core/auth.py). Une analyse n'existe
que pour son auteur : pour les autres, 404. Quota : analyses par 24 h et par utilisateur (429,
avec l'heure à laquelle une place se libère).

Les trois POST répondent tout de suite (202) avec l'identifiant, la position dans la file et
l'adresse du flux. Événements du flux (`data: {...}`) : en_attente (position) -> debut -> etape
(une par étape terminée) -> questions (l'analyse attend : envoyer les réponses) ou rapport (fin,
JSON et Markdown) ou erreur. Une analyse prend 2 à 4 minutes une fois son tour venu.
"""

import json
import time
import uuid
from datetime import timedelta
from functools import lru_cache
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.agents.document import MAX_BYTES, DocumentError, read_pdf
from app.agents.graph import AnalysisService, default_checkpointer
from app.agents.queue import TERMINAL, AnalysisQueue, local_queue, redis_queue
from app.core.auth import CurrentUser
from app.core.config import get_app_config, get_settings
from app.storage.db import get_database
from app.storage.records import AnalysisRecords, QuotaExceeded, check_quota

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
    return AnalysisService(checkpointer=default_checkpointer())


@lru_cache
def get_records() -> AnalysisRecords:
    return AnalysisRecords(get_database())


@lru_cache
def get_queue() -> AnalysisQueue:
    url = get_settings().redis_url
    return redis_queue(url) if url else local_queue(get_service(), get_records())


def _accepted(ticket: dict, **extra) -> dict:
    flux = f"/analyses/{ticket['analyse_id']}/flux?depuis={ticket['depuis']}"
    return {**ticket, "flux": flux, **extra}


def _owned(analysis_id: str, user) -> dict:
    record = get_records().get(analysis_id, user.id)
    if record is None:  # inconnue, ou d'un autre utilisateur : même réponse
        raise HTTPException(404, "Analyse inconnue")
    return record


def _check_quota(user) -> None:
    limit = get_app_config().quotas.analyses_par_jour
    try:
        check_quota(
            get_records().count_since, user.id, limit, timedelta(days=1), "analyses par 24 h"
        )
    except QuotaExceeded as exc:
        raise HTTPException(
            429, exc.detail(), headers={"Retry-After": str(exc.retry_after_s())}
        ) from exc


def _enqueue(user, text: str, answers: dict, titre: str, document: str | None = None) -> dict:
    analysis_id = uuid.uuid4().hex
    source = "document" if document else "description"
    get_records().create(analysis_id, user.id, titre, source)  # avant la file : le worker la suit
    return get_queue().start(text, answers, document=document, analysis_id=analysis_id)


def _resume_point(analysis_id: str) -> int:
    """Où relire le flux : après le dernier événement terminal (des questions déjà répondues,
    par exemple). Relu depuis 0, le flux s'arrêterait sur ces anciennes questions."""
    events = get_queue().events.read(analysis_id, 0)
    terminal = [i for i, e in enumerate(events) if e["type"] in TERMINAL]
    if terminal and terminal[-1] == len(events) - 1:  # l'analyse attend ou est finie
        return terminal[-1]
    return terminal[-1] + 1 if terminal else 0


def _finished(analysis_id: str) -> bool:
    """Dernier événement terminal (ou aucun) : l'analyse attend des réponses ou est finie."""
    events = get_queue().events
    count = events.count(analysis_id)
    last = events.read(analysis_id, count - 1) if count else []
    return not last or last[0]["type"] in TERMINAL


@router.get("")
def list_analyses(user: CurrentUser) -> list[dict]:
    return get_records().of_owner(user.id)


@router.post("", status_code=202)
def start(request: AnalysisRequest, user: CurrentUser) -> dict:
    _check_quota(user)
    ticket = _enqueue(user, request.description, request.reponses, request.description)
    return _accepted(ticket)


@router.post("/document", status_code=202)
def start_from_document(
    fichier: Annotated[UploadFile, File(description="PDF du projet, 30 pages et 10 Mo au plus")],
    user: CurrentUser,
    reponses: Annotated[str, Form(description="réponses déjà connues, en JSON")] = "{}",
) -> dict:
    _check_quota(user)
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
    ticket = _enqueue(user, document.text(), answers, document.name, document=document.name)
    info = {
        "nom": document.name,
        "pages": len(document.pages),
        "pages_sans_texte": document.empty_pages,  # images ou scans : non lues
    }
    return _accepted(ticket, document=info)


@router.get("/{analysis_id}/flux")
def stream(analysis_id: str, user: CurrentUser, depuis: int = 0) -> StreamingResponse:
    _owned(analysis_id, user)
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
def answer(analysis_id: str, request: AnswersRequest, user: CurrentUser) -> dict:
    _owned(analysis_id, user)
    service = get_service()
    # réponses déjà envoyées : la reprise est dans la file, l'état ne l'a pas encore vue
    if not service.waiting(analysis_id) or not _finished(analysis_id):
        raise HTTPException(409, "Cette analyse n'attend pas de réponse")
    get_records().set_status(analysis_id, "en_attente")
    return _accepted(get_queue().resume(analysis_id, request.reponses))


@router.get("/{analysis_id}")
def status(analysis_id: str, user: CurrentUser) -> dict:
    record = _owned(analysis_id, user)
    found = {
        "titre": record["titre"],
        "cree_le": record["created_at"],
        "flux": f"/analyses/{analysis_id}/flux?depuis={_resume_point(analysis_id)}",
    }
    result = get_service().get(analysis_id)
    if result is not None:
        if record["statut"] == "erreur":  # l'état de LangGraph ne garde pas la panne
            result["statut"] = "erreur"
        return {**result, **found}
    statut = "erreur" if record["statut"] == "erreur" else "en_attente"
    return {  # déposée, pas encore commencée (ou tombée avant la première étape)
        "analyse_id": analysis_id,
        "statut": statut,
        **found,
        "questions": [],
        "rapport": None,
    }


@router.delete("/{analysis_id}", status_code=204)
def delete(analysis_id: str, user: CurrentUser) -> None:
    _owned(analysis_id, user)
    if not _finished(analysis_id):
        raise HTTPException(409, "Analyse en cours : attendez ses questions ou son rapport")
    get_service().delete(analysis_id)
    get_records().delete(analysis_id, user.id)
