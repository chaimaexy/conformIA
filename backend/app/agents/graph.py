"""Agent d'analyse de conformité : graphe LangGraph à étapes fixes.

    profil ─> [questions : pause, l'utilisateur répond] ─> interdictions ─> haut_risque
           ─> obligations ─> rgpd ─> rapport

L'ordre des étapes est décidé par le code (procédure de conformité connue d'avance), pas par le
LLM : le trajet est reproductible et relisible par un juriste, et le nombre d'appels LLM est borné
(7 dans le cas général, au plus 10 avec les nouveaux essais de citation). Avec 2 requêtes par
minute en offre gratuite, un agent libre (ReAct) en ferait 15 à 30.

L'état est sauvegardé à chaque étape (checkpointer SQLite, Postgres en phase 7) : une analyse
peut attendre les réponses de l'utilisateur, puis reprendre là où elle s'était arrêtée.
"""

import logging
import operator
import sqlite3
import uuid
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Annotated, TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from app.agents import checklist, obligations, report
from app.agents.classification import (
    categories,
    classify_from_profile,
    classify_high_risk,
    classify_prohibited,
)
from app.agents.profile import (
    EXTRACTION_PROMPT,
    Profile,
    apply_answers,
    missing_fields,
    parse_profile,
    questions_for,
)
from app.core.config import DATA_DIR, get_app_config

CHECKPOINTS = DATA_DIR / "cache" / "analyses.sqlite"
MAX_TOKENS = 4096  # 4 thèmes CNIL (constat, actions, citation) dépassent 1 024 tokens
logger = logging.getLogger(__name__)
STEPS = {
    "profil": "Profil du projet",
    "questions": "Réponses intégrées au profil",
    "interdictions": "Pratiques interdites (article 5)",
    "haut_risque": "Systèmes à haut risque (article 6 et annexe III)",
    "obligations": "Obligations et dates",
    "rgpd": "Liste de vérification de la CNIL",
    "rapport": "Rapport",
}
NOT_HIGH_RISK = {
    "verdict": "non",
    "point": None,
    "citation": "",
    "citation_verifiee": False,
    "point_annonce": None,
    "nouvel_essai": False,
    "raison": "Non examiné : pratique interdite.",
    "derogation": "non",
    "condition": None,
    "citation_derogation": None,
    "information_manquante": None,
}


class AnalysisState(TypedDict, total=False):
    description: str
    reponses_initiales: dict  # réponses déjà connues (formulaire) : pas de question pour elles
    profil: dict
    manquants: list[str]
    interdit: dict
    haut_risque: dict
    classification: dict
    obligations: list[dict]
    note_111: dict | None
    rgpd: list[dict]
    rapport: dict
    markdown: str
    appels_llm: Annotated[int, operator.add]


def default_ask() -> Callable[[str], dict]:
    from app.llm.json_chat import chat_json

    cfg = get_app_config().llm
    effort = cfg.reasoning_effort.get(cfg.chat_model)

    def ask(prompt: str) -> dict:
        # JSON invalide (réponse coupée, virgule manquante) : rien n'est mis en cache, un nouvel
        # essai ; puis réponse vide, que chaque étape traite comme « à vérifier »
        for _ in range(2):
            try:
                data = chat_json(prompt, model=cfg.chat_model, effort=effort, max_tokens=MAX_TOKENS)
            except ValueError:
                logger.warning("Réponse JSON invalide, nouvel essai")
                continue
            return as_object(data)
        return {}

    return ask


def as_object(data) -> dict:
    """Le petit modèle ne respecte pas toujours la forme demandée : il enveloppe l'objet dans
    une liste ([{"conditions": ...}]) ou renvoie la liste seule ([{"id": "f.1", ...}, ...]).
    Une liste d'un seul objet est déballée ; une liste d'objets devient {"elements": [...]},
    que les étapes lisent à la place de la clé attendue."""
    if isinstance(data, list) and len(data) == 1 and isinstance(data[0], dict):
        data = data[0]
    if isinstance(data, list) and data and all(isinstance(d, dict) for d in data):
        return {"elements": data}
    return data if isinstance(data, dict) else {}


class Counter:
    """Compte les appels LLM d'une étape (pour l'état et l'évaluation)."""

    def __init__(self, ask: Callable[[str], dict]):
        self.ask, self.calls = ask, 0

    def __call__(self, prompt: str) -> dict:
        self.calls += 1
        return self.ask(prompt)


class Sources:
    """Textes exacts lus par l'agent (par numéro de section, sans recherche)."""

    def __init__(self, retriever=None):
        self._retriever = retriever

    @property
    def retriever(self):
        if self._retriever is None:
            from app.rag.service import default_retriever

            self._retriever = default_retriever()
        return self._retriever

    def text(self, source_id: str, *labels: str) -> str:
        parts = [c["content"] for label in labels for c in self.retriever.section(source_id, label)]
        return "\n\n".join(parts)

    def checklist_themes(self) -> dict[int, dict]:
        """Thèmes 1 à 11 de la liste de vérification de la CNIL."""
        document = self.retriever.documents.get("cnil_ia")
        themes: dict[int, dict] = {}
        for (doc, label), parts in self.retriever.section_parts.items():
            meta = parts[0]["metadata"]
            if doc != document or "Liste de vérification" not in meta["header"]:
                continue
            number = int(meta.get("number") or 0)
            if number:
                title = label.split("« ", 1)[-1].rstrip(" »")
                text = "\n".join(p["content"] for p in parts)
                themes[number] = {"titre": title, "texte": text, "url": meta.get("url")}
        return dict(sorted(themes.items()))


def build_graph(sources: Sources, ask: Callable[[str], dict], checkpointer=None):
    def profil(state: AnalysisState) -> dict:
        count = Counter(ask)
        data = count(EXTRACTION_PROMPT.format(description=state["description"]))
        profile = parse_profile(state["description"], data)
        profile = apply_answers(profile, state.get("reponses_initiales") or {})
        return {
            "profil": profile.model_dump(),
            "manquants": missing_fields(profile),
            "appels_llm": count.calls,
        }

    def questions(state: AnalysisState) -> dict:
        answers = interrupt({"questions": questions_for(state["manquants"])})
        profile = apply_answers(Profile.model_validate(state["profil"]), answers or {})
        return {"profil": profile.model_dump(), "manquants": missing_fields(profile)}

    def interdictions(state: AnalysisState) -> dict:
        count = Counter(ask)
        profile = Profile.model_validate(state["profil"])
        result = classify_prohibited(profile, sources.text("ai_act", "Article 5"), count)
        return {"interdit": result, "appels_llm": count.calls}

    def haut_risque(state: AnalysisState) -> dict:
        count = Counter(ask)
        profile = Profile.model_validate(state["profil"])
        text = sources.text("ai_act", "Article 6", "Annexe III")
        return {"haut_risque": classify_high_risk(profile, text, count), "appels_llm": count.calls}

    def obligations_step(state: AnalysisState) -> dict:
        profile = Profile.model_validate(state["profil"])
        high_risk = state.get("haut_risque") or NOT_HIGH_RISK
        simple = classify_from_profile(profile)
        derived = categories(profile, state["interdit"], high_risk, simple)
        classification = {
            **derived,
            "interdit": state["interdit"],
            "annexe_iii": high_risk,
            **simple,
        }
        return {
            "haut_risque": high_risk,
            "classification": classification,
            "obligations": obligations.select(profile, classification),
            "note_111": obligations.existing_high_risk_note(profile, classification),
        }

    def rgpd(state: AnalysisState) -> dict:
        count = Counter(ask)
        profile = Profile.model_validate(state["profil"])
        results, _ = checklist.evaluate(profile, sources.checklist_themes(), count)
        return {"rgpd": results, "appels_llm": count.calls}

    def rapport(state: AnalysisState) -> dict:
        count = Counter(ask)
        built = report.build(
            Profile.model_validate(state["profil"]),
            state["manquants"],
            state["classification"],
            state["obligations"],
            state.get("note_111"),
            state["rgpd"],
        )
        built["synthese"], built["synthese_origine"] = report.synthesize(built, count)
        built["appels_llm"] = state.get("appels_llm", 0) + count.calls
        return {"rapport": built, "markdown": report.to_markdown(built), "appels_llm": count.calls}

    graph = StateGraph(AnalysisState)
    for name, fn in [
        ("profil", profil),
        ("questions", questions),
        ("interdictions", interdictions),
        ("haut_risque", haut_risque),
        ("obligations", obligations_step),
        ("rgpd", rgpd),
        ("rapport", rapport),
    ]:
        graph.add_node(name, fn)
    graph.add_edge(START, "profil")
    graph.add_conditional_edges(
        "profil", lambda s: "questions" if s["manquants"] else "interdictions"
    )
    graph.add_edge("questions", "interdictions")
    graph.add_conditional_edges(
        "interdictions",
        lambda s: "obligations" if s["interdit"]["verdict"] == "oui" else "haut_risque",
    )
    graph.add_edge("haut_risque", "obligations")
    graph.add_edge("obligations", "rgpd")
    graph.add_edge("rgpd", "rapport")
    graph.add_edge("rapport", END)
    return graph.compile(checkpointer=checkpointer)


class AnalysisService:
    """Lance, reprend et relit les analyses ; produit les événements envoyés au client."""

    def __init__(self, sources: Sources | None = None, ask=None, db_path: Path = CHECKPOINTS):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(db_path, check_same_thread=False)
        self.graph = build_graph(sources or Sources(), ask or default_ask(), SqliteSaver(conn))

    @staticmethod
    def _config(analysis_id: str) -> dict:
        return {"configurable": {"thread_id": analysis_id}}

    def start(self, description: str, answers: dict | None = None) -> Iterator[dict]:
        analysis_id = uuid.uuid4().hex
        state = {"description": description, "reponses_initiales": answers or {}}
        yield {"type": "debut", "analyse_id": analysis_id}
        yield from self._run(analysis_id, state)

    def resume(self, analysis_id: str, answers: dict) -> Iterator[dict]:
        if not self.waiting(analysis_id):
            yield {"type": "erreur", "message": "Cette analyse n'attend pas de réponse."}
            return
        yield from self._run(analysis_id, Command(resume=answers))

    def waiting(self, analysis_id: str) -> bool:
        return bool(self.graph.get_state(self._config(analysis_id)).interrupts)

    def get(self, analysis_id: str) -> dict | None:
        snapshot = self.graph.get_state(self._config(analysis_id))
        if not snapshot.values:
            return None
        values = snapshot.values
        if snapshot.interrupts:
            status = "questions"
        elif values.get("rapport"):
            status = "termine"
        else:
            status = "en_cours"
        return {
            "analyse_id": analysis_id,
            "statut": status,
            "questions": snapshot.interrupts[0].value["questions"] if snapshot.interrupts else [],
            "rapport": values.get("rapport"),
            "markdown": values.get("markdown"),
        }

    def _run(self, analysis_id: str, payload) -> Iterator[dict]:
        config = self._config(analysis_id)
        for update in self.graph.stream(payload, config, stream_mode="updates"):
            for node, values in update.items():
                if node == "__interrupt__":
                    yield {
                        "type": "questions",
                        "analyse_id": analysis_id,
                        "questions": values[0].value["questions"],
                    }
                    return
                yield {"type": "etape", "etape": node, "libelle": STEPS.get(node, node)}
                if node == "rapport":
                    yield {
                        "type": "rapport",
                        "analyse_id": analysis_id,
                        "rapport": values["rapport"],
                        "markdown": values["markdown"],
                    }
