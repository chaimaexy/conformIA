"""Chaîne complète d'une question, du message de l'utilisateur à la réponse vérifiée.

message (+ historique)
  ─> routeur (si historique ou message très court) : intention + question autonome
  ─> recherche de production (mistral-embed + BM25, RRF, Cohere sur 50 candidats)
  ─> 4 chunks + sections qu'ils citent (graphe des citations)
  ─> garde-fou sur le score du 1er chunk :
       < seuil de refus          -> « je ne sais pas », sans appel au LLM
       < seuil de vérification   -> zone grise : un appel LLM séparé vérifie d'abord que
                                    les sources répondent (sinon refus)
  ─> réponse Mistral, chaque affirmation citée [n] ; « SOURCES_INSUFFISANTES » -> refus
  ─> vérification par le code : renvois valides, citations exactes, et aucune date ni aucun
     montant absent des sources (sinon avertissement visible dans la réponse)
"""

import logging
import uuid
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from functools import lru_cache

from app.core.config import get_app_config
from app.llm.json_chat import chat_json, chat_text, stream_text
from app.prompts.rag import (
    ANSWER_SYSTEM,
    ANSWER_USER,
    CHECK,
    CONVERSATION,
    GRAY_ZONE,
    INSUFFICIENT,
    REFUSAL,
    UNSOURCED_WARNING,
)
from app.rag.facts import humanize
from app.rag.memory import ConversationStore
from app.rag.router import Route, route
from app.rag.verify import Verification, verify

logger = logging.getLogger(__name__)
SEARCH_DEPTH = 20  # lignes classées gardées : leurs scores ordonnent les sections citées


@dataclass
class Source:
    n: int
    document: str  # « AI Act (règlement (UE) 2024/1689) »
    label: str  # « Article 113 »
    url: str
    score: float | None  # score du reranker ; None pour une section jointe par citation
    cited_by: str | None  # section qui la cite (graphe des citations)
    text: str
    cited: bool = False  # citée dans la réponse

    @classmethod
    def from_row(cls, n: int, row: dict) -> "Source":
        meta = row["metadata"]
        return cls(
            n=n,
            document=(meta.get("header") or "").split(" > ")[0],
            label=meta["label"],
            url=meta["url"],
            score=row.get("score"),
            cited_by=row.get("cited_by"),
            text=row["content"],
        )


@dataclass
class ChatAnswer:
    session_id: str
    question: str
    search_question: str
    intention: str
    zone: str  # conversation | hors_sujet | refus | verification | reponse
    top_score: float | None
    answer: str
    refused: bool
    sources: list[Source] = field(default_factory=list)
    verification: Verification | None = None
    coverage: dict | None = None  # zone grise : avis du contrôle de couverture

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Prepared:
    session_id: str
    message: str
    route: Route
    zone: str
    top_score: float | None = None
    sources: list[Source] = field(default_factory=list)
    messages: list[dict[str, str]] = field(default_factory=list)  # prompt de réponse
    coverage: dict | None = None  # zone grise : {"couvert": bool, "raison": str}

    @property
    def not_covered(self) -> bool:
        return self.coverage is not None and self.coverage.get("couvert") is False


def is_refusal(text: str | None) -> bool:
    """Le modèle refuse par le marqueur SEUL, en tête de réponse. Le même mot plus loin dans
    une réponse (« sur ce point : SOURCES_INSUFFISANTES ») n'est pas un refus."""
    return bool(text) and text.strip().strip("*").strip().startswith(INSUFFICIENT)


def zone_for(score: float | None) -> str:
    cfg = get_app_config().reranker
    if score is None or (cfg.answerability_threshold and score < cfg.answerability_threshold):
        return "refus"
    if cfg.verification_threshold and score < cfg.verification_threshold:
        return "verification"
    return "reponse"


def format_sources(sources: list[Source]) -> str:
    blocks = []
    for s in sources:
        origin = f" (section citée par {s.cited_by})" if s.cited_by else ""
        blocks.append(f"[{s.n}]{origin}\n{s.text}")
    return "\n\n".join(blocks)


def answer_messages(question: str, sources: list[Source], zone: str) -> list[dict[str, str]]:
    system = ANSWER_SYSTEM + ("\n\n" + GRAY_ZONE if zone == "verification" else "")
    user = ANSWER_USER.format(sources=format_sources(sources), question=question)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def check_coverage(
    question: str, sources: list[Source], model: str | None = None, effort: str | None = None
) -> dict | None:
    """Zone grise : les sources répondent-elles ? Un appel dédié, plus fiable qu'une consigne
    noyée dans le prompt de réponse (mesuré : DSA et cookies passaient). None si indisponible."""
    try:
        prompt = CHECK.format(sources=format_sources(sources), question=question)
        data = chat_json(prompt, model=model, effort=effort)
    except Exception:
        logger.exception("Contrôle de couverture indisponible : consigne de zone grise seule")
        return None
    if not isinstance(data, dict) or not isinstance(data.get("couvert"), bool):
        return None
    return {"couvert": data["couvert"], "raison": str(data.get("raison", ""))}


@lru_cache
def default_retriever():
    from app.retrieval.pipeline import RetrievalSettings, Retriever

    return Retriever(RetrievalSettings.from_config())


class ChatService:
    def __init__(
        self,
        retriever=None,
        memory: ConversationStore | None = None,
        model: str | None = None,
        effort: str | None = None,
    ):
        cfg = get_app_config().llm
        self.model = model  # modèle de génération ; None : celui de config.yaml
        # effort de raisonnement (mistral-large) ; None : celui de config.yaml pour ce modèle
        self.effort = effort or cfg.reasoning_effort.get(model or cfg.chat_model)
        self._retriever = retriever
        self.memory = memory or ConversationStore()

    @property
    def retriever(self):
        if self._retriever is None:
            self._retriever = default_retriever()
        return self._retriever

    def prepare(self, message: str, session_id: str | None = None) -> Prepared:
        session_id = session_id or uuid.uuid4().hex
        r = route(message, self.memory.get(session_id))
        if r.intention != "juridique":
            return Prepared(session_id, message, r, zone=r.intention)
        rows = self.retriever.search_many([r.question], k=SEARCH_DEPTH)[0]
        final_k = get_app_config().retrieval.final_k
        context = self.retriever.with_citations(rows, final_k)
        top_score = rows[0]["score"] if rows else None
        zone = zone_for(top_score)
        sources = [Source.from_row(i + 1, row) for i, row in enumerate(context)]
        prepared = Prepared(session_id, message, r, zone, top_score, sources)
        if zone == "verification":
            prepared.coverage = check_coverage(r.question, sources, self.model, self.effort)
        if zone != "refus" and not prepared.not_covered:
            prepared.messages = answer_messages(r.question, sources, zone)
        return prepared

    def finish(self, p: Prepared, generated: str | None) -> ChatAnswer:
        """Réponse finale à partir du texte généré (None si le LLM n'a pas été appelé)."""
        refused = p.zone in ("refus", "hors_sujet") or p.not_covered or is_refusal(generated)
        if p.zone == "conversation":
            text, refused = CONVERSATION, False
        elif refused:
            text = REFUSAL
        else:
            text = (generated or "").strip()
        verification = None
        if not refused and p.zone not in ("conversation",):
            verification = verify(text, [s.text for s in p.sources])
            for s in p.sources:
                s.cited = s.n in verification.cited
            if verification.unsourced_facts:  # chiffre tiré de la mémoire du modèle : on le dit
                facts = ", ".join(humanize(f) for f in verification.unsourced_facts)
                text += UNSOURCED_WARNING.format(facts=facts)
        result = ChatAnswer(
            session_id=p.session_id,
            question=p.message,
            search_question=p.route.question,
            intention=p.route.intention,
            zone=p.zone,
            top_score=p.top_score,
            answer=text,
            refused=refused,
            sources=p.sources,
            verification=verification,
            coverage=p.coverage,
        )
        self.memory.add(p.session_id, p.message, text)
        return result

    def answer(self, message: str, session_id: str | None = None) -> ChatAnswer:
        p = self.prepare(message, session_id)
        generated = None
        if p.messages:
            generated = chat_text(p.messages, model=self.model, effort=self.effort)
        return self.finish(p, generated)

    def stream(self, message: str, session_id: str | None = None) -> Iterator[dict]:
        """Événements pour l'interface : statut, sources, morceaux de texte, puis la fin."""
        yield {"type": "statut", "etape": "recherche"}
        p = self.prepare(message, session_id)
        yield {
            "type": "sources",
            "session_id": p.session_id,
            "zone": p.zone,
            "sources": [{k: v for k, v in asdict(s).items() if k != "text"} for s in p.sources],
        }
        generated = None
        if p.messages:
            yield {"type": "statut", "etape": "redaction"}
            parts: list[str] = []
            buffer, released = "", False
            for chunk in stream_text(p.messages, model=self.model, effort=self.effort):
                parts.append(chunk)
                if released:
                    yield {"type": "texte", "texte": chunk}
                    continue
                buffer += chunk  # retient le début : est-ce le marqueur de refus ?
                if len(buffer) > len(INSUFFICIENT) + 5 and not is_refusal(buffer):
                    released = True
                    yield {"type": "texte", "texte": buffer}
            generated = "".join(parts)
            if not released and generated and not is_refusal(generated):
                yield {"type": "texte", "texte": generated}
        result = self.finish(p, generated)
        yield {"type": "fin", **{k: v for k, v in result.to_dict().items() if k != "sources"}}
