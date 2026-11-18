"""Tests de la chaîne de réponse (sans réseau : faux retriever, faux LLM)."""

import json

import pytest
from fastapi.testclient import TestClient

from app.prompts.rag import CONVERSATION, INSUFFICIENT, REFUSAL
from app.rag import router as router_module
from app.rag import service as service_module
from app.rag.memory import ConversationStore, format_history
from app.rag.router import needs_router, route
from app.rag.service import ChatService, zone_for
from app.rag.verify import cited_numbers, verify

SOURCES = [
    "AI Act > Article 113\n\nIl est applicable à partir du 2 août 2026.",
    "AI Act > Article 99\n\nLe non-respect de l’interdiction des pratiques fait l’objet d’amendes.",
]


# ----------------------------------------------------------------------------- vérification
def test_verify_valid_answer():
    answer = "Le règlement s'applique à partir du 2 août 2026 [1]. Des amendes sont prévues [2]."
    v = verify(answer, SOURCES)
    assert v.ok and v.cited == [1, 2] and not v.uncited_sentences


def test_verify_detects_invalid_marker_and_invented_quote():
    answer = (
        "Selon le texte, « il est applicable à partir du 2 août 2026 » [1][3]. "
        "Il dit aussi « les amendes sont plafonnées à 10 euros » [2]."
    )
    v = verify(answer, SOURCES)
    assert v.invalid_markers == [3]
    assert v.quotes_checked == 2
    assert v.unverified_quotes == ["les amendes sont plafonnées à 10 euros"]
    assert not v.ok


def test_verify_quote_normalization_and_uncited_sentence():
    answer = (
        "« Le non-respect de l'interdiction des pratiques » est sanctionné [2]. "
        "Une phrase assez longue qui affirme quelque chose sans citer la moindre source."
    )
    v = verify(answer, SOURCES)
    assert not v.unverified_quotes  # apostrophe droite contre typographique : même citation
    assert len(v.uncited_sentences) == 1


def test_cited_numbers_order_and_lists():
    assert cited_numbers("a [2] b [1, 3] c [2]") == [2, 1, 3]


# ----------------------------------------------------------------------------- mémoire, routeur
def test_memory_keeps_last_messages():
    store = ConversationStore(max_messages=4)
    for i in range(3):
        store.add("s", f"q{i}", f"r{i}")
    assert [m["content"] for m in store.get("s")] == ["q1", "r1", "q2", "r2"]
    assert store.get("autre") == []
    assert "..." in format_history([{"role": "assistant", "content": "x" * 500}], max_chars=10)


def test_router_only_when_useful(monkeypatch):
    assert not needs_router("Que dit l'article 22 du RGPD ?", [])
    assert needs_router("bonjour", [])
    assert needs_router("et pour les PME ?", [{"role": "user", "content": "..."}])
    monkeypatch.setattr(router_module, "chat_json", lambda p, **k: pytest.fail("appel inutile"))
    assert route("Que dit l'article 22 du RGPD ?", []).routed is False


def test_router_rewrites_and_falls_back(monkeypatch):
    history = [{"role": "user", "content": "Amende pour une pratique interdite ?"}]
    monkeypatch.setattr(
        router_module,
        "chat_json",
        lambda p, **k: {"intention": "juridique", "question": "Amende pour une PME ?"},
    )
    assert route("et pour les PME ?", history).question == "Amende pour une PME ?"
    monkeypatch.setattr(router_module, "chat_json", lambda p, **k: {"intention": "???"})
    r = route("et pour les PME ?", history)
    assert (r.intention, r.question) == ("juridique", "et pour les PME ?")

    def broken(prompt, **k):
        raise RuntimeError("API indisponible")

    monkeypatch.setattr(router_module, "chat_json", broken)
    assert route("et pour les PME ?", history).question == "et pour les PME ?"


# ----------------------------------------------------------------------------- service
def test_zones():
    assert zone_for(None) == "refus"
    assert zone_for(0.1) == "refus"
    assert zone_for(0.3) == "verification"
    assert zone_for(0.8) == "reponse"


class FakeRetriever:
    def __init__(self, score):
        self.score = score

    def search_many(self, questions, k):
        rows = []
        for i, text in enumerate(SOURCES):
            label = text.split("\n")[0].split(" > ")[1]
            meta = {"header": text.split("\n")[0], "label": label, "url": f"u{i}"}
            rows.append({"content": text, "metadata": meta, "score": self.score - i / 10})
        return [rows]

    def with_citations(self, rows, final_k, question=""):
        return rows[:final_k]


def make_service(monkeypatch, score, generated, covered=True):
    calls = []

    def fake_chat_text(messages, model=None, effort=None):
        calls.append(messages)
        return generated

    def fake_check(prompt, model=None, effort=None):
        calls.append("controle")
        return {"couvert": covered, "raison": "test"}

    monkeypatch.setattr(service_module, "chat_text", fake_chat_text)
    monkeypatch.setattr(service_module, "chat_json", fake_check)
    return ChatService(retriever=FakeRetriever(score)), calls


def test_answer_with_verified_citations(monkeypatch):
    service, calls = make_service(monkeypatch, 0.8, "Applicable dès le 2 août 2026 [1].")
    result = service.answer("Quand l'AI Act s'applique-t-il ?", session_id="s1")
    assert result.zone == "reponse" and not result.refused
    assert result.verification.ok and [s.n for s in result.sources if s.cited] == [1]
    assert "Vérifie d'abord" not in calls[0][0]["content"] and "[2]" in calls[0][1]["content"]
    assert service.memory.get("s1")[1]["content"] == result.answer


def test_low_score_is_refused_without_llm(monkeypatch):
    service, calls = make_service(monkeypatch, 0.1, "ne doit pas être appelé")
    result = service.answer("Quel est le taux de TVA d'un logiciel SaaS ?")
    assert result.refused and result.answer == REFUSAL and calls == []


def test_gray_zone_not_covered_is_refused_without_generation(monkeypatch):
    service, calls = make_service(monkeypatch, 0.3, "ne doit pas être appelé", covered=False)
    result = service.answer("Que dit le Data Act sur les objets connectés ?")
    assert result.zone == "verification" and result.refused and result.answer == REFUSAL
    assert calls == ["controle"] and result.coverage == {"couvert": False, "raison": "test"}


def test_gray_zone_covered_still_honours_a_refusal(monkeypatch):
    service, calls = make_service(monkeypatch, 0.3, INSUFFICIENT)
    result = service.answer("Que dit le Data Act sur les objets connectés ?")
    assert result.refused and calls[0] == "controle"
    assert "Vérifie d'abord" in calls[1][0]["content"]


def test_marker_inside_a_real_answer_is_not_a_refusal(monkeypatch):
    partial = f"Applicable dès le 2 août 2026 [1]. Pour les sanctions : {INSUFFICIENT}."
    service, _ = make_service(monkeypatch, 0.8, partial)
    assert not service.answer("Quand l'AI Act s'applique-t-il ?").refused
    service, _ = make_service(monkeypatch, 0.8, f"**{INSUFFICIENT}**\n\nLes sources...")
    assert service.answer("Quand l'AI Act s'applique-t-il ?").refused


def test_unsourced_figure_triggers_a_visible_warning(monkeypatch):
    service, _ = make_service(monkeypatch, 0.8, "Applicable dès le 2 août 2028 [1].")
    result = service.answer("Quand l'AI Act s'applique-t-il ?")
    assert result.verification.unsourced_facts == ["date:2028-08-02"]
    assert not result.verification.ok and "2 août 2028 ne figure pas" in result.answer


def test_conversation_needs_no_search(monkeypatch):
    service, calls = make_service(monkeypatch, 0.9, "x")
    monkeypatch.setattr(
        router_module, "chat_json", lambda p, **k: {"intention": "conversation", "question": ""}
    )
    result = service.answer("bonjour")
    assert result.zone == "conversation" and result.answer == CONVERSATION and calls == []


def test_stream_hides_the_refusal_marker(monkeypatch):
    service, _ = make_service(monkeypatch, 0.3, "non utilisé")
    monkeypatch.setattr(
        service_module,
        "stream_text",
        lambda m, model=None, effort=None: iter([INSUFFICIENT[:6], INSUFFICIENT[6:]]),
    )
    events = list(service.stream("Que dit le Data Act ?"))
    assert [e["type"] for e in events] == ["statut", "sources", "statut", "fin"]
    assert events[-1]["refused"] and events[-1]["answer"] == REFUSAL

    pieces = ["Applicable dès le ", "2 août 2026 ", "[1]."]
    monkeypatch.setattr(
        service_module, "stream_text", lambda m, model=None, effort=None: iter(pieces)
    )
    events = list(service.stream("Quand l'AI Act s'applique-t-il ?"))
    text = "".join(e["texte"] for e in events if e["type"] == "texte")
    assert text == "".join(pieces) and events[-1]["verification"]["cited"] == [1]


def test_chat_endpoints(monkeypatch, db, as_user):
    from app.api.routes import chat
    from app.main import app
    from app.storage.records import Conversations

    service, _ = make_service(monkeypatch, 0.8, "Applicable dès le 2 août 2026 [1].")
    conversations = Conversations(db)
    service.memory = conversations
    monkeypatch.setattr(chat, "get_service", lambda: service)
    monkeypatch.setattr(chat, "get_conversations", lambda: conversations)
    monkeypatch.setattr(
        service_module,
        "stream_text",
        lambda m, model=None, effort=None: iter(["Applicable dès le 2 août 2026 [1]."]),
    )
    client = TestClient(app)
    body = client.post("/chat", json={"message": "Quand l'AI Act s'applique-t-il ?"}).json()
    assert body["sources"][0]["cited"] and body["verification"]["cited"] == [1]
    session = body["session_id"]  # conversation créée au premier message
    response = client.post("/chat/stream", json={"message": "Quand ?", "session_id": session})
    events = [json.loads(line[6:]) for line in response.text.split("\n\n") if line]
    assert events[-1]["type"] == "fin" and events[-1]["session_id"] == session
    assert client.post("/chat", json={"message": ""}).status_code == 422
    unknown = client.post("/chat", json={"message": "Quand ?", "session_id": "inventee"})
    assert unknown.status_code == 404


# ----------------------------------------------------------------------------- évaluation
def test_facts_are_normalized():
    from app.rag.facts import extract

    reference = "Jusqu'à 20 000 000 EUR ou 4 % ; applicable à partir du 2 décembre 2027."
    generated = (
        "Une amende de 20 millions d’euros (4 %), dès le 2 décembre 2027 ou le 1er août 2024."
    )
    assert extract(reference) == {"eur:20000000", "pct:4", "date:2027-12-02"}
    assert extract(reference) <= extract(generated)
    assert "date:2024-08-01" in extract(generated)  # « 1er »


def test_quote_with_ellipsis_checks_each_fragment():
    ok = "« Le non-respect de l'interdiction [...] fait l'objet d'amendes » [2]."
    bad = "« Le non-respect de l'interdiction [...] fait l'objet de prison ferme » [2]."
    assert not verify(ok, SOURCES).unverified_quotes
    assert verify(bad, SOURCES).unverified_quotes


def test_judge_parsing_is_strict_on_unknown_values():
    from app.evaluation.judge import parse

    j = parse(
        {
            "affirmations": [
                {"texte": "Amende de 35 M€", "verdict": "contredite"},
                {"texte": "Article 50", "verdict": "soutenue"},
                {"texte": "Vague", "verdict": "peut-être"},  # inconnu -> non soutenue
                {"texte": "  "},  # vide -> ignorée
            ],
            "exactitude": "plutôt bonne",  # inconnu -> incorrecte (le plus sévère)
            "erreurs": ["35 M€ au lieu de 15 M€", ""],
        }
    )
    assert [c["verdict"] for c in j.claims] == ["contredite", "soutenue", "non_soutenue"]
    assert j.accuracy == "incorrecte" and j.errors == ["35 M€ au lieu de 15 M€"]
    assert j.faithfulness == 1 / 3 and len(j.unsupported) == 2
    assert parse({}).faithfulness is None
