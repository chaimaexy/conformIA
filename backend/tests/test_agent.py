"""Tests de l'agent d'analyse (sans réseau : faux LLM, faux textes)."""

import json
from datetime import date

import pytest
from fastapi.testclient import TestClient

from app.agents import checklist, obligations, report
from app.agents.classification import (
    annex_points,
    article5_point,
    categories,
    classify_from_profile,
    classify_high_risk,
    classify_prohibited,
    conditions_of,
    point_text,
    practice_verdict,
)
from app.agents.graph import AnalysisService, Sources
from app.agents.profile import (
    Profile,
    apply_answers,
    market_status,
    missing_fields,
    parse_profile,
)

ARTICLE_5 = """AI Act > Article 5 - Pratiques interdites en matière d’IA
1. Les pratiques en matière d’IA suivantes sont interdites:
c) la mise sur le marché, la mise en service ou l’utilisation de systèmes d’IA pour l’évaluation ou la classification de personnes physiques ou de groupes de personnes au cours d’une période donnée en fonction de leur comportement social, la note sociale conduisant à l’une ou l’autre des situations suivantes, ou aux deux:
i) le traitement préjudiciable ou défavorable de certaines personnes physiques;
d) la mise sur le marché d’un système d’IA pour mener des évaluations des risques des personnes physiques;"""
ANNEX_III = """AI Act > Annexe III
4. Emploi, gestion de la main-d'œuvre et accès à l'emploi indépendant:
a) systèmes d'IA destinés à être utilisés pour le recrutement ou la sélection de personnes physiques, en particulier pour publier des offres d'emploi ciblées, analyser et filtrer les candidatures et évaluer les candidats;
b) systèmes d'IA destinés à être utilisés pour prendre des décisions influant sur les conditions des relations professionnelles, la promotion ou le licenciement dans le cadre de relations professionnelles contractuelles;"""
CV = (
    "Nous voulons utiliser un logiciel acheté à un éditeur qui analyse automatiquement les CV "
    "reçus et classe les candidats pour nos recrutements."
)


# ----------------------------------------------------------------------------- profil
def test_profile_values_need_verbatim_evidence():
    data = {
        "finalite": "Tri des CV",
        "roles": ["deployeur", "inconnu"],
        "decisions_sur_personnes": True,
        "deja_sur_le_marche": "apres_2026_08_02",  # deviné : aucune preuve
        "genere_contenus": False,
        "preuves": {
            "roles": "Nous voulons utiliser un logiciel acheté",
            "decisions_sur_personnes": "classe les candidats pour nos recrutements",
            "genere_contenus": "ne génère rien",  # absent de la description
        },
    }
    profile = parse_profile(CV, data)
    assert profile.roles == ["deployeur"] and profile.decisions_sur_personnes is True
    assert profile.deja_sur_le_marche is None and profile.genere_contenus is None
    assert profile.finalite == "Tri des CV"  # texte libre : pas de preuve exigée


def test_market_status_is_computed_from_the_month():
    today = date(2026, 11, 14)
    assert market_status("2025-10", today) == "entre_2025_08_02_et_2026_08_02"
    assert market_status("2024-03", today) == "avant_2025_08_02"
    assert market_status("2026-09", today) == "apres_2026_08_02"
    assert market_status("2027-01", today) == "non"  # pas encore sur le marché
    assert market_status("2025-08", today) is None  # avant ou après le 2 août : on demande
    assert market_status("2027-MM", today) == "non" and market_status("2026", today) is None
    # période calculée par le modèle (« à partir de 2027 » -> « après le 2 août 2026 ») : écartée
    later = "Nous allons proposer ce modèle en Europe à partir de 2027."
    guessed = {
        "deja_sur_le_marche": "apres_2026_08_02",
        "preuves": {"deja_sur_le_marche": "à partir de 2027"},
    }
    assert parse_profile(later, guessed).deja_sur_le_marche is None
    dated = {**guessed, "date_mise_sur_le_marche": "2027-MM"}
    dated["preuves"] = {**guessed["preuves"], "date_mise_sur_le_marche": "à partir de 2027"}
    assert parse_profile(later, dated).deja_sur_le_marche == "non"
    description = "Notre application est vendue depuis octobre 2025 sous notre marque."
    data = {
        "deja_sur_le_marche": "avant_2025_08_02",  # erreur de calcul du modèle
        "date_mise_sur_le_marche": "2025-10",
        "preuves": {
            "deja_sur_le_marche": "vendue depuis octobre 2025",
            "date_mise_sur_le_marche": "depuis octobre 2025",
        },
    }
    assert parse_profile(description, data).deja_sur_le_marche == "entre_2025_08_02_et_2026_08_02"


def test_missing_fields_and_closed_answers():
    profile = Profile(description=CV, roles=["deployeur"], genere_contenus=True)
    missing = missing_fields(profile)
    assert "roles" not in missing and "hypertrucage" in missing and "hors_ue" not in missing
    updated = apply_answers(
        profile,
        {"hypertrucage": "non", "secteur_public": "je_ne_sais_pas", "biometrie": "peut-être"},
    )
    assert updated.hypertrucage is False and updated.secteur_public is None
    assert updated.biometrie is None  # valeur hors des options : ignorée


# ----------------------------------------------------------------------------- classification
def test_points_are_normalized():
    assert article5_point("article 5, paragraphe 1, point f)") == "f)"
    assert article5_point("b bis)") == "b bis)"
    assert article5_point("f") == "f)" and article5_point("point g") == "g)"
    assert set(annex_points(ANNEX_III)) == {"4", "4 a)", "4 b)"}
    assert "i) le traitement" in point_text(ARTICLE_5, "c)")
    assert point_text(ARTICLE_5, "c)").count("d)") == 0


def test_prohibited_practice_is_dropped_when_a_condition_fails():
    calls = []

    def ask(prompt):
        calls.append(prompt)
        if "CONDITIONS :" in prompt:  # conditions fixées par le code, jugées par le modèle
            assert "[c.1]" in prompt and "[c.3]" in prompt
            return {
                "conditions": [
                    {"id": "c.1", "statut": "etablie"},
                    {"id": "c.2", "statut": "non"},  # i) et ii) : alternatives, toutes deux
                    {"id": "[c.3]", "statut": "non"},  # écartées
                ]
            }
        return {"points": [{"point": "c)"}, {"point": "z)"}]}  # z) : point inventé, ignoré

    result = classify_prohibited(Profile(description=CV), ARTICLE_5, ask)
    assert result["verdict"] == "non" and result["pratiques"] == [] and len(calls) == 2
    assert result["points_examines"] == [{"point": "c)", "verdict": "non"}]


def test_practice_verdict_follows_the_structure_of_the_point():
    def judged(point, statuses):
        return [{**c, "statut": s} for c, s in zip(conditions_of(point), statuses, strict=True)]

    # aucun élément établi : simple soupçon, pas retenu
    assert practice_verdict(judged("a)", ["inconnue", "inconnue", "inconnue"])) == "non"
    # c) : une seule des deux branches suffit (i) OU ii))
    assert practice_verdict(judged("c)", ["etablie", "etablie", "inconnue"])) == "oui"
    assert practice_verdict(judged("c)", ["etablie", "inconnue", "inconnue"])) == "incertain"
    # f) : l'exception (raisons médicales ou de sécurité) écarte l'interdiction
    assert practice_verdict(judged("f)", ["etablie", "etablie", "non"])) == "oui"
    assert practice_verdict(judged("f)", ["etablie", "etablie", "etablie"])) == "non"
    assert practice_verdict(judged("f)", ["etablie", "etablie", "inconnue"])) == "incertain"


def test_invented_quote_gets_one_retry_then_uncertain():
    calls = []

    def ask(prompt):
        calls.append(prompt)
        return {"annexe_iii": "oui", "point": "4 a)", "citation": "texte inventé par le modèle ici"}

    result = classify_high_risk(Profile(description=CV), ANNEX_III, ask)
    assert len(calls) == 2 and "ATTENTION" in calls[1]
    assert result["verdict"] == "incertain" and not result["citation_verifiee"]


def test_annex_point_is_found_from_the_quote():
    # point mal recopié (4 a)) pour une citation du 4 b), légèrement complétée par le modèle
    quote = (
        "systèmes d'IA destinés à être utilisés pour prendre des décisions influant sur les "
        "conditions des relations professionnelles, la promotion ou le licenciement des salariés"
    )

    def ask(prompt):
        return {"annexe_iii": "oui", "point": "4 a)", "citation": quote, "derogation": "non"}

    result = classify_high_risk(Profile(description=CV), ANNEX_III, ask)
    assert result["verdict"] == "oui" and result["point"] == "4 b)"
    assert result["point_annonce"] == "4 a)" and not result["nouvel_essai"]
    assert result["citation"].startswith("b) systèmes d'IA destinés")  # texte exact, lu par le code


def test_cv_screening_deployer_obligations_and_dates():
    profile = Profile(
        description=CV, roles=["deployeur"], decisions_sur_personnes=True, deja_sur_le_marche="non"
    )
    high_risk = {
        "verdict": "oui",
        "point": "4 a)",
        "derogation": "non",
    }
    prohibited = {"verdict": "non", "pratiques": []}
    simple = classify_from_profile(profile)
    derived = categories(profile, prohibited, high_risk, simple)
    assert derived["niveau"] == "haut_risque"
    classification = {**derived, "interdit": prohibited, "annexe_iii": high_risk, **simple}
    selected = {o["id"]: o for o in obligations.select(profile, classification)}
    assert {
        "maitrise_ia",
        "obligations_deployeur",
        "information_personnes",
        "droit_explication",
    } <= set(selected)
    assert "obligations_fournisseur" not in selected and "analyse_impact_droits" not in selected
    assert [d["date"] for d in selected["obligations_deployeur"]["dates"]] == ["2027-12-02"]
    assert (
        selected["droit_explication"]["dates"] == []
    )  # article 113 sans date directe : pas inventée


def test_dates_follow_market_status():
    profile = Profile(
        description="x",
        roles=["fournisseur"],
        genere_contenus=True,
        modele_usage_general=True,
        deja_sur_le_marche="avant_2025_08_02",
    )
    simple = classify_from_profile(profile)
    no = {"verdict": "non", "pratiques": []}
    derived = categories(profile, no, {"verdict": "non"}, simple)
    classification = {**derived, "interdit": no, "annexe_iii": {"verdict": "non"}, **simple}
    selected = {o["id"]: o for o in obligations.select(profile, classification)}
    assert [d["date"] for d in selected["obligations_modele"]["dates"]] == ["2027-08-02"]
    assert [d["date"] for d in selected["marquer_contenus"]["dates"]] == ["2026-12-02"]


def test_map_verification_detects_a_wrong_excerpt():
    assert obligations.verify_map(lambda label: "") != []  # aucune section : tout est signalé
    cfg = obligations.load_map()
    everything = "\n".join(
        [d["extrait"] for d in cfg["dates"].values()]
        + [cfg["haut_risque_deja_sur_le_marche"]["extrait"]]
        + [o["extrait"] for o in cfg["obligations"]]
    )
    assert obligations.verify_map(lambda label: everything) == []


# ----------------------------------------------------------------------------- RGPD, rapport
THEMES = {
    n: {"titre": f"Thème {n}", "texte": f"Texte du thème numéro {n} de la liste.", "url": "u"}
    for n in range(1, 12)
}


def test_checklist_without_personal_data_makes_no_call():
    results, calls = checklist.evaluate(
        Profile(description="x", donnees_personnelles=False), THEMES, lambda p: pytest.fail()
    )
    assert calls == 0 and {r["statut"] for r in results} == {"non_applicable"}


def test_checklist_conforme_needs_a_verified_quote():
    def ask(prompt):
        return {
            "themes": [
                {"numero": 1, "statut": "conforme", "citation": "faux"},
                {"numero": 2, "statut": "???"},
            ]
        }

    results, calls = checklist.evaluate(Profile(description="x"), THEMES, ask)
    by_number = {r["numero"]: r for r in results}
    assert (
        calls == 3
        and by_number[1]["statut"] == "a_verifier"
        and by_number[2]["statut"] == "a_verifier"
    )
    assert by_number[5]["constat"].startswith("Thème non évalué")


def test_synthesis_must_be_text_without_new_facts():
    built = {
        "classification": {"niveau": "haut_risque"},
        "obligations": [
            {
                "article": "Article 26",
                "titre": "Notice",
                "dates": [{"date": "2027-12-02"}],
                "statut": "applicable",
            }
        ],
        "rgpd": [],
        "informations_manquantes": [],
    }
    ok = "Système à haut risque : obligations du déployeur à partir du 2 décembre 2027."
    assert report.synthesize(built, lambda p: {"synthese": ok}) == (ok, "llm")
    invented = "Amende de 20 millions d'euros dès le 2 août 2026."
    assert report.synthesize(built, lambda p: {"synthese": invented})[1] == "code"
    assert report.synthesize(built, lambda p: {"synthese": {"niveau": 3}})[1] == "code"
    parts = {"niveau": "Système à haut risque.", "obligations": "Article 26."}
    assert report.synthesize(built, lambda p: {"synthese": parts}) == (
        "Système à haut risque. Article 26.",
        "llm",
    )


# ----------------------------------------------------------------------------- graphe, API
class FakeSources(Sources):
    def text(self, source_id, *labels):
        return ARTICLE_5 if "Article 5" in labels else ANNEX_III

    def checklist_themes(self):
        return THEMES


def fake_ask(prompt):
    if "DESCRIPTION DU PROJET" in prompt:
        return {
            "finalite": "Tri des CV",
            "roles": ["deployeur"],
            "decisions_sur_personnes": True,
            "preuves": {
                "roles": "Nous voulons utiliser",
                "decisions_sur_personnes": "classe les candidats",
            },
        }
    if "TEXTE DE L'ARTICLE 5" in prompt:
        return {"verdict": "non", "pratiques": []}
    if "annexe_iii" in prompt:
        quote = "systèmes d'IA destinés à être utilisés pour le recrutement ou la sélection de personnes physiques"
        return {"annexe_iii": "oui", "point": "4 a)", "citation": quote, "derogation": "non"}
    if "THÈMES DE LA LISTE" in prompt:
        return {"themes": []}
    return {"synthese": "Système à haut risque, obligations du déployeur."}


def test_graph_pauses_for_questions_then_reports(tmp_path):
    service = AnalysisService(FakeSources(), fake_ask, db_path=tmp_path / "a.sqlite")
    events = list(service.start(CV))
    analysis_id = events[0]["analyse_id"]
    assert [e["type"] for e in events] == ["debut", "etape", "questions"]
    asked = [q["champ"] for q in events[-1]["questions"]]
    assert "roles" not in asked and "biometrie" in asked
    assert service.get(analysis_id)["statut"] == "questions"

    answers = {name: "non" for name in asked} | {"donnees_personnelles": "oui"}
    events = list(service.resume(analysis_id, answers))
    steps = [e["etape"] for e in events if e["type"] == "etape"]
    assert steps == ["questions", "interdictions", "haut_risque", "obligations", "rgpd", "rapport"]
    final = events[-1]
    assert (
        final["type"] == "rapport" and final["rapport"]["classification"]["niveau"] == "haut_risque"
    )
    assert final["rapport"]["appels_llm"] == 1 + 1 + 1 + 3 + 1
    assert "| Article 26 |" in final["markdown"]
    assert service.get(analysis_id)["statut"] == "termine"
    assert list(service.resume(analysis_id, {}))[0]["type"] == "erreur"


def test_analysis_endpoints(monkeypatch, tmp_path):
    from app.api.routes import analyses
    from app.main import app

    service = AnalysisService(FakeSources(), fake_ask, db_path=tmp_path / "a.sqlite")
    monkeypatch.setattr(analyses, "get_service", lambda: service)
    client = TestClient(app)

    def events(response):
        return [json.loads(line[6:]) for line in response.text.split("\n\n") if line]

    started = events(client.post("/analyses", json={"description": CV}))
    analysis_id = started[0]["analyse_id"]
    assert started[-1]["type"] == "questions"
    assert client.get(f"/analyses/{analysis_id}").json()["statut"] == "questions"
    answers = {q["champ"]: "non" for q in started[-1]["questions"]}
    finished = events(client.post(f"/analyses/{analysis_id}/reponses", json={"reponses": answers}))
    assert finished[-1]["type"] == "rapport"
    assert (
        client.post(f"/analyses/{analysis_id}/reponses", json={"reponses": {}}).status_code == 409
    )
    assert client.get("/analyses/inconnue").status_code == 404
    assert client.post("/analyses", json={"description": "trop court"}).status_code == 422


def test_object_wrapped_in_a_list_is_unwrapped():
    from app.agents.graph import as_object

    assert as_object([{"conditions": []}]) == {"conditions": []}
    assert as_object([{"a": 1}, {"b": 2}]) == {"elements": [{"a": 1}, {"b": 2}]}
    assert as_object("texte") == {} and as_object([1, 2]) == {}


def test_doubt_without_any_annex_point_is_not_high_risk():
    def ask(prompt):
        return {"annexe_iii": "incertain", "point": None, "citation": None, "derogation": "non"}

    result = classify_high_risk(Profile(description=CV), ANNEX_III, ask)
    assert result["verdict"] == "non" and result["point"] is None and result["nouvel_essai"]
