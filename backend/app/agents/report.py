"""Rapport de pré-diagnostic : assemblé par le code, synthèse rédigée par le LLM puis contrôlée.

La synthèse ne doit contenir aucune date, aucun montant ni pourcentage absent des résultats :
sinon elle est remplacée par une synthèse écrite par le code (rien d'inventé ne passe).
"""

import json
from collections.abc import Callable

from app.agents.profile import QUESTIONS, Profile
from app.rag.facts import extract, humanize

LEVELS = {
    "interdit": "Pratique interdite (article 5)",
    "haut_risque": "Système d'IA à haut risque (article 6, paragraphe 2, et annexe III)",
    "haut_risque_a_confirmer": "Haut risque à confirmer (article 6, paragraphe 1, et annexe I)",
    "incertain": "Classification incertaine : des informations manquent pour conclure",
    "obligations_specifiques": "Obligations de transparence ou de modèle à usage général",
    "risque_minimal": "Risque minimal : pas d'obligation spécifique au-delà de la maîtrise de l'IA",
}
STATUS_LABELS = {
    "point_d_attention": "Point d'attention",
    "a_verifier": "À vérifier",
    "conforme": "Conforme",
    "non_applicable": "Non applicable",
}
DISCLAIMER = (
    "Pré-diagnostic automatique établi à partir de votre description et des textes cités. "
    "Il ne remplace pas l'avis d'un juriste ou de votre délégué à la protection des données."
)

SYNTHESIS_PROMPT = """Rédige en français la synthèse (4 à 6 phrases) d'un pré-diagnostic de \
conformité, pour un chef de projet. Utilise UNIQUEMENT les résultats ci-dessous : n'ajoute \
aucune date, aucun montant, aucun article qui n'y figure pas. Commence par le niveau de risque, \
puis les obligations principales et leurs dates, puis les points RGPD à traiter en priorité, \
puis ce qu'il reste à préciser.

RÉSULTATS :
{results}

Réponds en JSON, avec un seul paragraphe de texte (pas d'objet, pas de liste) : \
{{"synthese": "..."}}"""


def fr_date(iso: str) -> str:
    """« 2027-12-02 » -> « 2 décembre 2027 » (forme que le contrôle des faits reconnaît)."""
    return humanize(f"date:{iso}")


def key_results(report: dict) -> dict:
    """Ce que la synthèse a le droit de dire."""
    return {
        "niveau": LEVELS[report["classification"]["niveau"]],
        "obligations": [
            {
                "article": o["article"],
                "titre": o["titre"],
                "dates": [fr_date(d["date"]) for d in o["dates"]],
                "statut": o["statut"],
            }
            for o in report["obligations"]
        ],
        "rgpd_points_d_attention": [
            t["titre"] for t in report["rgpd"] if t["statut"] == "point_d_attention"
        ],
        "informations_manquantes": [m["question"] for m in report["informations_manquantes"]],
    }


def template_synthesis(report: dict) -> str:
    results = key_results(report)
    parts = [f"Niveau retenu : {results['niveau']}."]
    dated = [o for o in results["obligations"] if o["dates"]]
    if results["obligations"]:
        first = ", ".join(f"{o['article']} ({o['titre']})" for o in results["obligations"][:4])
        parts.append(f"{len(results['obligations'])} obligation(s) identifiée(s), dont : {first}.")
    if dated:
        earliest = min(d["date"] for o in report["obligations"] for d in o["dates"])
        parts.append(f"Première échéance : {fr_date(earliest)}.")
    priorities = results["rgpd_points_d_attention"]
    if priorities:
        listed = "; ".join(priorities[:3]) + (
            f" (et {len(priorities) - 3} autre(s))" if len(priorities) > 3 else ""
        )
        parts.append(f"Points RGPD prioritaires : {listed}.")
    if results["informations_manquantes"]:
        n = len(results["informations_manquantes"])
        parts.append(f"{n} information(s) restent à préciser pour confirmer ce diagnostic.")
    return " ".join(parts)


def synthesize(report: dict, ask: Callable[[str], dict]) -> tuple[str, str]:
    """(synthèse, origine) : « llm » si elle passe le contrôle, sinon « code »."""
    results = key_results(report)
    payload = json.dumps(results, ensure_ascii=False, indent=1)
    try:
        text = ask(SYNTHESIS_PROMPT.format(results=payload)).get("synthese")
    except Exception:
        text = None
    # le petit modèle découpe volontiers la synthèse en rubriques : on les met bout à bout
    if isinstance(text, dict) and all(isinstance(v, str) for v in text.values()):
        text = " ".join(v.strip() for v in text.values())
    # du texte, sans fait (date, montant, pourcentage) absent des résultats
    if isinstance(text, str) and text.strip() and extract(text) <= extract(payload):
        return text.strip(), "llm"
    return template_synthesis(report), "code"


def build(
    profile: Profile,
    missing: list[str],
    classification: dict,
    obligations: list[dict],
    note_111: dict | None,
    rgpd: list[dict],
) -> dict:
    # le texte d'un PDF (jusqu'à 30 pages) ne se recopie pas dans le rapport : ses preuves suffisent
    exclude = {"description"} if profile.document else set()
    return {
        "profil": profile.model_dump(exclude=exclude),
        "informations_manquantes": [
            {"champ": m, "question": QUESTIONS[m]["question"]} for m in missing
        ],
        "classification": classification,
        "obligations": obligations,
        "systeme_deja_sur_le_marche": note_111,
        "rgpd": rgpd,
        "avertissement": DISCLAIMER.replace("votre description", "votre document")
        if profile.document
        else DISCLAIMER,
    }


CONDITION_LABELS = {"etablie": "Oui.", "non": "Non.", "inconnue": "Inconnu."}
PROFILE_LABELS = {
    "roles": "Rôle",
    "decisions_sur_personnes": "Décisions sur des personnes",
    "interagit_avec_personnes": "Dialogue avec des personnes",
    "genere_contenus": "Génère des contenus",
    "hypertrucage": "Hypertrucage",
    "biometrie": "Biométrie ou émotions",
    "composant_securite_produit": "Composant de sécurité d'un produit",
    "modele_usage_general": "Modèle d'IA à usage général",
    "secteur_public": "Secteur public",
    "lieu_de_travail": "Sur le lieu de travail",
    "donnees_personnelles": "Données personnelles",
    "donnees_sensibles": "Données sensibles",
    "deja_sur_le_marche": "Déjà sur le marché",
    "hors_ue": "Établi hors de l'UE",
}


def _value_label(field: str, value) -> str:
    if isinstance(value, bool):
        return "oui" if value else "non"
    options = dict(QUESTIONS[field]["options"])
    if isinstance(value, list):
        return ", ".join(options.get(v, v) for v in value)
    return options.get(value, str(value))


def profile_lines(profile: dict) -> list[str]:
    """Une ligne par valeur connue, avec sa preuve (extrait et page) ou « votre réponse »."""
    lines = []
    for field, label in PROFILE_LABELS.items():
        value = profile.get(field)
        if value in (None, []):
            continue
        proof = (profile.get("preuves") or {}).get(field)
        if proof:
            page = f", page {proof['page']}" if proof.get("page") else ""
            source = f"« {_one_line(proof['extrait'])} »{page}"
        else:
            source = "votre réponse"
        lines.append(f"| {label} | {_value_label(field, value)} | {source} |")
    return lines


def _one_line(text: str) -> str:
    return " ".join(text.split())


def to_markdown(report: dict) -> str:
    c = report["classification"]
    lines = ["# Pré-diagnostic de conformité", "", f"> {report['avertissement']}", ""]
    lines += ["## Synthèse", "", report.get("synthese", ""), ""]
    profile = report["profil"]
    origin = f" (d'après le document « {profile['document']} »)" if profile.get("document") else ""
    lines += [f"## Profil du projet{origin}", ""]
    if profile.get("finalite"):
        lines += [f"Finalité : {profile['finalite']}", ""]
    lines += ["| Élément | Valeur | Source |", "|---|---|---|", *profile_lines(profile), ""]
    lines += ["## Classification AI Act", "", f"**{LEVELS[c['niveau']]}**", ""]
    for p in c["interdit"]["pratiques"]:
        lines.append(
            f"- Article 5, paragraphe 1, point {p['point']} ({p['verdict']}) : "
            f"« {_one_line(p['citation'])} »"
        )
        for cond in p.get("conditions") or []:
            lines.append(f"  - {cond['question']} {CONDITION_LABELS[cond['statut']]}")
    hr = c["annexe_iii"]
    if hr["verdict"] != "non":
        point = f"point {hr['point']}" if hr["point"] else "point non déterminé"
        quote = f" : « {_one_line(hr['citation'])} »" if hr["citation"] else ""
        lines.append(f"- Annexe III, {point} ({hr['verdict']}){quote}")
        if hr["derogation"] != "non":
            lines.append(f"- Dérogation de l'article 6, paragraphe 3 : {hr['derogation']}")
    if c["annexe_i"] == "a_confirmer":
        lines.append("- Annexe I : composant de sécurité d'un produit réglementé, à confirmer")
    for n in c["transparence"]:
        lines.append(f"- Transparence : article 50, paragraphe {n}")
    if c["modele_usage_general"]:
        lines.append("- Fournisseur d'un modèle d'IA à usage général (chapitre V)")
    lines += ["", "## Obligations et dates", "", "| Article | Obligation | À partir du | Statut |"]
    lines.append("|---|---|---|---|")
    for o in report["obligations"]:
        when = ", ".join(fr_date(d["date"]) for d in o["dates"]) or "voir article 113"
        status = "à confirmer" if o["statut"] == "a_confirmer" else "applicable"
        lines.append(f"| {o['article']} | {o['titre']} | {when} | {status} |")
    if report.get("systeme_deja_sur_le_marche"):
        note = report["systeme_deja_sur_le_marche"]
        lines += ["", f"Système déjà sur le marché ({note['article']}) : « {note['extrait']} »."]
    lines += ["", "## Liste de vérification de la CNIL (RGPD)", "", "| Thème | Statut | Constat |"]
    lines.append("|---|---|---|")
    for t in report["rgpd"]:
        lines.append(
            f"| {t['numero']}. {t['titre']} | {STATUS_LABELS[t['statut']]} | {t['constat']} |"
        )
    if report["informations_manquantes"]:
        lines += ["", "## À préciser", ""]
        lines += [f"- {m['question']}" for m in report["informations_manquantes"]]
    return "\n".join(lines) + "\n"
