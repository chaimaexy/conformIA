"""Profil du projet : ce que l'agent doit savoir pour qualifier un système d'IA.

Le LLM extrait le profil d'une description libre (un appel JSON). Tout ce que la description ne
dit pas explicitement reste à None : le code pose alors une question fermée, prise dans une liste
fixe (aucun appel LLM). Un « je ne sais pas » laisse le champ à None, et le rapport le signale.
"""

import re
from datetime import date
from typing import Literal, get_args

from pydantic import BaseModel, Field

from app.agents.document import page_of
from app.rag.verify import normalize

Role = Literal["fournisseur", "deployeur", "importateur", "distributeur", "mandataire"]
MarketStatus = Literal[
    "non", "avant_2025_08_02", "entre_2025_08_02_et_2026_08_02", "apres_2026_08_02"
]


class Profile(BaseModel):
    description: str = ""
    finalite: str | None = None  # ce que fait le système, en une phrase
    roles: list[Role] = Field(default_factory=list)
    personnes_concernees: list[str] = Field(default_factory=list)
    secteur_public: bool | None = None
    lieu_de_travail: bool | None = None
    decisions_sur_personnes: bool | None = None
    interagit_avec_personnes: bool | None = None
    genere_contenus: bool | None = None
    hypertrucage: bool | None = None
    biometrie: bool | None = None  # reconnaissance des émotions, catégorisation ou identification
    composant_securite_produit: bool | None = None
    modele_usage_general: bool | None = None
    hors_ue: bool | None = None
    donnees_personnelles: bool | None = None
    donnees_sensibles: bool | None = None
    deja_sur_le_marche: MarketStatus | None = None
    # tenus par le code, pas extraits par le LLM
    document: str | None = None  # nom du PDF quand l'analyse part d'un document
    preuves: dict[str, dict] = Field(default_factory=dict)  # champ -> {extrait, page}
    passages: list[dict] = Field(default_factory=list)  # passages clés du document, vérifiés


YES_NO = [("oui", "Oui"), ("non", "Non"), ("je_ne_sais_pas", "Je ne sais pas")]
QUESTIONS: dict[str, dict] = {
    "roles": {
        "question": "Quel est votre rôle vis-à-vis de ce système d'IA ?",
        "aide": "Fournisseur : vous le développez et le mettez sur le marché sous votre nom. "
        "Déployeur : vous l'utilisez sous votre autorité dans votre activité.",
        "options": [
            ("fournisseur", "Fournisseur"),
            ("deployeur", "Déployeur (utilisateur professionnel)"),
            ("importateur", "Importateur"),
            ("distributeur", "Distributeur"),
        ],
        "multiple": True,
    },
    "decisions_sur_personnes": {
        "question": "Le système prend-il ou aide-t-il à prendre des décisions concernant des "
        "personnes (embauche, crédit, accès à un service, notation, sanction...) ?",
        "options": YES_NO,
    },
    "interagit_avec_personnes": {
        "question": "Le système dialogue-t-il directement avec des personnes (chatbot, "
        "assistant vocal) ?",
        "options": YES_NO,
    },
    "genere_contenus": {
        "question": "Le système génère-t-il des textes, images, sons ou vidéos ?",
        "options": YES_NO,
    },
    "biometrie": {
        "question": "Le système reconnaît-il des émotions, ou identifie-t-il ou classe-t-il des "
        "personnes à partir de données biométriques (visage, voix, démarche) ?",
        "options": YES_NO,
    },
    "composant_securite_produit": {
        "question": "Le système est-il un composant de sécurité d'un produit réglementé "
        "(jouet, machine, dispositif médical, véhicule, ascenseur...) ?",
        "options": YES_NO,
    },
    "modele_usage_general": {
        "question": "Développez-vous vous-même un modèle d'IA à usage général (un grand modèle "
        "de langage ou d'image proposé à d'autres), et pas seulement un système qui en "
        "utilise un ?",
        "options": YES_NO,
    },
    "secteur_public": {
        "question": "Êtes-vous un organisme public, ou une entreprise privée qui fournit un "
        "service public ?",
        "options": YES_NO,
    },
    "lieu_de_travail": {
        "question": "Le système est-il utilisé sur le lieu de travail, sur vos salariés ?",
        "options": YES_NO,
    },
    "donnees_personnelles": {
        "question": "Le système traite-t-il des données personnelles (pour son entraînement ou "
        "son utilisation) ?",
        "options": YES_NO,
    },
    "donnees_sensibles": {
        "question": "Traite-t-il des données sensibles : santé, biométrie, opinions, origine, "
        "religion, orientation sexuelle, données d'infraction ?",
        "options": YES_NO,
    },
    "deja_sur_le_marche": {
        "question": "Le système (ou le modèle) est-il déjà sur le marché ou en service ?",
        "options": [
            ("non", "Pas encore"),
            ("avant_2025_08_02", "Oui, avant le 2 août 2025"),
            ("entre_2025_08_02_et_2026_08_02", "Oui, entre le 2 août 2025 et le 2 août 2026"),
            ("apres_2026_08_02", "Oui, après le 2 août 2026"),
            ("je_ne_sais_pas", "Je ne sais pas"),
        ],
    },
    "hors_ue": {
        "question": "Votre organisation est-elle établie hors de l'Union européenne ?",
        "options": YES_NO,
    },
    "hypertrucage": {
        "question": "Les contenus générés peuvent-ils imiter de façon réaliste des personnes, "
        "lieux ou événements existants (hypertrucage, « deepfake ») ?",
        "options": YES_NO,
    },
}


def missing_fields(profile: Profile) -> list[str]:
    """Champs à demander, dans l'ordre des questions ; certains seulement si utiles."""
    missing = []
    for name in QUESTIONS:
        value = getattr(profile, name)
        if name == "roles":
            needed = not value
        elif name == "donnees_sensibles":
            needed = value is None and profile.donnees_personnelles is not False
        elif name == "hors_ue":
            needed = value is None and profile.modele_usage_general is True
        elif name == "hypertrucage":
            needed = (
                value is None and profile.genere_contenus is True and "deployeur" in profile.roles
            )
        else:
            needed = value is None
        if needed:
            missing.append(name)
    return missing


def questions_for(fields: list[str]) -> list[dict]:
    """Questions fermées à poser (format de l'API : options [{valeur, libelle}])."""
    out = []
    for name in fields:
        q = QUESTIONS[name]
        out.append(
            {
                "champ": name,
                "question": q["question"],
                "aide": q.get("aide"),
                "multiple": q.get("multiple", False),
                "options": [{"valeur": v, "libelle": label} for v, label in q["options"]],
            }
        )
    return out


def apply_answers(profile: Profile, answers: dict) -> Profile:
    """Réponses aux questions fermées -> profil. Valeurs inconnues ignorées, « je ne sais pas »
    laisse le champ vide (le rapport le signalera comme incertain)."""
    data = profile.model_dump()
    for name, value in answers.items():
        if name not in QUESTIONS:
            continue
        allowed = {v for v, _ in QUESTIONS[name]["options"]}
        if name == "roles":
            values = value if isinstance(value, list) else [value]
            data["roles"] = [v for v in values if v in allowed]
        elif value == "je_ne_sais_pas" or value not in allowed:
            continue
        elif QUESTIONS[name]["options"] is YES_NO:
            data[name] = value == "oui"
        else:
            data[name] = value
        data["preuves"].pop(name, None)  # la valeur vient de l'utilisateur, plus de l'extrait
    return Profile.model_validate(data)


EXTRACTION_PROMPT = """Tu analyses la description d'un projet d'IA pour en extraire un profil \
juridique. Ne déduis RIEN qui ne soit pas dit explicitement : en cas de doute, mets null.

Définitions :
- fournisseur : développe le système (ou le fait développer) et le met sur le marché ou en \
service sous son propre nom ;
- deployeur : utilise le système sous sa propre autorité dans le cadre d'une activité \
professionnelle ;
- modele_usage_general : l'organisation développe elle-même un modèle d'IA à usage général \
(grand modèle polyvalent proposé à d'autres), pas seulement un système qui en utilise un ;
- interagit_avec_personnes : le système dialogue ou échange des messages directement avec des \
personnes (agent conversationnel, assistant vocal) ; analyser des personnes ou leurs données ne \
suffit pas ;
- donnees_sensibles : catégories particulières de l'article 9 du RGPD (origine raciale ou \
ethnique, opinions politiques, convictions religieuses ou philosophiques, appartenance \
syndicale, données génétiques, données biométriques servant à identifier une personne, santé, \
vie ou orientation sexuelle) ; les données financières, de performance ou de comportement n'en \
font pas partie ;
- deja_sur_le_marche : "non" (pas encore), "avant_2025_08_02", \
"entre_2025_08_02_et_2026_08_02" ou "apres_2026_08_02" ;
- date_mise_sur_le_marche : si la description donne le mois de mise sur le marché ou en \
service, recopie-le au format "AAAA-MM" (sans calcul), sinon null.

DESCRIPTION DU PROJET :
{description}

Pour CHAQUE valeur différente de null (et chaque rôle), donne dans "preuves" l'extrait de la \
description qui l'établit, recopié MOT POUR MOT. Sans extrait qui le dit, mets null.

Réponds en JSON avec exactement ces clés :
{{"finalite": "ce que fait le système, en une phrase", "roles": ["fournisseur" et/ou \
"deployeur" et/ou "importateur" et/ou "distributeur"], "personnes_concernees": ["..."], \
"secteur_public": null, "lieu_de_travail": null, "decisions_sur_personnes": null, \
"interagit_avec_personnes": null, "genere_contenus": null, "hypertrucage": null, \
"biometrie": null, "composant_securite_produit": null, "modele_usage_general": null, \
"hors_ue": null, "donnees_personnelles": null, "donnees_sensibles": null, \
"deja_sur_le_marche": null, "date_mise_sur_le_marche": null, \
"preuves": {{"roles": "extrait exact", "...": "extrait exact"}}}}
Chaque valeur booléenne vaut true, false ou null (non précisé)."""

FREE_TEXT = ("description", "finalite", "personnes_concernees")  # reformulations admises
INTERNAL = ("description", "document", "preuves", "passages")
MAX_PASSAGES = 8
MIN_PASSAGE_WORDS = 5

# Même profil, lu dans un document de plusieurs pages (cahier des charges, documentation,
# analyse d'impact). Il demande en plus des passages clés, recopiés mot pour mot : les étapes
# suivantes reçoivent ces passages et le profil, pas tout le document (limite de tokens/min).
DOCUMENT_PROMPT = (
    EXTRACTION_PROMPT.replace(
        "Tu analyses la description d'un projet d'IA",
        "Tu analyses le document d'un projet d'IA (cahier des charges, documentation, analyse "
        "d'impact)",
    )
    .replace(
        "DESCRIPTION DU PROJET :\n{description}",
        "DOCUMENT DU PROJET (texte extrait du PDF, chaque page commence par [page n]) :\n"
        "{description}",
    )
    .replace("l'extrait de la description qui", "l'extrait du document qui")
    .replace(
        "\nRéponds en JSON avec exactement ces clés :\n{{",
        '\nDonne aussi dans "passages_cles" jusqu\'à 8 passages du document, recopiés MOT POUR '
        "MOT (sans la marque de page), qui décrivent l'usage du système, les personnes "
        "concernées, les données utilisées et les décisions prises.\n\n"
        "Réponds en JSON avec exactement ces clés :\n"
        '{{"passages_cles": ["..."], ',
    )
)
assert DOCUMENT_PROMPT.count("passages_cles") == 2 and "l'extrait du document" in DOCUMENT_PROMPT


def has_evidence(evidence, description: str) -> bool:
    """L'extrait cité figure-t-il mot pour mot dans la description (au moins 2 mots) ?"""
    if not isinstance(evidence, str) or len(evidence.split()) < 2:
        return False
    return normalize(evidence.strip(" .«»\"'")) in normalize(description)


def parse_profile(description: str, data: dict) -> Profile:
    """JSON du LLM -> profil validé.

    Une valeur n'est gardée que si sa preuve (extrait de la description) est retrouvée mot pour
    mot : un petit modèle « devine » volontiers (« déjà sur le marché » pour un projet qui ne
    l'est pas). Sans preuve, la valeur redevient inconnue et l'agent posera la question.
    Une valeur invalide est aussi ramenée à « non précisé ».
    """
    clean: dict = {"description": description}
    evidence = data.get("preuves") if isinstance(data.get("preuves"), dict) else {}
    proofs: dict[str, dict] = {}
    for name in Profile.model_fields:
        if name in INTERNAL or name not in data or data[name] is None:
            continue
        value = data[name]
        if name not in FREE_TEXT and not has_evidence(evidence.get(name), description):
            continue
        if name == "roles" and isinstance(value, list):  # rôle inconnu : ignoré, pas tout le champ
            value = [r for r in value if r in get_args(Role)]
        try:
            Profile.model_validate({**clean, name: value})
        except ValueError:
            continue
        clean[name] = value
        if name not in FREE_TEXT:
            proofs[name] = _proof(evidence[name], description)
    # Période de mise sur le marché : le modèle la calculait mal (« à partir de 2027 » ->
    # « après le 2 août 2026 » au lieu de « pas encore »). Seul « non » est repris tel quel ;
    # une période ne vient que d'une date recopiée de la description, convertie par le code.
    if clean.get("deja_sur_le_marche") not in (None, "non"):
        del clean["deja_sur_le_marche"]
        proofs.pop("deja_sur_le_marche", None)
    if has_evidence(evidence.get("date_mise_sur_le_marche"), description):
        status = market_status(data.get("date_mise_sur_le_marche"))
        if status:
            clean["deja_sur_le_marche"] = status
            proofs["deja_sur_le_marche"] = _proof(evidence["date_mise_sur_le_marche"], description)
    clean["preuves"] = proofs
    clean["passages"] = key_passages(data.get("passages_cles"), description)
    return Profile.model_validate(clean)


def _proof(evidence: str, text: str) -> dict:
    excerpt = evidence.strip(" .«»\"'")
    return {"extrait": excerpt, "page": page_of(excerpt, text)}


def key_passages(passages, text: str) -> list[dict]:
    """Passages clés recopiés du document : seuls ceux retrouvés mot pour mot sont gardés."""
    if not isinstance(passages, list):
        return []
    kept: list[dict] = []
    for passage in passages:
        if not isinstance(passage, str) or len(passage.split()) < MIN_PASSAGE_WORDS:
            continue
        if has_evidence(passage, text) and passage not in [k["extrait"] for k in kept]:
            kept.append(_proof(passage, text))
    return kept[:MAX_PASSAGES]


def market_status(when, today: date | None = None) -> str | None:
    """« 2025-10 » -> « entre_2025_08_02_et_2026_08_02 » ; « 2027 » -> « non » (pas encore).
    Ce qui chevauche une date charnière (août 2025, août 2026, ou l'année seule 2025 ou 2026)
    est ambigu : None, et l'agent pose la question."""
    m = re.match(r"\s*(\d{4})(?:-(\d{2}))?", when) if isinstance(when, str) else None
    if not m:
        return None
    today = today or date.today()
    year = int(m[1])
    if not m[2]:  # année seule
        if year > today.year:
            return "non"
        return "avant_2025_08_02" if year < 2025 else None
    month = (year, int(m[2]))
    if not 1 <= month[1] <= 12 or month in ((2025, 8), (2026, 8)):
        return None
    if month > (today.year, today.month):
        return "non"
    if month < (2025, 8):
        return "avant_2025_08_02"
    return "entre_2025_08_02_et_2026_08_02" if month < (2026, 8) else "apres_2026_08_02"
