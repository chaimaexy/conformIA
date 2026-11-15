"""Classification AI Act : arbre de décision en code, le LLM ne juge que des faits.

L'ordre est fixé par le code : pratique interdite (article 5), puis haut risque (annexe III, avec
la dérogation de l'article 6, paragraphe 3 ; annexe I), puis transparence (article 50), puis
modèle à usage général (chapitre V).

Article 5 : un premier appel repère les points candidats ; un second juge, pour chacun, les
conditions fixées dans interdictions.yaml (cumulatives, alternatives, exceptions), et le code en
déduit le verdict. Laissé libre, un petit modèle découpait mal le texte : il rendait cumulatives
les deux branches du point c) ou inventait une condition (« le système est déjà en service »).
Le rapport cite le texte exact du point, lu par le code.

Annexe III : le LLM choisit un point et recopie le passage qui le décrit. Le code retrouve le
point d'où vient ce passage (au moins 8 mots consécutifs du texte), corrige un numéro mal
recopié et cite le texte exact du point. Sans ancrage, un seul nouvel essai (boucle corrective
bornée), puis « incertain ».

Le reste se déduit du profil sans LLM : l'article 50 vise des faits simples (le système dialogue,
génère des contenus, reconnaît des émotions), comme le fait de développer un modèle à usage général.
"""

import json
import re
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path

import yaml

from app.agents.profile import Profile
from app.rag.verify import longest_run, quote_found

AskJson = Callable[[str], dict]
VERDICTS = ("oui", "non", "incertain")
STATUSES = ("etablie", "non", "inconnue")
ADDED_PROHIBITIONS = ("b bis", "b ter")  # article 113 : applicables à partir du 2 décembre 2026
CONDITIONS_FILE = Path(__file__).with_name("interdictions.yaml")
GROUPS = {
    "toutes": "conditions cumulatives",
    "au_moins_une": "au moins une de ces conditions",
    "sauf": "exception (si elle est établie, la pratique n'est pas interdite)",
}
MIN_ANCHOR_WORDS = 8

PROHIBITED_PROMPT = """Tu es juriste spécialisé en droit de l'IA. Repère les points de \
l'article 5 du règlement sur l'IA (pratiques interdites) dont le projet ci-dessous pourrait \
relever, en t'appuyant UNIQUEMENT sur le texte fourni. Chaque point retenu sera ensuite vérifié \
condition par condition : retiens un point dès que l'usage décrit correspond à son objet, mais \
n'en invente pas.

PROFIL DU PROJET :
{profile}

TEXTE DE L'ARTICLE 5 :
{text}

Réponds en JSON : {{"points": [{{"point": "f)", "raison": "..."}}]}} (liste vide si aucun point \
ne correspond)."""

CONFIRM_PROMPT = """Tu es juriste spécialisé en droit de l'IA. Pour chaque condition ci-dessous, \
tirée de l'article 5 du règlement sur l'IA, indique si le projet la remplit :
- "etablie" : la description ou le profil montre qu'elle est remplie ;
- "non" : ils montrent qu'elle ne l'est pas (par exemple, la finalité décrite est autre) ;
- "inconnue" : ils ne permettent pas de le savoir.
Ne juge que la condition posée, sans en ajouter.

PROFIL DU PROJET :
{profile}

CONDITIONS :
{conditions}

Réponds en JSON, une entrée par condition : {{"conditions": [{{"id": "c.1", \
"statut": "etablie|non|inconnue", "raison": "..."}}]}}"""

HIGH_RISK_PROMPT = """Tu es juriste spécialisé en droit de l'IA. Détermine si le projet \
ci-dessous est un système d'IA à haut risque au sens de l'article 6, paragraphe 2, et de \
l'annexe III du règlement sur l'IA, et si la dérogation de l'article 6, paragraphe 3, peut \
s'appliquer. Appuie-toi UNIQUEMENT sur les textes fournis.

PROFIL DU PROJET :
{profile}

TEXTES :
{text}

Règles :
- "annexe_iii" : "oui" si l'usage prévu correspond à un point de l'annexe III ; "incertain" si \
cela dépend d'informations absentes du profil ; "non" sinon.
- "point" : numéro et lettre du point de l'annexe III (exemple : "4 a)"), ou null.
- "citation" : le passage de ce point recopié MOT POUR MOT (au moins 8 mots consécutifs).
- "derogation" : "oui" seulement si le profil établit qu'une condition de l'article 6, \
paragraphe 3, est remplie ; "incertain" si c'est possible mais non établi ; "non" sinon. Un \
système qui effectue un profilage de personnes physiques est toujours à haut risque.
- "citation_derogation" : le passage de l'article 6 recopié MOT POUR MOT, ou null.
{feedback}
Réponds en JSON :
{{"annexe_iii": "oui|non|incertain", "point": "4 a)", "citation": "...", "raison": "...", \
"derogation": "oui|non|incertain", "condition": "a)|b)|c)|d)|null", \
"citation_derogation": "...", "information_manquante": "ce qu'il faudrait savoir, ou null"}}"""

FEEDBACK = (
    "\nATTENTION : ta réponse précédente citait un passage absent de l'annexe III : {quote}. "
    "Recopie le passage du point exactement, mot pour mot, ou mets le verdict à "
    '"incertain".\n'
)


def profile_summary(profile: Profile) -> str:
    """Profil envoyé aux étapes suivantes. Pour un document, pas ses 30 pages (limite de tokens
    par minute) : les extraits recopiés qui prouvent le profil, et les passages clés."""
    data = profile.model_dump(exclude_none=True, exclude={"preuves", "passages", "document"})
    if profile.document:
        data.pop("description")
        excerpts = [p for p in profile.passages] + list(profile.preuves.values())
        seen = []
        for e in excerpts:
            line = f"(page {e['page']}) {e['extrait']}" if e.get("page") else e["extrait"]
            if line not in seen:
                seen.append(line)
        data["extraits_du_document"] = seen
    data = {k: v for k, v in data.items() if v not in ([], "")}
    return json.dumps(data, ensure_ascii=False, indent=1)


def _verdict(value) -> str:
    return value if value in VERDICTS else "incertain"


def annex_point(value) -> str | None:
    """« 4 a) », « point 4, a) », « annexe III, point 5 b) » -> « 4 a) », « 5 b) »."""
    if not value or not isinstance(value, str):
        return None
    m = re.search(r"(\d+)\s*[,.]?\s*(?:point\s*)?([a-h])?\s*\)?\s*$", value.strip())
    if not m:
        return value.strip()
    return f"{m[1]} {m[2]})" if m[2] else m[1]


def article5_point(value) -> str | None:
    """« f) », « f », « article 5, paragraphe 1, point f) », « b bis) » -> « f) », « b bis) »."""
    if not value or not isinstance(value, str):
        return None
    m = re.search(r"(?<![a-z])([a-h](?: bis| ter)?)\s*\)", value) or re.fullmatch(
        r"\s*(?:point\s*)?\(?([a-h](?: bis| ter)?)\s*", value
    )
    return f"{m[1]})" if m else value.strip()


NEXT_POINT = re.compile(r"^([a-h](?: bis| ter)?\)|\d+(?: bis| ter)?\.) ")


def point_text(article5: str, point: str | None) -> str | None:
    """Texte d'un point de l'article 5, paragraphe 1 (« c) » -> le point c) et ses sous-points)."""
    if not point:
        return None
    letter = point.removesuffix(")")
    lines = article5.splitlines()
    start = next((i for i, line in enumerate(lines) if line.startswith(f"{letter}) ")), None)
    if start is None:
        return None
    out = [lines[start]]
    for line in lines[start + 1 :]:
        if NEXT_POINT.match(line):
            break
        out.append(line)
    return "\n".join(out)


# ----------------------------------------------------------------------------- article 5
@lru_cache
def load_conditions() -> dict[str, dict[str, list[dict]]]:
    return yaml.safe_load(CONDITIONS_FILE.read_text(encoding="utf-8"))


def verify_conditions(article5: str) -> list[str]:
    """Extraits de interdictions.yaml introuvables dans le texte de l'article 5."""
    errors = []
    for point, groups in load_conditions().items():
        for group, items in groups.items():
            if group not in GROUPS:
                errors.append(f"article 5 {point} : groupe inconnu « {group} »")
            for item in items:
                if not quote_found(item["extrait"], [article5]):
                    errors.append(f"article 5 {point} : extrait introuvable « {item['extrait']} »")
    return errors


def conditions_of(point: str) -> list[dict]:
    """Conditions d'un point, numérotées « c.1 », « c.2 »... (« b bis) » -> « b_bis.1 »)."""
    prefix = point.removesuffix(")").replace(" ", "_")
    items = [
        {"groupe": group, **item}
        for group in GROUPS
        for item in load_conditions()[point].get(group) or []
    ]
    return [{"id": f"{prefix}.{n}", **item} for n, item in enumerate(items, start=1)]


def _conditions_prompt(points: list[str]) -> str:
    lines = []
    for point in points:
        conditions = conditions_of(point)
        for group, label in GROUPS.items():
            selected = [c for c in conditions if c["groupe"] == group]
            if selected:
                lines.append(f"Point {point}, {label} :")
                lines += [
                    f"- [{c['id']}] {c['question']} (texte : « {c['extrait']} »)" for c in selected
                ]
    return "\n".join(lines)


def practice_verdict(conditions: list[dict]) -> str:
    """Verdict d'un point à partir du statut de ses conditions (décidé par le code)."""

    def statuses(group):
        return [c["statut"] for c in conditions if c["groupe"] == group]

    required, alternatives, exceptions = (statuses(g) for g in GROUPS)
    if "non" in required or (alternatives and all(s == "non" for s in alternatives)):
        return "non"
    if "etablie" in exceptions:
        return "non"
    if "etablie" not in required + alternatives:
        # aucun élément établi : un soupçon du premier examen, pas une pratique à signaler
        return "non"
    if (
        all(s == "etablie" for s in required)
        and (not alternatives or "etablie" in alternatives)
        and all(s == "non" for s in exceptions)
    ):
        return "oui"
    return "incertain"


def classify_prohibited(profile: Profile, text: str, ask: AskJson) -> dict:
    summary = profile_summary(profile)
    known = load_conditions()
    candidates: dict[str, str] = {}
    data = ask(PROHIBITED_PROMPT.format(profile=summary, text=text))
    for p in data.get("points") or data.get("elements") or []:
        point = article5_point(p.get("point")) if isinstance(p, dict) else None
        if point in known and point not in candidates:  # point inventé : ignoré
            candidates[point] = str(p.get("raison") or "")
    if not candidates:
        return {"verdict": "non", "pratiques": [], "points_examines": []}

    prompt = CONFIRM_PROMPT.format(profile=summary, conditions=_conditions_prompt(list(candidates)))
    answers = {}
    data = ask(prompt)
    for c in data.get("conditions") or data.get("elements") or []:
        if isinstance(c, dict) and c.get("statut") in STATUSES:
            answers[str(c.get("id") or "").strip(" []")] = c
    examined = []
    for point, reason in candidates.items():
        conditions = [
            {
                **c,
                "statut": answers.get(c["id"], {}).get("statut", "inconnue"),
                "raison": str(answers.get(c["id"], {}).get("raison") or ""),
            }
            for c in conditions_of(point)
        ]
        examined.append(
            {
                "point": point,
                "verdict": practice_verdict(conditions),
                # texte exact du point, lu par le code (pas recopié par le modèle)
                "citation": point_text(text, point) or " ".join(c["extrait"] for c in conditions),
                "raison": reason,
                "conditions": conditions,
            }
        )
    practices = [p for p in examined if p["verdict"] != "non"]
    if any(p["verdict"] == "oui" for p in practices):
        verdict = "oui"
    elif practices:
        verdict = "incertain"
    else:
        verdict = "non"
    return {
        "verdict": verdict,
        "pratiques": practices,
        "points_examines": [{"point": p["point"], "verdict": p["verdict"]} for p in examined],
    }


# ----------------------------------------------------------------------------- annexe III
ANNEX_DOMAIN = re.compile(r"^(\d+)\. ")
ANNEX_LETTER = re.compile(r"^([a-h])\) ")


def annex_points(text: str) -> dict[str, str]:
    """Texte de chaque point de l'annexe III : {"4 a)": "...", "2": "..."}."""
    lines = text.splitlines()
    start = next((i for i, line in enumerate(lines) if "Annexe III" in line), len(lines))
    points: dict[str, str] = {}
    domain = current = None
    for raw in lines[start + 1 :]:
        line = raw.strip()
        if not line or line.startswith("AI Act"):  # en-tête de partie
            continue
        if m := ANNEX_DOMAIN.match(line):
            domain = current = m[1]
            points[current] = line
        elif (m := ANNEX_LETTER.match(line)) and domain:
            current = f"{domain} {m[1]})"
            points[current] = line
        elif current:  # suite du point (« Cela n'inclut pas... »)
            points[current] += "\n" + line
    return points


def locate_point(quote, points: dict[str, str], claimed: str | None) -> str | None:
    """Point de l'annexe III d'où vient la citation (au moins 8 mots consécutifs).

    Plusieurs points partagent leur début (« systèmes d'IA destinés à être utilisés par les
    autorités répressives... ») : on garde celui qui partage le plus long passage, et le point
    annoncé par le modèle en cas d'égalité."""
    if not isinstance(quote, str):
        return None
    runs = {p: longest_run(quote, t) for p, t in points.items()}
    best = max(runs.values(), default=0)
    if best < MIN_ANCHOR_WORDS:
        return None
    top = [p for p, n in runs.items() if n == best]
    return claimed if claimed in top else top[0]


def classify_high_risk(profile: Profile, text: str, ask: AskJson) -> dict:
    points = annex_points(text)
    values = {"profile": profile_summary(profile), "text": text}
    data = ask(HIGH_RISK_PROMPT.format(feedback="", **values))
    claimed = annex_point(data.get("point"))
    anchor = locate_point(data.get("citation"), points, claimed)
    retried = False
    if _verdict(data.get("annexe_iii")) != "non" and anchor is None:
        retried = True
        quote = json.dumps(str(data.get("citation") or ""), ensure_ascii=False)
        data = ask(HIGH_RISK_PROMPT.format(feedback=FEEDBACK.format(quote=quote), **values))
        claimed = annex_point(data.get("point"))
        anchor = locate_point(data.get("citation"), points, claimed)
    verdict = _verdict(data.get("annexe_iii"))
    if verdict == "oui" and anchor is None:
        verdict = "incertain"
    point = None if verdict == "non" else anchor or (claimed if claimed in points else None)
    if verdict == "incertain" and point is None:
        # le haut risque de l'annexe III suppose un point précis : un doute qui n'en désigne
        # aucun (« ne correspond à aucun point, mais... ») ne fonde pas un « incertain »
        verdict = "non"
    derogation_quote = data.get("citation_derogation")
    derogation_ok = bool(derogation_quote) and quote_found(str(derogation_quote), [text])
    derogation = _verdict(data.get("derogation"))
    if derogation == "oui" and not derogation_ok:
        derogation = "incertain"
    return {
        "verdict": verdict,
        "point": point,
        # texte exact du point, lu par le code ; la citation du modèle a servi à le retrouver
        "citation": points.get(point, "") if point else "",
        "citation_verifiee": anchor is not None,
        "point_annonce": claimed,
        "nouvel_essai": retried,
        "raison": str(data.get("raison") or ""),
        "derogation": derogation if verdict != "non" else "non",
        "condition": data.get("condition") if derogation != "non" else None,
        "citation_derogation": derogation_quote if derogation_ok else None,
        "information_manquante": data.get("information_manquante") or None,
    }


# ----------------------------------------------------------------------------- sans LLM
def classify_from_profile(profile: Profile) -> dict:
    """Ce que le profil suffit à établir : article 50, annexe I, modèle à usage général."""
    provider = "fournisseur" in profile.roles
    deployer = "deployeur" in profile.roles
    transparency = []
    if provider and profile.interagit_avec_personnes:
        transparency.append(1)
    if provider and profile.genere_contenus:
        transparency.append(2)
    if deployer and profile.biometrie:
        transparency.append(3)
    if deployer and profile.hypertrucage:
        transparency.append(4)
    return {
        "transparence": transparency,
        # annexe I : il faut aussi que le produit soit soumis à une évaluation par un tiers
        "annexe_i": "a_confirmer" if profile.composant_securite_produit else "non",
        "modele_usage_general": bool(provider and profile.modele_usage_general),
    }


def categories(profile: Profile, prohibited: dict, high_risk: dict, simple: dict) -> dict:
    """Catégories de la carte des obligations + niveau de risque retenu pour le rapport."""
    cats = {"tous"}
    if prohibited["verdict"] == "oui":
        cats.add("interdit")
    annex_iii = high_risk.get("verdict") == "oui" and high_risk.get("derogation") != "oui"
    if annex_iii:
        cats |= {"haut_risque", "haut_risque_annexe_iii"}
    if high_risk.get("verdict") == "oui" and high_risk.get("derogation") == "oui":
        cats.add("derogation_6_3")
    if simple["annexe_i"] == "a_confirmer":
        cats.add("haut_risque_annexe_i")
    cats |= {f"transparence_{n}" for n in simple["transparence"]}
    if simple["modele_usage_general"]:
        cats.add("modele_usage_general")
    point = high_risk.get("point") or ""
    flags = {
        "secteur_public": bool(profile.secteur_public),
        "lieu_de_travail": bool(profile.lieu_de_travail),
        "decisions_sur_personnes": bool(profile.decisions_sur_personnes),
        "hors_ue": bool(profile.hors_ue),
        "annexe_iii_5bc": bool(re.match(r"5 [bc]\)", point)),
    }
    if "interdit" in cats:
        level = "interdit"
    elif annex_iii:
        level = "haut_risque"
    elif "haut_risque_annexe_i" in cats:
        level = "haut_risque_a_confirmer"
    elif "incertain" in (prohibited["verdict"], high_risk.get("verdict")):
        level = "incertain"
    elif simple["transparence"] or simple["modele_usage_general"]:
        level = "obligations_specifiques"
    else:
        level = "risque_minimal"
    return {"categories": sorted(cats), "conditions": flags, "niveau": level}
