"""Liste de vérification de la CNIL (11 thèmes RGPD) appliquée au projet, par lots.

Un appel par thème coûterait 11 appels (plus de 5 minutes à 2 requêtes/min) ; tous les thèmes
en un appel donneraient un prompt où un petit modèle se perd. Compromis mesuré : 3 lots.
Chaque thème est évalué contre son texte exact (lu par son numéro), et la citation renvoyée est
vérifiée par le code. Sans données personnelles, le RGPD ne s'applique pas : aucun appel.
"""

from collections.abc import Callable

from app.agents.classification import profile_summary
from app.agents.profile import Profile
from app.rag.verify import quote_found

BATCHES = ((1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11))
STATUSES = ("non_applicable", "a_verifier", "point_d_attention", "conforme")

PROMPT = """Tu es délégué à la protection des données. Pour chaque thème de la liste de \
vérification de la CNIL ci-dessous, évalue le projet décrit par le profil, en t'appuyant \
UNIQUEMENT sur le texte des thèmes.

PROFIL DU PROJET :
{profile}

THÈMES DE LA LISTE DE VÉRIFICATION :
{themes}

Pour chaque thème, donne :
- "statut" :
  - "a_verifier" PAR DÉFAUT, quand le profil ne dit pas si c'est fait ;
  - "point_d_attention" seulement si un élément PRÉSENT dans le profil aggrave le risque pour \
ce thème (données sensibles, décisions automatisées sur des personnes, personnes vulnérables, \
réutilisation de données, biométrie) ; au plus 4 thèmes sur l'ensemble ;
  - "conforme" seulement si le profil dit explicitement que c'est fait ;
  - "non_applicable" si le thème ne concerne pas ce projet ;
- "constat" : une phrase sur ce que le thème implique pour ce projet. N'affirme RIEN sur le \
projet que le profil ne dit pas : si le profil ne dit pas si les personnes sont informées, écris \
qu'il faut le vérifier, pas qu'elles ne le sont pas ;
- "actions" : 1 à 3 actions concrètes tirées du texte du thème ;
- "citation" : une phrase du thème recopiée MOT POUR MOT (au moins 8 mots consécutifs).

Réponds en JSON :
{{"themes": [{{"numero": 1, "statut": "a_verifier", "constat": "...", "actions": ["..."], \
"citation": "..."}}]}}"""


def evaluate(
    profile: Profile, themes: dict[int, dict], ask: Callable[[str], dict]
) -> tuple[list[dict], int]:
    """themes : numéro -> {"titre", "texte", "url"}. Renvoie (résultats, nombre d'appels)."""
    if profile.donnees_personnelles is False:
        results = [
            _result(n, themes[n], "non_applicable", "Aucune donnée personnelle.") for n in themes
        ]
        return results, 0
    results, calls = [], 0
    for batch in BATCHES:
        numbers = [n for n in batch if n in themes]
        if not numbers:
            continue
        block = "\n\n".join(
            f"THÈME {n} : {themes[n]['titre']}\n{themes[n]['texte']}" for n in numbers
        )
        data = ask(PROMPT.format(profile=profile_summary(profile), themes=block))
        calls += 1
        answers = {}
        for item in data.get("themes") or data.get("elements") or []:
            if isinstance(item, dict) and isinstance(item.get("numero"), int):
                answers[item["numero"]] = item
        for n in numbers:
            results.append(_from_answer(n, themes[n], answers.get(n)))
    return results, calls


def _result(n: int, theme: dict, status: str, finding: str, **extra) -> dict:
    return {
        "numero": n,
        "titre": theme["titre"],
        "url": theme.get("url"),
        "statut": status,
        "constat": finding,
        "actions": extra.get("actions", []),
        "citation": extra.get("citation"),
        "citation_verifiee": extra.get("citation_verifiee", False),
    }


def _from_answer(n: int, theme: dict, answer: dict | None) -> dict:
    if not answer:
        return _result(n, theme, "a_verifier", "Thème non évalué par le modèle : à vérifier.")
    status = answer.get("statut") if answer.get("statut") in STATUSES else "a_verifier"
    quote = str(answer.get("citation") or "")
    ok = quote_found(quote, [theme["texte"]])
    if status == "conforme" and not ok:
        status = "a_verifier"  # « conforme » doit s'appuyer sur le texte du thème
    actions = [str(a) for a in answer.get("actions") or [] if str(a).strip()][:3]
    return _result(
        n,
        theme,
        status,
        str(answer.get("constat") or ""),
        actions=actions,
        citation=quote if ok else None,
        citation_verifiee=ok,
    )
