"""Faits vérifiables d'un texte juridique : dates, montants en euros, pourcentages.

Ce sont les erreurs les plus graves d'un assistant juridique (une date d'application
périmée, une amende fausse), et les plus faciles à contrôler sans LLM :
- rappel : les faits de la réponse de référence se retrouvent-ils dans la réponse générée ?
- hallucination : chaque fait de la réponse générée figure-t-il dans au moins une source ?
Les écritures sont normalisées : « 20 000 000 EUR » = « 20 millions d'euros » = 20000000.
"""

import re

MONTHS = [
    "janvier",
    "février",
    "mars",
    "avril",
    "mai",
    "juin",
    "juillet",
    "août",
    "septembre",
    "octobre",
    "novembre",
    "décembre",
]
_DATE = re.compile(rf"\b(\d{{1,2}})(?:er)?\s+({'|'.join(MONTHS)})\s+(\d{{4}})\b", re.IGNORECASE)
_NUMBER = r"\d{1,3}(?:[\s  .]\d{3})+|\d+(?:,\d+)?"
_MONEY = re.compile(
    rf"({_NUMBER})\s*(millions?|milliards?)?\s*(?:d['’]\s*)?(?:EUR|€|euros?)\b", re.IGNORECASE
)
_PERCENT = re.compile(r"(\d+(?:,\d+)?)\s*%")
_SCALE = {"million": 10**6, "millions": 10**6, "milliard": 10**9, "milliards": 10**9}


def _to_number(text: str) -> float:
    return float(re.sub(r"[\s  .]", "", text).replace(",", "."))


def extract(text: str) -> set[str]:
    """Faits normalisés : « date:2027-12-02 », « eur:20000000 », « pct:4 »."""
    facts = set()
    for day, month, year in _DATE.findall(text):
        facts.add(f"date:{year}-{MONTHS.index(month.lower()) + 1:02d}-{int(day):02d}")
    for number, scale in _MONEY.findall(text):
        value = _to_number(number) * _SCALE.get(scale.lower(), 1)
        facts.add(f"eur:{value:.0f}")
    for number in _PERCENT.findall(text):
        facts.add(f"pct:{_to_number(number):g}")
    return facts


def humanize(fact: str) -> str:
    """« date:2026-08-02 » -> « 2 août 2026 » ; « eur:20000000 » -> « 20 000 000 EUR »."""
    kind, value = fact.split(":", 1)
    if kind == "date":
        year, month, day = value.split("-")
        return f"{int(day)} {MONTHS[int(month) - 1]} {year}"
    if kind == "eur":
        return f"{int(value):,} EUR".replace(",", " ")
    return f"{value} %"
