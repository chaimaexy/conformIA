"""Vérification des citations d'une réponse générée, par du code (sans LLM).

Ce qu'on peut vérifier de façon certaine :
- chaque renvoi [n] désigne bien une source fournie au modèle ;
- chaque citation mot pour mot « ... » figure réellement dans l'une des sources ;
- la réponse cite au moins une source, et on repère les phrases qui n'en citent aucune ;
- chaque date, montant ou pourcentage de la réponse figure dans au moins une source
  (sinon le modèle l'a tiré de sa mémoire : c'est ainsi qu'il a donné « 2 août 2026 »
  au lieu du 2 août 2028 pour un jouet, quand l'article 113 manquait au contexte).
Ce qu'on ne peut pas vérifier ainsi : qu'une paraphrase est fidèle à sa source. C'est
l'objet de l'évaluation (faits clés, et juge LLM sur échantillon).
"""

import re
from dataclasses import dataclass, field

from app.rag.facts import extract

MARKER = re.compile(r"\[(\d+(?:\s*[,;]\s*\d+)*)\]")
QUOTE = re.compile(r"«\s*([^«»]+?)\s*»")
MIN_QUOTE_WORDS = 4  # en dessous, « haut risque » n'est pas une citation à vérifier
# coupure dans une citation : « début [...] fin » -> chaque morceau est vérifié
ELLIPSIS = re.compile(r"\[\s*(?:\.\.\.|…)\s*\]|\(\s*(?:\.\.\.|…)\s*\)|\.\.\.|…")
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-ZÀ-Ý«(])")


def normalize(text: str) -> str:
    text = text.replace("’", "'").replace("*", "").replace(" ", " ").replace(" ", " ")
    return re.sub(r"\s+", " ", text).strip().lower()


def cited_numbers(answer: str) -> list[int]:
    """Numéros cités, dans l'ordre de première apparition."""
    seen: list[int] = []
    for m in MARKER.finditer(answer):
        for n in re.findall(r"\d+", m[1]):
            if int(n) not in seen:
                seen.append(int(n))
    return seen


@dataclass
class Verification:
    cited: list[int] = field(default_factory=list)  # renvois valides
    invalid_markers: list[int] = field(default_factory=list)  # [n] sans source n
    quotes_checked: int = 0
    unverified_quotes: list[str] = field(default_factory=list)  # « ... » introuvables
    uncited_sentences: list[str] = field(default_factory=list)
    unsourced_facts: list[str] = field(default_factory=list)  # dates, montants hors sources

    @property
    def ok(self) -> bool:
        return (
            bool(self.cited)
            and not self.invalid_markers
            and not self.unverified_quotes
            and not self.unsourced_facts
        )


def quote_found(quote: str, sources: list[str]) -> bool:
    """Une citation (coupée ou non par « [...] ») figure-t-elle mot pour mot dans les sources ?
    Une citation trop courte pour être vérifiée (moins de 4 mots par morceau) est refusée."""
    fragments = [f for f in ELLIPSIS.split(quote) if len(f.split()) >= MIN_QUOTE_WORDS]
    haystack = [normalize(s) for s in sources]
    return bool(fragments) and all(any(normalize(f) in s for s in haystack) for f in fragments)


def longest_run(quote: str, text: str) -> int:
    """Nombre de mots du plus long passage de la citation présent tel quel dans le texte.

    Plus tolérant que quote_found : un petit modèle abrège ou complète volontiers une longue
    phrase (« ... de telles relations professionnelles contractuelles »). Sert à ancrer un choix
    (quel point de l'annexe III ?) ; le texte affiché vient alors du code, pas de la citation."""
    words = normalize(quote).split()
    haystack = normalize(text)
    best = 0
    for i in range(len(words)):
        j = i + best + 1  # on ne cherche que plus long que le meilleur déjà trouvé
        while j <= len(words) and " ".join(words[i:j]) in haystack:
            best, j = j - i, j + 1
    return best


def verify(answer: str, sources: list[str]) -> Verification:
    """`sources[i]` est le texte de la source numérotée i + 1."""
    numbers = cited_numbers(answer)
    result = Verification(
        cited=[n for n in numbers if 1 <= n <= len(sources)],
        invalid_markers=[n for n in numbers if not 1 <= n <= len(sources)],
    )
    haystack = [normalize(s) for s in sources]
    for quote in QUOTE.findall(answer):
        fragments = [f for f in ELLIPSIS.split(quote) if len(f.split()) >= MIN_QUOTE_WORDS]
        if not fragments:
            continue
        result.quotes_checked += 1
        if not all(any(normalize(f) in s for s in haystack) for f in fragments):
            result.unverified_quotes.append(quote)
    for line in answer.splitlines():  # paragraphes et puces, puis phrases
        for sentence in _SENTENCE.split(line.strip()):
            if len(sentence) > 60 and not MARKER.search(sentence):
                result.uncited_sentences.append(sentence)
    source_facts = set().union(*(extract(s) for s in sources)) if sources else set()
    result.unsourced_facts = sorted(extract(answer) - source_facts)
    return result
