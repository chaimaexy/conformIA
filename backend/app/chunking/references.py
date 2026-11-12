"""Résolution des renvois internes : « chapitre II » -> « chapitre II (Pratiques interdites...) ».

Un texte juridique renvoie à ses propres divisions par leur numéro. L'article 113 de l'AI Act
dit « les chapitres I et II s'appliquent à partir du 2 février 2025 » sans jamais écrire
« pratiques interdites » : ni l'embedding ni BM25 ne peuvent relier la question « depuis quand
les pratiques interdites s'appliquent-elles ? » à ce passage.

On ajoute donc à l'en-tête de contexte du chunk le titre des chapitres, sections et annexes
qu'il cite, tirés du texte lui-même (tables des matières), sans rien inventer ni modifier le
corps du texte. Seuls les renvois au MÊME texte sont résolus : « l'annexe II du règlement
(UE) 2019/1020 » n'est pas l'annexe II de l'AI Act.
"""

import re

from app.ingestion.models import LegalSection

ROMAN = r"[IVXLC]+"
_TITLE_SPLIT = re.compile(r"^(?:CHAPITRE|SECTION)\s+(\S+)\s+-\s+(.+)$")
# « chapitres I et II », « chapitre III, sections 1, 2 et 3 », « annexes I à III »
_CHAPTERS = re.compile(rf"\bchapitres?\s+({ROMAN}(?:\s*(?:,|et|à)\s*{ROMAN})*)\b(?!\s*bis)")
_CHAPTER_SECTIONS = re.compile(
    rf"\bchapitre\s+({ROMAN}),\s+sections?\s+(\d+(?:\s*(?:,|et|à)\s*\d+)*)"
)
_ANNEXES = re.compile(rf"\bannexes?\s+({ROMAN}(?:\s*(?:,|et|à)\s*{ROMAN})*)\b")
# « article 4 bis », « articles 53 et 55 », « articles 12 à 22 » (pas « article 5, paragraphe 1 »)
_SUFFIX = r"(?:\s+(?:bis|ter|quater|quinquies|sexies|septies|octies))?"
_ARTICLES = re.compile(rf"\barticles?\s+(\d+{_SUFFIX}(?:\s*(?:,|et|à)\s*\d+{_SUFFIX})*)\b")
_ARTICLE_NUMBER = re.compile(rf"\d+{_SUFFIX}|à")
MAX_ARTICLE_RANGE = 15  # « articles 25 à 39 » : au-delà, la plage n'est pas développée
# renvoi vers un autre acte juste après le numéro : « annexe I du règlement (UE) 2019/1020 »
_EXTERNAL = re.compile(
    r"^[^.;]{0,40}?\b(?:du|de la|de l’|de l')\s+(?:règlement|directive|décision|traité|charte)",
    re.IGNORECASE,
)


def _sentence_case(title: str) -> str:
    if title.isupper():  # titres de chapitres de l'AI Act en capitales
        title = title.capitalize()
        title = re.sub(r"\b(ia|ue)\b", lambda m: m.group(1).upper(), title)
    return title.strip()


def _roman_to_int(roman: str) -> int:
    values = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}
    total = 0
    for a, b in zip(roman, roman[1:] + " ", strict=True):
        total += -values[a] if b != " " and values[a] < values[b] else values[a]
    return total


def _int_to_roman(n: int) -> str:
    out = ""
    for value, sym in ((100, "C"), (90, "XC"), (50, "L"), (40, "XL"), (10, "X"), (9, "IX")):
        while n >= value:
            out, n = out + sym, n - value
    for value, sym in ((5, "V"), (4, "IV"), (1, "I")):
        while n >= value:
            out, n = out + sym, n - value
    return out


def _expand(listing: str, roman: bool) -> list[str]:
    """« I et II » -> [I, II] ; « 1 à 3 » -> [1, 2, 3] ; « I à III » -> [I, II, III]."""
    tokens = re.findall(rf"{ROMAN}|\d+|à", listing) if roman else re.findall(r"\d+|à", listing)
    items: list[str] = []
    for i, tok in enumerate(tokens):
        if tok == "à" and items and i + 1 < len(tokens):
            start = _roman_to_int(items[-1]) if roman else int(items[-1])
            end = _roman_to_int(tokens[i + 1]) if roman else int(tokens[i + 1])
            items += [_int_to_roman(n) if roman else str(n) for n in range(start + 1, end)]
        elif tok != "à":
            items.append(tok)
    return items


def _expand_articles(listing: str) -> list[str]:
    """« 53 et 55 » -> [53, 55] ; « 12 à 22 » -> [12, ..., 22] ; « 4 bis » reste « 4 bis »."""
    tokens = [" ".join(t.split()) for t in _ARTICLE_NUMBER.findall(listing)]
    items: list[str] = []
    for i, tok in enumerate(tokens):
        if tok == "à" and items and i + 1 < len(tokens):
            start, end = items[-1].split()[0], tokens[i + 1].split()[0]
            if start.isdigit() and end.isdigit() and int(end) - int(start) < MAX_ARTICLE_RANGE:
                items += [str(n) for n in range(int(start) + 1, int(end))]
        elif tok != "à":
            items.append(tok)
    return items


class ReferenceResolver:
    """Titres des chapitres, sections, annexes (et articles) d'UN texte, pour ses renvois."""

    def __init__(self, sections: list[LegalSection], articles: bool = False):
        self.chapters: dict[str, str] = {}
        self.sections: dict[tuple[str, str], str] = {}
        self.annexes: dict[str, str] = {}
        self.articles: dict[str, str] = {}
        self.resolve_articles = articles
        for s in sections:
            chapter = _TITLE_SPLIT.match(s.chapter or "")
            if s.section_type == "article" and s.title:
                self.articles[s.number] = s.title
            if s.section_type == "article" and chapter:
                self.chapters[chapter[1]] = _sentence_case(chapter[2])
                if section := _TITLE_SPLIT.match(s.section or ""):
                    self.sections[(chapter[1], section[1])] = _sentence_case(section[2])
            elif s.section_type == "annexe" and s.title:
                self.annexes[s.number] = s.title

    def resolve(self, text: str, own_article: str | None = None) -> list[str]:
        """Renvois internes trouvés dans le texte, avec leur titre, dans l'ordre d'apparition.

        `own_article` : numéro de l'article cité depuis lui-même, ignoré.
        """
        found: list[tuple[int, str]] = []
        if self.resolve_articles:
            for m in _ARTICLES.finditer(text):
                if _EXTERNAL.match(text[m.end() :]):
                    continue
                for n in _expand_articles(m[1]):
                    if n != own_article and (title := self.articles.get(n)):
                        found.append((m.start(), f"article {n} ({title})"))
        for m in _CHAPTER_SECTIONS.finditer(text):
            for n in _expand(m[2], roman=False):
                if title := self.sections.get((m[1], n)):
                    found.append((m.start(), f"chapitre {m[1]}, section {n} ({title})"))
        for pattern, table, name in (
            (_CHAPTERS, self.chapters, "chapitre"),
            (_ANNEXES, self.annexes, "annexe"),
        ):
            for m in pattern.finditer(text):
                if _EXTERNAL.match(text[m.end() :]):
                    continue
                for n in _expand(m[1], roman=True):
                    if title := table.get(n):
                        found.append((m.start(), f"{name} {n} ({title})"))
        return _unique(found)

    def cited_sections(self, text: str, own_article: str | None = None) -> list[str]:
        """Graphe des citations : libellés des articles et annexes du même texte cités
        (« Article 15 », « Annexe III »), pour joindre ces sections au contexte du LLM."""
        found: list[tuple[int, str]] = []
        for m in _ARTICLES.finditer(text):
            if not _EXTERNAL.match(text[m.end() :]):
                found += [
                    (m.start(), f"Article {n}")
                    for n in _expand_articles(m[1])
                    if n != own_article and n in self.articles
                ]
        for m in _ANNEXES.finditer(text):
            if not _EXTERNAL.match(text[m.end() :]):
                annexes = [n for n in _expand(m[1], roman=True) if n in self.annexes]
                found += [(m.start(), f"Annexe {n}") for n in annexes]
        return _unique(found)


def _unique(found: list[tuple[int, str]]) -> list[str]:
    """Dans l'ordre d'apparition, sans doublon."""
    seen: set[str] = set()
    ordered = []
    for _, ref in sorted(found):
        if ref not in seen:
            seen.add(ref)
            ordered.append(ref)
    return ordered
