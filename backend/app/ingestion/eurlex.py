"""Parsing des textes EUR-Lex (AI Act, RGPD) en sections juridiques.

Deux formats HTML coexistent :

| Élément        | Journal officiel (JO)            | Version consolidée                  |
|----------------|----------------------------------|-------------------------------------|
| considérant    | div#rct_N (tableau « (N) | ... ») | absent (pas de préambule)           |
| article        | div#art_N                        | div#art_N, div#art_4a               |
| paragraphe     | div#006.001                      | div.norm > span.no-parag            |
| énumération    | tableau « a) | texte »           | div.grid-container (2 colonnes)     |
| chapitre       | div#cpt_III > p.oj-ti-section-1  | div#cpt_III > p.title-division-1    |
| annexe         | div#anx_III > p.oj-doc-ti        | div#anx_III > p.title-annex-1/2     |
| modification   | -                                | p.modref « ▼M1 » (lien vers l'acte) |

On prend donc les considérants dans le JO d'origine, et les articles et annexes
dans la version consolidée la plus récente (qui intègre modifications et rectificatifs).
"""

import re
import warnings
from datetime import date, datetime

from bs4 import BeautifulSoup, NavigableString, Tag, XMLParsedAsHTMLWarning

from app.ingestion.cellar import EURLEX_URL
from app.ingestion.models import LegalSection, Paragraph

_ARTICLE_ID = re.compile(r"^art_(\d+[a-z]*)$")
_RECITAL_ID = re.compile(r"^rct_(\d+)$")
_ANNEX_ID = re.compile(r"^anx_([IVXLC]+)$")
_CHAPTER_ID = r"^cpt_[IVXLC]+$"
_SECTION_ID = r"^cpt_[IVXLC]+\.sct_\d+$"
_OJ_PARAGRAPH_ID = re.compile(r"^\d{3}\.\d{3}$")
_PARAGRAPH_NUMBER = re.compile(r"^(\d+(?: [a-z]+)?)\.\s")  # « 1. » ou « 1 bis. »

# Paragraphes qui ne sont pas du contenu juridique
_SKIP_P = {"modref", "arrow", "footnote", "oj-note"}
_HEADING_NUMBER = {"oj-ti-section-1", "title-division-1"}
_HEADING_TITLE = {"title-division-2"}
_ARTICLE_TITLE = {"oj-ti-art", "title-article-norm"}
_ANNEX_TITLE = {"oj-doc-ti", "title-annex-1", "title-annex-2"}


def classes(el: Tag) -> set[str]:
    return set(el.get("class") or [])


def soup_of(html: str) -> BeautifulSoup:
    """Parseur HTML (et non XML) : il découpe l'attribut class en liste de classes."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", XMLParsedAsHTMLWarning)
        return BeautifulSoup(html, "lxml")


# --------------------------------------------------------------------------- texte


def clean(text: str) -> str:
    """Normalise les espaces (le JO utilise des espaces insécables partout)."""
    return re.sub(r"\s+", " ", text).strip()


# Marqueurs des versions consolidées : ▼B (texte de base), ▼M1 (modifié), ►C1 ... ◄ (rectifié)
MARKER = re.compile(r"[▼►]\s?[A-Z]\d*|◄")
_INLINE = {"span", "a", "sup", "sub", "em", "strong", "b", "i", "u", "abbr"}


def text_of(el: Tag | NavigableString) -> str:
    # get_text() sans séparateur : les <span> portent déjà leurs espaces,
    # un séparateur produirait « européenne . »
    raw = el.get_text() if isinstance(el, Tag) else str(el)
    return clean(MARKER.sub("", raw))


def render_node(node: Tag) -> list[str]:
    """Transforme un élément HTML en lignes de texte, en recollant les énumérations.

    « a) texte » est codé en deux colonnes (tableau dans le JO, grille dans la
    version consolidée) : on fusionne les deux en une seule ligne.
    """
    cls = classes(node)
    if node.name == "p":
        if cls & _SKIP_P:
            return []
        text = text_of(node)
        return [text] if text else []
    if node.name in {"hr", "br", "col", "img"}:
        return []
    if node.name == "table":
        return _render_table(node)
    if "grid-container" in cls:
        label_el = node.find(class_="grid-list-column-1")
        body_el = node.find(class_="grid-list-column-2")
        return _merge(text_of(label_el) if label_el else "", render(body_el) if body_el else [])
    number = node.find("span", class_="no-parag", recursive=False)
    if number is not None:  # paragraphe numéroté d'une version consolidée
        return _merge(text_of(number), render(node, skip=number))
    if node.name in _INLINE:
        text = text_of(node)
        return [text] if text else []
    return render(node)


def render(el: Tag, skip: Tag | None = None) -> list[str]:
    """Rend les enfants d'un élément.

    Les éléments en ligne consécutifs (texte nu, <span>, <a>...) forment UNE ligne :
    « ►C1 texte rectifié ◄ suite » reste une seule phrase.
    """
    lines: list[str] = []
    inline: list[str] = []

    def flush() -> None:
        text = clean("".join(inline))  # les nœuds texte portent leurs propres espaces
        if text:
            lines.append(text)
        inline.clear()

    for child in el.children:
        if child is skip:
            continue
        if isinstance(child, NavigableString) or child.name in _INLINE:
            inline.append(child.get_text() if isinstance(child, Tag) else str(child))
        else:
            flush()
            lines.extend(render_node(child))
    flush()
    return [clean(MARKER.sub("", line)) for line in lines if clean(MARKER.sub("", line))]


def _merge(label: str, body: list[str]) -> list[str]:
    if not body:
        return [label] if label else []
    return [f"{label} {body[0]}".strip(), *body[1:]]


def _render_table(table: Tag) -> list[str]:
    lines: list[str] = []
    for row in table.select(":scope > tbody > tr, :scope > tr"):
        cells = row.find_all(["td", "th"], recursive=False)
        if len(cells) == 2 and len(text_of(cells[0])) <= 12:  # énumération « a) | texte »
            lines.extend(_merge(text_of(cells[0]), render(cells[1])))
        else:  # vrai tableau : une ligne par rangée, cellules séparées par « | »
            row_text = " | ".join(" ".join(render(c)) for c in cells)
            if clean(row_text.replace("|", "")):
                lines.append(row_text)
    return lines


# --------------------------------------------------------------------------- nettoyage


def strip_noise(soup: BeautifulSoup) -> None:
    """Supprime les appels de note « (1) » (le texte des notes est ignoré au rendu)."""
    for anchor in soup.select('a[id^="ntc"], a[id^="src."]'):
        prev, nxt = anchor.previous_sibling, anchor.next_sibling
        # version consolidée : « (<a>1</a>) », parenthèses hors du lien
        if (
            isinstance(prev, NavigableString)
            and prev.endswith("(")
            and isinstance(nxt, NavigableString)
            and nxt.startswith(")")
        ):
            prev.replace_with(prev[:-1])
            nxt.replace_with(nxt[1:])
            prev, nxt = anchor.previous_sibling, anchor.next_sibling
        # « Conseil (13). » -> « Conseil. » : on retire l'espace qui précédait l'appel
        if (
            isinstance(prev, NavigableString)
            and isinstance(nxt, NavigableString)
            and nxt[:1] in {".", ",", ";", ":", ")"}
        ):
            prev.replace_with(prev.rstrip())
        anchor.decompose()


def _ancestor(el: Tag, pattern: str) -> Tag | None:
    regex = re.compile(pattern)
    for parent in el.parents:
        if isinstance(parent, Tag) and regex.match(parent.get("id", "")):
            return parent
    return None


def _heading(container: Tag | None) -> str | None:
    """« CHAPITRE III - SYSTÈMES D'IA À HAUT RISQUE » à partir d'un div cpt_/sct_."""
    if container is None:
        return None
    parts = []
    for child in container.find_all(["p", "div"], recursive=False):
        cls = classes(child)
        if cls & _HEADING_NUMBER or cls & _HEADING_TITLE or "eli-title" in cls:
            parts.append(text_of(child))
        elif parts:
            break
    return " - ".join(p for p in parts if p) or None


def _changes(el: Tag, base_celex: str) -> tuple[list[str], list[str]]:
    """(actes modificatifs, rectificatifs) signalés par les marqueurs ▼M1 / ►C1 de la section."""
    amended: list[str] = []
    corrected: list[str] = []
    links = el.find_all("a", title=True)
    # Articles insérés (ex. 75 bis à 75 quinquies) : un seul « ▼M1 » placé AVANT le bloc,
    # au même niveau. Le dernier marqueur frère qui précède donne l'origine de l'article
    # (« ▼B » = texte de base, ignoré plus bas).
    previous = el.find_previous_sibling("p", class_="modref")
    if previous is not None:
        links = previous.find_all("a", title=True) + links
    for link in links:
        if not MARKER.fullmatch(clean(link.get_text())):
            continue
        celex = link["title"].split(":")[0].strip()
        target = corrected if re.search(r"R\(\d+\)$", celex) else amended
        if celex != base_celex and celex not in target:
            target.append(celex)
    return amended, corrected


# --------------------------------------------------------------------------- sections


def parse_recitals(html: str, source_id: str, celex: str) -> list[LegalSection]:
    soup = soup_of(html)
    strip_noise(soup)
    sections = []
    for div in soup.find_all("div", id=_RECITAL_ID):
        number = _RECITAL_ID.match(div["id"]).group(1)
        text = "\n".join(render(div))
        text = re.sub(rf"^\({number}\)\s*", "", text)  # retire le « (12) » en tête
        sections.append(
            LegalSection(
                source_id=source_id,
                section_type="considerant",
                number=number,
                text=text,
                url=f"{EURLEX_URL.format(celex=celex)}#rct_{number}",
                source_celex=celex,
            )
        )
    return sections


def official_number(div: Tag, fallback: str) -> str:
    """Numéro officiel : « Article 4 bis » -> "4 bis", « Article premier » -> "1"."""
    heading = next(
        (p for p in div.find_all("p", recursive=False) if classes(p) & _ARTICLE_TITLE), None
    )
    match = re.match(r"Article\s+(.+)$", text_of(heading)) if heading else None
    if not match:
        return fallback
    return "1" if match.group(1).lower() == "premier" else match.group(1)


def _parse_article(div: Tag, anchor: str, source_id: str, celex: str, base: str) -> LegalSection:
    number = official_number(div, anchor)
    title_div = div.find("div", class_="eli-title", recursive=False)
    paragraphs: list[Paragraph] = []
    body: list[str] = []
    for child in div.find_all(recursive=False):
        if child is title_div or classes(child) & _ARTICLE_TITLE:
            continue
        lines = render_node(child)
        if not lines:
            continue
        body.extend(lines)
        is_paragraph = _OJ_PARAGRAPH_ID.match(child.get("id", "")) or child.find(
            "span", class_="no-parag", recursive=False
        )
        if is_paragraph:
            match = _PARAGRAPH_NUMBER.match(lines[0])
            paragraphs.append(
                Paragraph(number=match.group(1) if match else None, text="\n".join(lines))
            )

    amended_by, corrected_by = _changes(div, base)
    return LegalSection(
        source_id=source_id,
        section_type="article",
        number=number,
        title=text_of(title_div) if title_div else None,
        chapter=_heading(_ancestor(div, _CHAPTER_ID)),
        section=_heading(_ancestor(div, _SECTION_ID)),
        text="\n".join(body),
        paragraphs=paragraphs,
        url=f"{EURLEX_URL.format(celex=celex)}#art_{anchor}",
        source_celex=celex,
        amended_by=amended_by,
        corrected_by=corrected_by,
    )


def _parse_annex(div: Tag, number: str, source_id: str, celex: str, base: str) -> LegalSection:
    titles = [p for p in div.find_all("p") if classes(p) & _ANNEX_TITLE]
    amended_by, corrected_by = _changes(div, base)
    if len(titles) > 1:
        title = text_of(titles[1])
    else:  # ex. annexe XIV consolidée : le titre est le premier paragraphe qui suit
        following = titles[0].find_next_sibling("p") if titles else None
        title = text_of(following) if following else None
    for p in titles:
        p.decompose()
    return LegalSection(
        source_id=source_id,
        section_type="annexe",
        number=number,
        title=title,
        text="\n".join(render(div)),
        url=f"{EURLEX_URL.format(celex=celex)}#anx_{number}",
        source_celex=celex,
        amended_by=amended_by,
        corrected_by=corrected_by,
    )


def parse_body(html: str, source_id: str, celex: str, base_celex: str) -> list[LegalSection]:
    """Articles et annexes (JO ou version consolidée)."""
    soup = soup_of(html)
    strip_noise(soup)
    sections = []
    for div in soup.find_all("div", id=True):
        if m := _ARTICLE_ID.match(div["id"]):
            sections.append(_parse_article(div, m.group(1), source_id, celex, base_celex))
        elif m := _ANNEX_ID.match(div["id"]):
            sections.append(_parse_annex(div, m.group(1), source_id, celex, base_celex))
    return sections


def publication_date(html: str) -> date | None:
    tag = soup_of(html).select_one(".oj-hd-date")
    if tag is None:
        return None
    try:
        return datetime.strptime(text_of(tag), "%d.%m.%Y").date()
    except ValueError:
        return None


# --------------------------------------------------------------------------- contrôle qualité


def normalize(text: str) -> str:
    """Forme de comparaison : espaces et apostrophes unifiés."""
    return clean(text.replace("’", "'").replace("‘", "'"))


def coverage(html: str, sections: list[LegalSection]) -> list[str]:
    """Paragraphes du HTML source absents du texte extrait (doit être vide).

    Contrôle de fidélité : chaque <p> de contenu situé dans un considérant, un article
    ou une annexe doit se retrouver dans le texte d'une section.
    """
    soup = soup_of(html)
    strip_noise(soup)
    extracted = normalize("\n".join(s.text for s in sections))
    prefix = {"considerant": "rct_", "article": "art_", "annexe": "anx_"}
    expected = {prefix[s.section_type] + s.url.rsplit("_", 1)[-1] for s in sections}
    missing = []
    for div in soup.find_all("div", id=lambda i: i in expected):
        for p in div.find_all("p"):
            cls = classes(p)
            if cls & (_SKIP_P | _ARTICLE_TITLE | _ANNEX_TITLE | {"stitle-article-norm"}):
                continue
            if p.find_parent(class_="eli-title"):
                continue
            if re.fullmatch(r"\(\d+\)", text_of(p)):  # numéro de considérant, retiré exprès
                continue
            text = normalize(text_of(p))
            if text and text not in extracted:
                missing.append(f"{div['id']}: {text[:80]}")
    return missing
