"""Ingestion des fiches pratiques IA de la CNIL (cnil.fr).

Licence : CC-BY-ND 4.0 (mentions légales de cnil.fr). On peut reproduire les textes
à l'identique, à condition de citer le titre, « Source : CNIL – https://www.cnil.fr »,
la date d'extraction et la licence. Pas de version modifiée : les citations restent
mot pour mot, et le contrôle de fidélité vérifie qu'aucun mot n'a été changé.

Structure d'une fiche :
    h1                       titre                -> chapter
    p.ctn-gen-auteur         « 22 juillet 2025 »  -> source_date
    div.page-introduction    chapô                -> section 0 « Introduction »
    div.ctn-gen-body         corps, découpé en une section par h2
      div.ctn-gen-ascenseur  bloc dépliable (h2 + bouton « Déplier » + contenu)
      div.apparte            encadré (« Exemple : ... »)
      button.glossary-tooltips  terme du glossaire (on garde le mot)
"""

import hashlib
import logging
import re
import time
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from urllib.parse import urljoin

import httpx
from bs4 import BeautifulSoup, NavigableString, Tag

from app.core.config import RAW_DIR
from app.ingestion.eurlex import clean, normalize
from app.ingestion.models import LegalSection

logger = logging.getLogger(__name__)

BASE_URL = "https://www.cnil.fr"
INDEX_URL = "https://www.cnil.fr/fr/les-fiches-pratiques-ia"
LICENSE = "CC-BY-ND 4.0 (https://creativecommons.org/licenses/by-nd/4.0/deed.fr)"
ATTRIBUTION = "Source : CNIL – https://www.cnil.fr"
USER_AGENT = "ConformIA/0.1 (student project; CNIL AI factsheets ingestion)"  # ASCII obligatoire
CACHE_DIR = RAW_DIR / "cnil"

_MONTHS = {
    m: i + 1
    for i, m in enumerate(
        [
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
    )
}
_INLINE = {"a", "span", "strong", "em", "b", "i", "u", "sup", "sub", "abbr", "button", "br"}
_NOISE = [
    "button.ctn-gen-ascenseur-collapse",  # « Déplier »
    ".sr-only",
    ".page--tools",  # imprimer / partager / taille de police
    ".ctn-gen-push-3",  # fiche précédente / suivante
    ".general-tags",
    "script",
    "style",
]


@dataclass
class Fiche:
    url: str
    path: Path
    sha256: str
    retrieved_at: datetime


# --------------------------------------------------------------------------- téléchargement


def _get(url: str) -> bytes:
    response = httpx.get(url, headers={"User-Agent": USER_AGENT}, follow_redirects=True, timeout=60)
    response.raise_for_status()
    time.sleep(0.5)  # politesse envers le serveur de la CNIL
    return response.content


def _cached(url: str, refresh: bool) -> Fiche:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    slug = url.rstrip("/").rsplit("/", 1)[-1]
    path = CACHE_DIR / f"{slug}.html"
    if refresh or not path.exists():
        path.write_bytes(_get(url))
        logger.info("Téléchargé %s", url)
    content = path.read_bytes()
    return Fiche(
        url=url,
        path=path,
        sha256=hashlib.sha256(content).hexdigest(),
        retrieved_at=datetime.fromtimestamp(path.stat().st_mtime, UTC),
    )


def discover(index_html: str) -> list[str]:
    """URLs des fiches HTML listées sur la page d'index (les PDF sont ignorés)."""
    main = BeautifulSoup(index_html, "lxml").find("main")
    urls: list[str] = []
    for link in main.find_all("a", href=True):
        url = urljoin(BASE_URL, link["href"])
        if "En savoir plus" in link.get_text() and not url.endswith(".pdf") and url not in urls:
            urls.append(url)
    return urls


def download_all(index_url: str = INDEX_URL, refresh: bool = False) -> tuple[Fiche, list[Fiche]]:
    index = _cached(index_url, refresh)
    urls = discover(index.path.read_text(encoding="utf-8"))
    return index, [_cached(url, refresh) for url in urls]


# --------------------------------------------------------------------------- parsing


def parse_date(text: str) -> date | None:
    """« 22 juillet 2025 » -> date(2025, 7, 22)"""
    m = re.search(r"(\d{1,2})(?:er)?\s+([a-zéû]+)\s+(\d{4})", text.lower())
    if not m or m.group(2) not in _MONTHS:
        return None
    return date(int(m.group(3)), _MONTHS[m.group(2)], int(m.group(1)))


def _prepare(html: str) -> BeautifulSoup:
    soup = BeautifulSoup(html, "lxml")
    for selector in _NOISE:
        for tag in soup.select(selector):
            tag.decompose()
    for button in soup.select("button.glossary-tooltips"):  # garder le terme, pas le bouton
        button.replace_with(button.get_text())
    return soup


def _inline_text(nodes: list) -> str:
    parts = []
    for node in nodes:
        if isinstance(node, NavigableString):
            parts.append(str(node))
        elif node.name == "br":
            parts.append(" ")
        else:
            parts.append(node.get_text())
    return clean("".join(parts))


def _render(el: Tag, depth: int = 0) -> list[str]:
    """Rend un bloc en lignes ; listes à puces « - », sous-listes indentées."""
    lines: list[str] = []
    inline: list = []

    def flush() -> None:
        text = _inline_text(inline)
        if text:
            lines.append(text)
        inline.clear()

    for child in el.children:
        if isinstance(child, NavigableString) or child.name in _INLINE:
            inline.append(child)
            continue
        flush()
        if child.name in {"ul", "ol"}:
            for li in child.find_all("li", recursive=False):
                own = [
                    c for c in li.children if not (isinstance(c, Tag) and c.name in {"ul", "ol"})
                ]
                sub_lines = _render_nodes(own, depth + 1)
                prefix = "  " * depth + "- "
                if sub_lines:
                    lines.append(prefix + sub_lines[0])
                    lines.extend("  " * (depth + 1) + s for s in sub_lines[1:])
                for sub in li.find_all(["ul", "ol"], recursive=False):
                    wrapper = BeautifulSoup("<div></div>", "lxml").div
                    wrapper.append(sub.__copy__())
                    lines.extend(_render(wrapper, depth + 1))
        elif child.name in {"hr", "img", "figure"}:
            continue
        else:
            lines.extend(_render(child, depth))
    flush()
    return lines


def _render_nodes(nodes: list, depth: int) -> list[str]:
    wrapper = BeautifulSoup("<div></div>", "lxml").div
    for node in nodes:
        wrapper.append(node.__copy__() if isinstance(node, Tag) else NavigableString(str(node)))
    return _render(wrapper, depth)


_CONTENT = ("div.ctn-gen-body", "div.ctn-gen-after")  # corps puis blocs dépliables


def _content(soup: BeautifulSoup) -> Tag | None:
    """Réunit dans un seul conteneur le corps de la fiche et ses blocs dépliables."""
    parts = [p for sel in _CONTENT if (p := soup.select_one(sel))]
    if not parts:
        return None
    container = soup.new_tag("div")
    for part in parts:
        for child in list(part.children):
            container.append(child.extract())
    return container


def _blocks(body: Tag) -> list[tuple[str | None, str | None, list[str]]]:
    """Découpe le corps en (titre h2, ancre, lignes). Les dépliables sont aplatis."""
    for box in body.select("div.ctn-gen-ascenseur"):
        heading = box.find("h2")
        content = box.select_one(".ctn-gen-ascenseur-texte")
        anchor = heading.find("span", id=True) if heading else None
        new_heading = BeautifulSoup("<h2></h2>", "lxml").h2
        new_heading.string = clean(heading.get_text()) if heading else ""
        if anchor:
            new_heading["id"] = anchor["id"]
        box.insert_before(new_heading)
        if content:
            box.insert_before(content)
        box.decompose()

    blocks: list[tuple[str | None, str | None, list[str]]] = [(None, None, [])]
    for child in body.children:
        if isinstance(child, Tag) and child.name == "h2":
            blocks.append((clean(child.get_text()), child.get("id"), []))
        elif isinstance(child, Tag):
            wrapper = BeautifulSoup("<div></div>", "lxml").div
            wrapper.append(child.__copy__())
            blocks[-1][2].extend(_render(wrapper))
        elif clean(str(child)):
            blocks[-1][2].append(clean(str(child)))
    return blocks


def parse(html: str, url: str, source_id: str = "cnil_ia") -> list[LegalSection]:
    soup = _prepare(html)
    h1 = soup.select_one(".ctn-gen-titre h1") or soup.find("h1")
    title = clean(h1.get_text()) if h1 else url
    date_el = soup.select_one("p.ctn-gen-auteur")
    published = parse_date(date_el.get_text()) if date_el else None
    intro = soup.select_one("div.page-introduction")
    body = _content(soup)
    if body is None:
        raise ValueError(f"{url} : corps de fiche introuvable (div.ctn-gen-body)")

    blocks = _blocks(body)
    intro_lines = _render(intro) if intro else []
    blocks[0] = (None, None, intro_lines + blocks[0][2])

    sections = []
    group = None  # intertitre sans contenu propre (« En pratique : ») qui chapeaute la suite
    for number, (heading, anchor, lines) in enumerate(blocks):
        if not lines:
            group = heading or group
            continue
        sections.append(
            LegalSection(
                source_id=source_id,
                section_type="fiche",
                number=str(number),
                title=heading or "Introduction",
                chapter=title,
                section=group,
                text="\n".join(lines),
                url=f"{url}#{anchor}" if anchor else url,
                source_date=published,
            )
        )
    return sections


# --------------------------------------------------------------------------- contrôle qualité

_LIST_MARKER = re.compile(r"^\s*- ", re.MULTILINE)


def chars(text: str) -> str:
    """Tous les caractères, ponctuation comprise ; seuls les espaces sont ignorés."""
    return re.sub(r"\s+", "", normalize(text))


def fidelity(html: str, sections: list[LegalSection]) -> tuple[bool, int, str]:
    """Compare caractère par caractère la page source et les sections extraites.

    Aucun caractère ajouté, perdu ou déplacé n'est toléré (seuls les espaces et les
    puces « - » ajoutées pour les listes sont ignorés).
    Renvoie (identique, nombre de caractères, détail).
    """
    soup = _prepare(html)
    parts = [soup.select_one(sel) for sel in ("div.page-introduction", *_CONTENT)]
    source = chars("".join(p.get_text() for p in parts if p))
    extracted = ""
    group = None
    for s in sections:
        if s.section and s.section != group:  # intertitre de groupe, compté une fois
            extracted += chars(s.section)
            group = s.section
        if s.title and s.title != "Introduction":
            extracted += chars(s.title)
        extracted += chars(_LIST_MARKER.sub("", s.text))
    if source == extracted:
        return True, len(source), f"{len(source)} caractères identiques"
    i = next(
        (i for i, (a, b) in enumerate(zip(source, extracted, strict=False)) if a != b),
        min(len(source), len(extracted)),
    )
    return (
        False,
        i,
        f"écart au caractère {i} : source « {source[i - 20 : i + 40]} » "
        f"/ extrait « {extracted[i - 20 : i + 40]} »",
    )
