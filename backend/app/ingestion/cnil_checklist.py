"""Liste de vérification CNIL (PDF) : « Développement des systèmes d'IA : que faut-il vérifier ? »

Le PDF est un tableau de 4 pages :
    | n° | thème (fiche) | sous-thème | mesure | case à cocher |
Les cellules fusionnées sont renvoyées vides (None) par pdfplumber : on reprend la
valeur de la ligne précédente. Chaque thème devient une section, chaque mesure un
paragraphe numéroté (« 5.3 »), ce qui servira de grille à l'agent de conformité.

Piège : les coupures de mots en fin de ligne (« déve-⏎loppeurs »). En français, un
tiret peut aussi être réel (« sous-⏎traitants »). On tranche avec le vocabulaire des
fiches CNIL déjà ingérées ; les cas non tranchés sont signalés.
"""

import hashlib
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import pdfplumber

from app.ingestion.eurlex import clean, normalize
from app.ingestion.models import LegalSection, Paragraph

PDF_URL = "https://www.cnil.fr/sites/default/files/2025-07/ia_liste_de_verification.pdf"
FILENAME = "ia_liste_de_verification.pdf"
_BULLET = "•"
_WORD_END = re.compile(r"(\w+)-$")
_WORD_START = re.compile(r"^(\w+)")


@dataclass
class Dehyphenator:
    """Décide si « mot-⏎suite » est une coupure de ligne ou un vrai trait d'union."""

    vocabulary: set[str]
    undecided: list[str] = field(default_factory=list)

    def join(self, left: str, right: str) -> str:
        end, start = _WORD_END.search(left), _WORD_START.match(right)
        if not end or not start:
            return f"{left} {right}"
        merged = (end.group(1) + start.group(1)).lower()
        hyphenated = f"{end.group(1)}-{start.group(1)}".lower()
        if hyphenated in self.vocabulary and merged not in self.vocabulary:
            return left + right  # vrai trait d'union : « sous-traitants »
        if merged not in self.vocabulary:
            self.undecided.append(f"{end.group(1)}-{start.group(1)}")
        return left[:-1] + right  # coupure de ligne : « déve-loppeurs » -> « développeurs »


def vocabulary(texts: list[str]) -> set[str]:
    return {w.lower() for t in texts for w in re.findall(r"\w+(?:-\w+)*", normalize(t))}


def _cell_lines(cell: str | None, dehyphen: Dehyphenator) -> list[str]:
    """Lignes logiques d'une cellule : on recolle les lignes, une puce ouvre une ligne."""
    lines: list[str] = []
    for raw in (cell or "").splitlines():
        raw = clean(raw)
        if not raw:
            continue
        if raw.startswith(_BULLET) or not lines:
            lines.append(raw)
        else:
            lines[-1] = dehyphen.join(lines[-1], raw)
    return [re.sub(rf"^{_BULLET}\s*", "- ", line) for line in lines]


def _cell_text(cell: str | None, dehyphen: Dehyphenator) -> str:
    return " ".join(_cell_lines(cell, dehyphen))


@dataclass
class Checklist:
    title: str
    published: date | None
    intro: str
    themes: list[dict]  # {number, title, items: [(sous-thème, [lignes de la mesure])]}
    undecided: list[str]


def extract(pdf_path: Path, known_texts: list[str]) -> Checklist:
    dehyphen = Dehyphenator(vocabulary(known_texts))
    themes: list[dict] = []
    with pdfplumber.open(pdf_path) as pdf:
        title = pdf.metadata.get("Title", "").strip()
        created = re.match(r"D:(\d{4})(\d{2})(\d{2})", pdf.metadata.get("CreationDate", ""))
        published = date(*map(int, created.groups())) if created else None

        first = pdf.pages[0]
        table_top = first.find_tables()[0].bbox[1]
        intro_lines = first.crop((0, 0, first.width, table_top)).extract_text().splitlines()
        intro_paragraphs: list[str] = []
        for line in intro_lines[1:]:  # la 1re ligne est le titre
            line = clean(line)
            if not line:
                continue
            ends_paragraph = intro_paragraphs and re.search(r"[.!?:]$", intro_paragraphs[-1])
            if not intro_paragraphs or (ends_paragraph and line[:1].isupper()):
                intro_paragraphs.append(line)
            else:
                intro_paragraphs[-1] = dehyphen.join(intro_paragraphs[-1], line)

        subtheme = None
        for page in pdf.pages:
            for row in page.extract_tables()[0][1:]:  # ligne 0 = en-tête FICHES / MESURES
                number, name, sub, measure = (row + [None] * 4)[:4]
                if number:
                    themes.append(
                        {"number": clean(number), "title": _cell_text(name, dehyphen), "items": []}
                    )
                    subtheme = None
                if sub is not None:  # None = cellule fusionnée avec la ligne du dessus
                    subtheme = _cell_text(sub, dehyphen) or None
                lines = _cell_lines(measure, dehyphen)
                if lines:
                    themes[-1]["items"].append((subtheme, lines))

    return Checklist(title, published, "\n".join(intro_paragraphs), themes, dehyphen.undecided)


def to_sections(checklist: Checklist, source_id: str = "cnil_ia") -> list[LegalSection]:
    common = {
        "source_id": source_id,
        "section_type": "checklist",
        "chapter": checklist.title,
        "url": PDF_URL,
        "source_date": checklist.published,
    }
    sections = [LegalSection(number="0", title="Présentation", text=checklist.intro, **common)]
    for theme in checklist.themes:
        lines, paragraphs, current = [], [], object()
        for k, (subtheme, measure) in enumerate(theme["items"], start=1):
            if subtheme != current and subtheme:
                lines.append(subtheme)
            current = subtheme
            lines.append("- " + measure[0])
            lines.extend("  " + extra for extra in measure[1:])
            prefix = f"[{subtheme}] " if subtheme else ""
            paragraphs.append(
                Paragraph(number=f"{theme['number']}.{k}", text=prefix + "\n".join(measure))
            )
        sections.append(
            LegalSection(
                number=theme["number"],
                title=theme["title"],
                text="\n".join(lines),
                paragraphs=paragraphs,
                **common,
            )
        )
    return sections


# --------------------------------------------------------------------------- contrôle qualité


def _chars(text: str) -> Counter:
    """Multiensemble des caractères, sans espaces, tirets ni puces (coupures de ligne)."""
    return Counter(re.sub(r"[\s\-•]", "", normalize(text)))


def fidelity(pdf_path: Path, sections: list[LegalSection]) -> tuple[bool, int, str]:
    """Chaque caractère imprimé dans le PDF doit se retrouver dans l'extraction, et inversement.

    Le tableau mélange les colonnes dans l'ordre de lecture du PDF : on compare donc
    les multiensembles de caractères (aucun perdu, aucun ajouté), pas leur ordre.
    """
    with pdfplumber.open(pdf_path) as pdf:
        source = Counter()
        for page in pdf.pages:
            source += _chars(page.extract_text())
        pages = len(pdf.pages)
        printed_title = pdf.pages[0].extract_text().splitlines()[0]  # en capitales
    # éléments de mise en page attendus : titre imprimé + en-tête du tableau sur chaque page
    # (le pied de page « CNIL. » est une image, absente du texte)
    extracted = _chars(printed_title + " FICHES MESURES" * pages)
    for s in sections:
        extracted += _chars(s.text)
        if s.section_type == "checklist" and s.number != "0":
            extracted += _chars(s.number + s.title)
    lost, added = source - extracted, extracted - source
    if lost or added:
        return False, 0, f"perdus : {dict(lost)} | ajoutés : {dict(added)}"

    ok, detail = _ordered_measures(pdf_path, sections)
    if not ok:
        return False, 0, detail
    return True, sum(source.values()), f"{sum(source.values())} caractères identiques ; {detail}"


def _seq(text: str) -> str:
    return re.sub(r"[\s\-•]", "", normalize(text))


def _ordered_measures(pdf_path: Path, sections: list[LegalSection]) -> tuple[bool, str]:
    """Contrôle ORDONNÉ de la colonne « mesures » : on découpe la colonne dans le PDF
    et on compare la suite exacte des caractères (une inversion de lettres est détectée)."""
    with pdfplumber.open(pdf_path) as pdf:
        column = ""
        for page in pdf.pages:
            table = page.find_tables()[0]
            first_row = table.rows[1].cells  # ligne 0 = en-tête
            x0, x1 = first_row[3][0], first_row[3][2]
            column += page.crop((x0, first_row[3][1], x1, table.bbox[3])).extract_text() or ""
    measures = "".join(
        re.sub(r"^\[[^\]]*\] ", "", p.text)
        for s in sections
        if s.section_type == "checklist"
        for p in s.paragraphs
    )
    source, extracted = _seq(column), _seq(measures)
    if source == extracted:
        return True, f"colonne des mesures identique dans l'ordre ({len(source)} caractères)"
    i = next((i for i, (a, b) in enumerate(zip(source, extracted, strict=False)) if a != b), 0)
    return (
        False,
        f"mesures : écart au caractère {i} "
        f"({source[i - 15 : i + 25]!r} / {extracted[i - 15 : i + 25]!r})",
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download(refresh: bool = False) -> Path:
    """PDF officiel, mis en cache dans data/raw/cnil/."""
    from app.ingestion import cnil  # import local : évite un cycle au chargement

    cnil.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = cnil.CACHE_DIR / FILENAME
    if refresh or not path.exists():
        content = cnil._get(PDF_URL)
        if not content.startswith(b"%PDF"):
            raise RuntimeError(f"{PDF_URL} n'a pas renvoyé un PDF")
        path.write_bytes(content)
    return path
