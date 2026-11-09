"""Application des rectificatifs (corrigenda) publiés au Journal officiel.

Un rectificatif corrige une erreur de la version publiée, sous la forme :
    « Page 14, au considérant 71, cinquième et sixième phrases:
      au lieu de: «...», lire: «...». »

- Considérants : ils viennent du JO d'origine (les versions consolidées n'ont pas de
  préambule), on APPLIQUE donc la correction.
- Articles / annexes : ils viennent de la version consolidée, qui intègre déjà les
  rectificatifs ; on VÉRIFIE seulement que le texte corrigé y figure.
Toute correction qui ne peut pas être appliquée ou vérifiée est signalée.
"""

import re
from dataclasses import dataclass, field

from app.ingestion.eurlex import normalize, soup_of, text_of
from app.ingestion.models import LegalSection, SectionType

_ENTRY = re.compile(
    r"(?P<header>Page\s+\d+[^\n]*?):\s*au lieu de\s*:\s*(?P<old>.*?)\s*,?\s*lire\s*:\s*(?P<new>.*?)"
    r"(?=\s*Page\s+\d+,|\s*ELI:|\Z)",
    re.DOTALL,
)
_TARGETS: list[tuple[SectionType, re.Pattern]] = [
    ("considerant", re.compile(r"considérant\s+(\d+)")),
    ("article", re.compile(r"article\s+(\d+(?: (?:bis|ter|quater|quinquies))?)")),
    ("annexe", re.compile(r"annexe\s+([IVXLC]+)")),
]
_ELLIPSIS = re.compile(r"\[\s*…\s*\]|\[\s*\.\.\.\s*\]")


@dataclass
class Correction:
    corrigendum: str  # CELEX du rectificatif
    header: str
    section_type: SectionType | None
    number: str | None
    pairs: list[tuple[str, str]] = field(default_factory=list)  # (ancien, nouveau)


def _lines(block: str) -> list[str]:
    """Nettoie un bloc cité : guillemets et numéro de considérant « (71) ».

    Les « […] » (passage inchangé non recopié) sont conservés : ils séparent
    des fragments, voir `fragments()`.
    """
    lines = []
    for line in block.splitlines():
        line = line.strip().strip("«»").strip()
        line = re.sub(r"^«?\(\d+\)\s*", "", line).strip("«»").strip()
        if line and _ELLIPSIS.sub("", line).strip():
            lines.append(line)
    # le point final après « » appartient à la phrase de la note, pas à la citation
    if lines and lines[-1].endswith("»."):
        lines[-1] = lines[-1][:-2]
    return lines


def parse(html: str, celex: str) -> list[Correction]:
    soup = soup_of(html)
    text = "\n".join(t for p in soup.find_all("p") if (t := text_of(p)))
    corrections = []
    for m in _ENTRY.finditer(text):
        header = m.group("header")
        section_type, number = None, None
        for kind, pattern in _TARGETS:
            if found := pattern.search(header):
                section_type, number = kind, found.group(1)
                break
        old, new = _lines(m.group("old")), _lines(m.group("new"))
        if len(old) == len(new):
            pairs = [(o, n) for o, n in zip(old, new, strict=True) if o != n]
        else:
            pairs = [(" ".join(old), " ".join(new))]
        corrections.append(Correction(celex, header, section_type, number, pairs))
    return corrections


def fragments(quote: str) -> list[str]:
    """Morceaux de texte réellement cités, séparés par les « […] »."""
    return [f.strip() for f in _ELLIPSIS.split(quote) if len(f.strip(" ;,.")) > 15]


def _replace(text: str, old: str, new: str) -> str | None:
    """Remplace `old` par `new` ligne par ligne, en ignorant ’ vs ' (même longueur)."""
    target = normalize(old)
    lines = text.split("\n")
    for i, line in enumerate(lines):
        idx = normalize(line).find(target)
        if idx >= 0:
            lines[i] = line[:idx] + new + line[idx + len(target) :]
            return "\n".join(lines)
    return None


def apply(sections: list[LegalSection], corrections: list[Correction]) -> list[str]:
    """Applique / vérifie les corrections. Renvoie un rapport ; lève une erreur si échec."""
    by_key = {(s.section_type, s.number): s for s in sections}
    report, errors = [], []
    for c in corrections:
        section = by_key.get((c.section_type, c.number))
        where = f"{c.corrigendum} | {c.header}"
        if section is None:
            errors.append(f"{where} : section introuvable")
            continue
        for old, new in c.pairs:
            new_parts = fragments(new)
            if section.section_type != "considerant":
                if new_parts and all(normalize(f) in normalize(section.text) for f in new_parts):
                    report.append(f"vérifié        {where}")
                else:
                    errors.append(f"{where} : correction absente de la version consolidée")
                continue

            old_parts = fragments(old)
            if len(old_parts) != len(new_parts):
                errors.append(f"{where} : citations ancien/nouveau non alignées")
                continue
            for old_part, new_part in zip(old_parts, new_parts, strict=True):
                if old_part == new_part or normalize(new_part) in normalize(section.text):
                    continue
                updated = _replace(section.text, old_part, new_part)
                if updated is None:
                    errors.append(f"{where} : texte à remplacer introuvable : {old_part[:80]}")
                    break
                section.text = updated
            else:
                if c.corrigendum not in section.corrected_by:
                    section.corrected_by.append(c.corrigendum)
                report.append(f"appliqué       {where}")
    if errors:
        raise ValueError("Rectificatifs non appliqués :\n" + "\n".join(errors))
    return report
