"""Outils communs aux stratégies de chunking : en-têtes de contexte, phrases, regroupement."""

import re
from collections.abc import Callable

from app.chunking.models import SectionRef
from app.ingestion.models import LegalSection

SOURCE_NAMES = {
    "ai_act": "AI Act (règlement (UE) 2024/1689)",
    "rgpd": "RGPD (règlement (UE) 2016/679)",
    "cnil_ia": "CNIL",
}

# Début de paragraphe numéroté d'article (« 1. », « 1 bis. ») ou de point d'annexe (« 3. »)
PARAGRAPH_START = re.compile(r"^(\d+(?: [a-z]+)?)\.\s")


_CELEX_ACT = re.compile(r"^3(\d{4})([RLD])(\d{4})$")
_ACT_KINDS = {"R": "règlement", "L": "directive", "D": "décision"}


def act_label(celex: str) -> str:
    """32026R1744 -> « règlement (UE) 2026/1744 »."""
    m = _CELEX_ACT.match(celex)
    if not m:
        return celex
    return f"{_ACT_KINDS[m.group(2)]} (UE) {m.group(1)}/{int(m.group(3))}"


def header(section: LegalSection) -> str:
    """Chemin de contexte placé devant le texte : aide l'embedding ET la lecture du LLM."""
    source = SOURCE_NAMES.get(section.source_id, section.source_id)
    if section.section_type == "considerant":
        note = ""
        if section.not_updated_by:  # le lecteur (et le LLM) doit savoir qu'il peut être périmé
            acts = ", ".join(act_label(c) for c in section.not_updated_by)
            note = f" (texte d'origine, non mis à jour par le {acts})"
        parts = [source, f"Considérant {section.number}{note}"]
    elif section.section_type == "article":
        title = f"Article {section.number}" + (f" - {section.title}" if section.title else "")
        parts = [source, section.chapter, section.section, title]
    elif section.section_type == "annexe":
        title = f"Annexe {section.number}" + (f" - {section.title}" if section.title else "")
        parts = [source, title]
    elif section.section_type == "fiche":
        parts = [source, f"Fiche « {section.chapter} »", section.section, section.title]
    else:  # checklist
        parts = [source, "Liste de vérification IA", f"{section.number}. {section.title}"]
    return " > ".join(p for p in parts if p)


def ref(section: LegalSection) -> SectionRef:
    return SectionRef(
        section_type=section.section_type,
        number=section.number,
        title=section.title,
        label=section.label,
        url=section.url,
    )


# « art. 5 », « p. 12 », « ex. : » ne terminent pas une phrase, ni un numéro de
# paragraphe ou de point (« 1. », « 12. », « 1 bis. »)
_ABBREVIATIONS = (
    r"(?<!\bart\.)(?<!\bp\.)(?<!\bex\.)(?<!\bpt\.)(?<!\bpts\.)(?<!\bcf\.)(?<!\bM\.)"
    r"(?<!\b\d\.)(?<!\b\d\d\.)(?<!\b\d\d\d\.)(?<!\bbis\.)(?<!\bter\.)(?<!\bquater\.)"
)
_SENTENCE_END = re.compile(r"(?<=[.!?])" + _ABBREVIATIONS + r"\s+(?=[«(A-ZÀ-ÖØ-Ý0-9])")


def sentences(text: str) -> list[str]:
    """Découpe en phrases (français) ; les retours à la ligne sont aussi des frontières."""
    out: list[str] = []
    for line in text.splitlines():
        out.extend(s.strip() for s in _SENTENCE_END.split(line) if s.strip())
    return out


def hard_split(text: str, max_chars: int) -> list[str]:
    """Dernier recours pour un bloc sans phrase plus courte que max_chars : coupe aux espaces."""
    pieces, current = [], ""
    for word in text.split(" "):
        if current and len(current) + 1 + len(word) > max_chars:
            pieces.append(current)
            current = word
        else:
            current = f"{current} {word}" if current else word
    return [*pieces, current] if current else pieces


def blocks(text: str, starts_block: Callable[[str], bool]) -> list[str]:
    """Regroupe les lignes en blocs ; un bloc commence à chaque ligne « de structure »."""
    result: list[list[str]] = []
    for line in text.splitlines():
        if not result or starts_block(line):
            result.append([line])
        else:
            result[-1].append(line)
    return ["\n".join(b) for b in result]


def split_to_fit(unit: str, max_chars: int) -> list[str]:
    """Garantit des morceaux <= max_chars : lignes, puis phrases, puis mots."""
    if len(unit) <= max_chars:
        return [unit]
    lines = unit.splitlines()
    if len(lines) > 1:
        return [p for line in lines for p in split_to_fit(line, max_chars)]
    parts = sentences(unit)
    if len(parts) > 1:
        return [p for s in parts for p in split_to_fit(s, max_chars)]
    return hard_split(unit, max_chars)


def split_tables(units: list[str], max_chars: int) -> list[str]:
    """Isole les tableaux Markdown ; un tableau trop long est coupé entre deux lignes, en
    répétant l'en-tête dans chaque morceau (une ligne de tableau ne se lit pas sans lui)."""
    out: list[str] = []
    for unit in units:
        runs: list[tuple[bool, list[str]]] = []
        for line in unit.splitlines():
            is_table = line.startswith("|") and line.endswith("|")
            if runs and runs[-1][0] == is_table:
                runs[-1][1].append(line)
            else:
                runs.append((is_table, [line]))
        for is_table, lines in runs:
            if not is_table or len("\n".join(lines)) <= max_chars:
                out.append("\n".join(lines))
                continue
            head, rows, current = lines[:2], lines[2:], []
            for row in rows:
                if current and len("\n".join([*head, *current, row])) > max_chars:
                    out.append("\n".join([*head, *current]))
                    current = []
                current.append(row)
            out.append("\n".join([*head, *current]))
    return out


def pack(units: list[str], max_chars: int, sep: str = "\n") -> list[str]:
    """Regroupe des unités consécutives tant que la taille reste <= max_chars."""
    chunks: list[str] = []
    current = ""
    for unit in (p for u in units for p in split_to_fit(u, max_chars)):
        candidate = f"{current}{sep}{unit}" if current else unit
        if current and len(candidate) > max_chars:
            chunks.append(current)
            current = unit
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def paragraph_numbers(body: str) -> list[str]:
    return [m.group(1) for line in body.splitlines() if (m := PARAGRAPH_START.match(line))]
