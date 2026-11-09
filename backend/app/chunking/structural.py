"""Chunking structurel : on respecte les unités juridiques.

- un considérant, un article, une annexe, une partie de fiche = un chunk s'il tient ;
- sinon on coupe aux frontières naturelles : paragraphes numérotés (« 2. », « 1 bis. »),
  points d'annexe, intertitres de fiche, thèmes de la liste de vérification ;
- chaque chunk reçoit un en-tête de contexte (« AI Act > CHAPITRE III > Article 6 - ... »),
  y compris les morceaux d'un article découpé, qui sinon perdraient leur référence.
Un chunk ne mélange jamais deux sections : la citation reste exacte.
"""

import re

from app.chunking.common import PARAGRAPH_START, blocks, header, pack, paragraph_numbers, ref
from app.chunking.models import Chunk
from app.ingestion.models import LegalSection

_HEADING = re.compile(r"^[^-\s].{0,150}[^.;:,]$")  # ligne courte sans ponctuation finale


def _starts_block(section: LegalSection):
    if section.section_type in {"article", "annexe"}:
        return lambda line: bool(PARAGRAPH_START.match(line))
    if section.section_type == "fiche":
        return lambda line: bool(_HEADING.match(line)) and not line.startswith(" ")
    if section.section_type == "checklist":
        return lambda line: not line.startswith((" ", "- "))  # sous-thème
    return lambda line: True  # considérant : chaque alinéa


def chunk_section(section: LegalSection, max_chars: int) -> list[Chunk]:
    base_header = header(section)
    bodies = pack(blocks(section.text, _starts_block(section)), max_chars)
    is_article = section.section_type == "article"
    chunks = []
    current_paragraph: str | None = None  # paragraphe en cours au début du morceau
    for i, body in enumerate(bodies):
        numbers = paragraph_numbers(body) if is_article else []
        starts_mid_paragraph = bool(current_paragraph) and not PARAGRAPH_START.match(body)
        if starts_mid_paragraph:  # « c) ... » : on est toujours dans le paragraphe précédent
            numbers = [current_paragraph, *numbers]
        part = ""
        if len(bodies) > 1:
            where = f" - paragraphe {numbers[0]}" if numbers else ""  # là où le morceau commence
            part = f" (partie {i + 1}/{len(bodies)}{where})"
        if numbers:
            current_paragraph = numbers[-1]
        chunks.append(
            Chunk(
                source_id=section.source_id,
                strategy="structural",
                chunk_index=0,  # renuméroté par chunk_document
                header=base_header + part,
                body=body,
                sections=[ref(section)],
                paragraphs=numbers,
                source_date=section.source_date.isoformat() if section.source_date else None,
                amended_by=section.amended_by,
                corrected_by=section.corrected_by,
            )
        )
    return chunks


def chunk_document(sections: list[LegalSection], max_chars: int = 2000) -> list[Chunk]:
    chunks = [c for s in sections for c in chunk_section(s, max_chars)]
    for i, chunk in enumerate(chunks):
        chunk.chunk_index = i
    return chunks
