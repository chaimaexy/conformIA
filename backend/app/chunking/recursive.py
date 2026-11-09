"""Chunking récursif : la base des tutoriels (RecursiveCharacterTextSplitter).

Le document est traité comme un texte brut : on coupe tous les `chunk_size` caractères
en préférant les sauts de paragraphe, puis de ligne, puis les fins de phrase, avec un
chevauchement. Aucune connaissance de la structure juridique, pas d'en-tête de contexte :
c'est la référence que les stratégies plus fines doivent battre.

Pour pouvoir citer quand même, on retrouve les sections couvertes grâce aux positions
(start_index) de chaque chunk dans le texte complet.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.chunking.common import paragraph_numbers, ref
from app.chunking.models import Chunk
from app.ingestion.models import LegalSection


def _section_title_line(section: LegalSection) -> str:
    """Le titre tel qu'il apparaîtrait dans le document (« Article 6 - ... »)."""
    if section.section_type == "considerant":
        return f"({section.number})"
    if section.section_type == "fiche":
        return section.title or ""
    title = f" - {section.title}" if section.title else ""
    return f"{section.label}{title}"


def chunk_document(
    sections: list[LegalSection], chunk_size: int = 1000, chunk_overlap: int = 150
) -> list[Chunk]:
    text_parts: list[str] = []
    spans: list[tuple[int, int, LegalSection]] = []
    position = 0
    for section in sections:
        part = f"{_section_title_line(section)}\n{section.text}".strip()
        spans.append((position, position + len(part), section))
        text_parts.append(part)
        position += len(part) + 2  # séparateur "\n\n"
    full_text = "\n\n".join(text_parts)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True,
    )
    chunks = []
    for i, doc in enumerate(splitter.create_documents([full_text])):
        start = doc.metadata["start_index"]
        end = start + len(doc.page_content)
        covered = [s for a, b, s in spans if a < end and b > start]
        chunks.append(
            Chunk(
                source_id=sections[0].source_id,
                strategy="recursive",
                chunk_index=i,
                body=doc.page_content,
                sections=[ref(s) for s in covered],
                paragraphs=paragraph_numbers(doc.page_content)
                if covered[0].section_type == "article"
                else [],
                source_date=covered[0].source_date.isoformat() if covered[0].source_date else None,
                amended_by=sorted({c for s in covered for c in s.amended_by}),
                corrected_by=sorted({c for s in covered for c in s.corrected_by}),
            )
        )
    return chunks
