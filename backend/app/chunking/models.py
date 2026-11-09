"""Chunk : unité indexée et retrouvée par le RAG."""

from typing import Literal

from pydantic import BaseModel, Field

Strategy = Literal["structural", "recursive", "semantic"]


class SectionRef(BaseModel):
    """Section juridique (au moins en partie) contenue dans un chunk, pour les citations."""

    section_type: str
    number: str
    title: str | None = None
    label: str
    url: str


class Chunk(BaseModel):
    source_id: str
    strategy: Strategy
    chunk_index: int  # position dans la source, pour cette stratégie
    header: str | None = None  # contexte ajouté devant le texte (« AI Act > Article 6 - ... »)
    body: str  # texte du chunk, tel qu'il figure dans la source
    sections: list[SectionRef]  # la 1re est la section principale
    # n° des paragraphes couverts (« 1 », « 1 bis »)
    paragraphs: list[str] = Field(default_factory=list)
    source_date: str | None = None
    amended_by: list[str] = Field(default_factory=list)
    corrected_by: list[str] = Field(default_factory=list)

    @property
    def content(self) -> str:
        """Texte indexé (embeddings et plein texte) : en-tête de contexte + corps."""
        return f"{self.header}\n\n{self.body}" if self.header else self.body

    def metadata(self) -> dict:
        return {
            "header": self.header,
            "sections": [s.model_dump() for s in self.sections],
            "section_type": self.sections[0].section_type,
            "number": self.sections[0].number,
            "label": self.sections[0].label,
            "url": self.sections[0].url,
            "paragraphs": self.paragraphs,
            "source_date": self.source_date,
            "amended_by": self.amended_by,
            "corrected_by": self.corrected_by,
        }
