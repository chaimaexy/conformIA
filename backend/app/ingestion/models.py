"""Modèles de données produits par l'ingestion.

Chaque source est découpée selon SA structure, qui est l'unité naturelle de citation :
- textes européens : considérant, article, annexe (« article 6, paragraphe 2 »)
- fiches CNIL : une section par partie de la fiche (titre h2)
- liste de vérification CNIL (PDF) : une section par thème, une mesure par paragraphe
Le chunking de la phase 2 partira de ces sections.
"""

from datetime import UTC, date, datetime
from typing import Literal

from pydantic import BaseModel, Field

SectionType = Literal["considerant", "article", "annexe", "fiche", "checklist"]


class Paragraph(BaseModel):
    """Paragraphe numéroté d'un article (« 1. », « 1 bis. »...)."""

    number: str | None = None
    text: str


class LegalSection(BaseModel):
    source_id: str
    section_type: SectionType
    # "6" (article 6), "III" (annexe III), "57" (considérant 57), "2" (2e partie d'une fiche)
    number: str
    title: str | None = None
    chapter: str | None = None  # « CHAPITRE III - ... » ou titre de la fiche CNIL
    section: str | None = None  # « SECTION 1 - ... »
    text: str
    paragraphs: list[Paragraph] = Field(default_factory=list)
    url: str  # lien direct vers la section
    source_celex: str | None = None  # textes UE : JO d'origine ou version consolidée
    source_date: date | None = None  # fiches CNIL : date de publication / mise à jour
    amended_by: list[str] = Field(default_factory=list)  # CELEX des actes modificatifs
    corrected_by: list[str] = Field(default_factory=list)  # CELEX des rectificatifs appliqués
    # Considérants : les actes modificatifs ne les mettent jamais à jour (seuls articles et
    # annexes le sont). Ils peuvent donc contredire la version en vigueur (dates notamment).
    not_updated_by: list[str] = Field(default_factory=list)  # CELEX des actes non répercutés

    @property
    def label(self) -> str:
        """Référence lisible, utilisée dans les citations."""
        if self.section_type == "fiche":
            return f"CNIL, « {self.chapter} », {self.title}"
        if self.section_type == "checklist":
            return f"CNIL, liste de vérification IA, point {self.number} « {self.title} »"
        names = {"considerant": "Considérant", "article": "Article", "annexe": "Annexe"}
        return f"{names[self.section_type]} {self.number}"


class LegalDocument(BaseModel):
    source_id: str
    title: str
    celex: str | None = None  # acte de base (textes UE)
    consolidated_celex: str | None = None  # version consolidée utilisée pour articles/annexes
    language: str = "fr"
    url: str
    publication_date: date | None = None
    license: str | None = None
    attribution: str | None = None  # mention obligatoire à afficher avec les citations
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    sections: list[LegalSection] = Field(default_factory=list)

    def count(self, section_type: SectionType) -> int:
        return sum(s.section_type == section_type for s in self.sections)
