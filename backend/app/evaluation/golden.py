"""Golden set : questions de référence et sections du corpus qui contiennent la réponse.

On note au niveau de la SECTION (« ai_act / Article 113 ») et non du chunk : la vérité
ne dépend pas de la stratégie de chunking, ce qui rend la comparaison équitable.
Pour les fiches CNIL, on peut viser une fiche entière (`fiche`) plutôt qu'une partie.
"""

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from app.core.config import BACKEND_DIR, PROCESSED_DIR
from app.ingestion.models import LegalDocument, LegalSection

EVAL_DIR = BACKEND_DIR / "eval"
GOLDEN_PATH = EVAL_DIR / "golden_set.jsonl"

Category = Literal[
    "directe",  # la question reprend le vocabulaire du texte
    "reformulee",  # cas pratique, vocabulaire éloigné du texte
    "reference",  # cite un numéro d'article / d'annexe
    "date",  # calendrier d'application (version consolidée)
    "multi_sources",  # la réponse combine plusieurs textes
    "renvoi",  # la réponse exige de suivre un renvoi par numéro (« articles 12 à 22 »)
    "cnil_pratique",  # recommandations pratiques CNIL
    "hors_corpus",  # le corpus ne contient pas la réponse : il faut savoir dire « je ne sais pas »
]


class ExpectedRef(BaseModel):
    source_id: Literal["ai_act", "rgpd", "cnil_ia"]
    label: str | None = None  # « Article 113 », « Annexe III », « Considérant 71 »...
    fiche: str | None = None  # titre d'une fiche CNIL entière

    @model_validator(mode="after")
    def one_target(self):
        if bool(self.label) == bool(self.fiche):
            raise ValueError("indiquer soit `label`, soit `fiche`")
        return self

    def matches(self, source_id: str, label: str) -> bool:
        if source_id != self.source_id:
            return False
        if self.fiche:
            return label.startswith(f"CNIL, « {self.fiche} »")
        return label == self.label

    def __str__(self) -> str:
        return f"{self.source_id} / {self.label or 'fiche « ' + self.fiche + ' »'}"


class GoldenItem(BaseModel):
    id: str
    question: str
    category: Category
    expected: list[ExpectedRef] = Field(default_factory=list)  # vide si hors corpus
    answer: str  # réponse de référence courte (sert en phase 5 pour juger la génération)
    origin: Literal["manuel", "genere"]
    status: Literal["a_relire", "valide", "rejete"] = "a_relire"
    note: str | None = None

    @model_validator(mode="after")
    def expected_consistency(self):
        if (self.category == "hors_corpus") != (not self.expected):
            raise ValueError(f"{self.id} : `expected` vide si et seulement si hors corpus")
        return self


def load(path: Path = GOLDEN_PATH) -> list[GoldenItem]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [GoldenItem.model_validate_json(line) for line in lines if line.strip()]


def save(items: list[GoldenItem], path: Path = GOLDEN_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(i.model_dump(exclude_none=True), ensure_ascii=False) + "\n" for i in items
        ),
        encoding="utf-8",
    )


def corpus_sections() -> list[LegalSection]:
    sections = []
    for source_id in ("ai_act", "rgpd", "cnil_ia"):
        doc = LegalDocument.model_validate_json(
            (PROCESSED_DIR / f"{source_id}.json").read_text(encoding="utf-8")
        )
        sections += doc.sections
    return sections


def check_against_corpus(items: list[GoldenItem], sections: list[LegalSection]) -> list[str]:
    """Chaque référence attendue doit désigner au moins une section réelle du corpus."""
    errors = []
    for item in items:
        for ref in item.expected:
            if not any(ref.matches(s.source_id, s.label) for s in sections):
                errors.append(f"{item.id} : « {ref} » introuvable dans le corpus")
        if len({i.id for i in items}) != len(items):
            errors.append("identifiants en double")
            break
    return errors
