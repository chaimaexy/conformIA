"""Document du projet (PDF) : contrôles, puis texte page par page.

Le PDF n'est jamais stocké : il est lu en mémoire dès l'envoi, et seul son texte entre dans
l'analyse. Lire tout de suite permet aussi de refuser aussitôt, avec un message clair, un fichier
qui n'est pas un PDF, un PDF protégé ou un PDF scanné (une image, sans texte : pas d'OCR ici).

Le texte garde les numéros de page (« [page 3] ») : chaque valeur du profil est prouvée par un
extrait recopié du document, et le rapport dit de quelle page il vient.
"""

import io
import logging
import re
from dataclasses import dataclass, field

import pdfplumber

from app.rag.verify import normalize

MAX_BYTES = 10 * 1024 * 1024  # 10 Mo
MAX_PAGES = 30
MAX_CHARS = 80_000  # environ 20 000 tokens : un seul appel, sous la limite de 40 000 tokens/min
MIN_PAGE_CHARS = 20  # en dessous, la page est une image (scan) ou presque vide
PAGE_MARK = re.compile(r"\[page (\d+)\]")
logger = logging.getLogger(__name__)
logging.getLogger("pdfminer").setLevel(logging.ERROR)  # avertissements de police sans intérêt


class DocumentError(ValueError):
    """Document refusé ; le message est destiné à l'utilisateur."""


@dataclass
class ProjectDocument:
    name: str
    pages: list[str]
    empty_pages: list[int] = field(default_factory=list)  # pages sans texte (images, scans)

    def text(self) -> str:
        """Texte complet, chaque page précédée de « [page n] »."""
        return "\n\n".join(
            f"[page {n}]\n{page}" for n, page in enumerate(self.pages, start=1) if page
        )


def _clean(text: str) -> str:
    lines = [" ".join(line.split()) for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def read_pdf(data: bytes, name: str = "document.pdf") -> ProjectDocument:
    if len(data) > MAX_BYTES:
        raise DocumentError(f"Le fichier dépasse {MAX_BYTES // (1024 * 1024)} Mo.")
    if not data.lstrip()[:5] == b"%PDF-":  # le contenu, pas l'extension du nom
        raise DocumentError("Le fichier n'est pas un PDF.")
    try:
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            if len(pdf.pages) > MAX_PAGES:
                raise DocumentError(
                    f"Le document a {len(pdf.pages)} pages ; {MAX_PAGES} au maximum."
                )
            pages = [_clean(page.extract_text() or "") for page in pdf.pages]
    except DocumentError:
        raise
    except Exception as exc:  # PDF chiffré, abîmé, ou que pdfminer ne sait pas lire
        logger.info("PDF illisible (%s) : %s", name, exc)
        raise DocumentError("Le PDF est illisible ou protégé par un mot de passe.") from exc
    if not pages:
        raise DocumentError("Le PDF ne contient aucune page.")
    empty = [n for n, page in enumerate(pages, start=1) if len(page) < MIN_PAGE_CHARS]
    if len(empty) == len(pages):
        raise DocumentError(
            "Le PDF ne contient pas de texte (document scanné ?). Exportez-le en PDF depuis "
            "votre traitement de texte, ou décrivez le projet par écrit."
        )
    document = ProjectDocument(name=name, pages=pages, empty_pages=empty)
    if len(document.text()) > MAX_CHARS:
        raise DocumentError(
            f"Le texte du document est trop long ({len(document.text())} caractères, "
            f"{MAX_CHARS} au maximum) : envoyez les parties qui décrivent le projet."
        )
    return document


def page_of(excerpt: str, text: str) -> int | None:
    """Page d'où vient un extrait (d'après les marques « [page n] » du texte), ou None."""
    haystack = normalize(text)
    position = haystack.find(normalize(excerpt))
    if position < 0:
        return None
    marks = PAGE_MARK.findall(haystack[:position])
    return int(marks[-1]) if marks else None
