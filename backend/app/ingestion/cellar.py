"""Accès au dépôt Cellar de l'Office des publications de l'UE.

Pourquoi Cellar et pas eur-lex.europa.eu ?
Le site EUR-Lex protège ses pages par une vérification anti-robot. Cellar est l'accès
officiel prévu pour les machines :
- documents : publications.europa.eu/resource/celex/{CELEX} (négociation de contenu)
- métadonnées : point d'accès SPARQL (versions consolidées, rectificatifs, modifications)
"""

import hashlib
import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import quote

import httpx

from app.core.config import RAW_DIR

logger = logging.getLogger(__name__)

RESOURCE_URL = "http://publications.europa.eu/resource/celex/{celex}"
SPARQL_URL = "https://publications.europa.eu/webapi/rdf/sparql"
EURLEX_URL = "https://eur-lex.europa.eu/legal-content/FR/TXT/HTML/?uri=CELEX:{celex}"


@dataclass
class RawFile:
    celex: str
    path: Path
    url: str
    sha256: str
    size: int
    retrieved_at: datetime


def filename(celex: str) -> str:
    """32024R1689R(01) -> 32024R1689R01.html"""
    return re.sub(r"[()]", "", celex) + ".html"


def download(celex: str, dest_dir: Path = RAW_DIR, refresh: bool = False) -> RawFile:
    """Télécharge la version française (XHTML) d'un document, avec cache dans data/raw/."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / filename(celex)
    url = RESOURCE_URL.format(celex=quote(celex))

    if refresh or not path.exists():
        response = httpx.get(
            url,
            headers={"Accept": "application/xhtml+xml, text/html;q=0.9", "Accept-Language": "fra"},
            follow_redirects=True,
            timeout=60,
        )
        response.raise_for_status()
        if len(response.content) < 1_000:
            raise RuntimeError(f"Réponse trop courte pour {celex} ({len(response.content)} octets)")
        path.write_bytes(response.content)
        logger.info("Téléchargé %s (%d Ko)", celex, len(response.content) // 1024)

    content = path.read_bytes()
    return RawFile(
        celex=celex,
        path=path,
        url=url,
        sha256=hashlib.sha256(content).hexdigest(),
        size=len(content),
        retrieved_at=datetime.fromtimestamp(path.stat().st_mtime, UTC),
    )


# --------------------------------------------------------------------------- métadonnées

_VERSIONS_QUERY = """
PREFIX cdm: <http://publications.europa.eu/ontology/cdm#>
PREFIX lang: <http://publications.europa.eu/resource/authority/language/>
SELECT DISTINCT ?celex ?kind WHERE {{
  ?base cdm:resource_legal_id_celex "{celex}"^^<http://www.w3.org/2001/XMLSchema#string> .
  {{ ?w cdm:act_consolidated_based_on_resource_legal ?base . BIND("consolidated" AS ?kind) }}
  UNION {{
    ?w cdm:resource_legal_corrects_resource_legal ?base .
    ?e cdm:expression_belongs_to_work ?w ;
       cdm:expression_uses_language lang:FRA .
    BIND("corrigendum_fr" AS ?kind)
  }}
  ?w cdm:resource_legal_id_celex ?celex .
}}
"""


@dataclass
class Versions:
    consolidated: list[str]  # triées, la plus récente en dernier
    corrigenda_fr: list[str]

    @property
    def latest_consolidated(self) -> str | None:
        return self.consolidated[-1] if self.consolidated else None


def versions(celex: str) -> Versions:
    """Liste les versions consolidées et les rectificatifs FRANÇAIS d'un acte."""
    response = httpx.get(
        SPARQL_URL,
        params={"query": _VERSIONS_QUERY.format(celex=celex), "format": "application/json"},
        timeout=60,
    )
    response.raise_for_status()
    rows = response.json()["results"]["bindings"]
    found: dict[str, list[str]] = {"consolidated": [], "corrigendum_fr": []}
    for row in rows:
        found[row["kind"]["value"]].append(row["celex"]["value"])
    # 02024R1689-20260727 : la date en suffixe permet un tri chronologique
    return Versions(
        consolidated=sorted(found["consolidated"], key=lambda c: c.rsplit("-", 1)[-1]),
        corrigenda_fr=sorted(found["corrigendum_fr"]),
    )
