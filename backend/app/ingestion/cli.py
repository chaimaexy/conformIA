"""Ingestion des sources du corpus, avec contrôles de fraîcheur et de fidélité.

Usage (depuis backend/) :
    python -m app.ingestion.cli                 # toutes les sources de config.yaml
    python -m app.ingestion.cli ai_act          # une seule source
    python -m app.ingestion.cli --refresh       # ignore le cache data/raw/
    python -m app.ingestion.cli --strict        # échoue si une version plus récente existe

Étapes pour chaque texte :
1. Fraîcheur : interroge Cellar (SPARQL) -> dernière version consolidée, rectificatifs FR
2. Téléchargement : JO d'origine, version consolidée, rectificatifs (cache data/raw/)
3. Parsing : considérants (JO) + articles/annexes (version consolidée)
4. Rectificatifs : appliqués aux considérants, vérifiés dans la version consolidée
5. Fidélité : chaque paragraphe source doit se retrouver dans le texte extrait
6. Sorties : data/processed/{id}.json + registre de provenance data/manifest.json
"""

import argparse
import json
import logging
from datetime import UTC, datetime

from app.core.config import DATA_DIR, PROCESSED_DIR, CorpusSource, get_app_config
from app.core.logging import setup_logging
from app.ingestion import cellar, cnil, cnil_checklist, corrigenda, eurlex
from app.ingestion.models import LegalDocument

logger = logging.getLogger("ingestion")
MANIFEST = DATA_DIR / "manifest.json"


class StaleSourceError(RuntimeError):
    pass


def check_freshness(source: CorpusSource, strict: bool) -> dict:
    """Compare la configuration avec les versions publiées sur Cellar."""
    try:
        published = cellar.versions(source.celex)
    except Exception as exc:  # réseau indisponible : on continue avec la version figée
        logger.warning("%s : contrôle de fraîcheur impossible (%s)", source.id, exc)
        return {"checked": False}

    problems = []
    latest = published.latest_consolidated
    if source.consolidated and latest and latest != source.consolidated:
        problems.append(
            f"version consolidée plus récente : {latest} (config : {source.consolidated})"
        )
    unknown = sorted(set(published.corrigenda_fr) - set(source.corrigenda))
    if unknown:
        problems.append(f"rectificatifs FR non pris en compte : {', '.join(unknown)}")

    for problem in problems:
        logger.warning("%s : %s -> mettre à jour config.yaml", source.id, problem)
    if problems and strict:
        raise StaleSourceError(f"{source.id} : " + " ; ".join(problems))
    if not problems:
        logger.info("%s : à jour (dernière consolidation %s)", source.id, latest)
    return {
        "checked": True,
        "checked_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "latest_consolidated": latest,
        "corrigenda_fr": published.corrigenda_fr,
        "up_to_date": not problems,
    }


def ingest_source(source: CorpusSource, refresh: bool = False, strict: bool = False) -> dict:
    if not source.celex:
        raise SystemExit(f"La source {source.id} n'a pas de CELEX (pas un texte EUR-Lex)")
    freshness = check_freshness(source, strict)

    base = cellar.download(source.celex, refresh=refresh)
    base_html = base.path.read_text(encoding="utf-8")
    body_file = (
        cellar.download(source.consolidated, refresh=refresh) if source.consolidated else base
    )
    body_html = body_file.path.read_text(encoding="utf-8")

    recitals = eurlex.parse_recitals(base_html, source.id, source.celex)
    body = eurlex.parse_body(body_html, source.id, body_file.celex, source.celex)

    # Fidélité AVANT les rectificatifs : on compare au texte source tel que publié
    missing = eurlex.coverage(base_html, recitals) + eurlex.coverage(body_html, body)
    if missing:
        raise ValueError(
            f"{source.id} : {len(missing)} paragraphes perdus\n" + "\n".join(missing[:20])
        )

    files = [
        {"role": "considérants (JO)", **_describe(base)},
        {"role": "articles et annexes", **_describe(body_file)},
    ]
    corrections = []
    for celex in source.corrigenda:
        raw = cellar.download(celex, refresh=refresh)
        files.append({"role": "rectificatif", **_describe(raw)})
        corrections += corrigenda.parse(raw.path.read_text(encoding="utf-8"), celex)
    report = corrigenda.apply(recitals + body, corrections)
    amending = sorted({celex for section in body for celex in section.amended_by})
    for recital in recitals:  # texte d'origine : les modifications ne s'y reportent pas
        recital.not_updated_by = amending

    document = LegalDocument(
        source_id=source.id,
        title=source.title,
        celex=source.celex,
        consolidated_celex=source.consolidated,
        url=cellar.EURLEX_URL.format(celex=source.consolidated or source.celex),
        publication_date=eurlex.publication_date(base_html),
        sections=recitals + body,
    )
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / f"{source.id}.json"
    out.write_text(document.model_dump_json(indent=2), encoding="utf-8")

    counts = {k: document.count(k) for k in ("considerant", "article", "annexe")}
    logger.info(
        "%s : %d considérants, %d articles, %d annexes | %d rectifications | fidélité 100%% -> %s",
        source.id,
        *counts.values(),
        len(report),
        out.name,
    )
    return {
        "id": source.id,
        "title": source.title,
        "freshness": freshness,
        "files": files,
        "counts": counts,
        "amended_sections": sum(bool(s.amended_by) for s in document.sections),
        "corrigenda_report": report,
        "coverage_missing": 0,
    }


CNIL_MAX_AGE_DAYS = 30  # au-delà, on conseille --refresh (les fiches sont mises à jour)


def ingest_cnil(source: CorpusSource, refresh: bool = False, strict: bool = False) -> dict:
    """Fiches pratiques IA de la CNIL : découverte via l'index, parsing, fidélité mot à mot."""
    index, fiches = cnil.download_all(source.url or cnil.INDEX_URL, refresh=refresh)

    previous = _previous_entry(source.id)
    old_urls = {f["url"] for f in previous.get("files", []) if f.get("role") == "fiche"}
    new_urls = {f.url for f in fiches}
    if old_urls and old_urls != new_urls:
        logger.warning(
            "%s : liste des fiches modifiée (ajoutées : %s, retirées : %s)",
            source.id,
            sorted(new_urls - old_urls) or "aucune",
            sorted(old_urls - new_urls) or "aucune",
        )
    age = (datetime.now(UTC) - min(f.retrieved_at for f in [index, *fiches])).days
    if age > CNIL_MAX_AGE_DAYS:
        message = f"{source.id} : cache de {age} jours, relancer avec --refresh"
        if strict:
            raise StaleSourceError(message)
        logger.warning(message)

    sections = []
    files = [
        {
            "role": "index",
            "url": index.url,
            "sha256": index.sha256,
            "retrieved_at": index.retrieved_at.isoformat(timespec="seconds"),
        }
    ]
    for fiche in fiches:
        html = fiche.path.read_text(encoding="utf-8")
        parsed = cnil.parse(html, fiche.url, source.id)
        ok, n_chars, detail = cnil.fidelity(html, parsed)
        if not ok:
            raise ValueError(f"{fiche.url} : fidélité non respectée ({detail})")
        sections += parsed
        files.append(
            {
                "role": "fiche",
                "title": parsed[0].chapter,
                "url": fiche.url,
                "published": parsed[0].source_date.isoformat() if parsed[0].source_date else None,
                "sha256": fiche.sha256,
                "retrieved_at": fiche.retrieved_at.isoformat(timespec="seconds"),
                "characters_verified": n_chars,
            }
        )

    # Liste de vérification (PDF) liée depuis l'index
    if cnil_checklist.PDF_URL.removeprefix(cnil.BASE_URL) not in index.path.read_text("utf-8"):
        logger.warning("%s : l'index ne pointe plus vers %s", source.id, cnil_checklist.PDF_URL)
    pdf_path = cnil_checklist.download(refresh=refresh)
    checklist = cnil_checklist.extract(pdf_path, [s.text for s in sections])
    checklist_sections = cnil_checklist.to_sections(checklist, source.id)
    ok, n_chars, detail = cnil_checklist.fidelity(pdf_path, checklist_sections)
    if not ok:
        raise ValueError(f"{cnil_checklist.PDF_URL} : fidélité non respectée ({detail})")
    if checklist.undecided:
        logger.info("Coupures de mots tranchées par défaut : %s", ", ".join(checklist.undecided))
    sections += checklist_sections
    files.append(
        {
            "role": "liste de vérification (PDF)",
            "title": checklist.title,
            "url": cnil_checklist.PDF_URL,
            "published": checklist.published.isoformat() if checklist.published else None,
            "sha256": cnil_checklist.sha256(pdf_path),
            "retrieved_at": datetime.fromtimestamp(pdf_path.stat().st_mtime, UTC).isoformat(
                timespec="seconds"
            ),
            "characters_verified": n_chars,
            "measures": sum(len(s.paragraphs) for s in checklist_sections),
            "hyphenation_defaults": checklist.undecided,
        }
    )

    document = LegalDocument(
        source_id=source.id,
        title=source.title,
        url=index.url,
        license=cnil.LICENSE,
        attribution=cnil.ATTRIBUTION,
        sections=sections,
    )
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / f"{source.id}.json"
    out.write_text(document.model_dump_json(indent=2), encoding="utf-8")
    logger.info(
        "%s : %d fiches + liste de vérification (%d mesures), %d sections | fidélité 100%% -> %s",
        source.id,
        len(fiches),
        files[-1]["measures"],
        len(sections),
        out.name,
    )
    return {
        "id": source.id,
        "title": source.title,
        "license": cnil.LICENSE,
        "attribution": cnil.ATTRIBUTION,
        "files": files,
        "counts": {
            "fiches": len(fiches),
            "checklist_measures": files[-1]["measures"],
            "sections": len(sections),
        },
    }


def _previous_entry(source_id: str) -> dict:
    if not MANIFEST.exists():
        return {}
    sources = json.loads(MANIFEST.read_text(encoding="utf-8")).get("sources", [])
    return next((s for s in sources if s["id"] == source_id), {})


def _describe(raw: cellar.RawFile) -> dict:
    return {
        "celex": raw.celex,
        "url": raw.url,
        "eurlex": cellar.EURLEX_URL.format(celex=raw.celex),
        "sha256": raw.sha256,
        "size": raw.size,
        "retrieved_at": raw.retrieved_at.isoformat(timespec="seconds"),
    }


def write_manifest(entries: list[dict]) -> None:
    """Registre de provenance versionné dans git : d'où vient chaque donnée."""
    existing = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {}
    sources = {s["id"]: s for s in existing.get("sources", [])}
    sources.update({e["id"]: e for e in entries})
    manifest = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "note": "Textes consolidés EUR-Lex : outil de documentation, seules les versions "
        "publiées au JO font foi. Réutilisation autorisée (décision 2011/833/UE). "
        "Fiches CNIL : licence CC-BY-ND 4.0, citer « Source : CNIL » et la date d'extraction.",
        "sources": sorted(sources.values(), key=lambda s: s["id"]),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingestion du corpus ConformIA")
    parser.add_argument("sources", nargs="*", help="ids de config.yaml (défaut : toutes)")
    parser.add_argument("--refresh", action="store_true", help="re-télécharger les sources")
    parser.add_argument("--strict", action="store_true", help="échouer si une source est périmée")
    args = parser.parse_args()

    setup_logging()
    configured = {s.id: s for s in get_app_config().corpus.sources}
    unknown = set(args.sources) - set(configured)
    if unknown:
        raise SystemExit(
            f"Sources inconnues : {', '.join(unknown)}. Disponibles : {', '.join(configured)}"
        )
    selected = [configured[i] for i in args.sources] if args.sources else list(configured.values())
    ingesters = {"eurlex": ingest_source, "cnil": ingest_cnil}
    write_manifest([ingesters[s.kind](s, args.refresh, args.strict) for s in selected])
    logger.info("Registre de provenance : %s", MANIFEST)


if __name__ == "__main__":
    main()
