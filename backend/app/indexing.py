"""Chunking + embeddings + indexation.

Usage (depuis backend/) :
    python -m app.indexing                       # stratégie retenue (config.yaml), envoi Supabase
    python -m app.indexing --no-upload           # calcule et sauvegarde localement seulement
    python -m app.indexing --strategies recursive semantic --no-upload   # expériences phase 3

Seule la stratégie structurelle (choisie en phase 3) est indexée dans Supabase. Les deux
autres restent disponibles pour refaire la comparaison : chunks et vecteurs sont en cache
local, les renvoyer ne coûte rien.
Sorties : data/processed/chunks/{stratégie}.jsonl et data/processed/chunks/stats.json
"""

import argparse
import json
import logging
import statistics
import time

import numpy as np

from app.chunking import recursive, semantic, structural
from app.chunking.models import Chunk
from app.core.config import PROCESSED_DIR, get_app_config
from app.core.logging import setup_logging
from app.embeddings import get_embedder
from app.ingestion.models import LegalDocument

logger = logging.getLogger("indexing")
CHUNKS_DIR = PROCESSED_DIR / "chunks"
STRATEGIES = ["structural", "recursive", "semantic"]


def load_documents() -> list[LegalDocument]:
    docs = []
    for source in get_app_config().corpus.sources:
        path = PROCESSED_DIR / f"{source.id}.json"
        if not path.exists():
            raise SystemExit(f"{path} absent : lancer d'abord python -m app.ingestion.cli")
        docs.append(LegalDocument.model_validate_json(path.read_text(encoding="utf-8")))
    return docs


def build_chunks(strategy: str, docs: list[LegalDocument]) -> tuple[list[Chunk], dict]:
    cfg = get_app_config().chunking
    extra: dict = {}
    if strategy == "structural":
        chunks = [
            c
            for d in docs
            for c in structural.chunk_document(d.sections, cfg.max_chars, cfg.article_references)
        ]
    elif strategy == "recursive":
        chunks = [
            c
            for d in docs
            for c in recursive.chunk_document(d.sections, cfg.chunk_size, cfg.chunk_overlap)
        ]
    else:
        embedder = get_embedder(get_app_config().embeddings.model)
        chunks, thresholds = [], []
        for d in docs:
            doc_chunks, threshold = semantic.chunk_document(
                d.sections, embedder.embed, cfg.semantic_percentile, cfg.max_chars, cfg.min_chars
            )
            chunks += doc_chunks
            thresholds.append(round(threshold, 4))
        extra["breakpoint_thresholds"] = thresholds
    return chunks, extra


def stats(chunks: list[Chunk]) -> dict:
    sizes = [len(c.body) for c in chunks]
    return {
        "chunks": len(chunks),
        "median_chars": int(statistics.median(sizes)),
        "min_chars": min(sizes),
        "max_chars": max(sizes),
        "under_100_chars": sum(s < 100 for s in sizes),
        "spanning_several_sections": sum(len(c.sections) > 1 for c in chunks),
        "with_context_header": sum(bool(c.header) for c in chunks),
    }


def save(strategy: str, chunks: list[Chunk]) -> None:
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    with (CHUNKS_DIR / f"{strategy}.jsonl").open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(chunk.model_dump_json() + "\n")


def models_for(strategy: str) -> list[str]:
    cfg = get_app_config().embeddings
    return [cfg.model] + (cfg.compare_with if strategy == "structural" else [])


def main() -> None:
    parser = argparse.ArgumentParser(description="Chunking + embeddings + indexation")
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=STRATEGIES,
        default=[get_app_config().chunking.strategy],
        help="défaut : la stratégie de config.yaml",
    )
    parser.add_argument("--no-upload", action="store_true", help="ne pas écrire dans Supabase")
    args = parser.parse_args()
    setup_logging()

    docs = load_documents()
    titles = {s.id: s for s in get_app_config().corpus.sources}
    all_stats: dict = {}
    stats_path = CHUNKS_DIR / "stats.json"
    if stats_path.exists():
        all_stats = json.loads(stats_path.read_text(encoding="utf-8"))

    db = None
    if not args.no_upload:
        from app.vectordb import store

        db = store.client()
        doc_ids = {
            d.source_id: store.ensure_document(
                db, d.source_id, titles[d.source_id].title, d.url, d.consolidated_celex or d.celex
            )
            for d in docs
        }

    for strategy in args.strategies:
        started = time.time()
        chunks, extra = build_chunks(strategy, docs)
        save(strategy, chunks)
        entry = {**stats(chunks), **extra, "embedding_models": models_for(strategy)}

        embeddings: dict[str, np.ndarray] = {}
        for model in models_for(strategy):
            t = time.time()
            embeddings[model] = get_embedder(model).embed([c.content for c in chunks])
            entry[f"seconds_{model}"] = round(time.time() - t, 1)
            logger.info("%s : %d chunks embarqués avec %s", strategy, len(chunks), model)

        if db is not None:
            from app.vectordb import store

            for source_id, doc_id in doc_ids.items():
                idx = [i for i, c in enumerate(chunks) if c.source_id == source_id]
                n = store.replace_chunks(
                    db,
                    doc_id,
                    strategy,
                    [chunks[i] for i in idx],
                    {m: e[idx] for m, e in embeddings.items()},
                )
                logger.info("%s / %s : %d chunks envoyés dans Supabase", strategy, source_id, n)
            entry["uploaded"] = True

        entry["seconds_total"] = round(time.time() - started, 1)
        all_stats[strategy] = entry
        stats_path.write_text(json.dumps(all_stats, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info(
            "%s : %s", strategy, {k: v for k, v in entry.items() if k != "embedding_models"}
        )


if __name__ == "__main__":
    main()
