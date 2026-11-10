"""Évaluation du retrieval sur le golden set (aucun appel au LLM).

Pour chaque configuration (stratégie de chunking x modèle d'embedding), on interroge
Supabase (match_chunks_dense) et on cherche à quel rang apparaît le premier chunk
contenant une section attendue.

Métriques (calculées sur les questions du corpus, hors « hors_corpus ») :
- hit@k     : part des questions dont une section attendue est dans les k premiers chunks
- recall@k  : part des sections attendues retrouvées dans les k premiers (questions multi-sources)
- MRR@20    : moyenne de 1/rang du premier bon chunk (0 si absent du top 20)
Pour les questions hors corpus, on relève le score du 1er résultat : c'est la donnée qui
servira à calibrer le garde-fou « je ne sais pas » (phase 4).

Usage (depuis backend/) :
    python -m app.evaluation.retrieval                 # toutes les questions non rejetées
    python -m app.evaluation.retrieval --validated     # seulement les questions validées
"""

import argparse
import json
import logging
import statistics
from dataclasses import dataclass
from datetime import UTC, datetime

from app.core.logging import setup_logging
from app.embeddings import get_embedder
from app.evaluation.golden import EVAL_DIR, GoldenItem, load
from app.vectordb.store import client, vector_literal

logger = logging.getLogger("eval.retrieval")
RESULTS_DIR = EVAL_DIR / "results"
TOP = 20
KS = (1, 4, 8, 20)


@dataclass(frozen=True)
class Config:
    name: str
    strategy: str
    model: str


CONFIGS = [
    Config("récursif + mistral-embed", "recursive", "mistral-embed"),
    Config("sémantique + mistral-embed", "semantic", "mistral-embed"),
    Config("structurel + mistral-embed", "structural", "mistral-embed"),
    Config("structurel + bge-m3", "structural", "local:BAAI/bge-m3"),
]


def first_rank(item: GoldenItem, rows: list[dict], sources: dict[str, str]) -> int | None:
    """Rang (1-based) du premier chunk qui contient une section attendue."""
    for rank, row in enumerate(rows, start=1):
        source_id = sources[row["document_id"]]
        labels = [s["label"] for s in row["metadata"]["sections"]]
        if any(ref.matches(source_id, label) for ref in item.expected for label in labels):
            return rank
    return None


def found_refs(item: GoldenItem, rows: list[dict], sources: dict[str, str], k: int) -> int:
    """Nombre de références attendues présentes dans les k premiers chunks."""
    seen = [
        (sources[r["document_id"]], s["label"]) for r in rows[:k] for s in r["metadata"]["sections"]
    ]
    return sum(any(ref.matches(src, lab) for src, lab in seen) for ref in item.expected)


def summarize(per_question: list[dict]) -> dict:
    in_corpus = [q for q in per_question if q["category"] != "hors_corpus"]
    out = [q for q in per_question if q["category"] == "hors_corpus"]
    summary = {"questions": len(in_corpus)}
    for k in KS:
        summary[f"hit@{k}"] = round(
            sum(q["rank"] is not None and q["rank"] <= k for q in in_corpus) / len(in_corpus), 3
        )
    summary["recall@8"] = round(
        sum(q["found@8"] for q in in_corpus) / sum(q["n_expected"] for q in in_corpus), 3
    )
    summary["mrr@20"] = round(
        statistics.mean(1 / q["rank"] if q["rank"] else 0 for q in in_corpus), 3
    )
    summary["top1_score_corpus"] = round(statistics.mean(q["top_score"] for q in in_corpus), 3)
    if out:
        summary["top1_score_hors_corpus"] = round(statistics.mean(q["top_score"] for q in out), 3)
    by_cat: dict[str, list[dict]] = {}
    for q in in_corpus:
        by_cat.setdefault(q["category"], []).append(q)
    summary["hit@8_par_categorie"] = {
        cat: round(sum(q["rank"] is not None and q["rank"] <= 8 for q in qs) / len(qs), 2)
        for cat, qs in sorted(by_cat.items())
    }
    return summary


def run(items: list[GoldenItem], configs: list[Config] = CONFIGS) -> dict:
    db = client()
    sources = {
        d["id"]: d["source_id"]
        for d in db.table("documents").select("id, source_id").execute().data
    }
    results = {}
    for config in configs:
        vectors = get_embedder(config.model).embed([i.question for i in items])
        per_question = []
        for item, vector in zip(items, vectors, strict=True):
            rows = (
                db.rpc(
                    "match_chunks_dense",
                    {
                        "query_embedding": vector_literal(vector),
                        "match_count": TOP,
                        "filter_strategy": config.strategy,
                        "embedding_model": config.model,
                    },
                )
                .execute()
                .data
            )
            rank = first_rank(item, rows, sources) if item.expected else None
            per_question.append(
                {
                    "id": item.id,
                    "category": item.category,
                    "rank": rank,
                    "found@8": found_refs(item, rows, sources, 8),
                    "n_expected": len(item.expected),
                    "top_score": round(rows[0]["score"], 4) if rows else 0.0,
                    "top_label": rows[0]["metadata"]["label"] if rows else None,
                }
            )
        results[config.name] = {
            "config": config.__dict__,
            "summary": summarize(per_question),
            "questions": per_question,
        }
        logger.info(
            "%s : %s",
            config.name,
            {k: v for k, v in results[config.name]["summary"].items() if not isinstance(v, dict)},
        )
    return results


def markdown_table(results: dict) -> str:
    lines = [
        "| Configuration | hit@1 | hit@4 | hit@8 | hit@20 | recall@8 | MRR@20 |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, r in results.items():
        s = r["summary"]
        lines.append(
            f"| {name} | {s['hit@1']:.2f} | {s['hit@4']:.2f} | {s['hit@8']:.2f} | "
            f"{s['hit@20']:.2f} | {s['recall@8']:.2f} | {s['mrr@20']:.3f} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Évaluation du retrieval")
    parser.add_argument("--validated", action="store_true", help="questions validées uniquement")
    parser.add_argument("--name", default="baseline", help="nom du fichier de résultats")
    args = parser.parse_args()
    setup_logging()

    items = [i for i in load() if i.status != "rejete"]
    if args.validated:
        items = [i for i in items if i.status == "valide"]
    if not items:
        raise SystemExit("Golden set vide : lancer python -m app.evaluation.review export")
    logger.info("%d questions évaluées", len(items))
    results = run(items)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{args.name}.json"
    payload = {
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
        "questions": len(items),
        "validated_only": args.validated,
        "results": results,
    }
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(markdown_table(results))
    logger.info("Résultats détaillés : %s", out)


if __name__ == "__main__":
    main()
