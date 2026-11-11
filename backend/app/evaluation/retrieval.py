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
    python -m app.evaluation.retrieval --validated --suite phase4 --name hybride
"""

import argparse
import json
import logging
import statistics
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime

from app.core.logging import setup_logging
from app.evaluation.golden import EVAL_DIR, GoldenItem, load
from app.retrieval.pipeline import RetrievalSettings, Retriever
from app.vectordb.store import client

logger = logging.getLogger("eval.retrieval")
RESULTS_DIR = EVAL_DIR / "results"
TOP = 20
KS = (1, 4, 8, 20)


@dataclass(frozen=True)
class Config:
    name: str
    settings: RetrievalSettings


MISTRAL, BGE = "mistral-embed", "local:BAAI/bge-m3"
CAMEMBERT = "antoinelouis/crossencoder-camembert-base-mmarcoFR"
BGE_RERANK = "BAAI/bge-reranker-v2-m3"
QWEN3 = "tomaarsen/Qwen3-Reranker-0.6B-seq-cls"
COHERE = "cohere:rerank-v5.0-pro"
COHERE_FAST = "cohere:rerank-v5.0-fast"
HYBRID = RetrievalSettings("structural", (MISTRAL, BGE), bm25=True)
SUITES = {
    # Phase 3 : stratégies de découpage x modèles d'embedding (recherche dense seule)
    "phase3": [
        Config("récursif + mistral-embed", RetrievalSettings("recursive", (MISTRAL,))),
        Config("sémantique + mistral-embed", RetrievalSettings("semantic", (MISTRAL,))),
        Config("structurel + mistral-embed", RetrievalSettings("structural", (MISTRAL,))),
        Config("structurel + bge-m3", RetrievalSettings("structural", (BGE,))),
    ],
    # Phase 4 : recherche hybride sur le découpage structurel
    "phase4": [
        Config("dense mistral-embed", RetrievalSettings("structural", (MISTRAL,))),
        Config("BM25 seul", RetrievalSettings("structural", (), bm25=True)),
        Config("mistral + BM25 (RRF)", RetrievalSettings("structural", (MISTRAL,), bm25=True)),
        Config("bge-m3 + BM25 (RRF)", RetrievalSettings("structural", (BGE,), bm25=True)),
        Config("hybride : mistral + bge-m3 + BM25 (RRF)", HYBRID),
        Config("hybride + multi-query", replace(HYBRID, multi_query=True)),
        Config("hybride + reranker CamemBERT", replace(HYBRID, reranker=CAMEMBERT)),
        Config("hybride + reranker bge-v2-m3", replace(HYBRID, reranker=BGE_RERANK)),
        Config(
            "hybride + multi-query + reranker bge-v2-m3",
            replace(HYBRID, multi_query=True, reranker=BGE_RERANK),
        ),
        Config(
            "hybride + multi-query + reranker CamemBERT",
            replace(HYBRID, multi_query=True, reranker=CAMEMBERT),
        ),
    ],
    # Comparaison finale, sur l'index avec renvois résolus (multi-query et CamemBERT écartés)
    "final": [
        Config("dense mistral-embed", RetrievalSettings("structural", (MISTRAL,))),
        Config("hybride : mistral + bge-m3 + BM25 (RRF)", HYBRID),
        Config("hybride + reranker bge-v2-m3", replace(HYBRID, reranker=BGE_RERANK)),
        Config("hybride + reranker Qwen3-0.6B", replace(HYBRID, reranker=QWEN3)),
        Config("hybride + reranker Cohere v5 fast", replace(HYBRID, reranker=COHERE_FAST)),
        Config("hybride + reranker Cohere v5 pro", replace(HYBRID, reranker=COHERE)),
    ],
    # Rerankers récents : open source (Qwen3) et API commerciale (Cohere)
    "rerankers": [
        Config("hybride + reranker Qwen3-0.6B", replace(HYBRID, reranker=QWEN3)),
        Config("hybride + reranker Cohere v5 pro", replace(HYBRID, reranker=COHERE)),
        Config("hybride + reranker Cohere v5 fast", replace(HYBRID, reranker=COHERE_FAST)),
        Config(
            "hybride + multi-query + reranker Qwen3-0.6B",
            replace(HYBRID, multi_query=True, reranker=QWEN3),
        ),
        Config(
            "hybride + multi-query + reranker Cohere v5 pro",
            replace(HYBRID, multi_query=True, reranker=COHERE),
        ),
    ],
}
CONFIGS = SUITES["phase3"]


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


def gate_analysis(per_question: list[dict]) -> dict:
    """Le score du 1er résultat sépare-t-il les questions hors corpus des autres ?

    Garde-fou : « je ne sais pas » si score < seuil. Deux points de fonctionnement :
    - sans_refus_a_tort : seuil = plus petit score d'une question du corpus (aucune refusée) ;
      part des questions hors corpus alors bloquées ;
    - tout_bloquer : seuil juste au-dessus du plus haut score hors corpus ; part des questions
      du corpus alors refusées à tort.
    """
    corpus = [q["top_score"] for q in per_question if q["category"] != "hors_corpus"]
    out = [q["top_score"] for q in per_question if q["category"] == "hors_corpus"]
    if not corpus or not out:
        return {}
    low, high = min(corpus), max(out)
    return {
        "sans_refus_a_tort": {
            "seuil": round(low, 4),
            "hors_corpus_bloquees": round(sum(s < low for s in out) / len(out), 3),
        },
        "tout_bloquer": {
            "seuil": round(high, 4),
            "refus_a_tort": round(sum(s <= high for s in corpus) / len(corpus), 3),
        },
        "separation_parfaite": high < low,
    }


def run(items: list[GoldenItem], configs: list[Config] = CONFIGS) -> dict:
    db = client()
    sources = {
        d["id"]: d["source_id"]
        for d in db.table("documents").select("id, source_id").execute().data
    }
    results = {}
    for config in configs:
        retriever = Retriever(config.settings, db)
        all_rows = retriever.search_many([i.question for i in items], k=TOP)
        per_question = []
        for item, rows in zip(items, all_rows, strict=True):
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
            "config": {"name": config.name, **asdict(config.settings)},
            "summary": summarize(per_question),
            "gate": gate_analysis(per_question),
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
    parser.add_argument("--suite", default="phase3", choices=sorted(SUITES), help="configurations")
    parser.add_argument("--skip", nargs="*", default=[], help="configurations à exclure")
    args = parser.parse_args()
    setup_logging()

    items = [i for i in load() if i.status != "rejete"]
    if args.validated:
        items = [i for i in items if i.status == "valide"]
    if not items:
        raise SystemExit("Golden set vide : lancer python -m app.evaluation.review export")
    logger.info("%d questions évaluées", len(items))
    configs = [c for c in SUITES[args.suite] if not any(s in c.name for s in args.skip)]
    results = run(items, configs)
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
