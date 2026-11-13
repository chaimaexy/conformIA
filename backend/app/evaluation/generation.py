"""Évaluation de la génération sur le golden set (chaîne complète, comme en production).

    python -m app.evaluation.generation --validated [--name generation] [--limit 20]
    python -m app.evaluation.generation --validated --judge --model mistral-large-latest

Métriques, toutes calculées par du code (pas de LLM juge) :
- refus : questions hors corpus refusées (garde-fou ou « SOURCES_INSUFFISANTES ») ;
  questions du corpus refusées à tort ;
- citations : réponses dont tous les renvois [n] sont valides et toutes les citations « »
  exactes ; part des phrases longues sans renvoi ;
- sources : part des sections attendues effectivement CITÉES dans la réponse ;
- faits clés : dates, montants et pourcentages de la réponse de référence retrouvés ;
- chiffres non sourcés : faits de la réponse absents de toutes les sources fournies
  (hallucination sur ce qui compte le plus en droit).
Avec --judge : fidélité (affirmations soutenues par les sources) et exactitude par rapport à
la réponse de référence, jugées par un modèle plus puissant (voir judge.py).
Le LLM est mis en cache : relancer ne coûte rien ; ~1 min par question au premier passage
(offre gratuite Mistral, 2 requêtes/min).
"""

import argparse
import json
import logging
import statistics
from datetime import UTC, datetime

from app.core.logging import setup_logging
from app.evaluation.golden import EVAL_DIR, load
from app.evaluation.judge import judge
from app.rag.facts import extract
from app.rag.service import ChatService

logger = logging.getLogger("eval.generation")
RESULTS_DIR = EVAL_DIR / "results"


def _source_id(document: str) -> str:
    if document.startswith("AI Act"):
        return "ai_act"
    if document.startswith("RGPD"):
        return "rgpd"
    return "cnil_ia"


def evaluate_item(service: ChatService, item, use_judge: bool = False) -> dict:
    result = service.answer(item.question, session_id=f"eval-{item.id}")
    row = {
        "id": item.id,
        "category": item.category,
        "zone": result.zone,
        "top_score": result.top_score,
        "refused": result.refused,
        "answer": result.answer,
    }
    if result.refused or item.category == "hors_corpus":
        return row
    cited = [s for s in result.sources if s.cited]
    v = result.verification
    found = sum(
        any(ref.matches(_source_id(s.document), s.label) for s in cited) for ref in item.expected
    )
    expected_facts = extract(item.answer)
    answer_facts = extract(result.answer)
    source_facts = set().union(*(extract(s.text) for s in result.sources))
    row.update(
        {
            "citations_ok": v.ok,
            "invalid_markers": v.invalid_markers,
            "unverified_quotes": v.unverified_quotes,
            "uncited_sentences": len(v.uncited_sentences),
            "sources_cited": [f"{s.document.split(' (')[0]} {s.label}" for s in cited],
            "expected_cited": found,
            "n_expected": len(item.expected),
            "facts_expected": sorted(expected_facts),
            "facts_found": sorted(expected_facts & answer_facts),
            "facts_unsourced": sorted(answer_facts - source_facts),
        }
    )
    if use_judge:
        try:
            j = judge(item.question, item.answer, [s.text for s in result.sources], result.answer)
        except Exception as exc:  # une réponse vide ou invalide du juge n'arrête pas l'évaluation
            logger.warning("Juge indisponible pour %s : %s", item.id, exc)
            row["judge_error"] = str(exc)[:200]
            return row
        row.update(
            {
                "judge_accuracy": j.accuracy,
                "judge_faithfulness": j.faithfulness,
                "judge_unsupported": j.unsupported,
                "judge_errors": j.errors,
            }
        )
    return row


def summarize(rows: list[dict]) -> dict:
    corpus = [r for r in rows if r["category"] != "hors_corpus"]
    out = [r for r in rows if r["category"] == "hors_corpus"]
    answered = [r for r in corpus if not r["refused"]]
    with_facts = [r for r in answered if r["facts_expected"]]
    summary = {
        "questions": len(rows),
        "hors_corpus_refusees": round(sum(r["refused"] for r in out) / len(out), 3)
        if out
        else None,
        "corpus_refusees_a_tort": round(sum(r["refused"] for r in corpus) / len(corpus), 3),
        "reponses": len(answered),
        "citations_valides": round(statistics.mean(r["citations_ok"] for r in answered), 3),
        "renvois_invalides": sum(len(r["invalid_markers"]) for r in answered),
        "citations_inexactes": sum(len(r["unverified_quotes"]) for r in answered),
        "sources_attendues_citees": round(
            sum(r["expected_cited"] for r in answered) / sum(r["n_expected"] for r in answered), 3
        ),
        "faits_cles_retrouves": round(
            sum(len(r["facts_found"]) for r in with_facts)
            / sum(len(r["facts_expected"]) for r in with_facts),
            3,
        )
        if with_facts
        else None,
        "reponses_avec_chiffre_non_source": sum(bool(r["facts_unsourced"]) for r in answered),
        "zones": {z: sum(r["zone"] == z for r in rows) for z in sorted({r["zone"] for r in rows})},
    }
    judged = [r for r in answered if "judge_accuracy" in r]
    if judged:
        faithful = [r["judge_faithfulness"] for r in judged if r["judge_faithfulness"] is not None]
        summary["juge"] = {
            "reponses_jugees": len(judged),
            "fidelite_moyenne": round(statistics.mean(faithful), 3) if faithful else None,
            "reponses_avec_affirmation_non_soutenue": sum(
                bool(r["judge_unsupported"]) for r in judged
            ),
            "affirmations_contredites": sum(
                c["verdict"] == "contredite" for r in judged for c in r["judge_unsupported"]
            ),
            "exactitude": {
                a: sum(r["judge_accuracy"] == a for r in judged)
                for a in ("correcte", "partielle", "incorrecte")
            },
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Évaluation de la génération")
    parser.add_argument("--validated", action="store_true", help="questions validées uniquement")
    parser.add_argument("--name", default="generation", help="nom du fichier de résultats")
    parser.add_argument("--limit", type=int, default=None, help="n premières questions (essai)")
    parser.add_argument("--ids", nargs="*", default=None, help="seulement ces questions")
    parser.add_argument("--model", default=None, help="modèle de génération (défaut : config)")
    parser.add_argument("--judge", action="store_true", help="juge LLM de fidélité/exactitude")
    parser.add_argument("--effort", default=None, help="effort de raisonnement : low/medium/high")
    args = parser.parse_args()
    setup_logging()

    items = [i for i in load() if i.status != "rejete"]
    if args.validated:
        items = [i for i in items if i.status == "valide"]
    if args.ids:
        items = [i for i in items if i.id in args.ids]
    items = items[: args.limit] if args.limit else items
    service = ChatService(model=args.model, effort=args.effort)
    rows = []
    for n, item in enumerate(items, start=1):
        rows.append(evaluate_item(service, item, use_judge=args.judge))
        logger.info("%d/%d %s : %s", n, len(items), item.id, rows[-1]["zone"])
    summary = summarize(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{args.name}.json"
    payload = {
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
        "model": args.model or "config",
        "effort": service.effort,
        "summary": summary,
        "questions": rows,
    }
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Résumé : %s", summary)
    logger.info("Résultats détaillés : %s", out)


if __name__ == "__main__":
    main()
