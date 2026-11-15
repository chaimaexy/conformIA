"""Évaluation de l'agent d'analyse sur des scénarios vérifiés (eval/agent_scenarios.yaml).

    python -m app.evaluation.agent                 # tous les scénarios
    python -m app.evaluation.agent --ids cv_deployeur notation_sociale

L'agent est joué comme par un utilisateur : il extrait le profil de la description, pose ses
questions, et l'évaluation y répond à partir des « faits » du scénario. On mesure :
- la classification (niveau, point de l'annexe III ou de l'article 5, article 50, chapitre V) ;
- les obligations attendues retrouvées, avec la bonne date, et l'absence des obligations exclues ;
- les erreurs d'extraction du profil (valeur affirmée contraire aux faits), les questions posées,
  les appels LLM et la durée.
"""

import argparse
import json
import logging
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

import yaml

from app.agents.graph import AnalysisService
from app.agents.profile import QUESTIONS, YES_NO

SCENARIOS = Path(__file__).resolve().parents[2] / "eval" / "agent_scenarios.yaml"
RESULTS_DIR = Path(__file__).resolve().parents[2] / "eval" / "results"
logger = logging.getLogger("eval.agent")


def load(path: Path = SCENARIOS) -> list[dict]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def fact_value(name: str, raw):
    """Fait du scénario -> valeur du profil (« oui » -> True pour les questions oui/non)."""
    if name in QUESTIONS and QUESTIONS[name]["options"] is YES_NO:
        return {"oui": True, "non": False}.get(raw)
    return raw


def profile_errors(extracted: dict, facts: dict) -> list[str]:
    """Valeurs affirmées par l'extraction et contraires aux faits (une valeur vide n'est pas une
    erreur : l'agent pose alors la question)."""
    errors = []
    for name, raw in facts.items():
        value = extracted.get(name)
        if value in (None, []):
            continue
        truth = fact_value(name, raw)
        if name == "roles":
            if set(value) - set(truth):
                errors.append(f"roles : {value} au lieu de {truth}")
        elif value != truth:
            errors.append(f"{name} : {value} au lieu de {truth}")
    return errors


def check(report: dict, expected: dict) -> dict:
    c = report["classification"]
    found = {o["id"]: o for o in report["obligations"]}
    result = {"niveau_ok": c["niveau"] == expected["niveau"]}
    if "annexe_iii" in expected:
        result["annexe_iii_ok"] = c["annexe_iii"].get("point") == expected["annexe_iii"]
    if "interdit" in expected:
        points = [p["point"] for p in c["interdit"]["pratiques"] if p["verdict"] == "oui"]
        result["interdit_ok"] = expected["interdit"] in points
    if "transparence" in expected:
        result["transparence_ok"] = sorted(c["transparence"]) == sorted(expected["transparence"])
    if "modele_usage_general" in expected:
        result["modele_usage_general_ok"] = (
            c["modele_usage_general"] == expected["modele_usage_general"]
        )
    wanted = expected.get("obligations") or {}
    result["obligations_trouvees"] = [o for o in wanted if o in found]
    result["obligations_manquantes"] = [o for o in wanted if o not in found]
    result["dates_fausses"] = [
        o
        for o, when in wanted.items()
        if o in found
        and [d["date"] for d in found[o]["dates"]][:1] != ([str(when)] if when else [])
    ]
    result["obligations_en_trop"] = [o for o in expected.get("absentes") or [] if o in found]
    return result


def run_scenario(service: AnalysisService, scenario: dict) -> dict:
    start = time.perf_counter()
    events = list(service.start(scenario["description"]))
    analysis_id = events[0]["analyse_id"]
    extracted = service.graph.get_state(service._config(analysis_id)).values["profil"]
    asked = []
    while events[-1]["type"] == "questions":  # un second tour si une réponse en appelle une autre
        fields = [q["champ"] for q in events[-1]["questions"]]
        asked += fields
        answers = {name: scenario["faits"].get(name, "je_ne_sais_pas") for name in fields}
        events = list(service.resume(analysis_id, answers))
    final = events[-1]
    if final["type"] != "rapport":
        return {"id": scenario["id"], "erreur": final}
    rapport = final["rapport"]
    return {
        "id": scenario["id"],
        "duree_s": round(time.perf_counter() - start, 1),
        "appels_llm": rapport["appels_llm"],
        "questions_posees": asked,
        "erreurs_profil": profile_errors(extracted, scenario["faits"]),
        "niveau": rapport["classification"]["niveau"],
        "annexe_iii": rapport["classification"]["annexe_iii"].get("point"),
        "interdit": [
            p["point"]
            for p in rapport["classification"]["interdit"]["pratiques"]
            if p["verdict"] == "oui"
        ],
        "transparence": rapport["classification"]["transparence"],
        "modele_usage_general": rapport["classification"]["modele_usage_general"],
        "obligations": {o["id"]: [d["date"] for d in o["dates"]] for o in rapport["obligations"]},
        "synthese_origine": rapport["synthese_origine"],
        **check(rapport, scenario["attendu"]),
    }


def summarize(rows: list[dict]) -> dict:
    done = [r for r in rows if "erreur" not in r]

    def rate(key):
        relevant = [r for r in done if key in r]
        return f"{sum(r[key] for r in relevant)}/{len(relevant)}" if relevant else None

    wanted = sum(len(r["obligations_trouvees"]) + len(r["obligations_manquantes"]) for r in done)
    return {
        "scenarios": len(rows),
        "erreurs_execution": len(rows) - len(done),
        "niveau_juste": rate("niveau_ok"),
        "point_annexe_iii_juste": rate("annexe_iii_ok"),
        "pratique_interdite_juste": rate("interdit_ok"),
        "transparence_juste": rate("transparence_ok"),
        "modele_usage_general_juste": rate("modele_usage_general_ok"),
        "obligations_retrouvees": f"{sum(len(r['obligations_trouvees']) for r in done)}/{wanted}",
        "dates_fausses": sum(len(r["dates_fausses"]) for r in done),
        "obligations_en_trop": sum(len(r["obligations_en_trop"]) for r in done),
        "erreurs_de_profil": sum(len(r["erreurs_profil"]) for r in done),
        "questions_par_analyse": round(sum(len(r["questions_posees"]) for r in done) / len(done), 1)
        if done
        else None,
        "appels_llm_moyens": round(sum(r["appels_llm"] for r in done) / len(done), 1)
        if done
        else None,
        "duree_moyenne_s": round(sum(r["duree_s"] for r in done) / len(done), 1) if done else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Évaluation de l'agent d'analyse")
    parser.add_argument("--ids", nargs="*", default=None)
    parser.add_argument("--name", default="agent")
    args = parser.parse_args()
    scenarios = [s for s in load() if not args.ids or s["id"] in args.ids]
    db = Path(tempfile.mkdtemp()) / "analyses.sqlite"  # état des analyses d'évaluation, jetable
    service = AnalysisService(db_path=db)
    rows = []
    for n, scenario in enumerate(scenarios, start=1):
        try:
            rows.append(run_scenario(service, scenario))
        except Exception as exc:  # une panne ne doit pas perdre les autres scénarios
            logger.exception("Scénario %s en échec", scenario["id"])
            rows.append({"id": scenario["id"], "erreur": str(exc)})
        logger.info("%d/%d %s : %s", n, len(scenarios), scenario["id"], rows[-1].get("niveau"))
        # sauvegarde au fil de l'eau : une panne en fin de course ne perd pas les scénarios faits
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        (RESULTS_DIR / f"{args.name}.partiel.json").write_text(
            json.dumps(rows, ensure_ascii=False), encoding="utf-8"
        )
    summary = summarize(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{args.name}.json"
    payload = {
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
        "summary": summary,
        "scenarios": rows,
    }
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    (RESULTS_DIR / f"{args.name}.partiel.json").unlink(missing_ok=True)  # passage complet
    logger.info("Résumé : %s", summary)
    logger.info("Résultats détaillés : %s", out)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    main()
