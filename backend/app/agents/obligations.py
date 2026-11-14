"""Obligations et dates : la carte vérifiée (obligations.yaml) appliquée au profil, sans LLM.

python -m app.agents.obligations --verifier   # chaque extrait de la carte est-il dans l'index ?
"""

import argparse
import logging
import sys
from datetime import date
from functools import lru_cache
from pathlib import Path

import yaml

from app.agents.profile import Profile
from app.rag.verify import quote_found

MAP_PATH = Path(__file__).with_name("obligations.yaml")
logger = logging.getLogger(__name__)


@lru_cache
def load_map(path: Path = MAP_PATH) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _date_key(key: str | None, profile: Profile, cats: set[str]) -> str | None:
    """Clé générique de la carte (« haut_risque ») -> date précise selon le profil."""
    market = profile.deja_sur_le_marche
    if key == "haut_risque":
        return (
            "haut_risque_annexe_iii" if "haut_risque_annexe_iii" in cats else "haut_risque_annexe_i"
        )
    if key == "modeles":
        return (
            "modeles_deja_sur_le_marche"
            if market == "avant_2025_08_02"
            else "modeles_usage_general"
        )
    return key


def _applies(entry: dict, profile: Profile, cats: set[str], conditions: dict, point: str) -> bool:
    if not set(entry["roles"]) & set(profile.roles):
        return False
    wanted = set(entry["categories"])
    if "haut_risque" in wanted and "haut_risque_annexe_i" in cats:
        wanted.add("haut_risque_annexe_i")
    if not wanted & cats:
        return False
    if entry.get("sauf_point") and point.split(" ")[0] == entry["sauf_point"]:
        return False
    return not entry.get("si") or any(conditions.get(flag) for flag in entry["si"])


def select(profile: Profile, classification: dict) -> list[dict]:
    """Obligations applicables, avec leur date (ou None : voir l'article 113) et leur statut.

    Statut « a_confirmer » : l'obligation dépend d'un point à confirmer (annexe I : le produit
    doit aussi être soumis à une évaluation par un tiers)."""
    cfg = load_map()
    cats = set(classification["categories"])
    conditions = classification["conditions"]
    point = classification.get("annexe_iii", {}).get("point") or ""
    only_annex_i = "haut_risque_annexe_i" in cats and "haut_risque" not in cats
    out = []
    for entry in cfg["obligations"]:
        if not _applies(entry, profile, cats, conditions, point):
            continue
        key = _date_key(entry.get("date"), profile, cats)
        if entry["id"] == "marquer_contenus" and profile.deja_sur_le_marche in (
            "avant_2025_08_02",
            "entre_2025_08_02_et_2026_08_02",
        ):
            key = "contenus_deja_sur_le_marche"
        dates = [cfg["dates"][key]] if key else []
        if entry["id"] == "interdiction":
            points = [p["point"] or "" for p in classification["interdit"]["pratiques"]]
            if any(p.removesuffix(")") in ("b bis", "b ter") for p in points):
                dates.append(cfg["dates"]["interdictions_ajoutees"])
        high_risk_entry = "haut_risque" in entry["categories"]
        out.append(
            {
                "id": entry["id"],
                "article": entry["article"],
                "titre": entry["titre"],
                "extrait": entry["extrait"],
                "source": entry.get("source", entry["article"]),
                "dates": [{**d, "date": str(d["date"])} for d in dates],
                "statut": "a_confirmer" if high_risk_entry and only_annex_i else "applicable",
            }
        )
    return out


def existing_high_risk_note(profile: Profile, classification: dict) -> dict | None:
    """Système à haut risque déjà sur le marché : article 111, paragraphe 2."""
    cats = set(classification["categories"])
    if profile.deja_sur_le_marche in (None, "non") or not cats & {
        "haut_risque",
        "haut_risque_annexe_i",
    }:
        return None
    return load_map()["haut_risque_deja_sur_le_marche"]


def verify_map(section_text) -> list[str]:
    """Extraits introuvables dans l'index (section_text(label) -> texte de la section)."""
    cfg = load_map()
    errors = []
    entries = [
        {"article": d["article"], "extrait": d["extrait"], "id": k} for k, d in cfg["dates"].items()
    ]
    entries.append(
        {**cfg["haut_risque_deja_sur_le_marche"], "id": "haut_risque_deja_sur_le_marche"}
    )
    entries += [{**e, "article": e.get("source", e["article"])} for e in cfg["obligations"]]
    for e in entries:
        text = section_text(e["article"])
        if not text:
            errors.append(f"{e['id']} : section « {e['article']} » absente de l'index")
        elif not quote_found(e["extrait"], [text]):
            errors.append(f"{e['id']} : extrait introuvable dans « {e['article']} »")
    for key, d in cfg["dates"].items():
        if not isinstance(d["date"], date):
            errors.append(f"{key} : date invalide")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Carte des obligations de l'AI Act")
    parser.add_argument("--verifier", action="store_true", help="vérifier les extraits")
    args = parser.parse_args()
    if not args.verifier:
        parser.print_help()
        return
    from app.retrieval.pipeline import RetrievalSettings, Retriever

    retriever = Retriever(RetrievalSettings.from_config())

    def section_text(label: str) -> str:
        return "\n".join(c["content"] for c in retriever.section("ai_act", label))

    from app.agents.classification import conditions_of, load_conditions, verify_conditions

    errors = verify_map(section_text) + verify_conditions(section_text("Article 5"))
    total = len(load_map()["obligations"]) + len(load_map()["dates"]) + 1
    total += sum(len(conditions_of(point)) for point in load_conditions())
    for error in errors:
        logger.error(error)
    logger.info("%d extraits vérifiés, %d erreur(s)", total, len(errors))
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    main()
